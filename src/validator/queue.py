"""Deterministic submission ordering and challenge states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256


@dataclass(frozen=True, order=True, slots=True)
class ChainPosition:
    """A total order for successful commitment calls."""

    block: int
    extrinsic_index: int
    event_index: int

    def __post_init__(self) -> None:
        if min(self.block, self.extrinsic_index, self.event_index) < 0:
            raise ValueError("chain position values must be non-negative")


def submission_id(netuid: int, block_hash: str, position: ChainPosition) -> str:
    """Return a domain-separated identifier for one finalized submission."""

    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")

    normalized_hash = block_hash.removeprefix("0x")
    try:
        block_hash_bytes = bytes.fromhex(normalized_hash)
    except ValueError as exc:
        raise ValueError("block_hash must be hexadecimal") from exc
    if len(block_hash_bytes) != 32:
        raise ValueError("block_hash must contain 32 bytes")

    digest = sha256()
    digest.update(b"veda-submission-id-v1\0")
    digest.update(netuid.to_bytes(2, "big"))
    digest.update(block_hash_bytes)
    digest.update(position.block.to_bytes(8, "big"))
    digest.update(position.extrinsic_index.to_bytes(4, "big"))
    digest.update(position.event_index.to_bytes(4, "big"))
    return digest.hexdigest()


def challenge_id(submission: str) -> str:
    """Return the stable challenge identifier for a submission."""

    try:
        submission_bytes = bytes.fromhex(submission)
    except ValueError as exc:
        raise ValueError("submission_id must be hexadecimal") from exc
    if len(submission_bytes) != 32:
        raise ValueError("submission_id must contain 32 bytes")

    digest = sha256()
    digest.update(b"veda-challenge-id-v1\0")
    digest.update(submission_bytes)
    return digest.hexdigest()


def passes_cooldown(
    *, current_block: int, last_accepted_block: int | None, cooldown_blocks: int
) -> bool:
    """Apply the track's per-hotkey rate limit using chain data only."""

    if current_block < 0 or cooldown_blocks < 0:
        raise ValueError("block values must be non-negative")
    if last_accepted_block is None:
        return True
    if last_accepted_block < 0 or current_block < last_accepted_block:
        raise ValueError("last accepted block cannot be in the future")
    return current_block - last_accepted_block >= cooldown_blocks


class ChallengeState(StrEnum):
    QUEUED = "QUEUED"
    FETCHING = "FETCHING"
    VALIDATING = "VALIDATING"
    SCREENING = "SCREENING"
    SCORING = "SCORING"
    PROVISIONAL = "PROVISIONAL"
    ROLLOUT = "ROLLOUT"
    SYSTEM_DEFERRED = "SYSTEM_DEFERRED"
    SYSTEM_FAILED = "SYSTEM_FAILED"
    SUCCEEDED = "SUCCEEDED"
    SCREENED_OUT = "SCREENED_OUT"
    PARTICIPANT_FAILED = "PARTICIPANT_FAILED"
    MALFORMED_SUBMISSION = "MALFORMED_SUBMISSION"
    REJECTED_RATE_LIMIT = "REJECTED_RATE_LIMIT"


TERMINAL_STATES = frozenset(
    {
        ChallengeState.SUCCEEDED,
        ChallengeState.SCREENED_OUT,
        ChallengeState.PARTICIPANT_FAILED,
        ChallengeState.SYSTEM_FAILED,
        ChallengeState.MALFORMED_SUBMISSION,
        ChallengeState.REJECTED_RATE_LIMIT,
    }
)

_ACTIVE_STAGES = frozenset(
    {
        ChallengeState.FETCHING,
        ChallengeState.VALIDATING,
        ChallengeState.SCREENING,
        ChallengeState.SCORING,
        ChallengeState.ROLLOUT,
    }
)

_TRANSITIONS: dict[ChallengeState, frozenset[ChallengeState]] = {
    ChallengeState.QUEUED: frozenset(
        {
            ChallengeState.FETCHING,
            ChallengeState.MALFORMED_SUBMISSION,
            ChallengeState.REJECTED_RATE_LIMIT,
        }
    ),
    ChallengeState.FETCHING: frozenset(
        {
            ChallengeState.VALIDATING,
            ChallengeState.PARTICIPANT_FAILED,
            ChallengeState.SYSTEM_DEFERRED,
            ChallengeState.SYSTEM_FAILED,
        }
    ),
    ChallengeState.VALIDATING: frozenset(
        {
            ChallengeState.SCREENING,
            ChallengeState.PARTICIPANT_FAILED,
            ChallengeState.SYSTEM_DEFERRED,
            ChallengeState.SYSTEM_FAILED,
        }
    ),
    ChallengeState.SCREENING: frozenset(
        {
            ChallengeState.SCORING,
            ChallengeState.SCREENED_OUT,
            ChallengeState.PARTICIPANT_FAILED,
            ChallengeState.SYSTEM_DEFERRED,
            ChallengeState.SYSTEM_FAILED,
        }
    ),
    ChallengeState.SCORING: frozenset(
        {
            ChallengeState.ROLLOUT,
            ChallengeState.PROVISIONAL,
            ChallengeState.SUCCEEDED,
            ChallengeState.PARTICIPANT_FAILED,
            ChallengeState.SYSTEM_DEFERRED,
            ChallengeState.SYSTEM_FAILED,
        }
    ),
    ChallengeState.ROLLOUT: frozenset(
        {
            ChallengeState.SUCCEEDED,
            ChallengeState.PARTICIPANT_FAILED,
            ChallengeState.SYSTEM_DEFERRED,
            ChallengeState.SYSTEM_FAILED,
        }
    ),
    ChallengeState.PROVISIONAL: frozenset({ChallengeState.ROLLOUT}),
    ChallengeState.SYSTEM_DEFERRED: _ACTIVE_STAGES,
}


def require_transition(current: ChallengeState, target: ChallengeState) -> None:
    """Raise when a challenge state change violates the frozen state machine."""

    if target not in _TRANSITIONS.get(current, frozenset()):
        raise ValueError(f"invalid challenge transition: {current} -> {target}")
