"""Finalized-chain synchronization for the validator."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from validator.competition import LEGACY_ACTIVATION_PREFIX, is_valid_legacy_activation
from validator.scan import (
    ChainCommitment,
    extract_commitment_reveals,
    extract_commitments,
    revealed_payload,
    submission_from_commitment,
)
from validator.store import ChallengeStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScanResult:
    first_block: int | None
    last_block: int | None
    blocks_scanned: int
    submissions_found: int


async def scan_finalized_blocks(
    client: Any,
    store: ChallengeStore,
    *,
    netuid: int,
    track_start_block: int,
    submission_end_block: int | None = None,
    max_blocks: int = 100,
) -> ScanResult:
    """Scan one bounded batch and commit the cursor after each complete block."""

    if track_start_block < 0:
        raise ValueError("track start block must be non-negative")
    if submission_end_block is not None and submission_end_block < track_start_block:
        raise ValueError("submission end block cannot precede the track")
    if max_blocks <= 0:
        raise ValueError("max blocks must be positive")

    finalized = await client.finalized_block()
    cursor = store.chain_cursor()
    first = track_start_block if cursor is None else cursor[0] + 1
    last = min(finalized, first + max_blocks - 1)
    if first > last:
        return ScanResult(None, None, 0, 0)

    submissions_found = 0
    for block_height in range(first, last + 1):
        current = store.chain_cursor()
        if current is not None and current[0] >= block_height:
            continue
        block_hash = await client._substrate.block_hash(block_height)
        submissions_closed = (
            submission_end_block is not None and block_height > submission_end_block
        )
        block = await client._substrate.get_block(block_hash=block_hash)
        if block is None:
            raise RuntimeError(f"finalized block {block_height} is unavailable")
        events = await client._substrate.events(block_hash)
        commitments = extract_commitments(
            block_height=block_height,
            block_hash=block_hash,
            block=block,
            events=events,
            netuid=netuid,
        )
        reveals = extract_commitment_reveals(
            block_height=block_height,
            events=events,
            netuid=netuid,
        )
        if reveals:
            snapshot = await client.at(block_height)
            current_commitments = await snapshot.subnets.commitments(netuid)
            by_hotkey = _commitments_by_hotkey(current_commitments)
            reveal_counts = Counter(reveal.hotkey for reveal in reveals)
            reveal_offsets: Counter[str] = Counter()
            for reveal in reveals:
                offset = reveal_offsets[reveal.hotkey]
                reveal_offsets[reveal.hotkey] += 1
                item = by_hotkey.get(reveal.hotkey)
                values = getattr(item, "revealed", ()) if item is not None else ()
                context = {
                    "block_height": block_height,
                    "event_index": reveal.position.event_index,
                    "hotkey": reveal.hotkey,
                    "netuid": netuid,
                }
                if not isinstance(values, Sequence) or isinstance(
                    values, (str, bytes, bytearray)
                ):
                    LOGGER.warning(
                        "skipping malformed finalized commitment reveal history",
                        extra=context,
                    )
                    values = ()
                matching = []
                for stored in values:
                    if not isinstance(stored, (tuple, list)) or len(stored) != 2:
                        LOGGER.warning(
                            "skipping malformed finalized commitment reveal entry",
                            extra=context,
                        )
                        continue
                    height, value = stored
                    if type(height) is not int or height < 0:
                        LOGGER.warning(
                            "skipping finalized commitment reveal with invalid block",
                            extra=context,
                        )
                        continue
                    if height != block_height:
                        continue
                    payload = revealed_payload(value)
                    if payload is None:
                        LOGGER.warning(
                            "skipping undecodable finalized commitment reveal",
                            extra=context,
                        )
                        continue
                    if not payload:
                        LOGGER.warning(
                            "skipping empty finalized commitment reveal",
                            extra=context,
                        )
                        continue
                    matching.append(payload)
                if len(matching) != reveal_counts[reveal.hotkey]:
                    LOGGER.warning(
                        "skipping unmatched finalized commitment reveal",
                        extra={
                            "block_height": block_height,
                            "event_index": reveal.position.event_index,
                            "hotkey": reveal.hotkey,
                            "matching_plaintexts": len(matching),
                            "reveal_events": reveal_counts[reveal.hotkey],
                            "netuid": netuid,
                        },
                    )
                    continue
                commitments.append(
                    ChainCommitment(
                        reveal.hotkey,
                        netuid,
                        block_hash,
                        reveal.position,
                        # The pallet appends plaintexts and emits their events in
                        # the same field order, including multiple fields per block.
                        matching[offset],
                    )
                )
        commitments.sort(key=lambda item: item.position)
        submissions = tuple(
            submission
            for commitment in commitments
            if not submissions_closed
            and (submission := submission_from_commitment(commitment)) is not None
        )
        submissions_by_position = {submission.position: submission for submission in submissions}
        for commitment in commitments:
            legacy_excluded = is_valid_legacy_activation(
                commitment.payload, hotkey=commitment.hotkey, netuid=netuid,
            )
            if submissions_closed and not legacy_excluded:
                continue
            submission = submissions_by_position.get(commitment.position)
            if submission is not None:
                store.add_submission(submission)
            store.observe_commitment(
                hotkey=commitment.hotkey,
                netuid=commitment.netuid,
                block_hash=commitment.block_hash,
                position=commitment.position,
                payload=commitment.payload,
                submission_id=None if submission is None else submission.submission_id,
                legacy_excluded=legacy_excluded,
            )
        try:
            store.advance_chain_cursor(block_height, block_hash)
        except ValueError:
            current = store.chain_cursor()
            if (
                current is None
                or current[0] < block_height
                or (current[0] == block_height and current[1] != block_hash)
            ):
                raise
        submissions_found += len(submissions)

    return ScanResult(first, last, last - first + 1, submissions_found)


async def seed_legacy_exclusions(
    client: Any,
    store: ChallengeStore,
    *,
    netuid: int,
    before_block: int,
) -> int:
    """Snapshot signed old-contest activations that predate this track."""

    if before_block < 0:
        return 0
    snapshot = await client.at(before_block)
    graph = await snapshot.subnets.metagraph(netuid, commitments=True)
    if graph is None:
        raise RuntimeError(f"subnet {netuid} does not exist at block {before_block}")
    exclusions: dict[str, int] = {}
    commitments: dict[str, Any] = {}
    for neuron in graph.neurons:
        hotkey = getattr(neuron, "hotkey", None)
        commitment = getattr(neuron, "commitment", None)
        if isinstance(hotkey, str) and hotkey and commitment is not None:
            commitments[hotkey] = commitment
    unregistered = getattr(graph, "unregistered_commitments", {})
    if isinstance(unregistered, Mapping):
        for key, commitment in unregistered.items():
            hotkey = getattr(commitment, "hotkey", key)
            if (
                isinstance(key, str)
                and key
                and hotkey == key
                and commitment is not None
            ):
                commitments.setdefault(key, commitment)
            else:
                LOGGER.warning("skipping inconsistent unregistered commitment identity")
    elif unregistered:
        LOGGER.warning("unregistered commitment snapshot has an unsupported shape")

    for hotkey, commitment in commitments.items():
        revealed = getattr(commitment, "revealed", ())
        for item in revealed or ():
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            block, value = item
            if type(block) is not int or block < 0 or block > before_block:
                continue
            payload = revealed_payload(value)
            if payload is None:
                continue
            if not payload.startswith(LEGACY_ACTIVATION_PREFIX):
                continue
            if is_valid_legacy_activation(payload, hotkey=hotkey, netuid=netuid):
                exclusions[hotkey] = min(block, exclusions.get(hotkey, block))
    store.seed_legacy_exclusions(exclusions)
    return len(exclusions)


def _commitments_by_hotkey(value: Any) -> dict[str, Any]:
    """Normalize the SDK's mapping shape and older sequence-shaped fixtures."""

    if isinstance(value, Mapping):
        rows = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = ((getattr(item, "hotkey", None), item) for item in value)
    else:
        LOGGER.warning("commitment snapshot has an unsupported shape")
        return {}
    result: dict[str, Any] = {}
    for key, item in rows:
        hotkey = getattr(item, "hotkey", key)
        if not isinstance(hotkey, str) or not hotkey or hotkey != key:
            LOGGER.warning("skipping commitment snapshot row with inconsistent hotkey")
            continue
        if hotkey in result:
            LOGGER.warning("skipping duplicate commitment snapshot row", extra={"hotkey": hotkey})
            continue
        result[hotkey] = item
    return result
