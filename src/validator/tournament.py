"""Deterministic FIFO tournament resolution over terminal challenge records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import Any

from scoring.spec import TrackSpec
from validator.queue import TERMINAL_STATES, ChainPosition, ChallengeState


class MatchOutcome(StrEnum):
    CHALLENGER_WON = "CHALLENGER_WON"
    CHAMPION_RETAINED = "CHAMPION_RETAINED"
    CHALLENGE_FAILED = "CHALLENGE_FAILED"


@dataclass(frozen=True, slots=True)
class Champion:
    challenge_id: str
    hotkey: str
    repo_id: str
    revision: str
    score: float
    position: ChainPosition
    uid: int | None = None
    source: str = "MINER"

    def public(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "hotkey": self.hotkey,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "score": self.score,
            "uid": self.uid,
            "source": self.source,
            "block_height": self.position.block,
            "extrinsic_index": self.position.extrinsic_index,
            "event_index": self.position.event_index,
        }


@dataclass(frozen=True, slots=True)
class MatchRecord:
    sequence: int
    challenge_id: str
    hotkey: str
    state: ChallengeState
    final_score: float | None
    score_gain: float | None
    outcome: MatchOutcome
    champion_after: Champion
    document: Mapping[str, Any]

    @property
    def object_name(self) -> str:
        return f"matches/{self.document['track_id']}/{self.sequence:012d}.json"


@dataclass(frozen=True, slots=True)
class Tournament:
    matches: tuple[MatchRecord, ...]
    champion: Champion
    leaderboard: tuple[dict[str, Any], ...]


def canonical_score_gain(score: float, frontier: float) -> float:
    """Subtract canonical JSON-style decimals without binary boundary drift."""

    if not isfinite(score) or not isfinite(frontier):
        raise ValueError("scores must be finite")
    return float(Decimal(str(score)) - Decimal(str(frontier)))


def genesis_champion(track: TrackSpec) -> Champion:
    """Return the code-defined UID0 champion that exists without an on-chain commit."""

    initial = track.initial_champion
    return Champion(
        challenge_id="genesis",
        hotkey=initial.hotkey,
        repo_id=initial.repo_id,
        revision=initial.revision,
        score=initial.score,
        position=ChainPosition(track.track_start_block, 0, 0),
        uid=initial.uid,
        source="VEDA_BASELINE",
    )


def challenge_from_match(document: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the public terminal challenge facts carried by one match object."""

    submission = document.get("submission")
    evaluation = document.get("evaluation")
    if not isinstance(submission, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("match submission and evaluation must be objects")
    return {
        "challenge_id": document.get("challenge_id"),
        "submission_id": document.get("submission_id"),
        "hotkey": submission.get("hotkey"),
        "repo_id": submission.get("repo_id"),
        "revision": submission.get("revision"),
        "block_height": submission.get("block_height"),
        "extrinsic_index": submission.get("extrinsic_index"),
        "event_index": submission.get("event_index"),
        "state": evaluation.get("state"),
        "error_code": evaluation.get("error_code"),
        "file_sha256": evaluation.get("file_sha256"),
        "model_digest": evaluation.get("model_digest"),
        "model_size_bytes": evaluation.get("model_size_bytes"),
        "final_score": evaluation.get("final_score"),
        "score_block": evaluation.get("score_block"),
        "incumbent_score": evaluation.get("incumbent_score"),
        "paired_gain_lower_bound": evaluation.get("paired_gain_lower_bound"),
        "evaluation_batch": evaluation.get("evaluation_batch"),
        "stages": evaluation.get("stages", []),
    }


def validate_match_document(
    document: Mapping[str, Any],
    *,
    sequence: int,
    track_id: str,
    track_spec_digest: str,
    netuid: int,
    minimum_gain: float,
    minimum_exact_gain: float | None = None,
    incumbent: Champion,
) -> MatchRecord:
    """Strictly validate one stored match and its link to the previous champion."""

    challenge = challenge_from_match(document)
    record, _ = _resolve_match(
        challenge,
        sequence=sequence,
        track_id=track_id,
        track_spec_digest=track_spec_digest,
        netuid=netuid,
        minimum_gain=minimum_gain,
        minimum_exact_gain=minimum_gain if minimum_exact_gain is None else minimum_exact_gain,
        incumbent=incumbent,
    )
    if dict(record.document) != dict(document):
        raise ValueError("stored match is not the canonical tournament decision")
    return record


def resolve_tournament(
    challenges: Iterable[Mapping[str, Any]],
    *,
    track_id: str,
    track_spec_digest: str,
    netuid: int,
    minimum_gain: float,
    minimum_exact_gain: float | None = None,
    initial_champion: Champion,
) -> Tournament:
    """Resolve the contiguous terminal prefix in finalized submission order."""

    if not track_id:
        raise ValueError("track ID cannot be empty")
    if len(track_spec_digest) != 64:
        raise ValueError("track specification digest must contain 32 bytes")
    try:
        bytes.fromhex(track_spec_digest)
    except ValueError as exc:
        raise ValueError("track specification digest must be hexadecimal") from exc
    if not 0 <= netuid <= 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    if not isfinite(minimum_gain) or not 0 < minimum_gain <= 1:
        raise ValueError("minimum gain must be within (0, 1]")
    exact_gain = minimum_gain if minimum_exact_gain is None else minimum_exact_gain
    if not isfinite(exact_gain) or not 0 < exact_gain <= 1:
        raise ValueError("minimum exact gain must be within (0, 1]")

    if initial_champion.uid != 0 or initial_champion.source != "VEDA_BASELINE":
        raise ValueError("initial champion must be the UID0 VEDA baseline")
    champion = initial_champion
    matches: list[MatchRecord] = []
    leaderboard_candidates: list[Champion] = [initial_champion]
    seen_hotkeys: set[str] = {initial_champion.hotkey}

    for challenge in challenges:
        state = ChallengeState(_required_string(challenge, "state"))
        origin = challenge.get("track_spec_digest") or track_spec_digest
        if origin != track_spec_digest:
            _optional_digest(origin, "historical track digest")
            if state not in TERMINAL_STATES:
                raise ValueError("an unfinished previous-track challenge cannot be rebound")
            historical_stages = challenge.get("stages", [])
            if len(historical_stages) == 2 and (
                historical_stages[0].get("paired_gain_lower_bound", -1) > exact_gain
                and historical_stages[1].get("paired_gain_lower_bound", -1) > minimum_gain
            ):
                raise ValueError(
                    "a previous-track promotion requires coordinated champion migration"
                )
            # Preserve completed facts and lifetime uniqueness, never re-label a
            # previous source/cache measurement as this track's match evidence.
            seen_hotkeys.add(_required_string(challenge, "hotkey"))
            continue
        if state not in TERMINAL_STATES:
            break
        sequence = len(matches) + 1
        hotkey = _required_string(challenge, "hotkey")
        repeated = hotkey in seen_hotkeys
        rejected_as_duplicate = state is ChallengeState.REJECTED_DUPLICATE_HOTKEY
        if repeated != rejected_as_duplicate:
            raise ValueError("exactly a hotkey's later submissions must be rejected")
        seen_hotkeys.add(hotkey)
        record, champion = _resolve_match(
            challenge,
            sequence=sequence,
            track_id=track_id,
            track_spec_digest=track_spec_digest.lower(),
            netuid=netuid,
            minimum_gain=minimum_gain,
            minimum_exact_gain=exact_gain,
            incumbent=champion,
        )
        matches.append(record)
        if state is ChallengeState.SUCCEEDED:
            position = _position(challenge)
            score = _score(challenge["final_score"], "final_score")
            _integer(challenge["score_block"], "score_block")
            repo_id = _required_string(challenge, "repo_id")
            revision = _required_string(challenge, "revision")
            candidate = Champion(
                record.challenge_id,
                record.hotkey,
                repo_id,
                revision,
                score,
                position,
            )
            leaderboard_candidates.append(candidate)

    ranked = [champion]
    ranked.extend(
        sorted(
            (
                candidate
                for candidate in leaderboard_candidates
                if candidate.challenge_id != champion.challenge_id
            ),
            key=_champion_order,
        )
    )
    leaderboard = tuple(
        {"rank": rank, **candidate.public()} for rank, candidate in enumerate(ranked, start=1)
    )
    return Tournament(tuple(matches), champion, leaderboard)


def _resolve_match(
    challenge: Mapping[str, Any],
    *,
    sequence: int,
    track_id: str,
    track_spec_digest: str,
    netuid: int,
    minimum_gain: float,
    minimum_exact_gain: float,
    incumbent: Champion,
) -> tuple[MatchRecord, Champion]:
    identifier = _required_string(challenge, "challenge_id")
    submission_id = _required_string(challenge, "submission_id")
    hotkey = _required_string(challenge, "hotkey")
    position = _position(challenge)
    state = ChallengeState(_required_string(challenge, "state"))
    if state not in TERMINAL_STATES:
        raise ValueError("a match must contain a terminal challenge")
    repo_id = _optional_string(challenge.get("repo_id"), "repo_id")
    revision = _optional_string(challenge.get("revision"), "revision")
    error_code = _optional_string(challenge.get("error_code"), "error_code")
    file_sha256 = _optional_digest(challenge.get("file_sha256"), "file_sha256")
    model_digest = _optional_digest(challenge.get("model_digest"), "model_digest")
    model_size_bytes = _optional_integer(challenge.get("model_size_bytes"), "model_size_bytes")
    before = incumbent
    score_gain: float | None = None
    stages = _evaluation_stages(
        challenge.get("stages", []),
    )
    if any(stage["score_block"] < position.block for stage in stages):
        raise ValueError("stage decision block cannot precede its submission")

    if state is ChallengeState.SUCCEEDED:
        if repo_id is None or revision is None:
            raise ValueError("successful challenge is missing its model reference")
        if error_code is not None:
            raise ValueError("successful challenge cannot contain an error code")
        if challenge.get("score_block") is None:
            raise ValueError("successful challenge is missing its score block")
        if file_sha256 is None or model_digest is None or model_size_bytes is None:
            raise ValueError("successful challenge is missing its model facts")
        score = _score(challenge.get("final_score"), "final_score")
        candidate = Champion(identifier, hotkey, repo_id, revision, score, position)
        incumbent_score = _score(challenge.get("incumbent_score"), "incumbent_score")
        score_block = _integer(challenge["score_block"], "score_block")
        if score_block < position.block:
            raise ValueError("decision block cannot precede its submission")
        batch = _evaluation_batch(
            challenge.get("evaluation_batch"),
        )
        if score_block < batch["generated_block"]:
            raise ValueError("decision block cannot precede its evaluation batch")
        stage_batches = [
            stage["evaluation_batch"] for stage in stages if "evaluation_batch" in stage
        ]
        if stage_batches and stage_batches[-1] != batch:
            raise ValueError("final evaluation batch does not match the last scored stage")
        score_gain = _gain(
            challenge.get("paired_gain_lower_bound"),
            "paired_gain_lower_bound",
        )
        if not stages:
            raise ValueError("successful challenge is missing its exact stage")
        exact_stage = stages[0]
        exact_gain = exact_stage.get("paired_gain_lower_bound")
        if exact_gain is None:
            raise ValueError("exact stage is missing its paired gain")
        if len(stages) == 1 and exact_gain > minimum_exact_gain:
            raise ValueError("an exact-stage promotion requires full-video confirmation")
        if len(stages) == 2:
            confirmation = stages[1]
            if exact_gain <= minimum_exact_gain:
                raise ValueError("full-video confirmation did not clear the exact gate")
            exact_batch = exact_stage.get("evaluation_batch")
            confirmation_batch = confirmation.get("evaluation_batch")
            if exact_batch is None or confirmation_batch is None:
                raise ValueError("both scoring stages must identify their evaluation batch")
            if confirmation_batch["plan_digest"] == exact_batch["plan_digest"]:
                raise ValueError("confirmation must use independent inventory")
            if confirmation["score_block"] < confirmation_batch["generated_block"]:
                raise ValueError("confirmation decision predates its evaluation batch")
        final_stage = stages[-1]
        if (
            final_stage["score"] != score
            or final_stage.get("incumbent_score") != incumbent_score
            or final_stage.get("paired_gain_lower_bound") != score_gain
        ):
            raise ValueError("final score does not match the last completed stage")
        cleared_both_stages = (
            len(stages) == 2 and exact_gain > minimum_exact_gain and score_gain > minimum_gain
        )
        if cleared_both_stages:
            outcome = MatchOutcome.CHALLENGER_WON
            incumbent = candidate
        else:
            outcome = MatchOutcome.CHAMPION_RETAINED
            incumbent = replace(incumbent, score=incumbent_score)
    else:
        if error_code is None:
            raise ValueError("failed challenge is missing its error code")
        score = None
        incumbent_score = None
        batch = None
        outcome = MatchOutcome.CHALLENGE_FAILED

    decided_block = (
        _integer(challenge["score_block"], "score_block")
        if challenge.get("score_block") is not None
        else position.block
    )
    if decided_block < position.block:
        raise ValueError("decision block cannot precede submission")
    evaluation = {
        "state": state,
        "error_code": error_code,
        "file_sha256": file_sha256,
        "model_digest": model_digest,
        "model_size_bytes": model_size_bytes,
        "final_score": score,
        "score_block": (
            _integer(challenge["score_block"], "score_block")
            if challenge.get("score_block") is not None
            else None
        ),
        "incumbent_score": incumbent_score,
        "paired_gain_lower_bound": score_gain,
        "evaluation_batch": batch,
        "model_object": (
            f"models/{track_id}/{file_sha256}/manifest.json"
            if state is ChallengeState.SUCCEEDED
            else None
        ),
        "stages": stages,
    }
    document = {
        "track_id": track_id,
        "track_spec_digest": track_spec_digest,
        "netuid": netuid,
        "sequence": sequence,
        "challenge_id": identifier,
        "submission_id": submission_id,
        "submission": {
            "hotkey": hotkey,
            "repo_id": repo_id,
            "revision": revision,
            "block_height": position.block,
            "extrinsic_index": position.extrinsic_index,
            "event_index": position.event_index,
        },
        "evaluation": evaluation,
        "incumbent_before": before.public(),
        "minimum_gain": minimum_gain,
        "score_gain": score_gain,
        "outcome": outcome,
        "champion_after": incumbent.public(),
        "decided_block": decided_block,
    }
    return (
        MatchRecord(
            sequence,
            identifier,
            hotkey,
            state,
            score,
            score_gain,
            outcome,
            incumbent,
            document,
        ),
        incumbent,
    )


def _position(value: Mapping[str, Any]) -> ChainPosition:
    return ChainPosition(
        _integer(value.get("block_height"), "block_height"),
        _integer(value.get("extrinsic_index"), "extrinsic_index"),
        _integer(value.get("event_index"), "event_index"),
    )


def _champion_order(value: Champion) -> tuple[float, ChainPosition]:
    return (-value.score, value.position)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be null or a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _gain(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or not -1 <= result <= 1:
        raise ValueError(f"{name} must be within [-1, 1]")
    return result


def _optional_digest(value: Any, name: str) -> str | None:
    if value is None:
        return None
    result = _optional_string(value, name)
    assert result is not None
    if len(result) != 64:
        raise ValueError(f"{name} must contain 32 bytes")
    try:
        bytes.fromhex(result)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return result.lower()


def _evaluation_batch(
    value: Any,
) -> dict[str, Any]:
    expected = {
        "sha256",
        "generated_block",
        "plan_digest",
        "seed_block_hash",
    }
    if not isinstance(value, Mapping) or set(value) not in (expected, expected | {"object"}):
        raise ValueError("evaluation_batch fields are invalid")
    sha256 = _optional_digest(value["sha256"], "evaluation batch sha256")
    plan_digest = _optional_digest(value["plan_digest"], "evaluation batch plan_digest")
    generated_block = _integer(value["generated_block"], "evaluation batch generated_block")
    assert sha256 is not None and plan_digest is not None
    seed_block_hash = _block_hash(value["seed_block_hash"])
    object_name = f"batches/{sha256}/manifest.json"
    if "object" in value and value["object"] != object_name:
        raise ValueError("evaluation batch object name is invalid")
    return {
        "sha256": sha256,
        "generated_block": generated_block,
        "plan_digest": plan_digest,
        "seed_block_hash": seed_block_hash,
        "object": object_name,
    }


def _evaluation_stages(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("evaluation stages must be a list")
    result = []
    seen = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("evaluation stage must be an object")
        stage = ChallengeState(_required_string(row, "stage"))
        if stage not in {ChallengeState.SCORING, ChallengeState.CONFIRMING} or stage in seen:
            raise ValueError("evaluation stage order is invalid")
        if stage is ChallengeState.SCORING and seen:
            raise ValueError("exact stage must precede confirmation")
        if stage is ChallengeState.CONFIRMING and ChallengeState.SCORING not in seen:
            raise ValueError("confirmation stage requires an exact stage")
        seen.add(stage)
        score = _score(row.get("score"), "stage score")
        score_block = _integer(row.get("score_block"), "stage score_block")
        if result and score_block < result[-1]["score_block"]:
            raise ValueError("stage decisions cannot run backwards")
        document: dict[str, Any] = {
            "score": score,
            "score_block": score_block,
            "stage": stage,
        }
        if row.get("evaluation_batch") is not None:
            batch = _evaluation_batch(
                row["evaluation_batch"],
            )
            if score_block < batch["generated_block"]:
                raise ValueError("stage decision block cannot precede its evaluation batch")
            document.update(
                {
                    "evaluation_batch": batch,
                    "incumbent_score": _score(row.get("incumbent_score"), "stage incumbent_score"),
                    "paired_gain_lower_bound": _gain(
                        row.get("paired_gain_lower_bound"),
                        "stage paired_gain_lower_bound",
                    ),
                }
            )
        result.append(document)
    return result


def _block_hash(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("evaluation batch seed block hash must be hexadecimal")
    normalized = value.removeprefix("0x").lower()
    try:
        decoded = bytes.fromhex(normalized)
    except ValueError as exc:
        raise ValueError("evaluation batch seed block hash must be hexadecimal") from exc
    if len(decoded) != 32:
        raise ValueError("evaluation batch seed block hash must contain 32 bytes")
    return "0x" + normalized
