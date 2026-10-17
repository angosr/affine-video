"""Parse finalized Bittensor blocks into ordered Veda submissions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from miner.submit import MAGIC, ModelSubmission, SubmissionFormatError
from validator.queue import ChainPosition, submission_id
from validator.store import SubmissionRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChainCommitment:
    """One successful commitment call in finalized chain order."""

    hotkey: str
    netuid: int
    block_hash: str
    position: ChainPosition
    payload: bytes


@dataclass(frozen=True, slots=True)
class CommitmentReveal:
    """One automatic timelock reveal in finalized event order."""

    hotkey: str
    position: ChainPosition


def extract_commitment_reveals(
    *,
    block_height: int,
    events: Sequence[Mapping[str, Any]],
    netuid: int,
) -> list[CommitmentReveal]:
    """Extract reveal identities; plaintext is read from finalized storage."""

    records: list[CommitmentReveal] = []
    for event_position, event in enumerate(events):
        if (
            event.get("module_id") != "Commitments"
            or event.get("event_id") != "CommitmentRevealed"
            or _attribute(event, "netuid") != netuid
        ):
            continue
        hotkey = _attribute(event, "who")
        if not isinstance(hotkey, str) or not hotkey:
            LOGGER.warning(
                "skipping commitment reveal with invalid hotkey",
                extra={"block_height": block_height, "event_index": event_position},
            )
            continue
        extrinsic_index = event.get("extrinsic_idx")
        if extrinsic_index is None:
            extrinsic_index = 0
        if type(extrinsic_index) is not int or extrinsic_index < 0:
            LOGGER.warning(
                "skipping commitment reveal with invalid extrinsic index",
                extra={
                    "block_height": block_height,
                    "event_index": event_position,
                    "hotkey": hotkey,
                },
            )
            continue
        records.append(
            CommitmentReveal(
                hotkey,
                ChainPosition(block_height, extrinsic_index, event_position),
            )
        )
    return records


def extract_commitments(
    *,
    block_height: int,
    block_hash: str,
    block: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    netuid: int,
) -> list[ChainCommitment]:
    """Extract every successful top-level commitment call in chain order."""

    successful = {
        event.get("extrinsic_idx")
        for event in events
        if event.get("module_id") == "System" and event.get("event_id") == "ExtrinsicSuccess"
    }
    commitment_events = {
        event.get("extrinsic_idx"): (event_position, event)
        for event_position, event in enumerate(events)
        if event.get("module_id") == "Commitments"
        and event.get("event_id") == "Commitment"
        and _attribute(event, "netuid") == netuid
    }

    records: list[ChainCommitment] = []
    for extrinsic_index, extrinsic in enumerate(block.get("extrinsics", ())):
        call = extrinsic.get("call", {})
        if (
            extrinsic_index not in successful
            or call.get("call_module") != "Commitments"
            or call.get("call_function") != "set_commitment"
        ):
            continue
        arguments = {argument["name"]: argument["value"] for argument in call.get("call_args", ())}
        if arguments.get("netuid") != netuid:
            continue
        payload = _raw_payload(arguments.get("info"))
        if payload is None:
            continue
        if extrinsic_index not in commitment_events:
            raise RuntimeError("successful commitment is missing its chain event")
        event_position, event = commitment_events[extrinsic_index]
        hotkey = _attribute(event, "who")
        if not isinstance(hotkey, str) or not hotkey:
            raise RuntimeError("commitment event contains an invalid hotkey")
        if extrinsic.get("address") != hotkey:
            raise RuntimeError("commitment signer does not match its chain event")
        records.append(
            ChainCommitment(
                hotkey,
                netuid,
                block_hash,
                ChainPosition(block_height, extrinsic_index, event_position),
                payload,
            )
        )
    return records


def extract_submissions(
    *,
    block_height: int,
    block_hash: str,
    block: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    netuid: int,
) -> list[SubmissionRecord]:
    """Extract successful top-level Veda commitment calls in chain order."""
    records: list[SubmissionRecord] = []
    commitments = extract_commitments(
        block_height=block_height,
        block_hash=block_hash,
        block=block,
        events=events,
        netuid=netuid,
    )
    for commitment in commitments:
        submission = submission_from_commitment(commitment)
        if submission is not None:
            records.append(submission)
    return records


def submission_from_commitment(commitment: ChainCommitment) -> SubmissionRecord | None:
    """Decode one Veda payload while retaining malformed Veda attempts."""

    payload = commitment.payload
    if not payload.startswith(MAGIC):
        return None
    try:
        model = ModelSubmission.decode(payload)
    except SubmissionFormatError:
        repo_id = None
        revision = None
    else:
        repo_id = model.repo_id
        revision = model.revision
    return SubmissionRecord(
        submission_id=submission_id(
            commitment.netuid,
            commitment.block_hash,
            commitment.position,
        ),
        hotkey=commitment.hotkey,
        netuid=commitment.netuid,
        block_hash=commitment.block_hash,
        position=commitment.position,
        payload=payload,
        repo_id=repo_id,
        revision=revision,
    )


def revealed_payload(value: Any) -> bytes | None:
    """Decode the SDK representation of one revealed commitment payload."""

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        return None
    if value.startswith("0x"):
        try:
            return bytes.fromhex(value[2:])
        except ValueError:
            return None
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        return None


def _attribute(event: Mapping[str, Any], name: str) -> Any:
    attributes = event.get("attributes")
    if isinstance(attributes, Mapping):
        return attributes.get(name)
    nested = event.get("event")
    if isinstance(nested, Mapping):
        nested_attributes = nested.get("attributes")
        if isinstance(nested_attributes, Mapping):
            return nested_attributes.get(name)
    return None


def _raw_payload(info: Any) -> bytes | None:
    if not isinstance(info, Mapping):
        return None
    fields = info.get("fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)) or len(fields) != 1:
        return None
    field = fields[0]
    if not isinstance(field, Mapping) or len(field) != 1:
        return None
    variant, encoded = next(iter(field.items()))
    if not isinstance(variant, str) or not variant.startswith("Raw"):
        return None
    try:
        expected_length = int(variant.removeprefix("Raw"))
    except ValueError:
        return None

    if isinstance(encoded, str) and encoded.startswith("0x"):
        try:
            payload = bytes.fromhex(encoded[2:])
        except ValueError:
            return None
    elif isinstance(encoded, (bytes, bytearray)):
        payload = bytes(encoded)
    elif isinstance(encoded, Sequence):
        try:
            payload = bytes(encoded)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return payload if len(payload) == expected_length else None
