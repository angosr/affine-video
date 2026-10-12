"""Parse finalized Bittensor blocks into ordered Veda submissions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from miner.submit import MAGIC, ModelSubmission, SubmissionFormatError
from validator.queue import ChainPosition, submission_id
from validator.store import SubmissionRecord


def extract_submissions(
    *,
    block_height: int,
    block_hash: str,
    block: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    netuid: int,
) -> list[SubmissionRecord]:
    """Extract successful top-level Veda commitment calls in chain order."""

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

    records: list[SubmissionRecord] = []
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
        if payload is None or not payload.startswith(MAGIC):
            continue
        if extrinsic_index not in commitment_events:
            raise RuntimeError("successful Veda commitment is missing its chain event")

        event_position, event = commitment_events[extrinsic_index]
        hotkey = _attribute(event, "who")
        if not isinstance(hotkey, str) or not hotkey:
            raise RuntimeError("Veda commitment event contains an invalid hotkey")
        if extrinsic.get("address") != hotkey:
            raise RuntimeError("Veda commitment signer does not match its chain event")

        position = ChainPosition(block_height, extrinsic_index, event_position)
        try:
            model = ModelSubmission.decode(payload)
        except SubmissionFormatError:
            repo_id = None
            revision = None
        else:
            repo_id = model.repo_id
            revision = model.revision

        records.append(
            SubmissionRecord(
                submission_id=submission_id(netuid, block_hash, position),
                hotkey=hotkey,
                netuid=netuid,
                block_hash=block_hash,
                position=position,
                payload=payload,
                repo_id=repo_id,
                revision=revision,
            )
        )
    return records


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
