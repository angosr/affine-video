from copy import deepcopy

import pytest

from miner.submit import ModelSubmission
from validator.scan import extract_commitment_reveals, extract_submissions, revealed_payload

BLOCK_HASH = "0x" + "ab" * 32
HOTKEY = "5C9ywXcKhSNrFY2ttziLTLNPZB9XPqGaKbLpvh6J6TuJyBQQ"


def fixture(payload: bytes, *, netuid: int = 7) -> tuple[dict, list[dict]]:
    block = {
        "extrinsics": [
            {"call": {"call_module": "Timestamp", "call_function": "set"}},
            {
                "address": HOTKEY,
                "call": {
                    "call_module": "Commitments",
                    "call_function": "set_commitment",
                    "call_args": [
                        {"name": "netuid", "value": netuid},
                        {
                            "name": "info",
                            "value": {"fields": [{f"Raw{len(payload)}": "0x" + payload.hex()}]},
                        },
                    ],
                },
            },
        ]
    }
    events = [
        {"extrinsic_idx": 1, "module_id": "System", "event_id": "ExtrinsicSuccess"},
        {
            "extrinsic_idx": 1,
            "module_id": "Commitments",
            "event_id": "Commitment",
            "attributes": {"netuid": netuid, "who": HOTKEY},
        },
    ]
    return block, events


def test_extracts_valid_submission_from_runtime_shape() -> None:
    model = ModelSubmission("owner/model", "12" * 20)
    block, events = fixture(model.encode())

    records = extract_submissions(
        block_height=100,
        block_hash=BLOCK_HASH,
        block=block,
        events=events,
        netuid=7,
    )

    assert len(records) == 1
    assert records[0].hotkey == HOTKEY
    assert records[0].repo_id == "owner/model"
    assert records[0].position.extrinsic_index == 1
    assert records[0].position.event_index == 1


def test_preserves_malformed_veda_payload_for_challenge_record() -> None:
    block, events = fixture(b"VDA1\x10short")

    records = extract_submissions(
        block_height=100,
        block_hash=BLOCK_HASH,
        block=block,
        events=events,
        netuid=7,
    )

    assert len(records) == 1
    assert records[0].repo_id is None
    assert records[0].revision is None


def test_ignores_failed_other_netuid_and_non_veda_calls() -> None:
    model = ModelSubmission("owner/model", "12" * 20)
    block, events = fixture(model.encode())

    failed_events = [event for event in events if event["module_id"] != "System"]
    assert not extract_submissions(
        block_height=100,
        block_hash=BLOCK_HASH,
        block=block,
        events=failed_events,
        netuid=7,
    )

    assert not extract_submissions(
        block_height=100,
        block_hash=BLOCK_HASH,
        block=block,
        events=events,
        netuid=8,
    )

    other_block, other_events = fixture(b"not-veda")
    assert not extract_submissions(
        block_height=100,
        block_hash=BLOCK_HASH,
        block=other_block,
        events=other_events,
        netuid=7,
    )


def test_requires_committing_hotkey_to_match_signer() -> None:
    model = ModelSubmission("owner/model", "12" * 20)
    block, events = fixture(model.encode())
    mismatched = deepcopy(block)
    mismatched["extrinsics"][1]["address"] = "different"

    with pytest.raises(RuntimeError, match="signer"):
        extract_submissions(
            block_height=100,
            block_hash=BLOCK_HASH,
            block=mismatched,
            events=events,
            netuid=7,
        )


def test_successful_veda_call_requires_matching_commitment_event() -> None:
    model = ModelSubmission("owner/model", "12" * 20)
    block, events = fixture(model.encode())
    events = [event for event in events if event["module_id"] != "Commitments"]

    with pytest.raises(RuntimeError, match="missing"):
        extract_submissions(
            block_height=100,
            block_hash=BLOCK_HASH,
            block=block,
            events=events,
            netuid=7,
        )


def test_extracts_automatic_timelock_reveal_event() -> None:
    records = extract_commitment_reveals(
        block_height=123,
        events=[
            {
                "extrinsic_idx": None,
                "module_id": "Commitments",
                "event_id": "CommitmentRevealed",
                "attributes": {"netuid": 7, "who": HOTKEY},
            }
        ],
        netuid=7,
    )

    assert len(records) == 1
    assert records[0].hotkey == HOTKEY
    assert records[0].position.block == 123
    assert records[0].position.extrinsic_index == 0


def test_decodes_revealed_payload_sdk_forms() -> None:
    assert revealed_payload("hello") == b"hello"
    assert revealed_payload("0x56444131") == b"VDA1"
    assert revealed_payload(b"bytes") == b"bytes"
    assert revealed_payload("0xzz") is None
    assert revealed_payload("\ud800") is None


def test_malformed_timelock_reveal_is_skipped_without_losing_valid_reveals(caplog) -> None:
    records = extract_commitment_reveals(
        block_height=123,
        events=[
            {
                "extrinsic_idx": -1,
                "module_id": "Commitments",
                "event_id": "CommitmentRevealed",
                "attributes": {"netuid": 7, "who": HOTKEY},
            },
            {
                "extrinsic_idx": None,
                "module_id": "Commitments",
                "event_id": "CommitmentRevealed",
                "attributes": {"netuid": 7, "who": HOTKEY},
            },
        ],
        netuid=7,
    )

    assert len(records) == 1
    assert "invalid extrinsic index" in caplog.text
    assert '"block_height": 123' in caplog.text
    assert '"event_index": 0' in caplog.text
