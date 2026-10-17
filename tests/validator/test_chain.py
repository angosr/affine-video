import asyncio
import logging
from types import SimpleNamespace

import pytest

from miner.submit import ModelSubmission
from validator import chain as chain_module
from validator.chain import scan_finalized_blocks, seed_legacy_exclusions
from validator.queue import ChallengeState
from validator.scan import extract_submissions
from validator.store import ChallengeStore

HOTKEY = "5C9ywXcKhSNrFY2ttziLTLNPZB9XPqGaKbLpvh6J6TuJyBQQ"


def chain_block(payload: bytes, block_height: int) -> tuple[str, dict, list[dict]]:
    block_hash = "0x" + f"{block_height:064x}"
    block = {
        "extrinsics": [
            {
                "address": HOTKEY,
                "call": {
                    "call_module": "Commitments",
                    "call_function": "set_commitment",
                    "call_args": [
                        {"name": "netuid", "value": 7},
                        {
                            "name": "info",
                            "value": {"fields": [{f"Raw{len(payload)}": "0x" + payload.hex()}]},
                        },
                    ],
                },
            }
        ]
    }
    events = [
        {"extrinsic_idx": 0, "module_id": "System", "event_id": "ExtrinsicSuccess"},
        {
            "extrinsic_idx": 0,
            "module_id": "Commitments",
            "event_id": "Commitment",
            "attributes": {"netuid": 7, "who": HOTKEY},
        },
    ]
    return block_hash, block, events


class FakeSubstrate:
    def __init__(self, blocks: dict[int, tuple[str, dict, list[dict]]]) -> None:
        self.blocks = blocks
        self.by_hash = {value[0]: value for value in blocks.values()}

    async def block_hash(self, height: int) -> str:
        return self.blocks[height][0]

    async def get_block(self, *, block_hash: str):
        return self.by_hash[block_hash][1]

    async def events(self, block_hash: str):
        return self.by_hash[block_hash][2]


class FakeClient:
    def __init__(self, blocks: dict[int, tuple[str, dict, list[dict]]]) -> None:
        self._substrate = FakeSubstrate(blocks)
        self._finalized = max(blocks)

    async def finalized_block(self) -> int:
        return self._finalized

    async def at(self, height: int):
        values = getattr(self, "revealed", {}).get(height, ())
        return SimpleNamespace(
            subnets=SimpleNamespace(
                commitments=lambda netuid: _ready(values),
            )
        )


async def _ready(value):
    return value


def test_scanner_advances_persistent_cursor_in_bounded_batches(tmp_path) -> None:
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    blocks = {height: chain_block(payload, height) for height in range(100, 103)}
    client = FakeClient(blocks)

    with ChallengeStore(tmp_path / "validator.db") as store:
        first = asyncio.run(
            scan_finalized_blocks(
                client,
                store,
                netuid=7,
                track_start_block=100,
                max_blocks=2,
            )
        )
        second = asyncio.run(
            scan_finalized_blocks(
                client,
                store,
                netuid=7,
                track_start_block=100,
                max_blocks=2,
            )
        )
        empty = asyncio.run(
            scan_finalized_blocks(
                client,
                store,
                netuid=7,
                track_start_block=100,
                max_blocks=2,
            )
        )

        assert first.blocks_scanned == 2
        assert second.blocks_scanned == 1
        assert empty.blocks_scanned == 0
        assert store.chain_cursor() == (102, blocks[102][0])
        challenge = store.public_snapshot()["challenges"][0]
        assert challenge["block_height"] == 100


def test_scanner_does_not_advance_cursor_past_failed_block(tmp_path) -> None:
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    blocks = {height: chain_block(payload, height) for height in range(100, 102)}
    client = FakeClient(blocks)

    async def unavailable(*, block_hash: str):
        if block_hash == blocks[101][0]:
            return None
        return blocks[100][1]

    client._substrate.get_block = unavailable
    with ChallengeStore(tmp_path / "validator.db") as store:
        with pytest.raises(RuntimeError, match="101"):
            asyncio.run(
                scan_finalized_blocks(
                    client,
                    store,
                    netuid=7,
                    track_start_block=100,
                )
            )

        assert store.chain_cursor() == (100, blocks[100][0])


def test_scanner_tolerates_another_process_advancing_finalized_cursor(tmp_path) -> None:
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    blocks = {height: chain_block(payload, height) for height in range(100, 102)}
    client = FakeClient(blocks)
    path = tmp_path / "validator.db"

    with ChallengeStore(path) as store, ChallengeStore(path) as other:
        original_events = client._substrate.events
        advanced = False

        async def racing_events(block_hash: str):
            nonlocal advanced
            if not advanced:
                for height in (100, 101):
                    for submission in extract_submissions(
                        block_height=height,
                        block_hash=blocks[height][0],
                        block=blocks[height][1],
                        events=blocks[height][2],
                        netuid=7,
                    ):
                        other.add_submission(submission)
                    other.advance_chain_cursor(height, blocks[height][0])
                advanced = True
            return await original_events(block_hash)

        client._substrate.events = racing_events
        result = asyncio.run(
            scan_finalized_blocks(
                client,
                store,
                netuid=7,
                track_start_block=100,
                max_blocks=2,
            )
        )

        assert result.blocks_scanned == 2
        assert store.chain_cursor() == (101, blocks[101][0])
        assert len(store.public_snapshot()["challenges"]) == 2


@pytest.mark.parametrize("mapping_shape", [False, True], ids=("sequence", "sdk-mapping"))
def test_scanner_processes_timelock_reveal_plaintext(tmp_path, mapping_shape) -> None:
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    block_hash = "0x" + f"{100:064x}"
    blocks = {
        100: (
            block_hash,
            {"extrinsics": []},
            [
                {
                    "extrinsic_idx": None,
                    "module_id": "Commitments",
                    "event_id": "CommitmentRevealed",
                    "attributes": {"netuid": 7, "who": HOTKEY},
                }
            ],
        )
    }
    client = FakeClient(blocks)
    commitment = SimpleNamespace(hotkey=HOTKEY, revealed=[(100, "0x" + payload.hex())])
    client.revealed = {100: {HOTKEY: commitment} if mapping_shape else [commitment]}

    with ChallengeStore(tmp_path / "validator.db") as store:
        result = asyncio.run(
            scan_finalized_blocks(
                client,
                store,
                netuid=7,
                track_start_block=100,
            )
        )

        assert result.submissions_found == 1
        challenge = store.public_snapshot()["challenges"][0]
        assert challenge["repo_id"] == "owner/model"


@pytest.mark.parametrize(
    "stored",
    [
        (),
        [(100, "0xnot-hex")],
        [(100, "first"), (100, "second")],
    ],
    ids=("missing", "undecodable", "ambiguous"),
)
def test_scanner_skips_bad_timelock_reveals_and_advances_cursor(tmp_path, caplog, stored) -> None:
    block_hash = "0x" + f"{100:064x}"
    blocks = {
        100: (
            block_hash,
            {"extrinsics": []},
            [
                {
                    "extrinsic_idx": None,
                    "module_id": "Commitments",
                    "event_id": "CommitmentRevealed",
                    "attributes": {"netuid": 7, "who": HOTKEY},
                }
            ],
        )
    }
    client = FakeClient(blocks)
    client.revealed = {100: [SimpleNamespace(hotkey=HOTKEY, revealed=stored)] if stored else []}

    with ChallengeStore(tmp_path / "validator.db") as store, caplog.at_level(logging.WARNING):
        result = asyncio.run(scan_finalized_blocks(client, store, netuid=7, track_start_block=100))

        assert result.submissions_found == 0
        assert store.chain_cursor() == (100, block_hash)
        assert store.public_snapshot()["challenges"] == []
    assert "skipping unmatched finalized commitment reveal" in caplog.text


def test_legacy_snapshot_decodes_hex_timelock_reveals(monkeypatch, tmp_path) -> None:
    activation = b"affine2|activate|signed"
    commitment = SimpleNamespace(revealed=[(99, "0x" + activation.hex())])
    neuron = SimpleNamespace(hotkey=HOTKEY, commitment=commitment)

    class SnapshotClient:
        async def at(self, height: int):
            assert height == 99
            return SimpleNamespace(
                subnets=SimpleNamespace(
                    metagraph=lambda _netuid, commitments: _ready(SimpleNamespace(neurons=[neuron]))
                )
            )

    monkeypatch.setattr(
        chain_module,
        "is_valid_legacy_activation",
        lambda payload, **_kwargs: payload == activation,
    )
    encoded = ModelSubmission("owner/model", "12" * 20).encode()
    block_hash, block, events = chain_block(encoded, 100)
    with ChallengeStore(tmp_path / "validator.db") as store:
        count = asyncio.run(
            seed_legacy_exclusions(
                SnapshotClient(),
                store,
                netuid=7,
                before_block=99,
            )
        )
        identifier, _ = store.add_submission(
            extract_submissions(
                block_height=100,
                block_hash=block_hash,
                block=block,
                events=events,
                netuid=7,
            )[0]
        )

        assert count == 1
        assert store.get_challenge(identifier).state is ChallengeState.REJECTED_OTHER_COMPETITION
