import asyncio
from types import SimpleNamespace

import pytest

from miner.submit import ModelSubmission
from validator.chain import scan_finalized_blocks
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


def test_scanner_processes_timelock_reveal_plaintext(tmp_path) -> None:
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
    client.revealed = {
        100: [SimpleNamespace(hotkey=HOTKEY, revealed=[(100, "0x" + payload.hex())])]
    }

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
