"""Finalized-chain synchronization for the validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from validator.scan import extract_submissions
from validator.store import ChallengeStore


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
    cooldown_blocks: int,
    max_blocks: int = 100,
) -> ScanResult:
    """Scan one bounded batch and commit the cursor after each complete block."""

    if track_start_block < 0:
        raise ValueError("track start block must be non-negative")
    if cooldown_blocks < 0:
        raise ValueError("cooldown blocks must be non-negative")
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
        block = await client._substrate.get_block(block_hash=block_hash)
        if block is None:
            raise RuntimeError(f"finalized block {block_height} is unavailable")
        events = await client._substrate.events(block_hash)
        submissions = extract_submissions(
            block_height=block_height,
            block_hash=block_hash,
            block=block,
            events=events,
            netuid=netuid,
        )
        for submission in submissions:
            store.add_submission(
                submission,
                cooldown_blocks=cooldown_blocks,
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
