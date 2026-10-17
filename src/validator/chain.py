"""Finalized-chain synchronization for the validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from validator.competition import LEGACY_ACTIVATION_PREFIX, is_valid_legacy_activation
from validator.scan import extract_commitments, extract_submissions
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
    max_blocks: int = 100,
) -> ScanResult:
    """Scan one bounded batch and commit the cursor after each complete block."""

    if track_start_block < 0:
        raise ValueError("track start block must be non-negative")
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
        commitments = extract_commitments(
            block_height=block_height,
            block_hash=block_hash,
            block=block,
            events=events,
            netuid=netuid,
        )
        submissions = extract_submissions(
            block_height=block_height,
            block_hash=block_hash,
            block=block,
            events=events,
            netuid=netuid,
        )
        submissions_by_position = {
            submission.position: submission for submission in submissions
        }
        for commitment in commitments:
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
                legacy_excluded=is_valid_legacy_activation(
                    commitment.payload,
                    hotkey=commitment.hotkey,
                    netuid=netuid,
                ),
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
    for neuron in graph.neurons:
        hotkey = getattr(neuron, "hotkey", None)
        commitment = getattr(neuron, "commitment", None)
        revealed = getattr(commitment, "revealed", ()) if commitment is not None else ()
        if not isinstance(hotkey, str) or not hotkey:
            continue
        for item in revealed or ():
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            block, value = item
            if type(block) is not int or block < 0 or block > before_block:
                continue
            if isinstance(value, str):
                payload = value.encode("utf-8")
            elif isinstance(value, (bytes, bytearray)):
                payload = bytes(value)
            else:
                continue
            if not payload.startswith(LEGACY_ACTIVATION_PREFIX):
                continue
            if is_valid_legacy_activation(payload, hotkey=hotkey, netuid=netuid):
                exclusions[hotkey] = min(block, exclusions.get(hotkey, block))
    store.seed_legacy_exclusions(exclusions)
    return len(exclusions)
