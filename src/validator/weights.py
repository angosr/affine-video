"""Deterministic leaderboard-to-chain weight planning and submission."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import bittensor as bt
from bittensor.intents import normalize as normalize_weights

from validator.store import ChallengeStore, LeaderboardEntry, WeightRun


class WeightSubmissionError(RuntimeError):
    """Raised when a weight transaction does not finalize successfully."""


@dataclass(frozen=True, slots=True)
class WeightPlan:
    """One normalized weight vector tied to a finalized source block."""

    source_block: int
    hotkeys: tuple[str, ...]
    uids: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.hotkeys)
        if self.source_block < 0:
            raise ValueError("source block must be non-negative")
        if size == 0 or len(self.uids) != size or len(self.weights) != size:
            raise ValueError("weight plan fields must be non-empty and aligned")
        if len(set(self.hotkeys)) != size or len(set(self.uids)) != size:
            raise ValueError("weight plan hotkeys and UIDs must be unique")
        if tuple(sorted(self.uids)) != self.uids:
            raise ValueError("weight plan must be ordered by UID")
        if any(uid < 0 or uid > 0xFFFF for uid in self.uids):
            raise ValueError("UIDs must fit in an unsigned 16-bit integer")
        if any(not isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError("planned weights must be finite and positive")
        if abs(sum(self.weights) - 1.0) > 1e-12:
            raise ValueError("planned weights must sum to one")


@dataclass(frozen=True, slots=True)
class WeightReceipt:
    block_hash: str
    extrinsic_id: str
    message: str


@dataclass(frozen=True, slots=True)
class WeightChainState:
    finalized_block: int
    uid_by_hotkey: Mapping[str, int]
    min_allowed_weights: int
    max_weight: float
    weights_rate_limit: int


async def read_weight_chain_state(client: Any, *, netuid: int) -> WeightChainState:
    """Read all planning inputs from one finalized chain snapshot."""

    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    finalized = await client.finalized_block()
    snapshot = await client.at(finalized)
    graph, minimum, maximum_raw, rate_limit = await asyncio.gather(
        snapshot.subnets.metagraph(netuid, commitments=False),
        snapshot.read("min_allowed_weights", netuid=netuid),
        snapshot.read("max_weight_limit", netuid=netuid),
        snapshot.read("weights_rate_limit", netuid=netuid),
    )
    if graph is None:
        raise RuntimeError(f"subnet {netuid} does not exist at finalized block {finalized}")
    if type(minimum) is not int or minimum < 0:
        raise RuntimeError("chain returned an invalid min_allowed_weights value")
    if type(maximum_raw) is not int or not 0 < maximum_raw <= 0xFFFF:
        raise RuntimeError("chain returned an invalid max_weight_limit value")
    if type(rate_limit) is not int or rate_limit < 0:
        raise RuntimeError("chain returned an invalid weights_rate_limit value")

    uid_by_hotkey: dict[str, int] = {}
    seen_uids: set[int] = set()
    for neuron in graph.neurons:
        hotkey = neuron.hotkey
        uid = neuron.uid
        if not isinstance(hotkey, str) or not hotkey:
            raise RuntimeError("metagraph contains an invalid hotkey")
        if type(uid) is not int or not 0 <= uid <= 0xFFFF:
            raise RuntimeError("metagraph contains an invalid UID")
        if hotkey in uid_by_hotkey or uid in seen_uids:
            raise RuntimeError("metagraph hotkey and UID mapping is not unique")
        uid_by_hotkey[hotkey] = uid
        seen_uids.add(uid)
    return WeightChainState(
        finalized_block=finalized,
        uid_by_hotkey=uid_by_hotkey,
        min_allowed_weights=minimum,
        max_weight=maximum_raw / 0xFFFF,
        weights_rate_limit=rate_limit,
    )


def frontier_gains(
    entries: Iterable[LeaderboardEntry], *, minimum_gain: float
) -> dict[str, float]:
    """Credit only score improvements over earlier on-chain submissions.

    Gains telescope to the best score, so copied or Sybil-split models cannot
    multiply their total utility merely by occupying more hotkeys.
    """

    if not isfinite(minimum_gain) or not 0 < minimum_gain <= 1:
        raise ValueError("minimum frontier gain must be within (0, 1]")
    ordered = sorted(entries, key=lambda entry: entry.position)
    frontier = 0.0
    gains: dict[str, float] = {}
    seen: set[str] = set()
    for entry in ordered:
        if entry.hotkey in seen:
            raise ValueError("leaderboard contains a duplicate hotkey")
        seen.add(entry.hotkey)
        if not isfinite(entry.final_score) or not 0 <= entry.final_score <= 1:
            raise ValueError("leaderboard scores must be finite and within [0, 1]")
        gain = entry.final_score - frontier
        if gain >= minimum_gain:
            gains[entry.hotkey] = gain
            frontier = entry.final_score
    return gains


def plan_weights(
    entries: Iterable[LeaderboardEntry],
    *,
    uid_by_hotkey: Mapping[str, int],
    current_block: int,
    min_allowed_weights: int,
    max_weight: float,
    minimum_frontier_gain: float,
) -> WeightPlan | None:
    """Build a chain-ready vector, or return None when it is infeasible."""

    if current_block < 0:
        raise ValueError("current block must be non-negative")
    if min_allowed_weights < 0:
        raise ValueError("minimum allowed weights must be non-negative")
    if not isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("maximum weight must be within (0, 1]")
    eligible_entries = [
        entry
        for entry in entries
        if entry.hotkey in uid_by_hotkey
        and not entry.provisional
        and entry.score_block <= current_block
    ]
    utilities = frontier_gains(eligible_entries, minimum_gain=minimum_frontier_gain)
    caps: dict[str, float] = {}
    seen_uids: dict[int, str] = {}
    for entry in eligible_entries:
        uid = uid_by_hotkey.get(entry.hotkey)
        if uid is None:
            raise AssertionError("eligible entry lost its UID")
        if uid < 0 or uid > 0xFFFF:
            raise ValueError("UIDs must fit in an unsigned 16-bit integer")
        other = seen_uids.setdefault(uid, entry.hotkey)
        if other != entry.hotkey:
            raise ValueError("multiple hotkeys map to the same UID")
        if entry.hotkey not in utilities:
            continue
        caps[entry.hotkey] = max_weight

    required = max(1, min_allowed_weights)
    if len(utilities) < required or sum(caps.values()) < 1 - 1e-12:
        return None

    allocated = waterfill_weights(utilities, caps=caps)
    ordered = sorted(allocated, key=lambda hotkey: uid_by_hotkey[hotkey])
    ordered_uids = tuple(uid_by_hotkey[hotkey] for hotkey in ordered)
    ordered_weights = tuple(allocated[hotkey] for hotkey in ordered)
    quantized_uids, _ = normalize_weights(ordered_uids, ordered_weights)
    if len(quantized_uids) < required:
        return None
    return WeightPlan(
        source_block=current_block,
        hotkeys=tuple(ordered),
        uids=ordered_uids,
        weights=ordered_weights,
    )


async def submit_weights(
    plan: WeightPlan,
    *,
    netuid: int,
    network: str,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None = None,
    mechid: int = 0,
    version_key: int = 0,
) -> WeightReceipt:
    """Submit a planned vector through Bittensor and wait for finalization."""

    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    if not network:
        raise ValueError("network cannot be empty")
    if mechid < 0 or version_key < 0:
        raise ValueError("mechanism and version identifiers must be non-negative")

    wallet_arguments = {"name": wallet_name, "hotkey": hotkey_name}
    if wallet_path is not None:
        wallet_arguments["path"] = wallet_path
    wallet = bt.Wallet(**wallet_arguments)
    intent = bt.SetWeights(
        netuid=netuid,
        uids=list(plan.uids),
        weights=list(plan.weights),
        mechid=mechid,
        version_key=version_key,
    )
    async with bt.Client(network) as client:
        result = await client.execute(
            intent,
            wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

    if not result.success:
        raise WeightSubmissionError(result.message or "weight transaction failed")
    if not result.block_hash or not result.extrinsic_id:
        raise WeightSubmissionError("finalized weight transaction is missing chain identifiers")
    return WeightReceipt(
        block_hash=result.block_hash,
        extrinsic_id=result.extrinsic_id,
        message=result.message,
    )


async def submit_recorded_weights(
    store: ChallengeStore,
    plan: WeightPlan,
    *,
    netuid: int,
    network: str,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None = None,
    mechid: int = 0,
    version_key: int = 0,
) -> tuple[WeightRun, WeightReceipt]:
    """Record one attempt, submit it, and persist the final chain outcome."""

    run = store.create_weight_run(
        netuid=netuid,
        mechid=mechid,
        source_block=plan.source_block,
        hotkeys=plan.hotkeys,
        uids=plan.uids,
        weights=plan.weights,
    )
    try:
        receipt = await submit_weights(
            plan,
            netuid=netuid,
            network=network,
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
            wallet_path=wallet_path,
            mechid=mechid,
            version_key=version_key,
        )
    except Exception as exc:
        store.fail_weight_run(
            run.run_id,
            message=str(exc) or type(exc).__name__,
        )
        raise
    finalized = store.finalize_weight_run(
        run.run_id,
        extrinsic_id=receipt.extrinsic_id,
        block_hash=receipt.block_hash,
        message=receipt.message,
    )
    return finalized, receipt


async def update_weights(
    client: Any,
    store: ChallengeStore,
    *,
    netuid: int,
    network: str,
    wallet_name: str,
    hotkey_name: str,
    interval_blocks: int,
    minimum_frontier_gain: float,
    wallet_path: str | None = None,
    mechid: int = 0,
    version_key: int = 0,
) -> WeightRun | None:
    """Submit at most one due vector built from a finalized chain snapshot."""

    if interval_blocks <= 0:
        raise ValueError("weight interval must be positive")
    chain = await read_weight_chain_state(client, netuid=netuid)
    cursor = store.chain_cursor()
    if cursor is None or cursor[0] != chain.finalized_block:
        return None
    latest = store.latest_weight_run()
    effective_interval = max(interval_blocks, chain.weights_rate_limit)
    if latest is not None and chain.finalized_block < latest.source_block + effective_interval:
        return None
    plan = plan_weights(
        store.leaderboard(),
        uid_by_hotkey=chain.uid_by_hotkey,
        current_block=chain.finalized_block,
        min_allowed_weights=chain.min_allowed_weights,
        max_weight=chain.max_weight,
        minimum_frontier_gain=minimum_frontier_gain,
    )
    if plan is None:
        return None
    run, _receipt = await submit_recorded_weights(
        store,
        plan,
        netuid=netuid,
        network=network,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        wallet_path=wallet_path,
        mechid=mechid,
        version_key=version_key,
    )
    return run


def waterfill_weights(
    utilities: Mapping[str, float], *, caps: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Normalize positive utilities while respecting per-hotkey caps."""

    caps = caps or {}
    positive: dict[str, float] = {}
    limits: dict[str, float] = {}

    for hotkey in sorted(utilities):
        utility = utilities[hotkey]
        cap = caps.get(hotkey, 1.0)
        if not isfinite(utility) or utility < 0:
            raise ValueError("utilities must be finite and non-negative")
        if not isfinite(cap) or not 0 < cap <= 1:
            raise ValueError("caps must be finite and within (0, 1]")
        if utility > 0:
            positive[hotkey] = utility
            limits[hotkey] = cap

    if not positive:
        return {}
    if sum(limits.values()) < 1 - 1e-12:
        raise ValueError("positive hotkey caps cannot hold a normalized vector")

    result: dict[str, float] = {}
    active = set(positive)
    remaining = 1.0

    while active:
        utility_sum = sum(positive[hotkey] for hotkey in active)
        capped = [
            hotkey
            for hotkey in sorted(active)
            if remaining * positive[hotkey] / utility_sum > limits[hotkey] + 1e-15
        ]
        if not capped:
            for hotkey in sorted(active):
                result[hotkey] = remaining * positive[hotkey] / utility_sum
            break

        for hotkey in capped:
            result[hotkey] = limits[hotkey]
            remaining -= limits[hotkey]
            active.remove(hotkey)

        if remaining < -1e-12:
            raise AssertionError("water-filling allocated more than one")

    total = sum(result.values())
    if abs(total - 1.0) > 1e-12:
        raise AssertionError("water-filling did not produce a normalized vector")
    return result
