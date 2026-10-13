"""Deterministic leaderboard-to-chain weight planning and submission."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import bittensor as bt
from bittensor.intents import normalize as normalize_weights

from validator.store import LeaderboardEntry

_HEX_32 = re.compile(r"^[0-9a-f]{64}$")


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
class WeightArtifact:
    """One public recommendation produced by the evaluation owner."""

    track_id: str
    track_spec_digest: str
    netuid: int
    mechid: int
    version_key: int
    source_block: int
    plan: WeightPlan | None

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track ID cannot be empty")
        if not _HEX_32.fullmatch(self.track_spec_digest):
            raise ValueError("track specification digest is invalid")
        if not 0 <= self.netuid <= 0xFFFF:
            raise ValueError("netuid must fit in an unsigned 16-bit integer")
        if min(self.mechid, self.version_key, self.source_block) < 0:
            raise ValueError("weight artifact chain values must be non-negative")
        if self.plan is not None and self.plan.source_block != self.source_block:
            raise ValueError("weight plan source block does not match its artifact")


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
    last_update_by_hotkey: Mapping[str, int]


async def read_weight_chain_state(
    client: Any,
    *,
    netuid: int,
    finalized_block: int | None = None,
) -> WeightChainState:
    """Read all planning inputs from one finalized chain snapshot."""

    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    finalized = await client.finalized_block() if finalized_block is None else finalized_block
    if finalized < 0:
        raise ValueError("finalized block must be non-negative")
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
    last_update_by_hotkey: dict[str, int] = {}
    seen_uids: set[int] = set()
    for neuron in graph.neurons:
        hotkey = neuron.hotkey
        uid = neuron.uid
        last_update = neuron.last_update
        if not isinstance(hotkey, str) or not hotkey:
            raise RuntimeError("metagraph contains an invalid hotkey")
        if type(uid) is not int or not 0 <= uid <= 0xFFFF:
            raise RuntimeError("metagraph contains an invalid UID")
        if hotkey in uid_by_hotkey or uid in seen_uids:
            raise RuntimeError("metagraph hotkey and UID mapping is not unique")
        if type(last_update) is not int or last_update < 0 or last_update > finalized:
            raise RuntimeError("metagraph contains an invalid last-update block")
        uid_by_hotkey[hotkey] = uid
        last_update_by_hotkey[hotkey] = last_update
        seen_uids.add(uid)
    return WeightChainState(
        finalized_block=finalized,
        uid_by_hotkey=uid_by_hotkey,
        min_allowed_weights=minimum,
        max_weight=maximum_raw / 0xFFFF,
        weights_rate_limit=rate_limit,
        last_update_by_hotkey=last_update_by_hotkey,
    )


def frontier_gains(entries: Iterable[LeaderboardEntry], *, minimum_gain: float) -> dict[str, float]:
    """Credit only score improvements over earlier on-chain submissions.

    Gains telescope to the best score, so copied or Sybil-split models cannot
    multiply their total utility merely by occupying more hotkeys.
    """

    if not isfinite(minimum_gain) or not 0 < minimum_gain <= 1:
        raise ValueError("minimum frontier gain must be within (0, 1]")
    ordered = sorted(entries, key=lambda entry: entry.position)
    frontier = 0.0
    gains: dict[str, float] = {}
    for entry in ordered:
        if not isfinite(entry.final_score) or not 0 <= entry.final_score <= 1:
            raise ValueError("leaderboard scores must be finite and within [0, 1]")
        gain = entry.final_score - frontier
        if gain >= minimum_gain:
            gains[entry.hotkey] = gains.get(entry.hotkey, 0.0) + gain
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
    client: Any,
    wallet: Any,
    netuid: int,
    mechid: int = 0,
    version_key: int = 0,
) -> WeightReceipt:
    """Submit a planned vector through Bittensor and wait for finalization."""

    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    if mechid < 0 or version_key < 0:
        raise ValueError("mechanism and version identifiers must be non-negative")

    intent = bt.SetWeights(
        netuid=netuid,
        uids=list(plan.uids),
        weights=list(plan.weights),
        mechid=mechid,
        version_key=version_key,
    )
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


def validate_weight_artifact(
    artifact: WeightArtifact,
    chain: WeightChainState,
    *,
    expected_netuid: int,
    expected_track_digest: str,
    max_staleness_blocks: int,
) -> WeightPlan | None:
    """Validate an untrusted public recommendation against current finalized chain state."""

    if expected_netuid < 0 or expected_netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    normalized_digest = expected_track_digest.lower()
    if not _HEX_32.fullmatch(normalized_digest):
        raise ValueError("expected track specification digest is invalid")
    if max_staleness_blocks <= 0:
        raise ValueError("maximum staleness must be positive")
    if artifact.netuid != expected_netuid:
        raise ValueError("public weights target a different subnet")
    if artifact.track_spec_digest != normalized_digest:
        raise ValueError("public weights target a different track specification")
    if artifact.source_block > chain.finalized_block:
        raise ValueError("public weights come from a future block")
    if chain.finalized_block - artifact.source_block > max_staleness_blocks:
        raise ValueError("public weights are stale")
    plan = artifact.plan
    if plan is None:
        return None
    required = max(1, chain.min_allowed_weights)
    if len(plan.hotkeys) < required:
        raise ValueError("public weights do not meet the chain minimum")
    quantized_uids, _ = normalize_weights(plan.uids, plan.weights)
    if len(quantized_uids) < required:
        raise ValueError("public weights fall below the chain minimum after quantization")
    if any(weight > chain.max_weight + 1e-12 for weight in plan.weights):
        raise ValueError("public weights exceed the current chain cap")
    for hotkey, uid in zip(plan.hotkeys, plan.uids, strict=True):
        if chain.uid_by_hotkey.get(hotkey) != uid:
            raise ValueError("public hotkey and UID mapping is no longer current")
    return plan


def weight_update_due(
    artifact: WeightArtifact,
    chain: WeightChainState,
    *,
    validator_hotkey: str,
) -> tuple[bool, str]:
    """Decide whether this Validator can submit the fetched artifact now."""

    if validator_hotkey not in chain.uid_by_hotkey:
        raise ValueError("validator hotkey is not registered on the subnet")
    last_update = chain.last_update_by_hotkey[validator_hotkey]
    if artifact.source_block <= last_update:
        return False, "ALREADY_APPLIED"
    if chain.finalized_block < last_update + chain.weights_rate_limit:
        return False, "RATE_LIMIT"
    return True, "READY"


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
