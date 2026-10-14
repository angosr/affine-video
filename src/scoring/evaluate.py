"""Deterministic aggregation for authoritative exact-cache scoring."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite, sqrt

import numpy as np

from scoring.metrics import log_cost_auc


@dataclass(frozen=True, slots=True)
class BudgetEvidence:
    """Per-sample relative errors for one fixed block budget."""

    budget: int
    calibrated_cost: float
    errors_by_group: Mapping[str, Sequence[float]]


@dataclass(frozen=True, slots=True)
class BudgetScore:
    budget: int
    calibrated_cost: float
    aggregate_error: float
    quality: float
    group_tail_errors: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ExactScore:
    score: float
    budgets: tuple[BudgetScore, ...]


def panel_seed(track_spec_digest: str, *, stage: str) -> int:
    """Derive one common hidden-panel seed for every submission on a track."""

    if not stage or not stage.isascii():
        raise ValueError("panel stage must be non-empty ASCII")
    digest = sha256()
    digest.update(b"veda-panel-seed\0")
    digest.update(_hex_bytes(track_spec_digest, name="track specification digest"))
    digest.update(stage.encode("ascii"))
    return int.from_bytes(digest.digest(), "big")


def stratified_sample_indices(
    case_strata: Sequence[str],
    count: int,
    seed: int,
    strata: Mapping[str, float],
) -> tuple[int, ...]:
    """Select deterministic case indices while covering every frozen stratum."""

    if count < len(strata) or count > len(case_strata):
        raise ValueError("sample count cannot cover the frozen strata")
    if not strata:
        raise ValueError("at least one frozen stratum is required")
    if (
        any(not isfinite(weight) or weight <= 0 for weight in strata.values())
        or abs(fsum(strata.values()) - 1.0) > 1e-12
    ):
        raise ValueError("stratum weights must be positive and sum to one")
    groups: dict[str, list[int]] = {stratum: [] for stratum in strata}
    for index, stratum in enumerate(case_strata):
        if stratum not in groups:
            raise ValueError("case references an unknown stratum")
        groups[stratum].append(index)
    if any(not indices for indices in groups.values()):
        raise ValueError("cache does not cover every frozen stratum")

    allocations = {stratum: 1 for stratum in strata}
    for _ in range(count - len(strata)):
        chosen = max(
            sorted(strata),
            key=lambda stratum: (strata[stratum] * count - allocations[stratum], stratum),
        )
        allocations[chosen] += 1
    if any(allocations[stratum] > len(groups[stratum]) for stratum in strata):
        raise ValueError("cache does not contain the frozen allocation for every stratum")

    generator = random.Random(seed)
    selected: list[int] = []
    for stratum in sorted(groups):
        candidates = groups[stratum].copy()
        generator.shuffle(candidates)
        selected.extend(candidates[: allocations[stratum]])
    generator.shuffle(selected)
    return tuple(selected)


def route_stratified_sample_indices(
    case_routes: Sequence[str],
    case_strata: Sequence[str],
    count: int,
    seed: int,
    routes: Sequence[str],
    strata: Mapping[str, float],
) -> tuple[int, ...]:
    """Sample by route and stratum so no selector route can evade evaluation."""

    if len(case_routes) != len(case_strata):
        raise ValueError("case routes and strata must be aligned")
    if not routes or len(set(routes)) != len(routes):
        raise ValueError("routes must be non-empty and unique")
    route_index = {route: index for index, route in enumerate(routes)}
    stratum_index = {stratum: index for index, stratum in enumerate(strata)}
    labels: list[str] = []
    for route, stratum in zip(case_routes, case_strata, strict=True):
        if route not in route_index:
            raise ValueError("case references an unknown route")
        if stratum not in stratum_index:
            raise ValueError("case references an unknown stratum")
        labels.append(f"{route_index[route]}:{stratum_index[stratum]}")
    cell_weights = {
        f"{route_position}:{stratum_index[stratum]}": weight / len(routes)
        for route_position in range(len(routes))
        for stratum, weight in strata.items()
    }
    return stratified_sample_indices(labels, count, seed, cell_weights)


def trip_pool(tiles: np.ndarray) -> np.ndarray:
    """Return mean, maximum, and minimum descriptors for token tiles."""

    values = np.asarray(tiles)
    if values.ndim < 2 or values.shape[-2] == 0 or values.shape[-1] == 0:
        raise ValueError("tiles must end in non-empty token and channel dimensions")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("tiles must contain finite floating-point values")
    descriptor = np.concatenate(
        (
            values.mean(axis=-2),
            values.max(axis=-2),
            values.min(axis=-2),
        ),
        axis=-1,
    )
    return descriptor.astype(np.float32, copy=False)


def stable_top_k(
    logits: np.ndarray,
    valid_keys: np.ndarray,
    *,
    blocks: int,
    required_keys: np.ndarray | None = None,
) -> np.ndarray:
    """Keep required keys, then fill an exact per-query budget by stable Top-k."""

    scores = np.asarray(logits)
    validity = np.asarray(valid_keys)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("logits must be a non-empty [query_tile, key_tile] matrix")
    if validity.shape != (scores.shape[1],) or validity.dtype != np.bool_:
        raise ValueError("valid_keys must be a boolean vector aligned with key tiles")
    if type(blocks) is not int or blocks <= 0:
        raise ValueError("block budget must be a positive integer")
    valid_indices = np.flatnonzero(validity)
    if len(valid_indices) < blocks:
        raise ValueError("block budget exceeds the number of valid key tiles")
    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("logits must be floating point")
    valid_scores = scores[:, valid_indices]
    if not np.isfinite(valid_scores).all():
        raise ValueError("valid logits must be finite")

    if required_keys is None:
        required = np.zeros(scores.shape, dtype=np.bool_)
    else:
        required = np.asarray(required_keys)
        if required.shape != scores.shape or required.dtype != np.bool_:
            raise ValueError("required_keys must be boolean and aligned with logits")
        if np.any(required & ~validity[None, :]):
            raise ValueError("required_keys cannot include an invalid key")
        if np.any(required.sum(axis=1) > blocks):
            raise ValueError("required key count exceeds the block budget")

    mask = required.copy()
    for query_index, row in enumerate(valid_scores):
        order = np.lexsort((valid_indices, -row))
        needed = blocks - int(mask[query_index].sum())
        if needed == 0:
            continue
        for candidate in valid_indices[order]:
            if not mask[query_index, candidate]:
                mask[query_index, candidate] = True
                needed -= 1
                if needed == 0:
                    break
    return mask


def reference_attention(
    q_tiles: np.ndarray,
    k_tiles: np.ndarray,
    v_tiles: np.ndarray,
    block_mask: np.ndarray,
) -> np.ndarray:
    """Compute deterministic float32 attention for a tile-level block mask."""

    q = np.asarray(q_tiles, dtype=np.float32)
    k = np.asarray(k_tiles, dtype=np.float32)
    v = np.asarray(v_tiles, dtype=np.float32)
    mask = np.asarray(block_mask)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q, K, and V must use [tile, token, channel] layout")
    if q.shape[-1] == 0 or q.shape[1] == 0 or k.shape[0] == 0 or k.shape[1] == 0:
        raise ValueError("attention dimensions must be non-empty")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q, K, and V dimensions are incompatible")
    if mask.shape != (q.shape[0], k.shape[0]) or mask.dtype != np.bool_:
        raise ValueError("block mask must be boolean and aligned with Q/K tiles")
    if not mask.any(axis=1).all():
        raise ValueError("every query tile must select at least one key tile")
    if not all(np.isfinite(value).all() for value in (q, k, v)):
        raise ValueError("Q, K, and V must be finite")

    output = np.empty_like(q)
    scale = np.float32(1 / sqrt(q.shape[-1]))
    for query_tile in range(q.shape[0]):
        selected = np.flatnonzero(mask[query_tile])
        keys = k[selected].reshape(-1, k.shape[-1])
        values = v[selected].reshape(-1, v.shape[-1])
        attention_logits = q[query_tile] @ keys.T * scale
        attention_logits -= attention_logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(attention_logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        output[query_tile] = probabilities @ values
    if not np.isfinite(output).all():
        raise ValueError("attention output is not finite")
    return output


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile with an explicit definition."""

    checked = _finite_nonnegative(values, name="percentile values")
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("percentile probability must be within [0, 1]")
    ordered = sorted(checked)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    if fraction == 0:
        return ordered[lower]
    return ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction


def tail_error(errors: Sequence[float], *, mean_weight: float) -> float:
    """Combine mean error and P95 tail error using the frozen mixture."""

    checked = _finite_nonnegative(errors, name="errors")
    if not isfinite(mean_weight) or not 0 <= mean_weight <= 1:
        raise ValueError("mean weight must be within [0, 1]")
    mean = fsum(checked) / len(checked)
    return mean_weight * mean + (1 - mean_weight) * percentile(checked, 0.95)


def aggregate_exact_score(
    evidence: Sequence[BudgetEvidence],
    *,
    group_weights: Mapping[str, float],
    tail_mean_weight: float,
) -> ExactScore:
    """Aggregate fixed-budget exact-cache evidence into the authoritative score."""

    if len(evidence) < 3:
        raise ValueError("exact evaluation requires at least three budgets")
    expected_groups = set(group_weights)
    if not expected_groups:
        raise ValueError("at least one evaluation group is required")
    if (
        any(not isfinite(weight) or weight <= 0 for weight in group_weights.values())
        or abs(fsum(group_weights.values()) - 1.0) > 1e-12
    ):
        raise ValueError("group weights must be positive and sum to one")
    if not isfinite(tail_mean_weight) or not 0 <= tail_mean_weight <= 1:
        raise ValueError("tail mean weight must be within [0, 1]")

    budgets: list[BudgetScore] = []
    previous_budget = -1
    previous_cost = 0.0
    for item in evidence:
        if item.budget <= previous_budget:
            raise ValueError("budgets must be strictly increasing")
        if not isfinite(item.calibrated_cost) or item.calibrated_cost <= previous_cost:
            raise ValueError("calibrated costs must be finite and strictly increasing")
        if set(item.errors_by_group) != expected_groups:
            raise ValueError("every budget must contain the exact evaluation groups")

        tails = {
            group: tail_error(
                item.errors_by_group[group],
                mean_weight=tail_mean_weight,
            )
            for group in sorted(expected_groups)
        }
        aggregate = fsum(group_weights[group] * tails[group] for group in tails)
        budgets.append(
            BudgetScore(
                budget=item.budget,
                calibrated_cost=item.calibrated_cost,
                aggregate_error=aggregate,
                quality=1.0 / (1.0 + aggregate),
                group_tail_errors=tails,
            )
        )
        previous_budget = item.budget
        previous_cost = item.calibrated_cost

    score = log_cost_auc(
        [item.calibrated_cost for item in budgets],
        [item.quality for item in budgets],
    )
    return ExactScore(score=score, budgets=tuple(budgets))


def stratified_bootstrap_bound(
    values_by_group: Mapping[str, Sequence[float]],
    *,
    group_weights: Mapping[str, float],
    confidence: float,
    resamples: int,
    seed: int,
    upper: bool,
) -> float:
    """Bootstrap a fixed-weight macro mean while resampling within each group."""

    if set(values_by_group) != set(group_weights) or not values_by_group:
        raise ValueError("bootstrap values must cover the exact evaluation groups")
    checked = {
        group: _finite_unit_interval(values, name=f"bootstrap group {group}")
        for group, values in values_by_group.items()
    }
    if (
        any(not isfinite(weight) or weight <= 0 for weight in group_weights.values())
        or abs(fsum(group_weights.values()) - 1.0) > 1e-12
    ):
        raise ValueError("group weights must be positive and sum to one")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be within (0, 1)")
    if resamples <= 0 or seed < 0:
        raise ValueError("bootstrap resamples must be positive and seed non-negative")

    generator = random.Random(seed)
    statistics: list[float] = []
    for _ in range(resamples):
        macro = 0.0
        for group in sorted(checked):
            values = checked[group]
            mean = fsum(values[generator.randrange(len(values))] for _ in values) / len(values)
            macro += group_weights[group] * mean
        statistics.append(macro)
    return percentile(statistics, confidence if upper else 1 - confidence)


def _finite_nonnegative(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    checked = tuple(values)
    if not checked or any(not isfinite(value) or value < 0 for value in checked):
        raise ValueError(f"{name} must be non-empty, finite, and non-negative")
    return checked


def _finite_unit_interval(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    checked = tuple(values)
    if not checked or any(not isfinite(value) or not 0 <= value <= 1 for value in checked):
        raise ValueError(f"{name} must be non-empty, finite, and within [0, 1]")
    return checked


def _hex_bytes(value: str, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be hexadecimal")
    try:
        result = bytes.fromhex(value.removeprefix("0x"))
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    if len(result) != 32:
        raise ValueError(f"{name} must contain 32 bytes")
    return result
