"""Deterministic aggregation for the exact local scoring stage."""

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
    errors_by_stratum: Mapping[str, Sequence[float]]


@dataclass(frozen=True, slots=True)
class BudgetScore:
    budget: int
    calibrated_cost: float
    aggregate_error: float
    quality: float
    stratum_tail_errors: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LocalScore:
    score: float
    budgets: tuple[BudgetScore, ...]


def challenge_seed(
    track_spec_digest: str,
    submission_id: str,
    seed_block_hash: str,
) -> int:
    """Derive the deterministic hidden-sample seed fixed by the challenge record."""

    components = (
        _hex_bytes(track_spec_digest, name="track specification digest"),
        _hex_bytes(submission_id, name="submission identifier"),
        _hex_bytes(seed_block_hash, name="seed block hash"),
    )
    return int.from_bytes(sha256(b"".join(components)).digest(), "big")


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
    groups: dict[str, list[int]] = {stratum: [] for stratum in strata}
    for index, stratum in enumerate(case_strata):
        if stratum not in groups:
            raise ValueError("case references an unknown stratum")
        groups[stratum].append(index)
    if any(not indices for indices in groups.values()):
        raise ValueError("cache does not cover every frozen stratum")

    generator = random.Random(seed)
    selected: list[int] = []
    for stratum in sorted(groups):
        candidates = groups[stratum].copy()
        generator.shuffle(candidates)
        selected.append(candidates[0])
    selected_set = set(selected)
    remaining = [index for index in range(len(case_strata)) if index not in selected_set]
    generator.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    return tuple(selected)


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


def stable_top_k(logits: np.ndarray, valid_keys: np.ndarray, *, blocks: int) -> np.ndarray:
    """Select exactly ``blocks`` keys per query with key index as the tie-breaker."""

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

    mask = np.zeros(scores.shape, dtype=np.bool_)
    for query_index, row in enumerate(valid_scores):
        order = np.lexsort((valid_indices, -row))
        mask[query_index, valid_indices[order[:blocks]]] = True
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


def evaluate_local(
    evidence: Sequence[BudgetEvidence],
    *,
    stratum_weights: Mapping[str, float],
    tail_mean_weight: float,
) -> LocalScore:
    """Aggregate fixed-budget exact-cache evidence into the local score."""

    if len(evidence) < 3:
        raise ValueError("local evaluation requires at least three budgets")
    expected_strata = set(stratum_weights)
    if not expected_strata:
        raise ValueError("at least one stratum is required")
    if (
        any(not isfinite(weight) or weight <= 0 for weight in stratum_weights.values())
        or abs(fsum(stratum_weights.values()) - 1.0) > 1e-12
    ):
        raise ValueError("stratum weights must be positive and sum to one")
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
        if set(item.errors_by_stratum) != expected_strata:
            raise ValueError("every budget must contain the exact frozen strata")

        tails = {
            stratum: tail_error(
                item.errors_by_stratum[stratum],
                mean_weight=tail_mean_weight,
            )
            for stratum in sorted(expected_strata)
        }
        aggregate = fsum(stratum_weights[stratum] * tails[stratum] for stratum in tails)
        budgets.append(
            BudgetScore(
                budget=item.budget,
                calibrated_cost=item.calibrated_cost,
                aggregate_error=aggregate,
                quality=1.0 / (1.0 + aggregate),
                stratum_tail_errors=tails,
            )
        )
        previous_budget = item.budget
        previous_cost = item.calibrated_cost

    score = log_cost_auc(
        [item.calibrated_cost for item in budgets],
        [item.quality for item in budgets],
    )
    return LocalScore(score=score, budgets=tuple(budgets))


def bootstrap_mean_lower_bound(
    values: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> float:
    """Return a deterministic percentile-bootstrap lower bound for the mean."""

    checked = _finite_unit_interval(values, name="bootstrap values")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be within (0, 1)")
    if resamples <= 0:
        raise ValueError("resample count must be positive")
    if seed < 0:
        raise ValueError("bootstrap seed must be non-negative")

    generator = random.Random(seed)
    count = len(checked)
    means = [
        fsum(checked[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    return percentile(means, 1 - confidence)


def bootstrap_mean_upper_bound(
    values: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> float:
    """Return a deterministic percentile-bootstrap upper bound for the mean."""

    checked = _finite_unit_interval(values, name="bootstrap values")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be within (0, 1)")
    if resamples <= 0:
        raise ValueError("resample count must be positive")
    if seed < 0:
        raise ValueError("bootstrap seed must be non-negative")

    generator = random.Random(seed)
    count = len(checked)
    means = [
        fsum(checked[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    return percentile(means, confidence)


def stratified_bootstrap_bound(
    values_by_stratum: Mapping[str, Sequence[float]],
    *,
    stratum_weights: Mapping[str, float],
    confidence: float,
    resamples: int,
    seed: int,
    upper: bool,
) -> float:
    """Bootstrap a fixed-weight macro mean while resampling within each stratum."""

    if set(values_by_stratum) != set(stratum_weights) or not values_by_stratum:
        raise ValueError("bootstrap values must cover the exact frozen strata")
    checked = {
        stratum: _finite_unit_interval(values, name=f"bootstrap stratum {stratum}")
        for stratum, values in values_by_stratum.items()
    }
    if (
        any(not isfinite(weight) or weight <= 0 for weight in stratum_weights.values())
        or abs(fsum(stratum_weights.values()) - 1.0) > 1e-12
    ):
        raise ValueError("stratum weights must be positive and sum to one")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be within (0, 1)")
    if resamples <= 0 or seed < 0:
        raise ValueError("bootstrap resamples must be positive and seed non-negative")

    generator = random.Random(seed)
    statistics: list[float] = []
    for _ in range(resamples):
        macro = 0.0
        for stratum in sorted(checked):
            values = checked[stratum]
            mean = fsum(values[generator.randrange(len(values))] for _ in values) / len(values)
            macro += stratum_weights[stratum] * mean
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
