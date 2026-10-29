"""Exact cached-QKV scoring with the same segmented mask contract as VEDA."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np

from scoring.evaluate import (
    percentile,
    reference_attention,
    segmented_top_k,
    tail_error,
    trip_pool,
)
from scoring.metrics import log_cost_auc, relative_squared_error
from scoring.model import ModelValidationError
from scoring.selector import FixedSelector
from scoring.spec import BudgetSpec

TARGET_SEGMENT = 0
REFERENCE_SEGMENT = 1
GLOBAL_SEGMENT = 2


@dataclass(frozen=True, slots=True)
class ExactCase:
    route: str
    stratum: str
    rollout_id: str
    source_id: str
    step_index: int
    step_window: str
    scheduler_timestep: float
    sigma: float
    tile_plan_hash: str
    query_tile_ids: tuple[int, ...]
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    dense: np.ndarray
    query_valid_counts: np.ndarray
    key_valid_counts: np.ndarray
    key_segments: np.ndarray
    required_keys: np.ndarray
    epsilon: float


@dataclass(frozen=True, slots=True)
class CaseDesign:
    """Small immutable sampling identity retained after a case leaves memory."""

    route: str
    stratum: str
    rollout_id: str
    source_id: str
    step_index: int
    step_window: str


@dataclass(frozen=True, slots=True)
class PairedSelectorScore:
    challenger_score: float
    incumbent_score: float
    gain_lower_bound: float
    challenger_cells: tuple[float, ...]
    incumbent_cells: tuple[float, ...]


def score_selector_pair(
    challenger: FixedSelector,
    incumbent: FixedSelector,
    cases: Iterable[ExactCase],
    budgets: Sequence[BudgetSpec],
    *,
    stratum_weights: Mapping[str, float],
    step_window_weights: Mapping[str, float],
    tail_mean_weight: float,
    confidence: float,
    resamples: int,
    seed: int,
    device: str | None = None,
) -> PairedSelectorScore:
    """Score both sides case-by-case so Q/K/V never accumulate in memory."""

    design: list[CaseDesign] = []
    challenger_cells: list[float] = []
    incumbent_cells: list[float] = []
    costs = [budget.calibrated_cost for budget in budgets]
    for case in cases:
        if challenger.architecture != incumbent.architecture:
            raise ValueError("paired selectors must use the same architecture")
        if device is None:
            challenger_quality, _ = _score_case_or_zero(
                challenger, case, budgets, tail_mean_weight
            )
            incumbent_quality, _ = _score_case_or_zero(
                incumbent, case, budgets, tail_mean_weight
            )
        else:
            from scoring.torch_backend import TorchCaseScorer

            _validate_case(case, challenger)
            gpu_case = TorchCaseScorer(case, device=device)
            challenger_quality, _ = _score_gpu_case_or_zero(
                gpu_case, challenger, budgets, tail_mean_weight
            )
            incumbent_quality, _ = _score_gpu_case_or_zero(
                gpu_case, incumbent, budgets, tail_mean_weight
            )
        challenger_cells.append(log_cost_auc(costs, challenger_quality))
        incumbent_cells.append(log_cost_auc(costs, incumbent_quality))
        design.append(
            CaseDesign(
                case.route,
                case.stratum,
                case.rollout_id,
                case.source_id,
                case.step_index,
                case.step_window,
            )
        )
        del case
    if not design:
        raise ValueError("paired exact evaluation requires at least one case")
    challenger_score = _weighted_cell_mean(
        challenger_cells, design, stratum_weights, step_window_weights
    )
    incumbent_score = _weighted_cell_mean(
        incumbent_cells, design, stratum_weights, step_window_weights
    )
    bound = paired_score_gain_bound(
        challenger_cells,
        incumbent_cells,
        design,
        stratum_weights=stratum_weights,
        step_window_weights=step_window_weights,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return PairedSelectorScore(
        challenger_score,
        incumbent_score,
        bound,
        tuple(challenger_cells),
        tuple(incumbent_cells),
    )


def paired_score_gain_bound(
    challenger_scores: Sequence[float],
    incumbent_scores: Sequence[float],
    cases: Sequence[ExactCase | CaseDesign],
    *,
    stratum_weights: Mapping[str, float],
    step_window_weights: Mapping[str, float],
    confidence: float,
    resamples: int,
    seed: int,
) -> float:
    """Bootstrap paired gains by route, independent source video, and step."""

    if len(challenger_scores) != len(cases) or len(incumbent_scores) != len(cases):
        raise ValueError("paired scores must align with every exact case")
    if not 0 < confidence < 1 or resamples <= 0 or seed < 0:
        raise ValueError("bootstrap configuration is invalid")
    differences = tuple(
        challenger - incumbent
        for challenger, incumbent in zip(challenger_scores, incumbent_scores, strict=True)
    )
    _validate_complete_design(cases, stratum_weights, step_window_weights)
    _validate_independent_videos(cases, stratum_weights)
    by_key = {
        (case.route, case.stratum, case.rollout_id, case.step_index): differences[index]
        for index, case in enumerate(cases)
    }
    routes = tuple(sorted({case.route for case in cases}))
    rollouts = {
        stratum: tuple(sorted({case.rollout_id for case in cases if case.stratum == stratum}))
        for stratum in stratum_weights
    }
    steps = {
        window: tuple(sorted({case.step_index for case in cases if case.step_window == window}))
        for window in step_window_weights
    }
    generator = random.Random(seed)
    statistics = []
    for _ in range(resamples):
        sampled_routes = [routes[generator.randrange(len(routes))] for _ in routes]
        sampled_rollouts = {
            stratum: [items[generator.randrange(len(items))] for _ in items]
            for stratum, items in rollouts.items()
        }
        sampled_steps = {
            window: [items[generator.randrange(len(items))] for _ in items]
            for window, items in steps.items()
        }
        value = 0.0
        for stratum, stratum_weight in stratum_weights.items():
            for window, step_weight in step_window_weights.items():
                selected = [
                    by_key[(route, stratum, rollout, step)]
                    for route in sampled_routes
                    for rollout in sampled_rollouts[stratum]
                    for step in sampled_steps[window]
                ]
                value += stratum_weight * step_weight * fsum(selected) / len(selected)
        statistics.append(value)
    return percentile(statistics, 1 - confidence)


def _score_case(
    selector: FixedSelector,
    case: ExactCase,
    budgets: Sequence[BudgetSpec],
    tail_mean_weight: float,
) -> tuple[list[float], list[list[float]]]:
    _validate_case(case, selector)
    logits = selector.logits(
        case.route,
        trip_pool(case.q, case.query_valid_counts),
        trip_pool(case.k, case.key_valid_counts),
    )
    qualities = []
    all_errors = []
    for budget in budgets:
        mask = segmented_top_k(
            logits,
            case.key_segments,
            case.key_valid_counts,
            target_blocks=budget.target_blocks,
            reference_blocks=budget.reference_blocks,
            required_keys=case.required_keys,
        )
        sparse = reference_attention(
            case.q,
            case.k,
            case.v,
            mask,
            query_valid_counts=case.query_valid_counts,
            key_valid_counts=case.key_valid_counts,
        )
        errors = [
            relative_squared_error(
                case.dense[index, :count].reshape(-1),
                sparse[index, :count].reshape(-1),
                epsilon=case.epsilon,
            )
            for index, count in enumerate(case.query_valid_counts)
        ]
        all_errors.append(errors)
        qualities.append(1 / (1 + tail_error(errors, mean_weight=tail_mean_weight)))
    return qualities, all_errors


def _score_case_or_zero(
    selector: FixedSelector,
    case: ExactCase,
    budgets: Sequence[BudgetSpec],
    tail_mean_weight: float,
) -> tuple[list[float], list[list[float]]]:
    try:
        return _score_case(selector, case, budgets, tail_mean_weight)
    except ModelValidationError as exc:
        if exc.code != "MODEL_RUNTIME_INVALID":
            raise
        return _zero_case(budgets)


def _score_gpu_case_or_zero(
    scorer: object,
    selector: FixedSelector,
    budgets: Sequence[BudgetSpec],
    tail_mean_weight: float,
) -> tuple[list[float], list[list[float]]]:
    try:
        return scorer.score(selector, list(budgets), tail_mean_weight)
    except ModelValidationError as exc:
        if exc.code != "MODEL_RUNTIME_INVALID":
            raise
        return _zero_case(budgets)


def _zero_case(budgets: Sequence[BudgetSpec]) -> tuple[list[float], list[list[float]]]:
    return [0.0 for _ in budgets], [[] for _ in budgets]


def _weighted_cell_mean(
    values: Sequence[float],
    cases: Sequence[ExactCase | CaseDesign],
    stratum_weights: Mapping[str, float],
    step_window_weights: Mapping[str, float],
) -> float:
    _validate_complete_design(cases, stratum_weights, step_window_weights)
    total = 0.0
    for stratum, stratum_weight in stratum_weights.items():
        for window, step_weight in step_window_weights.items():
            selected = [
                values[index]
                for index, case in enumerate(cases)
                if case.stratum == stratum and case.step_window == window
            ]
            total += stratum_weight * step_weight * fsum(selected) / len(selected)
    return total


def _validate_complete_design(
    cases: Sequence[ExactCase | CaseDesign],
    stratum_weights: Mapping[str, float],
    step_window_weights: Mapping[str, float],
) -> None:
    if not cases or set(case.stratum for case in cases) != set(stratum_weights):
        raise ValueError("cases must cover every scoring stratum")
    if set(case.step_window for case in cases) != set(step_window_weights):
        raise ValueError("cases must cover every sampled step window")
    if abs(fsum(stratum_weights.values()) - 1.0) > 1e-12:
        raise ValueError("stratum weights must sum to one")
    if abs(fsum(step_window_weights.values()) - 1.0) > 1e-12:
        raise ValueError("step window weights must sum to one")
    routes = {case.route for case in cases}
    steps = {case.step_index for case in cases}
    for stratum in stratum_weights:
        rollouts = {case.rollout_id for case in cases if case.stratum == stratum}
        expected = {
            (route, rollout, step) for route in routes for rollout in rollouts for step in steps
        }
        actual = {
            (case.route, case.rollout_id, case.step_index)
            for case in cases
            if case.stratum == stratum
        }
        if actual != expected or len(actual) != sum(case.stratum == stratum for case in cases):
            raise ValueError("exact cases do not form a complete route/rollout/step design")


def _validate_independent_videos(
    cases: Sequence[ExactCase | CaseDesign],
    stratum_weights: Mapping[str, float],
) -> None:
    """Require enough one-source video clusters for a defensible paired bound."""

    for stratum in stratum_weights:
        source_by_rollout: dict[str, str] = {}
        rollout_by_source: dict[str, str] = {}
        for case in cases:
            if case.stratum != stratum:
                continue
            existing_source = source_by_rollout.setdefault(case.rollout_id, case.source_id)
            existing_rollout = rollout_by_source.setdefault(case.source_id, case.rollout_id)
            if existing_source != case.source_id or existing_rollout != case.rollout_id:
                raise ValueError("each rollout must represent one independent source video")
        if len(source_by_rollout) < 4:
            raise ValueError("paired scoring requires at least four source videos per stratum")


def _validate_case(case: ExactCase, selector: FixedSelector) -> None:
    if case.route not in selector.architecture.routes:
        raise ValueError("exact case references an unknown selector route")
    _validate_case_data(case)


def validate_dense_reference(
    case: ExactCase,
    *,
    device: str | None = None,
) -> tuple[float, float, float]:
    """Check stored dense semantics, not cross-device identity of FP16 encodings.

    The unchanged canonical path computes in FP32 then rounds to FP16. Nearby
    FP32 results on opposite sides of a rounding midpoint can therefore occupy
    adjacent FP16 bins. Allow the sum of their local half-bin widths, not a
    hardware-specific measured maximum, promotion threshold or global atol.
    This is a conservative storage-resolution policy, not a universal bound on
    FP32 attention error. Archive SHA/manifest verification remains bit-exact;
    this function never substitutes the recomputed tensor for the stored one.
    """

    _validate_case_data(case)
    expected = canonical_dense_reference(
        case.q,
        case.k,
        case.v,
        case.query_valid_counts,
        case.key_valid_counts,
        device=device,
    )
    observed_storage = np.asarray(case.dense)
    if observed_storage.dtype != np.float16:
        raise ValueError("exact dense reference must use canonical float16 storage")
    observed = observed_storage.astype(np.float32)
    expected_float = expected.astype(np.float32)
    if not np.isfinite(expected_float).all():
        raise ValueError("recomputed exact dense reference must be finite")
    valid = np.arange(observed.shape[1])[None, :] < np.asarray(
        case.query_valid_counts
    )[:, None]
    delta = np.abs(observed[valid] - expected_float[valid])
    maximum = float(delta.max())
    mean = float(delta.mean(dtype=np.float64))
    relative = relative_squared_error(
        observed[valid].reshape(-1),
        expected_float[valid].reshape(-1),
        epsilon=case.epsilon,
    )
    allowance = _float16_rounding_radius(observed) + _float16_rounding_radius(expected_float)
    if (np.any(np.abs(observed[valid] - expected_float[valid]) > allowance[valid])
            or np.any(observed[~valid] != 0)):
        raise ValueError(
            "exact dense reference diverges from its Q/K/V: "
            f"max_abs={maximum:.9g}, mean_abs={mean:.9g}, relative={relative:.9g}"
        )
    return maximum, mean, relative


def _float16_rounding_radius(values: np.ndarray) -> np.ndarray:
    """Half an FP16 bin: precision 11; subnormal spacing is exactly 2**-24.

    Clamp magnitudes to the smallest normal before frexp so zero/subnormals
    receive radius 2**-25. Unlike nextafter(max, inf), this stays finite at 65504.
    At exponent boundaries each operand contributes its own bin width.
    """
    storage = np.finfo(np.float16)
    _, exponent = np.frexp(np.maximum(np.abs(values), storage.tiny))
    return np.ldexp(np.ones_like(values), exponent - storage.nmant - 2)


def canonical_dense_reference(
    q_tiles: np.ndarray,
    k_tiles: np.ndarray,
    v_tiles: np.ndarray,
    query_valid_counts: np.ndarray,
    key_valid_counts: np.ndarray,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Compute the single float16 dense representation stored by exact-cache v2."""

    q = np.asarray(q_tiles)
    k = np.asarray(k_tiles)
    v = np.asarray(v_tiles)
    q_counts = np.asarray(query_valid_counts)
    k_counts = np.asarray(key_valid_counts)
    if any(value.dtype != np.float16 for value in (q, k, v)):
        raise ValueError("canonical dense input Q/K/V must use float16 storage")
    if device is None:
        dense = reference_attention(
            q,
            k,
            v,
            np.ones((q.shape[0], k.shape[0]), dtype=np.bool_),
            query_valid_counts=q_counts,
            key_valid_counts=k_counts,
        )
    else:
        from scoring.torch_backend import dense_reference_attention

        dense = dense_reference_attention(
            q,
            k,
            v,
            q_counts,
            k_counts,
            device=device,
        )
    return np.asarray(dense, dtype=np.float16)


def _validate_case_data(case: ExactCase) -> None:
    for value, name in (
        (case.stratum, "stratum"),
        (case.rollout_id, "rollout"),
        (case.source_id, "source"),
        (case.step_window, "step window"),
        (case.tile_plan_hash, "tile plan hash"),
    ):
        if not value:
            raise ValueError(f"exact case {name} cannot be empty")
    if case.step_index < 0 or not isfinite(case.scheduler_timestep) or not isfinite(case.sigma):
        raise ValueError("exact case step metadata is invalid")
    if not isfinite(case.epsilon) or case.epsilon <= 0:
        raise ValueError("exact case epsilon must be finite and positive")
    q, k, v, dense = map(np.asarray, (case.q, case.k, case.v, case.dense))
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or dense.shape != q.shape:
        raise ValueError("exact case Q/K/V/dense shapes are incompatible")
    if q.shape[0] != len(case.query_tile_ids) or len(set(case.query_tile_ids)) != len(
        case.query_tile_ids
    ):
        raise ValueError("exact case query tile IDs are incompatible")
    if case.required_keys.shape != (q.shape[0], k.shape[0]) or case.required_keys.dtype != np.bool_:
        raise ValueError("exact case required keys are incompatible")
    if case.query_valid_counts.shape != (q.shape[0],):
        raise ValueError("exact case query valid counts are incompatible")
    if case.key_valid_counts.shape != (k.shape[0],):
        raise ValueError("exact case key valid counts are incompatible")
    if case.key_segments.shape != (k.shape[0],):
        raise ValueError("exact case key segments are incompatible")
    if not all(np.isfinite(value).all() for value in (q, k, v, dense)):
        raise ValueError("exact case arrays must be finite")
