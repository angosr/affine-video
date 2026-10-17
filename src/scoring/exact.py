"""Exact cached-QKV evaluation for one fixed selector."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from scoring.evaluate import (
    BudgetEvidence,
    ExactScore,
    aggregate_exact_score,
    reference_attention,
    stable_top_k,
    stratified_bootstrap_bound,
    tail_error,
    trip_pool,
)
from scoring.metrics import log_cost_auc, relative_squared_error
from scoring.selector import FixedSelector
from scoring.spec import BudgetSpec


@dataclass(frozen=True, slots=True)
class ExactCase:
    route: str
    stratum: str
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    dense: np.ndarray
    valid_keys: np.ndarray
    required_keys: np.ndarray
    epsilon: float


@dataclass(frozen=True, slots=True)
class SelectorScore:
    aggregate: ExactScore
    lower_bound: float
    sample_scores: tuple[float, ...]


def score_selector(
    selector: FixedSelector,
    cases: Sequence[ExactCase],
    budgets: Sequence[BudgetSpec],
    *,
    stratum_weights: Mapping[str, float],
    tail_mean_weight: float,
    bootstrap_confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> SelectorScore:
    """Compute one deterministic authoritative score from the exact cache."""

    if not cases:
        raise ValueError("exact evaluation requires at least one case")
    if len(budgets) < 3:
        raise ValueError("exact evaluation requires at least three budgets")

    sampled_routes = tuple(sorted({case.route for case in cases}))
    group_weights, group_for = _route_stratum_groups(sampled_routes, stratum_weights)
    errors: list[dict[str, list[float]]] = [
        {group: [] for group in group_weights} for _ in budgets
    ]
    per_case_qualities: list[list[float]] = [[] for _ in cases]
    for case_index, case in enumerate(cases):
        _validate_case(case, selector)
        if case.stratum not in stratum_weights:
            raise ValueError("exact case references an unknown stratum")
        logits = selector.logits(case.route, trip_pool(case.q), trip_pool(case.k))
        for budget_index, budget in enumerate(budgets):
            mask = stable_top_k(
                logits,
                case.valid_keys,
                required_keys=case.required_keys,
                blocks=budget.blocks,
            )
            sparse = reference_attention(case.q, case.k, case.v, mask)
            row_errors = [
                relative_squared_error(
                    dense.reshape(-1),
                    candidate.reshape(-1),
                    epsilon=case.epsilon,
                )
                for dense, candidate in zip(case.dense, sparse, strict=True)
            ]
            errors[budget_index][group_for(case.route, case.stratum)].extend(row_errors)
            aggregate_error = tail_error(row_errors, mean_weight=tail_mean_weight)
            per_case_qualities[case_index].append(1 / (1 + aggregate_error))

    evidence = tuple(
        BudgetEvidence(
            budget=budget.blocks,
            calibrated_cost=budget.calibrated_cost,
            errors_by_group=errors[index],
        )
        for index, budget in enumerate(budgets)
    )
    aggregate = aggregate_exact_score(
        evidence,
        group_weights=group_weights,
        tail_mean_weight=tail_mean_weight,
    )
    costs = [budget.calibrated_cost for budget in budgets]
    sample_scores = tuple(log_cost_auc(costs, qualities) for qualities in per_case_qualities)
    lower_bound = stratified_bootstrap_bound(
        {
            stratum: [
                sample_scores[index]
                for index, case in enumerate(cases)
                if case.stratum == stratum
            ]
            for stratum in stratum_weights
        },
        group_weights=stratum_weights,
        confidence=bootstrap_confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        upper=False,
    )
    return SelectorScore(aggregate, lower_bound, sample_scores)


def paired_score_gain_bound(
    challenger_scores: Sequence[float],
    incumbent_scores: Sequence[float],
    cases: Sequence[ExactCase],
    *,
    stratum_weights: Mapping[str, float],
    confidence: float,
    resamples: int,
    seed: int,
) -> float:
    """Return the lower confidence bound of paired per-case score gains."""

    if len(challenger_scores) != len(cases) or len(incumbent_scores) != len(cases):
        raise ValueError("paired scores must align with every exact case")
    differences = tuple(
        candidate - current
        for candidate, current in zip(
            challenger_scores,
            incumbent_scores,
            strict=True,
        )
    )
    return stratified_bootstrap_bound(
        {
            stratum: [
                differences[index]
                for index, case in enumerate(cases)
                if case.stratum == stratum
            ]
            for stratum in stratum_weights
        },
        group_weights=stratum_weights,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        upper=False,
    )


def _route_stratum_groups(
    routes: Sequence[str],
    stratum_weights: Mapping[str, float],
) -> tuple[dict[str, float], Callable[[str, str], str]]:
    route_positions = {route: index for index, route in enumerate(routes)}
    stratum_positions = {
        stratum: index for index, stratum in enumerate(stratum_weights)
    }
    weights = {
        f"{route_index}:{stratum_positions[stratum]}": weight / len(routes)
        for route_index in range(len(routes))
        for stratum, weight in stratum_weights.items()
    }

    def group_for(route: str, stratum: str) -> str:
        return f"{route_positions[route]}:{stratum_positions[stratum]}"

    return weights, group_for


def _validate_case(case: ExactCase, selector: FixedSelector) -> None:
    if case.route not in selector.architecture.routes:
        raise ValueError("exact case references an unknown selector route")
    if case.stratum == "":
        raise ValueError("exact case stratum cannot be empty")
    if not isfinite(case.epsilon) or case.epsilon <= 0:
        raise ValueError("exact case epsilon must be finite and positive")
    q = np.asarray(case.q)
    k = np.asarray(case.k)
    v = np.asarray(case.v)
    dense = np.asarray(case.dense)
    valid = np.asarray(case.valid_keys)
    required = np.asarray(case.required_keys)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or dense.shape != q.shape:
        raise ValueError("exact case Q/K/V/dense shapes are incompatible")
    if valid.shape != (k.shape[0],) or valid.dtype != np.bool_:
        raise ValueError("exact case valid_keys are incompatible")
    if required.shape != (q.shape[0], k.shape[0]) or required.dtype != np.bool_:
        raise ValueError("exact case required_keys are incompatible")
    if np.any(required & ~valid[None, :]):
        raise ValueError("exact case requires an invalid key")
