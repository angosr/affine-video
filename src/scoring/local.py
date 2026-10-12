"""Exact cached-QKV evaluation for one fixed selector."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from scoring.evaluate import (
    BudgetEvidence,
    LocalScore,
    evaluate_local,
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
    epsilon: float


@dataclass(frozen=True, slots=True)
class SelectorLocalScore:
    aggregate: LocalScore
    lower_bound: float
    sample_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScreenCase:
    route: str
    stratum: str
    q_descriptors: np.ndarray
    k_descriptors: np.ndarray
    teacher_distribution: np.ndarray
    valid_keys: np.ndarray


@dataclass(frozen=True, slots=True)
class SelectorScreenScore:
    score: float
    upper_bound: float
    budget_scores: tuple[float, ...]
    sample_scores: tuple[float, ...]


def evaluate_selector_screen(
    selector: FixedSelector,
    cases: Sequence[ScreenCase],
    budgets: Sequence[BudgetSpec],
    *,
    stratum_weights: Mapping[str, float],
    bootstrap_confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> SelectorScreenScore:
    """Evaluate retained max-pooled teacher probability without token attention."""

    if not cases:
        raise ValueError("screening requires at least one lightweight case")
    if len(budgets) < 3:
        raise ValueError("screening requires at least three budgets")
    group_weights, group_for = _route_stratum_groups(
        selector.architecture.routes,
        stratum_weights,
    )
    per_case: list[list[float]] = []
    for case in cases:
        _validate_screen_case(case, selector)
        if case.stratum not in stratum_weights:
            raise ValueError("screen case references an unknown stratum")
        logits = selector.logits(case.route, case.q_descriptors, case.k_descriptors)
        qualities: list[float] = []
        for budget in budgets:
            mask = stable_top_k(logits, case.valid_keys, blocks=budget.blocks)
            retained = np.where(mask, case.teacher_distribution, 0).sum(axis=1)
            qualities.append(float(retained.mean()))
        per_case.append(qualities)

    budget_scores = []
    for budget_index in range(len(budgets)):
        score = 0.0
        for group, weight in group_weights.items():
            values = [
                per_case[index][budget_index]
                for index, case in enumerate(cases)
                if group_for(case.route, case.stratum) == group
            ]
            if not values:
                raise ValueError("screen cache does not cover every route and stratum")
            score += weight * float(np.mean(values, dtype=np.float64))
        budget_scores.append(score)
    budget_scores_tuple = tuple(budget_scores)
    costs = [budget.calibrated_cost for budget in budgets]
    score = log_cost_auc(costs, budget_scores_tuple)
    sample_scores = tuple(log_cost_auc(costs, qualities) for qualities in per_case)
    upper_bound = stratified_bootstrap_bound(
        {
            group: [
                sample_scores[index]
                for index, case in enumerate(cases)
                if group_for(case.route, case.stratum) == group
            ]
            for group in group_weights
        },
        group_weights=group_weights,
        confidence=bootstrap_confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        upper=True,
    )
    return SelectorScreenScore(score, upper_bound, budget_scores_tuple, sample_scores)


def evaluate_selector_local(
    selector: FixedSelector,
    cases: Sequence[ExactCase],
    budgets: Sequence[BudgetSpec],
    *,
    stratum_weights: Mapping[str, float],
    tail_mean_weight: float,
    bootstrap_confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> SelectorLocalScore:
    """Run deterministic multi-budget sparse attention over an exact cache."""

    if not cases:
        raise ValueError("local evaluation requires at least one exact case")
    if len(budgets) < 3:
        raise ValueError("local evaluation requires at least three budgets")

    group_weights, group_for = _route_stratum_groups(
        selector.architecture.routes,
        stratum_weights,
    )
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
            mask = stable_top_k(logits, case.valid_keys, blocks=budget.blocks)
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
    aggregate = evaluate_local(
        evidence,
        group_weights=group_weights,
        tail_mean_weight=tail_mean_weight,
    )
    costs = [budget.calibrated_cost for budget in budgets]
    sample_scores = tuple(log_cost_auc(costs, qualities) for qualities in per_case_qualities)
    lower_bound = stratified_bootstrap_bound(
        {
            group: [
                sample_scores[index]
                for index, case in enumerate(cases)
                if group_for(case.route, case.stratum) == group
            ]
            for group in group_weights
        },
        group_weights=group_weights,
        confidence=bootstrap_confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        upper=False,
    )
    return SelectorLocalScore(aggregate, lower_bound, sample_scores)


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
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or dense.shape != q.shape:
        raise ValueError("exact case Q/K/V/dense shapes are incompatible")
    if valid.shape != (k.shape[0],) or valid.dtype != np.bool_:
        raise ValueError("exact case valid_keys are incompatible")


def _validate_screen_case(case: ScreenCase, selector: FixedSelector) -> None:
    if case.route not in selector.architecture.routes:
        raise ValueError("screen case references an unknown selector route")
    if case.stratum == "":
        raise ValueError("screen case stratum cannot be empty")
    q = np.asarray(case.q_descriptors)
    k = np.asarray(case.k_descriptors)
    distribution = np.asarray(case.teacher_distribution)
    valid = np.asarray(case.valid_keys)
    expected_dim = selector.architecture.descriptor_dim
    if (
        q.ndim != 2
        or k.ndim != 2
        or q.shape[0] == 0
        or k.shape[0] == 0
        or q.shape[1] != expected_dim
        or k.shape[1] != expected_dim
    ):
        raise ValueError("screen case descriptor shapes are incompatible")
    if distribution.shape != (q.shape[0], k.shape[0]):
        raise ValueError("screen case teacher distribution is incompatible")
    if valid.shape != (k.shape[0],) or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("screen case valid_keys are incompatible")
    if not np.isfinite(distribution).all() or (distribution < 0).any():
        raise ValueError("screen case teacher distribution must be finite and non-negative")
    if np.any(distribution[:, ~valid] != 0) or not np.allclose(
        distribution.sum(axis=1), 1.0, rtol=0, atol=1e-6
    ):
        raise ValueError("screen case teacher distribution must normalize over valid keys")
