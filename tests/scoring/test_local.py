import numpy as np
import pytest

from scoring.evaluate import reference_attention
from scoring.local import ExactCase, ScreenCase, evaluate_selector_local, evaluate_selector_screen
from scoring.selector import FixedSelector, SelectorArchitecture
from scoring.spec import BudgetSpec


def tied_selector() -> FixedSelector:
    architecture = SelectorArchitecture(1, 1, 2, ("layer_0_head_0",), "float32")
    weights = {
        f"layer_0_head_0.proj_{side}": np.zeros((6, 2), dtype=np.float32)
        for side in ("q", "k")
    }
    weights["layer_0_head_0.proj_q"][:2] = -np.eye(2, dtype=np.float32)
    weights["layer_0_head_0.proj_k"][:2] = -np.eye(2, dtype=np.float32)
    return FixedSelector(architecture, weights)


def tied_selector_for(routes: tuple[str, ...]) -> FixedSelector:
    architecture = SelectorArchitecture(1, len(routes), 2, routes, "float32")
    weights = {
        f"{route}.proj_{side}": np.zeros((6, 2), dtype=np.float32)
        for route in routes
        for side in ("q", "k")
    }
    for projection in weights.values():
        projection[:2] = -np.eye(2, dtype=np.float32)
    return FixedSelector(architecture, weights)


def exact_case(seed: int, *, stratum: str = "all") -> ExactCase:
    generator = np.random.default_rng(seed)
    q = generator.normal(size=(2, 2, 2)).astype(np.float32)
    k = generator.normal(size=(3, 2, 2)).astype(np.float32)
    v = generator.normal(size=(3, 2, 2)).astype(np.float32)
    dense = reference_attention(q, k, v, np.ones((2, 3), dtype=np.bool_))
    return ExactCase(
        route="layer_0_head_0",
        stratum=stratum,
        q=q,
        k=k,
        v=v,
        dense=dense,
        valid_keys=np.ones(3, dtype=np.bool_),
        epsilon=1e-6,
    )


def test_exact_local_scoring_runs_fixed_selector_at_every_budget() -> None:
    budgets = (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0))

    first = evaluate_selector_local(
        tied_selector(),
        [exact_case(1), exact_case(2)],
        budgets,
        stratum_weights={"all": 1.0},
        tail_mean_weight=0.8,
        bootstrap_confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    second = evaluate_selector_local(
        tied_selector(),
        [exact_case(1), exact_case(2)],
        budgets,
        stratum_weights={"all": 1.0},
        tail_mean_weight=0.8,
        bootstrap_confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )

    assert first == second
    assert 0 <= first.lower_bound <= first.aggregate.score <= 1
    assert first.aggregate.budgets[-1].quality == pytest.approx(1.0)
    assert len(first.sample_scores) == 2


def test_exact_local_scoring_rejects_unknown_stratum_and_oversized_budget() -> None:
    with pytest.raises(ValueError, match="stratum"):
        evaluate_selector_local(
            tied_selector(),
            [exact_case(1, stratum="missing")],
            (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0)),
            stratum_weights={"all": 1.0},
            tail_mean_weight=0.8,
            bootstrap_confidence=0.95,
            bootstrap_resamples=10,
            bootstrap_seed=7,
        )

    case = exact_case(1)
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_selector_local(
            tied_selector(),
            [case],
            (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(4, 4.0)),
            stratum_weights={"all": 1.0},
            tail_mean_weight=0.8,
            bootstrap_confidence=0.95,
            bootstrap_resamples=10,
            bootstrap_seed=7,
        )


def test_lightweight_screening_scores_retained_teacher_distribution() -> None:
    case = exact_case(1)
    q_desc = np.concatenate((case.q.mean(1), case.q.max(1), case.q.min(1)), axis=-1).astype(
        np.float32
    )
    k_desc = np.concatenate((case.k.mean(1), case.k.max(1), case.k.min(1)), axis=-1).astype(
        np.float32
    )
    mass = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)

    result = evaluate_selector_screen(
        tied_selector(),
        [ScreenCase("layer_0_head_0", "all", q_desc, k_desc, mass, np.ones(3, dtype=np.bool_))],
        (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0)),
        stratum_weights={"all": 1.0},
        bootstrap_confidence=0.95,
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )

    assert result.budget_scores == pytest.approx((0.4, 0.7, 1.0))
    assert result.score == pytest.approx(result.upper_bound)


def test_screening_macro_averages_routes_instead_of_cache_frequency() -> None:
    descriptors = np.ones((1, 6), dtype=np.float32)
    valid = np.ones(3, dtype=np.bool_)
    cases = [
        ScreenCase(
            "layer_0_head_0",
            "all",
            descriptors,
            np.ones((3, 6), dtype=np.float32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            valid,
        )
    ]
    cases.extend(
        ScreenCase(
            "layer_0_head_1",
            "all",
            descriptors,
            np.ones((3, 6), dtype=np.float32),
            np.array([[0.0, 0.5, 0.5]], dtype=np.float32),
            valid,
        )
        for _ in range(3)
    )

    result = evaluate_selector_screen(
        tied_selector_for(("layer_0_head_0", "layer_0_head_1")),
        cases,
        (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0)),
        stratum_weights={"all": 1.0},
        bootstrap_confidence=0.95,
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )

    assert result.budget_scores == pytest.approx((0.5, 0.75, 1.0))
