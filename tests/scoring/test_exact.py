import numpy as np
import pytest

from scoring.evaluate import reference_attention
from scoring.exact import ExactCase, paired_score_gain_bound, score_selector
from scoring.selector import FixedSelector, SelectorArchitecture
from scoring.spec import BudgetSpec


def tied_selector() -> FixedSelector:
    architecture = SelectorArchitecture(1, 1, 2, "float32")
    weights = {
        f"layer_0_head_0.proj_{side}": np.zeros((6, 2), dtype=np.float32) for side in ("q", "k")
    }
    weights["layer_0_head_0.proj_q"][:2] = -np.eye(2, dtype=np.float32)
    weights["layer_0_head_0.proj_k"][:2] = -np.eye(2, dtype=np.float32)
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
        required_keys=np.array([[True, False, False], [False, True, False]], dtype=np.bool_),
        epsilon=1e-6,
        sample_id=f"sample-{seed}",
    )


def test_exact_scoring_runs_fixed_selector_at_every_budget() -> None:
    budgets = (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0))

    first = score_selector(
        tied_selector(),
        [exact_case(1), exact_case(2)],
        budgets,
        stratum_weights={"all": 1.0},
        tail_mean_weight=0.8,
        bootstrap_confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    second = score_selector(
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


def test_exact_scoring_rejects_unknown_stratum_and_oversized_budget() -> None:
    with pytest.raises(ValueError, match="stratum"):
        score_selector(
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
        score_selector(
            tied_selector(),
            [case],
            (BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(4, 4.0)),
            stratum_weights={"all": 1.0},
            tail_mean_weight=0.8,
            bootstrap_confidence=0.95,
            bootstrap_resamples=10,
            bootstrap_seed=7,
        )


def test_paired_gain_bound_supports_losses_and_is_deterministic() -> None:
    cases = [exact_case(seed) for seed in range(4)]
    arguments = {
        "challenger_scores": [0.4, 0.5, 0.6, 0.7],
        "incumbent_scores": [0.5, 0.6, 0.7, 0.8],
        "cases": cases,
        "stratum_weights": {"all": 1.0},
        "confidence": 0.95,
        "resamples": 1000,
        "seed": 7,
    }

    first = paired_score_gain_bound(**arguments)

    assert first == paired_score_gain_bound(**arguments)
    assert first == pytest.approx(-0.1)
