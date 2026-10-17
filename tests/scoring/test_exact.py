from dataclasses import replace

import numpy as np
import pytest

from scoring.evaluate import reference_attention
from scoring.exact import ExactCase, paired_score_gain_bound, score_selector_pair
from scoring.selector import FixedSelector, SelectorArchitecture
from scoring.spec import BudgetSpec


def exact_case(seed: int, *, stratum: str = "all") -> ExactCase:
    generator = np.random.default_rng(seed)
    q = generator.normal(size=(2, 2, 2)).astype(np.float32)
    k = generator.normal(size=(3, 2, 2)).astype(np.float32)
    v = generator.normal(size=(3, 2, 2)).astype(np.float32)
    dense = reference_attention(q, k, v, np.ones((2, 3), dtype=np.bool_))
    return ExactCase(
        route="layer_0_head_0",
        stratum=stratum,
        rollout_id=f"rollout-{seed}",
        source_id=f"source-{seed}",
        step_index=0,
        step_window="all",
        scheduler_timestep=1.0,
        sigma=1.0,
        tile_plan_hash="ab" * 32,
        query_tile_ids=(0, 1),
        q=q,
        k=k,
        v=v,
        dense=dense,
        query_valid_counts=np.full(2, 2, dtype=np.int16),
        key_valid_counts=np.full(3, 2, dtype=np.int16),
        key_segments=np.zeros(3, dtype=np.int8),
        required_keys=np.array([[True, False, False], [False, True, False]], dtype=np.bool_),
        epsilon=1e-6,
    )


def test_paired_gain_bound_supports_losses_and_is_deterministic() -> None:
    cases = [exact_case(seed) for seed in range(4)]
    arguments = {
        "challenger_scores": [0.4, 0.5, 0.6, 0.7],
        "incumbent_scores": [0.5, 0.6, 0.7, 0.8],
        "cases": cases,
        "stratum_weights": {"all": 1.0},
        "step_window_weights": {"all": 1.0},
        "confidence": 0.95,
        "resamples": 1000,
        "seed": 7,
    }

    first = paired_score_gain_bound(**arguments)

    assert first == paired_score_gain_bound(**arguments)
    assert first == pytest.approx(-0.1)


def test_paired_bound_rejects_too_few_independent_source_videos() -> None:
    cases = [exact_case(seed) for seed in range(3)]

    with pytest.raises(ValueError, match="four source videos"):
        paired_score_gain_bound(
            [0.6, 0.6, 0.6],
            [0.5, 0.5, 0.5],
            cases,
            stratum_weights={"all": 1.0},
            step_window_weights={"all": 1.0},
            confidence=0.95,
            resamples=100,
            seed=7,
        )


def test_paired_bound_rejects_multiple_rollouts_from_one_source_video() -> None:
    cases = [exact_case(seed) for seed in range(4)]
    cases[1] = replace(cases[1], source_id=cases[0].source_id)

    with pytest.raises(ValueError, match="independent source video"):
        paired_score_gain_bound(
            [0.6] * 4,
            [0.5] * 4,
            cases,
            stratum_weights={"all": 1.0},
            step_window_weights={"all": 1.0},
            confidence=0.95,
            resamples=100,
            seed=7,
        )


def test_runtime_invalid_selector_scores_zero_without_poisoning_its_opponent() -> None:
    architecture = SelectorArchitecture(1, 1, 2, "float32")
    shape = (architecture.descriptor_dim, architecture.head_dim)
    good = FixedSelector(
        architecture,
        {
            "layer_0_head_0.proj_q": np.zeros(shape, dtype=np.float32),
            "layer_0_head_0.proj_k": np.zeros(shape, dtype=np.float32),
        },
    )
    bad = FixedSelector(
        architecture,
        {
            "layer_0_head_0.proj_q": np.full(
                shape, np.finfo(np.float32).max, dtype=np.float32
            ),
            "layer_0_head_0.proj_k": np.full(
                shape, np.finfo(np.float32).max, dtype=np.float32
            ),
        },
    )
    cases = []
    for seed in range(4):
        case = exact_case(seed)
        q = np.ones_like(case.q)
        k = np.ones_like(case.k)
        cases.append(
            replace(
                case,
                q=q,
                k=k,
                dense=reference_attention(
                    q,
                    k,
                    case.v,
                    np.ones((q.shape[0], k.shape[0]), dtype=np.bool_),
                ),
            )
        )
    arguments = {
        "cases": cases,
        "budgets": (BudgetSpec(1, 0, 1.0), BudgetSpec(2, 0, 2.0)),
        "stratum_weights": {"all": 1.0},
        "step_window_weights": {"all": 1.0},
        "tail_mean_weight": 0.5,
        "confidence": 0.95,
        "resamples": 100,
        "seed": 7,
    }

    invalid_challenger = score_selector_pair(bad, good, **arguments)
    invalid_incumbent = score_selector_pair(good, bad, **arguments)

    assert invalid_challenger.challenger_score == 0.0
    assert invalid_challenger.incumbent_score > 0.0
    assert invalid_challenger.gain_lower_bound < 0.0
    assert invalid_incumbent.incumbent_score == 0.0
    assert invalid_incumbent.challenger_score > 0.0
    assert invalid_incumbent.gain_lower_bound > 0.0
