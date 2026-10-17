from dataclasses import replace

import numpy as np
import pytest

from scoring.evaluate import reference_attention
from scoring.exact import ExactCase, paired_score_gain_bound


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
