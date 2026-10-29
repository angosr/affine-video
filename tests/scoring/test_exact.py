from dataclasses import replace

import numpy as np
import pytest

from scoring.evaluate import reference_attention
from scoring.exact import (
    ExactCase,
    canonical_dense_reference,
    paired_score_gain_bound,
    score_selector_pair,
    validate_dense_reference,
)
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


def test_dense_reference_retains_float16_storage_and_rejects_wrong_values() -> None:
    case = exact_case(1)
    q = case.q.astype(np.float16)
    k = case.k.astype(np.float16)
    v = case.v.astype(np.float16)
    dense = canonical_dense_reference(
        q,
        k,
        v,
        case.query_valid_counts,
        case.key_valid_counts,
    )
    canonical = replace(case, q=q, k=k, v=v, dense=dense)

    assert dense.dtype == np.float16
    assert validate_dense_reference(canonical) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="diverges"):
        validate_dense_reference(replace(canonical, dense=dense + np.float16(100)))


def test_dense_reference_accepts_signed_zero_without_rewriting_stored_bytes() -> None:
    case = exact_case(1)
    q, k, v = (np.zeros_like(value, dtype=np.float16) for value in (case.q, case.k, case.v))
    dense = canonical_dense_reference(
        q, k, v, case.query_valid_counts, case.key_valid_counts
    )
    canonical = replace(case, q=q, k=k, v=v, dense=dense)
    assert validate_dense_reference(canonical) == (0.0, 0.0, 0.0)

    mutated = dense.copy()
    mutated.view(np.uint16).flat[0] ^= np.uint16(1 << 15)
    before = mutated.tobytes()
    assert np.array_equal(mutated, dense)
    assert validate_dense_reference(replace(canonical, dense=mutated)) == (0.0, 0.0, 0.0)
    assert mutated.tobytes() == before and before != dense.tobytes()


def constant_dense_case(value):
    case = exact_case(1)
    q, k = (np.zeros_like(item, dtype=np.float16) for item in (case.q, case.k))
    v = np.full_like(case.v, value, dtype=np.float16)
    dense = canonical_dense_reference(q, k, v, case.query_valid_counts, case.key_valid_counts)
    assert np.all(dense == np.float16(value))
    return replace(case, q=q, k=k, v=v, dense=dense)


@pytest.mark.parametrize("value", [0, 2**-24, 2**-14, 0.125, 1, -1, 4, -4, 32, 65504, -65504])
def test_dense_reference_uses_local_float16_quantization_not_observed_max_error(value):
    case = constant_dense_case(value)
    direction = np.float16(0 if abs(value) == 65504 else (np.inf if value >= 0 else -np.inf))
    adjacent = case.dense.copy()
    adjacent.flat[0] = np.nextafter(adjacent.flat[0], direction)
    before = adjacent.tobytes()
    assert validate_dense_reference(replace(case, dense=adjacent))[0] > 0
    assert adjacent.tobytes() == before

    two_steps = adjacent.copy()
    two_steps.flat[0] = np.nextafter(two_steps.flat[0], direction)
    with pytest.raises(ValueError, match="diverges"):
        validate_dense_reference(replace(case, dense=two_steps))


def test_fp32_roundoff_across_a_float16_midpoint_is_not_data_corruption(monkeypatch):
    import scoring.exact as module

    # Nearby FP32 results can round to distinct FP16 values even with the same precision.
    lower = np.float16(1)
    upper = np.nextafter(lower, np.float16(np.inf))
    midpoint = (np.float32(lower) + np.float32(upper)) / np.float32(2)
    left = np.nextafter(midpoint, np.float32(-np.inf))
    right = np.nextafter(midpoint, np.float32(np.inf))
    assert np.float16(left) == lower and np.float16(right) == upper
    assert right - left < np.float32(upper) - np.float32(lower)
    case = constant_dense_case(lower)
    recomputed = np.full_like(case.dense, np.float16(right))
    monkeypatch.setattr(module, "canonical_dense_reference", lambda *_a, **_k: recomputed)
    before = case.dense.tobytes()
    maximum, _mean, _relative = validate_dense_reference(case, device="cuda:0")
    assert maximum == np.float32(upper) - np.float32(lower)
    assert case.dense.tobytes() == before


@pytest.mark.parametrize("value", [1, -1, 2, -2, 4, -4])
def test_dense_reference_does_not_admit_two_smaller_bins_at_exponent_boundary(value):
    case = constant_dense_case(value)
    adjacent = case.dense.copy()
    adjacent.flat[0] = np.nextafter(adjacent.flat[0], np.float16(0))
    assert validate_dense_reference(replace(case, dense=adjacent))[0] > 0
    adjacent.flat[0] = np.nextafter(adjacent.flat[0], np.float16(0))
    with pytest.raises(ValueError, match="diverges"):
        validate_dense_reference(replace(case, dense=adjacent))


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_dense_reference_rejects_nonfinite_recomputation(monkeypatch, value):
    import scoring.exact as module

    case = constant_dense_case(1)
    monkeypatch.setattr(module, "canonical_dense_reference",
                        lambda *_a, **_k: np.full_like(case.dense, value))
    with pytest.raises(ValueError, match="recomputed.*finite"):
        validate_dense_reference(case)


@pytest.mark.parametrize("fault", ["dtype", "q-dtype", "shape", "nan", "inf", "padding",
                                   "v-semantic"])
def test_semantic_dense_guard_keeps_dtype_shape_finite_and_padding_checks(fault):
    case = constant_dense_case(1)
    if fault == "dtype":
        case = replace(case, dense=case.dense.astype(np.float32))
    elif fault == "q-dtype":
        case = replace(case, q=case.q.astype(np.float32))
    elif fault == "shape":
        case = replace(case, dense=case.dense[:1])
    elif fault in {"nan", "inf"}:
        case.dense.flat[0] = np.float16(np.nan if fault == "nan" else np.inf)
    elif fault == "v-semantic":
        case = replace(case, v=np.zeros_like(case.v))
    else:
        counts = np.array([1, 2], dtype=np.int16)
        dense = canonical_dense_reference(case.q, case.k, case.v, counts, case.key_valid_counts)
        dense[0, 1, 0] = np.float16(2**-24)  # Padding has no legitimate rounding error.
        case = replace(case, query_valid_counts=counts, dense=dense)
    with pytest.raises(ValueError):
        validate_dense_reference(case)


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
