import numpy as np
import pytest

from scoring.evaluate import (
    BudgetEvidence,
    bootstrap_mean_lower_bound,
    bootstrap_mean_upper_bound,
    challenge_seed,
    evaluate_local,
    percentile,
    reference_attention,
    stable_top_k,
    stratified_bootstrap_bound,
    tail_error,
    trip_pool,
)


def test_challenge_seed_binds_all_three_immutable_inputs() -> None:
    track = "11" * 32
    submission = "22" * 32
    block = "0x" + "33" * 32

    first = challenge_seed(track, submission, block)

    assert first == challenge_seed(track.upper(), submission, block.upper().replace("0X", "0x"))
    assert first != challenge_seed(track, submission, "0x" + "44" * 32)
    with pytest.raises(ValueError, match="32 bytes"):
        challenge_seed("11", submission, block)


def test_trip_pool_uses_mean_max_and_min_in_fixed_order() -> None:
    tiles = np.array([[[1.0, 4.0], [3.0, 2.0]]], dtype=np.float32)

    descriptor = trip_pool(tiles)

    assert descriptor.tolist() == [[2.0, 3.0, 3.0, 4.0, 1.0, 2.0]]


def test_stable_top_k_has_exact_budget_and_index_tie_break() -> None:
    logits = np.array([[1.0, 2.0, 2.0, 9.0], [3.0, 3.0, 1.0, 8.0]], dtype=np.float32)
    valid = np.array([True, True, True, False])

    mask = stable_top_k(logits, valid, blocks=2)

    assert mask.tolist() == [
        [False, True, True, False],
        [True, True, False, False],
    ]
    assert np.all(mask.sum(axis=1) == 2)


def test_reference_attention_full_mask_matches_flat_dense_attention() -> None:
    q = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    k = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    v = np.array([[[2.0, 0.0]], [[0.0, 4.0]]], dtype=np.float32)

    result = reference_attention(q, k, v, np.ones((1, 2), dtype=np.bool_))
    logits = q[0] @ k.reshape(-1, 2).T / np.sqrt(2)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = probabilities @ v.reshape(-1, 2)

    assert result[0] == pytest.approx(expected)


def test_reference_primitives_reject_invalid_budget_and_empty_rows() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        stable_top_k(np.ones((1, 2)), np.array([True, False]), blocks=2)
    with pytest.raises(ValueError, match="at least one"):
        reference_attention(
            np.ones((1, 1, 2)),
            np.ones((1, 1, 2)),
            np.ones((1, 1, 2)),
            np.zeros((1, 1), dtype=np.bool_),
        )


def test_linear_percentile_and_tail_mixture_are_explicit() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 100.0]

    assert percentile(values, 0.95) == pytest.approx(80.6)
    assert tail_error(values, mean_weight=0.8) == pytest.approx(33.08)


def test_local_score_macro_averages_strata_then_integrates_cost() -> None:
    result = evaluate_local(
        [
            BudgetEvidence(1, 1.0, {"a": [1.0, 1.0], "b": [3.0, 3.0]}),
            BudgetEvidence(2, 2.0, {"a": [0.5, 0.5], "b": [1.5, 1.5]}),
            BudgetEvidence(4, 4.0, {"a": [0.0, 0.0], "b": [1.0, 1.0]}),
        ],
        stratum_weights={"a": 0.75, "b": 0.25},
        tail_mean_weight=0.8,
    )

    assert [item.aggregate_error for item in result.budgets] == pytest.approx([1.5, 0.75, 0.25])
    assert [item.quality for item in result.budgets] == pytest.approx([0.4, 1 / 1.75, 0.8])
    assert result.score == pytest.approx((0.4 + 2 / 1.75 + 0.8) / 4)


def test_local_score_rejects_missing_stratum_and_two_budget_shortcut() -> None:
    with pytest.raises(ValueError, match="at least three"):
        evaluate_local(
            [
                BudgetEvidence(1, 1.0, {"a": [1.0]}),
                BudgetEvidence(2, 2.0, {"a": [0.5]}),
            ],
            stratum_weights={"a": 1.0},
            tail_mean_weight=0.8,
        )

    with pytest.raises(ValueError, match="exact frozen strata"):
        evaluate_local(
            [
                BudgetEvidence(1, 1.0, {"a": [1.0]}),
                BudgetEvidence(2, 2.0, {"a": [0.5], "b": [0.5]}),
                BudgetEvidence(3, 3.0, {"a": [0.1]}),
            ],
            stratum_weights={"a": 1.0},
            tail_mean_weight=0.8,
        )


def test_bootstrap_lower_bound_is_reproducible_and_below_mean() -> None:
    values = [0.55, 0.65, 0.75, 0.85]

    first = bootstrap_mean_lower_bound(
        values,
        confidence=0.95,
        resamples=1000,
        seed=123,
    )
    second = bootstrap_mean_lower_bound(
        values,
        confidence=0.95,
        resamples=1000,
        seed=123,
    )

    assert first == second
    assert first == pytest.approx(0.6)
    assert first < sum(values) / len(values)


def test_bootstrap_upper_bound_is_reproducible_and_above_mean() -> None:
    values = [0.55, 0.65, 0.75, 0.85]

    first = bootstrap_mean_upper_bound(
        values,
        confidence=0.95,
        resamples=1000,
        seed=123,
    )
    second = bootstrap_mean_upper_bound(
        values,
        confidence=0.95,
        resamples=1000,
        seed=123,
    )

    assert first == second
    assert first > sum(values) / len(values)


def test_stratified_bootstrap_keeps_frozen_macro_weights() -> None:
    values = {"large": [0.9] * 20, "small": [0.1, 0.1]}
    weights = {"large": 0.5, "small": 0.5}

    lower = stratified_bootstrap_bound(
        values,
        stratum_weights=weights,
        confidence=0.95,
        resamples=100,
        seed=7,
        upper=False,
    )
    upper = stratified_bootstrap_bound(
        values,
        stratum_weights=weights,
        confidence=0.95,
        resamples=100,
        seed=7,
        upper=True,
    )

    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.5)
