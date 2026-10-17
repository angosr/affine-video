import numpy as np
import pytest

from scoring.evaluate import (
    percentile,
    reference_attention,
    segmented_top_k,
    tail_error,
    trip_pool,
)


def test_trip_pool_uses_mean_max_and_min_in_fixed_order() -> None:
    tiles = np.array([[[1.0, 4.0], [3.0, 2.0]]], dtype=np.float32)

    descriptor = trip_pool(tiles)

    assert descriptor.tolist() == [[2.0, 3.0, 3.0, 4.0, 1.0, 2.0]]


def test_segmented_top_k_does_not_overfill_a_budget_already_met_by_required_keys() -> None:
    mask = segmented_top_k(
        np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.int8),
        np.ones(3, dtype=np.int16),
        target_blocks=1,
        reference_blocks=0,
        required_keys=np.array([[False, True, False]], dtype=np.bool_),
    )

    assert mask.tolist() == [[False, True, False]]


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
