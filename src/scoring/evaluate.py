"""Deterministic aggregation for authoritative exact-cache scoring."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite, sqrt

import numpy as np


def trip_pool(tiles: np.ndarray, valid_counts: np.ndarray | None = None) -> np.ndarray:
    """Return mean, maximum, and minimum descriptors for token tiles."""

    values = np.asarray(tiles)
    if values.ndim < 2 or values.shape[-2] == 0 or values.shape[-1] == 0:
        raise ValueError("tiles must end in non-empty token and channel dimensions")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("tiles must contain finite floating-point values")
    if valid_counts is None:
        counts = np.full(values.shape[0], values.shape[-2], dtype=np.int64)
    else:
        counts = np.asarray(valid_counts)
        if (
            counts.shape != (values.shape[0],)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0)
            or np.any(counts > values.shape[-2])
        ):
            raise ValueError("tile valid counts are incompatible")
    pooled = []
    for tile, count in zip(values, counts, strict=True):
        valid = tile[: int(count)]
        # VEDA accumulates the mean in float32 even when the captured tiles
        # are stored as float16.  NumPy otherwise accumulates a float16 mean
        # in float16 and can change a near-boundary Top-k decision.
        mean = valid.sum(axis=0, dtype=np.float32) / np.float32(count)
        pooled.append(
            np.concatenate(
                (
                    mean,
                    valid.max(axis=0).astype(np.float32),
                    valid.min(axis=0).astype(np.float32),
                )
            )
        )
    descriptor = np.stack(pooled)
    return descriptor.astype(np.float32, copy=False)


def segmented_top_k(
    logits: np.ndarray,
    key_segments: np.ndarray,
    key_valid_counts: np.ndarray,
    *,
    target_blocks: int,
    reference_blocks: int,
    required_keys: np.ndarray,
) -> np.ndarray:
    """Apply VEDA target/reference budgets while always retaining global keys."""

    scores = np.asarray(logits)
    segments = np.asarray(key_segments)
    counts = np.asarray(key_valid_counts)
    required = np.asarray(required_keys)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("logits must be a non-empty matrix")
    if segments.shape != (scores.shape[1],) or not np.issubdtype(segments.dtype, np.integer):
        raise ValueError("key segments must align with key tiles")
    if set(np.unique(segments)) - {0, 1, 2}:
        raise ValueError("key segments contain an unknown segment")
    if (
        counts.shape != (scores.shape[1],)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
    ):
        raise ValueError("key valid counts must align with non-empty key tiles")
    if required.shape != scores.shape or required.dtype != np.bool_:
        raise ValueError("required keys must align with logits")
    if not np.isfinite(scores).all():
        raise ValueError("selector logits must be finite")
    if target_blocks <= 0 or reference_blocks < 0:
        raise ValueError("segment budgets are invalid")

    mask = required.copy()
    mask[:, segments == 2] = True
    for segment, budget, name in (
        (0, target_blocks, "target"),
        (1, reference_blocks, "reference"),
    ):
        keys = np.flatnonzero(segments == segment)
        if not len(keys):
            continue
        if budget <= 0 or budget > len(keys):
            raise ValueError(f"{name} block budget cannot be satisfied")
        if np.any(required[:, keys].sum(axis=1) > budget):
            raise ValueError(f"required {name} keys exceed the block budget")
        for query, row in enumerate(scores[:, keys]):
            order = np.lexsort((keys, -row))
            needed = budget - int(mask[query, keys].sum())
            if needed == 0:
                continue
            for key in keys[order]:
                if not mask[query, key]:
                    mask[query, key] = True
                    needed -= 1
                    if needed == 0:
                        break
            if int(mask[query, keys].sum()) != budget:
                raise AssertionError(f"{name} block selection did not meet its exact budget")
    return mask


def reference_attention(
    q_tiles: np.ndarray,
    k_tiles: np.ndarray,
    v_tiles: np.ndarray,
    block_mask: np.ndarray,
    *,
    query_valid_counts: np.ndarray | None = None,
    key_valid_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Compute deterministic float32 attention for a tile-level block mask."""

    q = np.asarray(q_tiles, dtype=np.float32)
    k = np.asarray(k_tiles, dtype=np.float32)
    v = np.asarray(v_tiles, dtype=np.float32)
    mask = np.asarray(block_mask)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q, K, and V must use [tile, token, channel] layout")
    if q.shape[-1] == 0 or q.shape[1] == 0 or k.shape[0] == 0 or k.shape[1] == 0:
        raise ValueError("attention dimensions must be non-empty")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q, K, and V dimensions are incompatible")
    if mask.shape != (q.shape[0], k.shape[0]) or mask.dtype != np.bool_:
        raise ValueError("block mask must be boolean and aligned with Q/K tiles")
    if not mask.any(axis=1).all():
        raise ValueError("every query tile must select at least one key tile")
    if not all(np.isfinite(value).all() for value in (q, k, v)):
        raise ValueError("Q, K, and V must be finite")

    if query_valid_counts is None:
        q_counts = np.full(q.shape[0], q.shape[1], dtype=np.int64)
    else:
        q_counts = np.asarray(query_valid_counts)
    if key_valid_counts is None:
        k_counts = np.full(k.shape[0], k.shape[1], dtype=np.int64)
    else:
        k_counts = np.asarray(key_valid_counts)
    if (
        q_counts.shape != (q.shape[0],)
        or k_counts.shape != (k.shape[0],)
        or not np.issubdtype(q_counts.dtype, np.integer)
        or not np.issubdtype(k_counts.dtype, np.integer)
        or np.any(q_counts <= 0)
        or np.any(q_counts > q.shape[1])
        or np.any(k_counts <= 0)
        or np.any(k_counts > k.shape[1])
    ):
        raise ValueError("attention valid counts are incompatible")
    output = np.zeros_like(q)
    scale = np.float32(1 / sqrt(q.shape[-1]))
    for query_tile in range(q.shape[0]):
        selected = np.flatnonzero(mask[query_tile])
        keys = np.concatenate([k[index, : int(k_counts[index])] for index in selected])
        values = np.concatenate([v[index, : int(k_counts[index])] for index in selected])
        query_count = int(q_counts[query_tile])
        attention_logits = q[query_tile, :query_count] @ keys.T * scale
        attention_logits -= attention_logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(attention_logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        output[query_tile, :query_count] = probabilities @ values
    if not np.isfinite(output).all():
        raise ValueError("attention output is not finite")
    return output


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile with an explicit definition."""

    checked = _finite_values(values, name="percentile values")
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("percentile probability must be within [0, 1]")
    ordered = sorted(checked)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    if fraction == 0:
        return ordered[lower]
    return ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction


def tail_error(errors: Sequence[float], *, mean_weight: float) -> float:
    """Combine mean error and P95 tail error using the frozen mixture."""

    checked = _finite_nonnegative(errors, name="errors")
    if not isfinite(mean_weight) or not 0 <= mean_weight <= 1:
        raise ValueError("mean weight must be within [0, 1]")
    mean = fsum(checked) / len(checked)
    return mean_weight * mean + (1 - mean_weight) * percentile(checked, 0.95)


def _finite_nonnegative(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    checked = tuple(values)
    if not checked or any(not isfinite(value) or value < 0 for value in checked):
        raise ValueError(f"{name} must be non-empty, finite, and non-negative")
    return checked


def _finite_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    checked = tuple(values)
    if not checked or any(not isfinite(value) for value in checked):
        raise ValueError(f"{name} must be non-empty and finite")
    return checked
