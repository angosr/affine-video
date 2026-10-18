"""One production TripPool/projection/stable-mask contract for exact and video.

Torch is supplied by the caller so importing the miner/chain packages does not
require the GPU environment. NumPy arithmetic remains an independent reference.
"""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace
from typing import Any

from scoring.model import ModelValidationError


def gather_and_pool(packed: Any, layout: Any, heads: Any = None) -> tuple[Any, Any]:
    """Gather and pool in one pass using the pinned runtime's CUDA kernel."""

    from miowtion.kernels.tile_gather_triton import gather_and_pool as fused_pool

    return fused_pool(packed, layout, heads)


def trip_pool(torch: Any, tiles: Any, counts: Any) -> Any:
    """Pool [tile, token, channel] or [head, tile, token, channel] in FP32."""

    if tiles.ndim not in {3, 4} or counts.shape != (tiles.shape[-3],):
        raise ValueError("tile counts do not align with selector input")
    if bool(((counts < 0) | (counts > tiles.shape[-2])).any()):
        raise ValueError("tile valid prefixes are out of range")
    valid = torch.arange(tiles.shape[-2], device=tiles.device)[None, :] < counts[:, None]
    if tiles.is_cuda:
        if tiles.shape[-2] != 128:
            raise ValueError("CUDA selector pooling requires 128-token tiles")
        # Cached single-head cases and full-video packed heads must use the
        # same reduction kernel, not merely the same algebraic expression.
        packed = (
            tiles.reshape(-1, 1, tiles.shape[-1]) if tiles.ndim == 3
            else tiles.permute(1, 2, 0, 3).reshape(-1, tiles.shape[0], tiles.shape[-1])
        )
        layout = SimpleNamespace(
            n_tiles=counts.numel(), num_slots=counts.numel() * 128,
            gather_index=torch.arange(counts.numel() * 128, device=tiles.device),
            slot_valid=valid.flatten(), valid_count=counts,
        )
        _, features = gather_and_pool(packed, layout)
        return features[0] if tiles.ndim == 3 else features
    mean = tiles.masked_fill(~valid[..., None], 0).sum(dim=-2, dtype=torch.float32)
    mean = mean / counts.clamp(min=1)[:, None].float()
    maximum = tiles.masked_fill(~valid[..., None], float("-inf")).amax(dim=-2).float()
    minimum = tiles.masked_fill(~valid[..., None], float("inf")).amin(dim=-2).float()
    features = torch.cat((mean, maximum, minimum), dim=-1)
    return torch.where(counts[:, None] > 0, features, 0.0)


def selector_logits(
    torch: Any, q_features: Any, k_features: Any, q_projection: Any, k_projection: Any,
) -> Any:
    """Residual TripPool projections and scaled QK, independently per head."""

    dimension = q_projection.shape[-1]
    q_hat = q_features @ q_projection.float() + q_features[..., :dimension]
    k_hat = k_features @ k_projection.float() + k_features[..., :dimension]
    logits = q_hat @ k_hat.transpose(-1, -2) / sqrt(dimension)
    _require_finite_logits(torch, logits)
    return logits


def segmented_top_k(
    torch: Any,
    logits: Any,
    key_segments: Any,
    key_valid_counts: Any,
    *,
    target_blocks: int,
    reference_blocks: int,
    required_keys: Any,
) -> Any:
    """Select exact per-segment budgets; tie-break on ascending key index.

    Leading head dimensions broadcast over required keys. Empty runtime tiles
    never enter the budget or the global mask; exact caches contain no empties.
    """

    _require_finite_logits(torch, logits)
    if target_blocks <= 0 or reference_blocks < 0:
        raise ValueError("segment budgets are invalid")
    valid = key_valid_counts > 0
    mask = torch.broadcast_to(required_keys, logits.shape).clone()
    if bool((mask & ~valid).any()):
        raise ValueError("required keys include an empty tile")
    mask[..., key_segments == 2] = valid[key_segments == 2]
    for segment, budget, name in (
        (0, target_blocks, "target"),
        (1, reference_blocks, "reference"),
    ):
        keys = torch.nonzero((key_segments == segment) & valid, as_tuple=False).flatten()
        if not keys.numel():
            continue
        if budget <= 0 or budget > keys.numel():
            raise ValueError(f"{name} block budget cannot be satisfied")
        required = mask.index_select(-1, keys)
        if bool((required.sum(dim=-1) > budget).any()):
            raise ValueError(f"required {name} keys exceed the block budget")
        scores = logits.index_select(-1, keys).masked_fill(required, float("inf"))
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        chosen = order[..., :budget]
        segment_mask = torch.zeros_like(required)
        segment_mask.scatter_(-1, chosen, True)
        mask[..., keys] = segment_mask
    return mask


def _require_finite_logits(torch: Any, logits: Any) -> None:
    if not bool(torch.isfinite(logits).all()):
        raise ModelValidationError("MODEL_RUNTIME_INVALID", "selector produced non-finite logits")


def runtime_block_mask(
    torch: Any,
    logits: Any,
    layout: Any,
    *,
    target_blocks: int,
    reference_blocks: int,
    query_rows: Any | None = None,
) -> Any:
    """Adapt the pinned tile layout to the shared mask, including global rows.

    Logits contain video-query rows only. Passing explicit query rows returns
    their masks alone; otherwise all video rows plus dense global rows result.
    """

    columns = torch.arange(layout.n_tiles, device=logits.device)
    rows = (
        torch.arange(layout.n_video_tiles, device=logits.device)
        if query_rows is None else query_rows
    )
    if logits.shape[-2:] != (rows.numel(), layout.n_tiles):
        raise ValueError("runtime logits do not align with query rows and tile layout")
    segments = torch.where(columns < layout.n_ref_tiles, 1, 0)
    segments = torch.where(columns >= layout.n_video_tiles, 2, segments)
    required = (rows[:, None] == columns[None, :]) & (layout.valid_count > 0)
    mask = segmented_top_k(
        torch, logits, segments, layout.valid_count,
        target_blocks=target_blocks, reference_blocks=reference_blocks,
        required_keys=required,
    )
    if query_rows is None:
        global_rows = (layout.valid_count > 0).expand(
            *mask.shape[:-2], layout.n_tiles - layout.n_video_tiles, layout.n_tiles,
        )
        mask = torch.cat((mask, global_rows), dim=-2)
    return mask
