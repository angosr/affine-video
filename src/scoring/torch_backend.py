"""CUDA implementation of the frozen exact-case arithmetic.

Torch is imported lazily because miners and chain-only validators do not
need the GPU runtime.  The NumPy implementation remains the small reference;
the authoritative evaluator selects this backend explicitly.
"""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from scoring.evaluate import tail_error
from scoring.metrics import relative_squared_error
from scoring.torch_selector import segmented_top_k, selector_logits, trip_pool

if TYPE_CHECKING:
    from scoring.exact import ExactCase
    from scoring.selector import FixedSelector
    from scoring.spec import BudgetSpec


def dense_reference_attention(
    q_tiles: np.ndarray,
    k_tiles: np.ndarray,
    v_tiles: np.ndarray,
    query_valid_counts: np.ndarray,
    key_valid_counts: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Compute one all-key float32 reference on the authoritative CUDA backend."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the torch scoring backend is unavailable") from exc
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the authoritative torch backend requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    q = torch.as_tensor(q_tiles, device=resolved, dtype=torch.float32)
    k = torch.as_tensor(k_tiles, device=resolved, dtype=torch.float32)
    v = torch.as_tensor(v_tiles, device=resolved, dtype=torch.float32)
    q_counts = torch.as_tensor(query_valid_counts, device=resolved)
    k_counts = torch.as_tensor(key_valid_counts, device=resolved)
    mask = torch.ones((q.shape[0], k.shape[0]), dtype=torch.bool, device=resolved)
    return _attention(torch, q, k, v, q_counts, k_counts, mask).cpu().numpy()


class TorchCaseScorer:
    """Hold one exact case on a CUDA device while scoring both selectors."""

    def __init__(self, case: ExactCase, *, device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("the torch scoring backend is unavailable") from exc
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the authoritative torch backend requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.device = resolved
        self.case = case
        self.q = torch.as_tensor(case.q, device=resolved, dtype=torch.float32)
        self.k = torch.as_tensor(case.k, device=resolved, dtype=torch.float32)
        self.v = torch.as_tensor(case.v, device=resolved, dtype=torch.float32)
        self.dense = np.asarray(case.dense, dtype=np.float32)
        self.q_counts = torch.as_tensor(case.query_valid_counts, device=resolved)
        self.k_counts = torch.as_tensor(case.key_valid_counts, device=resolved)
        self.segments = torch.as_tensor(case.key_segments, device=resolved)
        self.required = torch.as_tensor(case.required_keys, device=resolved)
        self.q_descriptors = trip_pool(torch, self.q, self.q_counts)
        self.k_descriptors = trip_pool(torch, self.k, self.k_counts)

    def score(
        self,
        selector: FixedSelector,
        budgets: tuple[BudgetSpec, ...] | list[BudgetSpec],
        tail_mean_weight: float,
    ) -> tuple[list[float], list[list[float]]]:
        torch = self.torch
        q_projection = torch.as_tensor(
            selector.weights[f"{self.case.route}.proj_q"],
            device=self.device,
        )
        k_projection = torch.as_tensor(
            selector.weights[f"{self.case.route}.proj_k"],
            device=self.device,
        )
        logits = selector_logits(
            torch, self.q_descriptors, self.k_descriptors, q_projection, k_projection,
        )

        qualities: list[float] = []
        all_errors: list[list[float]] = []
        for budget in budgets:
            mask = segmented_top_k(
                torch, logits, self.segments, self.k_counts, required_keys=self.required,
                target_blocks=budget.target_blocks,
                reference_blocks=budget.reference_blocks,
            )
            sparse = self._attention(mask).cpu().numpy()
            errors = [
                relative_squared_error(
                    self.dense[index, : int(count)].reshape(-1),
                    sparse[index, : int(count)].reshape(-1),
                    epsilon=self.case.epsilon,
                )
                for index, count in enumerate(self.case.query_valid_counts)
            ]
            all_errors.append(errors)
            qualities.append(1 / (1 + tail_error(errors, mean_weight=tail_mean_weight)))
        return qualities, all_errors

    def _attention(self, mask: Any) -> Any:
        return _attention(
            self.torch,
            self.q,
            self.k,
            self.v,
            self.q_counts,
            self.k_counts,
            mask,
        )


def _attention(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    q_counts: Any,
    k_counts: Any,
    mask: Any,
) -> Any:
    output = torch.zeros_like(q)
    scale = 1 / sqrt(q.shape[-1])
    flat_k = k.flatten(0, 1)
    flat_v = v.flatten(0, 1)
    key_rows = torch.arange(k.shape[1], device=k.device)[None, :]
    key_token_valid = key_rows < k_counts[:, None]
    for query, count_value in enumerate(q_counts):
        selected_tokens = (mask[query, :, None] & key_token_valid).flatten()
        keys = flat_k[selected_tokens]
        values = flat_v[selected_tokens]
        count = int(count_value)
        probabilities = torch.softmax(q[query, :count] @ keys.T * scale, dim=-1)
        output[query, :count] = probabilities @ values
    return output
