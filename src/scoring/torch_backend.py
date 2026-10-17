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
from scoring.model import ModelValidationError

if TYPE_CHECKING:
    from scoring.exact import ExactCase
    from scoring.selector import FixedSelector
    from scoring.spec import BudgetSpec


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
        token_rows = torch.arange(self.k.shape[1], device=resolved)[None, :]
        self.key_token_valid = token_rows < self.k_counts[:, None]
        self.q_descriptors = self._pool(self.q, self.q_counts)
        self.k_descriptors = self._pool(self.k, self.k_counts)

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
        dimension = selector.architecture.head_dim
        q_hat = self.q_descriptors @ q_projection + self.q_descriptors[:, :dimension]
        k_hat = self.k_descriptors @ k_projection + self.k_descriptors[:, :dimension]
        logits = q_hat @ k_hat.T / sqrt(dimension)
        _require_finite_logits(torch, logits)

        qualities: list[float] = []
        all_errors: list[list[float]] = []
        for budget in budgets:
            mask = self._segmented_top_k(
                logits,
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

    def _pool(self, tiles: Any, counts: Any) -> Any:
        torch = self.torch
        rows = torch.arange(tiles.shape[1], device=self.device)[None, :]
        valid = rows < counts[:, None]
        mean = (tiles * valid[:, :, None]).sum(dim=1) / counts[:, None].float()
        maximum = tiles.masked_fill(~valid[:, :, None], float("-inf")).amax(dim=1)
        minimum = tiles.masked_fill(~valid[:, :, None], float("inf")).amin(dim=1)
        return torch.cat((mean, maximum, minimum), dim=-1)

    def _segmented_top_k(
        self,
        logits: Any,
        *,
        target_blocks: int,
        reference_blocks: int,
    ) -> Any:
        torch = self.torch
        mask = self.required.clone()
        mask[:, self.segments == 2] = True
        for segment, budget, name in (
            (0, target_blocks, "target"),
            (1, reference_blocks, "reference"),
        ):
            keys = torch.nonzero(self.segments == segment, as_tuple=False).flatten()
            if not keys.numel():
                continue
            if budget <= 0 or budget > keys.numel():
                raise ValueError(f"{name} block budget cannot be satisfied")
            required = mask.index_select(1, keys)
            if bool((required.sum(dim=1) > budget).any()):
                raise ValueError(f"required {name} keys exceed the block budget")
            scores = logits.index_select(1, keys).masked_fill(required, float("inf"))
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            chosen = order[:, :budget]
            segment_mask = torch.zeros_like(required)
            segment_mask.scatter_(1, chosen, True)
            mask[:, keys] = segment_mask
        return mask

    def _attention(self, mask: Any) -> Any:
        torch = self.torch
        output = torch.zeros_like(self.q)
        scale = 1 / sqrt(self.q.shape[-1])
        flat_k = self.k.flatten(0, 1)
        flat_v = self.v.flatten(0, 1)
        for query, count_value in enumerate(self.case.query_valid_counts):
            selected_tokens = (mask[query, :, None] & self.key_token_valid).flatten()
            keys = flat_k[selected_tokens]
            values = flat_v[selected_tokens]
            count = int(count_value)
            probabilities = torch.softmax(self.q[query, :count] @ keys.T * scale, dim=-1)
            output[query, :count] = probabilities @ values
        return output


def _require_finite_logits(torch: Any, logits: Any) -> None:
    if not bool(torch.isfinite(logits).all()):
        raise ModelValidationError("MODEL_RUNTIME_INVALID", "selector produced non-finite logits")
