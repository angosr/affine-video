"""Veda predictor arithmetic loaded from weight-only submissions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open

from scoring.model import ModelValidationError, TensorSpec

_ROUTE = re.compile(r"^layer_(0|[1-9][0-9]*)_head_(0|[1-9][0-9]*)$")
_DTYPES = frozenset({"float16", "float32"})
_SAFETENSOR_DTYPES = {"float16": "F16", "float32": "F32"}


def route_name(layer: int, head: int) -> str:
    """Return the canonical cache route for one Veda layer and head."""

    if type(layer) is not int or layer < 0 or type(head) is not int or head < 0:
        raise ValueError("selector layer and head must be non-negative integers")
    return f"layer_{layer}_head_{head}"


def route_indices(route: str) -> tuple[int, int]:
    """Parse a canonical cache route into its layer and head indices."""

    if not isinstance(route, str) or (match := _ROUTE.fullmatch(route)) is None:
        raise ValueError("selector route must use layer_<index>_head_<index>")
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True, slots=True)
class SelectorArchitecture:
    """Code-owned Veda v1 predictor contract covering every layer and head."""

    num_layers: int
    num_heads: int
    head_dim: int
    storage_dtype: str

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_heads, self.head_dim) <= 0:
            raise ValueError("selector layer, head, and channel counts must be positive")
        if self.storage_dtype not in _DTYPES:
            raise ValueError("selector storage dtype must be float16 or float32")

    @property
    def routes(self) -> tuple[str, ...]:
        """Return every layer/head pair in one canonical order."""

        return tuple(
            route_name(layer, head)
            for layer in range(self.num_layers)
            for head in range(self.num_heads)
        )

    @property
    def descriptor_dim(self) -> int:
        """TripPool width: mean, maximum, and minimum for each channel."""

        return 3 * self.head_dim

    @property
    def tensor_schema(self) -> dict[str, TensorSpec]:
        shape = (self.num_heads, self.descriptor_dim, self.head_dim)
        return {
            f"layers.{layer}.proj_{side}": TensorSpec(shape, self.storage_dtype)
            for layer in range(self.num_layers)
            for side in ("q", "k")
        }


@dataclass(frozen=True, slots=True)
class FixedSelector:
    architecture: SelectorArchitecture
    weights: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        expected = {
            f"{route}.proj_{side}"
            for route in self.architecture.routes
            for side in ("q", "k")
        }
        if set(self.weights) != expected:
            raise ValueError("loaded selector projections do not cover the evaluation routes")
        shape = (self.architecture.descriptor_dim, self.architecture.head_dim)
        for projection in self.weights.values():
            if projection.shape != shape or projection.dtype != np.float32:
                raise ValueError("loaded selector projection has the wrong shape or dtype")
            if not np.isfinite(projection).all():
                raise ModelValidationError(
                    "MODEL_NON_FINITE", "loaded selector projection is not finite"
                )

    def logits(
        self, route: str, q_descriptors: np.ndarray, k_descriptors: np.ndarray
    ) -> np.ndarray:
        """Return Veda v1 logits for one layer/head cache route.

        Veda TripPools each tile into ``[mean | max | min]``. Its learned
        projection is residual: ``x_hat = pooled @ P + mean``. The submitted
        file therefore needs only ``proj_q`` and ``proj_k`` tensors.
        """

        if route not in self.architecture.routes:
            raise ValueError("unknown selector route")
        q = _descriptors(q_descriptors, self.architecture.descriptor_dim, "Q")
        k = _descriptors(k_descriptors, self.architecture.descriptor_dim, "K")
        q_projected = self._project(q, self.weights[f"{route}.proj_q"])
        k_projected = self._project(k, self.weights[f"{route}.proj_k"])
        result = q_projected @ k_projected.T / np.float32(sqrt(self.architecture.head_dim))
        if not np.isfinite(result).all():
            raise ModelValidationError(
                "MODEL_RUNTIME_INVALID", "selector produced non-finite logits"
            )
        return result.astype(np.float32, copy=False)

    def _project(self, descriptors: np.ndarray, projection: np.ndarray) -> np.ndarray:
        mean = descriptors[:, : self.architecture.head_dim]
        return descriptors @ projection + mean


def load_selector(
    path: str | Path,
    architecture: SelectorArchitecture,
    *,
    expected_hotkey: str | None = None,
) -> FixedSelector:
    """Validate the file schema and materialize every submitted head.

    Admission already streams every tensor to validate finite values. Scoring
    rechecks the immutable file hash before this call.
    """

    expected = architecture.tensor_schema
    weights: dict[str, np.ndarray] = {}
    try:
        with safe_open(Path(path), framework="numpy") as tensors:
            metadata = tensors.metadata() or {}
            if metadata not in ({}, {"hotkey": expected_hotkey}):
                raise ModelValidationError(
                    "MODEL_OWNER_MISMATCH",
                    "selector metadata does not match the submitting hotkey",
                )
            if sorted(tensors.keys()) != sorted(expected):
                raise ModelValidationError(
                    "MODEL_SCHEMA_INVALID", "tensor keys do not match the selector architecture"
                )
            for name, tensor_spec in expected.items():
                view = tensors.get_slice(name)
                if (
                    tuple(view.get_shape()) != tensor_spec.shape
                    or view.get_dtype() != _SAFETENSOR_DTYPES[tensor_spec.dtype]
                ):
                    raise ModelValidationError(
                        "MODEL_SCHEMA_INVALID",
                        f"tensor {name} does not match the selector architecture",
                    )
            for route in architecture.routes:
                layer, head = route_indices(route)
                for side in ("q", "k"):
                    name = f"layers.{layer}.proj_{side}"
                    value = tensors.get_slice(name)[head]
                    if not np.isfinite(value).all():
                        raise ModelValidationError(
                            "MODEL_NON_FINITE", f"tensor {name} is not finite"
                        )
                    weights[f"{route}.proj_{side}"] = np.asarray(value, dtype=np.float32)
    except ModelValidationError:
        raise
    except (SafetensorError, OSError, ValueError) as exc:
        raise ModelValidationError("MODEL_SCHEMA_INVALID", "invalid selector weights") from exc
    return FixedSelector(architecture, weights)


def _descriptors(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != expected_dim:
        raise ValueError(f"{name} descriptors have the wrong shape")
    if not np.isfinite(array).all():
        raise ModelValidationError(
            "MODEL_RUNTIME_INVALID", f"{name} descriptors contain non-finite values"
        )
    return array
