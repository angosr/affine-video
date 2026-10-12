"""Fixed, code-owned selector architecture loaded from weight-only submissions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open

from scoring.model import ModelValidationError, TensorSpec

_ROUTE = re.compile(r"^[A-Za-z0-9_-]+$")
_DTYPES = frozenset({"float16", "float32"})


@dataclass(frozen=True, slots=True)
class SelectorArchitecture:
    routes: tuple[str, ...]
    descriptor_dim: int
    hidden_dim: int
    projection_dim: int
    storage_dtype: str
    activation: str = "relu"

    def __post_init__(self) -> None:
        if not self.routes or len(set(self.routes)) != len(self.routes):
            raise ValueError("selector routes must be non-empty and unique")
        if any(not _ROUTE.fullmatch(route) for route in self.routes):
            raise ValueError("selector route names contain unsupported characters")
        if min(self.descriptor_dim, self.hidden_dim, self.projection_dim) <= 0:
            raise ValueError("selector dimensions must be positive")
        if self.storage_dtype not in _DTYPES:
            raise ValueError("selector storage dtype must be float16 or float32")
        if self.activation != "relu":
            raise ValueError("selector activation must be relu")

    @property
    def tensor_schema(self) -> dict[str, TensorSpec]:
        schema: dict[str, TensorSpec] = {}
        for route in self.routes:
            prefix = f"selector.{route}"
            for side in ("q", "k"):
                schema[f"{prefix}.{side}_hidden.weight"] = TensorSpec(
                    (self.hidden_dim, self.descriptor_dim), self.storage_dtype
                )
                schema[f"{prefix}.{side}_hidden.bias"] = TensorSpec(
                    (self.hidden_dim,), self.storage_dtype
                )
                schema[f"{prefix}.{side}_output.weight"] = TensorSpec(
                    (self.projection_dim, self.hidden_dim), self.storage_dtype
                )
                schema[f"{prefix}.{side}_output.bias"] = TensorSpec(
                    (self.projection_dim,), self.storage_dtype
                )
        return schema


@dataclass(frozen=True, slots=True)
class FixedSelector:
    architecture: SelectorArchitecture
    weights: dict[str, np.ndarray]

    def logits(
        self, route: str, q_descriptors: np.ndarray, k_descriptors: np.ndarray
    ) -> np.ndarray:
        """Return one finite score for every Query/Key tile pair."""

        if route not in self.architecture.routes:
            raise ValueError("unknown selector route")
        q = _descriptors(q_descriptors, self.architecture.descriptor_dim, "Q")
        k = _descriptors(k_descriptors, self.architecture.descriptor_dim, "K")
        prefix = f"selector.{route}"
        q_projected = self._project(prefix, "q", q)
        k_projected = self._project(prefix, "k", k)
        result = q_projected @ k_projected.T / np.float32(sqrt(self.architecture.projection_dim))
        if not np.isfinite(result).all():
            raise ModelValidationError(
                "MODEL_RUNTIME_INVALID", "selector produced non-finite logits"
            )
        return result.astype(np.float32, copy=False)

    def _project(self, prefix: str, side: str, descriptors: np.ndarray) -> np.ndarray:
        hidden = (
            descriptors @ self.weights[f"{prefix}.{side}_hidden.weight"].T
            + self.weights[f"{prefix}.{side}_hidden.bias"]
        )
        np.maximum(hidden, 0, out=hidden)
        return (
            hidden @ self.weights[f"{prefix}.{side}_output.weight"].T
            + self.weights[f"{prefix}.{side}_output.bias"]
        )


def load_selector(path: str | Path, architecture: SelectorArchitecture) -> FixedSelector:
    """Load exactly the tensors required by the fixed selector architecture."""

    expected = architecture.tensor_schema
    weights: dict[str, np.ndarray] = {}
    try:
        with safe_open(Path(path), framework="numpy") as tensors:
            if tensors.metadata():
                raise ModelValidationError(
                    "MODEL_METADATA_INVALID", "safetensors metadata must be empty"
                )
            if sorted(tensors.keys()) != sorted(expected):
                raise ModelValidationError(
                    "MODEL_SCHEMA_INVALID", "tensor keys do not match the selector architecture"
                )
            for name, tensor_spec in expected.items():
                value = tensors.get_tensor(name)
                if value.shape != tensor_spec.shape or value.dtype.name != tensor_spec.dtype:
                    raise ModelValidationError(
                        "MODEL_SCHEMA_INVALID",
                        f"tensor {name} does not match the selector architecture",
                    )
                if not np.isfinite(value).all():
                    raise ModelValidationError("MODEL_NON_FINITE", f"tensor {name} is not finite")
                weights[name] = np.asarray(value, dtype=np.float32)
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
