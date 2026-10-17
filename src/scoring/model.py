"""CPU validation and canonical digesting of selector weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open


class ModelValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ValidatedModel:
    file_sha256: str
    model_digest: str
    size_bytes: int
    tensor_count: int


def validate_model(
    path: str | Path,
    *,
    schema: dict[str, TensorSpec],
    max_bytes: int,
    expected_hotkey: str | None = None,
) -> ValidatedModel:
    """Validate exact tensor keys, shapes, dtypes, and finite values on CPU."""

    model_path = Path(path)
    size = model_path.stat().st_size
    if size > max_bytes:
        raise ModelValidationError("MODEL_TOO_LARGE", "model exceeds the byte limit")

    digest = hashlib.sha256()
    digest.update(b"veda-model-digest-v1\0")
    try:
        with safe_open(model_path, framework="numpy") as tensors:
            metadata = tensors.metadata() or {}
            expected_metadata = {} if expected_hotkey is None else {"hotkey": expected_hotkey}
            if metadata != expected_metadata:
                code = (
                    "MODEL_OWNER_MISMATCH"
                    if expected_hotkey is not None and set(metadata) == {"hotkey"}
                    else "MODEL_METADATA_INVALID"
                )
                raise ModelValidationError(
                    code,
                    "safetensors metadata must contain only the submitting hotkey"
                    if expected_hotkey is not None
                    else "safetensors metadata must be empty",
                )
            keys = sorted(tensors.keys())
            if keys != sorted(schema):
                raise ModelValidationError(
                    "MODEL_SCHEMA_INVALID", "tensor keys do not match the track schema"
                )
            for key in keys:
                array = tensors.get_tensor(key)
                expected = schema[key]
                if array.shape != expected.shape or array.dtype.name != expected.dtype:
                    raise ModelValidationError(
                        "MODEL_SCHEMA_INVALID", f"tensor {key} does not match the track schema"
                    )
                if not np.isfinite(array).all():
                    raise ModelValidationError("MODEL_NON_FINITE", f"tensor {key} is not finite")
                canonical = np.array(
                    array.astype(array.dtype.newbyteorder("<"), copy=False),
                    copy=True,
                    order="C",
                )
                canonical[canonical == 0] = 0
                key_bytes = key.encode("utf-8")
                dtype_bytes = array.dtype.name.encode("ascii")
                digest.update(len(key_bytes).to_bytes(4, "big"))
                digest.update(key_bytes)
                digest.update(len(dtype_bytes).to_bytes(2, "big"))
                digest.update(dtype_bytes)
                digest.update(len(array.shape).to_bytes(2, "big"))
                for dimension in array.shape:
                    digest.update(dimension.to_bytes(8, "big"))
                digest.update(canonical.tobytes(order="C"))
    except ModelValidationError:
        raise
    except (SafetensorError, OSError, ValueError) as exc:
        raise ModelValidationError("MODEL_SCHEMA_INVALID", "invalid safetensors file") from exc

    return ValidatedModel(
        file_sha256=_file_sha256(model_path),
        model_digest=digest.hexdigest(),
        size_bytes=size,
        tensor_count=len(schema),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
