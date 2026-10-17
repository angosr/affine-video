"""Prepare the single owner-bound safetensors file submitted by a Miner."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

_SS58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,49}$")
_MAX_MODEL_BYTES = 560_000_000


@dataclass(frozen=True, slots=True)
class PreparedModel:
    path: Path
    sha256: str
    size_bytes: int


def prepare_model(source: str | Path, output: str | Path, *, hotkey: str) -> PreparedModel:
    """Copy tensor values unchanged and replace metadata with the owner hotkey."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not _SS58.fullmatch(hotkey):
        raise ValueError("hotkey must be a valid SS58-shaped address")
    if not source_path.is_file() or source_path.stat().st_size > _MAX_MODEL_BYTES:
        raise ValueError("source model is missing or exceeds the track byte limit")
    if source_path == output_path:
        raise ValueError("source and output paths must differ")
    if output_path.exists():
        raise ValueError("output path already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with safe_open(source_path, framework="numpy") as model:
            tensors = {name: model.get_tensor(name) for name in list(model.keys())}
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        save_file(tensors, temporary, metadata={"hotkey": hotkey})
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, output_path)
        temporary = None
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return PreparedModel(output_path, _sha256(output_path), output_path.stat().st_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
