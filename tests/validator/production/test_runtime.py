from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from validator.production.runtime import _canonicalize_safetensors_header


def test_latent_header_is_stable_without_changing_tensor_payload(tmp_path):
    tensors = {
        "video_rows": np.arange(32, dtype=np.float32).reshape(4, 8),
        "audio_rows": np.arange(16, dtype=np.float16).reshape(2, 8),
    }
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_file(tensors, first, metadata={"geometry": "small", "rollout_id": "sample"})
    original = first.read_bytes()
    header_size = int.from_bytes(original[:8], "little")
    header = json.loads(original[8:8 + header_size])
    reordered = dict(reversed(tuple(header.items())))
    reordered["__metadata__"] = dict(reversed(tuple(header["__metadata__"].items())))
    encoded = json.dumps(reordered, separators=(",", ":")).encode()
    second.write_bytes(original[:8] + encoded.ljust(header_size, b" ") + original[8 + header_size:])
    assert first.read_bytes() != second.read_bytes()

    _canonicalize_safetensors_header(first)
    _canonicalize_safetensors_header(second)
    canonical = first.read_bytes()
    assert canonical == second.read_bytes()
    assert canonical[:8] == original[:8]
    assert canonical[8 + header_size:] == original[8 + header_size:]
    assert first.stat().st_size == len(original)
    for name, values in tensors.items():
        np.testing.assert_array_equal(load_file(first)[name], values)
        np.testing.assert_array_equal(load_file(second)[name], values)
    _canonicalize_safetensors_header(first)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(canonical).digest()


@pytest.mark.parametrize("content", [
    b"",
    (0).to_bytes(8, "little"),
    (1024 * 1024 + 1).to_bytes(8, "little"),
    (32).to_bytes(8, "little") + b"{}",
    (8).to_bytes(8, "little") + b"invalid!",
    (8).to_bytes(8, "little") + b"[]      ",
])
def test_latent_header_rejects_invalid_or_unbounded_input(tmp_path, content):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match="safetensors header"):
        _canonicalize_safetensors_header(path)
    assert path.read_bytes() == content
