from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from validator.production.runtime import (
    DenseRolloutRuntime,
    RolloutCapture,
    _canonicalize_safetensors_header,
)


def test_dense_video_excludes_audio_decoding_but_preserves_audio_latents(tmp_path, monkeypatch):
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    video_rows = np.arange(32, dtype=np.float32).reshape(4, 8)
    audio_rows = np.arange(16, dtype=np.float32).reshape(2, 8)
    calls = []

    class Decoder:
        sample_rate = 24_000

        def video(self, rows, geometry):
            np.testing.assert_array_equal(rows, video_rows)
            assert geometry == "parsed-geometry"
            return frames

        def audio(self, _rows):
            pytest.fail("the dense video artifact must not decode an unscored audio track")

    def write_mp4(path, decoded, audio_tracks, sample_rate):
        np.testing.assert_array_equal(decoded, frames)
        assert audio_tracks == [] and sample_rate == 24_000
        from pathlib import Path

        calls.append(path)
        Path(path).write_bytes(decoded.tobytes())

    package = ModuleType("miowtion")
    infer = ModuleType("miowtion.infer")
    decode = ModuleType("miowtion.infer.decode")
    decode.write_mp4 = write_mp4
    package.infer = infer
    infer.decode = decode
    monkeypatch.setitem(sys.modules, "miowtion", package)
    monkeypatch.setitem(sys.modules, "miowtion.infer", infer)
    monkeypatch.setitem(sys.modules, "miowtion.infer.decode", decode)
    runtime = object.__new__(DenseRolloutRuntime)
    runtime.decoder = Decoder()
    runtime.torch = SimpleNamespace(from_numpy=lambda value: value)
    runtime.data = SimpleNamespace(parse_geometry=lambda _value: "parsed-geometry")
    capture = RolloutCapture(
        (), 0.0, 0, (), video_rows, audio_rows, ((2, "late", video_rows.copy()),),
    )

    first = runtime.write_dense_outputs(
        capture, geometry="small", rollout_id="sample", directory=tmp_path / "first",
    )
    second = runtime.write_dense_outputs(
        capture, geometry="small", rollout_id="sample", directory=tmp_path / "second",
    )
    assert len(calls) == 2
    assert first.video_path.read_bytes() == second.video_path.read_bytes() == frames.tobytes()
    assert first.latent_path.read_bytes() == second.latent_path.read_bytes()
    saved = load_file(first.latent_path)
    np.testing.assert_array_equal(saved["video_rows"], video_rows)
    np.testing.assert_array_equal(saved["audio_rows"], audio_rows)
    np.testing.assert_array_equal(saved["video_step_002_rows"], video_rows)
    assert not tuple(tmp_path.rglob("*.tmp*"))


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
