import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from miner.prepare import prepare_model

HOTKEY = "5C9ywXcKhSNrFY2ttziLTLNPZB9XPqGaKbLpvh6J6TuJyBQQ"


def test_prepare_model_replaces_metadata_without_changing_tensors(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    output = tmp_path / "selector.safetensors"
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_file({"weight": expected}, source, metadata={"ignored": "value"})

    result = prepare_model(source, output, hotkey=HOTKEY)

    assert result.path == output.resolve()
    assert result.size_bytes == output.stat().st_size
    with safe_open(output, framework="numpy") as model:
        assert model.metadata() == {"hotkey": HOTKEY}
        np.testing.assert_array_equal(model.get_tensor("weight"), expected)


def test_prepare_model_refuses_overwrite_and_invalid_hotkey(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    output = tmp_path / "selector.safetensors"
    save_file({"weight": np.ones(1, dtype=np.float32)}, source)
    output.write_bytes(b"existing")

    with pytest.raises(ValueError, match="hotkey"):
        prepare_model(source, tmp_path / "new.safetensors", hotkey="invalid")
    with pytest.raises(ValueError, match="already exists"):
        prepare_model(source, output, hotkey=HOTKEY)
