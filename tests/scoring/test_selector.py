import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.model import ModelValidationError
from scoring.selector import SelectorArchitecture, load_selector


def architecture() -> SelectorArchitecture:
    return SelectorArchitecture(
        num_layers=1,
        num_heads=1,
        head_dim=2,
        storage_dtype="float32",
    )


def weights(spec: SelectorArchitecture) -> dict[str, np.ndarray]:
    values = {
        name: np.zeros(tensor.shape, dtype=tensor.dtype)
        for name, tensor in spec.tensor_schema.items()
    }
    values["layers.0.proj_q"][0, :2] = np.eye(2)
    values["layers.0.proj_k"][0, :2] = np.eye(2)
    return values


def test_schema_and_routes_cover_every_layer_and_head() -> None:
    spec = architecture()

    assert len(spec.tensor_schema) == 2
    assert spec.tensor_schema["layers.0.proj_q"].shape == (1, 6, 2)
    full = SelectorArchitecture(2, 2, 2, "float32")
    assert full.routes == (
        "layer_0_head_0",
        "layer_0_head_1",
        "layer_1_head_0",
        "layer_1_head_1",
    )
    assert set(full.tensor_schema) == {
        "layers.0.proj_q",
        "layers.0.proj_k",
        "layers.1.proj_q",
        "layers.1.proj_k",
    }


def test_weight_only_selector_produces_deterministic_pair_logits(tmp_path) -> None:
    spec = architecture()
    path = tmp_path / "selector.safetensors"
    save_file(weights(spec), path)
    selector = load_selector(path, spec)
    q = np.array([[1, 0, 9, 9, -9, -9], [0, 2, 9, 9, -9, -9]], dtype=np.float32)
    k = np.array([[3, 0, 8, 8, -8, -8], [0, 4, 8, 8, -8, -8]], dtype=np.float32)

    first = selector.logits("layer_0_head_0", q, k)
    second = selector.logits("layer_0_head_0", q, k)

    assert first == pytest.approx(second)
    assert first == pytest.approx(4 * np.array([[3, 0], [0, 8]]) / np.sqrt(2))
    assert set(selector.weights) == {
        "layer_0_head_0.proj_q",
        "layer_0_head_0.proj_k",
    }
    assert sum(value.nbytes for value in selector.weights.values()) == 2 * 6 * 2 * 4


def test_zero_veda_projection_equals_mean_pooled_qk(tmp_path) -> None:
    spec = architecture()
    path = tmp_path / "selector.safetensors"
    save_file(
        {
            name: np.zeros(tensor.shape, dtype=tensor.dtype)
            for name, tensor in spec.tensor_schema.items()
        },
        path,
    )
    selector = load_selector(path, spec)
    q = np.array([[1, 2, 7, 8, -2, -3]], dtype=np.float32)
    k = np.array([[3, 4, 9, 10, -4, -5]], dtype=np.float32)

    result = selector.logits("layer_0_head_0", q, k)

    assert result == pytest.approx(np.array([[11]], dtype=np.float32) / np.sqrt(2))


def test_selector_requires_exact_owner_metadata_when_hotkey_is_expected(tmp_path) -> None:
    spec = architecture()
    path = tmp_path / "selector.safetensors"
    save_file(weights(spec), path)

    with pytest.raises(ModelValidationError, match="submitting hotkey"):
        load_selector(path, spec, expected_hotkey="owner")

    save_file(weights(spec), path, metadata={"hotkey": "owner"})
    load_selector(path, spec, expected_hotkey="owner")

    with pytest.raises(ModelValidationError, match="submitting hotkey"):
        load_selector(path, spec, expected_hotkey="different")


def test_selector_rejects_extra_tensors_and_non_finite_runtime_input(tmp_path) -> None:
    spec = architecture()
    values = weights(spec)
    values["extra"] = np.zeros(1, dtype=np.float32)
    path = tmp_path / "selector.safetensors"
    save_file(values, path)

    with pytest.raises(ModelValidationError, match="keys"):
        load_selector(path, spec)

    values.pop("extra")
    save_file(values, path)
    selector = load_selector(path, spec)
    with pytest.raises(ModelValidationError, match="non-finite"):
        selector.logits(
            "layer_0_head_0",
            np.array([[np.nan, 0, 0, 0, 0, 0]], dtype=np.float32),
            np.zeros((1, 6), dtype=np.float32),
        )
