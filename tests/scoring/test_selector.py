import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.model import ModelValidationError
from scoring.selector import SelectorArchitecture, load_selector


def architecture() -> SelectorArchitecture:
    return SelectorArchitecture(
        routes=("block00_head00",),
        descriptor_dim=3,
        hidden_dim=2,
        projection_dim=2,
        storage_dtype="float32",
    )


def weights(spec: SelectorArchitecture) -> dict[str, np.ndarray]:
    values = {
        name: np.zeros(tensor.shape, dtype=tensor.dtype)
        for name, tensor in spec.tensor_schema.items()
    }
    prefix = "selector.block00_head00"
    values[f"{prefix}.q_hidden.weight"][:] = [[1, 0, 0], [0, 1, 0]]
    values[f"{prefix}.k_hidden.weight"][:] = [[1, 0, 0], [0, 1, 0]]
    values[f"{prefix}.q_output.weight"][:] = np.eye(2)
    values[f"{prefix}.k_output.weight"][:] = np.eye(2)
    return values


def test_schema_contains_only_fixed_route_projection_weights() -> None:
    spec = architecture()

    assert len(spec.tensor_schema) == 8
    assert spec.tensor_schema["selector.block00_head00.q_hidden.weight"].shape == (2, 3)
    with pytest.raises(ValueError, match="route"):
        SelectorArchitecture(("bad.route",), 3, 2, 2, "float32")


def test_weight_only_selector_produces_deterministic_pair_logits(tmp_path) -> None:
    spec = architecture()
    path = tmp_path / "selector.safetensors"
    save_file(weights(spec), path)
    selector = load_selector(path, spec)
    q = np.array([[1, 0, 9], [0, 2, 9]], dtype=np.float32)
    k = np.array([[3, 0, 8], [0, 4, 8]], dtype=np.float32)

    first = selector.logits("block00_head00", q, k)
    second = selector.logits("block00_head00", q, k)

    assert first == pytest.approx(second)
    assert first == pytest.approx(np.array([[3, 0], [0, 8]]) / np.sqrt(2))


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
            "block00_head00",
            np.array([[np.nan, 0, 0]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
