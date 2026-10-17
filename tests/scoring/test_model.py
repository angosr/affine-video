import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.model import ModelValidationError, TensorSpec, validate_model

SCHEMA = {
    "adapter.bias": TensorSpec((2,), "float32"),
    "selector.weight": TensorSpec((2, 3), "float32"),
}


def write_model(path, *, metadata=None, non_finite: bool = False) -> None:
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    if non_finite:
        weight[0, 0] = np.nan
    save_file(
        {
            "selector.weight": weight,
            "adapter.bias": np.array([1, 2], dtype=np.float32),
        },
        path,
        metadata=metadata,
    )


def repack_header(source, target) -> None:
    blob = source.read_bytes()
    header_size = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8 : 8 + header_size])
    reversed_header = dict(reversed(header.items()))
    encoded = json.dumps(reversed_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    target.write_bytes(len(encoded).to_bytes(8, "little") + encoded + blob[8 + header_size :])


def test_validates_schema_and_canonicalizes_repacked_weights(tmp_path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    write_model(first)
    repack_header(first, second)

    first_result = validate_model(first, schema=SCHEMA, max_bytes=10_000)
    second_result = validate_model(second, schema=SCHEMA, max_bytes=10_000)

    assert first_result.file_sha256 != second_result.file_sha256
    assert first_result.model_digest == second_result.model_digest
    assert first_result.tensor_count == 2


def test_model_digest_normalizes_equivalent_signed_zero(tmp_path) -> None:
    positive = tmp_path / "positive.safetensors"
    negative = tmp_path / "negative.safetensors"
    schema = {"weight": TensorSpec((2,), "float32")}
    save_file({"weight": np.array([0.0, 1.0], dtype=np.float32)}, positive)
    save_file({"weight": np.array([-0.0, 1.0], dtype=np.float32)}, negative)

    positive_result = validate_model(positive, schema=schema, max_bytes=1_000)
    negative_result = validate_model(negative, schema=schema, max_bytes=1_000)

    assert positive_result.file_sha256 != negative_result.file_sha256
    assert positive_result.model_digest == negative_result.model_digest


def test_rejects_custom_safetensors_metadata(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    write_model(path, metadata={"configuration": "not allowed"})

    with pytest.raises(ModelValidationError) as caught:
        validate_model(path, schema=SCHEMA, max_bytes=10_000)

    assert caught.value.code == "MODEL_METADATA_INVALID"


def test_requires_exact_submitting_hotkey_metadata(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    write_model(path, metadata={"hotkey": "victim"})

    accepted = validate_model(
        path,
        schema=SCHEMA,
        max_bytes=10_000,
        expected_hotkey="victim",
    )
    assert accepted.tensor_count == 2

    with pytest.raises(ModelValidationError) as caught:
        validate_model(
            path,
            schema=SCHEMA,
            max_bytes=10_000,
            expected_hotkey="attacker",
        )
    assert caught.value.code == "MODEL_OWNER_MISMATCH"


def test_resigning_copied_weights_does_not_change_duplicate_fingerprint(tmp_path) -> None:
    victim = tmp_path / "victim.safetensors"
    attacker = tmp_path / "attacker.safetensors"
    write_model(victim, metadata={"hotkey": "victim"})
    write_model(attacker, metadata={"hotkey": "attacker"})

    victim_result = validate_model(
        victim,
        schema=SCHEMA,
        max_bytes=10_000,
        expected_hotkey="victim",
    )
    attacker_result = validate_model(
        attacker,
        schema=SCHEMA,
        max_bytes=10_000,
        expected_hotkey="attacker",
    )

    assert victim_result.file_sha256 != attacker_result.file_sha256
    assert victim_result.model_digest == attacker_result.model_digest


def test_rejects_non_finite_weights(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    write_model(path, non_finite=True)

    with pytest.raises(ModelValidationError) as caught:
        validate_model(path, schema=SCHEMA, max_bytes=10_000)

    assert caught.value.code == "MODEL_NON_FINITE"


def test_rejects_schema_mismatch_and_oversize(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    write_model(path)

    with pytest.raises(ModelValidationError) as mismatch:
        validate_model(
            path,
            schema={"wrong": TensorSpec((1,), "float32")},
            max_bytes=10_000,
        )
    assert mismatch.value.code == "MODEL_SCHEMA_INVALID"

    with pytest.raises(ModelValidationError) as oversize:
        validate_model(path, schema=SCHEMA, max_bytes=1)
    assert oversize.value.code == "MODEL_TOO_LARGE"
