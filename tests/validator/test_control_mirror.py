"""The recoverable asset copy must preserve both file and tensor identities."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.model import validate_model
from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec

SPEC = importlib.util.spec_from_file_location(
    "control_mirror",
    Path(__file__).resolve().parents[2] / "evidence/system-panel/launch/verify_control_assets.py",
)
assert SPEC and SPEC.loader
mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirror)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    document = valid_spec()
    track = parse_track_spec(document)
    (tmp_path / "controls").mkdir()
    (tmp_path / "predictor").mkdir()
    identities = {}
    checksums = []
    for i, name in enumerate(mirror.panel.EXPECTED_IDENTITIES):
        path = tmp_path / "controls" / f"{name}.safetensors"
        save_file(
            {
                key: np.full(value.shape, i, dtype=value.dtype)
                for key, value in track.model.tensor_schema.items()
            },
            path,
        )
        result = validate_model(
            path, schema=track.model.tensor_schema, max_bytes=track.model.max_bytes
        )
        identities[name] = {"file_sha256": result.file_sha256, "model_digest": result.model_digest}
        checksums.append(f"{result.file_sha256}  {path}\n")
    monkeypatch.setattr(mirror.panel, "EXPECTED_IDENTITIES", identities)
    predictor = tmp_path / "predictor/release_bf16.safetensors"
    predictor.write_bytes(b"predictor")
    predictor_hash = mirror.panel.sha256(predictor)
    checksums.append(f"{predictor_hash}  {predictor}\n")
    (tmp_path / "SHA256SUMS").write_text("".join(checksums))
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "calibration_track_document": document,
                "model_identities": identities,
                "predictor_bundle_sha256": predictor_hash,
            }
        )
    )
    return tmp_path, registration, mirror.panel.sha256(registration)


def test_seven_assets_verify_both_selector_identities(assets):
    result = mirror.verify(*assets)
    assert result["all_seven_assets_verified"]
    assert result["release_accepted"] is False
    assert len(result["controls"]) == 6


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "predictor", "selector", "model"])
def test_incomplete_or_corrupted_control_mirror_is_rejected(assets, mutation, monkeypatch):
    root, _, _ = assets
    sums = root / "SHA256SUMS"
    if mutation == "missing":
        sums.write_text("".join(sums.read_text().splitlines(keepends=True)[:-1]))
    elif mutation == "duplicate":
        sums.write_text(sums.read_text() * 2)
    elif mutation == "predictor":
        (root / "predictor/release_bf16.safetensors").write_bytes(b"corrupted")
    elif mutation == "selector":
        (root / "controls/baseline_self.safetensors").write_bytes(b"corrupted")
    else:
        monkeypatch.setitem(
            mirror.panel.EXPECTED_IDENTITIES,
            "baseline_self",
            {
                **mirror.panel.EXPECTED_IDENTITIES["baseline_self"],
                "model_digest": "0" * 64,
            },
        )
    with pytest.raises(ValueError):
        mirror.verify(*assets)
