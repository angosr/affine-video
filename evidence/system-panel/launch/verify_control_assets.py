"""Verify the off-GPU control mirror after its SHA256SUMS is complete."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from scoring.model import validate_model
from scoring.spec import parse_track_spec

_SPEC = importlib.util.spec_from_file_location(
    "panel_controls", Path(__file__).with_name("runner.py")
)
assert _SPEC and _SPEC.loader
panel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(panel)


def verify(root: Path, registration_path: Path, registration_sha256: str):
    if panel.sha256(registration_path) != registration_sha256:
        raise ValueError("control registration hash mismatch")
    registration = json.loads(registration_path.read_text())
    track = parse_track_spec(registration["calibration_track_document"])
    expected = {
        root / "controls" / f"{name}.safetensors": identity["file_sha256"]
        for name, identity in panel.EXPECTED_IDENTITIES.items()
    }
    predictor = root / "predictor/release_bf16.safetensors"
    expected[predictor] = registration["predictor_bundle_sha256"]
    checksum_path = root / "SHA256SUMS"
    declared = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = Path(name.removeprefix("*")).resolve()
        if path in declared:
            raise ValueError("duplicate mirrored asset in SHA256SUMS")
        declared[path] = digest
    if declared != {path.resolve(): digest for path, digest in expected.items()}:
        raise ValueError("SHA256SUMS differs from the seven registered assets")
    identities = {}
    for name, identity in panel.EXPECTED_IDENTITIES.items():
        path = root / "controls" / f"{name}.safetensors"
        actual = validate_model(
            path, schema=track.model.tensor_schema, max_bytes=track.model.max_bytes
        )
        values = {"file_sha256": actual.file_sha256, "model_digest": actual.model_digest}
        if values != identity:
            raise ValueError(f"mirrored selector identity mismatch: {name}")
        identities[name] = {**values, "size_bytes": actual.size_bytes}
    for name, identity in registration["model_identities"].items():
        control = "baseline_self" if name.startswith("pattern_") else name
        if identity != panel.EXPECTED_IDENTITIES[control]:
            raise ValueError(f"cost registration selector mismatch: {name}")
    if panel.sha256(predictor) != expected[predictor]:
        raise ValueError("mirrored predictor hash mismatch")
    return {
        "format": "veda-private-control-mirror-verification-v1",
        "scope": "asset_recovery_only_not_release_acceptance",
        "registration_sha256": registration_sha256,
        "verifier_sha256": panel.sha256(Path(__file__)),
        "selector_registry_sha256": panel.sha256(Path(panel.__file__)),
        "checksum_manifest_sha256": panel.sha256(checksum_path),
        "track_spec_digest_provenance": track.digest,
        "controls": identities,
        "predictor": {"file_sha256": expected[predictor], "size_bytes": predictor.stat().st_size},
        "all_seven_assets_verified": True,
        "release_accepted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.registration, args.registration_sha256)
    panel.write_once(args.output, result)
    print(json.dumps(result, sort_keys=True))
