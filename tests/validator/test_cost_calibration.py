"""Cost acceptance must use a frozen fit, complete dimensions and dense denominator."""

import importlib.util
import json
from itertools import product
from pathlib import Path

import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "cost_analysis", ROOT / "evidence/system-panel/launch/cost_analysis.py"
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(tmp_path, phase="calibration", fit=None, times=(10, 13, 19)):
    registration = tmp_path / "registration.json"
    if not registration.exists():
        track = valid_spec()
        for budget, target in zip(track["scoring"]["budgets"], (16, 32, 64), strict=True):
            budget["target_blocks"] = target
        write(
            registration,
            {
                "format": "veda-h200-cost-registration-v2",
                "calibration_track_document": track,
                "cost_independent_track_sha256": analysis.cost_independent_contract(track),
                "seeds": {"calibration": [1], "confirmation": [2]},
                "layers": [0, 1],
                "models": ["baseline_self", "shuffled_heads"],
                "model_identities": {
                    name: {"file_sha256": name, "model_digest": name}
                    for name in ("baseline_self", "shuffled_heads")
                },
                "predictor_bundle_sha256": "plans",
                "geometries": ["a", "b"],
                "text_token_counts": [77, 256],
                "budgets": [16, 32, 64],
                "warmup": 10,
                "repeats": 50,
                "profile_repeats": 10,
                "first_cost": 0.04,
                "measurement_script_sha256": "measurement",
                "pattern_script_sha256": analysis.sha256(Path(analysis.patterns.__file__)),
                "source_bundle_sha256": "source-zip",
                "dependency_lock_sha256": "lock",
                "analysis_script_sha256": analysis.sha256(Path(analysis.__file__)),
                "e10_thresholds": {
                    "spearman_min": 0.95,
                    "median_relative_error_max": 0.05,
                    "p95_relative_error_max": 0.10,
                },
            },
        )
    registered = json.loads(registration.read_text())
    root = tmp_path / phase
    root.mkdir()
    protocol = {
        **registered,
        "format": "veda-h200-cost-measurement-v2",
        "phase": phase,
        "registration_sha256": analysis.sha256(registration),
        "seed": registered["seeds"][phase][0],
        "source_sha256": "source",
        "environment_sha256": "environment",
        "predictor_bundle_sha256": "plans",
        "model_identities": {
            name: {"file_sha256": name, "model_digest": name} for name in registered["models"]
        },
        "collect_bytes": 256 << 20,
        "track_spec_digest": (
            parse_track_spec(registered["calibration_track_document"]).digest
            if fit is None
            else json.loads(fit.read_text())["provisional_cost_track_spec_digest"]
        ),
        "track_document": (
            registered["calibration_track_document"]
            if fit is None
            else json.loads(fit.read_text())["provisional_cost_track_document"]
        ),
        "script_sha256": "measurement",
        "fit_sha256": None if fit is None else analysis.sha256(fit),
    }
    write(root / "protocol.json", protocol)
    protocol_hash = analysis.sha256(root / "protocol.json")
    for geometry, text in product(registered["geometries"], registered["text_token_counts"]):
        dense_name = f"dense-{geometry}-{text}.json"
        write(
            root / dense_name,
            {
                "geometry": geometry,
                "text_tokens": text,
                "protocol_sha256": protocol_hash,
                "end_to_end_samples_ms": [60] * 50,
                "end_to_end_cuda_samples_ms": [59] * 50,
            },
        )
        for model, layer, (budget, measured) in product(
            registered["models"],
            registered["layers"],
            zip(registered["budgets"], times, strict=True),
        ):
            write(
                root / f"{geometry}-{text}-{model}-{layer}-{budget}.json",
                {
                    "geometry": geometry,
                    "text_tokens": text,
                    "model": model,
                    "layer": layer,
                    "budget": budget,
                    "protocol_sha256": protocol_hash,
                    "dense_record": dense_name,
                    "dense_sha256": analysis.sha256(root / dense_name),
                    "end_to_end_samples_ms": [measured] * 50,
                    "end_to_end_cuda_samples_ms": [measured - 0.1] * 50,
                    "component_samples_ms": [
                        {
                            "sparse_attention_including_packing": measured - 2,
                            "kernel_mask_packing": 1,
                        }
                    ]
                    * 10,
                },
            )
    rehash(root)
    return root, registration


def rehash(root):
    protocol = json.loads((root / "protocol.json").read_text())
    expected = (
        len(protocol["geometries"])
        * len(protocol["text_token_counts"])
        * len(protocol["model_identities"])
        * len(protocol["layers"])
        * len(protocol["budgets"])
    )
    write(
        root / "summary.json",
        {
            "format": "veda-h200-cost-measurement-summary-v2",
            "row_count": expected,
            "expected_rows": expected,
            "all_dimensions_present": True,
            "protocol_sha256": analysis.sha256(root / "protocol.json"),
            "raw_sha256": {
                p.name: analysis.sha256(p) for p in root.glob("*.json") if p.name != "summary.json"
            },
        },
    )


def test_frozen_cost_fit_and_dense_overhead_denominator(tmp_path):
    root, registration = fixture(tmp_path)
    fitted = analysis.analyze([root], registration, "calibration")
    assert fitted["e10_confirmation_pass"] is False
    assert fitted["e11_confirmation_pass"] is False
    assert fitted["costs"] == pytest.approx({"16": 0.04, "32": 0.052, "64": 0.076})
    fit = tmp_path / "fit.json"
    write(fit, fitted)
    confirmed, _ = fixture(tmp_path, "confirmation", fit)
    result = analysis.analyze([confirmed], registration, "confirmation", fit)
    assert result["e10_confirmation_pass"] is True
    assert result["e11_confirmation_pass"] is True
    assert result["release_accepted"] is False
    assert all(
        g["maximum_overhead_fraction"] == pytest.approx(3 / 60) for g in result["gates"].values()
    )


def test_confirmation_cannot_refit_bad_cost_shape(tmp_path):
    root, registration = fixture(tmp_path)
    fit = tmp_path / "fit.json"
    write(fit, analysis.analyze([root], registration, "calibration"))
    confirmed, _ = fixture(tmp_path, "confirmation", fit, times=(10, 20, 40))
    result = analysis.analyze([confirmed], registration, "confirmation", fit)
    assert result["e10_confirmation_pass"] is False
    assert result["costs"] == json.loads(fit.read_text())["costs"]
    assert all(not g["e10_pass"] for g in result["gates"].values())


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "tamper", "dense_binding", "dense_context", "runtime", "overlap"],
)
def test_incomplete_or_mixed_cost_evidence_is_rejected(tmp_path, failure):
    root, registration = fixture(tmp_path)
    row_path = next(root.glob("a-77-*.json"))
    row = json.loads(row_path.read_text())
    if failure == "missing":
        row_path.unlink()
    elif failure == "duplicate":
        write(root / "duplicate.json", row)
    elif failure == "tamper":
        row_path.write_text("{}")
    elif failure == "dense_binding":
        row["dense_sha256"] = "wrong"
        write(row_path, row)
    elif failure == "dense_context":
        row["dense_record"] = "dense-b-77.json"
        row["dense_sha256"] = analysis.sha256(root / row["dense_record"])
        write(row_path, row)
    elif failure == "runtime":
        document = json.loads((root / "protocol.json").read_text())
        document["seed"] = 2
        write(root / "protocol.json", document)
    else:
        document = json.loads(registration.read_text())
        document["seeds"]["confirmation"] = [1]
        write(registration, document)
    if failure != "tamper":
        rehash(root)
    with pytest.raises(ValueError):
        analysis.analyze([root], registration, "calibration")


def test_one_failed_geometry_cannot_be_averaged_away(tmp_path):
    root, registration = fixture(tmp_path)
    fit = tmp_path / "fit.json"
    write(fit, analysis.analyze([root], registration, "calibration"))
    confirmed, _ = fixture(tmp_path, "confirmation", fit)
    for path in confirmed.glob("b-*.json"):
        row = json.loads(path.read_text())
        row["end_to_end_samples_ms"] = [2 * x for x in row["end_to_end_samples_ms"]]
        write(path, row)
    rehash(confirmed)
    result = analysis.analyze([confirmed], registration, "confirmation", fit)
    assert result["gates"]["a/77"]["e10_pass"] is True
    assert result["gates"]["b/77"]["e10_pass"] is False
    assert result["e10_confirmation_pass"] is False
    assert result["e11_confirmation_pass"] is False


@pytest.mark.parametrize("pattern", analysis.patterns.PATTERNS)
def test_legal_pattern_proof_recomputes_and_rejects_mutation(pattern):
    import hashlib

    import numpy as np

    counts = np.array([128] * 35 + [0, 3, 17, 0, 128])
    rows = np.flatnonzero(counts)
    logits = analysis.patterns.scores(pattern, counts, rows, 16, 2)
    digest = hashlib.sha256(logits.tobytes()).hexdigest()
    logits[:, :, counts == 0] = -np.inf
    logits[:, np.arange(len(rows)), rows] = np.inf
    index = np.argsort(-logits, axis=-1, kind="stable")[..., :17].astype(np.int64)
    keep = np.broadcast_to(np.arange(17) < 16, index.shape).copy()
    proof = analysis.patterns.selection_proof(index, keep, counts, rows, 16)
    proof.update(
        pattern=pattern,
        logits_sha256=digest,
        rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
    )
    analysis.patterns.verify_proof(proof)
    proof["index_sha256"] = "wrong"
    with pytest.raises(ValueError, match="deterministic legal allocation"):
        analysis.patterns.verify_proof(proof)


def test_patterns_exercise_distinct_locality_and_partial_tiles():
    import numpy as np

    counts = np.array([128] * 128 + [4, 8, 0])
    top = {
        name: np.argsort(-analysis.patterns.scores(name, counts, [65], 16, 1)[0, 0])[:16]
        for name in analysis.patterns.PATTERNS
    }
    assert top["clustered"].max() - top["clustered"].min() == 15
    assert top["spread"].max() - top["spread"].min() > 110
    assert set(top["boundary_padding"][:2]) == {128, 129}
    assert top["repeated_keys"].tolist() == list(range(16))
    assert all(130 not in indices for indices in top.values())


@pytest.mark.parametrize(
    "mutation",
    [
        {"format": "unrecognized"},
        {"row_count": 1},
        {"row_count": True},
        {"expected_rows": 1},
        {"all_dimensions_present": False},
    ],
)
def test_summary_schema_and_completeness_fail_closed(tmp_path, mutation):
    root, registration = fixture(tmp_path)
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(mutation)
    write(path, summary)
    with pytest.raises(ValueError, match="measurement summary"):
        analysis.analyze([root], registration, "calibration")


def test_cost_transition_requires_exact_derived_track_and_phase(tmp_path):
    root, registration_path = fixture(tmp_path)
    registration = json.loads(registration_path.read_text())
    fitted = analysis.analyze([root], registration_path, "calibration")
    original = registration["calibration_track_document"]
    derived = fitted["provisional_cost_track_document"]
    assert analysis.cost_independent_contract(original) == analysis.cost_independent_contract(
        derived
    )
    analysis.verify_phase_track(registration, derived, "confirmation", fitted)
    with pytest.raises(ValueError, match="registered phase track"):
        analysis.verify_phase_track(registration, original, "confirmation", fitted)
    with pytest.raises(ValueError, match="calibration rejects"):
        analysis.verify_phase_track(registration, original, "calibration", fitted)
    with pytest.raises(ValueError, match="requires the frozen"):
        analysis.verify_phase_track(registration, derived, "confirmation", None)
    altered = json.loads(json.dumps(derived))
    altered["track_id"] = "different-track"
    with pytest.raises(ValueError, match="registered phase track"):
        analysis.verify_phase_track(registration, altered, "confirmation", fitted)
    altered = json.loads(json.dumps(derived))
    altered["artifact_manifest_sha256"] = "22" * 32
    assert analysis.cost_independent_contract(altered) == analysis.cost_independent_contract(
        derived
    )
    actual_digest = analysis.verify_phase_track(registration, altered, "confirmation", fitted)
    assert actual_digest != fitted["provisional_cost_track_spec_digest"]
    altered["scoring"]["budgets"][1]["calibrated_cost"] *= 1.01
    with pytest.raises(ValueError, match="registered phase track"):
        analysis.verify_phase_track(registration, altered, "confirmation", fitted)


def test_control_host_raw_evidence_is_immutable_and_hash_checked(tmp_path):
    import hashlib

    spec = importlib.util.spec_from_file_location(
        "cost_collector", ROOT / "evidence/system-panel/launch/collect_cost_evidence.py"
    )
    assert spec and spec.loader
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    payload = {"samples": [1, 2, 3]}
    event = {
        "format": "veda-cost-raw-event-v1",
        "filename": "raw.json",
        "payload": payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    assert collector.persist(tmp_path, event) is True
    assert collector.persist(tmp_path, event) is False
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.persist(tmp_path, {**event, "sha256": "wrong"})
    with pytest.raises(ValueError, match="unsafe"):
        collector.persist(tmp_path, {**event, "filename": "../raw.json"})
    (tmp_path / "raw.json").write_bytes(b"old data")
    with pytest.raises(ValueError, match="conflict"):
        collector.persist(tmp_path, event)
    assert (tmp_path / "raw.json").read_bytes() == b"old data"
