"""Fit and independently confirm registered costs from hash-bound native timings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from itertools import product
from math import ceil, isfinite
from pathlib import Path
from statistics import median

import numpy as np

_PATTERN_SPEC = importlib.util.spec_from_file_location(
    "cost_patterns", Path(__file__).with_name("cost_patterns.py")
)
assert _PATTERN_SPEC and _PATTERN_SPEC.loader
patterns = importlib.util.module_from_spec(_PATTERN_SPEC)
_PATTERN_SPEC.loader.exec_module(patterns)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rank(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("rank observations must be finite")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def samples(values, count):
    if len(values) != count or not all(isfinite(x) and x > 0 for x in values):
        raise ValueError("invalid latency samples")
    return median(values)


def key(row):
    return (
        row["seed"],
        row["geometry"],
        row["text_tokens"],
        row["model"],
        row["layer"],
        row["budget"],
    )


def cost_independent_contract(document):
    from scoring.spec import parse_track_spec, track_spec_dict

    normalized = track_spec_dict(parse_track_spec(document))
    # Runtime/environment/assets are independently pinned in registration.
    # Release packaging changes the full track digest without changing costs.
    normalized.pop("artifact_manifest_sha256")
    for budget in normalized["scoring"]["budgets"]:
        budget.pop("calibrated_cost")
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def derive_track(document, costs):
    from scoring.spec import parse_track_spec, track_spec_dict

    normalized = track_spec_dict(parse_track_spec(document))
    if set(costs) != {str(b["target_blocks"]) for b in normalized["scoring"]["budgets"]}:
        raise ValueError("cost fit does not cover exact track budgets")
    for budget in normalized["scoring"]["budgets"]:
        budget["calibrated_cost"] = costs[str(budget["target_blocks"])]
    derived = parse_track_spec(normalized)
    return track_spec_dict(derived), derived.digest


def verify_phase_track(registration, document, phase, fit):
    from scoring.spec import parse_track_spec

    if phase == "calibration":
        if fit is not None:
            raise ValueError("calibration rejects a frozen fit")
        expected = registration["calibration_track_document"]
    elif phase == "confirmation":
        if fit is None:
            raise ValueError("confirmation requires the frozen calibration fit")
        expected, expected_digest = derive_track(
            registration["calibration_track_document"], fit["costs"]
        )
        if (
            fit["provisional_cost_track_spec_digest"] != expected_digest
            or fit["provisional_cost_track_document"] != expected
            or fit["phase"] != "calibration"
        ):
            raise ValueError("fit changed the deterministic cost derivation")
    else:
        raise ValueError("unknown cost phase")
    actual = parse_track_spec(document)
    expected_track = parse_track_spec(expected)
    if (
        cost_independent_contract(document) != registration["cost_independent_track_sha256"]
        or actual.scoring.budgets != expected_track.scoring.budgets
    ):
        raise ValueError("measurement does not use the registered phase track")
    # Full digest is provenance only: release packaging may bind new evidence.
    return actual.digest


def load_measurements(roots, registration_path, phase, fit_path=None):
    registration = json.loads(registration_path.read_text())
    if registration["format"] != "veda-h200-cost-registration-v2":
        raise ValueError("unknown cost registration")
    if set(registration["seeds"]["calibration"]) & set(registration["seeds"]["confirmation"]):
        raise ValueError("calibration and confirmation seeds overlap")
    fit = None if fit_path is None else json.loads(fit_path.read_text())
    document = (
        registration["calibration_track_document"]
        if phase == "calibration"
        else None
        if fit is None
        else fit["provisional_cost_track_document"]
    )
    verify_phase_track(registration, document, phase, fit)
    rows, protocols, references = [], [], []
    for root in roots:
        protocol_path = root / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        expected_track = verify_phase_track(registration, protocol["track_document"], phase, fit)
        protocol_hash = sha256(protocol_path)
        summary = json.loads((root / "summary.json").read_text())
        if (
            protocol["format"] != "veda-h200-cost-measurement-v2"
            or protocol["registration_sha256"] != sha256(registration_path)
            or protocol["phase"] != phase
            or protocol["track_spec_digest"] != expected_track
            or protocol["cost_independent_track_sha256"]
            != registration["cost_independent_track_sha256"]
            or protocol["seed"] not in registration["seeds"][phase]
            or summary["protocol_sha256"] != protocol_hash
            or protocol["layers"] != registration["layers"]
            or protocol["text_token_counts"] != registration["text_token_counts"]
            or protocol["budgets"] != registration["budgets"]
            or protocol["model_identities"] != registration["model_identities"]
            or protocol["predictor_bundle_sha256"] != registration["predictor_bundle_sha256"]
            or sorted(protocol["model_identities"]) != sorted(registration["models"])
            or (protocol["warmup"], protocol["repeats"], protocol["profile_repeats"])
            != (registration["warmup"], registration["repeats"], registration["profile_repeats"])
            or protocol["source_bundle_sha256"] != registration["source_bundle_sha256"]
            or protocol["dependency_lock_sha256"] != registration["dependency_lock_sha256"]
            or protocol["pattern_script_sha256"] != registration["pattern_script_sha256"]
            or protocol["script_sha256"] != registration["measurement_script_sha256"]
            or protocol["fit_sha256"] != (None if fit_path is None else sha256(fit_path))
        ):
            raise ValueError("measurement protocol differs from registered phase")
        expected_rows = (
            len(protocol["geometries"])
            * len(protocol["text_token_counts"])
            * len(protocol["model_identities"])
            * len(protocol["layers"])
            * len(protocol["budgets"])
        )
        if (
            summary.get("format") != "veda-h200-cost-measurement-summary-v2"
            or type(summary.get("row_count")) is not int
            or summary["row_count"] != expected_rows
            or summary.get("expected_rows") != expected_rows
            or summary.get("all_dimensions_present") is not True
            or not isinstance(summary.get("raw_sha256"), dict)
            or summary["raw_sha256"].get("protocol.json") != protocol_hash
        ):
            raise ValueError("invalid or incomplete measurement summary")
        before_rows = len(rows)
        for name, digest in summary["raw_sha256"].items():
            if Path(name).name != name or sha256(root / name) != digest:
                raise ValueError("raw timing hash mismatch")
            if name == "protocol.json":
                continue
            row = json.loads((root / name).read_text())
            if row["protocol_sha256"] != protocol_hash:
                raise ValueError("timing row belongs to another protocol")
            if name.startswith("dense-"):
                continue
            dense_name = row["dense_record"]
            if (
                Path(dense_name).name != dense_name
                or summary["raw_sha256"].get(dense_name) != row["dense_sha256"]
            ):
                raise ValueError("dense denominator is not bound")
            dense = json.loads((root / dense_name).read_text())
            if (dense["geometry"], dense["text_tokens"], dense["protocol_sha256"]) != (
                row["geometry"],
                row["text_tokens"],
                protocol_hash,
            ):
                raise ValueError("dense denominator uses different inputs")
            if row["model"].startswith("pattern_"):
                if not row["pattern_proofs"]:
                    raise ValueError("missing legal pattern allocation proofs")
                for proof in row["pattern_proofs"]:
                    if proof["pattern"] != row["model"].removeprefix("pattern_"):
                        raise ValueError("wrong pattern identity")
                    patterns.verify_proof(proof)
            measured = samples(row["end_to_end_samples_ms"], protocol["repeats"])
            dense_ms = samples(dense["end_to_end_samples_ms"], protocol["repeats"])
            samples(row["end_to_end_cuda_samples_ms"], protocol["repeats"])
            samples(dense["end_to_end_cuda_samples_ms"], protocol["repeats"])
            components = row["component_samples_ms"]
            if len(components) != protocol["profile_repeats"]:
                raise ValueError("missing component profiles")
            sparse = samples(
                [
                    c["sparse_attention_including_packing"] - c["kernel_mask_packing"]
                    for c in components
                ],
                protocol["profile_repeats"],
            )
            rows.append(
                {
                    **row,
                    "seed": protocol["seed"],
                    "measured_ms": measured,
                    "dense_ms": dense_ms,
                    "sparse_ms": sparse,
                    "overhead_fraction": (measured - sparse) / dense_ms,
                }
            )
        if len(rows) - before_rows != summary["row_count"]:
            raise ValueError("declared row count differs from raw measurements")
        protocols.append(protocol)
        references.append(
            {"protocol_sha256": protocol_hash, "summary_sha256": sha256(root / "summary.json")}
        )
    expected = set(
        product(
            registration["seeds"][phase],
            registration["geometries"],
            registration["text_token_counts"],
            registration["models"],
            registration["layers"],
            registration["budgets"],
        )
    )
    if {key(r) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("measurements omit or duplicate a required dimension")
    for field in (
        "source_sha256",
        "environment_sha256",
        "predictor_bundle_sha256",
        "model_identities",
        "collect_bytes",
    ):
        if any(p[field] != protocols[0][field] for p in protocols):
            raise ValueError(f"mixed runtime or input identity: {field}")
    return registration, rows, protocols[0], references


def fit_costs(rows, budgets, first_cost):
    """Median matched latency ratios; no use of confirmation observations."""
    matched = defaultdict(dict)
    for row in rows:
        matched[key(row)[:-1]][row["budget"]] = row["measured_ms"]
    costs = {
        str(b): first_cost * median(r[b] / r[budgets[0]] for r in matched.values()) for b in budgets
    }
    if any(costs[str(b)] >= costs[str(c)] for b, c in zip(budgets, budgets[1:], strict=False)):
        raise ValueError("fit costs must increase with budget")
    contexts = defaultdict(list)
    for row in rows:
        contexts[f"{row['geometry']}/{row['text_tokens']}"].append(
            row["measured_ms"] / costs[str(row["budget"])]
        )
    return costs, {name: median(values) for name, values in contexts.items()}


def evaluate(rows, costs, scales, thresholds):
    contexts = defaultdict(list)
    for row in rows:
        context = f"{row['geometry']}/{row['text_tokens']}"
        predicted = costs[str(row["budget"])] * scales[context]
        measurement = {
            **row,
            "predicted_ms": predicted,
            "relative_error": abs(predicted - row["measured_ms"]) / row["measured_ms"],
        }
        contexts[context].append(measurement)
        contexts[f"{context}/{row['model']}"].append(measurement)
    gates = {}
    for context, measurements in contexts.items():
        errors = sorted(r["relative_error"] for r in measurements)
        # Evaluate each matched budget curve. Avoid inventing rank distinctions
        # between repeated observations having identical predicted cost.
        curves = defaultdict(list)
        for row in measurements:
            curves[key(row)[:-1]].append(row)
        correlations = []
        for curve in curves.values():
            corr = np.corrcoef(
                rank([r["predicted_ms"] for r in curve]), rank([r["measured_ms"] for r in curve])
            )[0, 1]
            correlations.append(float(corr) if np.isfinite(corr) else -1.0)
        value = {
            "row_count": len(measurements),
            "matched_curve_count": len(curves),
            "minimum_matched_curve_spearman": min(correlations),
            "median_relative_error": median(errors),
            "p95_relative_error": errors[ceil(0.95 * len(errors)) - 1],
            "minimum_overhead_fraction": min(r["overhead_fraction"] for r in measurements),
            "maximum_overhead_fraction": max(r["overhead_fraction"] for r in measurements),
        }
        by_budget = defaultdict(list)
        for measurement in measurements:
            by_budget[str(measurement["budget"])].append(measurement["relative_error"])
        value["budget_error_gates"] = {
            budget: {
                "median_relative_error": median(values),
                "p95_relative_error": sorted(values)[ceil(0.95 * len(values)) - 1],
            }
            for budget, values in by_budget.items()
        }
        budget_pass = all(
            gate["median_relative_error"] <= thresholds["median_relative_error_max"]
            and gate["p95_relative_error"] <= thresholds["p95_relative_error_max"]
            for gate in value["budget_error_gates"].values()
        )
        value["e10_pass"] = (
            budget_pass
            and value["minimum_matched_curve_spearman"] >= thresholds["spearman_min"]
            and value["median_relative_error"] <= thresholds["median_relative_error_max"]
            and value["p95_relative_error"] <= thresholds["p95_relative_error_max"]
        )
        value["e11_pass"] = (
            value["minimum_overhead_fraction"] >= 0 and value["maximum_overhead_fraction"] <= 0.10
        )
        gates[context] = value
    return gates


def analyze(roots, registration_path, phase, fit_path=None):
    registration, rows, protocol, references = load_measurements(
        roots, registration_path, phase, fit_path
    )
    if (
        sha256(Path(__file__).with_name("cost_patterns.py"))
        != registration["pattern_script_sha256"]
    ):
        raise ValueError("pattern verifier differs from preregistered implementation")
    if sha256(Path(__file__)) != registration["analysis_script_sha256"]:
        raise ValueError("analysis differs from preregistered method")
    if phase == "calibration":
        costs, scales = fit_costs(rows, registration["budgets"], registration["first_cost"])
    else:
        if fit_path is None:
            raise ValueError("confirmation cannot refit costs")
        frozen = json.loads(fit_path.read_text())
        if (
            frozen["registration_sha256"] != sha256(registration_path)
            or frozen["phase"] != "calibration"
            or frozen["source_sha256"] != protocol["source_sha256"]
            or frozen["environment_sha256"] != protocol["environment_sha256"]
            or frozen["model_identities"] != protocol["model_identities"]
            or frozen["predictor_bundle_sha256"] != protocol["predictor_bundle_sha256"]
        ):
            raise ValueError("confirmation cannot reuse another fit's contract")
        costs, scales = frozen["costs"], frozen["scales"]
    derived, derived_digest = derive_track(registration["calibration_track_document"], costs)
    from scoring.spec import parse_track_spec
    from validator.system_matches import _evidence_contract_digest

    confirmed_contract = _evidence_contract_digest(parse_track_spec(derived))
    gates = evaluate(rows, costs, scales, registration["e10_thresholds"])
    return {
        "format": "veda-h200-cost-fit-v2",
        "phase": phase,
        "provisional_cost_track_document": derived,
        "provisional_cost_track_spec_digest": derived_digest,
        "cost_independent_track_sha256": registration["cost_independent_track_sha256"],
        "release_accepted": False,
        "scope": "native_attention_costs_not_complete_release_acceptance",
        "registration_sha256": sha256(registration_path),
        "fit_sha256": None if fit_path is None else sha256(fit_path),
        "source_sha256": protocol["source_sha256"],
        "environment_sha256": protocol["environment_sha256"],
        "model_identities": protocol["model_identities"],
        "predictor_bundle_sha256": protocol["predictor_bundle_sha256"],
        "costs": costs,
        "confirmed_evidence_contract_sha256": confirmed_contract,
        "scales": scales,
        "gates": gates,
        "raw_evidence": references,
        "e10_confirmation_pass": phase == "confirmation"
        and all(g["e10_pass"] for g in gates.values()),
        "e11_confirmation_pass": phase == "confirmation"
        and all(g["e11_pass"] for g in gates.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.roots, args.registration, args.phase, args.fit)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))
