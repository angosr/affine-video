"""Private, immutable stage evidence; public matches carry only its digest reference."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scoring.exact import (
    CaseDesign,
    _weighted_cell_mean,
    paired_score_gain_bound,
    score_selector_pair,
)
from validator.production.commitment import canonical_json


def stage_reference(name: str, payload: bytes) -> dict[str, Any]:
    return {"object": name, "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


def validate_stage_reference(value: Any, challenge_id: str | None = None,
                             stage: str | None = None) -> None:
    if (not isinstance(value, dict) or set(value) != {"object", "sha256", "size_bytes"}
            or not isinstance(value["object"], str)
            or not re.fullmatch(r"stages/[0-9a-f]{64}/(?:SCORING|CONFIRMING)\.json",
                                value["object"])
            or not isinstance(value["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or type(value["size_bytes"]) is not int or not 0 < value["size_bytes"] <= 2**20
            or (challenge_id is not None and stage is not None
                and value["object"] != f"stages/{challenge_id}/{stage}.json")):
        raise ValueError("stage evidence reference is invalid")


def _execution(device: str | None, runtime_identity: dict[str, Any] | None,
               started_at_ms: int) -> dict[str, Any]:
    release = None if runtime_identity is None else runtime_identity["release"]
    source = None
    if release is not None:
        from validator.production.environment import evaluation_source_digest

        source = evaluation_source_digest()
        if source != release["source_sha256"]:
            raise ValueError("scoring process source differs from its approved release")
        for field in ("dependency_lock", "source_bundle"):
            if hashlib.sha256(Path(runtime_identity[f"{field}_path"]).read_bytes()).hexdigest() != (
                runtime_identity[f"{field}_sha256"]
            ):
                raise ValueError("scoring process runtime artifact differs from its approval")
    functions = {"score_selector_pair": score_selector_pair,
                 "paired_score_gain_bound": paired_score_gain_bound}
    hardware = None
    torch_runtime = None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)"
                                               r"(?:,(?:[0-9]+|GPU-[0-9a-fA-F-]+))*", visible):
        visible = "unrecognized"
    driver = None
    try:
        match = re.search(r"Kernel Module\s+([0-9]+(?:\.[0-9]+)+)",
                          Path("/proc/driver/nvidia/version").read_text())
        driver = None if match is None else match[1]
    except OSError:
        pass  # Unknown, not a driver version copied from a different machine.
    if device is not None:
        import torch

        from scoring.torch_backend import TorchCaseScorer

        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("stage execution device must be authoritative CUDA")
        properties = torch.cuda.get_device_properties(resolved)
        hardware = {
            "name": properties.name, "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
        }
        torch_runtime = {
            "version": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        functions["TorchCaseScorer"] = TorchCaseScorer
    paths = {}
    for name, function in functions.items():
        path = Path(inspect.getfile(function)).resolve()
        paths[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "started_at_ms": started_at_ms, "finished_at_ms": time.time_ns() // 1_000_000,
        "clock": "local-wall-clock", "hostname": platform.node(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "runtime_identity": runtime_identity, "actual_source_sha256": source,
        "backend": "numpy-cpu" if device is None else "torch-cuda-fp32",
        "input_storage_dtype": "float16", "working_dtype": "float32",
        "dense_reference": "original-stored-dense", "logical_device": device,
        "hardware": hardware, "torch": torch_runtime, "code_paths": paths,
        "cuda_visible_devices": visible, "driver_version": driver,
    }


def pair_evidence(track: Any, metadata: dict[str, Any], paired: Any, *,
                  challenger_sha256: str, incumbent_sha256: str, device: str | None,
                  runtime_identity: dict[str, Any] | None,
                  started_at_ms: int) -> dict[str, Any]:
    """Retain calculated cells, never evaluate either model a second time."""

    design = [asdict(CaseDesign(**{key: row[key] for key in CaseDesign.__dataclass_fields__}))
              for row in metadata["cases"]]
    scoring = track.scoring
    result = {
        "format": "veda-stage-reproducibility-v1",
        "challenger_sha256": challenger_sha256, "incumbent_sha256": incumbent_sha256,
        "case_order": design,
        "challenger_cells": list(paired.challenger_cells),
        "incumbent_cells": list(paired.incumbent_cells),
        "stratum_weights": [list(row) for row in scoring.stratum_weight_map.items()],
        "step_window_weights": [list(row) for row in scoring.step_window_weight_map.items()],
        "bootstrap": {"confidence": scoring.match_confidence,
                      "resamples": scoring.bootstrap_resamples, "seed": scoring.bootstrap_seed},
        "budgets": [asdict(budget) for budget in scoring.budgets],
        "tail_mean_weight": scoring.tail_mean_weight,
        "execution": _execution(device, runtime_identity, started_at_ms),
    }
    validate_pair_evidence(result, paired.challenger_score, paired.incumbent_score,
                           paired.gain_lower_bound, track=track)
    return result


def validate_pair_evidence(value: Any, challenger_score: float, incumbent_score: float,
                           bound: float, *, track: Any = None) -> None:
    """Recompute only CPU statistics from recorded cells, including on cold restore.

    Stored execution facts are not replaced by the restoring machine's identity.
    No Q/K/V, weights, prompts, environment variables or credentials belong here.
    """

    if not isinstance(value, dict) or set(value) != {
        "format", "challenger_sha256", "incumbent_sha256", "case_order",
        "challenger_cells", "incumbent_cells", "stratum_weights", "step_window_weights",
        "bootstrap", "budgets", "tail_mean_weight", "execution",
    } or value["format"] != "veda-stage-reproducibility-v1":
        raise ValueError("stage reproducibility evidence is invalid")
    for key in ("challenger_sha256", "incumbent_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("stage model digest evidence is invalid")
    if (not isinstance(value["case_order"], list) or not value["case_order"]
            or len(value["case_order"]) > 100_000):
        raise ValueError("stage case order is invalid")
    design = []
    for row in value["case_order"]:
        if (not isinstance(row, dict) or set(row) != set(CaseDesign.__dataclass_fields__)
                or type(row["step_index"]) is not int or row["step_index"] < 0
                or any(not isinstance(row[key], str) or not row[key] for key in row
                       if key != "step_index")):
            raise ValueError("stage case design is invalid")
        design.append(CaseDesign(**row))
    for key in ("challenger_cells", "incumbent_cells"):
        cells = value[key]
        if (not isinstance(cells, list) or len(cells) != len(design)
                or any(type(cell) not in (float, int) or not math.isfinite(cell)
                       or not 0 <= cell <= 1 for cell in cells)):
            raise ValueError("stage cell scores are invalid")
    weights = []
    for key in ("stratum_weights", "step_window_weights"):
        rows = value[key]
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, list) or len(row) != 2
                       or not isinstance(row[0], str) or not row[0]
                       or type(row[1]) not in (float, int) or not math.isfinite(row[1])
                       or row[1] <= 0 for row in rows)
                or len(dict(rows)) != len(rows)):
            raise ValueError("stage aggregation weights are invalid")
        weights.append(dict(rows))
    bootstrap = value["bootstrap"]
    if (not isinstance(bootstrap, dict) or set(bootstrap) != {"confidence", "resamples", "seed"}
            or type(bootstrap["resamples"]) is not int or bootstrap["resamples"] <= 0
            or type(bootstrap["seed"]) is not int or bootstrap["seed"] < 0
            or type(bootstrap["confidence"]) not in (int, float)
            or not 0 < bootstrap["confidence"] < 1):
        raise ValueError("stage bootstrap parameters are invalid")
    if track is not None:
        scoring = track.scoring
        if (value["stratum_weights"] != [list(row) for row in scoring.stratum_weight_map.items()]
                or value["step_window_weights"] != [list(row) for row in
                                                   scoring.step_window_weight_map.items()]
                or bootstrap != {"confidence": scoring.match_confidence,
                                 "resamples": scoring.bootstrap_resamples,
                                 "seed": scoring.bootstrap_seed}
                or value["budgets"] != [asdict(budget) for budget in scoring.budgets]
                or value["tail_mean_weight"] != scoring.tail_mean_weight):
            raise ValueError("stage statistical policy differs from its track")
    actual = (
        _weighted_cell_mean(value["challenger_cells"], design, *weights),
        _weighted_cell_mean(value["incumbent_cells"], design, *weights),
        paired_score_gain_bound(value["challenger_cells"], value["incumbent_cells"], design,
                               stratum_weights=weights[0], step_window_weights=weights[1],
                               **bootstrap),
    )
    if actual != (challenger_score, incumbent_score, bound):
        raise ValueError("stage cell evidence does not reproduce its score and LCB")
    execution = value["execution"]
    if (not isinstance(execution, dict) or set(execution) != {
        "started_at_ms", "finished_at_ms", "clock", "hostname", "python_executable",
        "python_version", "runtime_identity", "actual_source_sha256", "backend",
        "input_storage_dtype", "working_dtype", "dense_reference", "logical_device",
        "hardware", "torch", "code_paths", "cuda_visible_devices", "driver_version",
    } or type(execution["started_at_ms"]) is not int
            or type(execution["finished_at_ms"]) is not int
            or not 0 < execution["started_at_ms"] <= execution["finished_at_ms"]
            or execution["clock"] != "local-wall-clock"
            or execution["input_storage_dtype"] != "float16"
            or execution["working_dtype"] != "float32"
            or execution["dense_reference"] != "original-stored-dense"):
        raise ValueError("stage execution evidence is invalid")
    if (not isinstance(execution["python_executable"], str)
            or not Path(execution["python_executable"]).is_absolute()
            or any(not isinstance(execution[key], str) or not execution[key]
                   for key in ("python_version", "hostname"))):
        raise ValueError("stage Python execution identity is invalid")
    paths = execution["code_paths"]
    names = {"score_selector_pair", "paired_score_gain_bound"}
    if execution["backend"] == "torch-cuda-fp32":
        names.add("TorchCaseScorer")
        hardware, runtime = execution["hardware"], execution["torch"]
        if (not isinstance(execution["logical_device"], str)
                or not re.fullmatch(r"cuda(?::[0-9]+)?", execution["logical_device"])
                or not isinstance(hardware, dict) or set(hardware) != {
                    "name", "uuid", "compute_capability", "total_memory",
                } or not isinstance(hardware["name"], str) or not hardware["name"]
                or not isinstance(hardware["uuid"], str) or not hardware["uuid"]
                or not isinstance(hardware["compute_capability"], list)
                or len(hardware["compute_capability"]) != 2
                or any(type(x) is not int or x < 0 for x in hardware["compute_capability"])
                or type(hardware["total_memory"]) is not int or hardware["total_memory"] <= 0
                or not isinstance(runtime, dict) or set(runtime) != {
                    "version", "cuda", "cudnn", "matmul_tf32", "cudnn_tf32",
                    "deterministic_algorithms", "cudnn_benchmark", "float32_matmul_precision",
                } or runtime["matmul_tf32"] is not False or runtime["cudnn_tf32"] is not False):
            raise ValueError("stage CUDA execution identity is invalid")
    elif (execution["backend"] != "numpy-cpu" or execution["logical_device"] is not None
          or execution["hardware"] is not None or execution["torch"] is not None):
        raise ValueError("stage backend identity is invalid")
    if (not isinstance(paths, dict) or set(paths) != names
            or any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
                   or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                   or not isinstance(row["sha256"], str)
                   or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in paths.values())):
        raise ValueError("stage callable path identity is invalid")
    identity = execution["runtime_identity"]
    if identity is not None and (not isinstance(identity, dict) or set(identity) != {
        "release", "dependency_lock_sha256", "source_bundle_sha256", "verified_at_ms",
        "dependency_lock_path", "source_bundle_path",
    } or not isinstance(identity["release"], dict)
            or identity["release"].get("source_sha256") != execution["actual_source_sha256"]):
        raise ValueError("stage release evidence differs from its execution")
    if identity is None:
        if execution["actual_source_sha256"] is not None:
            raise ValueError("unapproved execution cannot claim an approved source identity")
    else:
        release = identity["release"]
        required = {"track_spec_digest", "competition_manifest_sha256", "artifact_manifest_sha256",
                    "acceptance_sha256", "source_sha256", "environment_sha256", "image_id"}
        if (set(release) not in (required, required | {"data_spec_sha256", "data_contract_sha256"})
                or any(not isinstance(release[key], str)
                       or not re.fullmatch(r"[0-9a-f]{64}", release[key])
                       for key in release if key != "image_id")
                or not isinstance(release["image_id"], str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", release["image_id"])
                or type(identity["verified_at_ms"]) is not int
                or not 0 < identity["verified_at_ms"] <= execution["started_at_ms"]
                or any(not isinstance(identity[f"{key}_sha256"], str)
                       or not re.fullmatch(r"[0-9a-f]{64}", identity[f"{key}_sha256"])
                       or not isinstance(identity[f"{key}_path"], str)
                       or not Path(identity[f"{key}_path"]).is_absolute()
                       for key in ("source_bundle", "dependency_lock"))):
            raise ValueError("stage approved release artifacts are invalid")
        if track is not None and release["track_spec_digest"] != track.digest:
            raise ValueError("stage approved release belongs to another competition")
    # Ensure the retained object is canonical/finite JSON, not a live object graph.
    canonical_json(value)
