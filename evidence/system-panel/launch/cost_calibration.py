"""Replayable H200 E10/E11 measurements using the tournament's production selector.

Synthetic QKV avoids consuming hidden acceptance families. This is a kernel and
selection-cost experiment, not complete final-release acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack
from functools import partial
from math import ceil
from pathlib import Path
from statistics import median
from unittest.mock import patch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def control_identity(path, *, schema, max_bytes):
    """Validate fixed benchmark controls, not a miner's chain ownership."""

    from safetensors import safe_open

    from scoring.model import validate_model

    with safe_open(path, framework="numpy") as tensors:
        owner = (tensors.metadata() or {}).get("hotkey")
    model = validate_model(path, schema=schema, max_bytes=max_bytes, expected_hotkey=owner)
    return {"file_sha256": model.file_sha256, "model_digest": model.model_digest}


def write_once(path, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable calibration evidence conflicts: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "format": "veda-cost-raw-event-v1",
                "filename": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "payload": value,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def summarize(samples):
    if not samples or any(value <= 0 for value in samples):
        raise ValueError("latency samples must be positive")
    ordered = sorted(samples)
    return {"median_ms": median(samples), "p95_ms": ordered[ceil(0.95 * len(ordered)) - 1]}


def benchmark(torch, operation, warmup, repeats, post_check=None):
    for _ in range(warmup):
        operation()
        if post_check is not None:
            post_check()
    torch.cuda.synchronize()
    values = []
    cuda_values = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter_ns()
        begin.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        cuda_values.append(begin.elapsed_time(end))
        values.append((time.perf_counter_ns() - started) / 1e6)
        if post_check is not None:
            post_check()
    return values, cuda_values


def profile_components(torch, attention, runtime, fa4, operation, repeats, post_check=None):
    """Bracket production calls without adding synchronizations inside attention."""

    events = defaultdict(list)

    def wrapper(name, original):
        def measured(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            finish.record()
            events[name].append((start, finish))
            return result

        return measured

    targets = (
        (runtime, "gather_and_pool", "gather_and_trippool"),
        (attention, "_gather", "value_gather"),
        (runtime, "selector_logits", "residual_projection_and_logits"),
        (runtime, "runtime_block_mask", "stable_topk_and_anchors"),
        (fa4, "index_lists", "kernel_mask_packing"),
        (fa4, "block_sparse_attention", "sparse_attention_including_packing"),
        (attention, "_scatter_", "output_scatter"),
    )
    samples = []
    with ExitStack() as stack:
        for module, attribute, name in targets:
            stack.enter_context(
                patch.object(module, attribute, wrapper(name, getattr(module, attribute)))
            )
        for _ in range(repeats):
            events.clear()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            operation()
            torch.cuda.synchronize()
            row = {name: sum(a.elapsed_time(b) for a, b in pairs) for name, pairs in events.items()}
            row["instrumented_end_to_end_ms"] = (time.perf_counter_ns() - started) / 1e6
            row["sparse_attention_excluding_packing"] = (
                row["sparse_attention_including_packing"] - row["kernel_mask_packing"]
            )
            samples.append(row)
            if post_check is not None:
                post_check()
    return samples


def run(arguments):
    import torch
    from cost_analysis import cost_independent_contract, verify_phase_track
    from cost_patterns import PATTERNS, SelectorPattern
    from miowtion.h3 import attention as dense_runtime
    from miowtion.h3 import layout as packed_layout
    from miowtion.kernels import fa4
    from miowtion.train.data import parse_geometry
    from miowtion.veda import attention, bundle, mask
    from miowtion.veda.plan import PlanTable, TilePlan
    from safetensors import safe_open

    from scoring.spec import load_track_spec, track_spec_dict
    from validator.production import runtime
    from validator.production.environment import (
        evaluation_environment_digest,
        evaluation_source_digest,
        verify_runtime_environment,
    )

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise RuntimeError("expose exactly one H200 from the reserved calibration GPUs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_runtime_environment(arguments.source_bundle, arguments.dependency_lock)
    track = load_track_spec(arguments.track)
    plans = PlanTable(
        [
            TilePlan.from_json(value)
            for value in json.loads(
                bundle.read_metadata(str(arguments.predictor_bundle))["plans"]
            ).values()
        ]
    )
    registration = json.loads(arguments.registration.read_text())
    if registration["format"] != "veda-h200-cost-registration-v2":
        raise ValueError("unknown cost registration")
    if registration["measurement_script_sha256"] != sha256(__file__):
        raise ValueError("measurement script differs from preregistration")
    if registration["geometries"] != sorted(track.scoring.stratum_weight_map) or registration[
        "budgets"
    ] != [b.target_blocks for b in track.scoring.budgets]:
        raise ValueError("registered dimensions differ from production track")
    if registration["source_bundle_sha256"] != sha256(arguments.source_bundle) or registration[
        "dependency_lock_sha256"
    ] != sha256(arguments.dependency_lock):
        raise ValueError("cost registration does not bind this runtime")
    seeds = registration["seeds"][arguments.phase]
    if arguments.seed not in seeds:
        raise ValueError("seed does not belong to preregistered phase")
    if set(registration["seeds"]["calibration"]) & set(registration["seeds"]["confirmation"]):
        raise ValueError("calibration and confirmation seeds overlap")
    if registration["layers"] != list(range(track.model.selector.num_layers)):
        raise ValueError("cost registration must cover every production layer")
    if (arguments.warmup, arguments.repeats, arguments.profile_repeats) != (
        registration["warmup"],
        registration["repeats"],
        registration["profile_repeats"],
    ):
        raise ValueError("timing counts differ from preregistration")
    controls = tuple(registration["models"])
    if len(set(arguments.geometries)) != len(arguments.geometries) or set(
        arguments.geometries
    ) - set(track.scoring.stratum_weight_map):
        raise ValueError("invalid geometry shard")
    if arguments.phase == "calibration" and arguments.fit is not None:
        raise ValueError("calibration rejects --fit")
    if arguments.phase == "confirmation" and arguments.fit is None:
        raise ValueError("confirmation requires the immutable calibration fit")
    fit = None if arguments.fit is None else json.loads(arguments.fit.read_text())
    if fit is not None and (
        fit["registration_sha256"] != sha256(arguments.registration)
        or fit["source_sha256"] != evaluation_source_digest()
        or fit["environment_sha256"] != evaluation_environment_digest()
    ):
        raise ValueError("confirmation fit does not match registered runtime")
    if registration["pattern_script_sha256"] != sha256(
        Path(__file__).with_name("cost_patterns.py")
    ):
        raise ValueError("mask pattern implementation differs from preregistration")
    if registration["analysis_script_sha256"] != sha256(
        Path(__file__).with_name("cost_analysis.py")
    ):
        raise ValueError("cost contract verifier differs from preregistration")
    verify_phase_track(registration, json.loads(arguments.track.read_text()), arguments.phase, fit)
    identities = {}
    for name in controls:
        model_name = "baseline_self" if name.startswith("pattern_") else name
        if name.startswith("pattern_") and name.removeprefix("pattern_") not in PATTERNS:
            raise ValueError("unknown mask pattern probe")
        identities[name] = control_identity(
            arguments.controls / f"{model_name}.safetensors",
            schema=track.model.tensor_schema,
            max_bytes=track.model.max_bytes,
        )
    if identities != registration["model_identities"]:
        raise ValueError("cost model identities differ from preregistration")
    if sha256(arguments.predictor_bundle) != registration["predictor_bundle_sha256"]:
        raise ValueError("cost tile plans differ from preregistration")
    arguments.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "veda-h200-cost-measurement-v2",
        "phase": arguments.phase,
        "registration_sha256": sha256(arguments.registration),
        "fit_sha256": None if arguments.fit is None else sha256(arguments.fit),
        "scope": "registered_production_selector_cost_measurement",
        "source_sha256": evaluation_source_digest(),
        "environment_sha256": evaluation_environment_digest(),
        "dependency_lock_sha256": sha256(arguments.dependency_lock),
        "source_bundle_sha256": sha256(arguments.source_bundle),
        "script_sha256": sha256(__file__),
        "track_spec_digest": track.digest,
        "track_document": track_spec_dict(track),
        "cost_independent_track_sha256": cost_independent_contract(
            json.loads(arguments.track.read_text())
        ),
        "predictor_bundle_sha256": sha256(arguments.predictor_bundle),
        "model_identities": identities,
        "pattern_script_sha256": registration["pattern_script_sha256"],
        "pattern_estimand": (
            "real production projections plus precomputed legal logits before production selection"
        ),
        "seed": arguments.seed,
        "warmup": arguments.warmup,
        "repeats": arguments.repeats,
        "profile_repeats": arguments.profile_repeats,
        "input": "seeded synthetic BF16 full packed QKV; no hidden source families",
        "layers": registration["layers"],
        "text_token_counts": registration["text_token_counts"],
        "geometries": sorted(arguments.geometries),
        "budgets": [budget.target_blocks for budget in track.scoring.budgets],
        "candidate_costs": [budget.calibrated_cost for budget in track.scoring.budgets],
        "percentile": "nearest_rank",
        "e10_thresholds": {
            "spearman_min": 0.95,
            "median_relative_error_max": 0.05,
            "p95_relative_error_max": 0.10,
        },
        "e11_non_attention_fraction_max": 0.10,
        "e11_denominator": "installed dense H3 attention with FA4, same packed QKV and used rows",
        "e10_estimand": "complete tournament SelectorAttention call, synchronized wall latency",
        "dense_estimand": "complete production h3.attention.dense_attention call, backend=fa4",
        "collect_bytes": attention.VedaConfig().collect_bytes,
        "profiling": "CUDA spans around actual production functions; packing subtracted once",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_identity": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_once(arguments.output / "protocol.json", protocol)
    device = torch.device("cuda:0")
    rows = []
    with torch.inference_mode():
        for geometry_name in protocol["geometries"]:
            geometry = parse_geometry(geometry_name)
            plan = plans.select(geometry)
            for text_tokens in protocol["text_token_counts"]:
                layout = packed_layout.pack(torch.ones(text_tokens, dtype=torch.long), geometry)
                generator = torch.Generator(device=device).manual_seed(arguments.seed)
                shape = (
                    layout.seq_len,
                    track.model.selector.num_heads,
                    track.model.selector.head_dim,
                )
                q, k, v = [
                    torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
                    for _ in range(3)
                ]
                dense_path = arguments.output / (
                    f"dense-{geometry_name.replace(':', 'x')}-{text_tokens}.json"
                )
                if dense_path.exists():
                    dense_row = json.loads(dense_path.read_text())
                    if dense_row["protocol_sha256"] != sha256(arguments.output / "protocol.json"):
                        raise ValueError("dense timing belongs to another protocol")
                else:
                    dense_samples, dense_cuda = benchmark(
                        torch,
                        partial(dense_runtime.dense_attention, q, k, v, layout.used, backend="fa4"),
                        arguments.warmup,
                        arguments.repeats,
                    )
                    dense_row = {
                        "geometry": geometry_name,
                        "text_tokens": text_tokens,
                        "protocol_sha256": sha256(arguments.output / "protocol.json"),
                        "end_to_end_samples_ms": dense_samples,
                        "end_to_end_cuda_samples_ms": dense_cuda,
                        "end_to_end": summarize(dense_samples),
                    }
                write_once(dense_path, dense_row)
                dense_samples = dense_row["end_to_end_samples_ms"]
                for name in controls:
                    model_name = "baseline_self" if name.startswith("pattern_") else name
                    with safe_open(
                        arguments.controls / f"{model_name}.safetensors", framework="pt"
                    ) as tensors:
                        projections = {
                            f"layers.{layer}.{side}": tensors.get_tensor(
                                f"layers.{layer}.{side}"
                            ).to(device, torch.float32)
                            for layer in protocol["layers"] for side in ("proj_q", "proj_k")
                        }
                    for layer in protocol["layers"]:
                        for budget in track.scoring.budgets:
                            label = (
                                f"{geometry_name.replace(':', 'x')}-{text_tokens}"
                                f"-{name}-layer{layer}-{budget.target_blocks}"
                            )
                            path = arguments.output / f"{label}.json"
                            if path.exists():
                                row = json.loads(path.read_text())
                                if row["protocol_sha256"] != sha256(
                                    arguments.output / "protocol.json"
                                ):
                                    raise RuntimeError("resumed cost row uses another protocol")
                                write_once(path, row)
                                rows.append(row)
                                continue
                            clip = attention.ClipTiling(
                                layout,
                                attention.VedaConfig(
                                    target_budget=mask.Budget(tiles=budget.target_blocks)
                                ),
                                device,
                            )
                            operation = runtime._selector_attention(
                                torch, attention, clip, plan, projections, budget,
                            )

                            call = partial(operation, q, k, v, layer)

                            pattern = (
                                SelectorPattern(
                                    torch, runtime.runtime_block_mask, name.removeprefix("pattern_")
                                )
                                if name.startswith("pattern_")
                                else None
                            )
                            with ExitStack() as stack:
                                if pattern is not None:
                                    stack.enter_context(
                                        patch.object(runtime, "runtime_block_mask", pattern)
                                    )
                                    pattern.prime(call)
                                    call = pattern.replay(call)
                                post_check = None if pattern is None else pattern.verify_replay
                                torch.cuda.reset_peak_memory_stats()
                                samples, cuda_samples = benchmark(
                                    torch, call, arguments.warmup, arguments.repeats, post_check
                                )
                                components = profile_components(
                                    torch,
                                    attention,
                                    runtime,
                                    fa4,
                                    call,
                                    arguments.profile_repeats,
                                    post_check,
                                )
                            sparse = median(
                                row["sparse_attention_excluding_packing"] for row in components
                            )
                            non_attention_fraction = (median(samples) - sparse) / median(
                                dense_samples
                            )
                            row = {
                                "label": label,
                                "geometry": geometry_name,
                                "text_tokens": text_tokens,
                                "sequence_length": layout.seq_len,
                                "model": name,
                                "pattern_proofs": [] if pattern is None else pattern.proofs,
                                "layer": layer,
                                "dense_record": dense_path.name,
                                "dense_sha256": sha256(dense_path),
                                "budget": budget.target_blocks,
                                "candidate_cost": budget.calibrated_cost,
                                "protocol_sha256": sha256(arguments.output / "protocol.json"),
                                "end_to_end_samples_ms": samples,
                                "end_to_end_cuda_samples_ms": cuda_samples,
                                "end_to_end": summarize(samples),
                                "component_samples_ms": components,
                                "estimated_non_attention_fraction": non_attention_fraction,
                                "e11_measurement_pass": 0 <= non_attention_fraction <= 0.10,
                                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                            }
                            write_once(path, row)
                            rows.append(row)
                            print(
                                json.dumps(
                                    {
                                        "label": label,
                                        "median_ms": median(samples),
                                        "non_attention_fraction": non_attention_fraction,
                                    }
                                ),
                                flush=True,
                            )
                    projections.clear()
                del q, k, v
    expected = (
        len(protocol["geometries"])
        * len(protocol["text_token_counts"])
        * len(controls)
        * len(protocol["layers"])
        * len(protocol["budgets"])
    )
    write_once(
        arguments.output / "summary.json",
        {
            "format": "veda-h200-cost-measurement-summary-v2",
            "scope": "registered_production_selector_cost_measurement",
            "protocol_sha256": sha256(arguments.output / "protocol.json"),
            "row_count": len(rows),
            "expected_rows": expected,
            "all_dimensions_present": len(rows) == expected,
            "e11_measurement_pass": len(rows) == expected
            and all(row["e11_measurement_pass"] for row in rows),
            "maximum_estimated_non_attention_fraction": max(
                row["estimated_non_attention_fraction"] for row in rows
            ),
            "e10_confirmation_pass": False,
            "e10_missing": (
                "fit registered shared cost shape and fixed geometry/text scales on calibration; "
                "on disjoint seeds"
            ),
            "raw_sha256": {
                path.name: sha256(path)
                for path in sorted(arguments.output.glob("*.json"))
                if path.name != "summary.json"
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in (
        "track",
        "source-bundle",
        "dependency-lock",
        "predictor-bundle",
        "controls",
        "output",
        "registration",
    ):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--geometries", nargs="+", required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 5 or args.repeats < 20 or args.profile_repeats < 5:
        parser.error("insufficient warmup or repeat counts")
    run(args)
