"""Replayable H200 pilot for E10/E11 using the installed production SparseStudent.

Synthetic QKV avoids consuming hidden acceptance families. This is a kernel and
selection-cost pilot, not full-rollout or final-release acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def write_once(path, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable calibration evidence conflicts: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def summarize(samples):
    if not samples or any(value <= 0 for value in samples):
        raise ValueError("latency samples must be positive")
    ordered = sorted(samples)
    return {"median_ms": median(samples), "p95_ms": ordered[ceil(0.95 * len(ordered)) - 1]}


def benchmark(torch, operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return values


def profile_components(torch, attention, mask, fa4, predictor, operation, repeats):
    """Bracket native calls without inserting synchronizations inside the attention call."""

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
        (attention, "_gather_and_pool", "gather_and_trippool"),
        (attention, "_gather", "value_gather"),
        (predictor, "forward", "residual_projection_and_logits"),
        (mask, "select_video_blocks", "stable_topk_and_anchors"),
        (mask, "dense_block_mask", "dense_mask_construction"),
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
    return samples


def run(arguments):
    import torch
    from miowtion.h3 import layout as packed_layout
    from miowtion.kernels import fa4
    from miowtion.train.data import parse_geometry
    from miowtion.veda import attention, bundle, mask
    from miowtion.veda.plan import PlanTable, TilePlan
    from miowtion.veda.predictor import TileScorePredictor
    from safetensors import safe_open

    from scoring.model import validate_model
    from scoring.spec import load_track_spec
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
    controls = ("baseline_self", "mean_pool", "shuffled_heads")
    identities = {}
    for name in controls:
        model = validate_model(
            arguments.controls / f"{name}.safetensors",
            schema=track.model.tensor_schema,
            max_bytes=track.model.max_bytes,
        )
        identities[name] = {"file_sha256": model.file_sha256, "model_digest": model.model_digest}
    arguments.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "veda-h200-cost-pilot-v1",
        "scope": "pilot_not_release_acceptance",
        "source_sha256": evaluation_source_digest(),
        "environment_sha256": evaluation_environment_digest(),
        "dependency_lock_sha256": sha256(arguments.dependency_lock),
        "source_bundle_sha256": sha256(arguments.source_bundle),
        "script_sha256": sha256(__file__),
        "track_spec_digest": track.digest,
        "predictor_bundle_sha256": sha256(arguments.predictor_bundle),
        "model_identities": identities,
        "seed": arguments.seed,
        "warmup": arguments.warmup,
        "repeats": arguments.repeats,
        "profile_repeats": arguments.profile_repeats,
        "input": "seeded synthetic BF16 full packed QKV; no hidden source families",
        "layers": [0],
        "text_token_counts": [77, 256],
        "geometries": sorted(track.scoring.stratum_weight_map),
        "budgets": [budget.target_blocks for budget in track.scoring.budgets],
        "candidate_costs": [budget.calibrated_cost for budget in track.scoring.budgets],
        "percentile": "nearest_rank",
        "e10_thresholds": {
            "spearman_min": 0.95,
            "median_relative_error_max": 0.05,
            "p95_relative_error_max": 0.10,
        },
        "e11_non_attention_fraction_max": 0.10,
        "profiling": "CUDA event spans around installed native functions; packing subtracted once",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_once(arguments.output / "protocol.json", protocol)
    device = torch.device("cuda:0")
    predictor = TileScorePredictor(1, track.model.selector.num_heads, track.model.selector.head_dim)
    predictor = predictor.to(device, torch.float32).eval()
    rows = []
    with torch.inference_mode():
        for geometry_name in sorted(track.scoring.stratum_weight_map):
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
                for name in controls:
                    with safe_open(
                        arguments.controls / f"{name}.safetensors", framework="pt"
                    ) as tensors:
                        for side in ("proj_q", "proj_k"):
                            getattr(predictor.layers[0], side).copy_(
                                tensors.get_tensor(f"layers.0.{side}")
                            )
                    for budget in track.scoring.budgets:
                        label = (
                            f"{geometry_name.replace(':', 'x')}-{text_tokens}"
                            f"-{name}-{budget.target_blocks}"
                        )
                        path = arguments.output / f"{label}.json"
                        if path.exists():
                            row = json.loads(path.read_text())
                            if row["protocol_sha256"] != sha256(arguments.output / "protocol.json"):
                                raise RuntimeError("resumed cost row uses another protocol")
                            rows.append(row)
                            continue
                        clip = attention.ClipTiling(
                            layout,
                            attention.VedaConfig(
                                target_budget=mask.Budget(tiles=budget.target_blocks)
                            ),
                            device,
                        )
                        operation = attention.SparseStudent(clip, plan, predictor)

                        call = partial(operation, q, k, v, 0)

                        torch.cuda.reset_peak_memory_stats()
                        samples = benchmark(torch, call, arguments.warmup, arguments.repeats)
                        components = profile_components(
                            torch,
                            attention,
                            mask,
                            fa4,
                            predictor.layers[0],
                            call,
                            arguments.profile_repeats,
                        )
                        sparse = median(
                            row["sparse_attention_excluding_packing"] for row in components
                        )
                        non_attention_fraction = 1 - sparse / median(samples)
                        row = {
                            "label": label,
                            "geometry": geometry_name,
                            "text_tokens": text_tokens,
                            "sequence_length": layout.seq_len,
                            "model": name,
                            "budget": budget.target_blocks,
                            "candidate_cost": budget.calibrated_cost,
                            "protocol_sha256": sha256(arguments.output / "protocol.json"),
                            "end_to_end_samples_ms": samples,
                            "end_to_end": summarize(samples),
                            "component_samples_ms": components,
                            "estimated_non_attention_fraction": non_attention_fraction,
                            "e11_pilot_pass": 0 <= non_attention_fraction <= 0.10,
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
                del q, k, v
    write_once(
        arguments.output / "summary.json",
        {
            "format": "veda-h200-cost-pilot-summary-v1",
            "scope": "pilot_not_release_acceptance",
            "protocol_sha256": sha256(arguments.output / "protocol.json"),
            "row_count": len(rows),
            "expected_rows": 4 * 2 * 3 * 3,
            "all_dimensions_present": len(rows) == 72,
            "e11_pilot_pass": len(rows) == 72 and all(row["e11_pilot_pass"] for row in rows),
            "maximum_estimated_non_attention_fraction": max(
                row["estimated_non_attention_fraction"] for row in rows
            ),
            "e10_confirmation_pass": False,
            "e10_missing": (
                "fit fixed geometry scale on pilot and evaluate preregistered errors "
                "on disjoint seeds"
            ),
            "raw_sha256": {
                path.name: sha256(path) for path in sorted(arguments.output.glob("*.json"))
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
    ):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 5 or args.repeats < 20 or args.profile_repeats < 5:
        parser.error("insufficient warmup or repeat counts")
    run(args)
