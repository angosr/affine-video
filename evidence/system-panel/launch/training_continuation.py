"""Bound history replay for an exact training continuation.

This adapter is installed explicitly by a registered launch, never at import.
It does not edit Miowtion. Its own bytes and the upstream monitor bytes must
both be bound by that launch. Training metrics are not E3 ranking evidence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


def bound_document(reference):
    if sha256(reference["path"]) != reference["sha256"]:
        raise ValueError("registered artifact hash mismatch")
    return json.loads(Path(reference["path"]).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_history(path, expected_sha256, step, metric):
    if sha256(path) != expected_sha256:
        raise ValueError("training history hash mismatch")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    updates = [row for row in rows if "step" in row and "event" not in row]
    if [row["step"] for row in updates] != list(range(1, step + 1)):
        raise ValueError("training history is not the complete checkpoint prefix")
    if any(not math.isfinite(float(row[metric])) for row in updates):
        raise ValueError("training history has a non-finite divergence metric")
    return updates


def continuation_config(original, *, run_name, out_dir, world_size, saved_world_size):
    if world_size != saved_world_size or world_size != 2:
        raise ValueError("exact continuation must retain both original ranks")
    if original["steps"] != 100 or original["abort_max_drop"] <= 0:
        raise ValueError("expected bounded 100-step schedule and active divergence guard")
    config = dict(original)
    config.update(
        run_name=run_name, out_dir=out_dir, persistent_dir=None, save_every=2, abort_patience=0
    )
    return config


def verify_ranking_sources(protocol, track, training_families):
    from validator.production.commitment import SourceProof, verify_source_proof

    if protocol.get("format") != "veda-e3-ranking-protocol-v1":
        raise ValueError("unsupported E3 ranking protocol")
    if protocol["source_pool_merkle_root"] != track.source_pool_merkle_root:
        raise ValueError("E3 source root differs from candidate")
    counts = Counter()
    families = set(training_families)
    used = set()
    commitments = tuple(item.sha256 for item in track.source_commitments)
    for row in protocol["source_proofs"]:
        proof = SourceProof.from_document(row["proof"])
        if proof.leaf.partition != "production" or not verify_source_proof(
            proof, track.source_pool_merkle_root, commitment_digests=commitments
        ):
            raise ValueError("E3 proof is outside the original production partition")
        if proof.leaf.source_family_id in families or proof.leaf.digest in used:
            raise ValueError("E3 stages overlap each other or the training families")
        families.add(proof.leaf.source_family_id)
        used.add(proof.leaf.digest)
        counts[(row["stage"], row["batch"], proof.leaf.generation_geometry)] += 1
    expected = {
        (stage, batch, geometry): 4
        for stage in ("exact", "full_video")
        for batch in range(2)
        for geometry in track.scoring.stratum_weight_map
    }
    if len(track.scoring.stratum_weight_map) != 4 or dict(counts) != expected:
        raise ValueError("E3 requires two disjoint batches per stage and four videos per geometry")
    if protocol["selected_commitment_digests"] != sorted(used):
        raise ValueError("E3 commitment reservation differs from exact source proofs")
    if protocol["ranking_gates"] != {
        "minimum_real_checkpoints": 30,
        "mixed_spearman_bootstrap_95_lower_min": 0.8,
        "mixed_minus_dense_bootstrap_95_lower_min": -0.02,
    }:
        raise ValueError("E3 ranking gates differ from the design")


@contextmanager
def restored_early_abort(trainer, *, history, configuration, monitor_sha256):
    original = trainer.monitor_lib.EarlyAbort
    source = inspect.getsourcefile(original)
    if source is None or sha256(source) != monitor_sha256:
        raise ValueError("upstream EarlyAbort source differs from registration")
    if tuple(inspect.signature(original).parameters) != (
        "window",
        "patience",
        "metric",
        "max_drop",
    ):
        raise ValueError("upstream EarlyAbort signature changed")
    metric = configuration["abort_metric"]
    if not history or [row["step"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("cannot restore an incomplete divergence history")
    if any(not math.isfinite(float(row[metric])) for row in history):
        raise ValueError("non-finite divergence history")
    instances = []

    class RestoredEarlyAbort(original):
        def __init__(self, window=8, patience=0, metric="kept_over_ceiling", max_drop=0.05):
            expected = (
                configuration["abort_window"],
                0,
                configuration["abort_metric"],
                configuration["abort_max_drop"],
            )
            if (window, patience, metric, max_drop) != expected:
                raise ValueError("continuation altered the registered divergence guard")
            super().__init__(window, patience, metric, max_drop)
            for row in history:
                reason = super().update(row)
                if reason:
                    raise ValueError(f"original prefix already violates divergence guard: {reason}")
            self.next_step = len(history) + 1
            instances.append(self)

        def update(self, record):
            if record.get("step") != self.next_step:
                raise ValueError("continuation divergence history skipped or repeated an update")
            if not math.isfinite(float(record[metric])):
                raise ValueError("non-finite continuation divergence metric")
            self.next_step += 1
            return super().update(record)

    trainer.monitor_lib.EarlyAbort = RestoredEarlyAbort
    try:
        yield instances
    finally:
        trainer.monitor_lib.EarlyAbort = original


def history_self_test(
    trainer, *, history_path, history_sha256, configuration, step, monitor_sha256
):
    """Exercise the actual installed guard, including update 29 divergence."""
    history = read_history(history_path, history_sha256, step, configuration["abort_metric"])
    original = trainer.monitor_lib.EarlyAbort
    args = (
        configuration["abort_window"],
        0,
        configuration["abort_metric"],
        configuration["abort_max_drop"],
    )
    reference = original(*args)
    for row in history:
        if reference.update(row):
            raise ValueError("original history already diverged")
    with restored_early_abort(
        trainer, history=history, configuration=configuration, monitor_sha256=monitor_sha256
    ) as instances:
        resumed = trainer.monitor_lib.EarlyAbort(*args)
        if resumed.__dict__ != {**reference.__dict__, "next_step": step + 1}:
            raise AssertionError("history replay did not restore the original guard state")
        first = resumed._first
        # A first resumed observation far below the original reference must
        # trip immediately. A freshly reset guard incorrectly waits a window.
        record = {"step": step + 1, configuration["abort_metric"]: -10.0}
        expected_reason = reference.update(record)
        if not expected_reason or resumed.update(record) != expected_reason:
            raise AssertionError("resumed divergence decision differs from uninterrupted guard")
        if original(*args).update(record) is not None:
            raise AssertionError("regression no longer distinguishes a reset baseline")
        assert len(instances) == 1
    if trainer.monitor_lib.EarlyAbort is not original:
        raise AssertionError("history adapter was not removed")
    return {
        "format": "veda-early-abort-continuity-v1",
        "prefix_updates": step,
        "history_sha256": history_sha256,
        "monitor_sha256": monitor_sha256,
        "original_reference": first,
        "update_29_divergence_identical": True,
        "reset_baseline_counterexample": True,
        "release_accepted": False,
    }


def verify_launch(registration):
    """CPU-only checks must finish before constructing the training model."""
    if registration.get("format") != "veda-e3-training-continuation-v1":
        raise ValueError("unsupported continuation registration")
    if sha256(__file__) != registration["continuation_script_sha256"]:
        raise ValueError("continuation script differs from registration")
    inventory = bound_document(registration["recovery_inventory"])
    receipt = bound_document(registration["recovery_receipt"])
    if (
        receipt.get("verified") is not True
        or receipt["inventory_sha256"] != registration["recovery_inventory"]["sha256"]
    ):
        raise ValueError("complete off-GPU recovery is required before training")
    if inventory["format"] != "veda-training-recovery-inventory-v2":
        raise ValueError("recovery inventory omits the base model or adapter")
    if receipt["files"] != len(inventory["inputs"]) or receipt["bytes"] != inventory["total_bytes"]:
        raise ValueError("recovery receipt is incomplete")
    for row in inventory["inputs"] + inventory["public_base_model"]["files"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"training input changed after registration: {path}")
    original = inventory["configuration"]
    if int(os.environ.get("WORLD_SIZE", "0")) != inventory["world_size"]:
        raise ValueError("launch must preserve the original world size")
    config = continuation_config(
        original,
        run_name=registration["run_name"],
        out_dir=registration["out_dir"],
        world_size=inventory["world_size"],
        saved_world_size=inventory["world_size"],
    )
    if config != registration["configuration"]:
        raise ValueError("configuration differs from the exact continuation derivation")
    run_directory = Path(config["out_dir"]) / config["run_name"]
    seeded = run_directory / "ckpt" / f"step_{inventory['step']:07d}"
    if set((run_directory / "ckpt").iterdir()) != {seeded}:
        raise ValueError("continuation directory must contain only the registered resume step")
    originals = {
        Path(row["path"]).name: row
        for row in inventory["inputs"]
        if Path(row["path"]).parent == Path(inventory["checkpoint"])
    }
    for name in ("state.pt", "optim.pt", "done.json"):
        if sha256(seeded / name) != originals[name]["sha256"]:
            raise ValueError("seeded resume state differs from the durable original")
    if (
        registration["required_gpu_free_bytes"] < 120 * 1024**3
        or shutil.disk_usage(run_directory).free < registration["required_gpu_free_bytes"]
    ):
        raise ValueError("insufficient registered checkpoint and selector storage")
    protocol = bound_document(registration["ranking_protocol"])
    from scoring.spec import load_track_spec

    bound_document(registration["export_track"])
    track = load_track_spec(registration["export_track"]["path"])
    training_manifest = next(
        row
        for row in inventory["inputs"]
        if Path(row["path"]) == Path(original["sample_cache"]).parent / "manifest.json"
    )
    families = bound_document(training_manifest)["source_families"]
    verify_ranking_sources(protocol, track, families)
    if registration["timeout_seconds"] != 14400 or registration["physical_gpus"] != [4, 5]:
        raise ValueError("unexpected supervision bound or GPU allocation")
    if registration["selector_steps"] != list(range(inventory["step"], 101, 2)):
        raise ValueError("registration omits real checkpoints from the ladder")
    if len(registration["selector_steps"]) < 30:
        raise ValueError("E3 requires at least 30 real checkpoints")
    from validator.production.environment import build_dependency_lock

    lock = registration["training_dependency_lock"]
    if (
        sha256(lock["path"]) != lock["sha256"]
        or hashlib.sha256(build_dependency_lock()).hexdigest() != lock["sha256"]
    ):
        raise ValueError("training environment differs from the frozen dependency bytes")
    return inventory, config


def export_checkpoint(directory, output, track):
    """Export live trained weights, preserving both file and tensor identities."""
    import torch
    from safetensors.torch import save_file

    from scoring.model import validate_model

    marker = json.loads((directory / "done.json").read_text())
    state_path = directory / "state.pt"
    if state_path.stat().st_size != marker["bytes"]:
        raise ValueError("cannot export an incomplete checkpoint")
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    if state["step"] != marker["step"]:
        raise ValueError("checkpoint step and completion marker differ")
    weights = {
        name.removeprefix("predictor."): value.to(torch.float16).contiguous()
        for name, value in state["weights"].items()
    }
    if set(weights) != set(track.model.selector.tensor_schema):
        raise ValueError("training checkpoint differs from the registered selector schema")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"step-{state['step']:04d}-live-f16.safetensors"
    temporary = path.with_suffix(".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError("checkpoint export already exists")
    save_file(dict(sorted(weights.items())), temporary)
    validated = validate_model(
        temporary, schema=track.model.selector.tensor_schema, max_bytes=track.model.max_bytes
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    record = {
        "format": "veda-e3-selector-checkpoint-v1",
        "step": state["step"],
        "selector": path.name,
        "file_sha256": validated.file_sha256,
        "model_digest": validated.model_digest,
        "bytes": validated.size_bytes,
        "checkpoint_state_sha256": sha256(state_path),
        "source_weights": "live",
        "export_track_digest_provenance": track.digest,
        "release_accepted": False,
    }
    with path.with_suffix(".json").open("x") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"event": "selector_ready", **record}), flush=True)


def run_registered(path, expected_sha256):
    registration = bound_document({"path": path, "sha256": expected_sha256})
    inventory, config = verify_launch(registration)
    from miowtion.train import trainer

    from scoring.spec import load_track_spec

    history_ref = registration["history"]
    history = read_history(
        history_ref["path"], history_ref["sha256"], inventory["step"], config["abort_metric"]
    )
    track_ref = registration["export_track"]
    bound_document(track_ref)
    track = load_track_spec(track_ref["path"])

    class LadderTrainer(trainer.Trainer):
        def _save(self):
            super()._save()
            self.ckpt.wait()
            if self.env.is_main:
                export_checkpoint(
                    Path(self.run_dir) / "ckpt" / f"step_{self.step:07d}",
                    Path(registration["selector_output"]),
                    track,
                )

    with restored_early_abort(
        trainer,
        history=history,
        configuration=config,
        monitor_sha256=registration["monitor_sha256"],
    ):
        instance = LadderTrainer(trainer.TrainConfig(**config))
        if instance.step != inventory["step"] or instance.env.world_size != inventory["world_size"]:
            raise ValueError("trainer did not restore the registered checkpoint and rank count")
        rank = instance.env.rank
        if instance.samples.state() != inventory["samplers"][rank]:
            raise ValueError("trainer did not restore the original sampler position")
        for tensor, expected in (
            (instance.noise_gen.get_state(), inventory["noise_state_sha256"][rank]),
            (instance.geometries.state(), inventory["geometry_state_sha256"]),
        ):
            if hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest() != expected:
                raise ValueError("trainer did not restore the original generator state")
        if instance.env.is_main:
            export_checkpoint(
                Path(inventory["checkpoint"]), Path(registration["selector_output"]), track
            )
        instance.train()
        if instance.step != config["steps"]:
            raise RuntimeError("training stopped before the registered final step")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--registration", required=True)
    run.add_argument("--registration-sha256", required=True)
    self_test = commands.add_parser("self-test")
    self_test.add_argument("--inventory", required=True)
    self_test.add_argument("--inventory-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        run_registered(arguments.registration, arguments.registration_sha256)
    else:
        from miowtion.train import trainer

        inventory = bound_document(
            {"path": arguments.inventory, "sha256": arguments.inventory_sha256}
        )
        history = next(row for row in inventory["inputs"] if row["path"].endswith("/log.jsonl"))
        monitor = next(
            row for row in inventory["inputs"] if row["path"].endswith("/miowtion/train/monitor.py")
        )
        result = history_self_test(
            trainer,
            history_path=history["path"],
            history_sha256=history["sha256"],
            configuration=inventory["configuration"],
            step=inventory["step"],
            monitor_sha256=monitor["sha256"],
        )
        result["continuation_script_sha256"] = sha256(__file__)
        print(json.dumps(result, sort_keys=True))
