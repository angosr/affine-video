"""Inventory and mirror training recovery inputs without starting training.

Run ``inspect`` in the original training environment. Run ``mirror`` on the
durable host with the resulting immutable inventory. Neither command uses CUDA.
The receipt proves recoverability only, never E3 ranking acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def validate_resume(state, optimizer, marker):
    config = state["config"]
    step = state["step"]
    if marker["step"] != step or marker.get("init_only"):
        raise ValueError("not a complete resume checkpoint")
    if not 0 < step < config["steps"]:
        raise ValueError("checkpoint is outside the training schedule")
    if config["trajectory_steps"] is not None:
        raise ValueError("truncated training trajectories are not supported")
    if step * config["accum"] % config["num_steps"]:
        raise ValueError("checkpoint omits an in-progress trajectory")
    ranks = state["rank_states"]
    if not ranks or state["shared_generator"] is None:
        raise ValueError("missing rank or geometry generator state")
    if set(optimizer) != set(state["weights"]) or set(state["ema"]) != set(optimizer):
        raise ValueError("optimizer, live weights and EMA identities differ")
    for value in optimizer.values():
        if set(value) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("expected complete Adam moments")
        if int(value["step"]) != step:
            raise ValueError("optimizer step differs from checkpoint")
    for rank in ranks:
        if set(rank) != {"sampler", "noise"} or not rank["sampler"]:
            raise ValueError("missing sampler or noise state")
    return len(ranks)


def inspect(args):
    import torch

    root = args.root.resolve()
    checkpoint = args.checkpoint.resolve()
    source = args.source.resolve()
    marker = json.loads((checkpoint / "done.json").read_text())
    state_path = checkpoint / "state.pt"
    if state_path.stat().st_size != marker["bytes"]:
        raise ValueError("checkpoint size disagrees with completion marker")
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    optimizer = torch.load(
        checkpoint / "optim.pt", map_location="cpu", weights_only=True, mmap=True
    )
    world_size = validate_resume(state, optimizer, marker)
    config = state["config"]
    history = [json.loads(line) for line in args.history.read_text().splitlines()]
    updates = [row for row in history if "step" in row and "event" not in row]
    if [row["step"] for row in updates] != list(range(1, state["step"] + 1)):
        raise ValueError("training history is not a complete update prefix")
    for row in updates:
        if config["abort_metric"] not in row:
            raise ValueError("history cannot restore divergence reference")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise ValueError("training source contains unbound changes")
    files = {checkpoint / name for name in ("state.pt", "optim.pt", "done.json")} | {
        args.history.resolve(),
        args.config.resolve(),
        root / "manifest.json",
    }
    for directory in (Path(config["sample_cache"]), Path(config["plan_dir"])):
        files.update(path for path in directory.rglob("*") if path.is_file())
    # Bind and preserve the complete tracked source, including submodules.
    tracked = (
        subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "--recurse-submodules", "-z"]
        )
        .decode()
        .split("\0")
    )
    files.update(source / name for name in tracked if name)
    inventory = []
    for path in sorted(files):
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        inventory.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    return {
        "format": "veda-training-recovery-inventory-v1",
        "release_accepted": False,
        "checkpoint": str(checkpoint),
        "step": state["step"],
        "world_size": world_size,
        "configuration": config,
        "samplers": [rank["sampler"] for rank in state["rank_states"]],
        "noise_state_sha256": [tensor_hash(rank["noise"]) for rank in state["rank_states"]],
        "geometry_state_sha256": tensor_hash(state["shared_generator"]),
        "miowtion_revision": revision,
        "python_executable": sys.executable,
        "python_executable_sha256": digest(sys.executable),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "history_updates": len(updates),
        "inputs": inventory,
        "total_bytes": sum(row["bytes"] for row in inventory),
    }


def validate_inventory(document):
    if document.get("format") != "veda-training-recovery-inventory-v1":
        raise ValueError("unsupported recovery inventory")
    seen = set()
    for row in document["inputs"]:
        path = Path(row["path"])
        if not path.is_absolute() or ".." in path.parts or str(path) != row["path"]:
            raise ValueError("input path must be canonical and absolute")
        if path in seen or row["bytes"] < 0:
            raise ValueError("duplicate input or invalid size")
        seen.add(path)
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("invalid input digest")
    if not seen or document["total_bytes"] != sum(row["bytes"] for row in document["inputs"]):
        raise ValueError("empty inventory or incorrect total bytes")


def mirror(args):
    document = json.loads(args.inventory.read_text())
    validate_inventory(document)
    if digest(args.inventory) != args.inventory_sha256:
        raise ValueError("recovery inventory hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    # Preserve the irreplaceable resume state before re-fetchable source files.
    checkpoint = document.get("checkpoint", "")
    ordered = sorted(
        document["inputs"],
        key=lambda row: (not row["path"].startswith(checkpoint + "/"), row["path"]),
    )
    for row in ordered:
        target = args.output / row["path"].lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                raise ValueError(f"existing recovery input changed: {target}")
            continue
        partial = target.with_name(target.name + ".partial")
        subprocess.run(
            [
                "rsync",
                "--protect-args",
                "--partial",
                "--append-verify",
                "--timeout=120",
                "-e",
                "ssh -o ControlMaster=auto -o ControlPersist=120 "
                "-o ControlPath=/tmp/veda-training-recovery-%C",
                f"{args.host}:{row['path']}",
                str(partial),
            ],
            check=True,
            timeout=7200,
        )
        if partial.stat().st_size != row["bytes"] or digest(partial) != row["sha256"]:
            raise ValueError(f"recovery input hash mismatch: {target}")
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        # A completed target is never overwritten, including on service restart.
        os.link(partial, target)
        partial.unlink()
        print(json.dumps({"mirrored": row["path"], "sha256": row["sha256"]}), flush=True)
    receipt = {
        "format": "veda-training-recovery-receipt-v1",
        "release_accepted": False,
        "inventory_sha256": args.inventory_sha256,
        "files": len(document["inputs"]),
        "bytes": document["total_bytes"],
        "verified": True,
    }
    path = args.output / "receipt.json"
    payload = canonical(receipt)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("recovery receipt changed")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect")
    for name in ("root", "checkpoint", "source", "history", "config"):
        inspect_parser.add_argument("--" + name, type=Path, required=True)
    mirror_parser = sub.add_parser("mirror")
    mirror_parser.add_argument("--host", required=True)
    mirror_parser.add_argument("--inventory", type=Path, required=True)
    mirror_parser.add_argument("--inventory-sha256", required=True)
    mirror_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        print(canonical(inspect(args)).decode(), end="")
    else:
        mirror(args)


if __name__ == "__main__":
    main()
