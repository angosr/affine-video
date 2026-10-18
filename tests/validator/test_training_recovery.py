"""Training recovery preserves complete state and never overwrites verified inputs."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "training_recovery",
    Path(__file__).resolve().parents[2] / "evidence/system-panel/launch/training_recovery.py",
)
assert SPEC and SPEC.loader
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def resume():
    state = {
        "step": 28,
        "config": {"steps": 100, "accum": 4, "num_steps": 8, "trajectory_steps": None},
        "weights": {"p": 1},
        "ema": {"p": 1},
        "shared_generator": [14],
        "rank_states": [
            {"sampler": {"position": 14}, "noise": [1]},
            {"sampler": {"position": 14}, "noise": [2]},
        ],
    }
    return state, {"p": {"step": 28, "exp_avg": 1, "exp_avg_sq": 2}}, {"step": 28}


def test_original_two_ranks_are_preserved():
    assert recovery.validate_resume(*resume()) == 2


@pytest.mark.parametrize("mutation", ["moments", "optimizer_step", "sampler", "trajectory"])
def test_incomplete_resume_is_rejected(mutation):
    state, optimizer, marker = resume()
    if mutation == "moments":
        optimizer["p"].pop("exp_avg")
    elif mutation == "optimizer_step":
        optimizer["p"]["step"] = 27
    elif mutation == "sampler":
        state["rank_states"][1].pop("noise")
    else:
        state["step"] = marker["step"] = 27
    with pytest.raises(ValueError):
        recovery.validate_resume(state, optimizer, marker)


def test_mirror_checks_hashes_and_reuses_only_verified_files(tmp_path, monkeypatch):
    content = b"immutable resume state"
    source = tmp_path / "source"
    source.write_bytes(content)
    inventory = {
        "format": "veda-training-recovery-inventory-v1",
        "total_bytes": len(content),
        "inputs": [
            {
                "path": "/checkpoint/state.pt",
                "bytes": len(content),
                "sha256": recovery.digest(source),
            }
        ],
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    args = SimpleNamespace(
        inventory=path,
        inventory_sha256=recovery.digest(path),
        output=tmp_path / "mirror",
        host="gpu",
    )
    calls = []

    def transfer(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(content)

    monkeypatch.setattr(recovery.subprocess, "run", transfer)
    recovery.mirror(args)
    recovery.mirror(args)
    assert len(calls) == 1
    target = args.output / "checkpoint/state.pt"
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        recovery.mirror(args)
    assert len(calls) == 1
    bad = copy.deepcopy(inventory)
    bad["inputs"][0]["path"] = "/checkpoint/../escape"
    with pytest.raises(ValueError, match="canonical"):
        recovery.validate_inventory(bad)
