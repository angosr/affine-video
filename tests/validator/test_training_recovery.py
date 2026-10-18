"""Training recovery preserves complete state and never overwrites verified inputs."""

import copy
import importlib.util
import io
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
        "format": "veda-training-recovery-inventory-v2",
        "configuration": {"teacher_adapter": "/checkpoint/state.pt"},
        "teacher_adapter_sha256": recovery.digest(source),
        "public_base_model": {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": "a" * 40,
            "files": [
                {
                    "repository_path": "FL2VA/transformer/model.safetensors",
                    "sha256": "b" * 64,
                    "url": "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/"
                    + "a" * 40
                    + "/FL2VA/transformer/model.safetensors",
                }
            ],
        },
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
    bad = copy.deepcopy(inventory)
    bad["teacher_adapter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter is absent"):
        recovery.validate_inventory(bad)
    bad = copy.deepcopy(inventory)
    bad["public_base_model"]["files"][0]["url"] = "https://example.com/mutable"
    with pytest.raises(ValueError, match="recovery binding"):
        recovery.validate_inventory(bad)


def test_base_binding_checks_actual_required_shards_against_public_revision(tmp_path, monkeypatch):
    root = tmp_path / "base"
    transformer = root / "FL2VA/transformer"
    transformer.mkdir(parents=True)
    (root / "FL2VA/model_index.json").write_bytes(b"model config")
    (transformer / "config.json").write_bytes(b"transformer config")
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"q": "shard.safetensors", "k": "shard.safetensors"}})
    )
    shard = transformer / "shard.safetensors"
    shard.write_bytes(b"checkpoint weights")
    tree = [
        {
            "path": "FL2VA/transformer/shard.safetensors",
            "lfs": {"oid": recovery.digest(shard), "size": shard.stat().st_size},
        }
    ]

    def public(url, **kwargs):
        if "/tree/" in url:
            return io.BytesIO(json.dumps(tree).encode())
        relative = url.split("/resolve/" + "a" * 40 + "/")[1]
        return io.BytesIO((root / relative).read_bytes())

    monkeypatch.setattr(recovery.urllib.request, "urlopen", public)
    config = {"checkpoint_root": str(root), "variant": "FL2VA"}
    binding = recovery.public_base_inputs(config, "MiniMaxAI/MiniMax-H3", "a" * 40)
    assert len(binding["files"]) == 4
    shard.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="differs from pinned"):
        recovery.public_base_inputs(config, "MiniMaxAI/MiniMax-H3", "a" * 40)
