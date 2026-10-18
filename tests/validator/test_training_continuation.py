"""Continuation launch helpers reject changed schedules and incomplete history."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "training_continuation",
    Path(__file__).resolve().parents[2] / "evidence/system-panel/launch/training_continuation.py",
)
assert SPEC and SPEC.loader
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


def test_configuration_preserves_absolute_schedule_and_optimizer():
    original = {
        "steps": 100,
        "accum": 4,
        "lr": 0.001,
        "lr_decay": "cosine",
        "warmup": 20,
        "seed": 20261019,
        "abort_max_drop": 0.05,
        "abort_patience": 20,
        "save_every": 20,
        "persistent_dir": "/old",
        "run_name": "old",
        "out_dir": "/old-runs",
    }
    updated = continuation.continuation_config(
        original, run_name="continuation", out_dir="/new-runs", world_size=2, saved_world_size=2
    )
    assert {key for key in original if original[key] != updated[key]} == {
        "run_name",
        "out_dir",
        "persistent_dir",
        "save_every",
        "abort_patience",
    }
    assert original["abort_patience"] == 20
    with pytest.raises(ValueError, match="both original ranks"):
        continuation.continuation_config(
            original, run_name="new", out_dir="/new", world_size=1, saved_world_size=2
        )


@pytest.mark.parametrize("mutation", ["hash", "missing", "nonfinite"])
def test_history_requires_bound_complete_finite_prefix(tmp_path, mutation):
    path = tmp_path / "log.jsonl"
    rows = [{"step": i, "quality": 0.8} for i in range(1, 29)]
    if mutation == "missing":
        rows.pop(4)
    if mutation == "nonfinite":
        rows[4]["quality"] = float("nan")
    path.write_text("\n".join(json.dumps(row) for row in rows))
    digest = continuation.sha256(path) if mutation != "hash" else "0" * 64
    with pytest.raises(ValueError):
        continuation.read_history(path, digest, 28, "quality")


class RecordingGuard:
    """Interface double only; actual Miowtion decisions are tested in its OCI."""

    def __init__(self, window=8, patience=0, metric="quality", max_drop=0.05):
        self.records = []

    def update(self, record):
        self.records.append(record)


def test_explicit_patch_replays_before_update_and_restores_on_exception():
    trainer = SimpleNamespace(monitor_lib=SimpleNamespace(EarlyAbort=RecordingGuard))
    rows = [{"step": i, "quality": 0.8} for i in range(1, 29)]
    config = {"abort_window": 8, "abort_metric": "quality", "abort_max_drop": 0.05}
    with (
        pytest.raises(ValueError, match="skipped or repeated"),
        continuation.restored_early_abort(
            trainer,
            history=rows,
            configuration=config,
            monitor_sha256=continuation.sha256(__file__),
        ),
    ):
        guard = trainer.monitor_lib.EarlyAbort(8, 0, "quality", 0.05)
        assert guard.records == rows
        guard.update({"step": 29, "quality": 0.79})
        assert len(guard.records) == 29
        guard.update({"step": 31, "quality": 0.78})
    assert trainer.monitor_lib.EarlyAbort is RecordingGuard
    with (
        pytest.raises(ValueError, match="source differs"),
        continuation.restored_early_abort(
            trainer, history=rows, configuration=config, monitor_sha256="0" * 64
        ),
    ):
        pytest.fail("changed upstream source was accepted")


def ranking_protocol():
    from validator.production.commitment import SourceLeaf, SourcePool

    geometries = ("16:9@37", "1:1@72", "4:3@102", "9:16@37")
    leaves = tuple(
        SourceLeaf(
            "production",
            f"family-{geometry}-{i}",
            f"source-{geometry}-{i}",
            "prompt",
            (),
            geometry,
            "license",
            f"{i:064x}",
        )
        for geometry in geometries
        for i in range(16)
    )
    pool = SourcePool(leaves)
    track = SimpleNamespace(
        source_pool_merkle_root=pool.root,
        source_commitments=[SimpleNamespace(sha256=leaf.digest) for leaf in pool.leaves],
        scoring=SimpleNamespace(stratum_weight_map=dict.fromkeys(geometries, 0.25)),
    )
    rows = []
    for geometry in geometries:
        for i in range(16):
            rows.append(
                {
                    "stage": "exact" if i < 8 else "full_video",
                    "batch": (i // 4) % 2,
                    "proof": pool.proof(f"source-{geometry}-{i}").document(),
                }
            )
    protocol = {
        "format": "veda-e3-ranking-protocol-v1",
        "source_pool_merkle_root": pool.root,
        "source_proofs": rows,
        "selected_commitment_digests": sorted(leaf.digest for leaf in leaves),
        "ranking_gates": {
            "minimum_real_checkpoints": 30,
            "mixed_spearman_bootstrap_95_lower_min": 0.8,
            "mixed_minus_dense_bootstrap_95_lower_min": -0.02,
        },
    }
    return protocol, track


@pytest.mark.parametrize("mutation", [None, "training_overlap", "missing", "forged", "gates"])
def test_ranking_sources_bind_disjoint_complete_original_merkle_proofs(mutation):
    protocol, track = ranking_protocol()
    protocol = copy.deepcopy(protocol)
    families = []
    if mutation == "training_overlap":
        families = [protocol["source_proofs"][0]["proof"]["leaf"]["source_family_id"]]
    elif mutation == "missing":
        protocol["source_proofs"].pop()
    elif mutation == "forged":
        protocol["source_proofs"][0]["proof"]["leaf"]["prompt"] = "altered"
    elif mutation == "gates":
        protocol["ranking_gates"]["minimum_real_checkpoints"] = 20
    if mutation is None:
        continuation.verify_ranking_sources(protocol, track, families)
    else:
        with pytest.raises(ValueError):
            continuation.verify_ranking_sources(protocol, track, families)
