import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from scoring.exact import CaseDesign, _weighted_cell_mean, paired_score_gain_bound
from tests.validator.production.test_storage import MemoryS3
from tests.validator.test_scoring import ready_for_scoring, scoring_spec
from validator.production.batches import ReadyBatchSource
from validator.production.commitment import canonical_json
from validator.production.storage import PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState
from validator.reproducibility import (
    _execution,
    pair_evidence,
    stage_reference,
    validate_pair_evidence,
    validate_stage_reference,
)
from validator.scoring import ScoringSettings, StageOutcome, process_next_scoring
from validator.store import ChallengeStore


@pytest.fixture
def proof():
    scoring = scoring_spec(routes=1, rollouts=4)
    track = SimpleNamespace(digest="ab" * 32, scoring=scoring)
    design = [CaseDesign("route", "all", f"rollout-{i}", f"source-{i}", 0, "all")
              for i in range(4)]
    challenger, incumbent = [.6, .65, .7, .75], [.5, .55, .65, .65]
    weights = scoring.stratum_weight_map, scoring.step_window_weight_map
    paired = SimpleNamespace(
        challenger_score=_weighted_cell_mean(challenger, design, *weights),
        incumbent_score=_weighted_cell_mean(incumbent, design, *weights),
        gain_lower_bound=paired_score_gain_bound(
            challenger, incumbent, design, stratum_weights=weights[0],
            step_window_weights=weights[1], confidence=scoring.match_confidence,
            resamples=scoring.bootstrap_resamples, seed=scoring.bootstrap_seed),
        challenger_cells=tuple(challenger), incumbent_cells=tuple(incumbent),
    )
    evidence = pair_evidence(track, {"cases": [asdict(row) for row in design]}, paired,
                             challenger_sha256="12" * 32, incumbent_sha256="23" * 32,
                             device=None, runtime_identity=None,
                             started_at_ms=time.time_ns() // 1_000_000)
    return track, paired, evidence


@pytest.mark.parametrize("damage", ["cell", "order", "seed", "weights", "runtime"])
def test_recorded_cells_recompute_exact_statistics_and_reject_changes(proof, damage):
    track, paired, evidence = proof
    validate_pair_evidence(json.loads(canonical_json(evidence)), paired.challenger_score,
                           paired.incumbent_score, paired.gain_lower_bound, track=track)
    changed = copy.deepcopy(evidence)
    if damage == "cell":
        changed["challenger_cells"][0] = .1
    elif damage == "order":
        changed["case_order"][0] = changed["case_order"][1]
    elif damage == "seed":
        changed["bootstrap"]["seed"] += 1
    elif damage == "weights":
        changed["stratum_weights"][0][1] = .5
    else:
        changed["execution"]["runtime_identity"] = {"release": {"source_sha256": "bad"}}
    with pytest.raises(ValueError):
        validate_pair_evidence(changed, paired.challenger_score, paired.incumbent_score,
                               paired.gain_lower_bound, track=track)


def test_recorded_weight_order_survives_canonical_json_without_changing_bootstrap():
    scoring = scoring_spec(routes=1, rollouts=4)
    scoring = SimpleNamespace(**{
        field: getattr(scoring, field) for field in (
            "budgets", "tail_mean_weight", "match_confidence", "bootstrap_resamples",
            "bootstrap_seed",
        )
    }, stratum_weight_map={"z": .6, "a": .4}, step_window_weight_map={"late": .3, "early": .7})
    track = SimpleNamespace(scoring=scoring)
    design = [CaseDesign("r", s, f"{s}-{i}", f"{s}-source-{i}", step, window)
              for s in scoring.stratum_weight_map for i in range(4)
              for step, window in enumerate(scoring.step_window_weight_map)]
    a = [.55 + i / 100 for i in range(len(design))]
    b = [.5 + i / 200 for i in range(len(design))]
    weights = scoring.stratum_weight_map, scoring.step_window_weight_map
    paired = SimpleNamespace(
        challenger_cells=tuple(a), incumbent_cells=tuple(b),
        challenger_score=_weighted_cell_mean(a, design, *weights),
        incumbent_score=_weighted_cell_mean(b, design, *weights),
        gain_lower_bound=paired_score_gain_bound(
            a, b, design, stratum_weights=weights[0], step_window_weights=weights[1],
            confidence=scoring.match_confidence, resamples=scoring.bootstrap_resamples,
            seed=scoring.bootstrap_seed),
    )
    body = pair_evidence(track, {"cases": [asdict(row) for row in design]}, paired,
                         challenger_sha256="12" * 32, incumbent_sha256="23" * 32,
                         device=None, runtime_identity=None,
                         started_at_ms=time.time_ns() // 1_000_000)
    restored = json.loads(canonical_json(body))
    assert [row[0] for row in restored["stratum_weights"]] == ["z", "a"]
    validate_pair_evidence(restored, paired.challenger_score, paired.incumbent_score,
                           paired.gain_lower_bound, track=track)


def test_private_stage_is_single_immutable_object_and_cold_restore_keeps_original_execution(
    tmp_path, monkeypatch, proof,
):
    import validator.reproducibility as module

    track, paired, evidence = proof
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/original")
    source = ReadyBatchSource(track=track, objects=objects, cache_dir=tmp_path)
    monkeypatch.setattr(source, "stage_binding", lambda *_a, **_k: {})
    task = SimpleNamespace(challenge_id="34" * 32, submission_id="45" * 32, hotkey="miner",
                           repo_id="owner/model", revision="56" * 20,
                           file_sha256="12" * 32, state=ChallengeState.SCORING,
                           position=ChainPosition(100, 0, 0))
    metrics = {"reproducibility": evidence, "incumbent_score": paired.incumbent_score,
               "paired_gain_lower_bound": paired.gain_lower_bound}
    result = StageOutcome(paired.challenger_score, metrics)
    reference = source.archive_stage(task, result, 120)
    validate_stage_reference(reference, task.challenge_id, "SCORING")
    payload = objects.get(reference["object"]).body
    assert reference == stage_reference(reference["object"], payload)
    assert "stage_evidence" not in json.loads(payload)["metrics"]  # No circular digest.
    assert objects.list_names("stages/" + task.challenge_id) == (reference["object"],)
    monkeypatch.setattr(module, "_execution", lambda *_a: pytest.fail("history recaptured now"))
    cold = ReadyBatchSource(track=track, objects=objects, cache_dir=tmp_path / "cold")
    restored = cold.restore_stage(task)
    assert restored.metrics["reproducibility"] == evidence
    assert restored.metrics["stage_evidence"] == reference
    assert source.archive_stage(task, restored, 120) == reference
    assert objects.get(reference["object"]).body == payload
    damaged = copy.deepcopy(reference)
    damaged["sha256"] = "00" * 32
    with pytest.raises(ValueError, match="changed during archive retry"):
        source.archive_stage(task, StageOutcome(result.score, {**metrics,
                                                              "stage_evidence": damaged}), 120)


def test_archive_reference_is_persisted_before_result_and_private_cells_are_not_projected(
    tmp_path, proof,
):
    _, paired, evidence = proof
    with ChallengeStore(tmp_path / "local.db") as store:
        identifier = ready_for_scoring(store)
        reference = stage_reference(f"stages/{identifier}/SCORING.json", b"private fixture")
        outcome = process_next_scoring(
            store, worker_id="worker", current_finalized_block=200, now_ms=20,
            settings=ScoringSettings(archive_stage=lambda *_a: reference),
            evaluator=lambda _task: StageOutcome(paired.challenger_score, {
                "reproducibility": evidence, "claim_evidence": {"secret": "private-only"},
            }),
        )
        assert outcome.state is ChallengeState.SUCCEEDED
        row = store.public_challenge(identifier)
        assert row["stages"][0]["stage_evidence"] == reference
        assert "private-only" not in json.dumps(row)
        assert "challenger_cells" not in json.dumps(row)
        assert "execution" not in json.dumps(row)
        assert store.stage_score(identifier, ChallengeState.SCORING).metrics[
            "stage_evidence"] == reference


def test_execution_binds_original_release_small_artifacts_and_does_not_read_secret_env(
    tmp_path, monkeypatch,
):
    from validator.production import environment

    source, lock = tmp_path / "source.zip", tmp_path / "dependencies.lock"
    source.write_bytes(b"frozen source")
    lock.write_bytes(b"frozen lock")
    identity = {
        "release": {"source_sha256": "ab" * 32, "environment_sha256": "cd" * 32},
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "source_bundle_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependency_lock_path": str(lock), "source_bundle_path": str(source),
        "verified_at_ms": 1,
    }
    monkeypatch.setattr(environment, "evaluation_source_digest", lambda: "ab" * 32)
    monkeypatch.setenv("CLOUDFLARE_CONTROL_API_TOKEN", "NEVER-COPY-SECRET")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NEVER-COPY-SECRET")
    captured = _execution(None, identity, time.time_ns() // 1_000_000)
    assert captured["actual_source_sha256"] == "ab" * 32
    assert captured["runtime_identity"] == identity
    assert captured["cuda_visible_devices"] == "unrecognized"
    assert "NEVER-COPY-SECRET" not in json.dumps(captured)
    assert captured["python_executable"]
    lock.write_bytes(b"changed lock")
    with pytest.raises(ValueError, match="runtime artifact differs"):
        _execution(None, identity, time.time_ns() // 1_000_000)


def test_recorded_approval_matches_competition_without_recapturing_the_runtime(proof):
    track, paired, body = proof
    execution = body["execution"]
    release = {key: "cd" * 32 for key in (
        "competition_manifest_sha256", "artifact_manifest_sha256", "acceptance_sha256",
        "source_sha256", "environment_sha256", "data_spec_sha256", "data_contract_sha256",
    )}
    release.update(track_spec_digest=track.digest, image_id="sha256:" + "ef" * 32)
    execution["runtime_identity"] = {
        "release": release, "verified_at_ms": execution["started_at_ms"],
        "dependency_lock_path": "/original/runtime/lock.json",
        "dependency_lock_sha256": "12" * 32,
        "source_bundle_path": "/original/runtime/source.zip",
        "source_bundle_sha256": "23" * 32,
    }
    execution["actual_source_sha256"] = release["source_sha256"]
    # Those original paths deliberately do not exist on this restoring host.
    validate_pair_evidence(body, paired.challenger_score, paired.incumbent_score,
                           paired.gain_lower_bound, track=track)
    release["track_spec_digest"] = "00" * 32
    with pytest.raises(ValueError, match="another competition"):
        validate_pair_evidence(body, paired.challenger_score, paired.incumbent_score,
                               paired.gain_lower_bound, track=track)


def test_cuda_execution_metadata_is_observed_not_inherited_from_release(monkeypatch):
    fake = SimpleNamespace(
        __version__="fixture", version=SimpleNamespace(cuda="fixture"),
        device=lambda _name: SimpleNamespace(type="cuda"),
        cuda=SimpleNamespace(get_device_properties=lambda _device: SimpleNamespace(
            name="fixture device", uuid="GPU-fixture", major=8, minor=0, total_memory=123)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=False, benchmark=False,
                                                       version=lambda: 123)),
        are_deterministic_algorithms_enabled=lambda: True,
        get_float32_matmul_precision=lambda: "highest",
    )
    monkeypatch.setitem(sys.modules, "torch", fake)  # No CUDA, tensors, or GPU execution.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    observed = _execution("cuda:0", None, time.time_ns() // 1_000_000)
    assert observed["hardware"]["uuid"] == "GPU-fixture"
    assert observed["logical_device"] == "cuda:0"
    assert observed["cuda_visible_devices"] == "2"
    assert observed["torch"]["matmul_tf32"] is False
    assert observed["runtime_identity"] is None
