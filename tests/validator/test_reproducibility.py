import asyncio
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from scoring.exact import CaseDesign, _weighted_cell_mean, paired_score_gain_bound
from tests.validator.production.test_inventory import inventory_case as inventory_fixture
from tests.validator.production.test_inventory import make_plan
from tests.validator.production.test_storage import MemoryS3
from tests.validator.test_scoring import ready_for_scoring, scoring_spec
from validator.production.batches import ReadyBatchSource
from validator.production.commitment import canonical_json
from validator.production.storage import PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState
from validator.reproducibility import (
    _execution,
    pair_evidence,
    plan_case_order,
    stage_reference,
    validate_pair_evidence,
    validate_stage_reference,
    verify_public_score_audit,
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


@pytest.fixture
def plan_inputs_case(monkeypatch):
    return inventory_fixture.__wrapped__(monkeypatch)


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


def cell_statistics(evidence):
    design = [CaseDesign(**row) for row in evidence["case_order"]]
    weights = dict(evidence["stratum_weights"]), dict(evidence["step_window_weights"])
    a, b = evidence["challenger_cells"], evidence["incumbent_cells"]
    return (_weighted_cell_mean(a, design, *weights), _weighted_cell_mean(b, design, *weights),
            paired_score_gain_bound(a, b, design, stratum_weights=weights[0],
                                    step_window_weights=weights[1], **evidence["bootstrap"]))


def test_whole_route_removal_is_not_the_original_384_cases():
    from scoring.spec import load_track_spec

    track = load_track_spec("track.json")
    track = replace(track, scoring=replace(track.scoring, bootstrap_resamples=20))
    scoring = track.scoring
    design = [CaseDesign(f"r{r}", s, f"{s}-{i}", f"source-{s}-{i}", j, w)
              for r in range(scoring.sampled_route_count) for s in scoring.stratum_weight_map
              for i in range(scoring.rollouts_per_stratum)
              for j, w in enumerate(scoring.step_window_weight_map)]
    assert len(design) == scoring.case_count == 384
    evidence = {"case_order": [asdict(row) for row in design],
                "challenger_cells": [.6] * len(design), "incumbent_cells": [.5] * len(design),
                "stratum_weights": [list(row) for row in scoring.stratum_weight_map.items()],
                "step_window_weights": [list(row) for row in
                                        scoring.step_window_weight_map.items()],
                "bootstrap": {"confidence": scoring.match_confidence,
                              "resamples": scoring.bootstrap_resamples,
                              "seed": scoring.bootstrap_seed}}
    a, b, bound = cell_statistics(evidence)
    paired = SimpleNamespace(challenger_score=a, incumbent_score=b, gain_lower_bound=bound,
                             challenger_cells=evidence["challenger_cells"],
                             incumbent_cells=evidence["incumbent_cells"])
    evidence = pair_evidence(track, {"cases": evidence["case_order"]}, paired,
                             challenger_sha256="12" * 32, incumbent_sha256="23" * 32,
                             device=None, runtime_identity=None,
                             started_at_ms=time.time_ns() // 1_000_000)
    keep = [i for i, row in enumerate(evidence["case_order"]) if row["route"] != "r0"]
    for key in ("case_order", "challenger_cells", "incumbent_cells"):
        evidence[key] = [evidence[key][i] for i in keep]
    scores = cell_statistics(evidence)
    # This subset is still a complete Cartesian product and internally consistent.
    validate_pair_evidence(evidence, *scores)
    with pytest.raises(ValueError, match="geometry"):
        validate_pair_evidence(evidence, *scores, track=track)
    with pytest.raises(ValueError, match="geometry"):
        pair_evidence(track, {"cases": evidence["case_order"]}, SimpleNamespace(
            challenger_score=scores[0], incumbent_score=scores[1], gain_lower_bound=scores[2],
            challenger_cells=evidence["challenger_cells"],
            incumbent_cells=evidence["incumbent_cells"]),
            challenger_sha256="12" * 32, incumbent_sha256="23" * 32,
            device=None, runtime_identity=None, started_at_ms=time.time_ns() // 1_000_000)


@pytest.mark.parametrize("change", ["order", "route", "rollout", "source", "step", "window"])
def test_self_consistent_cells_must_match_original_plan_design(audit_case, change):
    case = audit_case
    evidence = copy.deepcopy(case.metrics["reproducibility"])
    if change == "order":
        for key in ("case_order", "challenger_cells", "incumbent_cells"):
            evidence[key].reverse()
    else:
        field = {"route": "route", "rollout": "rollout_id", "source": "source_id",
                 "step": "step_index", "window": "step_window"}[change]
        values = list(dict.fromkeys(row[field] for row in evidence["case_order"]))
        for row in evidence["case_order"]:
            row[field] = (row[field] + "-other" if change in {"route", "source", "rollout"} else
                          values[(values.index(row[field]) + 1) % len(values)])
    scores = cell_statistics(evidence)
    validate_pair_evidence(evidence, *scores, track=case.track)  # Statistical replay is valid.
    with pytest.raises(ValueError, match="authenticated READY plan"):
        validate_pair_evidence(evidence, *scores, track=case.track, plan=case.plan)


def test_cache_metadata_order_is_checked_before_any_tensor_read(audit_case, monkeypatch):
    from validator.production import batches

    case = audit_case
    metadata = {"sample_cache": case.plan.sample_cache.document(),
                "cases": list(reversed(plan_case_order(case.plan)))}
    monkeypatch.setattr(batches, "ExactCacheReader", lambda *_a, **_k: SimpleNamespace(
        metadata=metadata))
    monkeypatch.setattr(batches.np, "load", lambda *_a, **_k: pytest.fail("opened tensor payload"))
    with pytest.raises(batches.CacheLoadError, match="frozen batch plan"):
        batches._validate_plan_case_binding(None, case.plan, case.track)


@pytest.mark.parametrize("operation", ["capture", "restore"])
def test_stage_capture_and_restore_bind_plan_without_cache_payload(
    audit_case, tmp_path, monkeypatch, operation,
):
    case = audit_case
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="original")
    source = ReadyBatchSource(track=case.track, objects=objects, cache_dir=tmp_path / "unread")
    monkeypatch.setattr(source, "_allocated_plan", lambda *_a, **_k: case.plan)
    monkeypatch.setattr(source, "stage_binding", case.source.stage_binding)
    monkeypatch.setattr(source, "_restore_exact", lambda *_a: pytest.fail("restored 10GB cache"))
    metrics = copy.deepcopy(case.metrics)
    for row in metrics["reproducibility"]["case_order"]:
        row["source_id"] += "-replacement"
    scores = cell_statistics(metrics["reproducibility"])
    validate_pair_evidence(metrics["reproducibility"], *scores, track=case.track)
    metrics["incumbent_score"], metrics["paired_gain_lower_bound"] = scores[1:]
    original = case.source.objects.get(case.reference["object"]).body
    if operation == "capture":
        with pytest.raises(ValueError, match="authenticated READY plan"):
            source.archive_stage(case.task, StageOutcome(scores[0], metrics),
                                 case.item["score_block"])
        assert objects.list_names("stages/") == ()
    else:
        document = json.loads(original)
        document.update(metrics=metrics, score=scores[0])
        objects.create_plan(case.reference["object"], canonical_json(document))
        claim = metrics["cache_allocation"]
        monkeypatch.setattr(source.inventory, "read", lambda: (None, {"claims": [claim]}))
        objects.create_plan(f"admissions/{case.task.challenge_id}.json", canonical_json({
            "track_spec_digest": case.track.digest, "model_digest": claim["model_digest"],
            "file_sha256": case.task.file_sha256, "challenge_id": case.task.challenge_id,
        }))
        with pytest.raises(ValueError, match="authenticated READY plan"):
            source.restore_stage(case.task)
    assert case.source.objects.get(case.reference["object"]).body == original


def test_production_evidence_cannot_drop_plan_binding_on_restore(audit_case, tmp_path):
    case = audit_case
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="original")
    source = ReadyBatchSource(track=case.track, objects=objects, cache_dir=tmp_path / "unread")
    source.track = SimpleNamespace(digest=case.track.digest, netuid=120)
    document = json.loads(case.source.objects.get(case.reference["object"]).body)
    del document["metrics"]["cache_generated_block"]
    objects.create_plan(case.reference["object"], canonical_json(document))
    with pytest.raises(RuntimeError, match="lacks its allocated READY batch"):
        source.restore_stage(case.task)


def test_recorded_weight_order_survives_canonical_json_without_changing_bootstrap():
    scoring = scoring_spec(routes=1, rollouts=4)
    scoring = SimpleNamespace(**{
        field: getattr(scoring, field) for field in (
            "budgets", "tail_mean_weight", "match_confidence", "bootstrap_resamples",
            "bootstrap_seed",
        )
    }, stratum_weight_map={"z": .6, "a": .4}, step_window_weight_map={"late": .3, "early": .7},
        sampled_route_count=1, rollouts_per_stratum=4, sampled_step_count=2, case_count=16)
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


@pytest.mark.parametrize("missing", ["cells", "approved-runtime"])
def test_new_scoring_cannot_silently_become_a_legacy_stage(tmp_path, proof, missing):
    track, paired, evidence = proof
    track = SimpleNamespace(digest=track.digest, scoring=track.scoring, netuid=120)
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="original")
    source = ReadyBatchSource(track=track, objects=objects, cache_dir=tmp_path)
    task = SimpleNamespace(challenge_id="34" * 32, state=ChallengeState.SCORING)
    metrics = {} if missing == "cells" else {"reproducibility": evidence}
    with pytest.raises(ValueError, match="new .*requires"):
        source.archive_stage(task, StageOutcome(paired.challenger_score, metrics), 120)
    assert objects.list_names("stages/") == ()


def test_legacy_stage_replay_never_captures_the_new_process_runtime(tmp_path, monkeypatch):
    import validator.reproducibility as module
    from tests.validator.production.test_storage import seed_legacy_stage

    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="old")
    source = ReadyBatchSource(track=SimpleNamespace(digest="ab" * 32), objects=objects,
                              cache_dir=tmp_path)
    task = SimpleNamespace(challenge_id="34" * 32, submission_id="45" * 32, hotkey="miner",
                           repo_id="owner/model", revision="56" * 20, file_sha256="12" * 32,
                           state=ChallengeState.SCORING, position=ChainPosition(100, 0, 0))
    seed_legacy_stage(source, task, StageOutcome(.7, {}), 120)
    before = source.objects.get(source._stage_name(task.challenge_id, "SCORING")).body
    monkeypatch.setattr(module, "_execution", lambda *_a: pytest.fail("recaptured current runtime"))
    old = source.restore_stage(task)
    assert old.metrics == {}
    assert source.archive_stage(task, StageOutcome(old.score, old.metrics), old.score_block) is None
    assert source.objects.get(source._stage_name(task.challenge_id, "SCORING")).body == before


def test_archive_reference_is_persisted_before_result_and_private_cells_are_not_projected(
    tmp_path, proof,
):
    _, paired, evidence = proof
    captured = []

    def archive(_task, result, _block):
        captured.append(copy.deepcopy(result.metrics))
        return reference

    with ChallengeStore(tmp_path / "local.db") as store:
        identifier = ready_for_scoring(store)
        reference = stage_reference(f"stages/{identifier}/SCORING.json", b"private fixture")
        outcome = process_next_scoring(
            store, worker_id="worker", current_finalized_block=200, now_ms=20,
            settings=ScoringSettings(archive_stage=archive),
            evaluator=lambda _task: StageOutcome(paired.challenger_score, {
                "reproducibility": evidence, "claim_evidence": {"secret": "private-only"},
                "audit_context": {"secret": "original-only"},
            }),
        )
        assert outcome.state is ChallengeState.SUCCEEDED
        row = store.public_challenge(identifier)
        assert row["stages"][0]["stage_evidence"] == reference
        assert "private-only" not in json.dumps(row)
        assert "challenger_cells" not in json.dumps(row)
        assert "execution" not in json.dumps(row)
        stored = store.stage_score(identifier, ChallengeState.SCORING).metrics
        assert stored == {"stage_evidence": reference}
        assert captured[0]["reproducibility"] == evidence
        assert captured[0]["claim_evidence"] == {"secret": "private-only"}
        assert captured[0]["audit_context"] == {"secret": "original-only"}


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


@pytest.fixture
def observed_cuda_execution(monkeypatch):
    fake = SimpleNamespace(
        __version__="fixture", version=SimpleNamespace(cuda="fixture"),
        device=lambda _name: SimpleNamespace(type="cuda"),
        cuda=SimpleNamespace(get_device_properties=lambda _device: SimpleNamespace(
            name="fixture device", uuid="GPU-fixture", major=8, minor=0, total_memory=123)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=False, benchmark=False,
                                                       version=lambda: 123)),
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
    )
    monkeypatch.setitem(sys.modules, "torch", fake)  # No CUDA, tensors, or GPU execution.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return _execution("cuda:0", None, time.time_ns() // 1_000_000)


def test_cuda_execution_metadata_is_observed_not_inherited_from_release(observed_cuda_execution):
    observed = observed_cuda_execution
    assert observed["hardware"]["uuid"] == "GPU-fixture"
    assert observed["logical_device"] == "cuda:0"
    assert observed["cuda_visible_devices"] == "2"
    assert observed["torch"]["matmul_tf32"] is False
    assert observed["torch"]["cublas_workspace_config"] == ":4096:8"
    assert observed["torch"]["deterministic_warn_only"] is False
    assert observed["runtime_identity"] is None


@pytest.mark.parametrize("legacy", [False, True])
def test_cuda_policy_evidence_cold_read_preserves_old_facts(
    monkeypatch, proof, observed_cuda_execution, legacy,
):
    track, paired, evidence = proof
    evidence["execution"] = observed_cuda_execution
    if legacy:
        runtime = evidence["execution"]["torch"]
        del runtime["cublas_workspace_config"], runtime["deterministic_warn_only"]
        runtime.update(deterministic_algorithms=False, cudnn_benchmark=True,
                       float32_matmul_precision="high")
    original = canonical_json(evidence)
    monkeypatch.setitem(sys.modules, "torch", None)  # Cold JSON verification cannot use Torch.
    validate_pair_evidence(json.loads(original), paired.challenger_score, paired.incumbent_score,
                           paired.gain_lower_bound, track=track)
    assert canonical_json(evidence) == original


@pytest.mark.parametrize("field,value", [
    ("cublas_workspace_config", ":16:8"), ("deterministic_warn_only", True),
    ("deterministic_algorithms", False), ("cudnn_benchmark", True),
    ("float32_matmul_precision", "high"), ("cublas_workspace_config", None),
    ("deterministic_warn_only", None),
])
def test_new_cuda_policy_evidence_rejects_drift_or_partial_fields(
    proof, observed_cuda_execution, field, value,
):
    track, paired, evidence = proof
    evidence["execution"] = observed_cuda_execution
    if value is None:
        del evidence["execution"]["torch"][field]
    else:
        evidence["execution"]["torch"][field] = value
    with pytest.raises(ValueError, match="CUDA"):
        validate_pair_evidence(evidence, paired.challenger_score, paired.incumbent_score,
                               paired.gain_lower_bound, track=track)


@pytest.fixture(params=[False, True], ids=["legacy-claim-unknown", "first-cas-evidence"])
def audit_case(tmp_path, monkeypatch, request, plan_inputs_case):
    from scoring.spec import parse_track_spec, track_spec_dict
    from tests.validator.test_tournament import tenure_challenge
    from validator.tournament import genesis_champion, resolve_tournament

    inventory_case = plan_inputs_case
    raw = track_spec_dict(inventory_case.track)
    raw["promotion"] = {"format": "veda-exact-tenure-v1", "minimum_gain": .002,
                        "decay_seconds": 604800}
    raw["scoring"].update(minimum_frontier_gain=.003, match_confidence=.9975,
                           bootstrap_resamples=4000)
    raw["full_video"]["minimum_gain"] = .003
    track = parse_track_spec(raw)
    inventory_case.track = track
    scoring = track.scoring
    plan = make_plan(inventory_case, 95)
    design = [CaseDesign(**row) for row in plan_case_order(plan)]
    parameters = request.param if isinstance(request.param, tuple) else (request.param, True)
    has_claim, won = parameters[:2]
    vda2 = len(parameters) == 3 and parameters[2]
    b = [.65] * len(design)
    a = [.75 + i / 1000 for i in range(len(design))] if won else b[:]
    weights = scoring.stratum_weight_map, scoring.step_window_weight_map
    paired = SimpleNamespace(
        challenger_cells=a, incumbent_cells=b,
        challenger_score=_weighted_cell_mean(a, design, *weights),
        incumbent_score=_weighted_cell_mean(b, design, *weights),
        gain_lower_bound=paired_score_gain_bound(
            a, b, design, stratum_weights=weights[0], step_window_weights=weights[1],
            confidence=scoring.match_confidence, resamples=scoring.bootstrap_resamples,
            seed=scoring.bootstrap_seed),
    )
    evidence = pair_evidence(track, {"cases": [asdict(row) for row in design]}, paired,
                             challenger_sha256="12" * 32,
                             incumbent_sha256=track.initial_champion.evaluation_file_sha256,
                             device=None, runtime_identity=None,
                             started_at_ms=time.time_ns() // 1_000_000)
    baseline = replace(genesis_champion(track), tenure_block=100)  # Authenticated fixture fact.
    item = tenure_challenge(1, baseline, gain=paired.gain_lower_bound,
                            submitted_ms=200_000)
    item.update(challenge_id="34" * 32, submission_id="45" * 32, file_sha256="12" * 32,
                final_score=paired.challenger_score, incumbent_score=paired.incumbent_score)
    if vda2:
        item["declared_file_sha256"] = item["file_sha256"]
    batch = {"sha256": plan.exact.sha256, "plan_digest": plan.plan_digest,
             "seed_block_hash": plan.block_hash, "generated_block": 95,
             "completed_block": plan.completed_block, "completed_at_ms": plan.completed_at_ms,
             "ready_sha256": hashlib.sha256(plan.encode()).hexdigest()}
    allocation = dict(ordinal=1, challenge_id=item["challenge_id"], stage="SCORING",
                      plan_digest=batch["plan_digest"], ready_sha256=batch["ready_sha256"],
                      file_sha256=item["file_sha256"], model_digest=item["model_digest"],
                      submission_block=item["block_height"], submission_timestamp_ms=200_000)
    batch["allocation"] = allocation
    stage = item["stages"][0]
    stage.update(score=paired.challenger_score, incumbent_score=paired.incumbent_score,
                 evaluation_batch=batch)
    item["evaluation_batch"] = batch
    metrics = {"reproducibility": evidence, "promotion": stage["promotion"],
               "incumbent_score": paired.incumbent_score,
               "paired_gain_lower_bound": paired.gain_lower_bound,
               **{f"cache_{key}": batch[key] for key in batch if key not in
                  {"sha256", "generated_block", "plan_digest", "seed_block_hash"}},
               "cache_sha256": batch["sha256"], "cache_generated_block": 95,
               "cache_plan_digest": batch["plan_digest"],
               "cache_seed_block_hash": batch["seed_block_hash"]}
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="original")
    if has_claim:
        from validator.production.inventory import Inventory

        metrics["claim_evidence"] = Inventory(objects, track.digest).claim(
            **{k: v for k, v in allocation.items() if k not in
               {"ordinal", "plan_digest", "ready_sha256"}},
            plans=(plan, make_plan(inventory_case, 150)),
            disclosure_delay_seconds=track.cache.disclosure_delay_seconds)
    metrics["private_payload"] = "NEVER-PUBLISH-MODEL-OR-SECRET"
    source = ReadyBatchSource(track=track, objects=objects, cache_dir=tmp_path)
    monkeypatch.setattr(source, "_allocated_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(source, "stage_binding", lambda *_a, **_k: {
        key: metrics[key] for key in ("cache_ready_sha256", "cache_completed_block",
                                     "cache_completed_at_ms", "cache_allocation", "claim_evidence")
        if key in metrics})
    task = SimpleNamespace(challenge_id=item["challenge_id"], submission_id=item["submission_id"],
                           hotkey=item["hotkey"], repo_id=item["repo_id"],
                           revision=item["revision"],
                           declared_file_sha256=item.get("declared_file_sha256"),
                           file_sha256=item["file_sha256"], state=ChallengeState.SCORING,
                           position=ChainPosition(item["block_height"], 0, 0))
    reference = source.archive_stage(task, StageOutcome(paired.challenger_score, metrics),
                                     item["score_block"])
    stage["stage_evidence"] = reference
    record = resolve_tournament(
        [item], track_id=track.track_id, track_spec_digest=track.digest, netuid=track.netuid,
        minimum_gain=.003, minimum_exact_gain=.003, promotion=track.promotion,
        initial_champion=baseline,
    ).matches[0]
    due = reference["public_audit"]["not_before_ms"]
    payload = source.public_stage_audit(reference, now_ms=due)
    return SimpleNamespace(track=track, source=source, reference=reference, item=item,
                           record=record, payload=payload, due=due, raw=raw,
                           has_claim_evidence=has_claim, task=task, metrics=metrics, plan=plan)


@pytest.mark.parametrize("audit_case", [(True, True, True)], indirect=True)
def test_vda2_public_score_audit_and_verify_match_share_original_declaration(
    audit_case, tmp_path, capsys,
):
    from cli import main

    case = audit_case
    assert case.record.document["submission"]["declared_file_sha256"] == "12" * 32
    before = case.source.objects.get(case.reference["object"]).body
    result = verify_public_score_audit(
        case.payload, canonical_json(case.record.document), case.track,
    )
    assert result["gain_lower_bound"] == case.record.score_gain
    paths = [tmp_path / name for name in ("rule.json", "match.json", "score-audit.json")]
    for path, body in zip(paths, (canonical_json(case.raw), canonical_json(case.record.document),
                                  case.payload), strict=True):
        path.write_bytes(body)
    main(["verify-match", "--track", str(paths[0]), "--match", str(paths[1]),
          "--artifact", str(paths[2])])
    assert json.loads(capsys.readouterr().out)["gain_lower_bound"] == case.record.score_gain
    assert case.source.objects.get(case.reference["object"]).body == before


def test_public_cpu_audit_is_bound_to_actual_match_and_preserves_original_stage(audit_case):
    case = audit_case
    before = case.source.objects.get(case.reference["object"]).body
    result = verify_public_score_audit(case.payload, canonical_json(case.record.document),
                                      case.track)
    assert result["case_count"] == 24 and result["outcome"] == "CHALLENGER_WON"
    assert result["verification_scope"] == "statistics-only"
    assert result["model_inference_reproduced"] is False
    diagnostics = result["cell_diagnostics"]
    body = json.loads(case.payload)
    assert diagnostics["bootstrap"] == body["metrics"]["reproducibility"]["bootstrap"]
    assert diagnostics["weighted_mean_difference"] == (
        result["challenger_score"] - result["incumbent_score"])
    assert diagnostics["decision_margin"] == result["gain_lower_bound"] - result["minimum_gain"]
    assert diagnostics["design_coverage"] == {
        "route_count": 2, "rollouts_by_stratum": {"all": 4},
        "steps_by_window": {"early": 1, "late": 1, "middle": 1},
    }
    dispersion = diagnostics["descriptive_unweighted_cell_differences"]
    assert dispersion["minimum"] == pytest.approx(.1)
    assert dispersion["maximum"] == pytest.approx(.123)
    assert dispersion["population_standard_deviation"] == pytest.approx(
        ((24**2 - 1) / 12) ** .5 * .001)
    assert "calibration" not in result
    assert "challenger_cells" not in json.dumps(result)  # Refer to, do not duplicate, the artifact.
    assert case.source.objects.get(case.reference["object"]).body == before
    assert b"source_proofs" not in case.payload and b"model_path" not in case.payload
    assert (body["context"]["claim_cas_interval_ms"] is not None) == case.has_claim_evidence
    assert b"NEVER-PUBLISH-MODEL-OR-SECRET" not in case.payload
    if case.has_claim_evidence:
        claim = body["metrics"]["claim_evidence"]
        assert len(claim["selection"]["candidates"]) == 2
        assert claim["selection"]["candidates"][1]["s_eligible"] is False
        assert body["context"]["claim_cas_interval_ms"][0] == claim["selection"]["captured_at_ms"]
        assert body["context"]["claim_sha256"] == hashlib.sha256(canonical_json(claim)).hexdigest()
    with pytest.raises(ValueError, match="not due"):
        case.source.public_stage_audit(case.reference, now_ms=case.due - 1)


def disclosed_plan(plan):
    from validator.production.batches import public_selected_inputs_document

    return canonical_json({
        "format": "veda-batch-disclosure-v3", "plan": plan.document(),
        "public_exact_object": f"batches/{plan.exact.sha256}/manifest.json",
        "public_dense_references": [{
            "rollout_id": row.rollout_id,
            **{kind: {"object": f"batch_assets/{item.sha256}/manifest.json",
                      "sha256": item.sha256, "size_bytes": item.size_bytes}
               for kind, item in (("video", row.video), ("latent", row.latent))},
        } for row in plan.dense_references],
        "public_selected_inputs": public_selected_inputs_document(plan.sample_cache),
    })


def test_public_audit_authenticates_case_design_only_with_original_disclosure(audit_case):
    case = audit_case
    disclosure = disclosed_plan(case.plan)
    result = verify_public_score_audit(case.payload, canonical_json(case.record.document),
                                      case.track, disclosure=disclosure)
    assert result["verification_scope"] == "statistics-and-authenticated-plan-design"
    assert result["model_inference_reproduced"] is False
    changed = json.loads(disclosure)
    changed["plan"]["completed_at_ms"] += 1
    with pytest.raises(ValueError, match="authenticated READY"):
        verify_public_score_audit(case.payload, canonical_json(case.record.document),
                                  case.track, disclosure=canonical_json(changed))
    # Even rehashing a self-consistent score audit cannot change original source grouping.
    body, match = json.loads(case.payload), copy.deepcopy(case.record.document)
    for row in body["metrics"]["reproducibility"]["case_order"]:
        row["source_id"] += "-replacement"
    payload = canonical_json(body)
    match["evaluation"]["stages"][0]["stage_evidence"]["public_audit"] = {
        **stage_reference(case.reference["public_audit"]["object"], payload),
        "not_before_ms": case.due,
    }
    assert verify_public_score_audit(payload, canonical_json(match), case.track)[
        "verification_scope"] == "statistics-only"
    with pytest.raises(ValueError, match="authenticated READY plan"):
        verify_public_score_audit(payload, canonical_json(match), case.track,
                                  disclosure=disclosure)


@pytest.mark.parametrize("audit_case", [(True, False)], indirect=True)
def test_public_cpu_audit_recomputes_retained_champion_at_strict_gate(audit_case):
    case = audit_case
    result = verify_public_score_audit(case.payload, canonical_json(case.record.document),
                                      case.track)
    assert result["gain_lower_bound"] == 0 and result["outcome"] == "CHAMPION_RETAINED"
    assert result["cell_diagnostics"]["descriptive_unweighted_cell_differences"] == {
        "minimum": 0, "maximum": 0, "population_standard_deviation": 0,
    }


@pytest.mark.parametrize("audit_case", [True], indirect=True)
@pytest.mark.parametrize("damage", ["cell", "missing_cells", "plan"])
def test_cell_diagnostics_run_only_after_original_validation(audit_case, monkeypatch, damage):
    from validator import reproducibility

    case = audit_case
    body, match = json.loads(case.payload), copy.deepcopy(case.record.document)
    disclosure = disclosed_plan(case.plan)
    if damage == "plan":
        document = json.loads(disclosure)
        document["plan"]["completed_at_ms"] += 1
        disclosure = canonical_json(document)
    else:
        if damage == "cell":
            body["metrics"]["reproducibility"]["challenger_cells"][0] -= .1
        else:
            del body["metrics"]["reproducibility"]["challenger_cells"]
        match["evaluation"]["stages"][0]["stage_evidence"]["public_audit"] = {
            **stage_reference(case.reference["public_audit"]["object"], canonical_json(body)),
            "not_before_ms": case.due,
        }
    monkeypatch.setattr(reproducibility.math, "sqrt",
                        lambda *_a: pytest.fail("diagnostics ran before evidence validation"))
    with pytest.raises(ValueError):
        verify_public_score_audit(canonical_json(body), canonical_json(match), case.track,
                                  disclosure=disclosure)


def test_cold_private_stage_keeps_original_public_audit_hash_and_cas_interval(
    audit_case, tmp_path, monkeypatch,
):
    case = audit_case
    from validator.production.inventory import ALLOCATION_OBJECT

    claim = case.metrics.get("claim_evidence", case.metrics["cache_allocation"])
    case.source.objects.create_plan(ALLOCATION_OBJECT, canonical_json({
        "format": "veda-inventory-allocations-v1", "track_spec_digest": case.track.digest,
        "claims": [claim],
    }))
    case.source.objects.create_plan(f"admissions/{case.task.challenge_id}.json", canonical_json({
        "track_spec_digest": case.track.digest, "model_digest": claim["model_digest"],
        "file_sha256": case.task.file_sha256, "challenge_id": case.task.challenge_id,
    }))
    cold = ReadyBatchSource(track=case.track, objects=case.source.objects,
                            cache_dir=tmp_path / "cold")
    monkeypatch.setattr(cold, "stage_binding", case.source.stage_binding)
    monkeypatch.setattr(cold, "_allocated_plan", lambda *_a, **_k: case.plan)
    restored = cold.restore_stage(case.task)
    assert restored.metrics["stage_evidence"] == case.reference
    assert cold.archive_stage(case.task, restored, restored.score_block) == case.reference
    assert cold.public_stage_audit(case.reference, now_ms=case.due + 10000) == case.payload


@pytest.mark.parametrize("damage", ["cell", "threshold", "identity", "order", "runtime"])
def test_public_cpu_audit_rejects_changes_without_gpu(audit_case, damage):
    case = audit_case
    body = json.loads(case.payload)
    if damage == "cell":
        body["metrics"]["reproducibility"]["challenger_cells"][0] = .2
    elif damage == "threshold":
        body["metrics"]["promotion"]["minimum_gain"] = .001
    elif damage == "identity":
        body["challenge_id"] = "00" * 32
    elif damage == "order":
        body["metrics"]["reproducibility"]["case_order"].reverse()
    else:
        body["metrics"]["reproducibility"]["execution"]["working_dtype"] = "float16"
    document = copy.deepcopy(case.record.document)
    if damage in {"cell", "runtime"}:
        # Even a replacement digest cannot excuse wrong statistics/backend.
        document["evaluation"]["stages"][0]["stage_evidence"]["public_audit"] = {
            **stage_reference(case.reference["public_audit"]["object"], canonical_json(body)),
            "not_before_ms": case.due,
        }
    with pytest.raises(ValueError):
        verify_public_score_audit(canonical_json(body), canonical_json(document),
                                  case.track)


def test_public_cpu_audit_cannot_replace_original_s_window_even_with_a_new_digest(audit_case):
    case = audit_case
    body, document = json.loads(case.payload), copy.deepcopy(case.record.document)
    body["metrics"]["cache_completed_at_ms"] = 200_001  # Original S timestamp is 200000.
    body["context"]["not_before_ms"] = 200_001 + case.track.cache.disclosure_delay_seconds * 1000
    for batch in (document["evaluation"]["evaluation_batch"],
                  document["evaluation"]["stages"][0]["evaluation_batch"]):
        batch["completed_at_ms"] = 200_001
    document["evaluation"]["stages"][0]["stage_evidence"]["public_audit"] = {
        **stage_reference(case.reference["public_audit"]["object"], canonical_json(body)),
        "not_before_ms": body["context"]["not_before_ms"],
    }
    with pytest.raises(ValueError, match="batch or decision"):
        verify_public_score_audit(canonical_json(body), canonical_json(document), case.track)


def test_public_audit_cli_is_read_only_cpu_and_needs_no_role(audit_case, tmp_path, capsys):
    from cli import main

    case = audit_case
    paths = [tmp_path / name for name in ("track.json", "match.json", "audit.json")]
    for path, body in zip(paths, (canonical_json({"track_spec_digest": case.track.digest,
                                                 "spec": case.raw}),
                                  canonical_json(case.record.document),
                                  case.payload), strict=True):
        path.write_bytes(body)
    main(["verify-match", "--track", str(paths[0]), "--match", str(paths[1]),
          "--artifact", str(paths[2])])
    result = json.loads(capsys.readouterr().out)
    assert result["gain_lower_bound"] == case.record.score_gain
    assert result["verification_scope"] == "statistics-only"
    assert result["cell_diagnostics"]["bootstrap"]["confidence"] == (
        case.track.scoring.match_confidence)
    disclosure = tmp_path / "batch.json"
    disclosure.write_bytes(disclosed_plan(case.plan))
    main(["verify-match", "--track", str(paths[0]), "--match", str(paths[1]),
          "--artifact", str(paths[2]), "--batch-disclosure", str(disclosure)])
    assert json.loads(capsys.readouterr().out)[
        "verification_scope"] == "statistics-and-authenticated-plan-design"
    paths[0].write_bytes(canonical_json(case.raw))  # Same immutable rule object used by full audit.
    main(["verify-match", "--track", str(paths[0]), "--match", str(paths[1]),
          "--artifact", str(paths[2])])
    assert json.loads(capsys.readouterr().out)["gain_lower_bound"] == case.record.score_gain
    paths[0].write_bytes(canonical_json({"track_spec_digest": "00" * 32, "spec": case.raw}))
    with pytest.raises(SystemExit) as stopped:
        main(["verify-match", "--track", str(paths[0]), "--match", str(paths[1]),
              "--artifact", str(paths[2])])
    assert stopped.value.code == 1 and "digest" in capsys.readouterr().err


@pytest.mark.parametrize("origin", ["present", "missing", "corrupt", "legacy"])
def test_full_audit_mixed_rules_uses_original_spec_or_reports_unavailable(
    audit_case, monkeypatch, capsys, origin,
):
    import httpx

    import cli as cli_module
    from cli import main
    from scoring.spec import parse_track_spec
    from tests.validator.test_tournament import challenge
    from validator.production.commitment import Artifact, ArtifactManifest
    from validator.publication import (
        HISTORICAL_MATCHES_PREFIX,
        _summary_document,
        build_genesis_artifacts,
        encode_weight_artifact,
    )
    from validator.tournament import genesis_champion, resolve_tournament
    from validator.weights import WeightArtifact, WeightChainState, WeightPlan

    case = audit_case
    if origin == "legacy":
        document = copy.deepcopy(case.record.document)
        for stage in document["evaluation"]["stages"]:
            stage.pop("stage_evidence", None)
        case.record = replace(case.record, document=document)
    original = canonical_json(case.record.document)
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + case.record.object_name,
        hashlib.sha256(original).hexdigest(), len(original), case.record.object_name,
    ),))
    raw = copy.deepcopy(case.raw)
    raw.update(track_id="new-audit-rules", track_start_block=case.item["score_block"] + 100,
               artifact_manifest_sha256=manifest.digest)
    raw["scoring"]["bootstrap_seed"] += 99  # The original statistics must use the old seed.
    current = parse_track_spec(raw)
    failure = challenge(2, "new-hotkey", state=ChallengeState.PARTICIPANT_FAILED,
                        error_code="MODEL_SCHEMA_INVALID")
    failure.update(block_height=current.track_start_block + 1,
                   score_block=current.track_start_block + 2)
    tournament = resolve_tournament(
        [failure], track_id=current.track_id, track_spec_digest=current.digest,
        netuid=current.netuid, minimum_gain=current.full_video.minimum_gain,
        minimum_exact_gain=current.scoring.minimum_frontier_gain, promotion=current.promotion,
        initial_champion=genesis_champion(current), historical_matches=(case.record,),
    )
    block = current.track_start_block + 10
    chain = WeightChainState(block, {current.initial_champion.hotkey: 0,
                                    case.item["hotkey"]: 3, "new-hotkey": 4}, 1, 1.0, 1, {})
    public = build_genesis_artifacts(current, chain, version_key=0,
                                     runtime_manifest=manifest.encode(),
                                     historical_matches=(case.record,))
    new_match = canonical_json(tournament.matches[0].document)
    weights = WeightArtifact(
        current.track_id, current.digest, current.netuid, current.mechid, 0, block,
        WeightPlan(block, (case.item["hotkey"],), (3,), (1.0,)), match_sequence=1,
        terminal_match_sha256=hashlib.sha256(new_match).hexdigest(),
    )
    objects = {
        "track.json": public.track, "weights.json": encode_weight_artifact(weights),
        "summary.json": canonical_json(_summary_document(current, block, tournament)),
        f"runtime/{manifest.digest}/manifest.json": manifest.encode(),
        case.record.object_name: original, tournament.matches[0].object_name: new_match,
        case.reference["public_audit"]["object"]: case.payload,
        f"batch_groups/{case.plan.plan_digest}.json": disclosed_plan(case.plan),
    }
    spec_name = f"artifacts/{case.track.digest}"
    if origin != "missing":
        objects[spec_name] = (canonical_json(case.raw) if origin in {"present", "legacy"}
                              else canonical_json(raw))
    requests = []

    def get(request):
        assert request.method == "GET" and "authorization" not in request.headers
        name = request.url.path.removeprefix("/tracks/main/")
        requests.append(name)
        body = objects.get(name)
        return (httpx.Response(404) if body is None else httpx.Response(
            200, content=body, headers={"ETag": hashlib.sha256(body).hexdigest()}))

    client = httpx.Client
    monkeypatch.setattr(cli_module.httpx, "Client", lambda **kwargs: client(
        transport=httpx.MockTransport(get), **kwargs))
    if origin == "corrupt":
        with pytest.raises(SystemExit) as stopped:
            main(["audit", "--public-base-url", "https://public.example/tracks/main"])
        assert stopped.value.code == 1 and "frozen specification bytes" in capsys.readouterr().err
    else:
        main(["audit", "--public-base-url", "https://public.example/tracks/main"])
        output = json.loads(capsys.readouterr().out)
        assert output["historical_matches"] == output["committed_matches"] == 1
        if origin == "present":
            assert len(output["score_audits"]) == 1 and output["unavailable_score_audits"] == []
            assert output["score_audits"][0]["gain_lower_bound"] == case.record.score_gain
            assert output["score_audits"][0]["cell_diagnostics"]["bootstrap"] == (
                json.loads(case.payload)["metrics"]["reproducibility"]["bootstrap"])
        elif origin == "legacy":
            assert output["score_audits"] == output["unavailable_score_audits"] == []
            assert output["unknown_stage_evidence"] == 1
            assert case.reference["public_audit"]["object"] not in requests
        else:
            assert output["score_audits"] == []
            assert output["unavailable_score_audits"] == [{
                "challenge_id": case.task.challenge_id, "stage": "SCORING",
                "track_spec_digest": case.track.digest, "reason": "frozen_spec_unavailable",
            }]
    assert requests.count(spec_name) == (0 if origin == "legacy" else 1)
    assert objects[case.record.object_name] == original  # No history rewrite or re-evaluation.


def test_public_audit_disclosure_requires_r_match_fence_and_is_idempotent(audit_case, monkeypatch):
    import httpx

    from tests.validator.test_publication import MemoryS3 as PublicS3
    from validator.publication import PublicArtifactError, S3Repository, encode_weight_artifact
    from validator.weights import WeightArtifact, WeightPlan

    case = audit_case
    client = PublicS3()
    repository = S3Repository(client, bucket="public", prefix="test")
    with pytest.raises(PublicArtifactError, match="premature"):
        asyncio.run(repository.publish_score_audit(case.payload, track=case.track,
                                                    now_ms=case.due - 1))
    assert not asyncio.run(repository.publish_score_audit(case.payload, track=case.track,
                                                          now_ms=case.due))
    match_payload = canonical_json(case.record.document)
    client.objects["test/" + case.record.object_name] = match_payload
    fence = WeightArtifact(
        case.track.track_id, case.track.digest, case.track.netuid, case.track.mechid, 0, 500,
        WeightPlan(500, (case.item["hotkey"],), (1,), (1.0,)), match_sequence=1,
        terminal_match_sha256=hashlib.sha256(match_payload).hexdigest())
    client.objects["test/weights.json"] = encode_weight_artifact(replace(
        fence, terminal_match_sha256="ab" * 32))
    with pytest.raises(PublicArtifactError, match="fence"):
        asyncio.run(repository.publish_score_audit(case.payload, track=case.track, now_ms=case.due))
    client.objects["test/weights.json"] = encode_weight_artifact(replace(
        fence, source_block=200, plan=WeightPlan(200, (case.item["hotkey"],), (1,), (1.0,))))
    with pytest.raises(PublicArtifactError, match="fence"):
        asyncio.run(repository.publish_score_audit(case.payload, track=case.track, now_ms=case.due))
    client.objects["test/weights.json"] = encode_weight_artifact(fence)
    before = case.source.objects.get(case.reference["object"]).body
    committed = dict(client.objects)
    assert asyncio.run(repository.publish_score_audit(case.payload, track=case.track,
                                                      now_ms=case.due))
    # Publication exposes actual public bytes, not a private-only cells reference.
    group_name = f"batch_groups/{case.plan.plan_digest}.json"
    client.objects["test/" + group_name] = disclosed_plan(case.plan)

    def get(request):
        assert request.method == "GET" and "authorization" not in request.headers
        name = request.url.path.removeprefix("/tracks/main/")
        body = client.objects["test/" + name]
        return httpx.Response(200, content=body, headers={"ETag": hashlib.sha256(body).hexdigest()})

    with httpx.Client(transport=httpx.MockTransport(get)) as http:
        public = S3Repository.from_public_http(http, base_url="https://public.example/tracks/main")
        result = verify_public_score_audit(
            public._get_optional(case.reference["public_audit"]["object"]),
            public._get_optional(case.record.object_name), case.track,
            disclosure=public._get_optional(group_name),
        )
    assert result["verification_scope"] == "statistics-and-authenticated-plan-design"
    assert result["gain_lower_bound"] == case.record.score_gain
    assert result["model_inference_reproduced"] is False
    assert all(client.objects[key] == value for key, value in committed.items())
    assert case.source.objects.get(case.reference["object"]).body == before
    cold = S3Repository(client, bucket="public", prefix="test")
    assert not asyncio.run(cold.publish_score_audit(case.payload, track=case.track,
                                                 now_ms=case.due))
    assert [call["Key"] for call in client.calls] == ["test/" + case.reference[
        "public_audit"]["object"]]
    from validator.run import _publish_due_score_audit

    cold = S3Repository(client, bucket="public", prefix="test")
    original_get = cold._get_optional
    receipt_reads = []

    def receipt_only(name):
        assert name == case.reference["public_audit"]["object"]
        receipt_reads.append(name)
        return original_get(name)

    monkeypatch.setattr(cold, "_get_optional", receipt_only)
    monkeypatch.setattr(case.source, "public_stage_audit",
                        lambda *_a, **_k: pytest.fail("redundant private stage read"))
    asyncio.run(_publish_due_score_audit(case.source, cold, challenges=[case.item],
                                        now_ms=case.due))
    assert len(receipt_reads) == 1  # Cold receipt verifies bytes without rewalking match history.
    monkeypatch.setattr(cold, "_get_optional",
                        lambda *_a: pytest.fail("confirmed audit should use its existing cache"))
    asyncio.run(_publish_due_score_audit(case.source, cold, challenges=[case.item],
                                        now_ms=case.due))
    assert len(receipt_reads) == 1
    client.objects["test/" + case.reference["public_audit"]["object"]] = b"corrupt receipt"
    with pytest.raises(PublicArtifactError, match="receipt differs"):
        asyncio.run(S3Repository(client, bucket="public", prefix="test").score_audit_published(
            case.reference))


def test_cold_audit_receipt_scan_is_bounded_and_progresses_without_a_cursor():
    from validator.run import _publish_due_score_audit

    confirmed, reads = set(), []

    class Publisher:
        def score_audit_confirmed(self, reference):
            return reference["object"] in confirmed

        async def score_audit_published(self, reference, **_kwargs):
            reads.append(reference["object"])
            confirmed.add(reference["object"])
            return True

        async def publish_score_audit(self, *_a, **_k):
            pytest.fail("already public history started a new upload")

    challenges = [{"state": "SUCCEEDED", "stages": [{"stage_evidence": {
        "object": str(i), "public_audit": {"not_before_ms": 1},
    }}]} for i in range(20)]
    source = SimpleNamespace(track=None, public_stage_audit=lambda *_a, **_k:
                             pytest.fail("already public history read private evidence"))
    publisher = Publisher()
    for expected in (8, 16, 20, 20):
        asyncio.run(_publish_due_score_audit(source, publisher, challenges=challenges, now_ms=2))
        assert len(reads) == expected
    assert len(set(reads)) == 20  # Warm cache skips all old objects without network IO.


@pytest.mark.parametrize("committed", [False, True])
def test_one_unfinished_audit_per_pass_even_before_match_commits(committed):
    from validator.run import _publish_due_score_audit

    calls = []

    class Publisher:
        def score_audit_confirmed(self, _reference):
            return False

        async def score_audit_published(self, *_a, **_k):
            calls.append("receipt")
            return False

        async def publish_score_audit(self, *_a, **_k):
            calls.append("publish")
            return committed

    def read(*_a, **_k):
        calls.append("private")
        return b"sealed original"

    source = SimpleNamespace(track=None, public_stage_audit=read)
    item = {"state": "SUCCEEDED", "stages": [{"stage_evidence": {
        "public_audit": {"not_before_ms": 1},
    }}]}
    asyncio.run(_publish_due_score_audit(source, Publisher(), challenges=[item] * 10, now_ms=2))
    assert calls == ["receipt", "private", "publish"]


@pytest.mark.parametrize("failure", [None, "audit", "bulk"])
def test_existing_disclosure_loop_bounds_late_audits_and_bulk_failures_are_independent(
    audit_case, monkeypatch, failure,
):
    from validator.run import _publish_due_batch

    case = audit_case
    calls = []
    source = SimpleNamespace(
        track=case.track, data_track=case.track, objects=case.source.objects,
        protected_plan_digests=lambda *_a: frozenset(), prune_disclosed=lambda *_a: 0,
        # A late stage must publish even when its data was already disclosed.
        due_plan_digests=lambda _now: () if failure is None else ("de" * 32,),
        public_stage_audit=case.source.public_stage_audit,
        public_disclosure=lambda *_a, **_k: "original batch", record_disclosed=lambda *_a:
        calls.append("batch-receipt"),
    )

    class Publisher:
        def score_audit_confirmed(self, *_a):
            return False

        async def score_audit_published(self, *_a, **_k):
            return False

        async def publish_score_audit(self, payload, **_kwargs):
            assert payload == case.payload
            calls.append("audit")
            if failure == "audit":
                raise OSError("small audit unavailable")
            return True

        async def publish_disclosure(self, *_a, **_k):
            calls.append("batch")
            if failure == "bulk":
                raise OSError("bulk unavailable")

    monkeypatch.setattr(time, "time_ns", lambda: case.due * 1_000_000)
    execute = _publish_due_batch(source, Publisher(), finalized_block=500,
                                 challenges=[case.item, case.item])
    if failure:
        with pytest.raises(OSError):
            asyncio.run(execute)
    else:
        assert asyncio.run(execute) is None
    assert calls.count("audit") == 1  # Same pass cannot start two new audit uploads.
    if failure == "audit":
        assert calls == ["audit", "batch", "batch-receipt"]
    if failure == "bulk":
        assert calls == ["audit", "batch"]
