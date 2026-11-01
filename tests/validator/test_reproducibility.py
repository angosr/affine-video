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


@pytest.fixture(params=[False, True], ids=["legacy-claim-unknown", "first-cas-evidence"])
def audit_case(tmp_path, monkeypatch, request):
    from scoring.spec import parse_track_spec
    from tests.scoring.test_spec import valid_spec
    from tests.validator.test_tournament import tenure_challenge
    from validator.tournament import genesis_champion, resolve_tournament

    raw = valid_spec()
    raw["promotion"] = {"format": "veda-exact-tenure-v1", "minimum_gain": .002,
                        "decay_seconds": 604800}
    raw["scoring"].update(minimum_frontier_gain=.003, match_confidence=.9975,
                           bootstrap_resamples=4000)
    raw["full_video"]["minimum_gain"] = .003
    track = parse_track_spec(raw)
    scoring = track.scoring
    design = [CaseDesign(f"route-{r}", "all", f"rollout-{i}", f"source-{i}", step, window)
              for i in range(4) for r in range(2)
              for step, window in enumerate(scoring.step_window_weight_map)]
    has_claim, won = request.param if isinstance(request.param, tuple) else (request.param, True)
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
                            submitted_ms=2000)
    item.update(challenge_id="34" * 32, submission_id="45" * 32, file_sha256="12" * 32,
                final_score=paired.challenger_score, incumbent_score=paired.incumbent_score)
    batch = {**item["evaluation_batch"], "generated_block": 95,
             "completed_block": 98, "completed_at_ms": 1000, "ready_sha256": "56" * 32}
    allocation = dict(ordinal=1, challenge_id=item["challenge_id"], stage="SCORING",
                      plan_digest=batch["plan_digest"], ready_sha256=batch["ready_sha256"],
                      file_sha256=item["file_sha256"], model_digest=item["model_digest"],
                      submission_block=item["block_height"], submission_timestamp_ms=2000)
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
    if has_claim:
        from validator.production.inventory import _pool_digest

        candidates = [{
            "plan_digest": batch["plan_digest"], "plan_sha256": batch["ready_sha256"],
            "generation_block": 95, "state": "READY", "completed_block": 98,
            "completed_at_ms": 1000, "inventory_slot": 0,
            "source_selection_sha256": "78" * 32, "s_eligible": True,
            "slot_eligible": True, "families_eligible": True,
        }, {
            "plan_digest": "89" * 32, "plan_sha256": "9a" * 32,
            "generation_block": 150, "state": "READY", "completed_block": 151,
            "completed_at_ms": 3000, "inventory_slot": 1,
            "source_selection_sha256": "ab" * 32, "s_eligible": False,
            "slot_eligible": True, "families_eligible": True,
        }]
        metrics["claim_evidence"] = {**allocation, "selection": {
            "algorithm": "earliest-expiry-stable-v1", "source_sha256": "bc" * 32,
            "cas_attempt": 2, "captured_at_ms": 4000, "clock": "local-before-successful-cas",
            "pool_read_interval_ms": [3000, 3500],
            "disclosure_delay_seconds": track.cache.disclosure_delay_seconds,
            "previous_authority_sha256": hashlib.sha256(canonical_json({
                "format": "veda-inventory-allocations-v1", "track_spec_digest": track.digest,
                "claims": [],
            })).hexdigest(), "pool_sha256": _pool_digest(candidates),
            "candidates": candidates,
        }}
    metrics["private_payload"] = "NEVER-PUBLISH-MODEL-OR-SECRET"
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="original")
    source = ReadyBatchSource(track=track, objects=objects, cache_dir=tmp_path)
    monkeypatch.setattr(source, "stage_binding", lambda *_a, **_k: {
        key: metrics[key] for key in ("cache_ready_sha256", "cache_completed_block",
                                     "cache_completed_at_ms", "cache_allocation", "claim_evidence")
        if key in metrics})
    task = SimpleNamespace(challenge_id=item["challenge_id"], submission_id=item["submission_id"],
                           hotkey=item["hotkey"], repo_id=item["repo_id"],
                           revision=item["revision"],
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
                           has_claim_evidence=has_claim, task=task, metrics=metrics)


def test_public_cpu_audit_is_bound_to_actual_match_and_preserves_original_stage(audit_case):
    case = audit_case
    before = case.source.objects.get(case.reference["object"]).body
    result = verify_public_score_audit(case.payload, canonical_json(case.record.document),
                                      case.track)
    assert result["case_count"] == 24 and result["outcome"] == "CHALLENGER_WON"
    assert case.source.objects.get(case.reference["object"]).body == before
    assert b"source_proofs" not in case.payload and b"model_path" not in case.payload
    body = json.loads(case.payload)
    assert (body["context"]["claim_cas_interval_ms"] is not None) == case.has_claim_evidence
    assert b"NEVER-PUBLISH-MODEL-OR-SECRET" not in case.payload
    if case.has_claim_evidence:
        claim = body["metrics"]["claim_evidence"]
        assert len(claim["selection"]["candidates"]) == 2
        assert claim["selection"]["candidates"][1]["s_eligible"] is False
        assert body["context"]["claim_cas_interval_ms"][0] == 4000
        assert body["context"]["claim_sha256"] == hashlib.sha256(canonical_json(claim)).hexdigest()
    with pytest.raises(ValueError, match="not due"):
        case.source.public_stage_audit(case.reference, now_ms=case.due - 1)


@pytest.mark.parametrize("audit_case", [(True, False)], indirect=True)
def test_public_cpu_audit_recomputes_retained_champion_at_strict_gate(audit_case):
    case = audit_case
    result = verify_public_score_audit(case.payload, canonical_json(case.record.document),
                                      case.track)
    assert result["gain_lower_bound"] == 0 and result["outcome"] == "CHAMPION_RETAINED"


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
    monkeypatch.setattr(cold, "_allocated_plan", lambda *_a, **_k: SimpleNamespace(
        generation_block=case.metrics["cache_generated_block"],
        plan_digest=case.metrics["cache_plan_digest"],
        block_hash=case.metrics["cache_seed_block_hash"],
        exact=SimpleNamespace(sha256=case.metrics["cache_sha256"])))
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
    body["metrics"]["cache_completed_at_ms"] = 2500  # Original S timestamp is 2000.
    body["context"]["not_before_ms"] = 2500 + case.track.cache.disclosure_delay_seconds * 1000
    for batch in (document["evaluation"]["evaluation_batch"],
                  document["evaluation"]["stages"][0]["evaluation_batch"]):
        batch["completed_at_ms"] = 2500
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
    main(["evaluator", "audit", "--track", str(paths[0]), "--match", str(paths[1]),
          "--artifact", str(paths[2])])
    assert json.loads(capsys.readouterr().out)["gain_lower_bound"] == case.record.score_gain
    paths[0].write_bytes(canonical_json({"track_spec_digest": "00" * 32, "spec": case.raw}))
    with pytest.raises(SystemExit) as stopped:
        main(["evaluator", "audit", "--track", str(paths[0]), "--match", str(paths[1]),
              "--artifact", str(paths[2])])
    assert stopped.value.code == 1 and "digest" in capsys.readouterr().err


def test_public_audit_disclosure_requires_r_match_fence_and_is_idempotent(audit_case):
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
    assert asyncio.run(repository.publish_score_audit(case.payload, track=case.track,
                                                      now_ms=case.due))
    cold = S3Repository(client, bucket="public", prefix="test")
    assert not asyncio.run(cold.publish_score_audit(case.payload, track=case.track,
                                                 now_ms=case.due))
    assert [call["Key"] for call in client.calls] == ["test/" + case.reference[
        "public_audit"]["object"]]


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
