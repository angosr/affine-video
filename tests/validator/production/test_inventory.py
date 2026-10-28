import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scoring.spec import parse_track_spec, track_spec_dict
from tests.scoring.test_spec import rules_for_data, valid_spec
from tests.validator.test_store import submission_at
from validator.production.batches import ReadyBatchSource
from validator.production.commitment import (
    Artifact,
    ArtifactManifest,
    SourceLeaf,
    SourcePool,
    canonical_json,
)
from validator.production.inventory import ALLOCATION_OBJECT, Inventory
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    build_plan,
    generation_seed,
    select_source_proofs,
    validate_source_selection_sequence,
)
from validator.production.service import PlanCoordinator
from validator.production.storage import PrivateObjectStore
from validator.queue import ChallengeState
from validator.release import verify_competition_manifest
from validator.scoring import ScoringSettings, StageOutcome, process_next_scoring
from validator.store import ChallengeStore

from .test_storage import MemoryS3


@pytest.fixture
def inventory_case():
    pool = SourcePool(tuple(SourceLeaf(
        "production" if index < 24 else "audit",
        f"family-{index}", f"source-{index}", "prompt", (), "all",
        "test-license", f"{index:064x}",
    ) for index in range(26)))
    raw = valid_spec()
    raw["source_commitments"] = {partition: {"all": [
        row["sha256"] for row in pool.public_commitments() if row["partition"] == partition
    ]} for partition in ("production", "audit")}
    raw["source_pool_merkle_root"] = pool.root
    track = parse_track_spec(raw)
    objects = PrivateObjectStore(MemoryS3(), bucket="test", prefix="inventory")
    return SimpleNamespace(track=track, pool=pool, objects=objects)


def make_plan(case, block, slot=0, *, ready=True):
    secret = f"{block:064x}"
    block_hash = "0x" + f"{block + 1000:064x}"
    proofs = select_source_proofs(
        case.pool, commitments=case.track.source_commitments, partition="production",
        strata=("all",), sources_per_stratum=4,
        batch_seed=generation_seed(case.track.digest, block_hash, secret),
        inventory_slot=slot, excluded_source_digests=frozenset(),
    )
    plan = build_plan(
        track_spec_digest=case.track.digest, generation_block=block, block_hash=block_hash,
        secret=secret, inventory_slot=slot, source_proofs=proofs, scoring=case.track.scoring,
        available_routes=case.track.model.selector.routes,
    )
    if ready:
        item = ReadyObject("exact/file.npz", "23" * 32, 100)
        plan = plan.ready(
            completed_at_ms=(plan.generation_block + 1) * 1000,
            completed_block=plan.generation_block + 1,
            exact=item,
            sample_cache=ReadySampleCache(
                replace(item, object_key="inputs/index.json"),
                replace(item, object_key="inputs/text.safetensors"),
            ),
            dense_references=tuple(
                DenseReference(
                    rollout.rollout_id,
                    replace(item, object_key="videos/file.mp4"),
                    replace(item, object_key="latents/file.safetensors"),
                )
                for rollout in plan.rollouts
            ),
        )
    return plan


def claim(inventory, plans, identifier="a1" * 32, stage="SCORING", **kwargs):
    return inventory.claim(
        plans=plans, challenge_id=identifier, stage=stage,
        file_sha256=kwargs.get("file_sha256", "b2" * 32), model_digest="c3" * 32,
        submission_block=kwargs.get("submission_block", 200),
        submission_timestamp_ms=kwargs.get(
            "submission_timestamp_ms", kwargs.get("submission_block", 200) * 1000,
        ),
        disclosure_delay_seconds=kwargs.get("disclosure_delay_seconds", 1800),
    )


def test_inventory_plan_is_secret_committed_and_unassigned(inventory_case):
    plan = make_plan(inventory_case, 101)
    assert BatchPlan.decode(plan.encode()) == plan
    assert not {"exact_challenge_ids", "confirmation_challenge_ids", "submission_first_block"} & (
        plan.document().keys()
    )
    assert generation_seed(plan.track_spec_digest, plan.block_hash, "ff" * 32) != plan.batch_seed
    with pytest.raises(ValueError, match="commitment"):
        replace(plan, secret="ff" * 32)
    validate_source_selection_sequence(
        (plan,), commitments=inventory_case.track.source_commitments,
        scoring=inventory_case.track.scoring,
    )


def test_runtime_update_reuses_original_ready_and_claim_without_rebinding(inventory_case):
    case = inventory_case
    assets = Artifact("weights/teacher", "ab" * 32, 1, "artifacts/teacher")
    competition = ArtifactManifest((Artifact(
        "runtime/source.zip", "11" * 32, 1, "artifacts/old-source",
    ), assets))
    raw = track_spec_dict(case.track)
    raw["artifact_manifest_sha256"] = competition.digest
    case.track = parse_track_spec(raw)
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    for plan in plans:
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
    original = claim(Inventory(case.objects, case.track.digest), plans)
    before = dict(case.objects._client.objects)
    encoded = tuple(plan.encode() for plan in plans)
    runtime = ArtifactManifest((Artifact(
        "runtime/source.zip", "22" * 32, 1, "artifacts/new-source",
    ), assets))
    assert verify_competition_manifest(
        case.track, runtime.encode(), competition.encode(),
    ) == competition
    cold = Inventory(case.objects, case.track.digest)
    assert claim(cold, plans) == original
    assert case.objects._client.objects == before
    assert tuple(plan.encode() for plan in plans) == encoded
    assert all(plan.completed_at_ms == (plan.generation_block + 1) * 1000 for plan in plans)


def test_new_rules_keep_all_four_ready_seals_and_original_allocation_authority(
    inventory_case, tmp_path, monkeypatch,
):
    from validator.production import batches as batches_module
    from validator.queue import ChainPosition

    case = inventory_case
    raw = track_spec_dict(case.track)
    raw["cache"].update(inventory_target_batches=4, inventory_max_bytes=512 * 1024**3)
    case.track = parse_track_spec(raw)
    plans = tuple(make_plan(case, 101 + index, index % 2) for index in range(4))
    for plan in plans:
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
    previous = claim(Inventory(case.objects, case.track.digest), plans)
    before = {name: case.objects.get(name).body for name in
              case.objects.list_names("plans/") if name != ALLOCATION_OBJECT}
    rules = parse_track_spec(rules_for_data(case.track))
    source = ReadyBatchSource(track=rules, data_track=case.track, objects=case.objects,
                             cache_dir=tmp_path)
    task = SimpleNamespace(challenge_id="d4" * 32, submission_id="e5" * 32, hotkey="miner",
                           repo_id="owner/model", revision="12" * 20,
                           position=ChainPosition(200, 0, 0))
    case.objects.create_plan(source._admission_name(task.challenge_id), canonical_json({
        "format": "veda-admission-model-v1", **source._admission_identity(task),
        "file_sha256": "b2" * 32, "model_digest": "c3" * 32, "size_bytes": 100,
        "submission_timestamp_ms": 200_000,
    }))
    monkeypatch.setattr(batches_module, "_submission_timestamp_ms", lambda block: block * 1000)
    assert source.eligible_submission(task)
    assert source.claim_ready(task.challenge_id, confirmation=False, file_sha256="b2" * 32,
                              model_digest="c3" * 32, submission_block=200)
    with pytest.raises(RuntimeError, match="disabled"):
        source.claim_ready(task.challenge_id, confirmation=True, file_sha256="b2" * 32,
                           model_digest="c3" * 32, submission_block=200)
    cold = ReadyBatchSource(track=rules, data_track=case.track, objects=case.objects,
                           cache_dir=tmp_path / "cold")
    _, authority = cold.inventory.read()
    assert authority["track_spec_digest"] == case.track.digest
    assert authority["claims"][0] == previous
    assert len(authority["claims"]) == 2
    assert cold.stage_binding(task.challenge_id, confirmation=False) == source.stage_binding(
        task.challenge_id, confirmation=False,
    )
    task.state, task.file_sha256 = ChallengeState.SCORING, "b2" * 32
    plan = cold._allocated_plan(task.challenge_id, confirmation=False)
    metrics = {**cold.stage_binding(task.challenge_id, confirmation=False),
               "cache_generated_block": plan.generation_block,
               "cache_plan_digest": plan.plan_digest,
               "cache_seed_block_hash": plan.block_hash, "cache_sha256": plan.exact.sha256}
    cold.archive_stage(task, StageOutcome(.8, metrics), 210)
    assert source.restore_stage(task).score == .8
    archived = json.loads(case.objects.get(source._stage_name(task.challenge_id, "SCORING")).body)
    assert archived["track_spec_digest"] == rules.digest
    assert metrics["cache_ready_sha256"] == hashlib.sha256(plan.encode()).hexdigest()
    original = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path / "old")
    assert original.due_plan_digests(10**12) == cold.due_plan_digests(10**12)
    assert [p.encode() for _, p in cold._plans()] == [p.encode() for p in plans]
    assert {name: case.objects.get(name).body for name in before} == before
    assert cold.inventory_status(plans)["max_bytes"] == 768 * 1024**3
    coordinator = PlanCoordinator(track=rules, data_track=case.track, store=SimpleNamespace(
        finalized_block_hash=lambda _block: None,
    ), objects=case.objects, source_pool=case.pool)
    assert coordinator.create_next(now_ms=110_000, finalized_head=(110, "ab" * 32)) is None
    assert tuple(p for _, _, p in coordinator.plans()) == plans
    replacement = coordinator.create_next(disk_bytes=512 * 1024**3, now_ms=10**12,
                                           finalized_head=(110, "ab" * 32))
    assert replacement is not None and replacement.track_spec_digest == case.track.digest
    assert {name: case.objects.get(name).body for name in before} == before


def test_rule_upgrade_never_moves_original_ready_clock_or_submission_window(
    inventory_case, tmp_path, monkeypatch,
):
    from validator.production import batches as batches_module
    from validator.production.storage import ARTIFACT_MANIFEST_OBJECT

    case = inventory_case
    manifest = ArtifactManifest((Artifact("weights/teacher", "ab" * 32, 1, "artifacts/model"),))
    raw = track_spec_dict(case.track)
    raw["artifact_manifest_sha256"] = manifest.digest
    raw["cache"].update(disclosure_delay_seconds=86400, inventory_target_batches=4,
                         inventory_max_bytes=512 * 1024**3)
    case.track = parse_track_spec(raw)
    case.objects.create_plan(ARTIFACT_MANIFEST_OBJECT, manifest.encode())
    original = ReadyBatchSource(track=case.track, objects=case.objects,
                                cache_dir=tmp_path / "old", batch_dir=tmp_path / "content")
    plans, content = [], {}
    for index in range(4):
        payload = f"original immutable cache {index}".encode()
        path = tmp_path / f"original-{index}.npz"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        original.content.put(path, sha256=digest, size_bytes=len(payload))
        plan = replace(make_plan(case, 101 + index, index % 2), exact=ReadyObject(
            f"exact/{digest}.npz", digest, len(payload),
        ))
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
        plans.append(plan)
        content[digest] = payload
    plans = tuple(plans)
    sealed = {plan.plan_digest: plan.encode() for plan in plans}
    bootstrap = case.objects.get(ARTIFACT_MANIFEST_OBJECT).body
    rules = parse_track_spec(rules_for_data(case.track))
    upgrade_ms = max(plan.completed_at_ms for plan in plans) + 60_000
    monkeypatch.setattr(batches_module.time, "time_ns", lambda: upgrade_ms * 1_000_000)
    upgraded = ReadyBatchSource(track=rules, data_track=case.track, objects=case.objects,
                                cache_dir=tmp_path / "new", batch_dir=tmp_path / "content")
    assert upgraded.objects is original.objects
    assert upgraded.objects._prefix == original.objects._prefix
    assert upgraded.data_track.digest == case.track.digest != rules.digest
    assert upgraded.data_track.artifact_manifest_sha256 == manifest.digest
    assert (rules.cache.disclosure_delay_seconds
            == case.track.cache.disclosure_delay_seconds == 86400)
    for index, plan in enumerate(plans):
        c, r = plan.completed_at_ms, plan.completed_at_ms + 86_400_000
        assert plan.completed_at_ms != upgrade_ms
        assert plan.inventory_slot == index % 2
        assert hashlib.sha256(upgraded._plan_by_digest(plan.plan_digest).encode()).hexdigest() == (
            hashlib.sha256(sealed[plan.plan_digest]).hexdigest()
        )
        for block, stamp, eligible in (
            (plan.completed_block, c + 1, False),
            (plan.completed_block + 1, c, False),
            (plan.completed_block + 1, c + 1, True),
            (plan.completed_block + 1, r - 1, True),
            (plan.completed_block + 1, r, False),
            (plan.completed_block + 1, r + 1, False),
        ):
            assert Inventory.eligible(plan, block, stamp, 86400) is eligible
            identifier = hashlib.sha256(f"{index}:{block}:{stamp}".encode()).hexdigest()
            result = claim(upgraded.inventory, (plan,), identifier=identifier,
                           submission_block=block, submission_timestamp_ms=stamp,
                           disclosure_delay_seconds=rules.cache.disclosure_delay_seconds)
            assert (result is not None) is eligible
            if result is not None:
                # Upgrade/restart/admission after R cannot move the original S.
                monkeypatch.setattr(batches_module.time, "time_ns",
                                    lambda expiry=r: (expiry + 3_600_000) * 1_000_000)
                assert claim(upgraded.inventory, (plan,), identifier=identifier,
                             submission_block=block, submission_timestamp_ms=stamp,
                             disclosure_delay_seconds=86400) == result
        for now in (r - 1, r, r + 1):
            expected = tuple(p.plan_digest for p in plans
                             if p.completed_at_ms + 86_400_000 <= now)
            assert upgraded.due_plan_digests(now) == original.due_plan_digests(now) == expected
        restored = upgraded.content.get(sha256=plan.exact.sha256, size_bytes=plan.exact.size_bytes)
        assert restored.read_bytes() == content[plan.exact.sha256]
        assert hashlib.sha256(restored.read_bytes()).hexdigest() == plan.exact.sha256
    assert {p.plan_digest: p.encode() for _, p in upgraded._plans()} == sealed
    assert case.objects.get(ARTIFACT_MANIFEST_OBJECT).body == bootstrap == manifest.encode()
    kwargs = {"objects": case.objects, "source_pool": case.pool,
              "store": SimpleNamespace(finalized_block_hash=lambda _block: None)}
    before_upgrade = PlanCoordinator(track=case.track, **kwargs)
    after_upgrade = PlanCoordinator(track=rules, data_track=case.track, **kwargs)
    half_life = min(plan.completed_at_ms for plan in plans) + 43_200_000
    for now in (upgrade_ms, half_life - 1):
        assert before_upgrade.create_next(now_ms=now, finalized_head=(110, "ab" * 32)) is None
        assert after_upgrade.create_next(now_ms=now, finalized_head=(110, "ab" * 32)) is None
    replacement = after_upgrade.create_next(now_ms=half_life, finalized_head=(110, "ab" * 32))
    assert replacement is not None and replacement.state == "PLANNED"
    assert replacement.track_spec_digest == case.track.digest
    assert replacement.inventory_slot == plans[0].inventory_slot
    assert sum(p.state == "PLANNED" for _, _, p in after_upgrade.plans()) == 1
    assert {p.plan_digest: p.encode() for _, p in upgraded._plans()
            if p.plan_digest in sealed} == sealed


@pytest.mark.parametrize("stage", [ChallengeState.SCORING, ChallengeState.CONFIRMING])
def test_preproduced_archive_survives_cold_db_without_bulk(inventory_case, tmp_path, stage):
    case = inventory_case
    plans = (make_plan(case, 150), make_plan(case, 151, 1))
    for plan in plans:
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path / "empty")

    def prepare(store):
        identifier, _ = store.add_submission(submission_at(200))
        store.transition(identifier, ChallengeState.FETCHING)
        store.transition(identifier, ChallengeState.VALIDATING)
        store.transition(identifier, ChallengeState.SCORING)
        if stage is ChallengeState.CONFIRMING:
            store.transition(identifier, stage)
        task = store.claim_next_scoring("test", lease_ms=10, now_ms=20)
        return replace(task, file_sha256="b2" * 32)

    with ChallengeStore(tmp_path / "original.db", track_spec_digest=case.track.digest) as store:
        task = prepare(store)
        identity = {"format": "veda-admission-model-v1", **source._admission_identity(task),
                    "file_sha256": "b2" * 32, "model_digest": "c3" * 32, "size_bytes": 100,
                    "submission_timestamp_ms": 200_000}
        case.objects.create_plan(
            source._admission_name(task.challenge_id), canonical_json(identity),
        )
        claim(source.inventory, plans, task.challenge_id)
        if stage is ChallengeState.CONFIRMING:
            claim(source.inventory, plans, task.challenge_id, "CONFIRMING")
        plan = plans[stage is ChallengeState.CONFIRMING]
        metrics = {
            **source.stage_binding(task.challenge_id,
                                   confirmation=stage is ChallengeState.CONFIRMING),
            "cache_generated_block": plan.generation_block, "cache_plan_digest": plan.plan_digest,
            "cache_seed_block_hash": plan.block_hash, "cache_sha256": plan.exact.sha256,
            "incumbent_score": 0.8, "paired_gain_lower_bound": -0.1,
        }
        source.archive_stage(task, StageOutcome(0.7, metrics), 210)

    with ChallengeStore(tmp_path / "cold.db", track_spec_digest=case.track.digest) as store:
        cold = prepare(store)
        restored = source.restore_stage(cold)
        assert restored.score == 0.7 and restored.score_block == 210
        assert restored.metrics == metrics
        store._connection.execute(
            "UPDATE challenges SET file_sha256=? WHERE challenge_id=?",
            (cold.file_sha256, cold.challenge_id),
        )
        store._connection.commit()
        # The initial lease expires; the original archived decision, not GPU I/O, resumes FIFO.
        result = process_next_scoring(
            store, worker_id="recover", current_finalized_block=220, now_ms=31,
            settings=ScoringSettings(restore_stage=source.restore_stage),
            evaluator=lambda _task: pytest.fail("an archived stage was re-scored"),
        )
        assert result.state is ChallengeState.SUCCEEDED
        assert store.stage_score(cold.challenge_id, stage).score_block == 210
        assert not source.content.root.exists() or not any(source.content.root.iterdir())

    for key, value in {
        "cache_plan_digest": "ff" * 32, "cache_sha256": "ee" * 32,
        "cache_seed_block_hash": "0x" + "dd" * 32, "cache_generated_block": 152,
    }.items():
        name = source._stage_name(task.challenge_id, stage.value)
        stored = case.objects.get(name)
        altered = json.loads(stored.body)
        altered["metrics"][key] = value
        case.objects.replace_plan(name, canonical_json(altered), etag=stored.etag)
        with pytest.raises(RuntimeError, match="allocated READY"):
            source.restore_stage(task)
        case.objects.replace_plan(name, stored.body, etag=case.objects.get(name).etag)


def test_overlap_uses_oldest_unpublished_ready_without_changing_seals(inventory_case):
    case = inventory_case
    old = tuple(make_plan(case, block, index % 2) for index, block in enumerate(range(101, 105)))
    fresh = tuple(make_plan(case, block, index % 2)
                  for index, block in enumerate(range(43301, 43305)))
    plans = old + fresh
    sealed = {plan.plan_digest: plan.encode() for plan in plans}
    inventory = Inventory(case.objects, case.track.digest)
    for identifier, supplied in (("a1" * 32, tuple(reversed(plans))), ("a2" * 32, plans)):
        row = claim(inventory, supplied, identifier=identifier,
                    submission_block=50000, disclosure_delay_seconds=86400)
        assert row["plan_digest"] == old[0].plan_digest
        assert row["ready_sha256"] == hashlib.sha256(sealed[old[0].plan_digest]).hexdigest()
        assert row["submission_block"] == 50000
        assert row["submission_timestamp_ms"] == 50_000_000
    assert {plan.plan_digest: plan.encode() for plan in plans} == sealed


@pytest.mark.parametrize("disclosed_index,selected_index", [(1, 2), (2, 1)])
def test_new_claim_skips_expired_disclosed_incomplete_and_after_submission_plans(
    inventory_case, disclosed_index, selected_index,
):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1), make_plan(case, 103),
             make_plan(case, 1902, 1), make_plan(case, 100, ready=False))
    disclosed = plans[disclosed_index]
    case.objects.create_plan(f"disclosed/{disclosed.plan_digest}.json", canonical_json({
        "plan_digest": disclosed.plan_digest, "exact_sha256": disclosed.exact.sha256,
    }))
    row = claim(Inventory(case.objects, case.track.digest), tuple(reversed(plans)),
                submission_block=1902)
    assert row["plan_digest"] == plans[selected_index].plan_digest


def test_earliest_expiry_ties_are_stable_across_input_order(inventory_case):
    case = inventory_case
    plans = tuple(replace(make_plan(case, block, slot), completed_at_ms=150_000,
                          completed_block=150)
                  for block, slot in ((101, 0), (101, 1), (102, 0)))
    expected = min(plans, key=lambda plan: (plan.generation_block, plan.plan_digest))
    inventory = Inventory(case.objects, case.track.digest)
    for identifier, supplied in (("a1" * 32, plans), ("a2" * 32, tuple(reversed(plans)))):
        assert claim(inventory, supplied, identifier=identifier)["plan_digest"] == (
            expected.plan_digest
        )


def test_old_policy_claim_survives_new_priority_disclosure_and_cold_retry(inventory_case):
    case = inventory_case
    old, newest = make_plan(case, 101), make_plan(case, 43301, 1)
    inventory = Inventory(case.objects, case.track.digest)
    original = claim(inventory, (newest,), submission_block=50000,
                     disclosure_delay_seconds=86400)
    stored = case.objects.get(ALLOCATION_OBJECT)
    case.objects.create_plan(f"disclosed/{newest.plan_digest}.json", canonical_json({
        "plan_digest": newest.plan_digest, "exact_sha256": newest.exact.sha256,
    }))
    cold = Inventory(case.objects, case.track.digest)
    assert claim(cold, (old, newest), submission_block=50000,
                 disclosure_delay_seconds=86400) == original
    assert case.objects.get(ALLOCATION_OBJECT) == stored


def test_claim_is_fifo_idempotent_model_bound_and_cold_recoverable(inventory_case):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    inventory = Inventory(case.objects, case.track.digest)
    first = claim(inventory, tuple(reversed(plans)))
    assert first["plan_digest"] == plans[0].plan_digest
    assert claim(Inventory(case.objects, case.track.digest), plans) == first
    assert len(inventory.read()[1]["claims"]) == 1
    with pytest.raises(RuntimeError, match="original admission"):
        claim(inventory, plans, file_sha256="dd" * 32)
    second = claim(inventory, plans, identifier="a2" * 32)
    assert second["plan_digest"] == plans[0].plan_digest
    assert claim(inventory, plans, identifier="a3" * 32)["plan_digest"] == plans[0].plan_digest
    assert {row["plan_digest"] for row in inventory.read()[1]["claims"]} == {
        plans[0].plan_digest,
    }
    with pytest.raises(RuntimeError, match="original admission"):
        claim(inventory, plans, submission_block=201)


@pytest.mark.parametrize("submission,selected", [
    (102, None), (103, 101), (104, 101), (1902, 102), (1903, None),
])
def test_eligibility_uses_real_completion_and_original_submission(
    inventory_case, submission, selected,
):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    result = claim(Inventory(case.objects, case.track.digest), plans,
                   submission_block=submission)
    assert (None if result is None else result["plan_digest"]) == (
        None if selected is None else plans[selected - 101].plan_digest
    )


def test_plan_start_is_not_completion_and_newer_data_cannot_fill_old_submission(inventory_case):
    case = inventory_case
    plan = replace(make_plan(case, 101), completed_block=210)
    inventory = Inventory(case.objects, case.track.digest)
    assert claim(inventory, (plan,), submission_block=200) is None
    # An arbitrarily late admission is still compared to S, never to wall-clock A.
    assert claim(inventory, (plan,), submission_block=211)["plan_digest"] == plan.plan_digest
    changed = replace(plan, completed_block=211)
    with pytest.raises(RuntimeError, match="immutable READY"):
        claim(inventory, (changed,), submission_block=211)


def test_local_byte_loss_cannot_resample_the_original_pool(inventory_case, tmp_path):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    inventory = Inventory(case.objects, case.track.digest)
    original = claim(inventory, plans, submission_block=200)
    assert original["plan_digest"] == plans[0].plan_digest
    reader = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="unavailable"):
        reader.content.get(sha256=plans[0].exact.sha256, size_bytes=plans[0].exact.size_bytes)
    assert claim(inventory, plans, submission_block=200) == original
    later = claim(inventory, plans, identifier="a2" * 32, submission_block=251)
    assert later["plan_digest"] == plans[0].plan_digest
    assert set(inventory.read()[1]) == {"format", "track_spec_digest", "claims"}


def test_prune_protects_original_eligible_submissions_not_all_past_data(
    inventory_case, tmp_path, monkeypatch,
):
    case = inventory_case
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    old = make_plan(case, 101)
    new = make_plan(case, 1903, 1)
    monkeypatch.setattr(source, "_plans", lambda: (("old", old), ("new", new)))
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms", lambda b: b * 1000)
    pending = [{"state": "QUEUED", "block_height": 200,
                "track_spec_digest": case.track.digest}]
    # At a late scan, an original pre-expiry submission still protects its data.
    assert source.protected_plan_digests(pending, 2000) == frozenset({
        old.plan_digest, new.plan_digest,
    })
    # A model submitted after old expiry must not pin all historical datasets.
    pending[0]["block_height"] = 1904
    assert source.protected_plan_digests(pending, 2000) == frozenset({new.plan_digest})
    # A lagged scan must first discover pre-expiry submissions before pruning.
    assert old.plan_digest in source.protected_plan_digests([], 1901)


def test_time_rpc_failure_cannot_prune_or_invent_non_admission(
    inventory_case, tmp_path, monkeypatch
):
    source = ReadyBatchSource(track=inventory_case.track, objects=inventory_case.objects,
                              cache_dir=tmp_path)

    def unavailable(_block):
        raise OSError("time RPC unavailable")

    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms", unavailable)
    with pytest.raises(OSError, match="time RPC"):
        source.protected_plan_digests([], 2000)
    with pytest.raises(OSError, match="time RPC"):
        source.eligible_submission(SimpleNamespace(position=SimpleNamespace(block=200)))


def test_exact_tenure_admits_one_original_ready_without_confirmation_inventory(
    inventory_case, tmp_path, monkeypatch,
):
    case = inventory_case
    raw = track_spec_dict(case.track)
    raw["promotion"] = {"format": "veda-exact-tenure-v1", "minimum_gain": 0.002,
                        "decay_seconds": 604800}
    raw["scoring"].update(minimum_frontier_gain=0.003, match_confidence=0.9975,
                           bootstrap_resamples=4000)
    raw["full_video"]["minimum_gain"] = 0.003
    case.track = parse_track_spec(raw)
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    plan = make_plan(case, 101)
    monkeypatch.setattr(source, "_plans", lambda: (("single", plan),))
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms", lambda b: b * 1000)
    assert source.eligible_submission(SimpleNamespace(position=SimpleNamespace(block=200)))
    assert not source.eligible_submission(SimpleNamespace(position=SimpleNamespace(block=101)))
    with pytest.raises(RuntimeError, match="full-video inventory is disabled"):
        source.claim_ready("ab" * 32, confirmation=True, file_sha256="cd" * 32,
                           model_digest="ef" * 32, submission_block=200)


def test_confirmation_uses_independent_half_even_when_already_generated(inventory_case):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102), make_plan(case, 103, 1))
    inventory = Inventory(case.objects, case.track.digest)
    with pytest.raises(RuntimeError, match="exact allocation"):
        claim(inventory, plans, stage="CONFIRMING")
    exact = claim(inventory, plans)
    confirmation = claim(inventory, plans, stage="CONFIRMING")
    assert exact["plan_digest"] == plans[0].plan_digest
    assert confirmation["plan_digest"] == plans[2].plan_digest
    assert not {p.leaf.source_family_id for p in plans[2].source_proofs} & {
        p.leaf.source_family_id for p in plans[0].source_proofs
    }
    assert claim(inventory, plans, stage="CONFIRMING") == confirmation


def test_claimed_bytes_are_never_relabelled(inventory_case):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    inventory = Inventory(case.objects, case.track.digest)
    row = claim(inventory, plans)
    assert row["plan_digest"] == plans[0].plan_digest
    changed = replace(plans[0], exact=replace(plans[0].exact, sha256="dd" * 32))
    with pytest.raises(RuntimeError, match="immutable READY"):
        claim(inventory, (changed, plans[1]))


def test_cas_race_cannot_double_claim_or_discard_a_winning_claim(inventory_case, monkeypatch):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    create = case.objects.create_plan
    raced = False

    def race(name, payload):
        nonlocal raced
        if name == ALLOCATION_OBJECT and not raced:
            raced = True
            claim(Inventory(case.objects, case.track.digest), plans, identifier="a2" * 32)
        return create(name, payload)

    monkeypatch.setattr(case.objects, "create_plan", race)
    row = claim(inventory, plans)
    assert row["plan_digest"] == plans[0].plan_digest
    assert [r["ordinal"] for r in inventory.read()[1]["claims"]] == [1, 2]


def test_zero_miners_preproduce_and_full_inventory_pauses_then_refills(inventory_case):
    case = inventory_case
    class Store:
        def chain_cursor(self):
            return self.height, "0x" + f"{self.height + 1000:064x}"

        def finalized_block_hash(self, block):
            return "0x" + f"{block + 1000:064x}"

    store = Store()
    store.height = 101
    coordinator = PlanCoordinator(track=case.track, store=store, objects=case.objects,
                                  source_pool=case.pool)
    first = coordinator.create_next(now_ms=101_000, finalized_head=store.chain_cursor())
    assert first.inventory_slot == 0
    assert coordinator.create_next(now_ms=101_000, finalized_head=store.chain_cursor()) is None
    ready = make_plan(case, 101)
    # Keep the actual reservation's secret/selection, not a newly generated plan.
    ready = first.ready(
        completed_at_ms=(first.generation_block + 1) * 1000,
        completed_block=first.generation_block + 1,
        exact=ready.exact,
        dense_references=tuple(
            DenseReference(rollout.rollout_id, ready.exact, ready.exact)
            for rollout in first.rollouts
        ),
        sample_cache=ready.sample_cache,
    )
    stored = case.objects.get("plans/101.json")
    case.objects.replace_plan("plans/101.json", ready.encode(), etag=stored.etag)
    store.height = 102
    second = coordinator.create_next(now_ms=102_000, finalized_head=store.chain_cursor())
    assert second.inventory_slot == 1
    assert coordinator.create_next(now_ms=102_000, finalized_head=store.chain_cursor()) is None
    ready_second = second.ready(
        exact=ready.exact, sample_cache=ready.sample_cache,
        completed_block=103, completed_at_ms=103_000,
        dense_references=tuple(DenseReference(r.rollout_id, ready.exact, ready.exact)
                               for r in second.rollouts),
    )
    stored = case.objects.get("plans/102.json")
    case.objects.replace_plan("plans/102.json", ready_second.encode(), etag=stored.etag)
    claim(Inventory(case.objects, case.track.digest), (ready, ready_second))
    store.height = 103
    assert (
        coordinator.create_next(
            disk_bytes=case.track.cache.inventory_max_bytes, finalized_head=store.chain_cursor()
        )
        is None
    )
    assert coordinator.create_next(now_ms=103_000, finalized_head=store.chain_cursor()) is None
    # A claim does not refill the pool. Age starts renewal while old data is hidden.
    store.height = 1003
    assert coordinator.create_next(
        now_ms=1_003_000, finalized_head=store.chain_cursor(),
    ).inventory_slot == 0


def test_four_batch_target_fills_two_per_slot_without_rewriting_ready(inventory_case, tmp_path):
    case = inventory_case
    raw = valid_spec()
    raw["cache"]["inventory_target_batches"] = 4
    raw["cache"]["inventory_max_bytes"] = 12_000_000
    case.track = replace(case.track, cache=parse_track_spec(raw).cache)
    store = SimpleNamespace(finalized_block_hash=lambda _block: None)
    coordinator = PlanCoordinator(track=case.track, store=store, objects=case.objects,
                                  source_pool=case.pool)
    originals = {}
    slots = []
    for block in range(101, 105):
        head = (block, "0x" + f"{block + 1000:064x}")
        plan = coordinator.create_next(now_ms=block * 1000, finalized_head=head)
        slots.append(plan.inventory_slot)
        assert coordinator.create_next(now_ms=block * 1000, finalized_head=head) is None
        fixture = make_plan(case, block, plan.inventory_slot)
        ready = plan.ready(
            completed_at_ms=(block + 1) * 1000, completed_block=block + 1,
            exact=fixture.exact, sample_cache=fixture.sample_cache,
            dense_references=tuple(DenseReference(r.rollout_id, fixture.exact, fixture.exact)
                                   for r in plan.rollouts),
        )
        name = f"plans/{block}.json"
        case.objects.replace_plan(name, ready.encode(), etag=case.objects.get(name).etag)
        originals[name] = ready.encode()
    assert slots == [0, 1, 0, 1]
    assert coordinator.create_next(now_ms=106_000, finalized_head=(106, "0x" + "ab" * 32)) is None
    plans = tuple(plan for _, _, plan in coordinator.plans())
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    assert source.inventory_status(plans)["target_batches"] == 4
    assert all(case.objects.get(name).body == body for name, body in originals.items())


def test_authority_rejects_corrupt_or_cross_track_facts(inventory_case):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    row = claim(inventory, (make_plan(case, 101),))
    assert row["ready_sha256"] == hashlib.sha256(make_plan(case, 101).encode()).hexdigest()
    with pytest.raises(RuntimeError, match="invalid"):
        Inventory(case.objects, "ff" * 32).read()
    stored, document = inventory.read()
    document["claims"].append(document["claims"][0])
    case.objects.replace_plan(ALLOCATION_OBJECT, canonical_json(document), etag=stored.etag)
    with pytest.raises(RuntimeError, match="sequence"):
        inventory.read()


def test_claim_lost_response_keeps_original_allocation(inventory_case, monkeypatch):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    put = case.objects._client.put_object
    lost = False

    def put_then_lose(**kwargs):
        nonlocal lost
        result = put(**kwargs)
        if not lost:
            lost = True
            raise OSError("PUT committed but response was lost")
        return result

    monkeypatch.setattr(case.objects._client, "put_object", put_then_lose)
    with pytest.raises(OSError, match="response was lost"):
        claim(inventory, plans)
    before = case.objects.get(ALLOCATION_OBJECT)
    assert claim(Inventory(case.objects, case.track.digest), plans)["plan_digest"] == (
        plans[0].plan_digest
    )
    assert case.objects.get(ALLOCATION_OBJECT) == before
    assert len(inventory.read()[1]["claims"]) == 1


def test_same_stage_cas_race_reuses_only_the_original_claim(inventory_case, monkeypatch):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    create = case.objects.create_plan
    raced = False

    def race(name, payload):
        nonlocal raced
        if name == ALLOCATION_OBJECT and not raced:
            raced = True
            claim(Inventory(case.objects, case.track.digest), plans)
        return create(name, payload)

    monkeypatch.setattr(case.objects, "create_plan", race)
    assert claim(inventory, plans)["plan_digest"] == plans[0].plan_digest
    assert len(inventory.read()[1]["claims"]) == 1


def test_disclosure_blocks_new_claims_but_not_durable_original_claims(inventory_case):
    case = inventory_case
    plan = make_plan(case, 101)
    inventory = Inventory(case.objects, case.track.digest)
    original = claim(inventory, (plan,))
    case.objects.create_plan(f"disclosed/{plan.plan_digest}.json", canonical_json({
        "plan_digest": plan.plan_digest, "exact_sha256": plan.exact.sha256,
    }))
    assert claim(inventory, (plan,), identifier="a2" * 32,
                 submission_block=1902) is None
    assert claim(inventory, (plan,), identifier="a3" * 32) is None
    assert claim(inventory, (plan,)) == original


def test_public_inventory_contains_no_seed_secret_proofs_or_inputs(inventory_case, tmp_path):
    case = inventory_case
    plans = (replace(make_plan(case, 101), completed_at_ms=10**15),
             make_plan(case, 102, 1, ready=False))
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    status = source.inventory_status(plans)
    assert status["ready_batches"] == status["producing_batches"] == 1
    assert status["target_batches"] == 2
    assert status["commitments"][0]["secret_commitment"] == plans[0].secret_commitment
    encoded = canonical_json(status)
    for plan in plans:
        assert plan.secret.encode() not in encoded
        assert plan.batch_seed.encode() not in encoded
    assert b"source_proofs" not in encoded and b"sample_cache" not in encoded


def test_slots_remain_family_disjoint_across_different_seeds(inventory_case):
    case = inventory_case
    families = [set(), set()]
    for block in range(101, 121):
        for slot in (0, 1):
            plan = make_plan(case, block, slot)
            families[slot].update(proof.leaf.source_family_id for proof in plan.source_proofs)
            validate_source_selection_sequence(
                (plan,), commitments=case.track.source_commitments, scoring=case.track.scoring,
            )
    assert not families[0] & families[1]
    assert make_plan(case, 101).rollouts != make_plan(case, 102).rollouts


@pytest.mark.parametrize("field", ["file_sha256", "model_digest"])
def test_reader_claim_requires_both_registered_model_fingerprints(
    inventory_case, tmp_path, field,
):
    case = inventory_case
    identity = {
        "format": "veda-admission-model-v1", "track_spec_digest": case.track.digest,
        "challenge_id": "a1" * 32, "file_sha256": "b2" * 32, "model_digest": "c3" * 32,
    }
    identity[field] = "dd" * 32
    case.objects.create_plan("admissions/" + "a1" * 32 + ".json", canonical_json(identity))
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="admission identity"):
        source.claim_ready("a1" * 32, confirmation=False,
                           submission_block=200,
                           file_sha256="b2" * 32, model_digest="c3" * 32)
    assert source.inventory.read()[1]["claims"] == []


def test_reader_binds_original_metadata_even_when_payload_needs_recovery(
    inventory_case, tmp_path, monkeypatch,
):
    case = inventory_case
    case.objects.create_plan("admissions/" + "a1" * 32 + ".json", canonical_json({
        "format": "veda-admission-model-v1", "track_spec_digest": case.track.digest,
        "challenge_id": "a1" * 32, "file_sha256": "b2" * 32, "model_digest": "c3" * 32,
        "submission_block": 200,
        "submission_timestamp_ms": 200_000,
    }))
    plan = make_plan(case, 101)
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    monkeypatch.setattr(source, "_plans", lambda: (("plans/101.json", plan),))
    assert source.claim_ready("a1" * 32, confirmation=False, submission_block=200,
                                  file_sha256="b2" * 32, model_digest="c3" * 32)
    authority = source.inventory.read()[1]
    assert len(authority["claims"]) == 1
    assert authority["claims"][0]["plan_digest"] == plan.plan_digest


@pytest.mark.parametrize("height", [102, 1901])
def test_disclosure_does_not_restore_files_before_production_age(
    tmp_path, monkeypatch, height, inventory_case,
) -> None:
    plan = make_plan(inventory_case, 101)
    source = ReadyBatchSource(
        track=inventory_case.track, objects=inventory_case.objects, cache_dir=tmp_path,
    )
    monkeypatch.setattr(source, "_plan_by_digest", lambda _digest: plan)

    def unexpected_restore(_plan):
        raise AssertionError("nonterminal inventory must not restore any payload")

    monkeypatch.setattr(source, "_restore_exact", unexpected_restore)
    monkeypatch.setattr(source, "_restore_sample_cache", unexpected_restore)
    assert source.public_disclosure(plan.plan_digest, now_ms=height * 1000) is None
    assert list(tmp_path.iterdir()) == []
