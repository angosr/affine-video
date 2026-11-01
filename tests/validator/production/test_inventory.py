import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from miner.submit import ModelSubmission
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
    PublishedPlanReference,
    ReadyObject,
    ReadySampleCache,
    build_plan,
    generation_seed,
    public_disclosure_document,
    select_source_proofs,
    validate_source_selection_sequence,
)
from validator.production.retries import PrivateRetryStore
from validator.production.service import PlanCoordinator
from validator.production.storage import PreconditionFailed, PrivateObjectStore
from validator.publication import PublicArtifactError, S3Repository
from validator.queue import ChallengeState
from validator.release import verify_competition_manifest
from validator.scoring import (
    ScoringSettings,
    StageOutcome,
    TemporaryScoringError,
    process_next_scoring,
)
from validator.store import ChallengeStore

from .test_storage import MemoryS3, seed_legacy_stage


@pytest.fixture
def inventory_case(monkeypatch):
    from validator.production import environment

    # This metadata-only fixture has no frozen teacher package or GPU runtime.
    monkeypatch.setattr(environment, "evaluation_source_digest", lambda: "c0" * 32)
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


def test_first_claim_captures_full_order_and_retries_keep_its_original_evidence(
    inventory_case, monkeypatch,
):
    from validator.production import environment
    from validator.production import inventory as module

    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 190), make_plan(case, 101), make_plan(case, 250),
             make_plan(case, 90, ready=False))
    monkeypatch.setattr(module.time, "time_ns", lambda: 1_000_000_000_000)
    monkeypatch.setattr(environment, "evaluation_source_digest", lambda: "de" * 32)
    first = claim(inventory, plans)
    assert first["submission_block"] == 200
    assert first["selection"]["captured_at_ms"] == 1_000_000
    assert first["selection"]["clock"] == "local-before-successful-cas"
    assert first["selection"]["source_sha256"] == "de" * 32
    candidates = first["selection"]["candidates"]
    assert [row["generation_block"] for row in candidates] == [90, 101, 190, 250]
    assert [row["s_eligible"] for row in candidates] == [False, True, True, False]
    assert first["plan_digest"] == plans[1].plan_digest
    assert all(len(row["source_selection_sha256"]) == 64 for row in candidates)
    before = case.objects.get(ALLOCATION_OBJECT).body
    monkeypatch.setattr(module.time, "time_ns", lambda: 2_000_000_000_000)
    monkeypatch.setattr(environment, "evaluation_source_digest", lambda: "ef" * 32)
    assert claim(inventory, (*plans, make_plan(case, 102))) == first
    assert case.objects.get(ALLOCATION_OBJECT).body == before
    assert Inventory(case.objects, case.track.digest).read()[1]["claims"] == [first]


@pytest.mark.parametrize("damage", ["order", "eligibility", "previous_authority"])
def test_claim_evidence_rejects_changed_selection_without_reassigning(inventory_case, damage):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    claim(inventory, (make_plan(case, 101), make_plan(case, 102)))
    stored, document = inventory.read()
    selection = document["claims"][0]["selection"]
    if damage == "order":
        selection["candidates"].reverse()
    elif damage == "eligibility":
        selection["candidates"][0]["s_eligible"] = False
    else:
        selection["previous_authority_sha256"] = "00" * 32
    case.objects.replace_plan(ALLOCATION_OBJECT, canonical_json(document), etag=stored.etag)
    with pytest.raises(RuntimeError):
        inventory.read()


def test_legacy_claim_has_unknown_first_selection_and_is_never_backfilled(inventory_case):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 101),)
    claim(inventory, plans)
    stored, document = inventory.read()
    document["claims"][0].pop("selection")
    old_bytes = canonical_json(document)
    case.objects.replace_plan(ALLOCATION_OBJECT, old_bytes, etag=stored.etag)
    assert "selection" not in claim(inventory, plans)
    assert case.objects.get(ALLOCATION_OBJECT).body == old_bytes


@pytest.mark.parametrize("damage", [
    "omit", "replace", "forged-facts", "wrong-first", "input-bytes",
])
def test_selection_consumer_checks_original_plan_bytes_not_declared_booleans(
    inventory_case, damage,
):
    from validator.production.inventory import _pool_digest

    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 101), make_plan(case, 102))
    claim(inventory, plans)
    stored, document = inventory.read()
    row = document["claims"][0]
    candidates = row["selection"]["candidates"]
    if damage == "input-bytes":
        name = row["selection"]["inputs"]["object"]
        original = case.objects.get(name)
        case.objects.replace_plan(name, b"changed original input", etag=original.etag)
    else:
        if damage == "omit":
            candidates.pop(0)
        elif damage == "replace":
            candidates[0]["plan_sha256"] = "ff" * 32
        elif damage == "forged-facts":
            candidates[0].update(completed_at_ms=250_000, s_eligible=False)
            candidates.reverse()  # Self-consistent declared order, but false original plan facts.
        elif damage == "wrong-first":
            row.update(plan_digest=candidates[1]["plan_digest"],
                       ready_sha256=candidates[1]["plan_sha256"])
        if damage != "wrong-first":
            row.update(plan_digest=candidates[0]["plan_digest"],
                       ready_sha256=candidates[0]["plan_sha256"])
        row["selection"]["pool_sha256"] = _pool_digest(candidates)
        case.objects.replace_plan(ALLOCATION_OBJECT, canonical_json(document), etag=stored.etag)
    with pytest.raises(RuntimeError):
        inventory.read()


@pytest.mark.parametrize("damage", ["omitted-input", "unfinished", "limited", "wrong-origin"])
def test_claim_capture_rejects_incomplete_original_enumeration(inventory_case, damage):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102))
    for plan in plans:
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
    observations = [{}, {}]
    for observation in observations:
        case.objects.list_names("plans/", observation=observation)
    if damage == "unfinished":
        observations[1]["complete"] = False
    if damage == "limited":
        observations[1]["limit"] = 1000
    if damage == "wrong-origin":
        for observation in observations:
            observation["bucket"] = "other-authority"
    inventory = Inventory(case.objects, case.track.digest)
    with pytest.raises(RuntimeError, match="enumeration"):
        inventory.claim(
            challenge_id="a1" * 32, stage="SCORING", file_sha256="b2" * 32,
            model_digest="c3" * 32, plans=plans[:1] if damage == "omitted-input" else plans,
            submission_block=200, submission_timestamp_ms=200_000,
            disclosure_delay_seconds=1800,
            enumeration={"algorithm": "ReadyBatchSource._plans-v1", "observations": observations},
        )
    assert inventory.read()[1]["claims"] == []


def test_original_planned_candidate_remains_verifiable_after_ready_transition(inventory_case):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    planned = make_plan(case, 150, ready=False)
    original = claim(inventory, (make_plan(case, 101), planned))
    case.objects.create_plan("plans/150.json", make_plan(case, 150).encode())
    cold = Inventory(case.objects, case.track.digest)
    row = cold.read()[1]["claims"][0]
    assert row == original
    observed = next(p for p in cold.observed_plans(row) if p.plan_digest == planned.plan_digest)
    assert observed.state == "PLANNED"


def test_capture_rejects_enumerated_plan_under_a_substituted_key(inventory_case, tmp_path):
    case = inventory_case
    case.objects.create_plan("plans/999.json", make_plan(case, 101).encode())
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="key disagrees"):
        source._plans(observation={})


def test_successful_cas_row_seals_pool_version_and_only_winning_attempt_clock(
    inventory_case, monkeypatch,
):
    from validator.production import inventory as module

    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102))
    create = case.objects.create_plan
    raced, ticks = False, [100_000_000_000]

    def clock():
        ticks[0] += 1_000_000
        return ticks[0]

    def race(name, payload):
        nonlocal raced
        if name == ALLOCATION_OBJECT and not raced:
            raced = True
            claim(Inventory(case.objects, case.track.digest), plans, identifier="a2" * 32)
        return create(name, payload)

    monkeypatch.setattr(module.time, "time_ns", clock)
    monkeypatch.setattr(case.objects, "create_plan", race)
    row = claim(Inventory(case.objects, case.track.digest), plans)
    rows = Inventory(case.objects, case.track.digest).read()[1]["claims"]
    assert rows[-1] == row and row["selection"]["cas_attempt"] == 2
    assert row["selection"]["captured_at_ms"] > rows[0]["selection"]["captured_at_ms"]
    assert row["selection"]["pool_sha256"] == rows[0]["selection"]["pool_sha256"]
    sealed = case.objects.get(ALLOCATION_OBJECT).body
    assert claim(Inventory(case.objects, case.track.digest), tuple(reversed(plans))) == row
    assert case.objects.get(ALLOCATION_OBJECT).body == sealed


def test_generation_change_during_pool_observation_retries_without_resampling_bound_claim(
    inventory_case, tmp_path, monkeypatch,
):
    case = inventory_case
    identity = {"format": "veda-admission-model-v1", "track_spec_digest": case.track.digest,
                "challenge_id": "a1" * 32, "file_sha256": "b2" * 32,
                "model_digest": "c3" * 32, "submission_block": 200,
                "submission_timestamp_ms": 200_000}
    case.objects.create_plan("admissions/" + "a1" * 32 + ".json", canonical_json(identity))
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    case.objects.create_plan("plans/101.json", make_plan(case, 101).encode())
    actual_plans, reads = source._plans, [0]

    def changing(**kwargs):
        rows = actual_plans(**kwargs)
        reads[0] += 1
        if reads[0] == 1:
            case.objects.create_plan("plans/102.json", make_plan(case, 102, 1).encode())
        return rows

    monkeypatch.setattr(source, "_plans", changing)
    args = dict(confirmation=False, submission_block=200, file_sha256="b2" * 32,
                model_digest="c3" * 32)
    with pytest.raises(RuntimeError, match="changed during"):
        source.claim_ready("a1" * 32, **args)
    assert source.inventory.read()[1]["claims"] == []
    monkeypatch.setattr(source, "_plans", actual_plans)
    assert source.claim_ready("a1" * 32, **args)
    selection = source.inventory.read()[1]["claims"][0]["selection"]
    assert len(selection["candidates"]) == 2 and selection["pool_read_interval_ms"] is not None
    sealed = case.objects.get(ALLOCATION_OBJECT).body
    # Bound retry reads its original plan once, not a new two-read selection pool.
    case.objects.create_plan("plans/100.json", make_plan(case, 100).encode())
    reads[0] = 0

    def once(**kwargs):
        reads[0] += 1
        assert reads[0] == 1
        return actual_plans(**kwargs)

    monkeypatch.setattr(source, "_plans", once)
    assert source.claim_ready("a1" * 32, **args)
    assert case.objects.get(ALLOCATION_OBJECT).body == sealed


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
    seed_legacy_stage(cold, task, StageOutcome(.8, metrics), 210)
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
    ), objects=case.objects, source_pool=case.pool, confirm_disclosure=lambda _plan: False)
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
              "store": SimpleNamespace(finalized_block_hash=lambda _block: None),
              "confirm_disclosure": lambda _plan: False}
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
@pytest.mark.parametrize("vda2", [False, True])
def test_preproduced_archive_survives_cold_db_without_bulk(inventory_case, tmp_path, stage, vda2):
    case = inventory_case
    plans = (make_plan(case, 150), make_plan(case, 151, 1))
    for plan in plans:
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path / "empty")

    def prepare(store):
        submission = submission_at(200)
        if vda2:
            submission = replace(submission, payload=ModelSubmission(
                submission.repo_id, submission.revision, "b2" * 32,
            ).encode())
        identifier, _ = store.add_submission(submission)
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
        seed_legacy_stage(source, task, StageOutcome(0.7, metrics), 210)

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


@pytest.mark.parametrize("include_newer", [False, True])
def test_first_late_claim_keeps_original_s_eligibility_after_disclosure(
    inventory_case, include_newer,
):
    case = inventory_case
    oldest = make_plan(case, 101)
    plans = (oldest, make_plan(case, 150, 1)) if include_newer else (oldest,)
    sealed = {plan.plan_digest: plan.encode() for plan in plans}
    case.objects.create_plan(f"disclosed/{oldest.plan_digest}.json", canonical_json({
        "plan_digest": oldest.plan_digest, "exact_sha256": oldest.exact.sha256,
    }))
    inventory = Inventory(case.objects, case.track.digest)
    assert inventory.read()[1]["claims"] == []
    # Publication is at/after R; admission may finish later, but S remains before R.
    original = claim(inventory, tuple(reversed(plans)), submission_block=200)
    assert original is not None
    assert original["plan_digest"] == oldest.plan_digest
    assert original["ready_sha256"] == hashlib.sha256(sealed[oldest.plan_digest]).hexdigest()
    assert original["submission_block"] == 200
    assert original["submission_timestamp_ms"] == 200_000
    assert claim(Inventory(case.objects, case.track.digest), plans,
                 submission_block=200) == original
    assert {plan.plan_digest: plan.encode() for plan in plans} == sealed


@pytest.mark.parametrize("disclosed_index", [1, 2])
def test_new_claim_skips_expired_incomplete_and_after_submission_not_later_disclosure(
    inventory_case, disclosed_index,
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
    assert row["plan_digest"] == plans[1].plan_digest


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
                                  source_pool=case.pool, confirm_disclosure=lambda _plan: False)
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
                                  source_pool=case.pool, confirm_disclosure=lambda _plan: False)
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
        if not lost and kwargs["Key"].endswith(ALLOCATION_OBJECT):
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


def test_expiry_at_original_s_blocks_new_claims_but_later_disclosure_does_not(inventory_case):
    case = inventory_case
    plan = make_plan(case, 101)
    inventory = Inventory(case.objects, case.track.digest)
    original = claim(inventory, (plan,))
    case.objects.create_plan(f"disclosed/{plan.plan_digest}.json", canonical_json({
        "plan_digest": plan.plan_digest, "exact_sha256": plan.exact.sha256,
    }))
    assert claim(inventory, (plan,), identifier="a2" * 32,
                 submission_block=1902) is None
    assert claim(inventory, (plan,), identifier="a3" * 32)["plan_digest"] == plan.plan_digest
    assert claim(inventory, (plan,)) == original


def test_public_inventory_contains_no_seed_secret_proofs_or_inputs(inventory_case, tmp_path):
    case = inventory_case
    plans = (replace(make_plan(case, 101), completed_at_ms=10**15),
             make_plan(case, 102, 1, ready=False))
    original = tuple(plan.encode() for plan in plans)
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    status = source.inventory_status(plans)
    assert status["ready_batches"] == status["producing_batches"] == 1
    assert status["target_batches"] == 2
    assert status["commitments"][0]["secret_commitment"] == plans[0].secret_commitment
    assert [row["completed_block"] for row in status["commitments"]] == [102, None]
    for plan, payload, row in zip(plans, original, status["commitments"], strict=True):
        assert plan.encode() == payload
        assert row["completed_at_ms"] == plan.completed_at_ms
        assert row["disclose_at_ms"] == (
            None if plan.completed_at_ms is None else plan.completed_at_ms
            + case.track.cache.disclosure_delay_seconds * 1000
        )
        assert row["ready_sha256"] == (
            hashlib.sha256(payload).hexdigest() if plan.state == "READY" else None
        )
    encoded = canonical_json(status)
    for plan in plans:
        assert plan.secret.encode() not in encoded
        assert plan.batch_seed.encode() not in encoded
    assert b"source_proofs" not in encoded and b"sample_cache" not in encoded


def test_disclosure_waiver_drives_due_failed_and_public_status_from_original_data_cas(
    inventory_case, tmp_path, monkeypatch,
):
    from validator.production import batches as batches_module

    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    original = tuple(plan.encode() for plan in plans)
    rules = parse_track_spec(rules_for_data(case.track))
    assert rules.digest != case.track.digest
    now = max(plan.completed_at_ms for plan in plans) + (
        case.track.cache.disclosure_delay_seconds * 1000
    ) + 1
    monkeypatch.setattr(batches_module.time, "time_ns", lambda: now * 1_000_000)
    monkeypatch.setattr(ReadyBatchSource, "_plans", lambda _self: tuple(
        (f"plans/{plan.generation_block}.json", plan) for plan in plans))
    source = ReadyBatchSource(track=rules, data_track=case.track, objects=case.objects,
                              cache_dir=tmp_path)
    retries = PrivateRetryStore(case.objects, track_spec_digest=case.track.digest)
    digest = plans[0].plan_digest
    retries.disclosure_attempt(digest, now_ms=now - 100_000, lease_ms=100)
    retries.disclosure_attempt(digest, now_ms=now - 99_999, lease_ms=100,
                               completed_attempt=1, error="private transport detail")
    before = case.objects.get(f"retry_state/disclosures/{digest}.json")
    source.waive_disclosure(digest, now_ms=now, reason="Explicit operator decision")
    cold = ReadyBatchSource(track=rules, data_track=case.track, objects=case.objects,
                            cache_dir=tmp_path / "cold")
    assert cold.due_plan_digests(now) == (plans[1].plan_digest,)
    status = cold.inventory_status(plans)
    assert status["disclosure_pending_batches"] == 1
    assert [row["state"] for row in status["disclosure_status"]] == ["WAIVED", "OVERDUE"]
    after = json.loads(case.objects.get(f"retry_state/disclosures/{digest}.json").body)
    assert {key: after[key] for key in json.loads(before.body)} == json.loads(before.body)
    assert not retries.disclosure_failed(digest)
    assert case.objects.get_optional(f"disclosed/{digest}.json") is None
    assert original == tuple(plan.encode() for plan in plans)
    assert b"Explicit operator decision" not in canonical_json(status)
    assert b"private transport detail" not in canonical_json(status)
    assert Inventory(case.objects, case.track.digest).read()[1]["claims"] == []


def test_disclosure_waiver_checks_original_ready_deadline_and_receipt(
    inventory_case, tmp_path, monkeypatch,
):
    case = inventory_case
    plan = make_plan(case, 101)
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    monkeypatch.setattr(source, "_plans", lambda: (("plans/original.json", plan),))
    due = plan.completed_at_ms + case.track.cache.disclosure_delay_seconds * 1000
    with pytest.raises(ValueError, match="due READY"):
        source.waive_disclosure(plan.plan_digest, now_ms=due - 1, reason="Operator decision")
    assert case.objects.get_optional(f"retry_state/disclosures/{plan.plan_digest}.json") is None
    case.objects.create_plan(f"disclosed/{plan.plan_digest}.json", canonical_json({
        "plan_digest": plan.plan_digest, "exact_sha256": "ff" * 32,
    }))
    with pytest.raises(RuntimeError, match="receipt differs"):
        source.due_plan_digests(due)
    with pytest.raises(ValueError, match="completion receipt"):
        source.waive_disclosure(plan.plan_digest, now_ms=due, reason="Operator decision")


def test_disclosure_audit_checks_public_group_before_a_lagging_retry_failure(
    inventory_case, monkeypatch,
):
    from tests.validator.test_publication import MemoryS3 as PublicMemoryS3
    from validator import publication as publication_module
    from validator.publication import PublicLedger

    case = inventory_case
    plan = make_plan(case, 101)
    due = plan.completed_at_ms + case.track.cache.disclosure_delay_seconds * 1000
    monkeypatch.setattr(publication_module.time, "time_ns", lambda: (due + 2) * 1_000_000)
    client = PublicMemoryS3()
    client.objects["tracks/test/status.json"] = canonical_json({
        "track_id": case.track.track_id, "track_spec_digest": case.track.digest,
        "source_block": 800,
        "inventory": {"observed_at_ns": (due + 1) * 1_000_000, "disclosure_status": [{
            "plan_digest": plan.plan_digest, "state": "FAILED", "attempt": 1,
            "retry_after_ms": due + 1000, "waived_at_ms": None, "disclose_at_ms": due,
        }]},
    })
    repository = S3Repository(client, bucket="public", prefix="tracks/test")
    ledger = PublicLedger(800, ())
    assert repository._load_disclosure_status(case.track, case.track, ledger, frozenset())[0][
        "state"
    ] == "FAILED"
    group = f"tracks/test/batch_groups/{plan.plan_digest}.json"
    original = canonical_json(public_disclosure_document(plan))
    client.objects[group] = original
    assert repository._load_disclosure_status(case.track, case.track, ledger, frozenset()) == ()
    assert ledger.disclosed_plan_digests == frozenset()  # No claim that payloads were audited.
    tampered = json.loads(original)
    tampered["public_exact_object"] += ".changed"
    client.objects[group] = canonical_json(tampered)
    with pytest.raises(PublicArtifactError, match="receipt is invalid"):
        repository._load_disclosure_status(case.track, case.track, ledger, frozenset())


def test_operator_retry_keeps_original_inventory_cas_claim_and_ready_bytes(
    inventory_case, tmp_path,
):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    original_plans = tuple(plan.encode() for plan in plans)
    inventory = Inventory(case.objects, case.track.digest)
    retries = PrivateRetryStore(case.objects, track_spec_digest=case.track.digest)
    settings = ScoringSettings(
        max_attempts=1, restore_retry=retries.restore, record_retry=retries.record_failure,
    )
    attempts = []

    def evaluate(task):
        allocation = claim(inventory, plans, identifier=task.challenge_id,
                           submission_block=task.position.block)
        attempts.append((task.submission_id, task.position, allocation))
        if len(attempts) == 1:
            raise TemporaryScoringError("GPU_DOWN", "worker unavailable")
        return StageOutcome(0.7, {})

    path = tmp_path / "operator.db"
    with ChallengeStore(path) as store:
        identifier, _ = store.add_submission(submission_at(200), occurred_at_ms=1)
        for state in (ChallengeState.FETCHING, ChallengeState.VALIDATING, ChallengeState.SCORING):
            store.transition(identifier, state, occurred_at_ms=2)
        result = process_next_scoring(
            store, worker_id="worker", settings=settings, current_finalized_block=300,
            evaluator=evaluate, now_ms=10,
        )
        assert result.state is ChallengeState.SYSTEM_BLOCKED
        authority = case.objects.get(ALLOCATION_OBJECT)
    with ChallengeStore(path) as store:
        assert process_next_scoring(
            store, worker_id="automatic", settings=settings, current_finalized_block=300,
            evaluator=evaluate, now_ms=10**9,
        ) is None
        retries.reset(identifier, "SCORING")
        store.resume_blocked(identifier, occurred_at_ms=10**9)
        result = process_next_scoring(
            store, worker_id="operator-resumed", settings=settings, current_finalized_block=300,
            evaluator=evaluate, now_ms=10**9 + 1,
        )
        assert result.state is ChallengeState.SUCCEEDED
        assert store.get_challenge(identifier).attempts == 2
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert case.objects.get(ALLOCATION_OBJECT) == authority
    assert tuple(plan.encode() for plan in plans) == original_plans


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
    case.objects.create_plan("plans/101.json", plan.encode())
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


def _source_exclusion_case(case, tmp_path, *, write_receipt=True):
    from tests.validator.test_publication import MemoryS3 as PublicMemoryS3

    published, hidden = make_plan(case, 101), make_plan(case, 102)
    for plan in (published, hidden):
        case.objects.create_plan(f"plans/{plan.generation_block}.json", plan.encode())
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    client = PublicMemoryS3()
    repository = S3Repository(client, bucket="public", prefix="tracks/test")
    repository._put_immutable(f"batch_groups/{published.plan_digest}.json",
                              canonical_json(public_disclosure_document(published)))
    if write_receipt:
        source.record_disclosed(published.plan_digest)
    coordinator = PlanCoordinator(
        track=case.track, objects=case.objects, source_pool=case.pool,
        store=SimpleNamespace(finalized_block_hash=lambda _block: None),
        confirm_disclosure=repository.confirm_source_disclosure,
    )
    return SimpleNamespace(published=published, hidden=hidden, source=source, client=client,
                           repository=repository, coordinator=coordinator)


def test_new_plan_excludes_only_verified_public_sources_and_freezes_original_inputs(
    inventory_case, tmp_path,
):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path)
    old = (data.published, data.hidden)
    old_bytes = tuple(plan.encode() for plan in old)
    allocation = claim(data.source.inventory, old)
    authority = case.objects.get(ALLOCATION_OBJECT)
    new = data.coordinator.create_next(now_ms=10**12, finalized_head=(110, "0x" + "ab" * 32))
    reference = PublishedPlanReference.from_plan(data.published)
    assert new.source_exclusions == (reference,)
    assert json.loads(new.encode())["format"] == "veda-inventory-plan-v2"
    assert BatchPlan.decode(new.encode()) == new
    assert not {p.leaf.digest for p in new.source_proofs}.intersection(
        p.leaf.digest for p in data.published.source_proofs)
    assert tuple(case.objects.get(f"plans/{p.generation_block}.json").body
                 for p in old) == old_bytes
    assert case.objects.get(ALLOCATION_OBJECT) == authority

    # A later disclosure cannot change this PLANNED item's frozen exclusion input.
    data.repository._put_immutable(f"batch_groups/{data.hidden.plan_digest}.json",
                                   canonical_json(public_disclosure_document(data.hidden)))
    assert data.repository.confirm_source_disclosure(data.hidden) is True
    data.source.record_disclosed(data.hidden.plan_digest)
    cold = PlanCoordinator(track=case.track, objects=case.objects, source_pool=case.pool,
                           store=SimpleNamespace(finalized_block_hash=lambda _block: None),
                           confirm_disclosure=data.repository.confirm_source_disclosure)
    assert cold.create_next(now_ms=10**12 + 1, finalized_head=(111, "0x" + "cd" * 32)) is None
    assert cold.plans()[-1][2].encode() == new.encode()
    assert claim(data.source.inventory, old) == allocation
    assert case.objects.get(ALLOCATION_OBJECT) == authority
    assert tuple(case.objects.get(f"plans/{p.generation_block}.json").body
                 for p in old) == old_bytes

    # Public replay loads only the referenced original metadata, not hidden future inputs.
    reader = S3Repository(data.client, bucket="public", prefix="tracks/test")
    data.client.calls.clear()
    validate_source_selection_sequence(
        (new,), commitments=case.track.source_commitments, scoring=case.track.scoring,
        source_pool_merkle_root=case.pool.root, available_routes=case.track.model.selector.routes,
        read_disclosed_plan=reader._source_disclosure_plan,
    )
    assert reader._confirmed_immutable == {reference.object_name: reference.disclosure_sha256}


@pytest.mark.parametrize("damage", ["missing", "changed", "bad-receipt"])
def test_source_exclusion_capture_requires_original_verified_public_record(
    inventory_case, tmp_path, damage,
):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path)
    name = f"tracks/test/batch_groups/{data.published.plan_digest}.json"
    if damage == "missing":
        del data.client.objects[name]
    elif damage == "changed":
        data.client.objects[name] = b'{}'
    else:
        receipt_name = f"disclosed/{data.published.plan_digest}.json"
        stored = case.objects.get(receipt_name)
        case.objects.replace_plan(receipt_name, b'{}', etag=stored.etag)
    data.repository._confirmed_immutable.clear()  # Cold startup must authenticate original bytes.
    before = tuple(plan.encode() for _, _, plan in data.coordinator.plans())
    with pytest.raises((RuntimeError, PublicArtifactError), match="disclosure"):
        data.coordinator.create_next(now_ms=10**12, finalized_head=(110, "0x" + "ab" * 32))
    assert tuple(plan.encode() for _, _, plan in data.coordinator.plans()) == before


@pytest.mark.parametrize("damage", ["ready-clock", "disclosure-sha", "unavailable", "reuse"])
def test_private_and_public_replay_reject_changed_source_exclusion_inputs(
    inventory_case, tmp_path, damage,
):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path)
    original = data.coordinator.create_next(now_ms=10**12, finalized_head=(110, "0x" + "ab" * 32))
    reference = original.source_exclusions[0]
    proofs = original.source_proofs
    published = data.published
    if damage == "ready-clock":
        published = replace(published, completed_at_ms=published.completed_at_ms + 1)
        data.client.objects[f"tracks/test/{reference.object_name}"] = canonical_json(
            public_disclosure_document(published),
        )
    elif damage == "disclosure-sha":
        reference = replace(reference, disclosure_sha256="ff" * 32)
    elif damage == "reuse":
        proofs = data.published.source_proofs
    else:
        reference = replace(reference, generation_block=100)
    changed = build_plan(
        track_spec_digest=case.track.digest, generation_block=original.generation_block,
        block_hash=original.block_hash, secret=original.secret,
        inventory_slot=original.inventory_slot,
        source_proofs=proofs, scoring=case.track.scoring,
        available_routes=case.track.model.selector.routes, source_exclusions=(reference,),
    )
    for known, read in (((published, data.hidden, changed), None),
                        ((changed,), data.repository._source_disclosure_plan)):
        with pytest.raises((ValueError, PublicArtifactError)):
            validate_source_selection_sequence(
                known, commitments=case.track.source_commitments, scoring=case.track.scoring,
                source_pool_merkle_root=case.pool.root,
                available_routes=case.track.model.selector.routes, read_disclosed_plan=read,
            )


def test_source_exclusion_create_race_retains_first_cas_plan(inventory_case, tmp_path, monkeypatch):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path)
    create = case.objects.create_plan
    first = []

    def lost_race(name, body):
        create(name, body)
        first.append(body)
        raise PreconditionFailed("another producer sealed the first plan")

    monkeypatch.setattr(case.objects, "create_plan", lost_race)
    winner = data.coordinator.create_next(now_ms=10**12,
                                         finalized_head=(110, "0x" + "ab" * 32))
    assert first == [winner.encode()]
    assert winner.source_exclusions == (PublishedPlanReference.from_plan(data.published),)


def test_legacy_plan_bytes_remain_frozen_and_old_planned_item_is_not_replaced(inventory_case):
    case = inventory_case
    planned = make_plan(case, 101, ready=False)
    ready = make_plan(case, 101)
    assert hashlib.sha256(planned.encode()).hexdigest() == (
        "0b75a2b12b45d49c9aaeb561375a8cf35d1fdb7e2b91347bef908495e3342d05"
    )
    assert hashlib.sha256(ready.encode()).hexdigest() == (
        "682aa03a7674107f24d1912e7093c5b197f06e7b76465e7763e426f4cff4f8cb"
    )
    assert "source_exclusions" not in json.loads(ready.encode())
    assert BatchPlan.decode(ready.encode()).encode() == ready.encode()
    case.objects.create_plan("plans/101.json", planned.encode())
    coordinator = PlanCoordinator(track=case.track, objects=case.objects, source_pool=case.pool,
                                  store=SimpleNamespace(finalized_block_hash=lambda _block: None),
                                  confirm_disclosure=lambda _plan: False)
    assert coordinator.create_next(now_ms=10**12, finalized_head=(110, "0x" + "ab" * 32)) is None
    assert case.objects.get("plans/101.json").body == planned.encode()


def test_exhausted_published_sources_do_not_reserve_or_reuse_a_plan(inventory_case, tmp_path):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path)
    universe = sorted(item.sha256 for item in case.track.source_commitments
                      if item.partition == "production" and item.generation_geometry == "all")[::2]
    covered = {proof.leaf.digest for proof in data.published.source_proofs}
    for block in range(103, 163):
        if covered == set(universe):
            break
        plan = make_plan(case, block)
        selected = {proof.leaf.digest for proof in plan.source_proofs}
        if selected <= covered:
            continue
        case.objects.create_plan(f"plans/{block}.json", plan.encode())
        data.repository._put_immutable(f"batch_groups/{plan.plan_digest}.json",
                                       canonical_json(public_disclosure_document(plan)))
        data.source.record_disclosed(plan.plan_digest)
        covered.update(selected)
    assert covered == set(universe)
    before = tuple(plan.encode() for _, _, plan in data.coordinator.plans())
    with pytest.raises(ValueError, match="exhausted.*all"):
        data.coordinator.create_next(now_ms=10**12, finalized_head=(1000, "0x" + "ab" * 32))
    assert tuple(plan.encode() for _, _, plan in data.coordinator.plans()) == before
    assert case.objects.get_optional("plans/1000.json") is None
    assert case.objects.get_optional(ALLOCATION_OBJECT) is None


def test_publication_before_private_receipt_is_already_excluded(inventory_case, tmp_path):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path, write_receipt=False)
    marker = f"disclosed/{data.published.plan_digest}.json"
    assert case.objects.get_optional(marker) is None
    first = data.coordinator.create_next(now_ms=10**12, finalized_head=(110, "0x" + "ab" * 32))
    assert first.source_exclusions == (PublishedPlanReference.from_plan(data.published),)
    assert case.objects.get_optional(marker) is None  # Do not fabricate the missing receipt.


def test_new_exclusion_ready_claim_and_cold_retry_keep_original_allocation(
    inventory_case, tmp_path,
):
    case = inventory_case
    data = _source_exclusion_case(case, tmp_path)
    planned = data.coordinator.create_next(now_ms=10**12,
                                           finalized_head=(110, "0x" + "ab" * 32))
    exact = ReadyObject("exact/new.npz", "34" * 32, 101)
    ready = planned.ready(
        exact=exact, dense_references=tuple(
            DenseReference(row.rollout_id, data.published.dense_references[0].video,
                           data.published.dense_references[0].latent) for row in planned.rollouts
        ), sample_cache=data.published.sample_cache,
        completed_block=111, completed_at_ms=1_999_000,
    )
    stored = case.objects.get("plans/110.json")
    case.objects.replace_plan("plans/110.json", ready.encode(), etag=stored.etag)
    # A singleton bound-plan replay must resolve only its original referenced metadata.
    allocated = claim(data.source.inventory, (ready,), submission_block=2000,
                      submission_timestamp_ms=2_000_000)
    assert allocated["plan_digest"] == planned.plan_digest
    assert allocated["ready_sha256"] == hashlib.sha256(ready.encode()).hexdigest()
    authority = case.objects.get(ALLOCATION_OBJECT)
    cold = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path / "cold")
    assert claim(cold.inventory, (ready,), submission_block=2000,
                 submission_timestamp_ms=2_000_000) == allocated
    assert case.objects.get(ALLOCATION_OBJECT) == authority
    assert case.objects.get("plans/110.json").body == ready.encode()
    assert ready.source_exclusions == planned.source_exclusions
    assert (ready.completed_at_ms, cold._plan_by_digest(ready.plan_digest).completed_at_ms) == (
        1_999_000, 1_999_000,
    )


@pytest.mark.parametrize("damage", ["old-extra", "missing", "null", "duplicate", "future"])
def test_versioned_exclusion_fields_are_strict_and_never_backfilled(
    inventory_case, tmp_path, damage,
):
    data = _source_exclusion_case(inventory_case, tmp_path)
    plan = data.coordinator.create_next(now_ms=10**12,
                                        finalized_head=(110, "0x" + "ab" * 32))
    document = json.loads(plan.encode())
    if damage == "old-extra":
        document["format"] = "veda-inventory-plan-v1"
    elif damage == "missing":
        del document["source_exclusions"]
    elif damage == "null":
        document["source_exclusions"] = None
    elif damage == "duplicate":
        document["source_exclusions"] *= 2
    else:
        document["source_exclusions"][0]["generation_block"] = plan.generation_block
    with pytest.raises(ValueError):
        BatchPlan.decode(canonical_json(document))


@pytest.mark.parametrize("missing_status", [400, 404])
def test_source_public_confirmation_is_anonymous_metadata_only_and_never_caches_absence(
    inventory_case, missing_status,
):
    plan = make_plan(inventory_case, 101)
    payload = canonical_json(public_disclosure_document(plan))
    visible = [False]
    calls = []

    def respond(request):
        assert request.method == "GET"
        assert "authorization" not in request.headers
        assert request.url.path == f"/main/batch_groups/{plan.plan_digest}.json"
        calls.append(request.url.path)
        if visible[0]:
            return httpx.Response(200, content=payload, headers={"ETag": '"original"'})
        return httpx.Response(missing_status, json={"statusCode": 404, "error": "not_found"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        reader = S3Repository.from_public_http(client, base_url="https://public.test/main")
        assert reader.confirm_source_disclosure(plan) is False
        assert reader._confirmed_immutable == {}
        visible[0] = True
        assert reader.confirm_source_disclosure(plan) is True
        assert reader.confirm_source_disclosure(plan) is True
    assert len(calls) == 2
