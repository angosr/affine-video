import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator.production.batches import ReadyBatchSource
from validator.production.commitment import SourceLeaf, SourcePool, canonical_json
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
            exact=item, sample_cache=ReadySampleCache(item, item),
            dense_references=tuple(DenseReference(rollout.rollout_id, item, item)
                                   for rollout in plan.rollouts),
        )
    return plan


def claim(inventory, plans, identifier="a1" * 32, stage="SCORING", **kwargs):
    return inventory.claim(
        plans=plans, challenge_id=identifier, stage=stage,
        file_sha256=kwargs.get("file_sha256", "b2" * 32), model_digest="c3" * 32,
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
    assert second["plan_digest"] == plans[1].plan_digest
    assert claim(inventory, plans, identifier="a3" * 32) is None


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
    assert not {p.leaf.source_family_id for p in plans[0].source_proofs} & {
        p.leaf.source_family_id for p in plans[2].source_proofs
    }
    assert claim(inventory, plans, stage="CONFIRMING") == confirmation


def test_lost_unassigned_is_invalidated_but_claimed_bytes_are_never_relabelled(inventory_case):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    inventory = Inventory(case.objects, case.track.digest)
    assert inventory.invalidate_unassigned(plans[0].plan_digest)
    assert inventory.invalidate_unassigned(plans[0].plan_digest)
    row = claim(inventory, plans)
    assert row["plan_digest"] == plans[1].plan_digest
    assert not inventory.invalidate_unassigned(plans[1].plan_digest)
    assert len(inventory.read()[1]["invalidations"]) == 1
    changed = replace(plans[1], exact=replace(plans[1].exact, sha256="dd" * 32))
    with pytest.raises(RuntimeError, match="immutable READY"):
        claim(inventory, (plans[0], changed))


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
    assert row["plan_digest"] == plans[1].plan_digest
    assert [r["ordinal"] for r in inventory.read()[1]["claims"]] == [1, 2]
    assert not inventory.invalidate_unassigned(plans[0].plan_digest)


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
    first = coordinator.create_next()
    assert first.inventory_slot == 0
    assert coordinator.create_next() is None
    ready = make_plan(case, 101)
    # Keep the actual reservation's secret/selection, not a newly generated plan.
    ready = first.ready(exact=ready.exact, dense_references=tuple(
        DenseReference(rollout.rollout_id, ready.exact, ready.exact) for rollout in first.rollouts
    ), sample_cache=ready.sample_cache)
    stored = case.objects.get("plans/101.json")
    case.objects.replace_plan("plans/101.json", ready.encode(), etag=stored.etag)
    store.height = 102
    second = coordinator.create_next()
    assert second.inventory_slot == 1
    assert coordinator.create_next() is None
    assert Inventory(case.objects, case.track.digest).invalidate_unassigned(second.plan_digest)
    store.height = 103
    assert coordinator.create_next(disk_bytes=case.track.cache.inventory_max_bytes) is None
    assert coordinator.create_next().inventory_slot == 1


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


def test_invalidation_winning_cas_cannot_be_claimed(inventory_case, monkeypatch):
    case = inventory_case
    inventory = Inventory(case.objects, case.track.digest)
    plans = (make_plan(case, 101), make_plan(case, 102, 1))
    create = case.objects.create_plan
    raced = False

    def race(name, payload):
        nonlocal raced
        if name == ALLOCATION_OBJECT and not raced:
            raced = True
            Inventory(case.objects, case.track.digest).invalidate_unassigned(plans[0].plan_digest)
        return create(name, payload)

    monkeypatch.setattr(case.objects, "create_plan", race)
    assert claim(inventory, plans)["plan_digest"] == plans[1].plan_digest
    authority = inventory.read()[1]
    assert len(authority["invalidations"]) == len(authority["claims"]) == 1


def test_terminal_disclosure_never_returns_a_batch_to_inventory(inventory_case):
    case = inventory_case
    plan = make_plan(case, 101)
    inventory = Inventory(case.objects, case.track.digest)
    original = claim(inventory, (plan,))
    case.objects.create_plan(f"disclosed/{plan.plan_digest}.json", canonical_json({
        "plan_digest": plan.plan_digest, "exact_sha256": plan.exact.sha256,
    }))
    assert claim(inventory, (plan,), identifier="a2" * 32) is None
    assert claim(inventory, (plan,)) == original
    assert not inventory.invalidate_unassigned(plan.plan_digest)


def test_public_inventory_contains_no_seed_secret_proofs_or_inputs(inventory_case, tmp_path):
    case = inventory_case
    plans = (make_plan(case, 101), make_plan(case, 102, 1, ready=False))
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
                           file_sha256="b2" * 32, model_digest="c3" * 32)
    assert source.inventory.read()[1]["claims"] == []


def test_reader_fences_missing_unclaimed_payload_before_allocation(
    inventory_case, tmp_path, monkeypatch,
):
    case = inventory_case
    case.objects.create_plan("admissions/" + "a1" * 32 + ".json", canonical_json({
        "format": "veda-admission-model-v1", "track_spec_digest": case.track.digest,
        "challenge_id": "a1" * 32, "file_sha256": "b2" * 32, "model_digest": "c3" * 32,
    }))
    plan = make_plan(case, 101)
    source = ReadyBatchSource(track=case.track, objects=case.objects, cache_dir=tmp_path)
    monkeypatch.setattr(source, "_plans", lambda: (("plans/101.json", plan),))
    assert not source.claim_ready("a1" * 32, confirmation=False,
                                  file_sha256="b2" * 32, model_digest="c3" * 32)
    authority = source.inventory.read()[1]
    assert authority["claims"] == []
    assert authority["invalidations"] == [{
        "plan_digest": plan.plan_digest, "reason": "UNASSIGNED_DATA_LOST",
    }]


@pytest.mark.parametrize("terminal_ids", [frozenset(), frozenset({"b2" * 32})])
def test_disclosure_does_not_restore_files_until_the_allocated_challenge_is_terminal(
    tmp_path, monkeypatch, terminal_ids, inventory_case,
) -> None:
    plan = make_plan(inventory_case, 101)
    source = ReadyBatchSource(
        track=inventory_case.track, objects=inventory_case.objects, cache_dir=tmp_path,
    )
    source.inventory.claim(
        challenge_id="a1" * 32, stage="SCORING", file_sha256="b2" * 32,
        model_digest="c3" * 32, plans=(plan,),
    )
    monkeypatch.setattr(source, "_plan_by_digest", lambda _digest: plan)

    def unexpected_restore(_plan):
        raise AssertionError("nonterminal inventory must not restore any payload")

    monkeypatch.setattr(source, "_restore_exact", unexpected_restore)
    monkeypatch.setattr(source, "_restore_sample_cache", unexpected_restore)
    assert source.public_disclosure(plan.plan_digest, terminal_ids) is None
    assert list(tmp_path.iterdir()) == []
