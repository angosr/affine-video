import hashlib
import json
import queue
import shutil
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

import validator.production.service as production_service
from scoring.cache import CacheLoadError, write_exact_cache_batch
from scoring.exact import ExactCase
from scoring.model import ValidatedModel
from scoring.spec import (
    BudgetSpec,
    ScoringSpec,
    SourceCommitmentSpec,
    StepWindowSpec,
    TrajectoryScheduleSpec,
)
from validator.production.batches import ReadyBatchSource, _validate_plan_case_binding
from validator.production.commitment import (
    SourceLeaf,
    SourcePool,
    canonical_json,
    verify_source_proof,
)
from validator.production.inventory import Inventory
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    build_audit_evidence_plan,
    build_plan,
    generation_seed,
    inventory_commitment,
    select_query_tiles,
    select_source_digests,
    select_source_proofs,
    validate_source_selection_sequence,
)
from validator.production.runtime import DenseRolloutRuntime, trajectory_tensor_key
from validator.production.service import (
    ProducerService,
    UnitResult,
    WorkerPool,
    _find_valid_shard,
    _plan_attempt_count,
    _prune_local_work,
    _reserve_plan_attempt,
    _validate_source_capacity,
    _work_directory_lock,
    _write_unit_completion,
)
from validator.production.storage import (
    LocalBatchStore,
    PreconditionFailed,
    PrivateObjectStore,
    StoredObject,
)
from validator.queue import ChainPosition, ChallengeState
from validator.store import ChallengeStore

from .test_storage import MemoryS3


def scoring() -> ScoringSpec:
    return ScoringSpec(
        budgets=(
            BudgetSpec(1, 0, 1.0),
            BudgetSpec(2, 0, 2.0),
            BudgetSpec(3, 0, 4.0),
        ),
        sampled_route_count=2,
        rollouts_per_stratum=2,
        sampled_step_count=3,
        query_tiles_per_case=4,
        step_selection_algorithm="sha256-window-without-replacement-v1",
        query_tile_selection_algorithm="sha256-segment-quota-v1",
        trajectory_schedule=TrajectoryScheduleSpec(
            "test",
            (1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
            (
                StepWindowSpec("early", (0, 1), 0.34),
                StepWindowSpec("middle", (2, 3), 0.33),
                StepWindowSpec("late", (4,), 0.33),
            ),
        ),
        tail_mean_weight=0.8,
        stratum_weights=(("16:9@37", 1.0),),
        minimum_frontier_gain=0.001,
        match_confidence=0.95,
        bootstrap_resamples=1000,
        bootstrap_seed=7,
    )


def test_runtime_rejects_a_rounded_trajectory_commitment() -> None:
    runtime = DenseRolloutRuntime.__new__(DenseRolloutRuntime)
    runtime.teacher = SimpleNamespace(
        schedule=SimpleNamespace(video=(1.0, 0.972972972972973, 0.0))
    )

    runtime.verify_trajectory_schedule((1.0, 0.972972972972973, 0.0))
    with pytest.raises(RuntimeError, match="trajectory schedule"):
        runtime.verify_trajectory_schedule((1.0, 0.973, 0.0))


def proof(index: int = 0):
    leaf = SourceLeaf(
        "production",
        f"family-{index}",
        f"source-{index}",
        f"prompt {index}",
        (),
        "16:9@37",
        "internal-cleared-v1",
        hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
    )
    return SourcePool((leaf,)).proof(f"source-{index}")


def source_commitments(pool: SourcePool) -> tuple[SourceCommitmentSpec, ...]:
    return tuple(SourceCommitmentSpec(**item) for item in pool.public_commitments())


def ready_sample_cache() -> ReadySampleCache:
    return ReadySampleCache(
        ReadyObject("inputs/index.json", "78" * 32, 10),
        ReadyObject("inputs/text.safetensors", "89" * 32, 20),
    )


def test_plan_is_deterministic_and_only_transitions_to_ready_once() -> None:
    arguments = { "secret": "a5" * 32, "inventory_slot": 0,
        "track_spec_digest": "ab" * 32,
        "generation_block": 120,
        "block_hash": "0x" + "cd" * 32,


        "source_proofs": (proof(0), proof(1)),
        "scoring": scoring(),
        "available_routes": ("route-0", "route-1", "route-2"),
    }
    first = build_plan(**arguments)
    second = build_plan(**arguments)

    assert first == second
    assert first.secret_commitment == inventory_commitment(first.secret)
    assert len(first.selected_routes) == 2
    assert len(first.selected_steps) == 3
    assert len(first.rollouts) == 2
    assert first.state == "PLANNED"
    ready = first.ready(
        exact=ReadyObject("exact/120-data.npz", "12" * 32, 100),
        dense_references=tuple(
            DenseReference(
                rollout.rollout_id,
                ReadyObject(f"videos/{rollout.rollout_id}.mp4", "34" * 32, 200),
                ReadyObject(f"latents/{rollout.rollout_id}.npz", "56" * 32, 300),
            )
            for rollout in first.rollouts
        ),
        sample_cache=ready_sample_cache(),
    )
    assert ready.state == "READY"
    assert ready.plan_digest == first.plan_digest
    assert BatchPlan.decode(first.encode()) == first
    assert BatchPlan.decode(ready.encode()) == ready
    with pytest.raises(ValueError, match="canonical"):
        BatchPlan.decode(first.encode() + b"\n")
    with pytest.raises(ValueError, match="only a PLANNED"):
        ready.ready(
            exact=ReadyObject("exact/other.npz", "56" * 32, 100),
            dense_references=ready.dense_references,
            sample_cache=ready_sample_cache(),
        )


def test_query_selection_excludes_global_rows_and_reserves_reference_quota() -> None:
    selected = select_query_tiles(
        query_selection_seed="ab" * 32,
        case_id="case-a",
        target_tiles=(0, 1, 2, 3, 4),
        reference_tiles=(10, 11),
        count=4,
    )

    assert len(selected) == 4
    assert len(set(selected) & {10, 11}) == 1
    assert not set(selected) & {20, 21}


def test_source_selection_excludes_the_same_matches_exact_families() -> None:
    first = proof(0).leaf
    second = SourceLeaf(
        "production",
        "family-1",
        "source-1",
        "another prompt",
        (),
        "16:9@37",
        "internal-cleared-v1",
        hashlib.sha256(b"nonce-1").hexdigest(),
    )
    pool = SourcePool((first, second))

    selected = select_source_proofs(
        pool,
        commitments=source_commitments(pool),
        partition="production",
        strata=("16:9@37",),
        sources_per_stratum=1,
        batch_seed="12" * 32,
        excluded_source_digests=frozenset({first.digest}),
    )

    assert selected[0].leaf.source_family_id == "family-1"
    with pytest.raises(ValueError, match="exhausted"):
        select_source_proofs(
            pool,
            commitments=source_commitments(pool),
            partition="production",
            strata=("16:9@37",),
            sources_per_stratum=1,
            batch_seed="12" * 32,
            excluded_source_digests=frozenset({first.digest, second.digest}),
        )


def test_source_selection_is_reproducible_from_the_public_index() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(4)))
    commitments = source_commitments(pool)
    selected = select_source_proofs(
        pool,
        commitments=commitments,
        partition="production",
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed="34" * 32,
        excluded_source_digests=frozenset(),
    )

    expected = select_source_digests(
        commitments,
        partition="production",
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed="34" * 32,
        excluded_source_digests=frozenset(),
    )

    assert tuple(item.leaf.digest for item in selected) == expected
    assert all(verify_source_proof(item, pool.root) for item in selected)


def test_source_selection_is_partition_explicit_and_audit_is_deterministic() -> None:
    leaves = tuple(
        SourceLeaf(
            partition,
            f"{partition}-family-{index}",
            f"{partition}-source-{index}",
            f"{partition} prompt {index}",
            (),
            "16:9@37",
            "internal-cleared-v1",
            hashlib.sha256(f"{partition}-nonce-{index}".encode()).hexdigest(),
        )
        for partition in ("production", "audit")
        for index in range(3)
    )
    pool = SourcePool(leaves)
    commitments = source_commitments(pool)
    track_digest = "ab" * 32
    block_hash = "0x" + "cd" * 32
    arguments = {
        "strata": ("16:9@37",),
        "sources_per_stratum": 2,
        "batch_seed": generation_seed(track_digest, block_hash, "a5" * 32),
        "excluded_source_digests": frozenset(),
    }

    selected = select_source_proofs(
        pool,
        commitments=commitments,
        partition="audit",
        **arguments,
    )
    expected = select_source_digests(commitments, partition="audit", **arguments)
    production = select_source_digests(commitments, partition="production", **arguments)
    production_proofs = select_source_proofs(
        pool,
        commitments=commitments,
        partition="production",
        **arguments,
    )

    assert tuple(proof.leaf.digest for proof in selected) == expected
    assert all(proof.leaf.partition == "audit" for proof in selected)
    assert set(expected).isdisjoint(production)
    assert expected == select_source_digests(commitments, partition="audit", **arguments)

    audit_plan = build_audit_evidence_plan(secret="a5" * 32,
        track_spec_digest=track_digest,
        generation_block=120,
        block_hash=block_hash,
        source_proofs=selected,
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    production_plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest=track_digest,
        generation_block=120,
        block_hash=block_hash,
        source_proofs=production_proofs,
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    commitment_digests = tuple(item.sha256 for item in commitments)
    assert BatchPlan.decode(audit_plan.encode()) == audit_plan
    assert tuple(proof.leaf.digest for proof in audit_plan.source_proofs) == expected
    assert audit_plan.selected_routes == production_plan.selected_routes
    assert audit_plan.selected_steps == production_plan.selected_steps
    assert audit_plan.query_selection_seed == production_plan.query_selection_seed
    assert all(
        verify_source_proof(
            proof,
            pool.root,
            commitment_digests=commitment_digests,
        )
        for proof in audit_plan.source_proofs
    )
    with pytest.raises(ValueError, match="inventory requires a source slot"):
        validate_source_selection_sequence(
            (audit_plan,), commitments=commitments, scoring=scoring()
        )


@pytest.mark.parametrize("partition", ["", "staging", True, None])
def test_source_selection_rejects_invalid_partition(partition) -> None:
    pool = SourcePool((proof(0).leaf,))
    arguments = {
        "partition": partition,
        "strata": ("16:9@37",),
        "sources_per_stratum": 1,
        "batch_seed": "78" * 32,
        "excluded_source_digests": frozenset(),
    }

    with pytest.raises(ValueError, match="production or audit"):
        select_source_digests(source_commitments(pool), **arguments)
    with pytest.raises(ValueError, match="production or audit"):
        select_source_proofs(pool, commitments=source_commitments(pool), **arguments)


def test_live_batch_plan_rejects_audit_sources() -> None:
    leaves = tuple(
        SourceLeaf(
            "audit",
            f"audit-family-{index}",
            f"audit-source-{index}",
            f"audit prompt {index}",
            (),
            "16:9@37",
            "internal-cleared-v1",
            hashlib.sha256(f"audit-nonce-{index}".encode()).hexdigest(),
        )
        for index in range(2)
    )
    pool = SourcePool(leaves)
    audit_proofs = tuple(pool.proof(leaf.source_id) for leaf in leaves)

    with pytest.raises(ValueError, match="production sources"):
        build_plan(secret="a5" * 32, inventory_slot=0,
            track_spec_digest="ab" * 32,
            generation_block=120,
            block_hash="0x" + "cd" * 32,
            source_proofs=audit_proofs,
            scoring=scoring(),
            available_routes=("route-0", "route-1", "route-2"),
        )


def test_source_selection_sequence_rejects_a_member_chosen_out_of_order() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(4)))
    commitments = source_commitments(pool)
    block_hash = "0x" + "cd" * 32
    selected = select_source_proofs(
        pool,
        commitments=commitments,
        partition="production",
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed=generation_seed("ab" * 32, block_hash, "a5" * 32),
        excluded_source_digests=frozenset(), inventory_slot=0,
    )
    arguments = { "secret": "a5" * 32, "inventory_slot": 0,
        "track_spec_digest": "ab" * 32,
        "generation_block": 120,
        "block_hash": block_hash,


        "scoring": scoring(),
        "available_routes": ("route-0", "route-1", "route-2"),
    }
    plan = build_plan(source_proofs=selected, **arguments)

    validate_source_selection_sequence(
        (plan,), commitments=commitments, scoring=scoring()
    )
    wrong_order = build_plan(source_proofs=tuple(reversed(selected)), **arguments)
    with pytest.raises(ValueError, match="public commitment index"):
        validate_source_selection_sequence(
            (wrong_order,), commitments=commitments, scoring=scoring()
        )


def test_source_pool_rejects_duplicate_families_before_selection() -> None:
    leaves = (
        SourceLeaf(
            "production",
            "family-a",
            "source-a-1",
            "prompt a1",
            (),
            "16:9@37",
            "internal-cleared-v1",
            hashlib.sha256(b"nonce-a1").hexdigest(),
        ),
        SourceLeaf(
            "production",
            "family-a",
            "source-a-2",
            "prompt a2",
            (),
            "16:9@37",
            "internal-cleared-v1",
            hashlib.sha256(b"nonce-a2").hexdigest(),
        ),
        SourceLeaf(
            "production",
            "family-b",
            "source-b",
            "prompt b",
            (),
            "16:9@37",
            "internal-cleared-v1",
            hashlib.sha256(b"nonce-b").hexdigest(),
        ),
    )
    with pytest.raises(ValueError, match="must be unique"):
        SourcePool(leaves)


def test_every_rollout_uses_a_distinct_source_family() -> None:
    proofs = tuple(proof(index) for index in range(4))
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=proofs,
        scoring=replace(scoring(), rollouts_per_stratum=4),
        available_routes=("route-0", "route-1", "route-2"),
    )

    assert len(plan.rollouts) == 4
    assert {rollout.source_id for rollout in plan.rollouts} == {
        item.leaf.source_id for item in proofs
    }


def test_local_work_pruning_only_removes_unretained_plan_directories(tmp_path) -> None:
    retained = "a" * 64
    old = "b" * 64
    (tmp_path / retained).mkdir()
    (tmp_path / old).mkdir()
    (tmp_path / "operator-notes").mkdir()

    _prune_local_work(tmp_path, frozenset({retained}))

    assert (tmp_path / retained).is_dir()
    assert not (tmp_path / old).exists()
    assert (tmp_path / "operator-notes").is_dir()


def test_local_work_pruning_removes_only_abandoned_batch_temporaries(tmp_path) -> None:
    work = tmp_path / "work"
    plan = work / ("a" * 64)
    batch = plan / "batch"
    batch.mkdir(parents=True)
    abandoned = batch / "tmpinterrupted.tmp"
    abandoned.write_bytes(b"interrupted NPZ assembly")
    sealed = batch / "120-sealed.npz"
    sealed.write_bytes(b"immutable complete cache")
    preserved = (
        batch / "operator.tmp",
        plan / "units" / "tmpunit.tmp",
        work / "content" / "tmpcontent.tmp",
        work / "operator-notes" / "tmpnotes.tmp",
    )
    for path in preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"outside abandoned batch assembly scope")
    outside = tmp_path / "outside.tmp"
    outside.write_bytes(b"external original")
    alias = batch / "tmpalias.tmp"
    alias.symlink_to(outside)

    _prune_local_work(work, frozenset({plan.name}))

    assert not abandoned.exists()
    assert sealed.read_bytes() == b"immutable complete cache"
    assert all(path.read_bytes() == b"outside abandoned batch assembly scope" for path in preserved)
    assert alias.is_symlink() and outside.read_bytes() == b"external original"


def test_local_work_pruning_does_not_follow_retained_plan_or_batch_links(tmp_path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    external = tmp_path / "external"
    (external / "batch").mkdir(parents=True)
    abandoned = external / "batch" / "tmpinterrupted.tmp"
    abandoned.write_bytes(b"not owned by this work directory")
    (work / ("a" * 64)).symlink_to(external, target_is_directory=True)
    second = work / ("b" * 64)
    second.mkdir()
    (second / "batch").symlink_to(external / "batch", target_is_directory=True)

    _prune_local_work(work, frozenset({"a" * 64, "b" * 64}))

    assert abandoned.read_bytes() == b"not owned by this work directory"


def test_local_work_directory_allows_only_one_producer(tmp_path) -> None:
    with (
        _work_directory_lock(tmp_path),
        pytest.raises(RuntimeError, match="another producer"),
        _work_directory_lock(tmp_path),
    ):
        pass

    with _work_directory_lock(tmp_path):
        pass


def test_worker_pool_has_a_plan_wall_clock_ceiling() -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    worker = object.__new__(WorkerPool)
    worker._timeout_seconds = 0.01
    worker._result = queue.Queue()
    worker._queues = [queue.Queue()]
    worker._processes = [SimpleNamespace(is_alive=lambda: True)]

    with pytest.raises(RuntimeError, match="wall-clock limit"):
        worker.run(plan)


class AttemptObjects:
    def __init__(self, values=None) -> None:
        self.values = {} if values is None else values

    def list_names(self, prefix):
        return tuple(sorted(name for name in self.values if name.startswith(prefix)))

    def get(self, name):
        return StoredObject(self.values[name], hashlib.sha256(self.values[name]).hexdigest())

    def get_optional(self, name):
        return self.get(name) if name in self.values else None

    def create_plan(self, name, payload):
        existing = self.values.get(name)
        if existing is not None and existing != payload:
            raise PreconditionFailed("different bytes")
        self.values[name] = payload
        return self.get(name)


@pytest.fixture
def local_producer(tmp_path, monkeypatch):
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32, generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)), scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )

    class Objects(AttemptObjects):
        def replace_plan(self, name, payload, *, etag):
            assert self.get(name).etag == etag
            self.values[name] = payload
            return self.get(name)

        def put_content(self, *_args, **_kwargs):
            pytest.fail("hidden batch bytes must never be uploaded")

    objects = Objects({"plans/120.json": plan.encode()})
    service = object.__new__(ProducerService)
    service.objects = objects
    service.config = SimpleNamespace(work_dir=tmp_path / "work")
    service.track = SimpleNamespace(
        digest=plan.track_spec_digest, artifact_manifest_sha256="34" * 32,
        source_pool_merkle_root="56" * 32, cache=SimpleNamespace(exact_max_bytes=10**9),
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    def input_file(name, payload):
        path = input_dir / name
        path.write_bytes(payload)
        return path, production_service._ready_file(path, "inputs", path.suffix)

    index_path, index = input_file("index.json", b'{"samples":[]}')
    text_path, text = input_file("text.safetensors", b"inputs")
    inputs = ReadySampleCache(index, text)
    monkeypatch.setattr(
        production_service, "_build_selected_sample_cache",
        lambda *_args: (inputs, ((index_path, index), (text_path, text))),
    )
    payload = [b"deterministic exact fixture"]

    def assemble(directory, *_args, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "batch.npz"
        path.write_bytes(payload[0])
        return SimpleNamespace(path=path, sha256=hashlib.sha256(payload[0]).hexdigest())

    monkeypatch.setattr(production_service, "assemble_exact_cache_batch", assemble)

    class Workers:
        def run(self, pending, **_kwargs):
            results = []
            for index, rollout in enumerate(pending.rollouts):
                video = tmp_path / f"{index}.mp4"
                latent = tmp_path / f"{index}.safetensors"
                video.write_bytes(f"video-{index}".encode())
                latent.write_bytes(f"latent-{index}".encode())
                results.append(UnitResult(
                    rollout.rollout_id, "unused-shard", str(video), str(latent), 1, 0,
                ))
            return tuple(results)

    return service, plan, objects, Workers(), payload


@pytest.mark.parametrize("winner_change", ["identical", "exact", "inputs", "dense", "plan"])
def test_producer_cas_loser_requires_identical_ready(local_producer, winner_change):
    service, plan, objects, workers, _payload = local_producer
    objects.create_plan("stages/challenge/SCORING.json", b"immutable scored result")
    winner_body = []

    def lose_replace(name, payload, *, etag):
        assert objects.get(name).etag == etag
        candidate = BatchPlan.decode(payload)
        if winner_change == "exact":
            candidate = replace(candidate, exact=replace(candidate.exact, sha256="11" * 32))
        elif winner_change == "inputs":
            candidate = replace(candidate, sample_cache=replace(
                candidate.sample_cache,
                text=replace(candidate.sample_cache.text, sha256="22" * 32),
            ))
        elif winner_change == "dense":
            reference, *remaining = candidate.dense_references
            candidate = replace(candidate, dense_references=(replace(
                reference, video=replace(reference.video, sha256="33" * 32),
            ), *remaining))
        elif winner_change == "plan":
            changed = {**candidate.planned_document(), "inventory_slot": 1}
            candidate = replace(
                candidate, inventory_slot=1,
                plan_digest=hashlib.sha256(canonical_json(changed)).hexdigest(),
            )
        winner_body.append(candidate.encode())
        objects.values[name] = candidate.encode()
        raise PreconditionFailed("another producer won the same PLANNED ETag")

    objects.replace_plan = lose_replace
    if winner_change == "identical":
        ready = service._execute(
            "plans/120.json", objects.get("plans/120.json"), plan, workers,
            deadline=time.monotonic() + 30,
        )
        assert ready.encode() == winner_body[0]
        assert not service._needs_regeneration(ready)
    else:
        with pytest.raises(RuntimeError, match="concurrent READY.*local batch"):
            service._execute(
                "plans/120.json", objects.get("plans/120.json"), plan, workers,
                deadline=time.monotonic() + 30,
            )
    assert objects.get("plans/120.json").body == winner_body[0]
    assert objects.get("stages/challenge/SCORING.json").body == b"immutable scored result"
    assert set(objects.values) == {"plans/120.json", "stages/challenge/SCORING.json"}


@pytest.mark.parametrize("interruption_reason", ["REGENERATION_MISMATCH", "REGENERATION_EXHAUSTED"])
def test_gpu_local_ready_regeneration_preserves_binding_and_recovers_original_bytes(
    tmp_path, monkeypatch, local_producer, interruption_reason,
) -> None:
    """Storage/orchestration regression, not GPU determinism or score evidence."""

    service, plan, objects, workers, payload = local_producer
    ready = service._execute(
        "plans/120.json", objects.get("plans/120.json"), plan, workers,
        deadline=time.monotonic() + 30,
    )
    assert ready.state == "READY"
    assert not service._needs_regeneration(ready)
    Inventory(objects, plan.track_spec_digest).claim(
        challenge_id="a1" * 32, stage="SCORING", file_sha256="b2" * 32,
        model_digest="c3" * 32, plans=(ready,),
    )
    immutable_ready = objects.get("plans/120.json")
    objects.create_plan("stages/challenge/SCORING.json", b"immutable scored result")
    shutil.rmtree(service.content.root)
    assert service._needs_regeneration(ready)
    recovered = service._execute(
        "plans/120.json", immutable_ready, ready, workers, deadline=time.monotonic() + 30,
    )
    assert recovered.encode() == immutable_ready.body
    assert objects.get("plans/120.json") == immutable_ready
    assert not service._needs_regeneration(ready)
    shutil.rmtree(service.content.root)
    if interruption_reason == "REGENERATION_EXHAUSTED":
        service._interrupt(ready, interruption_reason)
    payload[0] = b"non-reproducible replacement"
    with pytest.raises(RuntimeError, match="SYSTEM_BLOCKED"):
        service._execute(
            "plans/120.json", immutable_ready, ready, workers, deadline=time.monotonic() + 30,
        )
    assert objects.get("plans/120.json") == immutable_ready
    assert objects.get("stages/challenge/SCORING.json").body == b"immutable scored result"
    interruption = objects.get(f"interruptions/{plan.plan_digest}.json")
    assert json.loads(interruption.body)["reason"] == interruption_reason
    assert service._needs_regeneration(ready)  # the existing attempt cooldown bounds later probes
    source = ReadyBatchSource(
        track=service.track, objects=objects, cache_dir=tmp_path / "restored",
        batch_dir=service.content.root,
    )
    monkeypatch.setattr(source, "_plans", lambda: (("plans/120.json", ready),))
    with pytest.raises(production_service.StorageError):
        source._plan_for("a1" * 32, confirmation=False)
    payload[0] = b"deterministic exact fixture"
    recovered = service._execute(
        "plans/120.json", immutable_ready, ready, workers, deadline=time.monotonic() + 30,
    )
    assert recovered.encode() == immutable_ready.body
    assert source._plan_for("a1" * 32, confirmation=False) == ready
    assert not service._needs_regeneration(ready)
    assert objects.get(f"interruptions/{plan.plan_digest}.json") == interruption
    assert objects.get("plans/120.json") == immutable_ready
    assert objects.get("stages/challenge/SCORING.json").body == b"immutable scored result"


@pytest.mark.parametrize("state", ["PLANNED", "READY"])
@pytest.mark.parametrize("cursor_block", [200, 10_000_000])
@pytest.mark.parametrize("newer_ready_count", [0, 2])
def test_producer_continues_at_arbitrary_heights_and_releases_workers(
    monkeypatch, tmp_path, state, cursor_block, newer_ready_count,
) -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32, generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)), scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    if state == "READY":
        plan = plan.ready(
            exact=ReadyObject("exact/batch.npz", "12" * 32, 100),
            dense_references=tuple(
                DenseReference(
                    rollout.rollout_id,
                    ReadyObject(f"videos/{index}.mp4", "34" * 32, 100),
                    ReadyObject(f"latents/{index}.safetensors", "56" * 32, 100),
                )
                for index, rollout in enumerate(plan.rollouts)
            ),
            sample_cache=ready_sample_cache(),
        )
    abandoned = tmp_path / plan.plan_digest / "batch" / "tmpinterrupted.tmp"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"prior failed assembly must be reclaimed before the next attempt")
    saved_unit = tmp_path / plan.plan_digest / "units" / "saved-unit.npz"
    saved_unit.parent.mkdir()
    saved_unit.write_bytes(b"completed unit progress")
    newer = []
    for block in (140, 160)[:newer_ready_count]:
        item = build_plan(secret="a5" * 32, inventory_slot=0,
            track_spec_digest=plan.track_spec_digest, generation_block=block,
            block_hash="0x" + f"{block:064x}",
            source_proofs=(proof(0), proof(1)), scoring=scoring(),
            available_routes=("route-0", "route-1", "route-2"),
        )
        newer.append(item.ready(
            exact=ReadyObject(f"exact/{block}.npz", "12" * 32, 100),
            dense_references=tuple(
                DenseReference(
                    rollout.rollout_id,
                    ReadyObject(f"videos/{block}-{index}.mp4", "34" * 32, 100),
                    ReadyObject(f"latents/{block}-{index}.safetensors", "56" * 32, 100),
                )
                for index, rollout in enumerate(item.rollouts)
            ),
            sample_cache=ready_sample_cache(),
        ))
    database = tmp_path / "validator.db"
    with ChallengeStore(database, track_spec_digest=plan.track_spec_digest) as store:
        store.advance_chain_cursor(cursor_block, "0x" + "cd" * 32)
    objects = AttemptObjects({
        f"plans/{item.generation_block}.json": item.encode() for item in (plan, *newer)
    })
    original = objects.get("plans/120.json")
    executions, closed_workers, stale_before_execution = [], [], []
    progress_before_execution = []
    rows = tuple(
        (f"plans/{item.generation_block}.json",
         objects.get(f"plans/{item.generation_block}.json"), item)
        for item in (plan, *newer)
    )

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def create_next(self, **_kwargs):
            return None

        def plans(self):
            return rows

    service = object.__new__(ProducerService)
    service.config = SimpleNamespace(
        validator_database=database, work_dir=tmp_path, poll_seconds=0.01,
    )
    service.track = SimpleNamespace(digest=plan.track_spec_digest)
    service.objects = objects
    service.pool = None
    service.alert = None
    service.attempt_timeout_seconds = 10
    service._needs_regeneration = lambda candidate: candidate.plan_digest == plan.plan_digest

    def execute(*_args, **_kwargs):
        stale_before_execution.append(abandoned.exists())
        progress_before_execution.append(saved_unit.read_bytes() if saved_unit.exists() else None)
        executions.append(plan.plan_digest)
        assert len(executions) == 1
        return plan

    service._execute = execute

    monkeypatch.setattr(production_service, "PlanCoordinator", Coordinator)
    monkeypatch.setattr(
        production_service, "WorkerPool",
        lambda *_args: SimpleNamespace(close=lambda: closed_workers.append(True)),
    )
    service._run_locked(once=True)

    expected = 1
    assert len(executions) == len(closed_workers) == expected
    assert _plan_attempt_count(objects, plan) == expected
    assert objects.get("plans/120.json") == original
    assert stale_before_execution == [False]
    assert progress_before_execution == [b"completed unit progress"]
    assert saved_unit.read_bytes() == b"completed unit progress"
    assert not abandoned.exists()
    assert all(objects.get(name) == stored for name, stored, _plan in rows)


def test_producer_cooldown_survives_host_loss_and_recovers_without_reset(
    monkeypatch, tmp_path,
) -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    durable = {}

    for expected in range(1, 6):
        objects = AttemptObjects(durable)
        assert _reserve_plan_attempt(objects, plan, now_ms=100) == expected
        assert _plan_attempt_count(AttemptObjects(durable), plan) == expected

    assert _reserve_plan_attempt(AttemptObjects(durable), plan, now_ms=3_600_099) is None
    assert len(durable) == 5
    assert _reserve_plan_attempt(AttemptObjects(durable), plan, now_ms=3_600_100) == 6
    assert _reserve_plan_attempt(AttemptObjects(durable), plan, now_ms=3_600_101) is None
    completed = []

    class Store:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def create_next(self, **_kwargs):
            return None

        def plans(self):
            if completed:
                return ()
            return (("plans/120.json", StoredObject(plan.encode(), "etag"), plan),)

    service = object.__new__(ProducerService)
    service.config = SimpleNamespace(
        validator_database="unused.db", poll_seconds=0.01, work_dir=tmp_path,
    )
    service.track = SimpleNamespace(digest=plan.track_spec_digest)
    service.objects = AttemptObjects(durable)
    service.pool = None
    service.alert = None
    monkeypatch.setattr(
        production_service, "ChallengeStore", lambda *_args, **_kwargs: Store()
    )
    monkeypatch.setattr(production_service, "PlanCoordinator", Coordinator)
    monkeypatch.setattr(Store, "chain_cursor", lambda _self: (120, "0x" + "cd" * 32),
                        raising=False)
    monkeypatch.setattr(production_service.time, "time_ns", lambda: 3_600_101 * 1_000_000)
    with pytest.raises(RuntimeError, match="retry cooldown"):
        service._run_locked(once=True)
    clock = [3_600_101]
    closed = []
    sleeps = []
    probes = []
    alerts = []
    service.alert = lambda *args: alerts.append(args)
    service.attempt_timeout_seconds = 10
    monkeypatch.setattr(production_service.time, "time_ns", lambda: clock[0] * 1_000_000)
    monkeypatch.setattr(
        Store, "chain_cursor",
        lambda _self: (201 if completed else 120, "0x" + "cd" * 32),
    )
    monkeypatch.setattr(
        production_service, "WorkerPool",
        lambda *_args: SimpleNamespace(close=lambda: closed.append(True)),
    )

    def elapsed(seconds):
        if completed:
            raise KeyboardInterrupt
        assert seconds <= 60
        sleeps.append(seconds)
        clock[0] += 3_600_000

    def execute(*_args, **_kwargs):
        probes.append(True)
        if len(probes) == 1:
            raise RuntimeError("synthetic ongoing producer outage")
        completed.append(True)
        return SimpleNamespace(generation_block=120)

    monkeypatch.setattr(production_service.time, "sleep", elapsed)
    service._execute = execute
    with pytest.raises(KeyboardInterrupt):
        service._run_locked(once=False)
    assert completed == [True] and closed == [True, True]
    assert sleeps
    assert _plan_attempt_count(AttemptObjects(durable), plan) == 8
    assert alerts[0][0] == "PRODUCER_PLAN_FAILURE" and alerts[0][1]["attempt"] == 7

    class RacingObjects(AttemptObjects):
        def create_plan(self, name, payload):
            winner = json.loads(payload)
            winner["reservation"] = "0" * 32
            super().create_plan(name, canonical_json(winner))
            return super().create_plan(name, payload)

    # Two hosts reaching the same due probe cannot both reserve its ordinal.
    clock[0] += 3_600_000
    assert _reserve_plan_attempt(RacingObjects(durable), plan, now_ms=clock[0]) is None
    assert _plan_attempt_count(AttemptObjects(durable), plan) == 9


@pytest.mark.parametrize(
    "phase", ["database", "create", "list", "prune", "reserve", "timeout"],
)
@pytest.mark.parametrize("once", [False, True])
def test_producer_scheduling_failures_alert_and_resume_the_same_plan(
    monkeypatch, tmp_path, phase, once,
) -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    failures = []
    completed = []
    alerts = []
    sleeps = []
    original_sleep = time.sleep

    def fail(at):
        if at == phase and len(failures) < 6:
            failures.append(at)
            if phase == "timeout":
                original_sleep(0.1)
            raise OSError("synthetic scheduling outage")

    class Store:
        def __init__(self, *_args, **_kwargs):
            fail("database")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def create_next(self, **_kwargs):
            fail("create")
            return None

        def plans(self):
            fail("list")
            fail("timeout")
            if completed:
                return ()
            return (("plans/120.json", StoredObject(plan.encode(), "etag"), plan),)

    real_reserve = production_service._reserve_plan_attempt

    def reserve(objects, pending):
        fail("reserve")
        return real_reserve(objects, pending)

    real_prune = production_service._prune_local_work

    def prune(root, retained):
        fail("prune")
        real_prune(root, retained)

    def execute(_name, _stored, pending, _workers, **_kwargs):
        assert pending == plan
        completed.append(pending.plan_digest)
        return SimpleNamespace(generation_block=120)

    service = object.__new__(ProducerService)
    service.config = SimpleNamespace(
        validator_database="unused.db", poll_seconds=0.01, work_dir=tmp_path,
    )
    service.track = SimpleNamespace(digest=plan.track_spec_digest)
    service.objects = AttemptObjects()
    service.pool = None
    service.alert = lambda *args: alerts.append(args)
    service.attempt_timeout_seconds = 10
    service._execute = execute
    monkeypatch.setattr(production_service, "ChallengeStore", Store)
    monkeypatch.setattr(production_service, "PlanCoordinator", Coordinator)
    monkeypatch.setattr(production_service, "_reserve_plan_attempt", reserve)
    monkeypatch.setattr(production_service, "_prune_local_work", prune)
    monkeypatch.setattr(production_service, "_PRODUCER_ATTEMPT_TIMEOUT_SECONDS", 0.02)
    def sleep(seconds):
        if completed:
            raise KeyboardInterrupt
        sleeps.append(seconds)

    monkeypatch.setattr(production_service.time, "sleep", sleep)
    monkeypatch.setattr(
        production_service, "WorkerPool",
        lambda *_args: SimpleNamespace(close=lambda: None),
    )

    if once:
        with pytest.raises(RuntimeError, match="producer scheduling failed"):
            service._run_locked(once=True)
        assert completed == [] and sleeps == []
        assert _plan_attempt_count(service.objects, plan) == 0
        assert [details["failures"] for _, details in alerts] == [1]
    else:
        with pytest.raises(KeyboardInterrupt):
            service._run_locked(once=False)
        assert completed == [plan.plan_digest]
        assert _plan_attempt_count(service.objects, plan) == 1
        assert len(failures) == 6
        assert [details["failures"] for _, details in alerts] == [1, 5]
        assert sleeps and max(sleeps) <= 60
    assert all(event == "PRODUCER_SCHEDULING_FAILURE" for event, _ in alerts)
    assert alerts[0][1]["error_type"] == ("TimeoutError" if phase == "timeout" else "OSError")


@pytest.mark.parametrize("failure", ["configuration", "restore", "timeout"])
def test_producer_bootstrap_failures_are_bounded_and_alerted(monkeypatch, failure) -> None:
    alerts = []
    config = object()
    monkeypatch.setenv("VEDA_ALERT_WEBHOOK_URL", "https://alerts.example/producer")
    monkeypatch.setattr(production_service, "_PRODUCER_ATTEMPT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        production_service, "webhook_alerter",
        lambda *_a, **_k: lambda event, details: alerts.append((event, details)),
    )

    def load(_path):
        if failure == "configuration":
            raise ValueError("invalid configuration")
        return config

    def create(value, *, alert):
        assert value is config
        assert alert is not None
        if failure == "restore":
            raise RuntimeError("storage bootstrap failed")
        time.sleep(1)
        raise AssertionError("blocking bootstrap was not interrupted")

    monkeypatch.setattr(production_service, "load_producer_config", load)
    monkeypatch.setattr(production_service, "ProducerService", create)
    started = time.monotonic()
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        production_service.run_producer("producer.json")

    assert time.monotonic() - started < 0.5
    expected_type = {
        "configuration": "ValueError",
        "restore": "RuntimeError",
        "timeout": "TimeoutError",
    }[failure]
    assert alerts == [("PRODUCER_STARTUP_FAILURE", {"error_type": expected_type})]


def test_continuous_producer_starts_without_external_webhook(monkeypatch, caplog):
    monkeypatch.delenv("VEDA_ALERT_WEBHOOK_URL", raising=False)
    config = object()
    runs = []
    monkeypatch.setattr(production_service, "load_producer_config", lambda _path: config)

    def create(value, *, alert):
        assert value is config and callable(alert)

        def run(*, once):
            runs.append(once)
            alert("PRODUCER_PLAN_FAILURE", {"attempts": 5})

        return SimpleNamespace(run=run)

    monkeypatch.setattr(production_service, "ProducerService", create)
    production_service.run_producer("producer.json")
    assert runs == [False]
    assert "push alerts are not configured" in caplog.text
    assert "PRODUCER_PLAN_FAILURE" in caplog.text


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_producer_bootstrap_guard_ends_before_running_batches(monkeypatch, runtime_failure):
    alerts = []
    calls = []
    monkeypatch.setenv("VEDA_ALERT_WEBHOOK_URL", "https://alerts.example/producer")
    monkeypatch.setattr(
        production_service, "webhook_alerter",
        lambda *_a, **_k: lambda event, details: alerts.append((event, details)),
    )
    monkeypatch.setattr(production_service, "load_producer_config", lambda _path: "config")

    def run(*, once):
        assert production_service.signal.getitimer(production_service.signal.ITIMER_REAL) == (0, 0)
        calls.append(once)
        if runtime_failure:
            raise RuntimeError("batch failure, handled by its existing path")

    def create(config, *, alert):
        assert config == "config" and alert is not None
        return SimpleNamespace(run=run)

    monkeypatch.setattr(production_service, "ProducerService", create)
    if runtime_failure:
        with pytest.raises(RuntimeError, match="batch failure"):
            production_service.run_producer("producer.json", once=True)
    else:
        production_service.run_producer("producer.json", once=True)
    assert calls == [True]
    assert alerts == []


def test_cached_producer_worker_does_not_construct_teacher(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "prior")
    plan = SimpleNamespace(plan_digest="ab" * 32)
    rollout = SimpleNamespace(rollout_id="cd" * 32)
    config = SimpleNamespace(work_dir=tmp_path)
    tasks, results = queue.Queue(), queue.Queue()
    tasks.put((plan, rollout))
    tasks.put(None)
    validated = []

    def find(root, received_plan, received_rollout, track, *, dense_device):
        validated.append((received_plan, received_rollout, dense_device))
        return root / "cached.npz"

    def forbidden_teacher(**_kwargs):
        raise AssertionError("valid completed units must not load the teacher")

    monkeypatch.setattr(production_service, "_find_valid_shard", find)
    monkeypatch.setattr("validator.production.runtime.DenseRolloutRuntime", forbidden_teacher)
    production_service._worker_loop(2, config, "track", tasks, results)

    kind, result = results.get_nowait()
    assert kind == "ok"
    assert result.rollout_id == rollout.rollout_id
    assert result.elapsed_seconds == 0
    assert validated == [(plan, rollout, "cuda:0")]


def test_producer_worker_constructs_teacher_once_on_first_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "prior")
    calls = []
    instance = object()
    config = SimpleNamespace(
        artifact_path=lambda value: tmp_path / value,
        runtime=SimpleNamespace(
            model_root="model", adapter="adapter", predictor="predictor", sample_cache="samples",
            offload_blocks=0, prefetch=1, mlp_chunk_rows=1,
        ),
    )
    tasks, results = queue.Queue(), queue.Queue()
    tasks.put(("plan", "first"))
    tasks.put(("plan", "second"))
    tasks.put(None)

    def construct(**_kwargs):
        calls.append("construct")
        return instance

    def produce(get_runtime, received_config, track, plan, rollout):
        calls.append(rollout)
        assert get_runtime() is instance
        assert get_runtime() is instance
        return rollout

    monkeypatch.setattr("validator.production.runtime.DenseRolloutRuntime", construct)
    monkeypatch.setattr(production_service, "_produce_unit", produce)
    production_service._worker_loop(2, config, "track", tasks, results)

    assert calls == ["first", "construct", "second"]
    assert [results.get_nowait(), results.get_nowait()] == [("ok", "first"), ("ok", "second")]


def test_producer_worker_startup_failure_is_counted_and_alerted(monkeypatch, tmp_path) -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )

    class Store:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def create_next(self, **_kwargs):
            return None

        def plans(self):
            return (("plans/120.json", StoredObject(plan.encode(), "etag"), plan),)

    def failed_pool(*_args, **_kwargs):
        raise RuntimeError("gpu worker could not start")

    objects = AttemptObjects()
    alerts = []
    service = object.__new__(ProducerService)
    service.config = SimpleNamespace(
        validator_database="unused.db", poll_seconds=0.01, work_dir=tmp_path,
    )
    service.track = SimpleNamespace(digest=plan.track_spec_digest)
    service.objects = objects
    service.pool = None
    service.alert = lambda event, details: alerts.append((event, details))
    service.attempt_timeout_seconds = 10
    monkeypatch.setattr(
        production_service, "ChallengeStore", lambda *_args, **_kwargs: Store()
    )
    monkeypatch.setattr(production_service, "PlanCoordinator", Coordinator)
    monkeypatch.setattr(production_service, "WorkerPool", failed_pool)

    monkeypatch.setattr(Store, "chain_cursor", lambda _self: (120, "0x" + "cd" * 32),
                        raising=False)
    with pytest.raises(RuntimeError, match="producer attempt failed"):
        service._run_locked(once=True)

    assert _plan_attempt_count(objects, plan) == 1
    assert alerts[0][0] == "PRODUCER_PLAN_FAILURE"
    assert alerts[0][1]["attempt"] == 1
    assert "could not start" in alerts[0][1]["error"]


@pytest.mark.parametrize("slow_step", ["local-storage", "assembly"])
def test_producer_attempt_deadline_covers_post_generation_work(
    tmp_path, monkeypatch, slow_step
) -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(production_service.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        production_service,
        "_ready_file",
        lambda _path, prefix, suffix: ReadyObject(
            f"{prefix}/artifact{suffix}", "12" * 32, 1
        ),
    )
    monkeypatch.setattr(
        production_service,
        "_build_selected_sample_cache",
        lambda *_args: (ready_sample_cache(), ()),
    )

    class Workers:
        def run(self, _plan, *, deadline):
            assert deadline == 10.0
            return (UnitResult(plan.rollouts[0].rollout_id, "shard", "video", "latent", 1, 0),)

    def retain(*_args):
        if slow_step == "local-storage":
            clock["now"] = 11.0

    def assemble(*_args, **_kwargs):
        clock["now"] = 11.0
        return SimpleNamespace(path=tmp_path / "never-used.npz", sha256="34" * 32)

    if slow_step == "assembly":
        monkeypatch.setattr(production_service, "assemble_exact_cache_batch", assemble)
    service = object.__new__(ProducerService)
    service._retain_content = retain
    service.config = SimpleNamespace(work_dir=tmp_path)
    service.track = SimpleNamespace(
        digest="ab" * 32,
        artifact_manifest_sha256="34" * 32,
        source_pool_merkle_root="56" * 32,
        cache=SimpleNamespace(exact_max_bytes=10**9),
    )

    with pytest.raises(RuntimeError, match="attempt exceeded"):
        service._execute(
            "plans/120.json",
            StoredObject(plan.encode(), "etag"),
            plan,
            Workers(),
            deadline=10.0,
        )


@pytest.mark.parametrize("blocking_step", ["local-storage", "assembly"])
def test_producer_attempt_deadline_interrupts_blocking_post_generation_work(
    tmp_path, monkeypatch, blocking_step
) -> None:
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    monkeypatch.setattr(
        production_service,
        "_ready_file",
        lambda _path, prefix, suffix: ReadyObject(
            f"{prefix}/artifact{suffix}", "12" * 32, 1
        ),
    )
    monkeypatch.setattr(
        production_service,
        "_build_selected_sample_cache",
        lambda *_args: (ready_sample_cache(), ()),
    )

    class Workers:
        def run(self, _plan, *, deadline):
            assert deadline > time.monotonic()
            return (UnitResult(plan.rollouts[0].rollout_id, "shard", "video", "latent", 1, 0),)

    def retain(*_args):
        if blocking_step == "local-storage":
            time.sleep(1)

    def assemble(*_args, **_kwargs):
        time.sleep(1)
        raise AssertionError("blocking assembly was not interrupted")

    if blocking_step == "assembly":
        monkeypatch.setattr(production_service, "assemble_exact_cache_batch", assemble)
    service = object.__new__(ProducerService)
    service._retain_content = retain
    service.config = SimpleNamespace(work_dir=tmp_path)
    service.track = SimpleNamespace(
        digest="ab" * 32,
        artifact_manifest_sha256="34" * 32,
        source_pool_merkle_root="56" * 32,
        cache=SimpleNamespace(exact_max_bytes=10**9),
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="attempt exceeded"):
        service._execute(
            "plans/120.json",
            StoredObject(plan.encode(), "etag"),
            plan,
            Workers(),
            deadline=started + 0.05,
        )

    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize("dense_mutation", ["plus100", "lsb", "signed_zero"])
def test_completed_unit_recovery_requires_bound_valid_artifacts(
    tmp_path, monkeypatch, dense_mutation
) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=tuple(pool.proof(f"source-{index}") for index in range(2)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    rollout = plan.rollouts[0]
    step = plan.selected_steps[0]
    route = plan.selected_routes[0]
    segments = np.array([1, 0, 0, 0, 0, 2], dtype=np.int8)
    selected = select_query_tiles(
        query_selection_seed=plan.query_selection_seed,
        case_id=f"{rollout.rollout_id}:{step.index}:{route}",
        target_tiles=(1, 2, 3, 4),
        reference_tiles=(0,),
        count=4,
    )
    required = np.zeros((4, 6), dtype=np.bool_)
    for row, query_id in enumerate(selected):
        required[row, query_id] = True
    valid = np.full(6, 2, dtype=np.int16)
    case = ExactCase(
        route=route,
        stratum=rollout.stratum,
        rollout_id=rollout.rollout_id,
        source_id=rollout.source_id,
        step_index=step.index,
        step_window=step.window,
        scheduler_timestep=step.sigma,
        sigma=step.sigma,
        tile_plan_hash="45" * 32,
        query_tile_ids=selected,
        q=np.zeros((4, 2, 2), dtype=np.float32),
        k=np.zeros((6, 2, 2), dtype=np.float32),
        v=np.zeros((6, 2, 2), dtype=np.float32),
        dense=np.zeros((4, 2, 2), dtype=np.float32),
        query_valid_counts=valid[list(selected)],
        key_valid_counts=valid,
        key_segments=segments,
        required_keys=required,
        epsilon=1e-6,
    )
    shard = write_exact_cache_batch(
        tmp_path,
        (case,),
        generated_block=plan.generation_block,
        seed_block_hash=plan.block_hash,
        plan_digest=plan.plan_digest,
        track_spec_digest=plan.track_spec_digest,
        artifact_manifest_sha256="67" * 32,
        source_pool_merkle_root=pool.root,
        source_proofs=tuple(item.document() for item in plan.source_proofs),
        dense_references=(),
        sample_cache={},
    )
    video = tmp_path / f"{rollout.rollout_id}.mp4"
    latent = tmp_path / f"{rollout.rollout_id}.safetensors"
    video.write_bytes(b"valid encoded video")
    save_file(
        {
            "audio_rows": np.zeros((2, 32), dtype=np.float32),
            "video_rows": np.zeros((2, 96), dtype=np.float32),
            **{
                trajectory_tensor_key(item.index): np.zeros((2, 96), dtype=np.float32)
                for item in plan.selected_steps
            },
        },
        latent,
        metadata={"geometry": rollout.stratum, "rollout_id": rollout.rollout_id},
    )
    _write_unit_completion(tmp_path, plan, rollout, shard.path, video, latent)
    monkeypatch.setattr(production_service, "_video_decodes", lambda _path: True)
    track = SimpleNamespace(
        digest=plan.track_spec_digest,
        cache=SimpleNamespace(exact_max_bytes=10**7),
    )

    assert _find_valid_shard(tmp_path, plan, rollout, track) == shard.path

    corrupted_dense = case.dense.astype(np.float16)
    if dense_mutation == "plus100":
        corrupted_dense += np.float16(100)
    else:
        corrupted_dense.view(np.uint16).flat[0] ^= np.uint16(
            1 if dense_mutation == "lsb" else 0x8000
        )
    inconsistent = write_exact_cache_batch(
        tmp_path,
        (replace(case, dense=corrupted_dense),),
        generated_block=plan.generation_block,
        seed_block_hash=plan.block_hash,
        plan_digest=plan.plan_digest,
        track_spec_digest=plan.track_spec_digest,
        artifact_manifest_sha256="67" * 32,
        source_pool_merkle_root=pool.root,
        source_proofs=tuple(item.document() for item in plan.source_proofs),
        dense_references=(),
        sample_cache={},
    )
    _write_unit_completion(tmp_path, plan, rollout, inconsistent.path, video, latent)
    assert _find_valid_shard(tmp_path, plan, rollout, track) is None
    _write_unit_completion(tmp_path, plan, rollout, shard.path, video, latent)

    video.write_bytes(b"broken encoded video")
    assert _find_valid_shard(tmp_path, plan, rollout, track) is None

    video.write_bytes(b"valid encoded video")
    save_file(
        {"wrong_rows": np.zeros((2, 96), dtype=np.float32)},
        latent,
        metadata={"geometry": rollout.stratum, "rollout_id": rollout.rollout_id},
    )
    _write_unit_completion(tmp_path, plan, rollout, shard.path, video, latent)
    assert _find_valid_shard(tmp_path, plan, rollout, track) is None


@pytest.mark.parametrize("dense_mutation", ["plus100", "lsb", "signed_zero"])
def test_exact_cases_are_bound_field_by_field_to_the_batch_plan(
    tmp_path, dense_mutation
) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    planned = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=tuple(pool.proof(f"source-{index}") for index in range(2)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    segments = np.array([1, 0, 0, 0, 0, 2], dtype=np.int8)

    def make_case(route, rollout, step):
        case_id = f"{rollout.rollout_id}:{step.index}:{route}"
        selected = select_query_tiles(
            query_selection_seed=planned.query_selection_seed,
            case_id=case_id,
            target_tiles=(1, 2, 3, 4),
            reference_tiles=(0,),
            count=4,
        )
        required = np.zeros((4, 6), dtype=np.bool_)
        for row, query_id in enumerate(selected):
            required[row, query_id] = True
        valid = np.full(6, 2, dtype=np.int16)
        return ExactCase(
            route=route,
            stratum=rollout.stratum,
            rollout_id=rollout.rollout_id,
            source_id=rollout.source_id,
            step_index=step.index,
            step_window=step.window,
            scheduler_timestep=step.sigma,
            sigma=step.sigma,
            tile_plan_hash="45" * 32,
            query_tile_ids=selected,
            q=np.zeros((4, 2, 2), dtype=np.float32),
            k=np.zeros((6, 2, 2), dtype=np.float32),
            v=np.zeros((6, 2, 2), dtype=np.float32),
            dense=np.zeros((4, 2, 2), dtype=np.float32),
            query_valid_counts=valid[list(selected)],
            key_valid_counts=valid,
            key_segments=segments,
            required_keys=required,
            epsilon=1e-6,
        )

    cases = tuple(
        make_case(route, rollout, step)
        for route in planned.selected_routes
        for rollout in planned.rollouts
        for step in planned.selected_steps
    )
    references = tuple(
        DenseReference(
            rollout.rollout_id,
            ReadyObject(f"videos/{rollout.rollout_id}.mp4", "12" * 32, 1),
            ReadyObject(f"latents/{rollout.rollout_id}.safetensors", "23" * 32, 1),
        )
        for rollout in planned.rollouts
    )

    def write(directory, values, *, sample_cache=None):
        return write_exact_cache_batch(
            directory,
            values,
            generated_block=planned.generation_block,
            seed_block_hash=planned.block_hash,
            plan_digest=planned.plan_digest,
            track_spec_digest=planned.track_spec_digest,
            artifact_manifest_sha256="34" * 32,
            source_pool_merkle_root=pool.root,
            source_proofs=tuple(item.document() for item in planned.source_proofs),
            dense_references=tuple(item.document() for item in references),
            sample_cache=(
                ready_sample_cache().document() if sample_cache is None else sample_cache
            ),
        )

    track = SimpleNamespace(
        digest=planned.track_spec_digest,
        artifact_manifest_sha256="34" * 32,
        source_pool_merkle_root=pool.root,
        cache=SimpleNamespace(exact_max_bytes=10_000_000),
        scoring=scoring(),
        model=SimpleNamespace(selector=SimpleNamespace(head_dim=2)),
    )
    batch = write(tmp_path / "valid", cases)
    ready = planned.ready(
        exact=ReadyObject("exact/valid.npz", batch.sha256, batch.path.stat().st_size),
        dense_references=references,
        sample_cache=ready_sample_cache(),
    )
    assert batch.sample_cache == ready.sample_cache.document()
    _validate_plan_case_binding(batch.path, ready, track)

    substituted_inputs = replace(
        ready,
        sample_cache=ReadySampleCache(
            ReadyObject("inputs/substitute.json", "90" * 32, 10),
            ready_sample_cache().text,
        ),
    )
    assert substituted_inputs.plan_digest == ready.plan_digest
    substituted_batch = write(
        tmp_path / "substituted-cache",
        cases,
        sample_cache=substituted_inputs.sample_cache.document(),
    )
    assert substituted_batch.plan_digest == batch.plan_digest
    assert substituted_batch.sha256 != batch.sha256
    with pytest.raises(CacheLoadError, match="selected inputs"):
        _validate_plan_case_binding(batch.path, substituted_inputs, track)

    changed = (replace(cases[0], source_id="wrong-source"), *cases[1:])
    invalid = write(tmp_path / "invalid", changed)
    invalid_ready = planned.ready(
        exact=ReadyObject("exact/invalid.npz", invalid.sha256, invalid.path.stat().st_size),
        dense_references=references,
        sample_cache=ready_sample_cache(),
    )
    with pytest.raises(CacheLoadError, match="metadata conflicts"):
        _validate_plan_case_binding(invalid.path, invalid_ready, track)

    corrupted_dense = cases[0].dense.astype(np.float16)
    if dense_mutation == "plus100":
        corrupted_dense += np.float16(100)
    else:
        corrupted_dense.view(np.uint16).flat[0] ^= np.uint16(
            1 if dense_mutation == "lsb" else 0x8000
        )
    inconsistent = write(
        tmp_path / "inconsistent",
        (replace(cases[0], dense=corrupted_dense), *cases[1:]),
    )
    inconsistent_ready = planned.ready(
        exact=ReadyObject(
            "exact/inconsistent.npz",
            inconsistent.sha256,
            inconsistent.path.stat().st_size,
        ),
        dense_references=references,
        sample_cache=ready_sample_cache(),
    )

    source = ReadyBatchSource(
        track=track,
        objects=object(),
        cache_dir=tmp_path / "restored-inconsistent",
    )
    source.content.put(
        inconsistent.path, sha256=inconsistent.sha256,
        size_bytes=inconsistent.path.stat().st_size,
    )
    with pytest.raises(RuntimeError, match="restored exact batch is invalid"):
        source._restore_exact(inconsistent_ready)


def test_ready_batch_source_restores_only_the_challenge_bound_confirmation(tmp_path) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(8)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
        source_commitments=source_commitments(pool),
        track_start_block=100,
        scoring=scoring(),
        model=SimpleNamespace(selector=SimpleNamespace(routes=("route-0", "route-1", "route-2"))),
    )
    planned = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest=track.digest,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        source_proofs=select_source_proofs(
            pool, commitments=track.source_commitments, partition="production",
            strata=("16:9@37",), sources_per_stratum=2,
            batch_seed=generation_seed(track.digest, "0x" + "cd" * 32, "a5" * 32),
            inventory_slot=0, excluded_source_digests=frozenset(),
        ),
        scoring=track.scoring,
        available_routes=track.model.selector.routes,
    )
    payloads = {}
    references = []
    for index, rollout in enumerate(planned.rollouts):
        content = f"latent-{index}".encode()
        digest = hashlib.sha256(content).hexdigest()
        ready = ReadyObject(f"latents/{digest}.safetensors", digest, len(content))
        payloads[ready.object_key] = content
        video_content = f"video-{index}".encode()
        video_digest = hashlib.sha256(video_content).hexdigest()
        video = ReadyObject(f"videos/{video_digest}.mp4", video_digest, len(video_content))
        payloads[video.object_key] = video_content
        references.append(
            DenseReference(
                rollout.rollout_id,
                video,
                ready,
            )
        )
    index_content = json.dumps(
        {"samples": [{"id": proof.leaf.source_id} for proof in planned.source_proofs]}
    ).encode()
    index_digest = hashlib.sha256(index_content).hexdigest()
    index_object = ReadyObject("inputs/index.json", index_digest, len(index_content))
    payloads[index_object.object_key] = index_content
    text_content = b"selected-text-inputs"
    text_digest = hashlib.sha256(text_content).hexdigest()
    text_object = ReadyObject("inputs/text.safetensors", text_digest, len(text_content))
    payloads[text_object.object_key] = text_content
    ready_plan = planned.ready(
        exact=ReadyObject("exact/data.npz", "12" * 32, 10),
        dense_references=tuple(references),
        sample_cache=ReadySampleCache(index_object, text_object),
    )

    def make_objects(plan=ready_plan):
        objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="test")
        earlier = build_plan(
            track_spec_digest=track.digest, generation_block=119,
            block_hash="0x" + "ab" * 32, secret="b6" * 32, inventory_slot=1,
            source_proofs=select_source_proofs(
                pool, commitments=track.source_commitments, partition="production",
                strata=("16:9@37",), sources_per_stratum=2,
                batch_seed=generation_seed(track.digest, "0x" + "ab" * 32, "b6" * 32),
                inventory_slot=1, excluded_source_digests=frozenset(),
            ), scoring=track.scoring, available_routes=track.model.selector.routes,
        )
        earlier = earlier.ready(
            exact=plan.exact, sample_cache=plan.sample_cache,
            dense_references=tuple(
                replace(ref, rollout_id=rollout.rollout_id)
                for ref, rollout in zip(plan.dense_references, earlier.rollouts, strict=True)
            ),
        )
        objects.create_plan("plans/119.json", earlier.encode())
        objects.create_plan("plans/120.json", plan.encode())
        inventory = Inventory(objects, track.digest)
        for stage in ("SCORING", "CONFIRMING"):
            inventory.claim(
                challenge_id="a1" * 32, stage=stage, file_sha256="b2" * 32,
                model_digest="c3" * 32, plans=(earlier, plan),
            )
        return objects

    objects = make_objects()
    content_dir = tmp_path / "content"
    content_store = LocalBatchStore(content_dir)
    for key, content in payloads.items():
        path = tmp_path / "seed" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        content_store.put(path, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    source = ReadyBatchSource(
        track=track,
        objects=objects,
        cache_dir=tmp_path,
        batch_dir=content_dir,
    )
    exact_path = tmp_path / "exact-source.npz"
    exact_path.write_bytes(b"exact")
    source._restore_exact = lambda _plan: SimpleNamespace(sha256="12" * 32, path=exact_path)

    restored = source.confirmation_batch("a1" * 32)

    assert restored.plan == ready_plan
    assert restored.exact.sha256 == "12" * 32
    assert len(restored.latent_paths) == 2
    terminal_ids = frozenset({"a1" * 32})
    disclosure = source.public_disclosure(ready_plan.plan_digest, terminal_ids)
    assert disclosure is not None
    assert disclosure.challenge_ids == ("a1" * 32,)
    assert disclosure.record_name == f"batch_groups/{ready_plan.plan_digest}.json"
    assert len(disclosure.files) == 7
    with pytest.raises(RuntimeError, match="no allocated"):
        source.confirmation_batch("other")

    mismatched_track = SimpleNamespace(
        **{
            **track.__dict__,
            "source_pool_merkle_root": "ff" * 32,
        }
    )
    with pytest.raises(RuntimeError, match="source proof"):
        ReadyBatchSource(
            track=mismatched_track,
            objects=objects,
            cache_dir=tmp_path,
        ).public_disclosure(ready_plan.plan_digest, terminal_ids)

    wrong_namespace = replace(
        ready_plan,
        exact=replace(ready_plan.exact, object_key="source-pool/substituted.npz"),
    )
    with pytest.raises(RuntimeError, match="outside its runtime namespace"):
        ReadyBatchSource(
            track=track,
            objects=make_objects(wrong_namespace),
            cache_dir=tmp_path,
        ).public_disclosure(ready_plan.plan_digest, terminal_ids)

    exact = tmp_path / f"{ready_plan.generation_block}-{ready_plan.exact.sha256}.npz"
    exact.write_bytes(b"temporary")
    assert (tmp_path / "dense" / ready_plan.plan_digest).is_dir()
    assert source.prune_disclosed({"b2" * 32}) == 0
    assert source.prune_disclosed({"a1" * 32}) == 4
    assert not exact.exists()
    assert not (tmp_path / "dense" / ready_plan.plan_digest).exists()

    task = SimpleNamespace(
        challenge_id="exact-a",
        submission_id="submission-a",
        hotkey="hotkey-a",
        repo_id="owner/model",
        revision="12" * 20,
        file_sha256="34" * 32,
        state=ChallengeState.SCORING,
        position=ChainPosition(100, 0, 0),
    )
    result = SimpleNamespace(score=0.8, metrics={"requires_confirmation": True})
    source.archive_stage(task, result, 121)
    archived = source.restore_stage(task)
    assert archived is not None
    assert archived.score == 0.8
    assert archived.score_block == 121
    assert archived.metrics == {"requires_confirmation": True}


def test_admission_archive_restores_model_by_challenge_identity(tmp_path) -> None:
    class Objects:
        def __init__(self):
            self.small = {}
            self.content = {}

        def put_content(self, path, *, sha256, object_name):
            payload = Path(path).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == sha256
            self.content.setdefault(object_name, payload)

        def create_plan(self, name, payload):
            stored = StoredObject(payload, hashlib.sha256(payload).hexdigest())
            existing = self.small.setdefault(name, stored)
            if existing.body != payload:
                raise RuntimeError("conflicting archive")
            return existing

        def get_optional(self, name):
            return self.small.get(name)

        def download_content(self, object_name, destination, *, sha256, size_bytes):
            payload = self.content[object_name]
            assert len(payload) == size_bytes
            assert hashlib.sha256(payload).hexdigest() == sha256
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return path

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"validated selector")
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    validated = ValidatedModel(file_sha256, "cd" * 32, path.stat().st_size, 1)
    task = SimpleNamespace(
        challenge_id="challenge-a",
        submission_id="submission-a",
        hotkey="hotkey-a",
        repo_id="owner/model",
        revision="12" * 20,
    )
    objects = Objects()
    source = ReadyBatchSource(
        track=SimpleNamespace(digest="ab" * 32),
        objects=objects,
        cache_dir=tmp_path / "restored",
    )

    source.archive_model(task, path, validated)
    path.unlink()
    restored = source.restore_admission(task)

    assert restored is not None
    assert restored.path.read_bytes() == b"validated selector"
    assert restored.file_sha256 == file_sha256
    assert restored.download_host == "private-archive"


def test_source_pool_capacity_is_proved_before_production_starts() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(6)))
    track = SimpleNamespace(
        scoring=SimpleNamespace(
            rollouts_per_stratum=2,
            stratum_weight_map={"16:9@37": 1.0},
        ),
    )

    _validate_source_capacity(pool, track)
    track.scoring.rollouts_per_stratum = 4
    with pytest.raises(ValueError, match="two disjoint batches"):
        _validate_source_capacity(pool, track)


def test_source_capacity_is_independent_of_competition_duration() -> None:
    track = SimpleNamespace(
        scoring=SimpleNamespace(
            rollouts_per_stratum=4,
            stratum_weight_map={"16:9@37": 1.0},
        ),
    )
    _validate_source_capacity(
        SourcePool(tuple(proof(index).leaf for index in range(8))),
        track,
    )
    with pytest.raises(ValueError, match="two disjoint batches"):
        _validate_source_capacity(
            SourcePool(tuple(proof(index).leaf for index in range(7))),
            track,
        )
