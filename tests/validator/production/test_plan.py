import hashlib
import json
import queue
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
from validator.production.commitment import SourceLeaf, SourcePool, verify_source_proof
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    build_plan,
    generation_seed,
    scheduled_generation_block,
    select_query_tiles,
    select_source_digests,
    select_source_proofs,
    submission_interval,
    validate_source_selection_sequence,
)
from validator.production.runtime import DenseRolloutRuntime, trajectory_tensor_key
from validator.production.service import (
    PlanCoordinator,
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
from validator.production.storage import PreconditionFailed, StoredObject
from validator.queue import ChainPosition, ChallengeState


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
        season_false_promotion_alpha=0.05,
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


def test_batch_schedule_has_one_exact_submission_interval() -> None:
    assert (
        scheduled_generation_block(
            119,
            track_start_block=100,
            minimum_generation_delay_blocks=1,
            generation_interval_blocks=20,
        )
        == 120
    )
    assert submission_interval(
        120,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
    ) == (100, 119)
    assert submission_interval(
        140,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
    ) == (120, 139)


def test_plan_is_deterministic_and_only_transitions_to_ready_once() -> None:
    arguments = {
        "track_spec_digest": "ab" * 32,
        "generation_block": 120,
        "block_hash": "0x" + "cd" * 32,
        "track_start_block": 100,
        "minimum_generation_delay_blocks": 1,
        "generation_interval_blocks": 20,
        "exact_challenge_ids": ("exact-a",),
        "confirmation_challenge_ids": ("confirm-a",),
        "source_proofs": (proof(0), proof(1)),
        "scoring": scoring(),
        "available_routes": ("route-0", "route-1", "route-2"),
    }
    first = build_plan(**arguments)
    second = build_plan(**arguments)

    assert first == second
    assert first.submission_first_block == 100
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


def test_source_selection_never_reuses_a_family() -> None:
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
        strata=("16:9@37",),
        sources_per_stratum=1,
        batch_seed="12" * 32,
        used_source_digests=frozenset({first.digest}),
    )

    assert selected[0].leaf.source_family_id == "family-1"
    with pytest.raises(ValueError, match="exhausted"):
        select_source_proofs(
            pool,
            commitments=source_commitments(pool),
            strata=("16:9@37",),
            sources_per_stratum=1,
            batch_seed="12" * 32,
            used_source_digests=frozenset({first.digest, second.digest}),
        )


def test_source_selection_is_reproducible_from_the_public_index() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(4)))
    commitments = source_commitments(pool)
    selected = select_source_proofs(
        pool,
        commitments=commitments,
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed="34" * 32,
        used_source_digests=frozenset(),
    )

    expected = select_source_digests(
        commitments,
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed="34" * 32,
        used_source_digests=frozenset(),
    )

    assert tuple(item.leaf.digest for item in selected) == expected
    assert all(verify_source_proof(item, pool.root) for item in selected)


def test_source_selection_sequence_rejects_a_member_chosen_out_of_order() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(4)))
    commitments = source_commitments(pool)
    block_hash = "0x" + "cd" * 32
    selected = select_source_proofs(
        pool,
        commitments=commitments,
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed=generation_seed("ab" * 32, block_hash),
        used_source_digests=frozenset(),
    )
    arguments = {
        "track_spec_digest": "ab" * 32,
        "generation_block": 120,
        "block_hash": block_hash,
        "track_start_block": 100,
        "minimum_generation_delay_blocks": 1,
        "generation_interval_blocks": 20,
        "exact_challenge_ids": ("challenge",),
        "confirmation_challenge_ids": (),
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
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=(),
        confirmation_challenge_ids=(),
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
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=(),
        confirmation_challenge_ids=(),
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

    def create_plan(self, name, payload):
        existing = self.values.get(name)
        if existing is not None and existing != payload:
            raise PreconditionFailed("different bytes")
        self.values[name] = payload
        return self.get(name)


def test_producer_retry_ceiling_survives_process_and_host_loss(monkeypatch) -> None:
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
        source_proofs=(proof(0), proof(1)),
        scoring=scoring(),
        available_routes=("route-0", "route-1", "route-2"),
    )
    durable = {}

    for expected in range(1, 6):
        objects = AttemptObjects(durable)
        assert _reserve_plan_attempt(objects, plan) == expected
        assert _plan_attempt_count(AttemptObjects(durable), plan) == expected

    assert _reserve_plan_attempt(AttemptObjects(durable), plan) == 5
    assert len(durable) == 5

    class Store:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def create_next(self):
            return None

        def plans(self):
            return (("plans/120.json", StoredObject(plan.encode(), "etag"), plan),)

    service = object.__new__(ProducerService)
    service.config = SimpleNamespace(validator_database="unused.db", poll_seconds=0.01)
    service.track = SimpleNamespace(digest=plan.track_spec_digest)
    service.objects = AttemptObjects(durable)
    service.pool = None
    service.alert = None
    monkeypatch.setattr(
        production_service, "ChallengeStore", lambda *_args, **_kwargs: Store()
    )
    monkeypatch.setattr(production_service, "PlanCoordinator", Coordinator)
    with pytest.raises(RuntimeError, match="retry ceiling"):
        service._run_locked(once=True)


def test_producer_worker_startup_failure_is_counted_and_alerted(monkeypatch) -> None:
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
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

        def create_next(self):
            return None

        def plans(self):
            return (("plans/120.json", StoredObject(plan.encode(), "etag"), plan),)

    def failed_pool(*_args, **_kwargs):
        raise RuntimeError("gpu worker could not start")

    objects = AttemptObjects()
    alerts = []
    service = object.__new__(ProducerService)
    service.config = SimpleNamespace(validator_database="unused.db", poll_seconds=0.01)
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

    with pytest.raises(RuntimeError, match="retry ceiling"):
        service._run_locked(once=True)

    assert _plan_attempt_count(objects, plan) == 1
    assert alerts[0][0] == "PRODUCER_PLAN_FAILURE"
    assert alerts[0][1]["attempt"] == 1
    assert "could not start" in alerts[0][1]["error"]


@pytest.mark.parametrize("slow_step", ["upload", "assembly"])
def test_producer_attempt_deadline_covers_post_generation_work(
    tmp_path, monkeypatch, slow_step
) -> None:
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
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

    class Objects:
        def put_content(self, *_args, **_kwargs):
            if slow_step == "upload":
                clock["now"] = 11.0

    def assemble(*_args, **_kwargs):
        clock["now"] = 11.0
        return SimpleNamespace(path=tmp_path / "never-used.npz", sha256="34" * 32)

    if slow_step == "assembly":
        monkeypatch.setattr(production_service, "assemble_exact_cache_batch", assemble)
    service = object.__new__(ProducerService)
    service.objects = Objects()
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


@pytest.mark.parametrize("blocking_step", ["upload", "assembly"])
def test_producer_attempt_deadline_interrupts_blocking_post_generation_work(
    tmp_path, monkeypatch, blocking_step
) -> None:
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
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

    class Objects:
        def put_content(self, *_args, **_kwargs):
            if blocking_step == "upload":
                try:
                    time.sleep(1)
                except Exception as exc:
                    raise AssertionError("SDK swallowed the deadline interrupt") from exc

    def assemble(*_args, **_kwargs):
        time.sleep(1)
        raise AssertionError("blocking assembly was not interrupted")

    if blocking_step == "assembly":
        monkeypatch.setattr(production_service, "assemble_exact_cache_batch", assemble)
    service = object.__new__(ProducerService)
    service.objects = Objects()
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


def test_completed_unit_recovery_requires_bound_valid_artifacts(tmp_path, monkeypatch) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    plan = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
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
        selected_inputs={},
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

    inconsistent = write_exact_cache_batch(
        tmp_path,
        (replace(case, dense=case.dense + 100),),
        generated_block=plan.generation_block,
        seed_block_hash=plan.block_hash,
        plan_digest=plan.plan_digest,
        track_spec_digest=plan.track_spec_digest,
        artifact_manifest_sha256="67" * 32,
        source_pool_merkle_root=pool.root,
        source_proofs=tuple(item.document() for item in plan.source_proofs),
        dense_references=(),
        selected_inputs={},
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


def test_exact_cases_are_bound_field_by_field_to_the_batch_plan(tmp_path) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    planned = build_plan(
        track_spec_digest="ab" * 32,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
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

    def write(directory, values):
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
            selected_inputs=ready_sample_cache().document(),
        )

    track = SimpleNamespace(cache=SimpleNamespace(exact_max_bytes=10_000_000), scoring=scoring())
    batch = write(tmp_path / "valid", cases)
    ready = planned.ready(
        exact=ReadyObject("exact/valid.npz", batch.sha256, batch.path.stat().st_size),
        dense_references=references,
        sample_cache=ready_sample_cache(),
    )
    _validate_plan_case_binding(batch.path, ready, track)

    substituted_inputs = replace(
        ready,
        sample_cache=ReadySampleCache(
            ReadyObject("inputs/substitute.json", "90" * 32, 10),
            ready_sample_cache().text,
        ),
    )
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


def test_ready_batch_source_restores_only_the_challenge_bound_confirmation(tmp_path) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
        source_commitments=source_commitments(pool),
        track_start_block=100,
        cache=SimpleNamespace(generation_interval_blocks=20, minimum_generation_delay_blocks=1),
        scoring=scoring(),
        model=SimpleNamespace(selector=SimpleNamespace(routes=("route-0", "route-1", "route-2"))),
    )
    planned = build_plan(
        track_spec_digest=track.digest,
        generation_block=120,
        block_hash="0x" + "cd" * 32,
        track_start_block=100,
        minimum_generation_delay_blocks=1,
        generation_interval_blocks=20,
        exact_challenge_ids=("exact-a",),
        confirmation_challenge_ids=("confirm-a",),
        source_proofs=tuple(pool.proof(f"source-{index}") for index in range(2)),
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

    class Objects:
        def __init__(self):
            self.small = {}

        def list_names(self, _prefix):
            return ("plans/120.json",)

        def get(self, name):
            if name in self.small:
                return self.small[name]
            return StoredObject(ready_plan.encode(), "etag")

        def get_optional(self, name):
            return self.small.get(name)

        def create_plan(self, name, payload):
            stored = StoredObject(payload, hashlib.sha256(payload).hexdigest())
            existing = self.small.setdefault(name, stored)
            if existing.body != payload:
                raise RuntimeError("conflicting archive")
            return existing

        def download_content(self, object_name, destination, *, sha256, size_bytes):
            content = payloads[object_name]
            assert len(content) == size_bytes
            assert hashlib.sha256(content).hexdigest() == sha256
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return path

    objects = Objects()
    source = ReadyBatchSource(
        track=track,
        objects=objects,
        cache_dir=tmp_path,
    )
    exact_path = tmp_path / "exact-source.npz"
    exact_path.write_bytes(b"exact")
    source._restore_exact = lambda _plan: SimpleNamespace(sha256="12" * 32, path=exact_path)

    restored = source.confirmation_batch("confirm-a")

    assert restored.plan == ready_plan
    assert restored.exact.sha256 == "12" * 32
    assert len(restored.latent_paths) == 2
    assert source.batch_challenge_ids(ready_plan.plan_digest) == (
        "exact-a",
        "confirm-a",
    )
    disclosure = source.public_disclosure(ready_plan.plan_digest)
    assert disclosure.challenge_ids == ("exact-a", "confirm-a")
    assert disclosure.record_name == f"batch_groups/{ready_plan.plan_digest}.json"
    assert len(disclosure.files) == 7
    with pytest.raises(RuntimeError, match="no scheduled"):
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
            objects=Objects(),
            cache_dir=tmp_path,
        ).batch_challenge_ids(ready_plan.plan_digest)

    exact = tmp_path / f"{ready_plan.generation_block}-{ready_plan.exact.sha256}.npz"
    exact.write_bytes(b"temporary")
    assert (tmp_path / "dense" / ready_plan.plan_digest).is_dir()
    assert source.prune_disclosed({"exact-a"}) == 0
    assert source.prune_disclosed({"exact-a", "confirm-a"}) == 4
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


def test_plan_coordinator_skips_empty_intervals_and_rejects_a_conflicting_winner() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(4)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
        source_commitments=source_commitments(pool),
        track_start_block=100,
        maximum_generation_batches=2,
        maximum_effective_challenges=2,
        cache=SimpleNamespace(
            generation_interval_blocks=20,
            minimum_generation_delay_blocks=1,
        ),
        scoring=scoring(),
        model=SimpleNamespace(selector=SimpleNamespace(routes=("route-0", "route-1"))),
    )

    class Store:
        admission_complete = False

        def admission_interval_complete(self, _first, _last):
            return self.admission_complete

        def chain_cursor(self):
            return (140, "0x" + "ef" * 32)

        def finalized_block_hash(self, block):
            return "0x" + f"{block:064x}"

        def admitted_challenge_ids(self, first, last):
            return ("exact-a",) if (first, last) == (120, 139) else ()

        def confirmation_challenge_ids(self, _first, _last):
            return ()

    class Objects:
        def __init__(self):
            self.payload = None

        def list_names(self, _prefix):
            return ()

        def create_plan(self, name, payload):
            self.payload = (name, payload)
            return StoredObject(payload, "etag")

    store = Store()
    objects = Objects()
    coordinator = PlanCoordinator(
        track=track,
        store=store,
        objects=objects,
        source_pool=pool,
    )

    assert coordinator.create_next() is None
    assert objects.payload is None

    store.admission_complete = True
    plan = coordinator.create_next()

    assert plan is not None
    assert plan.generation_block == 140
    assert plan.exact_challenge_ids == ("exact-a",)
    assert objects.payload is not None and objects.payload[0] == "plans/140.json"

    conflicting = build_plan(
        track_spec_digest=track.digest,
        generation_block=plan.generation_block,
        block_hash=plan.block_hash,
        track_start_block=track.track_start_block,
        minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
        generation_interval_blocks=track.cache.generation_interval_blocks,
        exact_challenge_ids=("different-challenge",),
        confirmation_challenge_ids=(),
        source_proofs=plan.source_proofs,
        scoring=track.scoring,
        available_routes=track.model.selector.routes,
    )

    class ConflictObjects(Objects):
        def create_plan(self, _name, _payload):
            raise PreconditionFailed("lost conditional create")

        def get(self, _name):
            return StoredObject(conflicting.encode(), "winner-etag")

    with pytest.raises(RuntimeError, match="deterministic schedule"):
        PlanCoordinator(
            track=track,
            store=store,
            objects=ConflictObjects(),
            source_pool=pool,
        ).create_next()

    recovered = plan.ready(
        exact=ReadyObject("exact/batch.npz", "12" * 32, 100),
        dense_references=tuple(
            DenseReference(
                rollout.rollout_id,
                ReadyObject(f"videos/{index}.mp4", f"{index + 1:064x}", 100),
                ReadyObject(f"latents/{index}.safetensors", f"{index + 3:064x}", 200),
            )
            for index, rollout in enumerate(plan.rollouts)
        ),
        sample_cache=ready_sample_cache(),
    )

    class RecoveryObjects(ConflictObjects):
        def get(self, _name):
            return StoredObject(recovered.encode(), "winner-etag")

    assert (
        PlanCoordinator(
            track=track,
            store=store,
            objects=RecoveryObjects(),
            source_pool=pool,
        ).create_next()
        == recovered
    )


def test_plan_coordinator_maps_confirmation_from_its_exact_decision_block() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
        source_commitments=source_commitments(pool),
        track_start_block=100,
        maximum_generation_batches=2,
        maximum_effective_challenges=2,
        cache=SimpleNamespace(
            generation_interval_blocks=20,
            minimum_generation_delay_blocks=1,
        ),
        scoring=scoring(),
        model=SimpleNamespace(selector=SimpleNamespace(routes=("route-0", "route-1"))),
    )

    class Store:
        def admission_interval_complete(self, _first, _last):
            return True

        def chain_cursor(self):
            return (140, "0x" + "ef" * 32)

        def finalized_block_hash(self, block):
            return "0x" + f"{block:064x}"

        def admitted_challenge_ids(self, _first, _last):
            return ()

        def confirmation_challenge_ids(self, first, last):
            return ("confirm-a",) if (first, last) == (120, 139) else ()

    class Objects:
        def __init__(self):
            self.payload = None

        def list_names(self, _prefix):
            return ()

        def create_plan(self, name, payload):
            self.payload = (name, payload)
            return StoredObject(payload, "etag-next")

    objects = Objects()
    coordinator = PlanCoordinator(
        track=track,
        store=Store(),
        objects=objects,
        source_pool=pool,
    )

    plan = coordinator.create_next()

    assert plan is not None
    assert plan.generation_block == 140
    assert plan.exact_challenge_ids == ()
    assert plan.confirmation_challenge_ids == ("confirm-a",)


def test_plan_coordinator_waits_for_private_stage_recovery() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    challenge_id = "12" * 32
    archived = {
        "format": "veda-stage-result-v1",
        "track_spec_digest": "ab" * 32,
        "challenge_id": challenge_id,
        "submission_id": "submission-a",
        "hotkey": "hotkey-a",
        "repo_id": "owner/repo",
        "revision": "34" * 20,
        "file_sha256": "56" * 32,
        "stage": ChallengeState.SCORING.value,
        "score": 0.8,
        "score_block": 125,
        "metrics": {"cache_generated_block": 120, "requires_confirmation": True},
    }
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
        source_commitments=source_commitments(pool),
        track_start_block=100,
        maximum_generation_batches=2,
        maximum_effective_challenges=2,
        cache=SimpleNamespace(
            generation_interval_blocks=20,
            minimum_generation_delay_blocks=1,
        ),
        scoring=scoring(),
        model=SimpleNamespace(selector=SimpleNamespace(routes=("route-0", "route-1"))),
    )

    class Store:
        score_recovered = False
        transition_applied = False

        def admission_interval_complete(self, _first, _last):
            return True

        def stage_score(self, identifier, stage):
            assert identifier == challenge_id
            assert stage is ChallengeState.SCORING
            if not self.score_recovered:
                return None
            return SimpleNamespace(
                score=archived["score"],
                score_block=archived["score_block"],
                metrics=archived["metrics"],
            )

        def get_challenge(self, identifier):
            assert identifier == challenge_id
            state = (
                ChallengeState.CONFIRMING
                if self.transition_applied
                else ChallengeState.SCORING
            )
            return SimpleNamespace(state=state)

        def stage_transition_applied(self, identifier, stage):
            assert identifier == challenge_id
            assert stage is ChallengeState.SCORING
            return self.transition_applied

        def chain_cursor(self):
            return (140, "0x" + "ef" * 32)

        def finalized_block_hash(self, block):
            return "0x" + f"{block:064x}"

        def admitted_challenge_ids(self, _first, _last):
            return ()

        def confirmation_challenge_ids(self, first, last):
            if self.transition_applied and (first, last) == (120, 139):
                return (challenge_id,)
            return ()

    class Objects:
        plan = None

        def list_names(self, prefix):
            if prefix == "stages/":
                return (f"stages/{challenge_id}/SCORING.json",)
            return ()

        def get(self, name):
            assert name == f"stages/{challenge_id}/SCORING.json"
            return StoredObject(json.dumps(archived).encode(), "stage-etag")

        def create_plan(self, name, payload):
            self.plan = (name, payload)
            return StoredObject(payload, "plan-etag")

    store = Store()
    objects = Objects()
    coordinator = PlanCoordinator(
        track=track,
        store=store,
        objects=objects,
        source_pool=pool,
    )

    assert coordinator.create_next() is None
    assert objects.plan is None

    store.score_recovered = True
    assert coordinator.create_next() is None
    assert objects.plan is None

    store.transition_applied = True
    plan = coordinator.create_next()

    assert plan is not None
    assert plan.generation_block == 140
    assert plan.confirmation_challenge_ids == (challenge_id,)


def test_terminal_public_stage_projection_matches_private_archive() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    challenge_id = "12" * 32
    public_metrics = {
        "incumbent_score": 0.7,
        "paired_gain_lower_bound": 0.02,
        "cache_sha256": "34" * 32,
        "cache_generated_block": 120,
        "cache_plan_digest": "56" * 32,
        "cache_seed_block_hash": "0x" + "78" * 32,
    }
    archived = {
        "format": "veda-stage-result-v1",
        "track_spec_digest": "ab" * 32,
        "challenge_id": challenge_id,
        "submission_id": "submission-a",
        "hotkey": "hotkey-a",
        "repo_id": "owner/repo",
        "revision": "34" * 20,
        "file_sha256": "56" * 32,
        "stage": ChallengeState.SCORING.value,
        "score": 0.8,
        "score_block": 125,
        "metrics": {
            **public_metrics,
            "case_count": 384,
            "exact_score": 0.8,
            "requires_confirmation": True,
        },
    }
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
        source_commitments=source_commitments(pool),
    )

    class Store:
        state = ChallengeState.SUCCEEDED

        def stage_score(self, identifier, stage):
            assert identifier == challenge_id
            assert stage is ChallengeState.SCORING
            return SimpleNamespace(
                score=archived["score"],
                score_block=archived["score_block"],
                metrics=public_metrics,
            )

        def get_challenge(self, identifier):
            assert identifier == challenge_id
            return SimpleNamespace(state=self.state)

        def stage_transition_applied(self, identifier, stage):
            assert identifier == challenge_id
            assert stage is ChallengeState.SCORING
            return True

    class Objects:
        def list_names(self, prefix):
            assert prefix == "stages/"
            return (f"stages/{challenge_id}/SCORING.json",)

        def get(self, name):
            assert name == f"stages/{challenge_id}/SCORING.json"
            return StoredObject(json.dumps(archived).encode(), "stage-etag")

    store = Store()
    coordinator = PlanCoordinator(
        track=track,
        store=store,
        objects=Objects(),
        source_pool=pool,
    )

    assert coordinator._stage_recovery_complete()

    store.state = ChallengeState.CONFIRMING
    with pytest.raises(RuntimeError, match="conflicts with its private archive"):
        coordinator._stage_recovery_complete()


def test_source_pool_capacity_is_proved_before_production_starts() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(6)))
    track = SimpleNamespace(
        maximum_generation_batches=2,
        maximum_effective_challenges=1,
        scoring=SimpleNamespace(
            rollouts_per_stratum=2,
            stratum_weight_map={"16:9@37": 1.0},
        ),
    )

    _validate_source_capacity(pool, track)
    track.maximum_generation_batches = 2
    track.scoring.rollouts_per_stratum = 3
    with pytest.raises(ValueError, match="reserve batch"):
        _validate_source_capacity(pool, track)


def test_source_capacity_counts_only_possible_nonempty_plans() -> None:
    track = SimpleNamespace(
        maximum_generation_batches=80,
        maximum_effective_challenges=20,
        scoring=SimpleNamespace(
            rollouts_per_stratum=4,
            stratum_weight_map={"16:9@37": 1.0},
        ),
    )
    _validate_source_capacity(
        SourcePool(tuple(proof(index).leaf for index in range(164))),
        track,
    )
    with pytest.raises(ValueError, match="reserve batch"):
        _validate_source_capacity(
            SourcePool(tuple(proof(index).leaf for index in range(163))),
            track,
        )
