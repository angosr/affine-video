import hashlib
import queue
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scoring.cache import CacheLoadError, write_exact_cache_batch
from scoring.exact import ExactCase
from scoring.model import ValidatedModel
from scoring.spec import BudgetSpec, ScoringSpec, StepWindowSpec, TrajectoryScheduleSpec
from validator.production.batches import ReadyBatchSource, _validate_plan_case_binding
from validator.production.commitment import SourceLeaf, SourcePool
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    build_plan,
    scheduled_generation_block,
    select_query_tiles,
    select_source_proofs,
    submission_interval,
)
from validator.production.service import (
    PlanCoordinator,
    WorkerPool,
    _prune_local_work,
    _validate_source_capacity,
)
from validator.production.storage import StoredObject
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


def proof(index: int = 0):
    leaf = SourceLeaf(
        "production",
        f"family-{index}",
        f"source-{index}",
        f"prompt {index}",
        (),
        "16:9@37",
        "internal-cleared-v1",
    )
    return SourcePool((leaf,)).proof(f"source-{index}")


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
    )
    pool = SourcePool((first, second))

    selected = select_source_proofs(
        pool,
        strata=("16:9@37",),
        sources_per_stratum=1,
        batch_seed="12" * 32,
        used_families=frozenset({"family-0"}),
    )

    assert selected[0].leaf.source_family_id == "family-1"
    with pytest.raises(ValueError, match="exhausted"):
        select_source_proofs(
            pool,
            strata=("16:9@37",),
            sources_per_stratum=1,
            batch_seed="12" * 32,
            used_families=frozenset({"family-0", "family-1"}),
        )


def test_source_selection_skips_duplicate_families_without_losing_quota() -> None:
    leaves = (
        SourceLeaf(
            "production",
            "family-a",
            "source-a-1",
            "prompt a1",
            (),
            "16:9@37",
            "internal-cleared-v1",
        ),
        SourceLeaf(
            "production",
            "family-a",
            "source-a-2",
            "prompt a2",
            (),
            "16:9@37",
            "internal-cleared-v1",
        ),
        SourceLeaf(
            "production",
            "family-b",
            "source-b",
            "prompt b",
            (),
            "16:9@37",
            "internal-cleared-v1",
        ),
    )
    selected = select_source_proofs(
        SourcePool(leaves),
        strata=("16:9@37",),
        sources_per_stratum=2,
        batch_seed="34" * 32,
        used_families=frozenset(),
    )

    assert {item.leaf.source_family_id for item in selected} == {"family-a", "family-b"}


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
        )

    track = SimpleNamespace(cache=SimpleNamespace(exact_max_bytes=10_000_000), scoring=scoring())
    batch = write(tmp_path / "valid", cases)
    ready = planned.ready(
        exact=ReadyObject("exact/valid.npz", batch.sha256, batch.path.stat().st_size),
        dense_references=references,
    )
    _validate_plan_case_binding(batch.path, ready, track)

    changed = (replace(cases[0], source_id="wrong-source"), *cases[1:])
    invalid = write(tmp_path / "invalid", changed)
    invalid_ready = planned.ready(
        exact=ReadyObject("exact/invalid.npz", invalid.sha256, invalid.path.stat().st_size),
        dense_references=references,
    )
    with pytest.raises(CacheLoadError, match="metadata conflicts"):
        _validate_plan_case_binding(invalid.path, invalid_ready, track)


def test_ready_batch_source_restores_only_the_challenge_bound_confirmation(tmp_path) -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
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
    ready_plan = planned.ready(
        exact=ReadyObject("exact/data.npz", "12" * 32, 10),
        dense_references=tuple(references),
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
    assert len(disclosure.files) == 5
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
    assert source.prune_disclosed({"exact-a", "confirm-a"}) == 3
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


def test_plan_coordinator_skips_empty_intervals_without_losing_later_work() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(4)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
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
        def reserve_season_slots(self, maximum):
            assert maximum == 2

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
    assert plan.exact_challenge_ids == ("exact-a",)
    assert objects.payload is not None and objects.payload[0] == "plans/140.json"


def test_plan_coordinator_maps_confirmation_from_its_exact_decision_block() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(2)))
    track = SimpleNamespace(
        digest="ab" * 32,
        source_pool_merkle_root=pool.root,
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
        def reserve_season_slots(self, maximum):
            assert maximum == 2

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


def test_source_pool_capacity_is_proved_before_production_starts() -> None:
    pool = SourcePool(tuple(proof(index).leaf for index in range(6)))
    track = SimpleNamespace(
        maximum_generation_batches=2,
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
