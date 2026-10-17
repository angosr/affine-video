"""Single-process coordinator and fixed-GPU workers for private batch production."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoring.cache import ExactCacheReader, assemble_exact_cache_batch, write_exact_cache_batch
from scoring.spec import TrackSpec, load_track_spec
from validator.alerts import webhook_alerter
from validator.production.commitment import ArtifactManifest, SourcePool, verify_source_proof
from validator.production.config import ProducerConfig, load_producer_config
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    RolloutPlan,
    build_plan,
    select_source_proofs,
    submission_interval,
)
from validator.production.storage import PreconditionFailed, PrivateObjectStore, StoredObject
from validator.store import ChallengeStore


@dataclass(frozen=True, slots=True)
class UnitResult:
    rollout_id: str
    shard_path: str
    video_path: str
    latent_path: str
    elapsed_seconds: float
    black_frame_fraction: float


@dataclass(frozen=True, slots=True)
class ProducerCycle:
    planned_block: int | None
    ready_block: int | None
    waiting_for_cursor: bool


class PlanCoordinator:
    """Create immutable plans only from a fully scanned finalized SQLite snapshot."""

    def __init__(
        self,
        *,
        track: TrackSpec,
        store: ChallengeStore,
        objects: PrivateObjectStore,
        source_pool: SourcePool,
    ) -> None:
        if source_pool.root != track.source_pool_merkle_root:
            raise ValueError("source pool does not match the track commitment")
        self.track = track
        self.store = store
        self.objects = objects
        self.source_pool = source_pool

    def plans(self) -> tuple[tuple[str, StoredObject, BatchPlan], ...]:
        rows = []
        for name in self.objects.list_names("plans/"):
            if not name.startswith("plans/") or not name.endswith(".json"):
                raise RuntimeError("private plan index contains an unexpected object")
            stored = self.objects.get(name)
            plan = BatchPlan.decode(stored.body)
            self._validate_plan(name, plan)
            rows.append((name, stored, plan))
        rows.sort(key=lambda row: row[2].generation_block)
        if len({row[2].generation_block for row in rows}) != len(rows):
            raise RuntimeError("private plan index contains duplicate generation blocks")
        return tuple(rows)

    def create_next(self) -> BatchPlan | None:
        self.store.reserve_season_slots(self.track.maximum_effective_challenges)
        existing = self.plans()
        by_block = {plan.generation_block: plan for _, _, plan in existing}
        used_families = {
            proof.leaf.source_family_id for _, _, plan in existing for proof in plan.source_proofs
        }
        cursor = self.store.chain_cursor()
        if cursor is None:
            return None
        interval = self.track.cache.generation_interval_blocks
        for sequence in range(1, self.track.maximum_generation_batches + 1):
            block = self.track.track_start_block + sequence * interval
            if block in by_block:
                continue
            if cursor[0] < block:
                return None
            block_hash = self.store.finalized_block_hash(block)
            if block_hash is None:
                raise RuntimeError("generation block hash is absent from the finalized scan")
            first, last = submission_interval(
                block,
                track_start_block=self.track.track_start_block,
                minimum_generation_delay_blocks=(self.track.cache.minimum_generation_delay_blocks),
                generation_interval_blocks=interval,
            )
            exact_ids = self.store.admitted_challenge_ids(first, last)
            confirmation_ids = self.store.confirmation_challenge_ids(first, last)
            if not exact_ids and not confirmation_ids:
                continue
            seed = hashlib.sha256(
                b"veda-evaluation-batch-v1"
                + bytes.fromhex(self.track.digest)
                + bytes.fromhex(block_hash.removeprefix("0x"))
            ).hexdigest()
            proofs = select_source_proofs(
                self.source_pool,
                strata=tuple(self.track.scoring.stratum_weight_map),
                sources_per_stratum=self.track.scoring.rollouts_per_stratum,
                batch_seed=seed,
                used_families=frozenset(used_families),
            )
            plan = build_plan(
                track_spec_digest=self.track.digest,
                generation_block=block,
                block_hash=block_hash,
                track_start_block=self.track.track_start_block,
                minimum_generation_delay_blocks=(self.track.cache.minimum_generation_delay_blocks),
                generation_interval_blocks=interval,
                exact_challenge_ids=exact_ids,
                confirmation_challenge_ids=confirmation_ids,
                source_proofs=proofs,
                scoring=self.track.scoring,
                available_routes=self.track.model.selector.routes,
            )
            name = _plan_name(block)
            try:
                self.objects.create_plan(name, plan.encode())
            except PreconditionFailed:
                winner = BatchPlan.decode(self.objects.get(name).body)
                self._validate_plan(name, winner)
                return winner
            return plan
        return None

    def _validate_plan(self, name: str, plan: BatchPlan) -> None:
        if name != _plan_name(plan.generation_block):
            raise RuntimeError("plan key and generation block disagree")
        if plan.track_spec_digest != self.track.digest:
            raise RuntimeError("private plan belongs to another track")
        expected_hash = self.store.finalized_block_hash(plan.generation_block)
        if expected_hash is not None and expected_hash.lower() != plan.block_hash.lower():
            raise RuntimeError("private plan block hash conflicts with finalized chain history")
        if any(
            not verify_source_proof(proof, self.track.source_pool_merkle_root)
            for proof in plan.source_proofs
        ):
            raise RuntimeError("private plan contains an invalid source proof")


class WorkerPool:
    """Exactly one long-lived MiniMax teacher process on each formal producer GPU."""

    def __init__(
        self,
        config: ProducerConfig,
        track: TrackSpec,
        *,
        timeout_seconds: float = 4 * 60 * 60,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("producer worker timeout must be positive")
        self._timeout_seconds = timeout_seconds
        context = multiprocessing.get_context("spawn")
        self._result = context.Queue()
        self._queues = []
        self._processes = []
        for gpu_id in config.gpu_ids:
            tasks = context.Queue()
            process = context.Process(
                target=_worker_loop,
                args=(gpu_id, config, track, tasks, self._result),
                name=f"veda-producer-gpu-{gpu_id}",
            )
            process.start()
            self._queues.append(tasks)
            self._processes.append(process)

    def run(self, plan: BatchPlan) -> tuple[UnitResult, ...]:
        for index, rollout in enumerate(plan.rollouts):
            self._queues[index % len(self._queues)].put((plan, rollout))
        results: dict[str, UnitResult] = {}
        deadline = time.monotonic() + self._timeout_seconds
        while len(results) < len(plan.rollouts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("producer workers exceeded the plan wall-clock limit")
            try:
                kind, payload = self._result.get(timeout=min(30, remaining))
            except queue.Empty:
                failed = [process for process in self._processes if not process.is_alive()]
                if failed:
                    raise RuntimeError(
                        f"producer worker exited unexpectedly: {failed[0].exitcode}"
                    ) from None
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "producer workers exceeded the plan wall-clock limit"
                    ) from None
                continue
            if kind == "error":
                raise RuntimeError(payload)
            result = payload
            if not isinstance(result, UnitResult) or result.rollout_id in results:
                raise RuntimeError("producer worker returned an invalid or duplicate result")
            results[result.rollout_id] = result
        return tuple(results[rollout.rollout_id] for rollout in plan.rollouts)

    def close(self) -> None:
        for tasks in self._queues:
            tasks.put(None)
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


class ProducerService:
    def __init__(
        self,
        config: ProducerConfig,
        *,
        alert: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        self.config = config
        self.track = load_track_spec(config.track_spec)
        self.objects = PrivateObjectStore.from_environment(
            bucket=config.storage_bucket,
            prefix=f"{config.storage_prefix}/tracks/{self.track.digest}",
        )
        self.objects.verify_conditional_writes()
        self.pool = SourcePool.decode(
            self.objects.ensure_bootstrap("bootstrap/source_pool.json", config.source_pool)
        )
        _validate_source_capacity(self.pool, self.track)
        self.manifest = ArtifactManifest.decode(
            self.objects.ensure_bootstrap(
                "bootstrap/artifact_manifest.json", config.artifact_manifest
            )
        )
        if self.manifest.digest != self.track.artifact_manifest_sha256:
            raise ValueError("artifact manifest does not match the track commitment")
        self.objects.ensure_artifacts(self.manifest, config.artifact_root)
        config.verify_manifest(self.manifest)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        self.alert = alert

    def run(self, *, once: bool = False) -> None:
        workers: WorkerPool | None = None
        failures: dict[str, int] = {}
        try:
            while True:
                with ChallengeStore(
                    self.config.validator_database,
                    track_spec_digest=self.track.digest,
                ) as store:
                    coordinator = PlanCoordinator(
                        track=self.track,
                        store=store,
                        objects=self.objects,
                        source_pool=self.pool,
                    )
                    created = coordinator.create_next()
                    pending = next(
                        (row for row in coordinator.plans() if row[2].state == "PLANNED"),
                        None,
                    )
                    retained = _retained_plan_digests(coordinator.plans())
                ready_block = None
                if pending is not None:
                    if workers is None:
                        workers = WorkerPool(self.config, self.track)
                    try:
                        ready = self._execute(pending[0], pending[1], pending[2], workers)
                    except Exception as exc:
                        workers.close()
                        workers = None
                        plan = pending[2]
                        attempt = failures.get(plan.plan_digest, 0) + 1
                        failures[plan.plan_digest] = attempt
                        if self.alert is not None and (attempt == 1 or attempt == 5):
                            self.alert(
                                "PRODUCER_PLAN_FAILURE",
                                {
                                    "attempt": attempt,
                                    "generation_block": plan.generation_block,
                                    "plan_digest": plan.plan_digest,
                                    "error": (str(exc) or type(exc).__name__)[:500],
                                },
                            )
                        if once or attempt >= 5:
                            raise RuntimeError(
                                f"producer retry ceiling reached for block "
                                f"{plan.generation_block}: {exc}"
                            ) from exc
                        time.sleep(min(300.0, self.config.poll_seconds * (2 ** (attempt - 1))))
                        continue
                    failures.pop(pending[2].plan_digest, None)
                    ready_block = ready.generation_block
                _prune_local_work(self.config.work_dir, retained)
                _report_cycle(
                    ProducerCycle(
                        None if created is None else created.generation_block,
                        ready_block,
                        created is None and pending is None,
                    )
                )
                if once:
                    return
                time.sleep(self.config.poll_seconds)
        finally:
            if workers is not None:
                workers.close()

    def _execute(
        self,
        name: str,
        stored: StoredObject,
        plan: BatchPlan,
        workers: WorkerPool,
    ) -> BatchPlan:
        results = workers.run(plan)
        references = []
        for result in results:
            video = _ready_file(Path(result.video_path), "videos", ".mp4")
            latent = _ready_file(Path(result.latent_path), "latents", ".safetensors")
            self.objects.put_content(
                result.video_path, sha256=video.sha256, object_name=video.object_key
            )
            self.objects.put_content(
                result.latent_path, sha256=latent.sha256, object_name=latent.object_key
            )
            references.append(DenseReference(result.rollout_id, video, latent))
        directory = self.config.work_dir / plan.plan_digest / "batch"
        batch = assemble_exact_cache_batch(
            directory,
            tuple(result.shard_path for result in results),
            generated_block=plan.generation_block,
            seed_block_hash=plan.block_hash,
            plan_digest=plan.plan_digest,
            track_spec_digest=self.track.digest,
            artifact_manifest_sha256=self.track.artifact_manifest_sha256,
            source_pool_merkle_root=self.track.source_pool_merkle_root,
            source_proofs=tuple(proof.document() for proof in plan.source_proofs),
            dense_references=tuple(reference.document() for reference in references),
        )
        if batch.path.stat().st_size > self.track.cache.exact_max_bytes:
            raise RuntimeError("generated exact cache exceeds the frozen track byte limit")
        exact = ReadyObject(
            f"exact/{plan.generation_block}-{batch.sha256}.npz",
            batch.sha256,
            batch.path.stat().st_size,
        )
        self.objects.put_content(batch.path, sha256=exact.sha256, object_name=exact.object_key)
        ready = plan.ready(exact=exact, dense_references=tuple(references))
        try:
            self.objects.replace_plan(name, ready.encode(), etag=stored.etag)
        except PreconditionFailed:
            winner = BatchPlan.decode(self.objects.get(name).body)
            if winner.state != "READY":
                raise RuntimeError("plan changed without reaching READY") from None
            return winner
        return ready


def run_producer(config_path: str | Path, *, once: bool = False) -> None:
    alert_url = os.environ.get("VEDA_ALERT_WEBHOOK_URL")
    if not once and not alert_url:
        raise ValueError("a push alert webhook is required for the continuous producer")
    alert = webhook_alerter(alert_url, service="veda-producer") if alert_url is not None else None
    ProducerService(load_producer_config(config_path), alert=alert).run(once=once)


def _worker_loop(
    gpu_id: int,
    config: ProducerConfig,
    track: TrackSpec,
    tasks: Any,
    results: Any,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        from validator.production.runtime import DenseRolloutRuntime

        runtime = DenseRolloutRuntime(
            model_root=str(config.artifact_path(config.runtime.model_root)),
            adapter_path=str(config.artifact_path(config.runtime.adapter)),
            predictor_bundle=str(config.artifact_path(config.runtime.predictor)),
            sample_cache=str(config.artifact_path(config.runtime.sample_cache)),
            offload_blocks=config.runtime.offload_blocks,
            prefetch=config.runtime.prefetch,
            mlp_chunk_rows=config.runtime.mlp_chunk_rows,
        )
        while True:
            task = tasks.get()
            if task is None:
                return
            plan, rollout = task
            try:
                results.put(("ok", _produce_unit(runtime, config, track, plan, rollout)))
            except Exception as exc:  # worker boundary must return a useful failure
                results.put(("error", f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        results.put(("error", f"worker startup failed on GPU {gpu_id}: {exc}"))


def _produce_unit(
    runtime: Any,
    config: ProducerConfig,
    track: TrackSpec,
    plan: BatchPlan,
    rollout: RolloutPlan,
) -> UnitResult:
    root = config.work_dir / plan.plan_digest / "units" / rollout.rollout_id
    root.mkdir(parents=True, exist_ok=True)
    existing = _find_valid_shard(root, plan, rollout, track)
    video = root / f"{rollout.rollout_id}.mp4"
    latent = root / f"{rollout.rollout_id}.safetensors"
    if existing is not None and video.stat().st_size > 0 and latent.stat().st_size > 0:
        return UnitResult(rollout.rollout_id, str(existing), str(video), str(latent), 0.0, 0.0)
    capture = runtime.capture(
        sample_id=rollout.source_id,
        source_id=rollout.source_id,
        stratum=rollout.stratum,
        rollout_id=rollout.rollout_id,
        geometry=rollout.stratum,
        seed=rollout.noise_seed,
        routes=plan.selected_routes,
        steps=tuple((step.index, step.window) for step in plan.selected_steps),
        query_selection_seed=plan.query_selection_seed,
        query_tiles_per_case=track.scoring.query_tiles_per_case,
    )
    output = runtime.write_dense_outputs(
        capture,
        geometry=rollout.stratum,
        rollout_id=rollout.rollout_id,
        directory=root,
    )
    shard = write_exact_cache_batch(
        root,
        capture.cases,
        generated_block=plan.generation_block,
        seed_block_hash=plan.block_hash,
        plan_digest=plan.plan_digest,
        track_spec_digest=track.digest,
        artifact_manifest_sha256=track.artifact_manifest_sha256,
        source_pool_merkle_root=track.source_pool_merkle_root,
        source_proofs=tuple(proof.document() for proof in plan.source_proofs),
        dense_references=(),
    )
    return UnitResult(
        rollout.rollout_id,
        str(shard.path),
        str(output.video_path),
        str(output.latent_path),
        capture.elapsed_seconds + output.decode_seconds,
        output.black_frame_fraction,
    )


def _find_valid_shard(
    root: Path, plan: BatchPlan, rollout: RolloutPlan, track: TrackSpec
) -> Path | None:
    candidates = sorted(root.glob(f"{plan.generation_block}-*.npz"))
    if (
        not candidates
        or not (root / f"{rollout.rollout_id}.mp4").is_file()
        or not (root / f"{rollout.rollout_id}.safetensors").is_file()
    ):
        return None
    for path in candidates:
        digest = _file_sha256(path)
        try:
            reader = ExactCacheReader(
                path,
                expected_sha256=digest,
                max_bytes=track.cache.exact_max_bytes,
            )
            if reader.metadata["plan_digest"] == plan.plan_digest and {
                row["rollout_id"] for row in reader.metadata["cases"]
            } == {rollout.rollout_id}:
                return path
        except (OSError, ValueError):
            continue
    return None


def _ready_file(path: Path, directory: str, suffix: str) -> ReadyObject:
    if not path.is_file() or path.stat().st_size <= 0 or path.suffix != suffix:
        raise RuntimeError("producer output is missing or invalid")
    digest = _file_sha256(path)
    return ReadyObject(f"{directory}/{digest}{suffix}", digest, path.stat().st_size)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_name(block: int) -> str:
    return f"plans/{block}.json"


def _retained_plan_digests(
    rows: tuple[tuple[str, StoredObject, BatchPlan], ...],
) -> frozenset[str]:
    planned = {plan.plan_digest for _, _, plan in rows if plan.state == "PLANNED"}
    ready = sorted(
        (plan for _, _, plan in rows if plan.state == "READY"),
        key=lambda plan: plan.generation_block,
        reverse=True,
    )
    return frozenset(planned | {plan.plan_digest for plan in ready[:2]})


def _prune_local_work(root: Path, retained: frozenset[str]) -> None:
    """Remove only reconstructible, uploaded plan work older than the local safety tail."""

    for child in root.iterdir():
        if (
            child.is_dir()
            and len(child.name) == 64
            and all(character in "0123456789abcdef" for character in child.name)
            and child.name not in retained
        ):
            shutil.rmtree(child)


def _validate_source_capacity(pool: SourcePool, track: TrackSpec) -> None:
    required = track.maximum_generation_batches * track.scoring.rollouts_per_stratum
    for stratum in track.scoring.stratum_weight_map:
        families = {
            leaf.source_family_id
            for leaf in pool.leaves
            if leaf.partition == "production" and leaf.generation_geometry == stratum
        }
        if len(families) < required:
            raise ValueError(f"source pool cannot cover the track capacity for stratum {stratum}")


def _report_cycle(cycle: ProducerCycle) -> None:
    print(
        json.dumps(
            {
                "planned_block": cycle.planned_block,
                "ready_block": cycle.ready_block,
                "waiting_for_cursor": cycle.waiting_for_cursor,
            },
            sort_keys=True,
        ),
        flush=True,
    )
