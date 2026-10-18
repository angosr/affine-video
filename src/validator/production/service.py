"""Single-process coordinator and fixed-GPU workers for private batch production."""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

from scoring.cache import (
    ExactCacheReader,
    assemble_exact_cache_batch,
    validate_exact_cache_references,
    write_exact_cache_batch,
)
from scoring.spec import TrackSpec, load_track_spec, required_production_sources_per_stratum
from validator.alerts import webhook_alerter
from validator.production.commitment import (
    ArtifactManifest,
    SourcePool,
    canonical_json,
    decode_canonical_json,
    load_bound_source_pool,
    verify_source_proof,
)
from validator.production.config import ProducerConfig, load_producer_config
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    RolloutPlan,
    build_plan,
    select_source_proofs,
    submission_interval,
    validate_source_selection_sequence,
)
from validator.production.runtime import trajectory_tensor_key
from validator.production.storage import (
    ARTIFACT_MANIFEST_OBJECT,
    EXACT_NAMESPACE,
    INPUTS_NAMESPACE,
    LATENTS_NAMESPACE,
    PLANS_NAMESPACE,
    SOURCE_POOL_OBJECT,
    STAGES_NAMESPACE,
    VIDEOS_NAMESPACE,
    PreconditionFailed,
    PrivateObjectStore,
    StoredObject,
)
from validator.production.workers import close_workers
from validator.queue import TERMINAL_STATES, ChallengeState
from validator.store import ChallengeStore

_MAX_PLAN_ATTEMPTS = 5
_PRODUCER_ATTEMPT_TIMEOUT_SECONDS = 4 * 60 * 60


class _DeadlineInterrupt(BaseException):
    """Escape SDK retry loops before translating to a producer failure."""


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
        expected_commitments = tuple(
            {
                "generation_geometry": item.generation_geometry,
                "partition": item.partition,
                "sha256": item.sha256,
            }
            for item in track.source_commitments
        )
        if source_pool.public_commitments() != expected_commitments:
            raise ValueError("source pool differs from the public selection index")
        self.track = track
        self.store = store
        self.objects = objects
        self.source_pool = source_pool

    def plans(self) -> tuple[tuple[str, StoredObject, BatchPlan], ...]:
        rows = []
        prefix = f"{PLANS_NAMESPACE}/"
        for name in self.objects.list_names(prefix):
            if not name.startswith(prefix) or not name.endswith(".json"):
                raise RuntimeError("private plan index contains an unexpected object")
            stored = self.objects.get(name)
            plan = BatchPlan.decode(stored.body)
            self._validate_plan(name, plan)
            rows.append((name, stored, plan))
        rows.sort(key=lambda row: row[2].generation_block)
        if len({row[2].generation_block for row in rows}) != len(rows):
            raise RuntimeError("private plan index contains duplicate generation blocks")
        try:
            validate_source_selection_sequence(
                tuple(row[2] for row in rows),
                commitments=self.track.source_commitments,
                scoring=self.track.scoring,
            )
        except ValueError as exc:
            raise RuntimeError("private plan source selection is invalid") from exc
        return tuple(rows)

    def create_next(self) -> BatchPlan | None:
        if not self._stage_recovery_complete():
            return None
        existing = self.plans()
        by_block = {plan.generation_block: plan for _, _, plan in existing}
        used_source_digests = {
            proof.leaf.digest for _, _, plan in existing for proof in plan.source_proofs
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
            if not self.store.admission_interval_complete(first, last):
                return None
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
                commitments=self.track.source_commitments,
                partition="production",
                strata=tuple(self.track.scoring.stratum_weight_map),
                sources_per_stratum=self.track.scoring.rollouts_per_stratum,
                batch_seed=seed,
                used_source_digests=frozenset(used_source_digests),
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
            except PreconditionFailed as exc:
                winner = BatchPlan.decode(self.objects.get(name).body)
                self._validate_plan(name, winner)
                if winner.planned_document() != plan.planned_document():
                    raise RuntimeError(
                        "private plan conflicts with the deterministic schedule"
                    ) from exc
                return winner
            return plan
        return None

    def _stage_recovery_complete(self) -> bool:
        """Fence planning until every private stage fact is present in SQLite."""

        for name in self.objects.list_names(f"{STAGES_NAMESPACE}/"):
            parts = name.split("/")
            if (
                len(parts) != 3
                or parts[0] != STAGES_NAMESPACE
                or not parts[2].endswith(".json")
            ):
                raise RuntimeError("private stage index contains an unexpected object")
            challenge_id = parts[1]
            stage_name = parts[2][:-5]
            try:
                challenge_bytes = bytes.fromhex(challenge_id)
                stage = ChallengeState(stage_name)
            except ValueError as exc:
                raise RuntimeError("private stage index contains an invalid object") from exc
            if len(challenge_bytes) != 32 or stage not in {
                ChallengeState.SCORING,
                ChallengeState.CONFIRMING,
            }:
                raise RuntimeError("private stage index contains an invalid object")

            stored = self.objects.get(name)
            try:
                document = json.loads(stored.body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("private stage archive is invalid") from exc
            if (
                not isinstance(document, dict)
                or document.get("format") != "veda-stage-result-v1"
                or document.get("track_spec_digest") != self.track.digest
                or document.get("challenge_id") != challenge_id
                or document.get("stage") != stage.value
            ):
                raise RuntimeError("private stage archive conflicts with its object key")

            recovered = self.store.stage_score(challenge_id, stage)
            if recovered is None:
                return False
            local_state = self.store.get_challenge(challenge_id).state
            metrics_match = recovered.metrics == document.get("metrics")
            if not metrics_match and local_state in TERMINAL_STATES:
                metrics_match = recovered.metrics == _public_stage_metrics(
                    document.get("metrics")
                )
            if (
                recovered.score != document.get("score")
                or recovered.score_block != document.get("score_block")
                or not metrics_match
            ):
                raise RuntimeError("local stage fact conflicts with its private archive")
            if not self.store.stage_transition_applied(challenge_id, stage):
                return False
        return True

    def _validate_plan(self, name: str, plan: BatchPlan) -> None:
        if name != _plan_name(plan.generation_block):
            raise RuntimeError("plan key and generation block disagree")
        if plan.track_spec_digest != self.track.digest:
            raise RuntimeError("private plan belongs to another track")
        expected_hash = self.store.finalized_block_hash(plan.generation_block)
        if expected_hash is not None and expected_hash.lower() != plan.block_hash.lower():
            raise RuntimeError("private plan block hash conflicts with finalized chain history")
        commitment_digests = tuple(item.sha256 for item in self.track.source_commitments)
        if any(
            not verify_source_proof(
                proof,
                self.track.source_pool_merkle_root,
                commitment_digests=commitment_digests,
            )
            for proof in plan.source_proofs
        ):
            raise RuntimeError("private plan contains an invalid source proof")


def _public_stage_metrics(value: object) -> dict[str, object]:
    """Project a private score result onto the fields preserved in public matches."""

    if not isinstance(value, Mapping):
        return {}
    keys = (
        "incumbent_score",
        "paired_gain_lower_bound",
        "cache_sha256",
        "cache_generated_block",
        "cache_plan_digest",
        "cache_seed_block_hash",
    )
    return {key: value[key] for key in keys if key in value}


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
        try:
            for gpu_id in config.gpu_ids:
                tasks = context.Queue()
                self._queues.append(tasks)
                process = context.Process(
                    target=_worker_loop,
                    args=(gpu_id, config, track, tasks, self._result),
                    name=f"veda-producer-gpu-{gpu_id}",
                )
                self._processes.append(process)
                process.start()
        except Exception:
            self.close()
            raise

    def run(
        self,
        plan: BatchPlan,
        *,
        deadline: float | None = None,
    ) -> tuple[UnitResult, ...]:
        for index, rollout in enumerate(plan.rollouts):
            self._queues[index % len(self._queues)].put((plan, rollout))
        results: dict[str, UnitResult] = {}
        worker_deadline = time.monotonic() + self._timeout_seconds
        if deadline is not None:
            worker_deadline = min(worker_deadline, deadline)
        while len(results) < len(plan.rollouts):
            remaining = worker_deadline - time.monotonic()
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
                if time.monotonic() >= worker_deadline:
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
        queues, self._queues = self._queues, []
        processes, self._processes = self._processes, []
        result, self._result = self._result, None
        close_workers(queues, processes, result)


class ProducerService:
    def __init__(
        self,
        config: ProducerConfig,
        *,
        alert: Callable[[str, Mapping[str, object]], None] | None = None,
        attempt_timeout_seconds: float = _PRODUCER_ATTEMPT_TIMEOUT_SECONDS,
    ) -> None:
        if attempt_timeout_seconds <= 0:
            raise ValueError("producer attempt timeout must be positive")
        self.config = config
        self.track = load_track_spec(config.track_spec)
        self.objects = PrivateObjectStore.from_environment(
            bucket=config.storage_bucket,
            prefix=f"{config.storage_prefix}/tracks/{self.track.digest}",
        )
        self.objects.verify_conditional_writes()
        source_pool_payload = self.objects.ensure_bootstrap(
            SOURCE_POOL_OBJECT, config.source_pool
        )
        self.manifest = ArtifactManifest.decode(
            self.objects.ensure_bootstrap(
                ARTIFACT_MANIFEST_OBJECT, config.artifact_manifest
            )
        )
        if self.manifest.digest != self.track.artifact_manifest_sha256:
            raise ValueError("artifact manifest does not match the track commitment")
        self.objects.ensure_artifacts(self.manifest, config.artifact_root)
        config.verify_manifest(self.manifest)
        self.pool = load_bound_source_pool(
            source_pool_payload,
            config.artifact_path(config.runtime.source_binding).read_bytes(),
            expected_root=self.track.source_pool_merkle_root,
            expected_commitments=self.track.source_commitment_documents,
            sample_cache=config.artifact_path(config.runtime.sample_cache),
        )
        _validate_source_capacity(self.pool, self.track)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        self.alert = alert
        self.attempt_timeout_seconds = attempt_timeout_seconds

    def run(self, *, once: bool = False) -> None:
        with _work_directory_lock(self.config.work_dir):
            self._run_locked(once=once)

    def _run_locked(self, *, once: bool) -> None:
        workers: WorkerPool | None = None
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
                    plan = pending[2]
                    attempts = _plan_attempt_count(self.objects, plan)
                    if attempts >= _MAX_PLAN_ATTEMPTS:
                        raise RuntimeError(
                            f"producer retry ceiling reached for block "
                            f"{plan.generation_block}"
                        )
                    attempt = _reserve_plan_attempt(self.objects, plan)
                    deadline = time.monotonic() + self.attempt_timeout_seconds
                    try:
                        if workers is None:
                            workers = WorkerPool(self.config, self.track)
                        ready = self._execute(
                            pending[0], pending[1], plan, workers, deadline=deadline
                        )
                    except Exception as exc:
                        if workers is not None:
                            workers.close()
                        workers = None
                        if self.alert is not None and attempt in {1, _MAX_PLAN_ATTEMPTS}:
                            self.alert(
                                "PRODUCER_PLAN_FAILURE",
                                {
                                    "attempt": attempt,
                                    "generation_block": plan.generation_block,
                                    "plan_digest": plan.plan_digest,
                                    "error": (str(exc) or type(exc).__name__)[:500],
                                },
                            )
                        if once or attempt >= _MAX_PLAN_ATTEMPTS:
                            raise RuntimeError(
                                f"producer retry ceiling reached for block "
                                f"{plan.generation_block}: {exc}"
                            ) from exc
                        time.sleep(min(300.0, self.config.poll_seconds * (2 ** (attempt - 1))))
                        continue
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
        *,
        deadline: float,
    ) -> BatchPlan:
        try:
            with _interrupt_at_deadline(deadline):
                return self._execute_before_deadline(
                    name, stored, plan, workers, deadline=deadline
                )
        except _DeadlineInterrupt:
            raise RuntimeError("producer attempt exceeded the wall-clock limit") from None

    def _execute_before_deadline(
        self,
        name: str,
        stored: StoredObject,
        plan: BatchPlan,
        workers: WorkerPool,
        *,
        deadline: float,
    ) -> BatchPlan:
        _require_before_deadline(deadline)
        results = workers.run(plan, deadline=deadline)
        _require_before_deadline(deadline)
        references = []
        for result in results:
            video = _ready_file(Path(result.video_path), VIDEOS_NAMESPACE, ".mp4")
            latent = _ready_file(Path(result.latent_path), LATENTS_NAMESPACE, ".safetensors")
            self.objects.put_content(
                result.video_path, sha256=video.sha256, object_name=video.object_key
            )
            _require_before_deadline(deadline)
            self.objects.put_content(
                result.latent_path, sha256=latent.sha256, object_name=latent.object_key
            )
            _require_before_deadline(deadline)
            references.append(DenseReference(result.rollout_id, video, latent))
        directory = self.config.work_dir / plan.plan_digest / "batch"
        selected_cache, cache_files = _build_selected_sample_cache(
            self.config,
            plan,
            directory / "sample-cache",
        )
        for path, item in cache_files:
            self.objects.put_content(path, sha256=item.sha256, object_name=item.object_key)
            _require_before_deadline(deadline)
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
            sample_cache=selected_cache.document(),
        )
        _require_before_deadline(deadline)
        if batch.path.stat().st_size > self.track.cache.exact_max_bytes:
            raise RuntimeError("generated exact cache exceeds the frozen track byte limit")
        exact = ReadyObject(
            f"{EXACT_NAMESPACE}/{plan.generation_block}-{batch.sha256}.npz",
            batch.sha256,
            batch.path.stat().st_size,
        )
        self.objects.put_content(batch.path, sha256=exact.sha256, object_name=exact.object_key)
        _require_before_deadline(deadline)
        ready = plan.ready(
            exact=exact,
            dense_references=tuple(references),
            sample_cache=selected_cache,
        )
        try:
            self.objects.replace_plan(name, ready.encode(), etag=stored.etag)
        except PreconditionFailed:
            _require_before_deadline(deadline)
            winner = BatchPlan.decode(self.objects.get(name).body)
            _require_before_deadline(deadline)
            if winner.state != "READY":
                raise RuntimeError("plan changed without reaching READY") from None
            return winner
        _require_before_deadline(deadline)
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
    runtime.verify_trajectory_schedule(track.scoring.trajectory_schedule.sigmas)
    root = config.work_dir / plan.plan_digest / "units" / rollout.rollout_id
    root.mkdir(parents=True, exist_ok=True)
    existing = _find_valid_shard(root, plan, rollout, track, dense_device="cuda:0")
    video = root / f"{rollout.rollout_id}.mp4"
    latent = root / f"{rollout.rollout_id}.safetensors"
    if existing is not None:
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
        sample_cache={},
    )
    _write_unit_completion(root, plan, rollout, shard.path, output.video_path, output.latent_path)
    validated = _find_valid_shard(root, plan, rollout, track, dense_device="cuda:0")
    if validated != shard.path:
        raise RuntimeError("producer unit failed its completion validation")
    return UnitResult(
        rollout.rollout_id,
        str(validated),
        str(output.video_path),
        str(output.latent_path),
        capture.elapsed_seconds + output.decode_seconds,
        output.black_frame_fraction,
    )


def _find_valid_shard(
    root: Path,
    plan: BatchPlan,
    rollout: RolloutPlan,
    track: TrackSpec,
    *,
    dense_device: str | None = None,
) -> Path | None:
    completion = root / "unit-complete.json"
    if not completion.is_file():
        return None
    try:
        payload = completion.read_bytes()
        document = json.loads(payload)
        identity = {
            "format": "veda-production-unit-v1",
            "track_spec_digest": track.digest,
            "plan_digest": plan.plan_digest,
            "rollout_id": rollout.rollout_id,
        }
        if (
            not isinstance(document, dict)
            or payload != _canonical_json(document)
            or set(document) != set(identity) | {"files"}
            or any(document.get(key) != value for key, value in identity.items())
            or not isinstance(document.get("files"), dict)
            or set(document["files"]) != {"exact", "video", "latent"}
        ):
            return None
        exact = _completed_file(root, document["files"]["exact"], ".npz")
        video = _completed_file(root, document["files"]["video"], ".mp4")
        latent = _completed_file(root, document["files"]["latent"], ".safetensors")
        if video.name != f"{rollout.rollout_id}.mp4" or latent.name != (
            f"{rollout.rollout_id}.safetensors"
        ):
            return None
        reader = ExactCacheReader(
            exact,
            expected_sha256=document["files"]["exact"]["sha256"],
            max_bytes=track.cache.exact_max_bytes,
        )
        if (
            reader.metadata["track_spec_digest"] != track.digest
            or reader.metadata["plan_digest"] != plan.plan_digest
            or reader.metadata["generated_block"] != plan.generation_block
            or reader.metadata["seed_block_hash"].lower() != plan.block_hash.lower()
            or {row["rollout_id"] for row in reader.metadata["cases"]}
            != {rollout.rollout_id}
            or not _valid_dense_latent(latent, rollout, plan)
            or not _video_decodes(video)
        ):
            return None
        validate_exact_cache_references(
            exact,
            expected_sha256=document["files"]["exact"]["sha256"],
            max_bytes=track.cache.exact_max_bytes,
            device=dense_device,
        )
        return exact
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_unit_completion(
    root: Path,
    plan: BatchPlan,
    rollout: RolloutPlan,
    exact: Path,
    video: Path,
    latent: Path,
) -> None:
    document = {
        "format": "veda-production-unit-v1",
        "track_spec_digest": plan.track_spec_digest,
        "plan_digest": plan.plan_digest,
        "rollout_id": rollout.rollout_id,
        "files": {
            "exact": _completed_file_document(exact),
            "video": _completed_file_document(video),
            "latent": _completed_file_document(latent),
        },
    }
    destination = root / "unit-complete.json"
    temporary = root / ".unit-complete.tmp"
    payload = _canonical_json(document)
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_file_document(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("producer unit output is missing or empty")
    return {
        "name": path.name,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _completed_file(root: Path, value: object, suffix: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size_bytes"}:
        raise ValueError("unit completion file record is invalid")
    name = value["name"]
    digest = value["sha256"]
    size = value["size_bytes"]
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(suffix)
        or not isinstance(digest, str)
        or len(digest) != 64
        or type(size) is not int
        or size <= 0
    ):
        raise ValueError("unit completion file record is invalid")
    bytes.fromhex(digest)
    path = root / name
    if not path.is_file() or path.stat().st_size != size or _file_sha256(path) != digest:
        raise ValueError("unit completion file does not match its recorded content")
    return path


def _valid_dense_latent(path: Path, rollout: RolloutPlan, plan: BatchPlan) -> bool:
    try:
        with safe_open(path, framework="numpy") as tensors:
            trajectory_keys = {
                trajectory_tensor_key(step.index) for step in plan.selected_steps
            }
            if set(tensors.keys()) != {
                "audio_rows",
                "video_rows",
                *trajectory_keys,
            } or tensors.metadata() != {
                "geometry": rollout.stratum,
                "rollout_id": rollout.rollout_id,
            }:
                return False
            audio = tensors.get_slice("audio_rows")
            video = tensors.get_slice("video_rows")
            valid = (
                audio.get_dtype() == "F32"
                and video.get_dtype() == "F32"
                and len(audio.get_shape()) == 2
                and len(video.get_shape()) == 2
                and audio.get_shape()[0] > 0
                and video.get_shape()[0] > 0
                and audio.get_shape()[1] == 32
                and video.get_shape()[1] == 96
            )
            return valid and all(
                tensors.get_slice(key).get_dtype() == "F32"
                and tensors.get_slice(key).get_shape() == video.get_shape()
                for key in trajectory_keys
            )
    except (OSError, ValueError, SafetensorError):
        return False


def _video_decodes(path: Path) -> bool:
    try:
        subprocess.run(
            (
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _ready_file(path: Path, directory: str, suffix: str) -> ReadyObject:
    if not path.is_file() or path.stat().st_size <= 0 or path.suffix != suffix:
        raise RuntimeError("producer output is missing or invalid")
    digest = _file_sha256(path)
    return ReadyObject(f"{directory}/{digest}{suffix}", digest, path.stat().st_size)


def _build_selected_sample_cache(
    config: ProducerConfig,
    plan: BatchPlan,
    destination: Path,
) -> tuple[ReadySampleCache, tuple[tuple[Path, ReadyObject], ...]]:
    """Export only this plan's encoded inputs for least-privilege evaluation."""

    from miowtion.train.data import SampleCache, SampleCacheWriter

    if destination.exists():
        shutil.rmtree(destination)
    source = SampleCache(str(config.artifact_path(config.runtime.sample_cache)))
    samples = {sample.id: sample for sample in source.samples}
    selected_ids = {proof.leaf.source_id for proof in plan.source_proofs}
    if selected_ids != {rollout.source_id for rollout in plan.rollouts}:
        raise RuntimeError("plan source inputs conflict with its rollouts")
    if selected_ids - set(samples):
        raise RuntimeError("plan source input is absent from the private SampleCache")
    writer = SampleCacheWriter(str(destination))
    for source_id in sorted(selected_ids):
        sample = samples[source_id]
        hidden, tags = source.text(sample)
        video, audio = source.conditions(sample)
        writer.add(
            sample.id,
            sample.task,
            sample.split,
            hidden,
            tags,
            keyframes=sample.keyframes,
            references=sample.references,
            aspect=sample.aspect,
            latent_t=sample.latent_t,
            cond_video=video if video.numel() else None,
            cond_audio=audio if audio.numel() else None,
        )
    writer.finalize()
    index_path = destination / "index.json"
    text_path = destination / "text.safetensors"
    condition_path = destination / "cond.safetensors"
    index = _ready_file(index_path, INPUTS_NAMESPACE, ".json")
    text = _ready_file(text_path, INPUTS_NAMESPACE, ".safetensors")
    condition = (
        _ready_file(condition_path, INPUTS_NAMESPACE, ".safetensors")
        if condition_path.is_file()
        else None
    )
    files = [(index_path, index), (text_path, text)]
    if condition is not None:
        files.append((condition_path, condition))
    return ReadySampleCache(index, text, condition), tuple(files)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_name(block: int) -> str:
    return f"{PLANS_NAMESPACE}/{block}.json"


def _plan_attempt_count(objects: PrivateObjectStore, plan: BatchPlan) -> int:
    prefix = f"attempts/{plan.plan_digest}/"
    names = objects.list_names(prefix)
    attempts: list[int] = []
    expected_fields = {
        "attempt",
        "format",
        "generation_block",
        "plan_digest",
        "track_spec_digest",
    }
    for name in names:
        suffix = name.removeprefix(prefix)
        if not name.startswith(prefix) or not suffix.endswith(".json"):
            raise RuntimeError("private producer attempt index is invalid")
        try:
            attempt = int(suffix[:-5])
        except ValueError as exc:
            raise RuntimeError("private producer attempt index is invalid") from exc
        try:
            document = decode_canonical_json(objects.get(name).body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("private producer attempt fact is invalid") from exc
        if (
            not isinstance(document, dict)
            or set(document) != expected_fields
            or document.get("format") != "veda-producer-attempt-v1"
            or document.get("attempt") != attempt
            or document.get("generation_block") != plan.generation_block
            or document.get("plan_digest") != plan.plan_digest
            or document.get("track_spec_digest") != plan.track_spec_digest
        ):
            raise RuntimeError("private producer attempt fact is invalid")
        attempts.append(attempt)
    attempts.sort()
    if attempts != list(range(1, len(attempts) + 1)):
        raise RuntimeError("private producer attempt sequence is invalid")
    if len(attempts) > _MAX_PLAN_ATTEMPTS:
        raise RuntimeError("private producer attempt sequence exceeds its ceiling")
    return len(attempts)


def _reserve_plan_attempt(
    objects: PrivateObjectStore,
    plan: BatchPlan,
) -> int:
    while True:
        attempts = _plan_attempt_count(objects, plan)
        if attempts >= _MAX_PLAN_ATTEMPTS:
            return attempts
        attempt = attempts + 1
        payload = canonical_json(
            {
                "attempt": attempt,
                "format": "veda-producer-attempt-v1",
                "generation_block": plan.generation_block,
                "plan_digest": plan.plan_digest,
                "track_spec_digest": plan.track_spec_digest,
            }
        )
        try:
            objects.create_plan(
                f"attempts/{plan.plan_digest}/{attempt}.json",
                payload,
            )
        except PreconditionFailed:
            continue
        return attempt


def _require_before_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RuntimeError("producer attempt exceeded the wall-clock limit")


@contextmanager
def _interrupt_at_deadline(deadline: float):
    """Interrupt every synchronous phase when the attempt wall clock expires."""

    _require_before_deadline(deadline)
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("producer attempts must run on the process main thread")
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if previous_delay > 0 or previous_interval > 0:
        raise RuntimeError("producer attempt cannot replace an active process timer")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise _DeadlineInterrupt

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(deadline - time.monotonic(), 1e-6))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def _work_directory_lock(root: Path):
    """Keep disposable local unit files single-writer while remote CAS stays authoritative."""

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".producer.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another producer is using the configured work directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    required = required_production_sources_per_stratum(
        maximum_generation_batches=track.maximum_generation_batches,
        maximum_effective_challenges=track.maximum_effective_challenges,
        rollouts_per_stratum=track.scoring.rollouts_per_stratum,
    )
    for stratum in track.scoring.stratum_weight_map:
        families = {
            leaf.source_family_id
            for leaf in pool.leaves
            if leaf.partition == "production" and leaf.generation_geometry == stratum
        }
        if len(families) < required:
            raise ValueError(
                f"source pool cannot cover the track plus one reserve batch for stratum {stratum}"
            )


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
