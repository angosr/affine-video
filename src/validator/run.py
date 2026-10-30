"""Owner evaluation and independent weight-setting loops."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any, Literal, Protocol

import bittensor as bt
import httpx

from scoring.model import ValidatedModel, validate_model
from scoring.spec import TrackSpec, load_track_spec
from validator.admission import (
    AdmissionOutcome,
    AdmissionSettings,
    process_next_admission,
)
from validator.alerts import webhook_alerter
from validator.chain import (
    ScanResult,
    finalized_commitments,
    scan_finalized_blocks,
    seed_legacy_exclusions,
)
from validator.download import DownloadedModel, download_chunked_asset, download_public_model
from validator.production.batches import BatchDisclosure, ReadyBatchSource
from validator.production.commitment import ArtifactManifest
from validator.production.config import load_producer_config
from validator.production.confirmation import ConfirmationWorkerPool
from validator.production.retries import PrivateRetryStore
from validator.production.storage import ARTIFACT_MANIFEST_OBJECT, PrivateObjectStore
from validator.publication import (
    PublicArtifacts,
    PublicLedger,
    build_genesis_artifacts,
    build_public_artifacts,
    fetch_track_identity,
    fetch_weight_artifact,
    validate_public_weights_url,
)
from validator.queue import TERMINAL_STATES
from validator.release import (
    PRODUCTION_NETUID,
    require_release_approval,
    verify_competition_manifest,
)
from validator.scan import submission_from_commitment
from validator.scoring import (
    EvaluatedStage,
    PairedTrackEvaluator,
    ScoringOutcome,
    ScoringSettings,
    StageOutcome,
    commit_evaluated_scoring,
    defer_evaluated_scoring,
    evaluate_next_scoring,
    verify_initial_champion,
)
from validator.store import MAX_PUBLICATION_FAILURES, ChallengeStore, LeasedChallenge
from validator.tournament import MatchRecord, genesis_champion, resolve_tournament
from validator.uploads import PrivateUploads
from validator.weights import (
    WeightReceipt,
    read_weight_chain_state,
    submit_weights,
    validate_weight_artifact,
    weight_update_due,
)

PUBLIC_STORAGE_BUCKET = "affine_video"
PUBLIC_STORAGE_PREFIX = "tracks/main"
PUBLIC_STORAGE_URL = (
    "https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/"
    f"{PUBLIC_STORAGE_BUCKET}/{PUBLIC_STORAGE_PREFIX}"
)
DEFAULT_PUBLIC_WEIGHTS_URL = f"{PUBLIC_STORAGE_URL}/weights.json"
DEFAULT_BASELINE_MANIFEST_URL = f"{PUBLIC_STORAGE_URL}/assets/veda_baseline_f16/manifest.json"
DEFAULT_NETUID = PRODUCTION_NETUID
DEFAULT_MECHID = 1
_LOG = logging.getLogger(__name__)
_DATA_REFRESH_SECONDS = 30.0


class ArtifactRepository(Protocol):
    async def stage(self, artifacts: PublicArtifacts) -> None: ...

    async def publish(self, artifacts: PublicArtifacts) -> None: ...

    async def publish_disclosure(
        self, disclosure: BatchDisclosure, *, track: TrackSpec, now_ms: int,
        deadline: float | None = None,
    ) -> None: ...

    async def recover(
        self, track: TrackSpec, *, runtime_manifest: bytes | None = None,
    ) -> PublicLedger: ...

    async def load_system_matches(
        self, track: TrackSpec, runtime_manifest: bytes | None,
    ) -> tuple[tuple[str, bytes], ...]: ...

    async def finalize_abort(self, track: TrackSpec) -> None: ...


def _require_production_release(
    track: TrackSpec,
    runtime_manifest: bytes,
    publisher: ArtifactRepository,
    *, image_id: str | None = None,
) -> None:
    """Fail before evaluation when a production repository lacks release approval."""

    try:
        require_release_approval(
            track,
            runtime_manifest,
            getattr(publisher, "release_approval", None),
            image_id=image_id,
        )
    except ValueError as exc:
        raise ValueError(f"formal evaluator release rejected: {exc}") from exc


@dataclass(frozen=True, slots=True)
class EvaluationCycleResult:
    finalized_block: int | None
    scan: ScanResult | None
    admission: AdmissionOutcome | None
    scoring: ScoringOutcome | None
    restored_matches: int = 0
    published_block: int | None = None
    publication_error: str | None = None
    cycle_error: str | None = None


@dataclass(frozen=True, slots=True)
class WeightCycleResult:
    finalized_block: int | None
    source_block: int | None
    status: str
    receipt: WeightReceipt | None = None
    error: str | None = None


class _EvaluationWorker:
    """One FIFO task at a time, with SQLite connections owned by its worker thread."""

    def __init__(
        self,
        *,
        database: str | Path,
        track_spec_digest: str | None,
        evaluator_factory: Callable[[ChallengeStore], Callable[[LeasedChallenge], StageOutcome]],
        admission_settings: AdmissionSettings,
        scoring_settings: ScoringSettings,
        worker_id: str,
        downloader: Callable[..., DownloadedModel] = download_public_model,
        model_validator: Callable[..., ValidatedModel] = validate_model,
    ) -> None:
        if not worker_id:
            raise ValueError("worker id cannot be empty")
        self.database = database
        self.track_spec_digest = track_spec_digest
        self.evaluator_factory = evaluator_factory
        self.admission_settings = admission_settings
        self.scoring_settings = scoring_settings
        self.worker_id = worker_id
        self.downloader = downloader
        self.model_validator = model_validator
        self.allowed = True
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veda-evaluation")
        self._pending: Future[
            tuple[AdmissionOutcome | None, EvaluatedStage | ScoringOutcome | None]
        ] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    @property
    def done(self) -> bool:
        return self._pending is not None and self._pending.done()

    def start(self, finalized_block: int) -> None:
        if self.busy:
            raise RuntimeError("an evaluation task is already pending")
        self._pending = self._executor.submit(self._run, finalized_block)

    def _run(
        self, finalized_block: int,
    ) -> tuple[AdmissionOutcome | None, EvaluatedStage | ScoringOutcome | None]:
        with ChallengeStore(self.database, track_spec_digest=self.track_spec_digest) as store:
            if not self.allowed or store.publication_failures() >= MAX_PUBLICATION_FAILURES:
                return None, None
            evaluator = self.evaluator_factory(store)
            admission = process_next_admission(
                store, worker_id=f"{self.worker_id}:admission", settings=self.admission_settings,
                current_finalized_block=finalized_block, downloader=self.downloader,
                model_validator=self.model_validator,
            )
            if not self.allowed or store.publication_failures() >= MAX_PUBLICATION_FAILURES:
                return admission, None
            evaluated = evaluate_next_scoring(
                store, worker_id=f"{self.worker_id}:scoring", settings=self.scoring_settings,
                current_finalized_block=finalized_block, evaluator=evaluator,
            )
            return admission, evaluated

    async def take(
        self, *, wait: bool,
    ) -> tuple[AdmissionOutcome | None, EvaluatedStage | ScoringOutcome | None] | None:
        pending = self._pending
        if pending is None:
            return None
        try:
            if wait:
                await asyncio.shield(asyncio.wrap_future(pending))
            elif not pending.done():
                return None
            self._pending = None
            return pending.result()
        except Exception as exc:
            self._pending = None
            if self.scoring_settings.alert is not None:
                self.scoring_settings.alert("EVALUATOR_WORKER_FAILURE", {
                    "error_type": type(exc).__name__,
                    "error": (str(exc) or type(exc).__name__)[:500],
                })
            raise

    def close(self) -> None:
        self.allowed = False
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending = None


async def _stage_publication(
    artifacts: PublicArtifacts,
    *,
    publisher: ArtifactRepository,
) -> PublicArtifacts:
    # A decision depends on its admitted model, not publication of hidden input.
    await publisher.stage(artifacts)
    return artifacts


async def _publish_due_batch(
    source: ReadyBatchSource, publisher: ArtifactRepository, *, finalized_block: int,
    challenges: list[dict[str, object]],
) -> str | None | Literal[False]:
    """Run one bulk job; report any remaining durable failure before declaring success.

    False means due data is fenced without a known failure; None means nothing is due.
    """
    try:
        protected = await asyncio.to_thread(
            source.protected_plan_digests, challenges, finalized_block,
        )
        await asyncio.to_thread(source.prune_disclosed, protected)
    except Exception:
        # Retain bytes when safety cannot be established. Optional reclamation
        # must not delay the independent, durable disclosure deadline.
        _LOG.exception("optional disclosed-data cleanup failed; retaining files")
    now_ms = time.time_ns() // 1_000_000
    due = await asyncio.to_thread(source.due_plan_digests, now_ms)
    if not due:
        return None
    retries = PrivateRetryStore(source.objects, track_spec_digest=source.data_track.digest)
    published = None
    for digest in due:
        attempt = await asyncio.to_thread(
            retries.disclosure_attempt, digest, now_ms=now_ms, lease_ms=3 * 60 * 60 * 1000,
        )
        if attempt is None:
            continue
        error = None
        try:
            deadline = time.monotonic() + 3 * 60 * 60
            disclosure = await asyncio.to_thread(
                source.public_disclosure, digest, now_ms=now_ms, deadline=deadline,
            )
            if disclosure is None:
                raise RuntimeError("disclosure became unavailable after reserving its attempt")
            await publisher.publish_disclosure(
                disclosure, track=source.data_track, now_ms=now_ms, deadline=deadline,
            )
            await asyncio.to_thread(source.record_disclosed, digest)
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            raise
        finally:
            try:
                await asyncio.to_thread(
                    retries.disclosure_attempt, digest, now_ms=time.time_ns() // 1_000_000,
                    lease_ms=3 * 60 * 60 * 1000, completed_attempt=attempt, error=error,
                )
            except Exception:
                if error is None:
                    raise
                _LOG.exception("disclosure retry completion failed; original error retained")
        published = digest
        break
    if await asyncio.to_thread(lambda: any(
        retries.disclosure_failed(digest) for digest in due if digest != published
    )):
        raise RuntimeError("unresolved disclosure failure remains in durable retry state")
    return published if published is not None else False


async def _reclaim_private_uploads(
    uploads: PrivateUploads, *, database: str | Path, network: str,
    archive_endpoints: list[str] | None,
) -> dict[str, str | None]:
    """One bounded mailbox/GC pass inside the existing evaluator owner."""
    async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
        genesis = (await client.block_info(0)).hash
        if not isinstance(genesis, str) or genesis.removeprefix("0x") != uploads.genesis_hash:
            raise RuntimeError("upload cleanup network differs from its signature domain")
        finalized = await client.finalized_block()
        snapshot = await client.at(finalized)
        timestamp = int((await snapshot.timestamp()).timestamp())

    def reclaim() -> dict[str, str | None]:
        from copy import copy

        async def registration(hotkey: str) -> tuple[int | None, int]:
            # Qualification and the original scan fence must describe ONE
            # finalized block, not a newer head fetched during the RPC roundtrip.
            cursor = store.chain_cursor()
            if cursor is None:
                raise RuntimeError("private registration awaits a scanned snapshot")
            async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
                genesis = (await client.block_info(0)).hash
                if genesis.removeprefix("0x") != uploads.genesis_hash:
                    raise RuntimeError("private registration network differs")
                if cursor[0] > await client.finalized_block():
                    raise RuntimeError("private registration scan is not finalized")
                block_hash = await client._substrate.block_hash(cursor[0])
                if block_hash.removeprefix("0x") != cursor[1].removeprefix("0x"):
                    raise RuntimeError("private registration scan hash differs")
                snapshot = await client.at(cursor[0])
                return await snapshot.read("uid", hotkey_ss58=hotkey, netuid=120), cursor[0]

        # Created and used in this thread; never share the main SQLite connection.
        with ChallengeStore.readonly(database) as store:
            worker = copy(uploads)
            worker.store = store
            worker.registration = lambda hotkey: asyncio.run(registration(hotkey))
            try:
                issue_error = None
                try:
                    worker.issue_pending(now=int(time.time()))
                except ValueError:
                    # Original eligibility refusal does not consume an opportunity.
                    _LOG.info("PRIVATE_UPLOAD_INELIGIBLE; original qualification retained")
                except Exception:
                    issue_error = "PRIVATE_UPLOAD_ISSUANCE_FAILED"
                cleanup_error = None
                try:
                    worker.reclaim_expired(
                        store=store, finalized_block=finalized, finalized_timestamp=timestamp,
                        now=int(time.time()),
                    )
                except Exception:
                    cleanup_error = "PRIVATE_UPLOAD_CLEANUP_FAILED"
                status = {"private_upload_error": issue_error,
                          "private_upload_cleanup_error": cleanup_error}
                try:
                    for key, code in worker.maintenance_status().items():
                        status[key] = code or status[key]
                except Exception:
                    status["private_upload_error"] = "PRIVATE_UPLOAD_ISSUANCE_STATUS_UNAVAILABLE"
                return status
            finally:
                uploads._issue_after, uploads._gc_after = worker._issue_after, worker._gc_after

    return await asyncio.to_thread(reclaim)


async def run_evaluator(
    *,
    track_spec: str | Path,
    database: str | Path,
    model_cache: str | Path,
    exact_cache: str | Path,
    initial_model: str | Path | None,
    initial_model_manifest_url: str = DEFAULT_BASELINE_MANIFEST_URL,
    network: str,
    archive_endpoints: list[str] | None,
    worker_id: str,
    publisher: ArtifactRepository,
    production_config: str | Path | None = None,
    version_key: int = 0,
    poll_seconds: float = 12.0,
    max_scan_blocks: int = 100,
    scoring_timeout_seconds: float = 4 * 60 * 60,
    scoring_max_memory_bytes: int | None = None,
    scoring_max_cpu_seconds: int | None = None,
    scoring_device: str | None = None,
    alert_webhook_url: str | None = None,
    once: bool = False,
    report: Callable[[EvaluationCycleResult], None] | None = None,
) -> None:
    """Evaluate submissions and publish public artifacts without loading a wallet."""

    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if version_key < 0:
        raise ValueError("version identifier must be non-negative")
    alert = webhook_alerter(alert_webhook_url, service="veda-evaluator")
    track = load_track_spec(track_spec)
    data_config = None
    runtime_manifest_payload = None
    competition_manifest_payload = None
    if production_config is not None:
        data_config = load_producer_config(production_config)
        if data_config.track_spec.resolve() != Path(track_spec).resolve():
            raise ValueError("production configuration uses a different track specification")
        if data_config.validator_database.resolve() != Path(database).resolve():
            raise ValueError("production configuration uses a different validator database")
        runtime_manifest_payload = data_config.artifact_manifest.read_bytes()
        competition_manifest_payload = data_config.competition_payload(track)
        verify_competition_manifest(track, runtime_manifest_payload, competition_manifest_payload)
        approval = getattr(publisher, "release_approval", None)
        if approval is not None:
            publisher.release_approval = replace(
                approval, competition_manifest=competition_manifest_payload,
            )
    if production_config is not None:
        lock_publisher = getattr(publisher, "acquire_writer_lock", None)
        verify_public_storage = getattr(publisher, "verify_writes", None)
        if lock_publisher is None or verify_public_storage is None:
            raise ValueError("production publisher requires a single-writer lock and write checks")
        lock_publisher(database)
        await asyncio.to_thread(verify_public_storage)
    # Public storage can still be down after a restart. Keep ingesting finalized
    # facts, but do not initialize admission/scoring until ledger recovery succeeds.
    with ChallengeStore(database, track_spec_digest=track.digest) as recovery_store:
        recovery_store.reserve_uid_zero(
            track.initial_champion.hotkey,
            track_start_block=track.track_start_block,
        )
        while True:
            try:
                ledger = (
                    await publisher.recover(track) if runtime_manifest_payload is None
                    else await publisher.recover(
                        track, runtime_manifest=competition_manifest_payload,
                    )
                )
                break
            except Exception as exc:
                _record_publication_failure(recovery_store, alert, exc)
                scan = None
                finalized_block = None
                chain_error = None
                try:
                    async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
                        scan = await scan_finalized_blocks(
                            client, recovery_store, netuid=track.netuid,
                            track_start_block=track.track_start_block,
                            max_blocks=max_scan_blocks,
                        )
                        finalized_block = await client.finalized_block()
                except Exception as chain_exc:
                    if once:
                        raise
                    chain_error = str(chain_exc) or type(chain_exc).__name__
                if report is not None:
                    report(EvaluationCycleResult(
                        finalized_block, scan, None, None,
                        publication_error=str(exc) or type(exc).__name__,
                        cycle_error=chain_error,
                    ))
                if once:
                    return
                await asyncio.sleep(poll_seconds)
    historical_matches = tuple(getattr(ledger, "historical_matches", ()))
    recovery_matches = (*historical_matches, *ledger.matches)
    cache_dir = Path(exact_cache)
    if not cache_dir.is_dir():
        raise ValueError("exact cache path must be a batch directory")
    initial_model_path = (
        Path(initial_model)
        if initial_model is not None
        else await asyncio.to_thread(
            download_chunked_asset,
            initial_model_manifest_url,
            cache_dir=Path(model_cache) / "baseline",
            expected_sha256=track.initial_champion.evaluation_file_sha256,
            expected_model_digest=track.initial_champion.model_digest,
            max_bytes=track.model.max_bytes,
        )
    )
    verify_initial_champion(
        track,
        model_path=initial_model_path,
    )
    batch_source = None
    confirmation_pool = None
    retry_store = None
    if data_config is not None:
        scoring_device = data_config.validate_evaluation_device(scoring_device)
        data_track, data_manifest_payload = data_config.data_binding(
            track, runtime_manifest_payload,
        )
        objects = PrivateObjectStore.from_evaluator_environment(
            bucket=data_config.storage_bucket,
            prefix=f"{data_config.storage_prefix}/tracks/{data_track.digest}",
        )
        objects.verify_conditional_writes()
        stored_manifest = objects.get(ARTIFACT_MANIFEST_OBJECT).body
        if stored_manifest != data_manifest_payload:
            raise ValueError("private data manifest differs from the verified original binding")
        competition = ArtifactManifest.decode(stored_manifest)
        if competition.digest != data_track.artifact_manifest_sha256:
            raise ValueError("artifact manifest does not match the original data commitment")
        manifest = ArtifactManifest.decode(runtime_manifest_payload)
        _require_production_release(track, runtime_manifest_payload, publisher)
        hidden_artifacts = tuple(
            artifact
            for artifact in manifest.artifacts
            if artifact.logical_name == data_config.runtime.source_binding
            or artifact.logical_name.startswith(
                data_config.runtime.sample_cache.rstrip("/") + "/"
            )
        )
        objects.verify_evaluator_boundary(hidden_artifacts)
        image = data_config.verify_evaluator_manifest(manifest)
        _require_production_release(
            track, runtime_manifest_payload, publisher, image_id=image["image_id"],
        )
        batch_source = ReadyBatchSource(
            track=track,
            data_track=data_track,
            objects=objects,
            cache_dir=cache_dir,
            batch_dir=data_config.work_dir / "content",
            public_base_url=PUBLIC_STORAGE_URL,
        )
        retry_store = PrivateRetryStore(objects, track_spec_digest=track.digest)
        if track.promotion is None:
            confirmation_pool = ConfirmationWorkerPool(
                data_config,
                track,
                timeout_seconds=scoring_timeout_seconds,
            )
    scoring_settings = ScoringSettings(
        lease_ms=ceil(scoring_timeout_seconds * 2 * 1000) + 60_000,
        alert=alert,
        archive_stage=None if batch_source is None else batch_source.archive_stage,
        restore_stage=None if batch_source is None else batch_source.restore_stage,
        restore_retry=None if retry_store is None else retry_store.restore,
        record_retry=None if retry_store is None else retry_store.record_failure,
    )
    admission_settings = AdmissionSettings(
        cache_dir=Path(model_cache),
        max_model_bytes=track.model.max_bytes,
        tensor_schema=track.model.tensor_schema,
        alert=alert,
        archive=None if batch_source is None else batch_source.archive_model,
        archive_rejection=None if batch_source is None else batch_source.archive_rejection,
        restore=None if batch_source is None else batch_source.restore_admission,
        eligible=None if batch_source is None else batch_source.eligible_submission,
        restore_retry=None if retry_store is None else retry_store.restore,
        record_retry=None if retry_store is None else retry_store.record_failure,
    )

    async with AsyncExitStack() as resources:
        store = resources.enter_context(ChallengeStore(database, track_spec_digest=track.digest))
        if confirmation_pool is not None:
            resources.callback(confirmation_pool.close)
        uploads = PrivateUploads.from_environment(
            store=store, max_bytes=track.model.max_bytes,
            authority=None if batch_source is None else batch_source.objects,
        )
        if uploads is not None:
            if track.netuid != DEFAULT_NETUID or batch_source is None:
                raise ValueError("private upload admission requires SN120 and its models archive")
            admission_settings = replace(admission_settings, private_download=uploads.download)
            batch_source.private_uploads = uploads
            uploads.submission_timestamp = batch_source.block_timestamp_ms

            def current_private_champion(query_store):
                tournament = resolve_tournament(
                    query_store.public_snapshot()["challenges"], track_id=track.track_id,
                    track_spec_digest=track.digest, netuid=track.netuid,
                    minimum_gain=track.full_video.minimum_gain,
                    minimum_exact_gain=track.scoring.minimum_frontier_gain,
                    initial_champion=genesis_champion(track),
                    historical_matches=historical_matches, promotion=track.promotion,
                )
                return tournament.champion.private_submission

            uploads.current_champion = current_private_champion
        store.reserve_uid_zero(
            track.initial_champion.hotkey,
            track_start_block=track.track_start_block,
        )
        evaluator = PairedTrackEvaluator(
            track=track,
            store=store,
            cache_dir=cache_dir,
            model_cache=Path(model_cache),
            initial_model_path=initial_model_path,
            champion_archive_url=f"{PUBLIC_STORAGE_URL}/models/{track.track_id}",
            timeout_seconds=scoring_timeout_seconds,
            max_memory_bytes=scoring_max_memory_bytes,
            max_cpu_seconds=scoring_max_cpu_seconds,
            scoring_device=scoring_device,
            batch_source=batch_source,
            confirmation_pool=confirmation_pool,
            historical_matches=historical_matches,
        )
        work = _EvaluationWorker(
            database=database, track_spec_digest=track.digest,
            evaluator_factory=lambda worker_store: replace(evaluator, store=worker_store),
            admission_settings=admission_settings, scoring_settings=scoring_settings,
            worker_id=worker_id,
        )
        # Join the owner before closing its confirmation processes or main connection.
        resources.callback(work.close)
        recovery_index = 0
        publication: PublicArtifacts | None = None
        published_match_count = ledger.committed_match_count
        staged_match_count = published_match_count
        reconcile_publication = bool(ledger.matches)
        pending_artifacts: PublicArtifacts | None = None
        publication_task: asyncio.Task[PublicArtifacts] | None = None
        once_result: EvaluationCycleResult | None = None
        system_matches: tuple[tuple[str, bytes], ...] = ()
        system_match_task: asyncio.Task[tuple[tuple[str, bytes], ...]] | None = None
        next_system_match_refresh = 0.0
        system_match_loader = getattr(publisher, "load_system_matches", None)
        inventory_task: asyncio.Task[dict[str, object]] | None = None
        next_inventory_refresh = 0.0
        disclosure_task: asyncio.Task[str | None | Literal[False]] | None = None
        next_disclosure_refresh = 0.0
        upload_cleanup_task: asyncio.Task[dict[str, str | None]] | None = None
        next_upload_cleanup = 0.0

        def refresh_metadata() -> None:
            nonlocal system_match_task, system_matches, next_system_match_refresh
            nonlocal inventory_task, next_inventory_refresh
            nonlocal disclosure_task, next_disclosure_refresh
            nonlocal upload_cleanup_task, next_upload_cleanup
            if system_match_task is not None and system_match_task.done():
                try:
                    system_matches = system_match_task.result()
                except Exception as exc:
                    _LOG.warning(
                        "system benchmark refresh failed; retaining last valid summary: %s", exc,
                    )
                system_match_task = None
            now = asyncio.get_running_loop().time()
            if upload_cleanup_task is not None and upload_cleanup_task.done():
                inventory = dict(evaluator.inventory_status or {"allocated_stages": []})
                try:
                    upload_status = upload_cleanup_task.result()
                except Exception:
                    upload_status = {"private_upload_error":
                                     "PRIVATE_UPLOAD_ISSUANCE_STATUS_UNAVAILABLE"}
                for key, code in upload_status.items():
                    if code and code != inventory.get(key):
                        # Fixed code only: backend exceptions can contain credentials/URLs.
                        _LOG.warning("%s; original references retained", code)
                        if alert is not None:
                            alert(code, {})
                    inventory[key] = code
                evaluator.inventory_status = inventory
                upload_cleanup_task = None
            if uploads is not None and upload_cleanup_task is None and now >= next_upload_cleanup:
                upload_cleanup_task = asyncio.create_task(_reclaim_private_uploads(
                    uploads, database=database, network=network,
                    archive_endpoints=archive_endpoints,
                ))
                next_upload_cleanup = now + max(5.0, poll_seconds)
            if (
                system_match_loader is not None
                and system_match_task is None and now >= next_system_match_refresh
            ):
                system_match_task = asyncio.create_task(
                    system_match_loader(track, runtime_manifest_payload),
                )
                next_system_match_refresh = now + 300.0
            if inventory_task is not None and inventory_task.done():
                inventory = dict(evaluator.inventory_status or {"allocated_stages": []})
                try:
                    inventory.update(inventory_task.result(), refresh_error=None)
                except Exception as exc:
                    inventory["refresh_error"] = "INVENTORY_REFRESH_FAILED"
                    _LOG.warning("inventory refresh failed; retaining last observation: %s", exc)
                evaluator.inventory_status = inventory
                inventory_task = None
            source = evaluator.batch_source
            if source is not None and inventory_task is None and now >= next_inventory_refresh:
                def load_inventory() -> dict[str, object]:
                    return source.inventory_status(tuple(
                        plan for _, plan in source._plans()
                    ))
                inventory_task = asyncio.create_task(asyncio.to_thread(load_inventory))
                next_inventory_refresh = now + _DATA_REFRESH_SECONDS
            if disclosure_task is not None and disclosure_task.done():
                inventory = dict(evaluator.inventory_status or {"allocated_stages": []})
                try:
                    outcome = disclosure_task.result()
                except Exception as exc:
                    newly_failed = inventory.get("disclosure_error") != "DATA_DISCLOSURE_FAILURE"
                    inventory["disclosure_error"] = "DATA_DISCLOSURE_FAILURE"
                    _LOG.error("DATA_DISCLOSURE_FAILURE: %s", exc)
                    if alert is not None and newly_failed:
                        alert("DATA_DISCLOSURE_FAILURE", {"error": str(exc)[:500]})
                else:
                    if outcome is not False:
                        inventory.update(
                            disclosure_error=None, disclosure_observed_at_ns=time.time_ns(),
                        )
                evaluator.inventory_status = inventory
                disclosure_task = None
            cursor = store.chain_cursor()
            if (source is not None and cursor is not None and disclosure_task is None
                    and now >= next_disclosure_refresh):
                # A caught-up cursor fences late scans: do not prune any data
                # that an already-submitted, not-yet-discovered model could use.
                disclosure_task = asyncio.create_task(_publish_due_batch(
                    source, publisher, finalized_block=cursor[0],
                    challenges=store.public_snapshot()["challenges"],
                ))
                next_disclosure_refresh = now + _DATA_REFRESH_SECONDS

        async def close_system_match_refresh() -> None:
            tasks = [task for task in (system_match_task, inventory_task) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if disclosure_task is not None:
                # Joining preserves a single writer. Cancelling to_thread would
                # leave its S3 request alive and permit an overlapping restart.
                await asyncio.gather(asyncio.shield(disclosure_task), return_exceptions=True)
            if upload_cleanup_task is not None:
                # A live exact-key DELETE must end before the owner can be restarted.
                await asyncio.gather(asyncio.shield(upload_cleanup_task), return_exceptions=True)

        def start_publication() -> asyncio.Task[PublicArtifacts]:
            assert pending_artifacts is not None
            artifacts = pending_artifacts
            return asyncio.create_task(_stage_publication(
                artifacts,
                publisher=publisher,
            ))

        async def finish_publication() -> None:
            work.allowed = False
            if publication_task is not None:
                try:
                    await asyncio.shield(publication_task)
                except Exception as exc:
                    _record_publication_failure(store, alert, exc)

        resources.push_async_callback(finish_publication)
        resources.push_async_callback(close_system_match_refresh)
        consecutive_cycle_errors = 0
        while True:
            try:
                refresh_metadata()
                async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
                    while True:
                        refresh_metadata()
                        finalized_block = await client.finalized_block()
                        if finalized_block >= track.track_start_block:
                            break
                        if historical_matches:
                            await seed_legacy_exclusions(
                                client, store, netuid=track.netuid,
                                before_block=finalized_block,
                            )
                        chain = await read_weight_chain_state(
                            client,
                            netuid=track.netuid,
                            mechid=track.mechid,
                            finalized_block=finalized_block,
                        )
                        artifacts = build_genesis_artifacts(
                            track,
                            chain,
                            version_key=version_key,
                            runtime_manifest=runtime_manifest_payload,
                            release_approval=getattr(publisher, "release_approval", None),
                            system_matches=system_matches,
                            historical_matches=historical_matches,
                            excluded_hotkeys=store.legacy_excluded_hotkeys(),
                        )
                        try:
                            await publisher.publish(artifacts)
                        except Exception as exc:
                            _record_publication_failure(store, alert, exc)
                            raise
                        store.publication_succeeded()
                        if report is not None:
                            report(
                                EvaluationCycleResult(
                                    finalized_block,
                                    None,
                                    None,
                                    None,
                                    published_block=finalized_block,
                                )
                            )
                        if once:
                            return
                        await asyncio.sleep(poll_seconds)
                    await seed_legacy_exclusions(
                        client,
                        store,
                        netuid=track.netuid,
                        before_block=track.track_start_block - 1,
                    )
                    while True:
                        refresh_metadata()
                        result = await process_cycle(
                            client,
                            store,
                            netuid=track.netuid,
                            track_start_block=track.track_start_block,
                            work=work,
                            wait_for_work=once,
                            work_paused=(
                                reconcile_publication or pending_artifacts is not None
                                or staged_match_count != published_match_count
                            ),
                            published_match_count=published_match_count,
                            max_scan_blocks=max_scan_blocks,
                            recovery_matches=recovery_matches[recovery_index:],
                            private_uploads=uploads,
                            # One height identifies one immutable public snapshot.
                            # Keep finished work leased until it can enter the next
                            # snapshot, including after a partially failed publication.
                            minimum_finalized_block=max(
                                ledger.required_finalized_block,
                                0 if publication is None else publication.source_block + 1,
                            ),
                        )
                        recovery_index += result.restored_matches
                        refresh_metadata()
                        cursor = store.chain_cursor()
                        if (
                            cursor is not None
                            and cursor[0] == result.finalized_block
                            and result.finalized_block >= ledger.required_finalized_block
                        ):
                            try:
                                chain = await read_weight_chain_state(
                                    client,
                                    netuid=track.netuid,
                                    mechid=track.mechid,
                                    finalized_block=result.finalized_block,
                                )
                                stage_error = None
                                if (
                                    publication_task is not None and publication_task.done()
                                    and (
                                        publication is None
                                        or chain.finalized_block > publication.source_block
                                    )
                                ):
                                    finished, publication_task = publication_task, None
                                    try:
                                        prepared = finished.result()
                                    except Exception as exc:
                                        _record_publication_failure(store, alert, exc)
                                        stage_error = str(exc) or type(exc).__name__
                                    else:
                                        staged_match_count = len(prepared.matches)
                                        pending_artifacts = None
                                if (
                                    publication is None
                                    or publication.source_block != chain.finalized_block
                                ):
                                    arguments = {
                                        "version_key": version_key,
                                        "model_resolver": (
                                            None if batch_source is None
                                            else batch_source.restore_model
                                        ),
                                        "private_winner_archive": (
                                            None if batch_source is None
                                            else batch_source.archive_champion
                                        ),
                                        "runtime_manifest": runtime_manifest_payload,
                                        "system_matches": system_matches,
                                        "inventory_status": evaluator.inventory_status,
                                        "evaluation_running": work.busy and not work.done,
                                        "historical_matches": historical_matches,
                                        "release_approval": getattr(
                                            publisher, "release_approval", None,
                                        ),
                                    }
                                    candidate = None
                                    if (
                                        pending_artifacts is None
                                        and staged_match_count == published_match_count
                                    ):
                                        candidate = build_public_artifacts(
                                            store, track, chain, **arguments,
                                        )
                                        if (
                                            candidate is not None
                                            and (
                                                len(candidate.matches) > published_match_count
                                                or reconcile_publication
                                            )
                                        ):
                                            pending_artifacts = candidate
                                            reconcile_publication = False
                                            once_result = result
                                    publication = (
                                        candidate
                                        if candidate is not None
                                        and len(candidate.matches) == staged_match_count
                                        else build_public_artifacts(
                                            store, track, chain, match_limit=staged_match_count,
                                            **arguments,
                                        )
                                    )
                                if once and pending_artifacts is not None:
                                    if publication_task is None:
                                        publication_task = start_publication()
                                    await asyncio.shield(publication_task)
                                    # No old-prefix heartbeat was published in one-shot mode;
                                    # re-scan and bind the ready prefix to a fresh finalized head.
                                    publication = None
                                    continue
                                artifacts = publication
                                if artifacts is not None:
                                    await publisher.publish(artifacts)
                                    # Commit the small match prefix and fenced pointers.
                                    # Dataset publication has its own production-age clock.
                                    published_match_count = staged_match_count
                                    if pending_artifacts is None:
                                        store.publication_succeeded()
                                    if pending_artifacts is not None and publication_task is None:
                                        publication_task = start_publication()
                                    result = replace(
                                        result,
                                        published_block=artifacts.source_block,
                                        publication_error=stage_error,
                                    )
                                    if once and once_result is not None:
                                        result = replace(
                                            result, admission=once_result.admission,
                                            scoring=once_result.scoring,
                                        )
                            except Exception as exc:
                                _record_publication_failure(store, alert, exc)
                                result = EvaluationCycleResult(
                                    result.finalized_block,
                                    result.scan,
                                    result.admission,
                                    result.scoring,
                                    restored_matches=result.restored_matches,
                                    publication_error=str(exc) or type(exc).__name__,
                                )
                        if report is not None:
                            report(result)
                        consecutive_cycle_errors = 0
                        if once:
                            return
                        await asyncio.sleep(poll_seconds)
            except Exception as exc:
                if once:
                    raise
                if report is not None:
                    report(
                        EvaluationCycleResult(
                            None,
                            None,
                            None,
                            None,
                            cycle_error=str(exc) or type(exc).__name__,
                        )
                    )
                consecutive_cycle_errors += 1
                if alert is not None and (
                    consecutive_cycle_errors == 3 or consecutive_cycle_errors % 10 == 0
                ):
                    alert(
                        "EVALUATOR_CYCLE_FAILURE",
                        {
                            "consecutive_failures": consecutive_cycle_errors,
                            "error_type": type(exc).__name__,
                            "error": (str(exc) or type(exc).__name__)[:500],
                        },
                    )
                await asyncio.sleep(poll_seconds)


def _record_publication_failure(
    store: ChallengeStore,
    alert: Callable[[str, dict[str, Any]], None] | None,
    error: Exception,
) -> None:
    previous, current = store.record_publication_failure()
    if alert is not None and current != previous and current in {1, MAX_PUBLICATION_FAILURES}:
        alert(
            "PUBLICATION_FAILURE",
            {
                "consecutive_failures": current,
                "blocked": current >= MAX_PUBLICATION_FAILURES,
                "error": (str(error) or type(error).__name__)[:500],
            },
        )


async def run_validator(
    *,
    public_weights_url: str,
    network: str,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None = None,
    poll_seconds: float = 12.0,
    max_staleness_blocks: int = 100,
    once: bool = False,
    report: Callable[[WeightCycleResult], None] | None = None,
) -> None:
    """Fetch evaluator-produced weights, validate them, and submit with one Validator wallet."""

    if not network:
        raise ValueError("network cannot be empty")
    validate_public_weights_url(public_weights_url)
    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if max_staleness_blocks <= 0:
        raise ValueError("maximum staleness must be positive")
    wallet_arguments = {"name": wallet_name, "hotkey": hotkey_name}
    if wallet_path is not None:
        wallet_arguments["path"] = wallet_path
    wallet = bt.Wallet(**wallet_arguments)
    validator_hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if not isinstance(validator_hotkey, str) or not validator_hotkey:
        raise ValueError("validator wallet does not expose a hotkey address")

    async with (
        bt.Client(network) as client,
        httpx.AsyncClient(follow_redirects=False, timeout=30.0) as http_client,
    ):
        while True:
            try:
                result = await process_weight_cycle(
                    client,
                    http_client,
                    wallet,
                    validator_hotkey=validator_hotkey,
                    public_weights_url=public_weights_url,
                    max_staleness_blocks=max_staleness_blocks,
                )
            except Exception as exc:
                result = WeightCycleResult(
                    finalized_block=None,
                    source_block=None,
                    status="ERROR",
                    error=str(exc) or type(exc).__name__,
                )
            if report is not None:
                report(result)
            if once:
                if result.error is not None:
                    raise RuntimeError(result.error)
                return
            await asyncio.sleep(poll_seconds)


async def process_weight_cycle(
    client: Any,
    http_client: httpx.AsyncClient,
    wallet: Any,
    *,
    validator_hotkey: str,
    public_weights_url: str,
    max_staleness_blocks: int,
) -> WeightCycleResult:
    track = await fetch_track_identity(
        public_weights_url.rsplit("/", 1)[0] + "/track.json",
        client=http_client,
    )
    artifact = await fetch_weight_artifact(public_weights_url, client=http_client)
    if artifact.track_id != track.track_id:
        raise ValueError("public weights target a different track")
    chain = await read_weight_chain_state(client, netuid=track.netuid, mechid=track.mechid)
    plan = validate_weight_artifact(
        artifact,
        chain,
        expected_netuid=track.netuid,
        expected_track_digest=track.track_spec_digest,
        expected_mechid=track.mechid,
        max_staleness_blocks=max_staleness_blocks,
    )
    if plan is None:
        status = "TRACK_ABORTED" if artifact.track_aborted else "NO_WEIGHTS"
        return WeightCycleResult(chain.finalized_block, artifact.source_block, status)
    due, reason = weight_update_due(
        artifact,
        chain,
        validator_hotkey=validator_hotkey,
    )
    if not due:
        return WeightCycleResult(chain.finalized_block, artifact.source_block, reason)
    receipt = await submit_weights(
        plan,
        client=client,
        wallet=wallet,
        netuid=artifact.netuid,
        mechid=artifact.mechid,
        version_key=artifact.version_key,
    )
    return WeightCycleResult(
        chain.finalized_block,
        artifact.source_block,
        "SUBMITTED",
        receipt=receipt,
    )


async def process_cycle(
    client: Any,
    store: ChallengeStore,
    *,
    netuid: int,
    track_start_block: int,
    work: _EvaluationWorker,
    wait_for_work: bool = False,
    work_paused: bool = False,
    published_match_count: int = 0,
    max_scan_blocks: int = 100,
    recovery_matches: Sequence[MatchRecord] = (),
    minimum_finalized_block: int = 0,
    private_uploads: PrivateUploads | None = None,
) -> EvaluationCycleResult:
    """Scan/publish independently of one leased FIFO task; commit on the owner loop."""

    scoring_settings = work.scoring_settings
    if minimum_finalized_block < 0:
        raise ValueError("minimum finalized block cannot be negative")
    if type(published_match_count) is not int or published_match_count < 0:
        raise ValueError("published match count must be a non-negative integer")
    # A fresh DB scans from this release's start, not historical submissions.
    # Recover only manifest-authenticated decisions and their actual chain input
    # before scanning/admitting new work; never fabricate old payload/hash fields.
    known = {row["challenge_id"] for row in store.public_snapshot()["challenges"]}
    missing = [match for match in recovery_matches if match.challenge_id not in known]
    recovered_submissions = {}
    if missing:
        finalized = await client.finalized_block()
        by_block = {}
        for match in missing:
            height = match.document["submission"]["block_height"]
            if height > finalized:
                raise RuntimeError("historical submission is not finalized")
            if height not in by_block:
                _, commitments = await finalized_commitments(
                    client, netuid=netuid, block_height=height,
                )
                by_block[height] = {
                    record.submission_id: record for item in commitments
                    if (record := submission_from_commitment(item)) is not None
                }
            record = by_block[height].get(match.document["submission_id"])
            if record is None:
                raise RuntimeError("immutable match submission is absent from the finalized block")
            recovered_submissions[record.submission_id] = record
        store.restore_public_matches(recovery_matches, submissions=recovered_submissions)
    scan = await scan_finalized_blocks(
        client,
        store,
        netuid=netuid,
        track_start_block=track_start_block,
        max_blocks=max_scan_blocks,
        **({"private_uploads": private_uploads} if private_uploads is not None else {}),
    )
    finalized_block = await client.finalized_block()
    restored_matches = store.restore_public_matches(recovery_matches)
    cursor = store.chain_cursor()
    if (
        finalized_block < minimum_finalized_block
        or cursor is None or cursor[0] != finalized_block
        or restored_matches != len(recovery_matches)
    ):
        work.allowed = False
        return EvaluationCycleResult(
            finalized_block,
            scan,
            None,
            None,
            restored_matches=restored_matches,
        )
    terminal_prefix = 0
    for challenge in store.public_snapshot()["challenges"]:
        if challenge.get("track_spec_digest") not in (None, store.track_spec_digest()):
            continue
        if challenge["state"] not in TERMINAL_STATES:
            break
        terminal_prefix += 1
    work.allowed = not (
        work_paused or terminal_prefix > published_match_count
        or store.publication_failures() >= MAX_PUBLICATION_FAILURES
    )
    if not work.allowed:
        return EvaluationCycleResult(
            finalized_block,
            scan,
            None,
            None,
            restored_matches=restored_matches,
        )
    if not work.busy:
        work.start(finalized_block)
    completed = await work.take(wait=wait_for_work)
    if completed is None:
        return EvaluationCycleResult(
            finalized_block, scan, None, None, restored_matches=restored_matches,
        )
    admission, evaluated = completed
    if isinstance(evaluated, EvaluatedStage):
        try:
            finalized_block = await client.finalized_block()
        except Exception as exc:
            scoring = defer_evaluated_scoring(
                store,
                evaluated,
                settings=scoring_settings,
                code="FINALIZED_HEAD_UNAVAILABLE",
                summary=str(exc) or type(exc).__name__,
            )
        else:
            scoring = commit_evaluated_scoring(
                store,
                evaluated,
                current_finalized_block=finalized_block,
                settings=scoring_settings,
            )
    else:
        scoring = evaluated
    return EvaluationCycleResult(
        finalized_block,
        scan,
        admission,
        scoring,
        restored_matches=restored_matches,
    )
