"""Lease-safe scoring of admitted models.

Admission owns permanent model rejection. Subsequent input failures are retryable;
non-finite model outputs receive zero quality inside the exact/video scorers.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import resource
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

from scoring.cache import (
    CacheLoadError,
    ExactCacheBatch,
    ExactCacheReader,
)
from scoring.exact import score_selector_pair
from scoring.model import ModelValidationError, ValidatedModel, validate_model
from scoring.selector import load_selector
from scoring.spec import TrackSpec
from validator.download import ModelDownloadError, download_chunked_file, download_public_model
from validator.production.batches import ReadyBatchSource
from validator.production.confirmation import ConfirmationWorkerPool
from validator.queue import ChallengeState, require_transition
from validator.retries import (
    RetryRecorder,
    RetryRestore,
    persist_retry_failure,
    prepare_durable_retry,
    retry_clock,
    retry_stage,
)
from validator.store import ChallengeStore, LeasedChallenge
from validator.tournament import (
    MatchRecord,
    challenge_from_match,
    genesis_champion,
    resolve_tournament,
)


class TemporaryScoringError(RuntimeError):
    """An infrastructure failure that may succeed when retried."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Aggregate result returned by the trusted scoring implementation."""

    score: float
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("stage score must be finite and within [0, 1]")
        try:
            json.dumps(
                self.metrics,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("stage metrics must be canonical JSON values") from exc


@dataclass(frozen=True, slots=True)
class ScoringSettings:
    lease_ms: int = 30 * 60 * 1000
    retry_base_ms: int = 60 * 1000
    retry_max_ms: int = 60 * 60 * 1000
    max_attempts: int = 9
    alert: Callable[[str, Mapping[str, object]], None] | None = None
    archive_stage: Callable[[LeasedChallenge, StageOutcome, int], None] | None = None
    restore_stage: Callable[[LeasedChallenge], object | None] | None = None
    restore_retry: RetryRestore | None = None
    record_retry: RetryRecorder | None = None

    def __post_init__(self) -> None:
        if min(self.lease_ms, self.retry_base_ms, self.retry_max_ms, self.max_attempts) <= 0:
            raise ValueError("scoring timing limits must be positive")
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")
        if (self.restore_retry is None) != (self.record_retry is None):
            raise ValueError("durable scoring retry callbacks must be configured together")


@dataclass(frozen=True, slots=True)
class ScoringOutcome:
    challenge_id: str
    state: ChallengeState
    code: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluatedStage:
    """A leased stage whose clock stays live through decision and archival I/O."""

    task: LeasedChallenge
    result: StageOutcome
    recorded_score_block: int | None
    recorded_locally: bool
    clock: Callable[[], int]


@dataclass(frozen=True, slots=True)
class FilePairedEvaluator:
    """Load one batch once and score both sides of a match in one worker."""

    track: TrackSpec
    exact_cache: str
    exact_sha256: str
    incumbent: LeasedChallenge
    scoring_device: str | None = None

    def __call__(self, challenger: LeasedChallenge) -> StageOutcome:
        try:
            scoring = self.track.scoring
            challenger_selector = _load_task_selector(self.track, challenger)
            incumbent_selector = _load_task_selector(self.track, self.incumbent)
            reader = ExactCacheReader(
                self.exact_cache,
                expected_sha256=self.exact_sha256,
                max_bytes=self.track.cache.exact_max_bytes,
            )
            if len(reader.metadata["cases"]) != scoring.case_count:
                raise CacheLoadError("cache case count does not match the track")
            with reader:
                paired = score_selector_pair(
                    challenger_selector,
                    incumbent_selector,
                    reader,
                    scoring.budgets,
                    stratum_weights=scoring.stratum_weight_map,
                    step_window_weights=scoring.step_window_weight_map,
                    tail_mean_weight=scoring.tail_mean_weight,
                    confidence=self.track.scoring.match_confidence,
                    resamples=scoring.bootstrap_resamples,
                    seed=scoring.bootstrap_seed,
                    device=self.scoring_device,
                )
        except (CacheLoadError, ValueError, OSError) as exc:
            raise TemporaryScoringError("SCORING_CACHE_INVALID", str(exc)) from exc
        return StageOutcome(
            paired.challenger_score,
            {
                "case_count": len(paired.challenger_cells),
                "exact_score": paired.challenger_score,
                "incumbent_score": paired.incumbent_score,
                "paired_gain_lower_bound": paired.gain_lower_bound,
            },
        )


def _load_task_selector(track: TrackSpec, task: LeasedChallenge):
    if task.model_path is None or task.file_sha256 is None:
        raise OSError("validated model path is unavailable")
    path = Path(task.model_path)
    if _file_sha256(path) != task.file_sha256:
        raise OSError("validated model cache checksum changed")
    expected_hotkey = None if task.challenge_id == "genesis" else task.hotkey
    return load_selector(path, track.model.selector, expected_hotkey=expected_hotkey)


def verify_initial_champion(
    track: TrackSpec,
    *,
    model_path: str | Path,
) -> ValidatedModel:
    """Verify the fixed official VEDA baseline model before processing Miners."""

    initial = track.initial_champion
    try:
        validated = validate_model(
            model_path,
            schema=track.model.tensor_schema,
            max_bytes=track.model.max_bytes,
        )
    except (OSError, ModelValidationError) as exc:
        raise ValueError("initial VEDA evaluation weights are invalid") from exc
    if validated.file_sha256 != initial.evaluation_file_sha256:
        raise ValueError("initial VEDA evaluation file SHA-256 is incorrect")
    if validated.model_digest != initial.model_digest:
        raise ValueError("initial VEDA canonical model digest is incorrect")
    return validated


@dataclass(slots=True)
class PairedTrackEvaluator:
    """Score the FIFO challenger on data fixed at its original submission."""

    track: TrackSpec
    store: ChallengeStore
    cache_dir: Path
    model_cache: Path
    initial_model_path: Path
    champion_archive_url: str
    timeout_seconds: float
    max_memory_bytes: int | None = None
    max_cpu_seconds: int | None = None
    downloader: Callable[..., Any] = download_public_model
    seed_block_hashes: Mapping[int, str] | None = None
    scoring_device: str | None = None
    batch_source: ReadyBatchSource | None = None
    confirmation_pool: ConfirmationWorkerPool | None = None
    inventory_status: Mapping[str, Any] | None = None
    historical_matches: tuple[MatchRecord, ...] = ()

    def ready_through_block(self, current_finalized_block: int) -> int:
        """No artificial block delay: every finalized admitted submission is eligible."""
        return current_finalized_block

    def batch_ready(self, task: LeasedChallenge) -> bool:
        """Distinguish expected producer lag from an evaluation failure."""

        if self.batch_source is None:
            return True
        record = self.store.public_challenge(task.challenge_id)
        return self.batch_source.claim_ready(
            task.challenge_id,
            confirmation=task.state is ChallengeState.CONFIRMING,
            file_sha256=task.file_sha256,
            model_digest=record["model_digest"],
            submission_block=task.position.block,
        )

    def __call__(self, task: LeasedChallenge) -> StageOutcome:
        snapshot = self.store.public_snapshot()
        task = self._restore_challenger(task, snapshot["challenges"])
        tournament = resolve_tournament(
            snapshot["challenges"],
            track_id=self.track.track_id,
            track_spec_digest=self.track.digest,
            netuid=self.track.netuid,
            minimum_gain=self.track.full_video.minimum_gain,
            minimum_exact_gain=self.track.scoring.minimum_frontier_gain,
            initial_champion=genesis_champion(self.track),
            historical_matches=self.historical_matches,
        )
        incumbent = tournament.champion
        incumbent_task = self._incumbent_task(incumbent, snapshot["challenges"])
        if task.state is ChallengeState.CONFIRMING:
            return self._confirm(task, incumbent_task)
        batch = self._batch(task)
        evaluator = FilePairedEvaluator(
            self.track,
            str(batch.path),
            batch.sha256,
            incumbent_task,
            self.scoring_device,
        )
        isolated = IsolatedEvaluator(
            evaluator,
            timeout_seconds=self.timeout_seconds * 2,
            max_memory_bytes=self.max_memory_bytes,
            max_cpu_seconds=(None if self.max_cpu_seconds is None else self.max_cpu_seconds * 2),
        )
        paired = isolated(task)
        return StageOutcome(
            paired.score,
            {
                **paired.metrics,
                **self.batch_source.stage_binding(task.challenge_id, confirmation=False),
                "cache_sha256": batch.sha256,
                "cache_generated_block": batch.generated_block,
                "cache_plan_digest": batch.plan_digest,
                "cache_seed_block_hash": batch.seed_block_hash,
                "requires_confirmation": (
                    paired.metrics["paired_gain_lower_bound"]
                    > self.track.scoring.minimum_frontier_gain
                ),
            },
        )

    def _restore_challenger(
        self, task: LeasedChallenge, challenges: Sequence[Mapping[str, Any]]
    ) -> LeasedChallenge:
        """Recover the admitted identity from private storage before either scoring stage."""

        record = next(item for item in challenges if item["challenge_id"] == task.challenge_id)
        try:
            if task.file_sha256 is None or task.file_sha256 != record["file_sha256"]:
                raise RuntimeError("challenger identity is unavailable or changed")
            path = Path(task.model_path) if task.model_path is not None else None
            if (
                path is None
                or not path.is_file()
                or path.stat().st_size != record["model_size_bytes"]
                or _file_sha256(path) != task.file_sha256
            ):
                if self.batch_source is None:
                    raise RuntimeError("private challenger archive is unavailable")
                path = self.batch_source.restore_model(
                    task.file_sha256, record["model_size_bytes"]
                )
            validated = validate_model(
                path,
                schema=self.track.model.tensor_schema,
                max_bytes=self.track.model.max_bytes,
                expected_hotkey=task.hotkey,
            )
            if (
                validated.file_sha256 != task.file_sha256
                or validated.size_bytes != record["model_size_bytes"]
                or validated.model_digest != record["model_digest"]
            ):
                raise RuntimeError("restored challenger identity changed")
        except (OSError, RuntimeError, ModelValidationError) as exc:
            raise TemporaryScoringError(
                "CHALLENGER_MODEL_UNAVAILABLE", "admitted challenger cannot be restored"
            ) from exc
        return replace(task, model_path=str(path))

    def _confirm(
        self,
        challenger: LeasedChallenge,
        incumbent: LeasedChallenge,
    ) -> StageOutcome:
        if self.batch_source is None or self.confirmation_pool is None:
            raise TemporaryScoringError(
                "CONFIRMATION_BATCH_UNAVAILABLE",
                "full-video confirmation is not configured",
            )
        try:
            _load_task_selector(self.track, challenger)
            _load_task_selector(self.track, incumbent)
            batch = self.batch_source.confirmation_batch(challenger.challenge_id)
            exact = self.store.stage_score(challenger.challenge_id, ChallengeState.SCORING)
            if exact is None:
                raise RuntimeError("confirmation is missing its completed exact decision")
            exact_plan = self.batch_source._allocated_plan(
                challenger.challenge_id, confirmation=False,
            )
            if (batch.plan.plan_digest == exact_plan.plan_digest or {
                proof.leaf.source_family_id for proof in exact_plan.source_proofs
            }.intersection(proof.leaf.source_family_id for proof in batch.plan.source_proofs)):
                raise RuntimeError("confirmation inventory is not independent of exact")
            self._require_finalized_seed_hash(batch.plan.generation_block, batch.plan.block_hash)
            result = self.confirmation_pool.run(
                challenger_path=challenger.model_path or "",
                incumbent_path=incumbent.model_path or "",
                batch=batch,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise TemporaryScoringError("CONFIRMATION_BATCH_UNAVAILABLE", str(exc)) from exc
        return StageOutcome(
            result.challenger_score,
            {
                **self.batch_source.stage_binding(challenger.challenge_id, confirmation=True),
                "cache_generated_block": batch.plan.generation_block,
                "cache_plan_digest": batch.plan.plan_digest,
                "cache_seed_block_hash": batch.plan.block_hash,
                "cache_sha256": batch.exact.sha256,
                "case_count": len(result.challenger_cells),
                "confirmation_score": result.challenger_score,
                "incumbent_score": result.incumbent_score,
                "paired_gain_lower_bound": result.gain_lower_bound,
            },
        )

    def _batch(self, task: LeasedChallenge) -> ExactCacheBatch:
        try:
            if self.batch_source is None:
                raise CacheLoadError("a durable inventory allocation is required")
            batch = self.batch_source.exact_batch(task.challenge_id)
            self._require_finalized_seed_hash(batch.generated_block, batch.seed_block_hash)
            if self.track.cache.generator != "veda-private-rollout-v1":
                raise CacheLoadError("track cache generator is unsupported")
            return batch
        except (CacheLoadError, RuntimeError) as exc:
            raise TemporaryScoringError("SCORING_CACHE_UNAVAILABLE", str(exc)) from exc

    def _require_finalized_seed_hash(self, generated_block: int, seed_block_hash: str) -> None:
        """Reject both stages unless their seed matches the finalized chain snapshot."""

        expected_hash = (
            self.store.finalized_block_hash(generated_block)
            if self.seed_block_hashes is None else self.seed_block_hashes.get(generated_block)
        )
        if (expected_hash is None and self.seed_block_hashes is None
                and generated_block < self.track.track_start_block
                and self.batch_source is not None):
            expected_hash = self.batch_source.finalized_seed_hash(generated_block)
        if not isinstance(expected_hash, str) or seed_block_hash.lower() != expected_hash.lower():
            raise CacheLoadError("cache seed hash does not match its finalized generation block")

    def _incumbent_task(
        self,
        incumbent: Any,
        challenges: Sequence[Mapping[str, Any]],
    ) -> LeasedChallenge:
        if incumbent.challenge_id == "genesis":
            path = self.initial_model_path
            expected_digest = self.track.initial_champion.model_digest
            expected_hotkey = None
            expected_sha256 = self.track.initial_champion.evaluation_file_sha256
            expected_size = None
            validated = self._validate_incumbent_candidate(
                path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                expected_digest=expected_digest,
                expected_hotkey=expected_hotkey,
            )
        else:
            original = next((match for match in self.historical_matches
                             if match.challenge_id == incumbent.challenge_id), None)
            record = next(
                (item for item in challenges if item["challenge_id"] == incumbent.challenge_id),
                None if original is None else challenge_from_match(original.document),
            )
            if record is None:
                raise TemporaryScoringError(
                    "INCUMBENT_MODEL_UNAVAILABLE", "confirmed champion facts are missing",
                )
            expected_digest = record["model_digest"]
            expected_hotkey = incumbent.hotkey
            expected_sha256 = record["file_sha256"]
            expected_size = record["model_size_bytes"]
            try:
                local_path = self.store.model_path(incumbent.challenge_id)
            except KeyError:
                local_path = None
            path = None if local_path is None else Path(local_path)
            validated = self._validate_incumbent_candidate(
                path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                expected_digest=expected_digest,
                expected_hotkey=expected_hotkey,
            )
            if validated is None and path is not None and path.is_file():
                try:
                    path.unlink()
                except OSError as exc:
                    raise TemporaryScoringError(
                        "INCUMBENT_MODEL_UNAVAILABLE", "cannot discard corrupt champion cache"
                    ) from exc
            if validated is None and self.batch_source is not None:
                try:
                    path = self.batch_source.restore_model(expected_sha256, expected_size)
                except (OSError, RuntimeError):
                    path = None
                validated = self._validate_incumbent_candidate(
                    path,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                    expected_digest=expected_digest,
                    expected_hotkey=expected_hotkey,
                )
            if validated is None:
                try:
                    archive_url = self.champion_archive_url
                    if original is not None:
                        archive_url = (
                            f"{archive_url.rstrip('/').rsplit('/', 1)[0]}/"
                            f"{original.document['track_id']}"
                        )
                    path = download_chunked_file(
                        f"{archive_url.rstrip('/')}/{expected_sha256}/manifest.json",
                        cache_dir=self.model_cache / "champions",
                        expected_sha256=expected_sha256,
                        max_bytes=self.track.model.max_bytes,
                        suffix=".safetensors",
                    )
                except RuntimeError:
                    path = None
                validated = self._validate_incumbent_candidate(
                    path,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                    expected_digest=expected_digest,
                    expected_hotkey=expected_hotkey,
                )
            if validated is None:
                try:
                    downloaded = self.downloader(
                        incumbent.repo_id,
                        incumbent.revision,
                        cache_dir=self.model_cache,
                        max_bytes=self.track.model.max_bytes,
                    )
                except ModelDownloadError as exc:
                    raise TemporaryScoringError(
                        "INCUMBENT_MODEL_UNAVAILABLE", str(exc)
                    ) from exc
                path = Path(downloaded.path)
                validated = self._validate_incumbent_candidate(
                    path,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                    expected_digest=expected_digest,
                    expected_hotkey=expected_hotkey,
                )
        if validated is None or path is None:
            raise TemporaryScoringError(
                "INCUMBENT_MODEL_INVALID", "current champion model cannot be restored"
            )
        return LeasedChallenge(
            challenge_id=incumbent.challenge_id,
            submission_id=incumbent.challenge_id,
            state=ChallengeState.SCORING,
            hotkey=incumbent.hotkey,
            repo_id=incumbent.repo_id,
            revision=incumbent.revision,
            position=incumbent.position,
            lease_token="incumbent",
            lease_expires_at_ms=0,
            attempts=1,
            model_path=str(path),
            file_sha256=validated.file_sha256,
        )

    def _validate_incumbent_candidate(
        self,
        path: Path | None,
        *,
        expected_sha256: str,
        expected_size: int | None,
        expected_digest: str,
        expected_hotkey: str | None,
    ) -> ValidatedModel | None:
        if path is None:
            return None
        try:
            if (
                not path.is_file()
                or (expected_size is not None and path.stat().st_size != expected_size)
                or _file_sha256(path) != expected_sha256
            ):
                return None
            validated = validate_model(
                path,
                schema=self.track.model.tensor_schema,
                max_bytes=self.track.model.max_bytes,
                expected_hotkey=expected_hotkey,
            )
        except (OSError, ModelValidationError):
            return None
        if (
            validated.file_sha256 != expected_sha256
            or (expected_size is not None and validated.size_bytes != expected_size)
            or validated.model_digest != expected_digest
        ):
            return None
        return validated


@dataclass(frozen=True, slots=True)
class IsolatedEvaluator:
    """Run the trusted scorer in a fresh, resource-bounded child process."""

    evaluator: Callable[[LeasedChallenge], StageOutcome]
    timeout_seconds: float
    max_memory_bytes: int | None = None
    max_cpu_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("scoring timeout must be finite and positive")
        if self.max_memory_bytes is not None and self.max_memory_bytes <= 0:
            raise ValueError("memory limit must be positive")
        if self.max_cpu_seconds is not None and self.max_cpu_seconds <= 0:
            raise ValueError("CPU limit must be positive")

    def __call__(self, task: LeasedChallenge) -> StageOutcome:
        return evaluate_isolated(
            self.evaluator,
            task,
            timeout_seconds=self.timeout_seconds,
            max_memory_bytes=self.max_memory_bytes,
            max_cpu_seconds=self.max_cpu_seconds,
        )


def evaluate_isolated(
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    task: LeasedChallenge,
    *,
    timeout_seconds: float,
    max_memory_bytes: int | None = None,
    max_cpu_seconds: int | None = None,
) -> StageOutcome:
    """Execute one stage with spawn isolation and a bounded wall-clock wait."""

    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("scoring timeout must be finite and positive")
    if max_memory_bytes is not None and max_memory_bytes <= 0:
        raise ValueError("memory limit must be positive")
    if max_cpu_seconds is not None and max_cpu_seconds <= 0:
        raise ValueError("CPU limit must be positive")

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_entry,
        args=(sender, evaluator, task, max_memory_bytes, max_cpu_seconds),
    )
    try:
        process.start()
    except Exception as exc:
        sender.close()
        receiver.close()
        raise TemporaryScoringError(
            "SCORING_WORKER_FAILURE",
            "scoring process could not be started",
        ) from exc
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            _stop_process(process)
            raise TemporaryScoringError(
                "SCORING_TIMEOUT",
                "scoring process exceeded the wall-clock limit",
            )
        try:
            message = receiver.recv()
        except EOFError:
            message = None
    finally:
        receiver.close()

    process.join(timeout=1)
    if process.is_alive():
        _stop_process(process)
        raise TemporaryScoringError(
            "SCORING_WORKER_FAILURE",
            "scoring process did not exit after producing a result",
        )
    if message is None:
        if process.exitcode == -signal.SIGXCPU:
            raise TemporaryScoringError(
                "SCORING_CPU_LIMIT",
                "scoring process exceeded the fixed CPU limit",
            )
        raise TemporaryScoringError(
            "SCORING_WORKER_FAILURE",
            f"scoring process exited without a result ({process.exitcode})",
        )

    kind, *payload = message
    if kind == "ok":
        result = payload[0]
        if not isinstance(result, StageOutcome):
            raise TemporaryScoringError(
                "SCORING_WORKER_FAILURE",
                "scoring process returned an invalid result",
            )
        return result
    code, summary = payload
    raise TemporaryScoringError(code, summary)


def evaluate_next_scoring(
    store: ChallengeStore,
    *,
    worker_id: str,
    current_finalized_block: int,
    settings: ScoringSettings,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    now_ms: int | None = None,
) -> EvaluatedStage | ScoringOutcome | None:
    """Evaluate at most one stage while retaining its lease for atomic commit."""

    clock = retry_clock(now_ms)
    now = clock()
    readiness = getattr(evaluator, "ready_through_block", None)
    ready_through_block = None if readiness is None else readiness(current_finalized_block)
    task = store.claim_next_scoring(
        worker_id,
        lease_ms=settings.lease_ms,
        ready_through_block=ready_through_block,
        now_ms=now,
    )
    if task is None:
        return None

    challenge_id = task.challenge_id
    task, restored_retry = prepare_durable_retry(
        store,
        task,
        stage=retry_stage(task.state),
        clock=clock,
        restore=settings.restore_retry,
        wait_ms=settings.retry_base_ms,
    )
    if task is None:
        assert restored_retry is not None
        return ScoringOutcome(
            challenge_id,
            (
                ChallengeState.SYSTEM_BLOCKED
                if restored_retry.blocked
                else ChallengeState.SYSTEM_DEFERRED
            ),
            restored_retry.error_code,
        )

    recorded = store.stage_score(task.challenge_id, task.state)
    if recorded is None:
        try:
            archived = None if settings.restore_stage is None else settings.restore_stage(task)
            batch_ready = getattr(evaluator, "batch_ready", None)
            if archived is None and batch_ready is not None and not batch_ready(task):
                raise TemporaryScoringError(
                    "EVALUATION_BATCH_UNAVAILABLE",
                    "the original pre-submission data binding is unavailable; "
                    "later production cannot replace it",
                )
            if archived is None:
                result = evaluator(task)
                recorded_score_block = None
            else:
                result = StageOutcome(archived.score, archived.metrics)
                recorded_score_block = archived.score_block
                if recorded_score_block > current_finalized_block:
                    raise TemporaryScoringError(
                        "SCORING_ARCHIVE_AHEAD",
                        "archived stage result is newer than the finalized chain head",
                    )
            generated_block = result.metrics.get("cache_generated_block")
            if generated_block is not None and (
                type(generated_block) is not int or generated_block > current_finalized_block
            ):
                raise TemporaryScoringError(
                    "SCORING_CACHE_INVALID",
                    "cache generation block is not finalized",
                )
            target = (
                ChallengeState.CONFIRMING
                if task.state is ChallengeState.SCORING
                and result.metrics.get("requires_confirmation") is True
                else ChallengeState.SUCCEEDED
            )
            require_transition(task.state, target)
        except TemporaryScoringError as exc:
            state = _defer_or_fail(
                store,
                task,
                settings,
                clock(),
                code=exc.code,
                summary=str(exc),
            )
            return ScoringOutcome(task.challenge_id, state, exc.code)
        except Exception as exc:
            state = _defer_or_fail(
                store,
                task,
                settings,
                clock(),
                code="SCORING_WORKER_FAILURE",
                summary=str(exc) or type(exc).__name__,
            )
            return ScoringOutcome(
                task.challenge_id,
                state,
                "SCORING_WORKER_FAILURE",
            )
        recorded_locally = False
    else:
        if recorded.score_block > current_finalized_block:
            state = _defer_or_fail(
                store,
                task,
                settings,
                clock(),
                code="SCORING_RECORD_AHEAD",
                summary="recorded stage result is newer than the finalized chain head",
            )
            return ScoringOutcome(task.challenge_id, state, "SCORING_RECORD_AHEAD")
        result = StageOutcome(
            recorded.score,
            recorded.metrics,
        )
        recorded_score_block = recorded.score_block
        recorded_locally = True
    return EvaluatedStage(task, result, recorded_score_block, recorded_locally, clock)


def commit_evaluated_scoring(
    store: ChallengeStore,
    evaluated: EvaluatedStage,
    *,
    current_finalized_block: int,
    settings: ScoringSettings,
) -> ScoringOutcome:
    """Persist a successful stage against a finalized block read after evaluation."""

    task = evaluated.task
    result = evaluated.result
    result_score_block = (
        current_finalized_block
        if evaluated.recorded_score_block is None
        else evaluated.recorded_score_block
    )
    if not evaluated.recorded_locally:
        if settings.archive_stage is not None:
            try:
                settings.archive_stage(task, result, result_score_block)
            except Exception as exc:
                state = _defer_or_fail(
                    store,
                    task,
                    settings,
                    evaluated.clock(),
                    code="STAGE_ARCHIVE_UNAVAILABLE",
                    summary=str(exc) or type(exc).__name__,
                )
                return ScoringOutcome(task.challenge_id, state, "STAGE_ARCHIVE_UNAVAILABLE")
        store.record_stage_score(
            task.challenge_id,
            task.lease_token,
            task.state,
            score=result.score,
            score_block=result_score_block,
            metrics=result.metrics,
            occurred_at_ms=evaluated.clock(),
        )
    if task.state is ChallengeState.SCORING and result.metrics.get("requires_confirmation") is True:
        store.advance_leased(
            task.challenge_id,
            task.lease_token,
            ChallengeState.CONFIRMING,
            release=True,
            occurred_at_ms=evaluated.clock(),
            details={"exact_score": result.score},
        )
        return ScoringOutcome(task.challenge_id, ChallengeState.CONFIRMING)
    store.complete_challenge(
        task.challenge_id,
        ChallengeState.SUCCEEDED,
        final_score=result.score,
        occurred_at_ms=evaluated.clock(),
        lease_token=task.lease_token,
        score_block=result_score_block,
    )
    return ScoringOutcome(task.challenge_id, ChallengeState.SUCCEEDED)


def defer_evaluated_scoring(
    store: ChallengeStore,
    evaluated: EvaluatedStage,
    *,
    settings: ScoringSettings,
    code: str,
    summary: str,
) -> ScoringOutcome:
    """Release an evaluated stage when its post-evaluation commit precondition failed."""

    state = _defer_or_fail(
        store,
        evaluated.task,
        settings,
        evaluated.clock(),
        code=code,
        summary=summary,
    )
    return ScoringOutcome(evaluated.task.challenge_id, state, code)


def process_next_scoring(
    store: ChallengeStore,
    *,
    worker_id: str,
    current_finalized_block: int,
    settings: ScoringSettings,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    now_ms: int | None = None,
) -> ScoringOutcome | None:
    """Evaluate and commit one stage using a caller-supplied finalized block."""

    evaluated = evaluate_next_scoring(
        store,
        worker_id=worker_id,
        current_finalized_block=current_finalized_block,
        settings=settings,
        evaluator=evaluator,
        now_ms=now_ms,
    )
    if evaluated is None or isinstance(evaluated, ScoringOutcome):
        return evaluated
    return commit_evaluated_scoring(
        store,
        evaluated,
        current_finalized_block=current_finalized_block,
        settings=settings,
    )


def _defer_or_fail(
    store: ChallengeStore,
    task: LeasedChallenge,
    settings: ScoringSettings,
    now_ms: int,
    *,
    code: str,
    summary: str,
) -> ChallengeState:
    attempts = task.attempts
    multiplier = 1 << min(max(attempts - 1, 0), 20)
    delay = min(settings.retry_max_ms, settings.retry_base_ms * multiplier)
    if attempts >= settings.max_attempts:
        delay = settings.retry_max_ms
    checkpoint = persist_retry_failure(
        task,
        stage=retry_stage(task.state),
        code=code,
        summary=summary,
        retry_after_ms=now_ms + delay,
        max_attempts=settings.max_attempts,
        record=settings.record_retry,
    )
    if checkpoint is not None:
        attempts = checkpoint.attempts
    if attempts == 1 and attempts < settings.max_attempts and settings.alert is not None:
        settings.alert(
            "SCORING_RETRY_STARTED",
            {
                "challenge_id": task.challenge_id,
                "attempts": attempts,
                "code": code,
                "stage": str(task.state),
            },
        )
    blocked = checkpoint.blocked if checkpoint is not None else attempts >= settings.max_attempts
    retry_after_ms = now_ms + delay if checkpoint is None else checkpoint.retry_after_ms
    if blocked:
        store.block_leased(
            task.challenge_id,
            task.lease_token,
            error_code=code,
            error_summary=summary,
            retry_after_ms=retry_after_ms,
            occurred_at_ms=now_ms,
        )
        if settings.alert is not None:
            settings.alert(
                "SCORING_RETRY_EXHAUSTED",
                {
                    "challenge_id": task.challenge_id,
                    "attempts": attempts,
                    "code": code,
                    "stage": str(task.state),
                    "retry_after_ms": retry_after_ms,
                },
            )
        return ChallengeState.SYSTEM_BLOCKED
    store.defer_leased(
        task.challenge_id,
        task.lease_token,
        retry_after_ms=retry_after_ms,
        error_code=code,
        error_summary=summary,
        occurred_at_ms=now_ms,
    )
    return ChallengeState.SYSTEM_DEFERRED


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop_process(process: multiprocessing.Process) -> None:
    """Terminate a scoring child without an unbounded cleanup wait."""

    if not process.is_alive():
        process.join(timeout=0)
        return
    process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def _isolated_entry(
    sender: Any,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    task: LeasedChallenge,
    max_memory_bytes: int | None,
    max_cpu_seconds: int | None,
) -> None:
    try:
        if max_memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        if max_cpu_seconds is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (max_cpu_seconds, max_cpu_seconds + 1),
            )
        result = evaluator(task)
        if not isinstance(result, StageOutcome):
            raise TypeError("evaluator did not return StageOutcome")
        sender.send(("ok", result))
    except TemporaryScoringError as exc:
        sender.send(("temporary", exc.code, str(exc)))
    except MemoryError:
        sender.send(
            (
                "temporary",
                "SCORING_MEMORY_LIMIT",
                "scoring process exceeded the fixed memory limit",
            )
        )
    except BaseException as exc:
        sender.send(
            (
                "temporary",
                "SCORING_WORKER_FAILURE",
                str(exc) or type(exc).__name__,
            )
        )
    finally:
        sender.close()
