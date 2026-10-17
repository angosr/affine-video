"""Lease-safe orchestration of one staged scoring task."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import resource
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from scoring.cache import (
    CacheLoadError,
    ExactCacheBatch,
    ExactCacheReader,
    select_exact_cache_batch,
)
from scoring.exact import score_selector_pair
from scoring.model import ModelValidationError, ValidatedModel, validate_model
from scoring.selector import load_selector
from scoring.spec import TrackSpec
from validator.download import ModelDownloadError, download_chunked_file, download_public_model
from validator.production.batches import ReadyBatchSource
from validator.production.confirmation import ConfirmationWorkerPool
from validator.production.plan import scheduled_generation_block
from validator.queue import ChallengeState, require_transition
from validator.store import ChallengeStore, LeasedChallenge
from validator.tournament import genesis_champion, resolve_tournament


class ParticipantScoringError(RuntimeError):
    """A deterministic model failure that makes the submission invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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

    def __post_init__(self) -> None:
        if min(self.lease_ms, self.retry_base_ms, self.retry_max_ms, self.max_attempts) <= 0:
            raise ValueError("scoring timing limits must be positive")
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")


@dataclass(frozen=True, slots=True)
class ScoringOutcome:
    challenge_id: str
    state: ChallengeState
    code: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluatedStage:
    """A leased successful stage waiting for a fresh finalized decision block."""

    task: LeasedChallenge
    result: StageOutcome
    recorded_score_block: int | None
    recorded_locally: bool
    occurred_at_ms: int


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
                    confidence=self.track.match_confidence,
                    resamples=scoring.bootstrap_resamples,
                    seed=scoring.bootstrap_seed,
                    device=self.scoring_device,
                )
        except ModelValidationError as exc:
            raise ParticipantScoringError(exc.code, str(exc)) from exc
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
    return load_selector(path, track.model.selector, expected_hotkey=task.hotkey)


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
    """Score challenger and current champion on one post-submission cache batch."""

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

    def __call__(self, task: LeasedChallenge) -> StageOutcome:
        snapshot = self.store.public_snapshot()
        tournament = resolve_tournament(
            snapshot["challenges"],
            track_id=self.track.track_id,
            track_spec_digest=self.track.digest,
            netuid=self.track.netuid,
            minimum_gain=self.track.full_video.minimum_gain,
            minimum_exact_gain=self.track.scoring.minimum_frontier_gain,
            initial_champion=genesis_champion(self.track),
            minimum_generation_delay_blocks=(self.track.cache.minimum_generation_delay_blocks),
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
            expected_block = scheduled_generation_block(
                exact.score_block,
                track_start_block=self.track.track_start_block,
                minimum_generation_delay_blocks=self.track.cache.minimum_generation_delay_blocks,
                generation_interval_blocks=self.track.cache.generation_interval_blocks,
            )
            if batch.plan.generation_block != expected_block:
                raise RuntimeError("confirmation batch does not follow the exact decision block")
            result = self.confirmation_pool.run(
                challenger_path=challenger.model_path or "",
                incumbent_path=incumbent.model_path or "",
                batch=batch,
            )
        except ModelValidationError as exc:
            raise ParticipantScoringError(exc.code, str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise TemporaryScoringError("CONFIRMATION_BATCH_UNAVAILABLE", str(exc)) from exc
        return StageOutcome(
            result.challenger_score,
            {
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
        scoring = self.track.scoring
        try:
            if self.batch_source is None:
                batch = select_exact_cache_batch(
                    self.cache_dir,
                    submission_block=task.position.block,
                    minimum_delay_blocks=self.track.cache.minimum_generation_delay_blocks,
                    track_start_block=self.track.track_start_block,
                    generation_interval_blocks=self.track.cache.generation_interval_blocks,
                    expected_track_spec_digest=self.track.digest,
                    max_bytes=self.track.cache.exact_max_bytes,
                    head_dim=self.track.model.selector.head_dim,
                    tile_size=128,
                    routes=self.track.model.selector.routes,
                    strata=scoring.stratum_weight_map,
                    sampled_route_count=scoring.sampled_route_count,
                    rollouts_per_stratum=scoring.rollouts_per_stratum,
                    sampled_step_count=scoring.sampled_step_count,
                    query_tiles_per_case=scoring.query_tiles_per_case,
                )
            else:
                batch = self.batch_source.exact_batch(task.challenge_id)
                expected_block = scheduled_generation_block(
                    task.position.block,
                    track_start_block=self.track.track_start_block,
                    minimum_generation_delay_blocks=(
                        self.track.cache.minimum_generation_delay_blocks
                    ),
                    generation_interval_blocks=self.track.cache.generation_interval_blocks,
                )
                if batch.generated_block != expected_block:
                    raise CacheLoadError("challenge READY plan uses the wrong generation block")
            if self.seed_block_hashes is not None:
                expected_hash = self.seed_block_hashes.get(batch.generated_block)
                if expected_hash is None or batch.seed_block_hash.lower() != expected_hash.lower():
                    raise CacheLoadError(
                        "cache seed hash does not match its finalized generation block"
                    )
            if self.track.cache.generator != "veda-private-rollout-v1":
                raise CacheLoadError("track cache generator is unsupported")
            return batch
        except (CacheLoadError, RuntimeError) as exc:
            raise TemporaryScoringError("SCORING_CACHE_UNAVAILABLE", str(exc)) from exc

    def _incumbent_task(
        self,
        incumbent: Any,
        challenges: Sequence[Mapping[str, Any]],
    ) -> LeasedChallenge:
        if incumbent.challenge_id == "genesis":
            path = self.initial_model_path
            expected_digest = self.track.initial_champion.model_digest
            expected_hotkey = None
        else:
            record = next(
                item for item in challenges if item["challenge_id"] == incumbent.challenge_id
            )
            expected_digest = record["model_digest"]
            expected_hotkey = incumbent.hotkey
            local_path = self.store.model_path(incumbent.challenge_id)
            if local_path is None or not Path(local_path).is_file():
                path = None
                if self.batch_source is not None:
                    try:
                        path = self.batch_source.restore_model(
                            record["file_sha256"], record["model_size_bytes"]
                        )
                    except (OSError, RuntimeError):
                        path = None
                try:
                    if path is None:
                        path = download_chunked_file(
                            f"{self.champion_archive_url.rstrip('/')}/{record['file_sha256']}/manifest.json",
                            cache_dir=self.model_cache / "champions",
                            expected_sha256=record["file_sha256"],
                            max_bytes=self.track.model.max_bytes,
                            suffix=".safetensors",
                        )
                except RuntimeError:
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
            else:
                path = Path(local_path)
        try:
            validated = validate_model(
                path,
                schema=self.track.model.tensor_schema,
                max_bytes=self.track.model.max_bytes,
                expected_hotkey=expected_hotkey,
            )
        except (OSError, ModelValidationError) as exc:
            raise TemporaryScoringError(
                "INCUMBENT_MODEL_INVALID", "current champion model cannot be restored"
            ) from exc
        if validated.model_digest != expected_digest:
            raise TemporaryScoringError(
                "INCUMBENT_MODEL_INVALID", "current champion model digest changed"
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
    if kind == "participant":
        raise ParticipantScoringError(code, summary)
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

    now = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    task = store.claim_next_scoring(
        worker_id,
        lease_ms=settings.lease_ms,
        now_ms=now,
    )
    if task is None:
        return None

    recorded = store.stage_score(task.challenge_id, task.state)
    if recorded is None:
        try:
            archived = None if settings.restore_stage is None else settings.restore_stage(task)
            if archived is None:
                result = evaluator(task)
                recorded_score_block = None
            else:
                result = StageOutcome(archived.score, archived.metrics)
                recorded_score_block = archived.score_block
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
        except ParticipantScoringError as exc:
            store.complete_challenge(
                task.challenge_id,
                ChallengeState.PARTICIPANT_FAILED,
                error_code=exc.code,
                error_summary=str(exc),
                occurred_at_ms=now,
                lease_token=task.lease_token,
                score_block=current_finalized_block,
            )
            return ScoringOutcome(
                task.challenge_id,
                ChallengeState.PARTICIPANT_FAILED,
                exc.code,
            )
        except TemporaryScoringError as exc:
            state = _defer_or_fail(
                store,
                task,
                settings,
                now,
                code=exc.code,
                summary=str(exc),
            )
            return ScoringOutcome(task.challenge_id, state, exc.code)
        except Exception as exc:
            state = _defer_or_fail(
                store,
                task,
                settings,
                now,
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
        result = StageOutcome(
            recorded.score,
            recorded.metrics,
        )
        recorded_score_block = recorded.score_block
        recorded_locally = True
    return EvaluatedStage(task, result, recorded_score_block, recorded_locally, now)


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
                    evaluated.occurred_at_ms,
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
            occurred_at_ms=evaluated.occurred_at_ms,
        )
    if task.state is ChallengeState.SCORING and result.metrics.get("requires_confirmation") is True:
        store.advance_leased(
            task.challenge_id,
            task.lease_token,
            ChallengeState.CONFIRMING,
            release=True,
            occurred_at_ms=evaluated.occurred_at_ms,
            details={"exact_score": result.score},
        )
        return ScoringOutcome(task.challenge_id, ChallengeState.CONFIRMING)
    store.complete_challenge(
        task.challenge_id,
        ChallengeState.SUCCEEDED,
        final_score=result.score,
        occurred_at_ms=evaluated.occurred_at_ms,
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
        evaluated.occurred_at_ms,
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
    if task.attempts >= settings.max_attempts:
        store.block_leased(
            task.challenge_id,
            task.lease_token,
            error_code=code,
            error_summary=summary,
            occurred_at_ms=now_ms,
        )
        if settings.alert is not None:
            settings.alert(
                "SCORING_RETRY_EXHAUSTED",
                {
                    "challenge_id": task.challenge_id,
                    "attempts": task.attempts,
                    "code": code,
                    "stage": str(task.state),
                },
            )
        return ChallengeState.SYSTEM_BLOCKED
    multiplier = 1 << min(max(task.attempts - 1, 0), 20)
    delay = min(settings.retry_max_ms, settings.retry_base_ms * multiplier)
    store.defer_leased(
        task.challenge_id,
        task.lease_token,
        retry_after_ms=now_ms + delay,
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
    except ParticipantScoringError as exc:
        sender.send(("participant", exc.code, str(exc)))
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
