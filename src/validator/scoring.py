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
    load_exact_cache,
    select_exact_cache_batch,
)
from scoring.evaluate import panel_seed, route_stratified_sample_indices
from scoring.exact import ExactCase, score_selector
from scoring.model import ModelValidationError, ValidatedModel, validate_model
from scoring.selector import load_selector
from scoring.spec import TrackSpec
from validator.download import ModelDownloadError, download_public_model
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

    def __post_init__(self) -> None:
        if min(self.lease_ms, self.retry_base_ms, self.retry_max_ms) <= 0:
            raise ValueError("scoring timing limits must be positive")
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")


@dataclass(frozen=True, slots=True)
class ScoringOutcome:
    challenge_id: str
    state: ChallengeState
    code: str | None = None


@dataclass(frozen=True, slots=True)
class TrackEvaluator:
    """Trusted authoritative evaluator over the frozen exact cache."""

    track: TrackSpec
    exact_cases: Sequence[ExactCase]
    presampled: bool = False

    def __call__(self, task: LeasedChallenge) -> StageOutcome:
        if task.model_path is None or task.file_sha256 is None:
            raise TemporaryScoringError(
                "SCORING_MODEL_UNAVAILABLE", "validated model path is unavailable"
            )
        try:
            current_digest = _file_sha256(Path(task.model_path))
        except OSError as exc:
            raise TemporaryScoringError(
                "SCORING_MODEL_UNAVAILABLE", "validated model file cannot be read"
            ) from exc
        if current_digest != task.file_sha256:
            raise TemporaryScoringError(
                "SCORING_MODEL_CHANGED", "validated model cache checksum changed"
            )
        try:
            selector = load_selector(
                task.model_path,
                self.track.model.selector,
                expected_hotkey=task.hotkey,
            )
        except ModelValidationError as exc:
            raise ParticipantScoringError(exc.code, str(exc)) from exc

        scoring = self.track.scoring
        try:
            if task.state is ChallengeState.SCORING:
                cases = _exact_cases_for_stage(
                    self.exact_cases,
                    count=scoring.sample_count,
                    seed=panel_seed(self.track.digest, stage="exact"),
                    routes=self.track.model.selector.routes,
                    strata=scoring.stratum_weight_map,
                    presampled=self.presampled,
                )
                result = score_selector(
                    selector,
                    cases,
                    scoring.budgets,
                    stratum_weights=scoring.stratum_weight_map,
                    tail_mean_weight=scoring.tail_mean_weight,
                    bootstrap_confidence=scoring.bootstrap_confidence,
                    bootstrap_resamples=scoring.bootstrap_resamples,
                    bootstrap_seed=scoring.bootstrap_seed,
                )
                return StageOutcome(
                    result.lower_bound,
                    {
                        "exact_score": result.aggregate.score,
                        "lower_bound": result.lower_bound,
                        "budget_scores": [item.quality for item in result.aggregate.budgets],
                        "sample_count": len(cases),
                    },
                )
        except (ParticipantScoringError, TemporaryScoringError):
            raise
        except ModelValidationError as exc:
            raise ParticipantScoringError(exc.code, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise TemporaryScoringError(
                "SCORING_CACHE_INVALID", "frozen scoring data is invalid"
            ) from exc
        raise TemporaryScoringError(
            "SCORING_STAGE_INVALID", f"unsupported scoring stage: {task.state}"
        )


@dataclass(frozen=True, slots=True)
class FilePairedEvaluator:
    """Load one batch once and score both sides of a match in one worker."""

    track: TrackSpec
    exact_cache: str
    exact_sha256: str
    incumbent: LeasedChallenge

    def __call__(self, challenger: LeasedChallenge) -> StageOutcome:
        try:
            scoring = self.track.scoring
            exact_cases = load_exact_cache(
                self.exact_cache,
                expected_sha256=self.exact_sha256,
                max_bytes=self.track.cache.exact_max_bytes,
            )
            if len(exact_cases) != scoring.sample_count:
                raise CacheLoadError("cache case count does not match the track")
        except CacheLoadError as exc:
            raise TemporaryScoringError("SCORING_CACHE_INVALID", str(exc)) from exc
        evaluator = TrackEvaluator(self.track, exact_cases, presampled=True)
        challenger_result = evaluator(challenger)
        incumbent_result = evaluator(self.incumbent)
        return StageOutcome(
            challenger_result.score,
            {
                **challenger_result.metrics,
                "incumbent_score": incumbent_result.score,
            },
        )


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
    timeout_seconds: float
    max_memory_bytes: int | None = None
    max_cpu_seconds: int | None = None
    downloader: Callable[..., Any] = download_public_model

    def __call__(self, task: LeasedChallenge) -> StageOutcome:
        snapshot = self.store.public_snapshot()
        tournament = resolve_tournament(
            snapshot["challenges"],
            track_id=self.track.track_id,
            track_spec_digest=self.track.digest,
            netuid=self.track.netuid,
            minimum_gain=self.track.scoring.minimum_frontier_gain,
            initial_champion=genesis_champion(self.track),
            minimum_generation_delay_blocks=(
                self.track.cache.minimum_generation_delay_blocks
            ),
        )
        incumbent = tournament.champion
        batch = self._batch(task)
        incumbent_task = self._incumbent_task(incumbent, snapshot["challenges"])
        evaluator = FilePairedEvaluator(
            self.track,
            str(batch.path),
            batch.sha256,
            incumbent_task,
        )
        isolated = IsolatedEvaluator(
            evaluator,
            timeout_seconds=self.timeout_seconds * 2,
            max_memory_bytes=self.max_memory_bytes,
            max_cpu_seconds=(
                None if self.max_cpu_seconds is None else self.max_cpu_seconds * 2
            ),
        )
        paired = isolated(task)
        return StageOutcome(
            paired.score,
            {
                **paired.metrics,
                "cache_sha256": batch.sha256,
                "cache_generated_block": batch.generated_block,
                "cache_source_digest": batch.source_digest,
            },
        )

    def _batch(self, task: LeasedChallenge) -> ExactCacheBatch:
        scoring = self.track.scoring
        try:
            return select_exact_cache_batch(
                self.cache_dir,
                submission_block=task.position.block,
                minimum_delay_blocks=self.track.cache.minimum_generation_delay_blocks,
                max_bytes=self.track.cache.exact_max_bytes,
                min_blocks=scoring.budgets[0].blocks,
                max_blocks=scoring.budgets[-1].blocks,
                head_dim=self.track.model.selector.head_dim,
                routes=self.track.model.selector.routes,
                strata=scoring.stratum_weight_map,
                sample_count=scoring.sample_count,
            )
        except CacheLoadError as exc:
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


def process_next_scoring(
    store: ChallengeStore,
    *,
    worker_id: str,
    current_finalized_block: int,
    settings: ScoringSettings,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    now_ms: int | None = None,
) -> ScoringOutcome | None:
    """Evaluate and commit at most one ready scoring stage."""

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
            result = evaluator(task)
            generated_block = result.metrics.get("cache_generated_block")
            if generated_block is not None and (
                type(generated_block) is not int
                or generated_block > current_finalized_block
            ):
                raise TemporaryScoringError(
                    "SCORING_CACHE_INVALID",
                    "cache generation block is not finalized",
                )
            require_transition(task.state, ChallengeState.SUCCEEDED)
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
        store.record_stage_score(
            task.challenge_id,
            task.lease_token,
            task.state,
            score=result.score,
            score_block=current_finalized_block,
            metrics=result.metrics,
            occurred_at_ms=now,
        )
        result_score_block = current_finalized_block
    else:
        result = StageOutcome(
            recorded.score,
            recorded.metrics,
        )
        result_score_block = recorded.score_block
    store.complete_challenge(
        task.challenge_id,
        ChallengeState.SUCCEEDED,
        final_score=result.score,
        occurred_at_ms=now,
        lease_token=task.lease_token,
        score_block=result_score_block,
    )
    return ScoringOutcome(task.challenge_id, ChallengeState.SUCCEEDED)


def _defer_or_fail(
    store: ChallengeStore,
    task: LeasedChallenge,
    settings: ScoringSettings,
    now_ms: int,
    *,
    code: str,
    summary: str,
) -> ChallengeState:
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


def _exact_cases_for_stage(
    cases: Sequence[ExactCase],
    *,
    count: int,
    seed: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
    presampled: bool,
) -> tuple[ExactCase, ...]:
    if not presampled:
        indices = route_stratified_sample_indices(
            [case.route for case in cases],
            [case.stratum for case in cases],
            count,
            seed,
            routes,
            strata,
        )
        return tuple(cases[index] for index in indices)
    if len(cases) != count:
        raise ValueError("sampled cache returned the wrong number of cases")
    return tuple(cases)


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
