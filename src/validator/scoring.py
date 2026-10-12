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

from scoring.cache import CacheLoadError, sample_exact_cache, sample_screen_cache
from scoring.evaluate import challenge_seed, stratified_sample_indices
from scoring.local import (
    ExactCase,
    ScreenCase,
    evaluate_selector_local,
    evaluate_selector_screen,
)
from scoring.metrics import quality_score
from scoring.model import ModelValidationError
from scoring.selector import FixedSelector, load_selector
from scoring.spec import TrackSpec
from validator.queue import ChallengeState, require_transition
from validator.store import ChallengeStore, LeasedChallenge


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

    next_state: ChallengeState
    score: float
    attention_cost: float
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("stage score must be finite and within [0, 1]")
        if not isfinite(self.attention_cost) or self.attention_cost < 0:
            raise ValueError("attention cost must be finite and non-negative")
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
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if min(self.lease_ms, self.retry_base_ms, self.retry_max_ms) <= 0:
            raise ValueError("scoring timing limits must be positive")
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")
        if self.max_attempts <= 0:
            raise ValueError("maximum attempts must be positive")


@dataclass(frozen=True, slots=True)
class ScoringOutcome:
    challenge_id: str
    state: ChallengeState
    code: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutResult:
    audit_factor: float
    attention_cost: float
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isfinite(self.audit_factor) or not 0 <= self.audit_factor <= 1:
            raise ValueError("rollout audit factor must be within [0, 1]")
        if not isfinite(self.attention_cost) or self.attention_cost < 0:
            raise ValueError("rollout attention cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TrackEvaluator:
    """Trusted staged evaluator over frozen lightweight and exact caches."""

    track: TrackSpec
    screen_cases: Sequence[ScreenCase]
    exact_cases: Sequence[ExactCase]
    rollout: Callable[[LeasedChallenge, FixedSelector, TrackSpec], RolloutResult] | None = None
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
            selector = load_selector(task.model_path, self.track.model.selector)
        except ModelValidationError as exc:
            raise ParticipantScoringError(exc.code, str(exc)) from exc

        seed = challenge_seed(
            self.track.digest,
            task.submission_id,
            task.seed_block_hash,
        )
        scoring = self.track.scoring
        try:
            if task.state is ChallengeState.SCREENING:
                cases = _cases_for_stage(
                    self.screen_cases,
                    count=scoring.screen_sample_count,
                    seed=seed ^ 0x53435245,
                    strata=scoring.stratum_weight_map,
                    presampled=self.presampled,
                )
                result = evaluate_selector_screen(
                    selector,
                    cases,
                    scoring.budgets,
                    stratum_weights=scoring.stratum_weight_map,
                    bootstrap_confidence=scoring.bootstrap_confidence,
                    bootstrap_resamples=scoring.bootstrap_resamples,
                    bootstrap_seed=scoring.bootstrap_seed,
                )
                target = (
                    ChallengeState.SCREENED_OUT
                    if result.upper_bound < scoring.screening_threshold
                    else ChallengeState.SCORING
                )
                return StageOutcome(
                    target,
                    result.score,
                    scoring.budgets[-1].calibrated_cost,
                    {
                        "upper_bound": result.upper_bound,
                        "budget_scores": list(result.budget_scores),
                        "sample_count": len(cases),
                    },
                )
            if task.state is ChallengeState.SCORING:
                cases = _cases_for_stage(
                    self.exact_cases,
                    count=scoring.local_sample_count,
                    seed=seed ^ 0x4C4F4341,
                    strata=scoring.stratum_weight_map,
                    presampled=self.presampled,
                )
                result = evaluate_selector_local(
                    selector,
                    cases,
                    scoring.budgets,
                    stratum_weights=scoring.stratum_weight_map,
                    tail_mean_weight=scoring.tail_mean_weight,
                    bootstrap_confidence=scoring.bootstrap_confidence,
                    bootstrap_resamples=scoring.bootstrap_resamples,
                    bootstrap_seed=scoring.bootstrap_seed,
                )
                score = quality_score(
                    local_lcb=result.lower_bound,
                    q_floor=scoring.q_floor,
                    gamma=scoring.gamma,
                    audit_factor=1.0,
                )
                target = (
                    ChallengeState.ROLLOUT
                    if self.rollout is not None and result.lower_bound >= scoring.rollout_threshold
                    else ChallengeState.PROVISIONAL
                )
                return StageOutcome(
                    target,
                    score,
                    scoring.budgets[-1].calibrated_cost,
                    {
                        "local_score": result.aggregate.score,
                        "lower_bound": result.lower_bound,
                        "budget_scores": [item.quality for item in result.aggregate.budgets],
                        "sample_count": len(cases),
                    },
                )
            if task.state is ChallengeState.ROLLOUT:
                if self.rollout is None:
                    raise TemporaryScoringError(
                        "ROLLOUT_UNAVAILABLE", "rollout evaluator is not configured"
                    )
                if task.current_score is None:
                    raise TemporaryScoringError(
                        "ROLLOUT_SCORE_MISSING", "provisional score is unavailable"
                    )
                rollout = self.rollout(task, selector, self.track)
                if not isinstance(rollout, RolloutResult):
                    raise TemporaryScoringError(
                        "ROLLOUT_RESULT_INVALID", "rollout returned an invalid result"
                    )
                return StageOutcome(
                    ChallengeState.SUCCEEDED,
                    task.current_score * rollout.audit_factor,
                    rollout.attention_cost,
                    {**rollout.metrics, "audit_factor": rollout.audit_factor},
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
class FileTrackEvaluator:
    """Load only the cache needed by the current stage inside the worker."""

    track: TrackSpec
    screen_cache: str
    exact_cache: str
    rollout: Callable[[LeasedChallenge, FixedSelector, TrackSpec], RolloutResult] | None = None

    def __call__(self, task: LeasedChallenge) -> StageOutcome:
        try:
            seed = challenge_seed(
                self.track.digest,
                task.submission_id,
                task.seed_block_hash,
            )
            scoring = self.track.scoring
            if task.state is ChallengeState.SCREENING:
                screen_cases = sample_screen_cache(
                    self.screen_cache,
                    expected_sha256=self.track.cache.screen_sha256,
                    max_bytes=self.track.cache.screen_max_bytes,
                    count=scoring.screen_sample_count,
                    seed=seed ^ 0x53435245,
                    strata=scoring.stratum_weight_map,
                )
                exact_cases: Sequence[ExactCase] = ()
            elif task.state is ChallengeState.SCORING:
                screen_cases = ()
                exact_cases = sample_exact_cache(
                    self.exact_cache,
                    expected_sha256=self.track.cache.exact_sha256,
                    max_bytes=self.track.cache.exact_max_bytes,
                    count=scoring.local_sample_count,
                    seed=seed ^ 0x4C4F4341,
                    strata=scoring.stratum_weight_map,
                )
            else:
                screen_cases = ()
                exact_cases = ()
        except CacheLoadError as exc:
            raise TemporaryScoringError("SCORING_CACHE_INVALID", str(exc)) from exc
        return TrackEvaluator(
            self.track,
            screen_cases,
            exact_cases,
            rollout=self.rollout,
            presampled=True,
        )(task)


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
        current_finalized_block=current_finalized_block,
        lease_ms=settings.lease_ms,
        now_ms=now,
    )
    if task is None:
        return None

    try:
        result = evaluator(task)
        require_transition(task.state, result.next_state)
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
        state = _defer_or_fail(store, task, settings, now, code=exc.code, summary=str(exc))
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
        attention_cost=result.attention_cost,
        metrics=result.metrics,
        occurred_at_ms=now,
    )
    if (
        result.next_state in {ChallengeState.PROVISIONAL, ChallengeState.ROLLOUT}
        and task.state is ChallengeState.SCORING
    ):
        store.publish_provisional(
            task.challenge_id,
            task.lease_token,
            final_score=result.score,
            attention_cost=result.attention_cost,
            score_block=current_finalized_block,
            schedule_rollout=result.next_state is ChallengeState.ROLLOUT,
            occurred_at_ms=now,
        )
    elif result.next_state is ChallengeState.SUCCEEDED:
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.SUCCEEDED,
            final_score=result.score,
            attention_cost=result.attention_cost,
            occurred_at_ms=now,
            lease_token=task.lease_token,
            score_block=current_finalized_block,
        )
    elif result.next_state is ChallengeState.SCREENED_OUT:
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.SCREENED_OUT,
            error_code="SCREENING_THRESHOLD",
            error_summary="screening upper bound is below the fixed threshold",
            occurred_at_ms=now,
            lease_token=task.lease_token,
            score_block=current_finalized_block,
        )
    else:
        store.advance_leased(
            task.challenge_id,
            task.lease_token,
            result.next_state,
            release=True,
            occurred_at_ms=now,
            details={"stage_score": result.score},
        )
    return ScoringOutcome(task.challenge_id, result.next_state)


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
        store.fail_leased_system(
            task.challenge_id,
            task.lease_token,
            error_code=code,
            error_summary=summary,
            occurred_at_ms=now_ms,
        )
        return ChallengeState.SYSTEM_FAILED
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


def _sample_stratified[T: (ScreenCase, ExactCase)](
    cases: Sequence[T],
    count: int,
    seed: int,
    strata: Mapping[str, float],
) -> tuple[T, ...]:
    indices = stratified_sample_indices(
        [case.stratum for case in cases],
        count,
        seed,
        strata,
    )
    return tuple(cases[index] for index in indices)


def _cases_for_stage[T: (ScreenCase, ExactCase)](
    cases: Sequence[T],
    *,
    count: int,
    seed: int,
    strata: Mapping[str, float],
    presampled: bool,
) -> tuple[T, ...]:
    if not presampled:
        return _sample_stratified(cases, count, seed, strata)
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
