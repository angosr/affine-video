"""Durable retry state shared by admission and scoring."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from validator.queue import ChallengeState
from validator.store import ChallengeStore, LeasedChallenge


@dataclass(frozen=True, slots=True)
class RetryCheckpoint:
    attempts: int
    blocked: bool
    retry_after_ms: int
    error_code: str
    error_summary: str

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise ValueError("retry attempts must be positive")
        if type(self.retry_after_ms) is not int or self.retry_after_ms < 0:
            raise ValueError("retry time must be non-negative")
        if not self.error_code or not self.error_summary:
            raise ValueError("retry error identity cannot be empty")


RetryRestore = Callable[[LeasedChallenge, str], RetryCheckpoint | None]
RetryRecorder = Callable[..., RetryCheckpoint]


def retry_clock(fixed_time_ms: int | None) -> Callable[[], int]:
    """Read wall time at each event; an explicit test time remains fixed."""

    return lambda: time.time_ns() // 1_000_000 if fixed_time_ms is None else fixed_time_ms


def prepare_durable_retry(
    store: ChallengeStore,
    task: LeasedChallenge,
    *,
    stage: str,
    clock: Callable[[], int],
    restore: RetryRestore | None,
    wait_ms: int,
) -> tuple[LeasedChallenge | None, RetryCheckpoint | None]:
    """Restore a durable retry fence before any participant or GPU I/O."""

    if restore is None:
        return task, None
    checkpoint = restore(task, stage)
    if checkpoint is None:
        return task, None
    now_ms = clock()
    if checkpoint.blocked and checkpoint.retry_after_ms > now_ms:
        store.synchronize_leased_attempts(
            task.challenge_id,
            task.lease_token,
            attempts=checkpoint.attempts,
        )
        store.block_leased(
            task.challenge_id,
            task.lease_token,
            error_code=checkpoint.error_code,
            error_summary=checkpoint.error_summary,
            retry_after_ms=checkpoint.retry_after_ms,
            occurred_at_ms=now_ms,
        )
        return None, checkpoint
    ordinal = checkpoint.attempts + 1
    store.synchronize_leased_attempts(
        task.challenge_id,
        task.lease_token,
        attempts=ordinal,
    )
    task = replace(task, attempts=ordinal)
    if checkpoint.retry_after_ms > now_ms:
        store.wait_leased(
            task.challenge_id,
            task.lease_token,
            retry_after_ms=max(checkpoint.retry_after_ms, now_ms + wait_ms),
            reason="DURABLE_RETRY_BACKOFF",
            occurred_at_ms=now_ms,
        )
        return None, checkpoint
    return task, checkpoint


def persist_retry_failure(
    task: LeasedChallenge,
    *,
    stage: str,
    code: str,
    summary: str,
    retry_after_ms: int,
    max_attempts: int,
    record: RetryRecorder | None,
) -> RetryCheckpoint | None:
    """Commit one failed attempt before changing disposable local state."""

    if record is None:
        return None
    checkpoint = record(
        task,
        stage,
        code=code,
        summary=summary,
        retry_after_ms=retry_after_ms,
        max_attempts=max_attempts,
    )
    if checkpoint.attempts != task.attempts:
        raise RuntimeError("durable retry ordinal disagrees with the active lease")
    return checkpoint


def retry_stage(state: ChallengeState) -> str:
    if state in {ChallengeState.FETCHING, ChallengeState.VALIDATING}:
        return "ADMISSION"
    if state in {ChallengeState.SCORING, ChallengeState.CONFIRMING}:
        return state.value
    raise ValueError("challenge state has no retry stage")
