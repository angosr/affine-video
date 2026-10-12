"""One recoverable admission task: download, validate, and deduplicate."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scoring.model import ModelValidationError, TensorSpec, ValidatedModel, validate_model
from validator.download import (
    DownloadedModel,
    ModelDownloadError,
    download_public_model,
)
from validator.queue import ChallengeState
from validator.store import ChallengeStore


class AdmissionStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    DEFERRED = "DEFERRED"
    SYSTEM_FAILED = "SYSTEM_FAILED"


@dataclass(frozen=True, slots=True)
class AdmissionSettings:
    cache_dir: Path
    max_model_bytes: int
    tensor_schema: Mapping[str, TensorSpec]
    lease_ms: int = 5 * 60 * 1000
    retry_base_ms: int = 30 * 1000
    retry_max_ms: int = 30 * 60 * 1000
    max_attempts: int = 5

    def __post_init__(self) -> None:
        if (
            min(
                self.max_model_bytes,
                self.lease_ms,
                self.retry_base_ms,
                self.retry_max_ms,
            )
            <= 0
        ):
            raise ValueError("admission limits must be positive")
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")
        if self.max_attempts <= 0:
            raise ValueError("maximum attempts must be positive")


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    challenge_id: str
    status: AdmissionStatus
    code: str | None = None
    duplicate_of: str | None = None


def process_next_admission(
    store: ChallengeStore,
    *,
    worker_id: str,
    settings: AdmissionSettings,
    now_ms: int | None = None,
    downloader: Callable[..., DownloadedModel] = download_public_model,
    model_validator: Callable[..., ValidatedModel] = validate_model,
) -> AdmissionOutcome | None:
    """Process at most one FIFO admission task."""

    now = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    task = store.claim_next_admission(
        worker_id,
        lease_ms=settings.lease_ms,
        now_ms=now,
    )
    if task is None:
        return None

    reference_owner = store.earlier_reference_owner(task.challenge_id)
    if reference_owner is not None:
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.PARTICIPANT_FAILED,
            error_code="REFERENCE_DUPLICATE",
            error_summary="repository revision duplicates an earlier active submission",
            occurred_at_ms=now,
            lease_token=task.lease_token,
        )
        return AdmissionOutcome(
            task.challenge_id,
            AdmissionStatus.FAILED,
            "REFERENCE_DUPLICATE",
            reference_owner,
        )

    model_path = Path(task.model_path) if task.model_path is not None else None
    expected_file_sha256 = task.file_sha256
    if task.state is ChallengeState.FETCHING:
        try:
            downloaded = downloader(
                task.repo_id,
                task.revision,
                cache_dir=settings.cache_dir,
                max_bytes=settings.max_model_bytes,
            )
        except ModelDownloadError as exc:
            if exc.temporary:
                status = _defer_or_fail(
                    store,
                    task.challenge_id,
                    task.lease_token,
                    task.attempts,
                    settings,
                    now,
                    exc,
                )
                return AdmissionOutcome(task.challenge_id, status, exc.code)
            store.complete_challenge(
                task.challenge_id,
                ChallengeState.PARTICIPANT_FAILED,
                error_code=exc.code,
                error_summary=str(exc),
                occurred_at_ms=now,
                lease_token=task.lease_token,
            )
            return AdmissionOutcome(task.challenge_id, AdmissionStatus.FAILED, exc.code)

        store.record_downloaded_model(
            task.challenge_id,
            task.lease_token,
            model_path=downloaded.path,
            file_sha256=downloaded.file_sha256,
            size_bytes=downloaded.size_bytes,
            occurred_at_ms=now,
            details={"etag": downloaded.etag, "download_host": downloaded.download_host},
        )
        model_path = downloaded.path
        expected_file_sha256 = downloaded.file_sha256

    if model_path is None:
        error = ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE", "cached model path is unavailable", temporary=True
        )
        status = _defer_or_fail(
            store,
            task.challenge_id,
            task.lease_token,
            task.attempts,
            settings,
            now,
            error,
            resume_state=ChallengeState.FETCHING,
        )
        return AdmissionOutcome(task.challenge_id, status, error.code)

    try:
        validated = model_validator(
            model_path,
            schema=dict(settings.tensor_schema),
            max_bytes=settings.max_model_bytes,
        )
    except OSError:
        error = ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE", "cached model cannot be read", temporary=True
        )
        status = _defer_or_fail(
            store,
            task.challenge_id,
            task.lease_token,
            task.attempts,
            settings,
            now,
            error,
            resume_state=ChallengeState.FETCHING,
        )
        return AdmissionOutcome(task.challenge_id, status, error.code)
    except ModelValidationError as exc:
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.PARTICIPANT_FAILED,
            error_code=exc.code,
            error_summary=str(exc),
            occurred_at_ms=now,
            lease_token=task.lease_token,
        )
        return AdmissionOutcome(task.challenge_id, AdmissionStatus.FAILED, exc.code)

    if expected_file_sha256 is not None and validated.file_sha256 != expected_file_sha256:
        error = ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE", "cached model checksum changed", temporary=True
        )
        status = _defer_or_fail(
            store,
            task.challenge_id,
            task.lease_token,
            task.attempts,
            settings,
            now,
            error,
            resume_state=ChallengeState.FETCHING,
        )
        return AdmissionOutcome(task.challenge_id, status, error.code)

    registration = store.register_validated_model(
        task.challenge_id,
        task.lease_token,
        model_digest=validated.model_digest,
        occurred_at_ms=now,
    )
    if not registration.accepted:
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.PARTICIPANT_FAILED,
            error_code="EXACT_DUPLICATE",
            error_summary="model weights duplicate an earlier submission",
            occurred_at_ms=now,
            lease_token=task.lease_token,
        )
        return AdmissionOutcome(
            task.challenge_id,
            AdmissionStatus.FAILED,
            "EXACT_DUPLICATE",
            registration.duplicate_of,
        )
    return AdmissionOutcome(task.challenge_id, AdmissionStatus.ACCEPTED)


def _defer_or_fail(
    store: ChallengeStore,
    challenge_id: str,
    lease_token: str,
    attempts: int,
    settings: AdmissionSettings,
    now_ms: int,
    error: ModelDownloadError,
    *,
    resume_state: ChallengeState | None = None,
) -> AdmissionStatus:
    if attempts >= settings.max_attempts:
        store.fail_leased_system(
            challenge_id,
            lease_token,
            error_code=error.code,
            error_summary=str(error),
            occurred_at_ms=now_ms,
        )
        return AdmissionStatus.SYSTEM_FAILED
    multiplier = 1 << min(max(attempts - 1, 0), 20)
    delay = min(settings.retry_max_ms, settings.retry_base_ms * multiplier)
    store.defer_leased(
        challenge_id,
        lease_token,
        retry_after_ms=now_ms + delay,
        error_code=error.code,
        error_summary=str(error),
        resume_state=resume_state,
        occurred_at_ms=now_ms,
    )
    return AdmissionStatus.DEFERRED
