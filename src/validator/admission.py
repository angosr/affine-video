"""One recoverable admission task: download, validate, and deduplicate."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scoring.model import (
    ModelValidationError,
    TensorSpec,
    ValidatedModel,
    model_file_sha256,
    validate_model,
)
from validator.download import (
    DownloadedModel,
    ModelDownloadError,
    download_public_model,
)
from validator.queue import ChallengeState
from validator.store import ChallengeStore, LeasedChallenge


class AdmissionStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class AdmissionSettings:
    cache_dir: Path
    max_model_bytes: int
    tensor_schema: Mapping[str, TensorSpec]
    publication_grace_blocks: int
    maximum_effective_challenges: int | None = None
    last_submission_block: int | None = None
    lease_ms: int = 5 * 60 * 1000
    retry_base_ms: int = 30 * 1000
    retry_max_ms: int = 30 * 60 * 1000
    max_attempts: int = 5
    alert: Callable[[str, Mapping[str, object]], None] | None = None
    archive: Callable[[object, str | Path, ValidatedModel], None] | None = None
    restore: Callable[[object], DownloadedModel | None] | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.max_model_bytes,
                self.lease_ms,
                self.retry_base_ms,
                self.retry_max_ms,
                self.max_attempts,
            )
            <= 0
        ):
            raise ValueError("admission limits must be positive")
        if self.maximum_effective_challenges is not None and self.maximum_effective_challenges <= 0:
            raise ValueError("maximum effective challenges must be positive")
        if self.last_submission_block is not None and self.last_submission_block < 0:
            raise ValueError("last submission block must be non-negative")
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")
        if self.publication_grace_blocks <= 0:
            raise ValueError("publication grace blocks must be positive")


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
    current_finalized_block: int,
    now_ms: int | None = None,
    downloader: Callable[..., DownloadedModel] = download_public_model,
    model_validator: Callable[..., ValidatedModel] = validate_model,
) -> AdmissionOutcome | None:
    """Process at most one FIFO admission task."""

    if current_finalized_block < 0:
        raise ValueError("current finalized block must be non-negative")
    now = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    task = store.claim_next_admission(
        worker_id,
        lease_ms=settings.lease_ms,
        ready_through_block=current_finalized_block - settings.publication_grace_blocks,
        now_ms=now,
    )
    if task is None:
        return None

    if (
        settings.last_submission_block is not None
        and task.position.block > settings.last_submission_block
    ):
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.REJECTED_TRACK_CLOSED,
            error_code="TRACK_CLOSED",
            error_summary="submission is later than the final frozen evaluation interval",
            occurred_at_ms=now,
            lease_token=task.lease_token,
            score_block=current_finalized_block,
        )
        return AdmissionOutcome(task.challenge_id, AdmissionStatus.FAILED, "TRACK_CLOSED")

    if settings.maximum_effective_challenges is not None:
        store.reserve_season_slots(settings.maximum_effective_challenges)
        if not store.has_season_slot(task.challenge_id):
            return _reject_track_capacity(
                store,
                task.challenge_id,
                task.lease_token,
                current_finalized_block=current_finalized_block,
                now_ms=now,
            )

    reference_owner = store.earlier_reference_owner(task.challenge_id)
    if reference_owner is not None:
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.PARTICIPANT_FAILED,
            error_code="REFERENCE_DUPLICATE",
            error_summary="repository revision duplicates an earlier active submission",
            occurred_at_ms=now,
            lease_token=task.lease_token,
            score_block=current_finalized_block,
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
        downloaded = None
        if settings.restore is not None:
            try:
                downloaded = settings.restore(task)
            except Exception:
                error = ModelDownloadError(
                    "MODEL_ARCHIVE_UNAVAILABLE",
                    "archived admission model could not be restored",
                    temporary=True,
                )
                status = _defer_or_fail(
                    store,
                    task.challenge_id,
                    task.lease_token,
                    task.attempts,
                    settings,
                    now,
                    error,
                )
                return AdmissionOutcome(task.challenge_id, status, error.code)
        try:
            if downloaded is None:
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
                score_block=current_finalized_block,
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

    cache_error = None
    if expected_file_sha256 is None:
        cache_error = "cached model identity is unavailable"
    else:
        try:
            if model_file_sha256(model_path) != expected_file_sha256:
                cache_error = "cached model checksum changed"
        except OSError:
            cache_error = "cached model cannot be read"
    if cache_error is not None:
        return _defer_cached_model(
            store,
            task,
            settings,
            now_ms=now,
            summary=cache_error,
        )

    try:
        validated = model_validator(
            model_path,
            schema=dict(settings.tensor_schema),
            max_bytes=settings.max_model_bytes,
            expected_hotkey=task.hotkey,
        )
    except OSError:
        return _defer_cached_model(
            store,
            task,
            settings,
            now_ms=now,
            summary="cached model cannot be read",
        )
    except ModelValidationError as exc:
        try:
            identity_still_matches = (
                expected_file_sha256 is not None
                and model_file_sha256(model_path) == expected_file_sha256
            )
        except OSError:
            identity_still_matches = False
        if not identity_still_matches:
            return _defer_cached_model(
                store,
                task,
                settings,
                now_ms=now,
                summary="cached model changed during validation",
            )
        store.complete_challenge(
            task.challenge_id,
            ChallengeState.PARTICIPANT_FAILED,
            error_code=exc.code,
            error_summary=str(exc),
            occurred_at_ms=now,
            lease_token=task.lease_token,
            score_block=current_finalized_block,
        )
        return AdmissionOutcome(task.challenge_id, AdmissionStatus.FAILED, exc.code)

    if expected_file_sha256 is not None and validated.file_sha256 != expected_file_sha256:
        return _defer_cached_model(
            store,
            task,
            settings,
            now_ms=now,
            summary="cached model changed during validation",
        )

    duplicate_owner = store.model_owner(validated.model_digest)
    if duplicate_owner is not None:
        return _reject_exact_duplicate(
            store,
            task.challenge_id,
            task.lease_token,
            duplicate_owner,
            current_finalized_block=current_finalized_block,
            now_ms=now,
        )

    if settings.archive is not None:
        try:
            settings.archive(task, model_path, validated)
        except Exception:
            error = ModelDownloadError(
                "MODEL_ARCHIVE_UNAVAILABLE",
                "validated model could not be persisted outside the evaluator host",
                temporary=True,
            )
            status = _defer_or_fail(
                store,
                task.challenge_id,
                task.lease_token,
                task.attempts,
                settings,
                now,
                error,
            )
            return AdmissionOutcome(task.challenge_id, status, error.code)

    registration = store.register_validated_model(
        task.challenge_id,
        task.lease_token,
        model_digest=validated.model_digest,
        occurred_at_ms=now,
    )
    if not registration.accepted:
        return _reject_exact_duplicate(
            store,
            task.challenge_id,
            task.lease_token,
            registration.duplicate_of,
            current_finalized_block=current_finalized_block,
            now_ms=now,
        )
    return AdmissionOutcome(task.challenge_id, AdmissionStatus.ACCEPTED)


def _defer_cached_model(
    store: ChallengeStore,
    task: LeasedChallenge,
    settings: AdmissionSettings,
    *,
    now_ms: int,
    summary: str,
) -> AdmissionOutcome:
    error = ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", summary, temporary=True)
    status = _defer_or_fail(
        store,
        task.challenge_id,
        task.lease_token,
        task.attempts,
        settings,
        now_ms,
        error,
        resume_state=ChallengeState.FETCHING,
    )
    return AdmissionOutcome(task.challenge_id, status, error.code)


def _reject_exact_duplicate(
    store: ChallengeStore,
    challenge_id: str,
    lease_token: str,
    duplicate_of: str | None,
    *,
    current_finalized_block: int,
    now_ms: int,
) -> AdmissionOutcome:
    if duplicate_of is None:
        raise RuntimeError("duplicate model is missing its earlier owner")
    store.complete_challenge(
        challenge_id,
        ChallengeState.PARTICIPANT_FAILED,
        error_code="EXACT_DUPLICATE",
        error_summary="model weights duplicate an earlier submission",
        occurred_at_ms=now_ms,
        lease_token=lease_token,
        score_block=current_finalized_block,
    )
    return AdmissionOutcome(
        challenge_id,
        AdmissionStatus.FAILED,
        "EXACT_DUPLICATE",
        duplicate_of,
    )


def _reject_track_capacity(
    store: ChallengeStore,
    challenge_id: str,
    lease_token: str,
    *,
    current_finalized_block: int,
    now_ms: int,
) -> AdmissionOutcome:
    store.complete_challenge(
        challenge_id,
        ChallengeState.REJECTED_TRACK_CAPACITY,
        error_code="TRACK_CAPACITY_REACHED",
        error_summary="the frozen track has no remaining effective challenge slots",
        occurred_at_ms=now_ms,
        lease_token=lease_token,
        score_block=current_finalized_block,
    )
    return AdmissionOutcome(
        challenge_id,
        AdmissionStatus.FAILED,
        "TRACK_CAPACITY_REACHED",
    )


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
    if attempts == 1 and attempts < settings.max_attempts and settings.alert is not None:
        settings.alert(
            "ADMISSION_RETRY_STARTED",
            {"challenge_id": challenge_id, "attempts": attempts, "code": error.code},
        )
    if attempts >= settings.max_attempts:
        store.block_leased(
            challenge_id,
            lease_token,
            error_code=error.code,
            error_summary=str(error),
            occurred_at_ms=now_ms,
        )
        if settings.alert is not None:
            settings.alert(
                "ADMISSION_RETRY_EXHAUSTED",
                {"challenge_id": challenge_id, "attempts": attempts, "code": error.code},
            )
        return AdmissionStatus.BLOCKED
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
