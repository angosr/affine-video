"""One recoverable admission task: download, validate, and deduplicate."""

from __future__ import annotations

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
from validator.retries import (
    RetryRecorder,
    RetryRestore,
    persist_retry_failure,
    prepare_durable_retry,
    retry_clock,
)
from validator.store import ChallengeStore, LeasedChallenge


class AdmissionStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"
    NOT_ADMITTED = "NOT_ADMITTED"


@dataclass(frozen=True, slots=True)
class RejectedAdmission:
    code: str
    summary: str
    score_block: int
    file_sha256: str | None = None
    model_size_bytes: int | None = None
    duplicate_of: str | None = None


@dataclass(frozen=True, slots=True)
class AdmissionSettings:
    cache_dir: Path
    max_model_bytes: int
    tensor_schema: Mapping[str, TensorSpec]
    lease_ms: int = 5 * 60 * 1000
    retry_base_ms: int = 30 * 1000
    retry_max_ms: int = 30 * 60 * 1000
    max_attempts: int = 5
    alert: Callable[[str, Mapping[str, object]], None] | None = None
    archive: Callable[[object, str | Path, ValidatedModel], None] | None = None
    archive_rejection: Callable[[object, RejectedAdmission], None] | None = None
    restore: Callable[[object], DownloadedModel | RejectedAdmission | None] | None = None
    restore_retry: RetryRestore | None = None
    record_retry: RetryRecorder | None = None
    eligible: Callable[[object], bool] | None = None
    private_download: Callable[..., DownloadedModel] | None = None

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
        if self.retry_base_ms > self.retry_max_ms:
            raise ValueError("retry base cannot exceed retry maximum")
        if (self.restore_retry is None) != (self.record_retry is None):
            raise ValueError("durable admission retry callbacks must be configured together")
        if self.archive_rejection is not None and self.restore is None:
            raise ValueError("archived admission rejections require a restore callback")


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
    clock = retry_clock(now_ms)
    now = clock()
    task = store.claim_next_admission(
        worker_id,
        lease_ms=settings.lease_ms,
        ready_through_block=current_finalized_block,
        now_ms=now,
    )
    if task is None:
        return None

    challenge_id = task.challenge_id
    task, restored_retry = prepare_durable_retry(
        store,
        task,
        stage="ADMISSION",
        clock=clock,
        restore=settings.restore_retry,
        wait_ms=settings.retry_base_ms,
    )
    if task is None:
        assert restored_retry is not None
        return AdmissionOutcome(
            challenge_id,
            AdmissionStatus.BLOCKED if restored_retry.blocked else AdmissionStatus.DEFERRED,
            restored_retry.error_code,
        )

    downloaded = None
    if settings.restore is not None:
        try:
            downloaded = settings.restore(task)
        except Exception:
            error = ModelDownloadError(
                "MODEL_ARCHIVE_UNAVAILABLE",
                "archived admission could not be restored",
                temporary=True,
            )
            status = _defer_or_fail(store, task, settings, clock(), error)
            return AdmissionOutcome(task.challenge_id, status, error.code)
    if isinstance(downloaded, RejectedAdmission):
        return _reject_participant(
            store, task, settings, downloaded,
            current_finalized_block=current_finalized_block, clock=clock,
        )
    prior = store.earlier_hotkey_owner(task.challenge_id)
    if prior is not None:
        return _reject_participant(
            store,
            task,
            settings,
            RejectedAdmission(
                "DUPLICATE_HOTKEY",
                "the earlier submission consumed this opportunity",
                current_finalized_block,
                duplicate_of=prior,
            ),
            current_finalized_block=current_finalized_block,
            clock=clock,
        )
    if (downloaded is None and task.state is ChallengeState.FETCHING
            and settings.eligible is not None):
        try:
            eligible = settings.eligible(task)
        except Exception:
            error = ModelDownloadError(
                "INVENTORY_AUTHORITY_UNAVAILABLE", "submission-time pool cannot be verified",
                temporary=True,
            )
            status = _defer_or_fail(store, task, settings, clock(), error)
            return AdmissionOutcome(task.challenge_id, status, error.code)
        if not eligible:
            return _reject_participant(
                store, task, settings,
                RejectedAdmission(
                    "NO_PRE_SUBMISSION_DATA", "system did not admit this submission; "
                    "no eligible pre-submission data; opportunity retained, resubmission allowed",
                    current_finalized_block,
                ),
                current_finalized_block=current_finalized_block, clock=clock,
            )
    reference_owner = store.earlier_reference_owner(task.challenge_id)
    if reference_owner is not None:
        return _reject_participant(
            store, task, settings,
            _observed_rejection(
                store, task, "REFERENCE_DUPLICATE",
                "repository revision duplicates an earlier active submission",
                current_finalized_block, duplicate_of=reference_owner,
            ),
            current_finalized_block=current_finalized_block, clock=clock,
        )
    model_path = Path(task.model_path) if task.model_path is not None else None
    expected_file_sha256 = task.file_sha256
    if task.state is ChallengeState.FETCHING:
        try:
            if downloaded is None:
                if task.private_submission is not None:
                    if settings.private_download is None:
                        raise ModelDownloadError(
                            "PRIVATE_UPLOAD_UNAVAILABLE", "private source is not configured",
                            temporary=True,
                        )
                    downloaded = settings.private_download(
                        task.private_submission, hotkey=task.hotkey,
                        cache_dir=settings.cache_dir, max_bytes=settings.max_model_bytes,
                        submission_block=task.position.block,
                    )
                else:
                    downloaded = downloader(
                        task.repo_id, task.revision,
                        cache_dir=settings.cache_dir, max_bytes=settings.max_model_bytes,
                    )
            if (task.private_submission is not None
                    and downloaded.file_sha256 != task.private_submission.file_sha256):
                raise ModelDownloadError(
                    "PRIVATE_UPLOAD_HASH_MISMATCH", "private source differs from its chain binding",
                    temporary=True,
                )
        except ModelDownloadError as exc:
            if exc.temporary:
                status = _defer_or_fail(
                    store,
                    task,
                    settings,
                    clock(),
                    exc,
                )
                return AdmissionOutcome(task.challenge_id, status, exc.code)
            return _reject_participant(
                store, task, settings,
                _observed_rejection(store, task, exc.code, str(exc), current_finalized_block),
                current_finalized_block=current_finalized_block, clock=clock,
            )
        store.record_downloaded_model(
            task.challenge_id,
            task.lease_token,
            model_path=downloaded.path,
            file_sha256=downloaded.file_sha256,
            size_bytes=downloaded.size_bytes,
            occurred_at_ms=clock(),
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
            task,
            settings,
            clock(),
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
            now_ms=clock(),
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
            now_ms=clock(),
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
                now_ms=clock(),
                summary="cached model changed during validation",
            )
        return _reject_participant(
            store, task, settings,
            _observed_rejection(store, task, exc.code, str(exc), current_finalized_block),
            current_finalized_block=current_finalized_block, clock=clock,
        )

    if expected_file_sha256 is not None and validated.file_sha256 != expected_file_sha256:
        return _defer_cached_model(
            store,
            task,
            settings,
            now_ms=clock(),
            summary="cached model changed during validation",
        )

    duplicate_owner = store.model_owner(validated.model_digest)
    if duplicate_owner is not None:
        return _reject_exact_duplicate(
            store,
            task,
            settings,
            duplicate_owner,
            current_finalized_block=current_finalized_block,
            clock=clock,
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
                task,
                settings,
                clock(),
                error,
            )
            return AdmissionOutcome(task.challenge_id, status, error.code)

    registration = store.register_validated_model(
        task.challenge_id,
        task.lease_token,
        model_digest=validated.model_digest,
        occurred_at_ms=clock(),
    )
    if not registration.accepted:
        return _reject_exact_duplicate(
            store,
            task,
            settings,
            registration.duplicate_of,
            current_finalized_block=current_finalized_block,
            clock=clock,
        )
    return AdmissionOutcome(task.challenge_id, AdmissionStatus.ACCEPTED)


def _observed_rejection(
    store: ChallengeStore, task: LeasedChallenge, code: str, summary: str, score_block: int,
    *, duplicate_of: str | None = None,
) -> RejectedAdmission:
    row = store.public_challenge(task.challenge_id)
    return RejectedAdmission(
        code, summary, score_block, row["file_sha256"], row["model_size_bytes"], duplicate_of,
    )


def _reject_participant(
    store: ChallengeStore,
    task: LeasedChallenge,
    settings: AdmissionSettings,
    rejection: RejectedAdmission,
    *,
    current_finalized_block: int,
    clock: Callable[[], int],
) -> AdmissionOutcome:
    if rejection.score_block > current_finalized_block:
        error = ModelDownloadError(
            "ADMISSION_ARCHIVE_AHEAD", "archived admission is ahead of the finalized head",
            temporary=True,
        )
    else:
        try:
            if settings.archive_rejection is not None:
                settings.archive_rejection(task, rejection)
        except Exception:
            error = ModelDownloadError(
                "ADMISSION_ARCHIVE_UNAVAILABLE", "admission rejection could not be persisted",
                temporary=True,
            )
        else:
            not_admitted = rejection.code == "NO_PRE_SUBMISSION_DATA"
            target = (
                ChallengeState.SYSTEM_NOT_ADMITTED if not_admitted
                else ChallengeState.REJECTED_DUPLICATE_HOTKEY
                if rejection.code == "DUPLICATE_HOTKEY"
                else ChallengeState.PARTICIPANT_FAILED
            )
            store.complete_challenge(
                task.challenge_id, target,
                error_code=rejection.code, error_summary=rejection.summary,
                occurred_at_ms=clock(), lease_token=task.lease_token,
                score_block=rejection.score_block,
                file_sha256=rejection.file_sha256,
                model_size_bytes=rejection.model_size_bytes,
            )
            return AdmissionOutcome(
                task.challenge_id,
                AdmissionStatus.NOT_ADMITTED if not_admitted else AdmissionStatus.FAILED,
                rejection.code, rejection.duplicate_of,
            )
    status = _defer_or_fail(store, task, settings, clock(), error)
    return AdmissionOutcome(task.challenge_id, status, error.code)


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
        task,
        settings,
        now_ms,
        error,
        resume_state=ChallengeState.FETCHING,
    )
    return AdmissionOutcome(task.challenge_id, status, error.code)


def _reject_exact_duplicate(
    store: ChallengeStore,
    task: LeasedChallenge,
    settings: AdmissionSettings,
    duplicate_of: str | None,
    *,
    current_finalized_block: int,
    clock: Callable[[], int],
) -> AdmissionOutcome:
    if duplicate_of is None:
        raise RuntimeError("duplicate model is missing its earlier owner")
    return _reject_participant(
        store, task, settings,
        _observed_rejection(
            store, task, "EXACT_DUPLICATE", "model weights duplicate an earlier submission",
            current_finalized_block, duplicate_of=duplicate_of,
        ),
        current_finalized_block=current_finalized_block, clock=clock,
    )


def _defer_or_fail(
    store: ChallengeStore,
    task: LeasedChallenge,
    settings: AdmissionSettings,
    now_ms: int,
    error: ModelDownloadError,
    *,
    resume_state: ChallengeState | None = None,
) -> AdmissionStatus:
    attempts = task.attempts
    multiplier = 1 << min(max(attempts - 1, 0), 20)
    delay = min(settings.retry_max_ms, settings.retry_base_ms * multiplier)
    if attempts >= settings.max_attempts:
        delay = settings.retry_max_ms
    checkpoint = persist_retry_failure(
        task,
        stage="ADMISSION",
        code=error.code,
        summary=str(error),
        retry_after_ms=now_ms + delay,
        max_attempts=settings.max_attempts,
        record=settings.record_retry,
    )
    if checkpoint is not None:
        attempts = checkpoint.attempts
    if attempts == 1 and attempts < settings.max_attempts and settings.alert is not None:
        settings.alert(
            "ADMISSION_RETRY_STARTED",
            {"challenge_id": task.challenge_id, "attempts": attempts, "code": error.code},
        )
    blocked = checkpoint.blocked if checkpoint is not None else attempts >= settings.max_attempts
    retry_after_ms = now_ms + delay if checkpoint is None else checkpoint.retry_after_ms
    if blocked:
        store.block_leased(
            task.challenge_id,
            task.lease_token,
            error_code=error.code,
            error_summary=str(error),
            retry_after_ms=retry_after_ms,
            occurred_at_ms=now_ms,
        )
        if settings.alert is not None:
            settings.alert(
                "ADMISSION_RETRY_EXHAUSTED",
                {"challenge_id": task.challenge_id, "attempts": attempts, "code": error.code,
                 "retry_after_ms": retry_after_ms},
            )
        return AdmissionStatus.BLOCKED
    store.defer_leased(
        task.challenge_id,
        task.lease_token,
        retry_after_ms=retry_after_ms,
        error_code=error.code,
        error_summary=str(error),
        resume_state=resume_state,
        occurred_at_ms=now_ms,
    )
    return AdmissionStatus.DEFERRED
