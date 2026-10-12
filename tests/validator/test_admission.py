import hashlib
from dataclasses import replace

import numpy as np
from safetensors.numpy import save_file

from scoring.model import TensorSpec
from validator.admission import (
    AdmissionSettings,
    AdmissionStatus,
    process_next_admission,
)
from validator.download import DownloadedModel, ModelDownloadError
from validator.queue import ChainPosition, ChallengeState, submission_id
from validator.store import ChallengeStore, SubmissionRecord

SCHEMA = {"selector.weight": TensorSpec((2, 2), "float32")}


def submission_at(block: int, hotkey: str) -> SubmissionRecord:
    position = ChainPosition(block, 0, 0)
    block_hash = "0x" + f"{block:064x}"
    return SubmissionRecord(
        submission_id=submission_id(7, block_hash, position),
        hotkey=hotkey,
        netuid=7,
        block_hash=block_hash,
        position=position,
        payload=b"VDA1payload",
        repo_id="owner/model",
        revision="12" * 20,
    )


def model_downloader(path):
    content = path.read_bytes()
    downloaded = DownloadedModel(
        path=path,
        file_sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        etag='"etag"',
        download_host="cdn.example",
    )

    def download(*_args, **_kwargs):
        return downloaded

    return download


def settings(tmp_path) -> AdmissionSettings:
    return AdmissionSettings(
        cache_dir=tmp_path / "cache",
        max_model_bytes=10_000,
        tensor_schema=SCHEMA,
        lease_ms=1_000,
        retry_base_ms=30_000,
        retry_max_ms=120_000,
    )


def test_admission_accepts_first_digest_and_rejects_later_duplicate(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    save_file({"selector.weight": np.ones((2, 2), dtype=np.float32)}, model)

    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        first = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            now_ms=100,
            downloader=model_downloader(model),
        )
        assert first is not None and first.status is AdmissionStatus.ACCEPTED
        assert store.get_challenge(first_id).state is ChallengeState.SCREENING

        second_id, _ = store.add_submission(submission_at(101, "hk2"), occurred_at_ms=2)
        second = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            now_ms=200,
            downloader=model_downloader(model),
        )

        assert second is not None
        assert second.status is AdmissionStatus.FAILED
        assert second.code == "EXACT_DUPLICATE"
        assert second.duplicate_of == first_id
        duplicate = store.get_challenge(second_id)
        assert duplicate.state is ChallengeState.PARTICIPANT_FAILED
        assert duplicate.error_code == "EXACT_DUPLICATE"


def test_temporary_download_failure_is_deferred_with_backoff(tmp_path) -> None:
    def unavailable(*_args, **_kwargs):
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            now_ms=100,
            downloader=unavailable,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.DEFERRED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_DEFERRED
        assert (
            store.claim_next_admission("other", lease_ms=1_000, now_ms=30_099)
            is None
        )
        retry = store.claim_next_admission("other", lease_ms=1_000, now_ms=30_100)
        assert retry is not None and retry.state is ChallengeState.FETCHING


def test_repeated_download_infrastructure_failure_becomes_system_failed(tmp_path) -> None:
    def unavailable(*_args, **_kwargs):
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    configured = replace(settings(tmp_path), max_attempts=1)
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            now_ms=100,
            downloader=unavailable,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.SYSTEM_FAILED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_FAILED
        assert store.leaderboard() == []


def test_permanent_download_failure_records_zero_score(tmp_path) -> None:
    def missing(*_args, **_kwargs):
        raise ModelDownloadError("MODEL_FILE_MISSING", "selector is missing")

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            now_ms=100,
            downloader=missing,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.FAILED
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.PARTICIPANT_FAILED
        assert challenge.error_code == "MODEL_FILE_MISSING"
        assert store.leaderboard()[0].final_score == 0
