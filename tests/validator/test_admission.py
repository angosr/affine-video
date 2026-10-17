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


def submission_at(block: int, hotkey: str, repo_id: str = "owner/model") -> SubmissionRecord:
    position = ChainPosition(block, 0, 0)
    block_hash = "0x" + f"{block:064x}"
    return SubmissionRecord(
        submission_id=submission_id(7, block_hash, position),
        hotkey=hotkey,
        netuid=7,
        block_hash=block_hash,
        position=position,
        payload=b"VDA1payload",
        repo_id=repo_id,
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


def write_model(path, hotkey: str) -> None:
    save_file(
        {"selector.weight": np.ones((2, 2), dtype=np.float32)},
        path,
        metadata={"hotkey": hotkey},
    )


def settings(tmp_path) -> AdmissionSettings:
    return AdmissionSettings(
        cache_dir=tmp_path / "cache",
        max_model_bytes=10_000,
        tensor_schema=SCHEMA,
        publication_grace_blocks=1,
        lease_ms=1_000,
        retry_base_ms=30_000,
        retry_max_ms=120_000,
    )


def test_model_can_be_published_during_frozen_block_grace(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    downloads = 0

    def download_after_publication(*args, **kwargs):
        nonlocal downloads
        downloads += 1
        return model_downloader(model)(*args, **kwargs)

    configured = replace(settings(tmp_path), publication_grace_blocks=2)
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)

        waiting = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=101,
            now_ms=100,
            downloader=download_after_publication,
        )
        assert waiting is None
        assert downloads == 0
        assert store.get_challenge(identifier).state is ChallengeState.QUEUED

        admitted = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=102,
            now_ms=200,
            downloader=download_after_publication,
        )
        assert admitted is not None and admitted.status is AdmissionStatus.ACCEPTED
        assert downloads == 1
        assert store.get_challenge(identifier).state is ChallengeState.SCORING


def test_submission_after_final_evaluation_interval_is_rejected_without_download(tmp_path) -> None:
    configured = replace(settings(tmp_path), last_submission_block=100)

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("closed track submission was downloaded")

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(101, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=unexpected_download,
        )

        assert outcome is not None and outcome.code == "TRACK_CLOSED"
        assert store.get_challenge(identifier).state is ChallengeState.REJECTED_TRACK_CLOSED


def test_admission_accepts_first_digest_and_rejects_later_duplicate(tmp_path) -> None:
    first_model = tmp_path / "first.safetensors"
    second_model = tmp_path / "second.safetensors"
    write_model(first_model, "hk1")
    write_model(second_model, "hk2")

    archived = []
    configured = replace(
        settings(tmp_path),
        archive=lambda path, digest, size: archived.append((path, digest, size)),
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        first = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=model_downloader(first_model),
        )
        assert first is not None and first.status is AdmissionStatus.ACCEPTED
        assert store.get_challenge(first_id).state is ChallengeState.SCORING

        second_id, _ = store.add_submission(
            submission_at(101, "hk2", "other/model"), occurred_at_ms=2
        )
        second = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=200,
            downloader=model_downloader(second_model),
        )

        assert second is not None
        assert second.status is AdmissionStatus.FAILED
        assert second.code == "EXACT_DUPLICATE"
        assert second.duplicate_of == first_id
        duplicate = store.get_challenge(second_id)
        assert duplicate.state is ChallengeState.PARTICIPANT_FAILED
        assert duplicate.error_code == "EXACT_DUPLICATE"
        assert [item[0] for item in archived] == [first_model]


def test_frozen_effective_challenge_limit_is_enforced(tmp_path) -> None:
    first_model = tmp_path / "first.safetensors"
    second_model = tmp_path / "second.safetensors"
    write_model(first_model, "hk1")
    save_file(
        {"selector.weight": np.zeros((2, 2), dtype=np.float32)},
        second_model,
        metadata={"hotkey": "hk2"},
    )
    configured = replace(settings(tmp_path), maximum_effective_challenges=1)

    with ChallengeStore(tmp_path / "validator.db") as store:
        store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        first = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=model_downloader(first_model),
        )
        second_id, _ = store.add_submission(
            submission_at(101, "hk2", "other/model"), occurred_at_ms=2
        )
        second = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=200,
            downloader=model_downloader(second_model),
        )

        assert first is not None and first.status is AdmissionStatus.ACCEPTED
        assert second is not None and second.code == "TRACK_CAPACITY_REACHED"
        assert store.get_challenge(second_id).state is ChallengeState.REJECTED_TRACK_CAPACITY


def test_deferred_earlier_submission_keeps_its_non_refundable_season_slot(tmp_path) -> None:
    configured = replace(settings(tmp_path), maximum_effective_challenges=1)
    downloads = 0

    def unavailable(*_args, **_kwargs):
        nonlocal downloads
        downloads += 1
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        second_id, _ = store.add_submission(
            submission_at(101, "hk2", "other/model"), occurred_at_ms=2
        )
        first = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=unavailable,
        )
        second = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=200,
            downloader=unavailable,
        )

        assert first is not None and first.status is AdmissionStatus.DEFERRED
        assert store.has_season_slot(first_id)
        assert second is not None and second.code == "TRACK_CAPACITY_REACHED"
        assert store.get_challenge(second_id).state is ChallengeState.REJECTED_TRACK_CAPACITY
        assert downloads == 1


def test_same_repository_revision_is_rejected_before_redownload(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    downloads = 0

    def counted_download(*args, **kwargs):
        nonlocal downloads
        downloads += 1
        return model_downloader(model)(*args, **kwargs)

    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        first = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=counted_download,
        )
        assert first is not None and first.status is AdmissionStatus.ACCEPTED

        second_id, _ = store.add_submission(submission_at(101, "hk2"), occurred_at_ms=2)
        second = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=200,
            downloader=counted_download,
        )

        assert downloads == 1
        assert second is not None and second.code == "REFERENCE_DUPLICATE"
        assert second.duplicate_of == first_id
        assert store.get_challenge(second_id).state is ChallengeState.PARTICIPANT_FAILED


def test_same_reference_can_retry_after_earlier_participant_failure(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk2")

    def missing(*_args, **_kwargs):
        raise ModelDownloadError("REPOSITORY_UNAVAILABLE", "repository is private")

    with ChallengeStore(tmp_path / "validator.db") as store:
        store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        failed = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=missing,
        )
        assert failed is not None and failed.status is AdmissionStatus.FAILED

        store.add_submission(submission_at(101, "hk2"), occurred_at_ms=2)
        retried = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=200,
            downloader=model_downloader(model),
        )

        assert retried is not None and retried.status is AdmissionStatus.ACCEPTED


def test_copied_reference_cannot_front_run_model_owner(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "victim")

    with ChallengeStore(tmp_path / "validator.db") as store:
        attacker_id, _ = store.add_submission(submission_at(100, "attacker"), occurred_at_ms=1)
        attacker = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=model_downloader(model),
        )
        assert attacker is not None and attacker.code == "MODEL_OWNER_MISMATCH"
        assert store.get_challenge(attacker_id).state is ChallengeState.PARTICIPANT_FAILED

        victim_id, _ = store.add_submission(submission_at(101, "victim"), occurred_at_ms=2)
        victim = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=200,
            downloader=model_downloader(model),
        )
        assert victim is not None and victim.status is AdmissionStatus.ACCEPTED
        assert store.get_challenge(victim_id).state is ChallengeState.SCORING


def test_temporary_download_failure_is_deferred_with_backoff(tmp_path) -> None:
    def unavailable(*_args, **_kwargs):
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=unavailable,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.DEFERRED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_DEFERRED
        assert (
            store.claim_next_admission(
                "other", lease_ms=1_000, ready_through_block=10_000, now_ms=30_099
            )
            is None
        )
        retry = store.claim_next_admission(
            "other", lease_ms=1_000, ready_through_block=10_000, now_ms=30_100
        )
        assert retry is not None and retry.state is ChallengeState.FETCHING


def test_repeated_download_infrastructure_failure_remains_retryable(tmp_path) -> None:
    def unavailable(*_args, **_kwargs):
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=unavailable,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.DEFERRED
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.SYSTEM_DEFERRED
        assert challenge.score_block is None
        assert store.leaderboard() == []


def test_model_is_durably_archived_before_entering_scoring(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    archived = []
    configured = replace(
        settings(tmp_path),
        archive=lambda path, digest, size: archived.append((path, digest, size)),
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=model_downloader(model),
        )

        assert outcome is not None and outcome.status is AdmissionStatus.ACCEPTED
        assert store.get_challenge(identifier).state is ChallengeState.SCORING
        assert archived == [
            (model, hashlib.sha256(model.read_bytes()).hexdigest(), model.stat().st_size)
        ]


def test_archive_failure_defers_without_consuming_the_submission(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")

    def unavailable(*_args):
        raise RuntimeError("object store unavailable")

    configured = replace(settings(tmp_path), archive=unavailable)
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=model_downloader(model),
        )

        assert outcome is not None and outcome.status is AdmissionStatus.DEFERRED
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.SYSTEM_DEFERRED
        assert challenge.error_code == "MODEL_ARCHIVE_UNAVAILABLE"
        assert store.leaderboard() == []


def test_admission_retry_ceiling_blocks_and_pushes_an_alert(tmp_path) -> None:
    alerts = []

    def unavailable(*_args, **_kwargs):
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    configured = replace(
        settings(tmp_path),
        max_attempts=1,
        alert=lambda event, details: alerts.append((event, details)),
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=unavailable,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.BLOCKED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_BLOCKED
        assert (
            store.claim_next_admission(
                "other", lease_ms=1_000, ready_through_block=10_000, now_ms=1_000_000
            )
            is None
        )
        assert alerts == [
            (
                "ADMISSION_RETRY_EXHAUSTED",
                {
                    "challenge_id": identifier,
                    "attempts": 1,
                    "code": "DOWNLOAD_TEMPORARY_FAILURE",
                },
            )
        ]


def test_permanent_download_failure_records_no_authoritative_score(tmp_path) -> None:
    def missing(*_args, **_kwargs):
        raise ModelDownloadError("MODEL_FILE_MISSING", "selector is missing")

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=missing,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.FAILED
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.PARTICIPANT_FAILED
        assert challenge.error_code == "MODEL_FILE_MISSING"
        assert challenge.final_score is None
        assert challenge.score_block == 10_000
        assert store.leaderboard() == []
