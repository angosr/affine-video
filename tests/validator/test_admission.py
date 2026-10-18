import errno
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from scoring.model import TensorSpec
from validator import download as download_module
from validator.admission import (
    AdmissionSettings,
    AdmissionStatus,
    process_next_admission,
)
from validator.download import DownloadedModel, ModelDownloadError
from validator.production.retries import PrivateRetryStore
from validator.production.storage import PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState, submission_id
from validator.store import ChallengeStore, SubmissionRecord

from .production.test_storage import MemoryS3

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
        archive=lambda task, path, validated: archived.append(
            (
                task.challenge_id,
                path,
                validated.file_sha256,
                validated.size_bytes,
            )
        ),
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
        assert [(item[0], item[1]) for item in archived] == [(first_id, first_model)]


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
    second_downloads = 0

    def count_second_download(*args, **kwargs):
        nonlocal second_downloads
        second_downloads += 1
        return model_downloader(second_model)(*args, **kwargs)

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
            downloader=count_second_download,
        )

        assert first is not None and first.status is AdmissionStatus.ACCEPTED
        assert second is not None and second.code == "TRACK_CAPACITY_REACHED"
        assert store.get_challenge(second_id).state is ChallengeState.REJECTED_TRACK_CAPACITY
        assert second_downloads == 0


def test_deferred_earlier_submission_blocks_fifo_while_retaining_its_slot(tmp_path) -> None:
    configured = replace(settings(tmp_path), maximum_effective_challenges=1)
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    downloads = 0

    def initially_unavailable(*args, **kwargs):
        nonlocal downloads
        downloads += 1
        if downloads == 1:
            raise ModelDownloadError(
                "DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True
            )
        return model_downloader(model)(*args, **kwargs)

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
            downloader=initially_unavailable,
        )
        blocked_by_fifo = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=200,
            downloader=initially_unavailable,
        )

        assert first is not None and first.status is AdmissionStatus.DEFERRED
        assert store.has_season_slot(first_id)
        assert not store.admission_interval_complete(100, 100)
        assert blocked_by_fifo is None

        retried = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=30_100,
            downloader=initially_unavailable,
        )
        assert retried is not None and retried.status is AdmissionStatus.ACCEPTED
        assert store.has_season_slot(first_id)
        assert store.admission_interval_complete(100, 100)

        second = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=30_101,
            downloader=initially_unavailable,
        )
        assert second is not None and second.code == "TRACK_CAPACITY_REACHED"
        assert store.get_challenge(second_id).state is ChallengeState.REJECTED_TRACK_CAPACITY
        assert downloads == 2


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


def test_many_invalid_admissions_release_capacity_but_consume_each_hotkey(tmp_path) -> None:
    def missing(*_args, **_kwargs):
        raise ModelDownloadError("REPOSITORY_UNAVAILABLE", "repository is private")

    configured = replace(settings(tmp_path), maximum_effective_challenges=20)
    with ChallengeStore(tmp_path / "validator.db") as store:
        for index in range(20):
            identifier, _ = store.add_submission(
                submission_at(100 + index, f"invalid-{index}"), occurred_at_ms=index + 1
            )
            failed = process_next_admission(
                store, worker_id="admission", settings=configured,
                current_finalized_block=10_000, now_ms=100 + index, downloader=missing,
            )
            assert failed is not None and failed.status is AdmissionStatus.FAILED
            assert not store.has_season_slot(identifier)
            assert store.effective_challenge_count() == 0

        duplicate, _ = store.add_submission(
            submission_at(120, "invalid-0"), occurred_at_ms=200
        )
        assert store.get_challenge(duplicate).state is ChallengeState.REJECTED_DUPLICATE_HOTKEY
        model = tmp_path / "valid.safetensors"
        write_model(model, "valid")
        valid_id, _ = store.add_submission(submission_at(121, "valid"), occurred_at_ms=201)
        accepted = process_next_admission(
            store, worker_id="admission", settings=configured,
            current_finalized_block=10_000, now_ms=202, downloader=model_downloader(model),
        )
        assert accepted is not None and accepted.status is AdmissionStatus.ACCEPTED
        assert store.has_season_slot(valid_id)
        assert store.effective_challenge_count() == 1


def test_invalid_model_releases_the_only_effective_challenge_slot(tmp_path) -> None:
    invalid_model = tmp_path / "invalid.safetensors"
    valid_model = tmp_path / "valid.safetensors"
    save_file(
        {"unexpected.weight": np.ones((2, 2), dtype=np.float32)},
        invalid_model,
        metadata={"hotkey": "invalid"},
    )
    write_model(valid_model, "valid")
    configured = replace(settings(tmp_path), maximum_effective_challenges=1)

    with ChallengeStore(tmp_path / "validator.db") as store:
        invalid_id, _ = store.add_submission(
            submission_at(100, "invalid", "invalid/model"), occurred_at_ms=1
        )
        invalid = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=model_downloader(invalid_model),
        )

        assert invalid is not None and invalid.status is AdmissionStatus.FAILED
        assert store.get_challenge(invalid_id).error_code == "MODEL_SCHEMA_INVALID"
        assert store.effective_challenge_count() == 0

        valid_id, _ = store.add_submission(
            submission_at(101, "valid", "valid/model"), occurred_at_ms=2
        )
        valid = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=200,
            downloader=model_downloader(valid_model),
        )

        assert valid is not None and valid.status is AdmissionStatus.ACCEPTED
        assert store.effective_challenge_count() == 1
        assert store.has_season_slot(valid_id)


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


def test_temporary_failure_cannot_invert_identical_model_ownership(tmp_path) -> None:
    early_model = tmp_path / "early.safetensors"
    later_model = tmp_path / "later.safetensors"
    write_model(early_model, "early")
    write_model(later_model, "later")
    downloads = {
        "early/model": model_downloader(early_model),
        "later/model": model_downloader(later_model),
    }
    early_attempts = 0

    def temporarily_unavailable(repo_id, *args, **kwargs):
        nonlocal early_attempts
        if repo_id == "early/model" and early_attempts == 0:
            early_attempts += 1
            raise ModelDownloadError(
                "DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True
            )
        return downloads[repo_id](repo_id, *args, **kwargs)

    with ChallengeStore(tmp_path / "validator.db") as store:
        early_id, _ = store.add_submission(
            submission_at(100, "early", "early/model"), occurred_at_ms=1
        )
        later_id, _ = store.add_submission(
            submission_at(101, "later", "later/model"), occurred_at_ms=2
        )

        deferred = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=100,
            downloader=temporarily_unavailable,
        )
        assert deferred is not None and deferred.status is AdmissionStatus.DEFERRED

        assert (
            process_next_admission(
                store,
                worker_id="admission",
                settings=settings(tmp_path),
                current_finalized_block=10_000,
                now_ms=101,
                downloader=temporarily_unavailable,
            )
            is None
        )

        accepted = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=30_100,
            downloader=temporarily_unavailable,
        )
        assert accepted is not None and accepted.status is AdmissionStatus.ACCEPTED
        assert accepted.challenge_id == early_id

        duplicate = process_next_admission(
            store,
            worker_id="admission",
            settings=settings(tmp_path),
            current_finalized_block=10_000,
            now_ms=30_101,
            downloader=temporarily_unavailable,
        )
        assert duplicate is not None and duplicate.code == "EXACT_DUPLICATE"
        assert duplicate.challenge_id == later_id
        assert duplicate.duplicate_of == early_id


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
        archive=lambda task, path, validated: archived.append(
            (
                task.challenge_id,
                path,
                validated.file_sha256,
                validated.size_bytes,
            )
        ),
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
            (
                identifier,
                model,
                hashlib.sha256(model.read_bytes()).hexdigest(),
                model.stat().st_size,
            )
        ]


def test_archived_admission_is_restored_without_repository_access(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    restored = model_downloader(model)()
    restore_calls = []

    def restore(task):
        restore_calls.append(task.challenge_id)
        return restored

    def repository_must_not_be_used(*_args, **_kwargs):
        raise AssertionError("repository access is not allowed during archive recovery")

    configured = replace(settings(tmp_path), restore=restore)
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        outcome = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=repository_must_not_be_used,
        )

        assert outcome is not None and outcome.status is AdmissionStatus.ACCEPTED
        assert restore_calls == [identifier]


def test_archive_failure_defers_without_consuming_the_submission(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    alerts = []

    def unavailable(*_args):
        raise RuntimeError("object store unavailable")

    configured = replace(
        settings(tmp_path),
        archive=unavailable,
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
            downloader=model_downloader(model),
        )

        assert outcome is not None and outcome.status is AdmissionStatus.DEFERRED
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.SYSTEM_DEFERRED
        assert challenge.error_code == "MODEL_ARCHIVE_UNAVAILABLE"
        assert store.has_season_slot(identifier)
        assert store.effective_challenge_count() == 1
        assert store.leaderboard() == []
        assert alerts == [
            (
                "ADMISSION_RETRY_STARTED",
                {
                    "challenge_id": identifier,
                    "attempts": 1,
                    "code": "MODEL_ARCHIVE_UNAVAILABLE",
                },
            )
        ]


def test_corrupted_cached_model_is_recovered_instead_of_blaming_participant(tmp_path) -> None:
    model = tmp_path / "selector.safetensors"
    write_model(model, "hk1")
    download = model_downloader(model)

    def unavailable_archive(*_args):
        raise RuntimeError("object store unavailable")

    def repository_must_not_be_used(*_args, **_kwargs):
        raise AssertionError("validation retry must detect corruption before repository access")

    configured = replace(settings(tmp_path), archive=unavailable_archive)
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100, "hk1"), occurred_at_ms=1)
        first = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=100,
            downloader=download,
        )
        assert first is not None and first.code == "MODEL_ARCHIVE_UNAVAILABLE"

        model.write_bytes(b"damaged local cache")
        second = process_next_admission(
            store,
            worker_id="admission",
            settings=configured,
            current_finalized_block=10_000,
            now_ms=30_100,
            downloader=repository_must_not_be_used,
        )

        assert second is not None
        assert second.status is AdmissionStatus.DEFERRED
        assert second.code == "DOWNLOAD_TEMPORARY_FAILURE"
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.SYSTEM_DEFERRED
        assert challenge.error_code == "DOWNLOAD_TEMPORARY_FAILURE"


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


def test_admission_retry_ceiling_survives_total_local_state_loss(tmp_path) -> None:
    retries = PrivateRetryStore(
        PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc"),
        track_spec_digest="ab" * 32,
    )
    calls = 0

    def unavailable(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "temporary outage", temporary=True)

    configured = replace(
        settings(tmp_path),
        max_attempts=2,
        restore_retry=retries.restore,
        record_retry=retries.record_failure,
    )
    submission = submission_at(100, "hk1")
    outcomes = []
    for index, now in enumerate((100, 30_100, 1_000_000), start=1):
        with ChallengeStore(tmp_path / f"validator-{index}.db") as store:
            identifier, _ = store.add_submission(submission, occurred_at_ms=1)
            outcomes.append(
                process_next_admission(
                    store,
                    worker_id="admission",
                    settings=configured,
                    current_finalized_block=10_000,
                    now_ms=now,
                    downloader=unavailable,
                )
            )
            if index >= 2:
                assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_BLOCKED

    assert [outcome.status for outcome in outcomes if outcome is not None] == [
        AdmissionStatus.DEFERRED,
        AdmissionStatus.BLOCKED,
        AdmissionStatus.BLOCKED,
    ]
    assert calls == 2


def test_disk_full_during_download_preparation_uses_durable_retry_ceiling(
    monkeypatch, tmp_path
) -> None:
    retries = PrivateRetryStore(
        PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc"),
        track_spec_digest="ab" * 32,
    )
    calls = 0
    alerts = []

    class MetadataApi:
        def model_info(self, *_args, **_kwargs):
            sibling = SimpleNamespace(
                rfilename="selector.safetensors",
                size=6,
                lfs=None,
            )
            return SimpleNamespace(siblings=[sibling], sha="12" * 20)

    def disk_full(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(download_module, "HfApi", lambda **_kwargs: MetadataApi())
    monkeypatch.setattr(download_module.tempfile, "mkstemp", disk_full)
    configured = replace(
        settings(tmp_path),
        max_attempts=2,
        alert=lambda event, details: alerts.append((event, details)),
        restore_retry=retries.restore,
        record_retry=retries.record_failure,
    )
    submission = submission_at(100, "hk1")
    outcomes = []
    states = []
    for index, now in enumerate((100, 30_100, 1_000_000), start=1):
        with ChallengeStore(tmp_path / f"disk-full-{index}.db") as store:
            identifier, _ = store.add_submission(submission, occurred_at_ms=1)
            outcome = process_next_admission(
                store,
                worker_id="admission",
                settings=configured,
                current_finalized_block=10_000,
                now_ms=now,
            )
            assert outcome is not None
            outcomes.append(outcome.status)
            states.append(store.get_challenge(identifier).state)

    assert outcomes == [
        AdmissionStatus.DEFERRED,
        AdmissionStatus.BLOCKED,
        AdmissionStatus.BLOCKED,
    ]
    assert states == [
        ChallengeState.SYSTEM_DEFERRED,
        ChallengeState.SYSTEM_BLOCKED,
        ChallengeState.SYSTEM_BLOCKED,
    ]
    assert calls == 2
    assert [event for event, _details in alerts] == [
        "ADMISSION_RETRY_STARTED",
        "ADMISSION_RETRY_EXHAUSTED",
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
