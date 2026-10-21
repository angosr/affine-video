from dataclasses import replace

import pytest

from validator.production.retries import PrivateRetryStore
from validator.production.storage import RETRY_STATE_NAMESPACE, PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState
from validator.store import LeasedChallenge

from .test_storage import MemoryS3


def test_disclosure_retry_survives_cold_host_without_touching_miner_attempts():
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    assert retry.disclosure_attempt(digest, now_ms=100, lease_ms=1000) == 1
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    assert cold.disclosure_attempt(digest, now_ms=1099, lease_ms=1000) is None
    cold.disclosure_attempt(digest, now_ms=1100, lease_ms=1000,
                            completed_attempt=1, error="network unavailable")
    assert retry.disclosure_attempt(digest, now_ms=61_099, lease_ms=1000) is None
    assert retry.disclosure_attempt(digest, now_ms=61_100, lease_ms=1000) == 2
    with pytest.raises(RuntimeError, match="attempt fence"):
        cold.disclosure_attempt(digest, now_ms=61_101, lease_ms=1000,
                                completed_attempt=1, error=None)
    assert retry.restore(task(lease_token="miner", attempts=1), "SCORING") is None


def task(*, lease_token: str, attempts: int) -> LeasedChallenge:
    return LeasedChallenge(
        challenge_id="ab" * 32,
        submission_id="cd" * 32,
        state=ChallengeState.SCORING,
        hotkey="hotkey",
        repo_id="owner/model",
        revision="12" * 20,
        position=ChainPosition(100, 0, 0),
        lease_token=lease_token,
        lease_expires_at_ms=1_000,
        attempts=attempts,
        model_path=None,
        file_sha256=None,
    )


def test_retry_state_is_cas_idempotent_blocking_and_resettable() -> None:
    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    first_task = task(lease_token="lease-1", attempts=1)

    first = retries.record_failure(
        first_task,
        "SCORING",
        code="GPU_DOWN",
        summary="worker unavailable",
        retry_after_ms=200,
        max_attempts=2,
    )
    repeated = retries.record_failure(
        first_task,
        "SCORING",
        code="GPU_DOWN",
        summary="worker unavailable",
        retry_after_ms=200,
        max_attempts=2,
    )
    second_task = replace(first_task, lease_token="lease-2", attempts=2)
    blocked = retries.record_failure(
        second_task,
        "SCORING",
        code="GPU_DOWN",
        summary="worker unavailable",
        retry_after_ms=300,
        max_attempts=2,
    )

    assert first == repeated
    assert first.attempts == 1 and not first.blocked
    assert blocked.attempts == 2 and blocked.blocked
    assert blocked.retry_after_ms == 300
    assert retries.restore(replace(second_task, lease_token="lease-3"), "SCORING") == blocked
    assert {
        name for bucket, name in client.objects if bucket == "private"
    } == {
        f"tracks/abc/{RETRY_STATE_NAMESPACE}/{second_task.challenge_id}/scoring.json"
    }
    probe = retries.record_failure(
        replace(second_task, lease_token="automatic-probe", attempts=3), "SCORING",
        code="GPU_DOWN", summary="still unavailable", retry_after_ms=400, max_attempts=2,
    )
    assert probe.blocked and probe.attempts == 3 and probe.retry_after_ms == 400

    retries.reset(second_task.challenge_id, "SCORING")
    assert retries.restore(replace(second_task, lease_token="lease-3"), "SCORING") is None
