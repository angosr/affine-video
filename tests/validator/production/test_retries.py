import json
from dataclasses import replace

import pytest

from miner.submit import PrivateModelSubmission
from validator.production.commitment import canonical_json
from validator.production.retries import PrivateRetryStore
from validator.production.storage import RETRY_STATE_NAMESPACE, PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState
from validator.retries import persist_retry_failure
from validator.store import LeasedChallenge

from .test_storage import MemoryS3


def test_disclosure_retry_survives_cold_host_without_touching_miner_attempts():
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    assert not retry.disclosure_failed(digest)
    assert retry.disclosure_attempt(digest, now_ms=100, lease_ms=1000) == 1
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    assert cold.disclosure_attempt(digest, now_ms=1099, lease_ms=1000) is None
    cold.disclosure_attempt(digest, now_ms=1100, lease_ms=1000,
                            completed_attempt=1, error="network unavailable")
    assert cold.disclosure_failed(digest)
    assert retry.disclosure_attempt(digest, now_ms=61_099, lease_ms=1000) is None
    assert retry.disclosure_attempt(digest, now_ms=61_100, lease_ms=1000) == 2
    assert cold.disclosure_failed(digest)  # Reserving a retry is not recovery.
    with pytest.raises(RuntimeError, match="attempt fence"):
        cold.disclosure_attempt(digest, now_ms=61_101, lease_ms=1000,
                                completed_attempt=1, error=None)
    cold.disclosure_attempt(digest, now_ms=61_102, lease_ms=1000,
                            completed_attempt=2, error=None)
    assert not retry.disclosure_failed(digest)
    assert retry.restore(task(lease_token="miner", attempts=1), "SCORING") is None


@pytest.mark.parametrize("prior_failure", [False, True])
def test_operator_disclosure_waiver_is_the_original_cas_not_a_fake_receipt(prior_failure):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    name = f"{RETRY_STATE_NAMESPACE}/disclosures/{digest}.json"
    old = None
    if prior_failure:
        retry.disclosure_attempt(digest, now_ms=10, lease_ms=100)
        retry.disclosure_attempt(digest, now_ms=20, lease_ms=100,
                                 completed_attempt=1, error="original failure")
        old = json.loads(objects.get(name).body)
    retry.waive_disclosure(digest, now_ms=70_000, reason="Operator decision")
    sealed = objects.get(name)
    row = json.loads(sealed.body)
    assert row["operator_waiver"] == {"decided_at_ms": 70_000, "reason": "Operator decision"}
    if old is not None:
        assert {key: row[key] for key in old} == old
    else:
        assert row["attempt"] == row["retry_after_ms"] == 0 and row["error"] is None
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    cold.waive_disclosure(digest, now_ms=999_999, reason="Operator decision")
    assert objects.get(name) == sealed  # Idempotent after a lost response, not a new clock.
    assert cold.disclosure_attempt(digest, now_ms=10**12, lease_ms=100) is None
    assert not cold.disclosure_failed(digest)
    if old is not None:
        assert cold.disclosure_attempt(digest, now_ms=10**12, lease_ms=100,
                                       completed_attempt=1, error=None) is None
    assert objects.get_optional(f"disclosed/{digest}.json") is None
    assert cold.restore(task(lease_token="miner", attempts=1), "SCORING") is None
    public = cold.disclosure_status(digest, now_ms=10**12)
    assert public["state"] == "WAIVED" and public["retry_after_ms"] is None
    assert "reason" not in json.dumps(public) and "original failure" not in json.dumps(public)
    with pytest.raises(RuntimeError, match="different operator decision"):
        cold.waive_disclosure(digest, now_ms=10**12, reason="Different decision")
    assert objects.get(name) == sealed


def test_disclosure_projection_distinguishes_upload_failure_and_expired_fence():
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    assert retry.disclosure_status(digest, now_ms=0)["state"] == "OVERDUE"
    retry.disclosure_attempt(digest, now_ms=100, lease_ms=100)
    assert retry.disclosure_status(digest, now_ms=101)["state"] == "UPLOADING"
    assert retry.disclosure_status(digest, now_ms=200)["state"] == "OVERDUE"
    retry.disclosure_attempt(digest, now_ms=200, lease_ms=100,
                             completed_attempt=1, error="prior network failure")
    assert retry.disclosure_status(digest, now_ms=201)["state"] == "FAILED"
    assert retry.disclosure_attempt(digest, now_ms=60_200, lease_ms=100) == 2
    assert retry.disclosure_status(digest, now_ms=60_201)["state"] == "FAILED"


@pytest.mark.parametrize("failed", [False, True])
def test_disclosure_waiver_stops_future_work_during_upload_or_cooldown(failed):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    name = f"{RETRY_STATE_NAMESPACE}/disclosures/{digest}.json"
    retry.disclosure_attempt(digest, now_ms=100, lease_ms=1000)
    if failed:
        retry.disclosure_attempt(digest, now_ms=101, lease_ms=1000,
                                 completed_attempt=1, error="prior failure")
    before = json.loads(objects.get(name).body)
    assert before["retry_after_ms"] > 102
    retry.waive_disclosure(digest, now_ms=102, reason="Operator decision")
    sealed = objects.get(name)
    after = json.loads(sealed.body)
    assert {key: after[key] for key in before} == before
    assert after["operator_waiver"]["decided_at_ms"] == 102
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    assert cold.disclosure_attempt(digest, now_ms=10**12, lease_ms=1000) is None
    assert cold.disclosure_attempt(digest, now_ms=103, lease_ms=1000,
                                  completed_attempt=1, error=None) is None
    assert objects.get(name) == sealed
    assert objects.get_optional(f"disclosed/{digest}.json") is None


def test_disclosure_waiver_cas_race_preserves_the_attempt_that_reserved_first(monkeypatch):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    create = objects.create_plan
    raced = False

    def reserve_first(name, payload):
        nonlocal raced
        if not raced:
            raced = True
            retry.disclosure_attempt(digest, now_ms=10, lease_ms=100)
        return create(name, payload)

    monkeypatch.setattr(objects, "create_plan", reserve_first)
    retry.waive_disclosure(digest, now_ms=10, reason="Operator decision")
    name = f"{RETRY_STATE_NAMESPACE}/disclosures/{digest}.json"
    sealed = objects.get(name)
    row = json.loads(sealed.body)
    assert row["attempt"] == 1 and row["retry_after_ms"] == 110 and row["error"] is None
    assert row["operator_waiver"] == {"decided_at_ms": 10, "reason": "Operator decision"}
    assert retry.disclosure_status(digest, now_ms=11)["state"] == "WAIVED"
    assert retry.disclosure_attempt(digest, now_ms=12, lease_ms=100,
                                    completed_attempt=1, error=None) is None
    assert objects.get(name) == sealed


@pytest.mark.parametrize("operation", ["reserve", "completion"])
def test_disclosure_waiver_cas_winner_cannot_be_overwritten_by_old_writer(
    monkeypatch, operation,
):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    name = f"{RETRY_STATE_NAMESPACE}/disclosures/{digest}.json"
    retry.disclosure_attempt(digest, now_ms=100, lease_ms=100)
    before = json.loads(objects.get(name).body)
    replace_plan = objects.replace_plan
    raced = False

    def waive_first(key, payload, *, etag):
        nonlocal raced
        if not raced:
            raced = True
            retry.waive_disclosure(digest, now_ms=200, reason="Operator decision")
        return replace_plan(key, payload, etag=etag)

    monkeypatch.setattr(objects, "replace_plan", waive_first)
    kwargs = {} if operation == "reserve" else {"completed_attempt": 1, "error": "late failure"}
    assert retry.disclosure_attempt(digest, now_ms=200, lease_ms=100, **kwargs) is None
    after = json.loads(objects.get(name).body)
    assert {key: after[key] for key in before} == before
    assert after["operator_waiver"] == {"decided_at_ms": 200, "reason": "Operator decision"}


def test_disclosure_waiver_lost_ack_preserves_the_original_decision(monkeypatch):
    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    put = client.put_object
    lost = False

    def write_then_lose_response(**kwargs):
        nonlocal lost
        result = put(**kwargs)
        if not lost:
            lost = True
            raise OSError("response lost after commit")
        return result

    monkeypatch.setattr(client, "put_object", write_then_lose_response)
    with pytest.raises(OSError, match="response lost"):
        retry.waive_disclosure(digest, now_ms=100, reason="Operator decision")
    row = objects.get(f"retry_state/disclosures/{digest}.json")
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    cold.waive_disclosure(digest, now_ms=200, reason="Operator decision")
    assert objects.get(f"retry_state/disclosures/{digest}.json") == row
    assert cold.disclosure_status(digest, now_ms=200)["waived_at_ms"] == 100


@pytest.mark.parametrize("change", ["null", "missing_time", "negative_time", "control", "zero"])
def test_disclosure_waiver_invalid_rows_are_not_suppressed(change):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    retry.waive_disclosure(digest, now_ms=100, reason="Operator decision")
    name = f"{RETRY_STATE_NAMESPACE}/disclosures/{digest}.json"
    stored = objects.get(name)
    row = json.loads(stored.body)
    if change == "null":
        row["operator_waiver"] = None
    elif change == "missing_time":
        del row["operator_waiver"]["decided_at_ms"]
    elif change == "negative_time":
        row["operator_waiver"]["decided_at_ms"] = -1
    elif change == "control":
        row["operator_waiver"]["reason"] = "secret\nvalue"
    else:
        del row["operator_waiver"]  # Old non-waived records still require a real attempt.
    objects.replace_plan(name, canonical_json(row), etag=stored.etag)
    with pytest.raises(RuntimeError, match="authority is invalid"):
        retry.disclosure_status(digest, now_ms=200)


@pytest.mark.parametrize("field", ["track_spec_digest", "plan_digest", "error"])
def test_disclosure_failure_read_uses_the_same_strict_retry_decoder(field):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retry = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    digest = "ab" * 32
    retry.disclosure_attempt(digest, now_ms=100, lease_ms=1000)
    name = f"{RETRY_STATE_NAMESPACE}/disclosures/{digest}.json"
    stored = objects.get(name)
    document = json.loads(stored.body)
    document[field] = 0 if field == "error" else "00" * 32
    objects.replace_plan(name, canonical_json(document), etag=stored.etag)
    before = objects.get(name)
    with pytest.raises(RuntimeError, match="authority is invalid"):
        retry.disclosure_failed(digest)
    with pytest.raises(RuntimeError, match="authority is invalid"):
        retry.disclosure_attempt(digest, now_ms=1200, lease_ms=1000)
    assert objects.get(name) == before


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


@pytest.mark.parametrize("mutation", ["upload_id", "file_sha256", "hf"])
def test_private_retry_keeps_original_source_and_rejects_rebinding(mutation):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    reference = PrivateModelSubmission("ab" * 32, "cd" * 32)
    original = replace(task(lease_token="original", attempts=1), repo_id=None, revision=None,
                       private_submission=reference)
    checkpoint = retries.record_failure(original, "ADMISSION", code="PRIVATE_UPLOAD_UNAVAILABLE",
                                        summary="missing part", retry_after_ms=200, max_attempts=5)
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    assert cold.restore(original, "ADMISSION") == checkpoint
    if mutation == "hf":
        changed = replace(original, repo_id="owner/model", revision="12" * 20,
                          private_submission=None)
    else:
        changed = replace(original, private_submission=replace(reference, **{mutation: "ff" * 32}))
    with pytest.raises(RuntimeError, match="conflicts with its (submission|private source)"):
        cold.restore(changed, "ADMISSION")


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
    before = dict(client.objects)
    probe = retries.record_failure(
        replace(second_task, lease_token="automatic-probe", attempts=3), "SCORING",
        code="GPU_DOWN", summary="still unavailable", retry_after_ms=400, max_attempts=2,
    )
    assert probe == blocked
    assert client.objects == before

    retries.reset(second_task.challenge_id, "SCORING")
    assert retries.restore(replace(second_task, lease_token="lease-3"), "SCORING") is None
    reset = json.loads(next(iter(client.objects.values()))[0])
    assert reset["generation"] == 2 and reset["attempts"] == 0
    resumed = retries.record_failure(
        replace(first_task, lease_token="operator-resumed"), "SCORING",
        code="GPU_DOWN", summary="worker unavailable", retry_after_ms=500, max_attempts=2,
    )
    assert resumed.attempts == 1 and not resumed.blocked


def test_legacy_deferred_checkpoint_cannot_record_past_the_current_budget():
    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    original = task(lease_token="old", attempts=1)
    retries.record_failure(original, "SCORING", code="GPU_DOWN", summary="outage",
                           retry_after_ms=200, max_attempts=2)
    before = dict(client.objects)
    with pytest.raises(RuntimeError, match="operator reset"):
        retries.record_failure(replace(original, lease_token="new", attempts=2), "SCORING",
                               code="GPU_DOWN", summary="outage", retry_after_ms=300,
                               max_attempts=1)
    assert client.objects == before


@pytest.mark.parametrize("attempts", [1, 3])
def test_blocked_recorder_ordinal_mismatch_fails_without_new_io_or_cas(monkeypatch, attempts):
    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    first = task(lease_token="first", attempts=1)
    retries.record_failure(first, "SCORING", code="GPU_DOWN", summary="worker unavailable",
                           retry_after_ms=200, max_attempts=2)
    retries.record_failure(replace(first, lease_token="last", attempts=2), "SCORING",
                           code="GPU_DOWN", summary="worker unavailable",
                           retry_after_ms=300, max_attempts=2)
    before = dict(client.objects)
    reads = []
    get = client.get_object

    def read_checkpoint(**kwargs):
        reads.append(kwargs)
        return get(**kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("an exhausted ordinal cannot write CAS or perform other object I/O")

    monkeypatch.setattr(client, "get_object", read_checkpoint)
    for method in ("put_object", "delete_object", "head_object", "list_objects_v2"):
        monkeypatch.setattr(client, method, forbidden)
    with pytest.raises(RuntimeError, match="ordinal disagrees"):
        persist_retry_failure(
            replace(first, lease_token="stale-or-extra", attempts=attempts), stage="SCORING",
            code="GPU_DOWN", summary="still unavailable", retry_after_ms=400,
            max_attempts=2, record=retries.record_failure,
        )
    assert client.objects == before
    assert reads == [{"Bucket": "private", "Key": (
        f"tracks/abc/{RETRY_STATE_NAMESPACE}/{first.challenge_id}/scoring.json"
    )}]


def test_publication_clock_cas_lost_response_reset_and_legacy_unknown(monkeypatch):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    original = replace(task(lease_token="one", attempts=1), state=ChallengeState.FETCHING,
                       declared_file_sha256="ab" * 32)
    create = objects.create_plan

    def lost_response(name, body):
        create(name, body)
        raise OSError("response lost after commit")

    monkeypatch.setattr(objects, "create_plan", lost_response)
    committed = retries.record_publication_wait(original, http_status=403, observed_at_ms=100,
                                                max_attempts=5)
    assert committed.attempts == 0 and committed.http_grace.deadline_ms == 600_100
    cold = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    checkpoint = cold.record_publication_wait(original, http_status=404, observed_at_ms=200,
                                              max_attempts=5)
    assert checkpoint.attempts == 0 and checkpoint.http_grace.first_status == 403
    assert checkpoint.retry_after_ms == 600_100
    before = objects.get(f"retry_state/{original.challenge_id}/admission.json")
    cold.reset(original.challenge_id, "ADMISSION")
    active = cold.restore(original, "ADMISSION")
    assert active.http_grace == checkpoint.http_grace and active.attempts == 0
    assert json.loads(before.body)["http_grace"] == json.loads(objects.get(
        f"retry_state/{original.challenge_id}/admission.json").body)["http_grace"]
    ordinary = cold.record_failure(original, "ADMISSION", code="SERVER_DOWN", summary="HF 500",
                                    retry_after_ms=700_000, max_attempts=5)
    assert ordinary.attempts == 1 and ordinary.http_grace == checkpoint.http_grace
    name = f"retry_state/{original.challenge_id}/admission.json"
    stored = objects.get(name)
    legacy = json.loads(stored.body)
    del legacy["http_grace"]
    objects.replace_plan(name, canonical_json(legacy), etag=stored.etag)
    before = objects.get(name)
    with pytest.raises(RuntimeError, match="clock is unknown"):
        cold.record_publication_wait(replace(original, lease_token="two", attempts=2),
                                     http_status=404, observed_at_ms=900_000, max_attempts=5)
    assert objects.get(name) == before


@pytest.mark.parametrize("mutation", [None, "replace", "omit"])
def test_vda2_retry_declaration_is_bound_to_the_original_task(mutation):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    original = replace(task(lease_token="one", attempts=1),
                       declared_file_sha256="ab" * 32)
    saved = retries.record_failure(original, "SCORING", code="GPU_DOWN", summary="unavailable",
                                    retry_after_ms=200, max_attempts=2)
    assert retries.restore(original, "SCORING") == saved
    changed = replace(original, declared_file_sha256=("ff" * 32 if mutation == "replace" else None))
    if mutation is not None:
        with pytest.raises(RuntimeError, match="HF declaration"):
            retries.restore(changed, "SCORING")


@pytest.mark.parametrize("field,value", [("deadline_ms", 999_999), ("first_status", 500)])
def test_publication_clock_tampering_cannot_create_a_wait(field, value):
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    original = replace(task(lease_token="one", attempts=1), state=ChallengeState.FETCHING,
                       declared_file_sha256="ab" * 32)
    retries.record_publication_wait(original, http_status=401, observed_at_ms=100, max_attempts=5)
    name = f"retry_state/{original.challenge_id}/admission.json"
    stored = objects.get(name)
    row = json.loads(stored.body)
    row["http_grace"][field] = value
    objects.replace_plan(name, canonical_json(row), etag=stored.etag)
    with pytest.raises(RuntimeError, match="clock is invalid"):
        retries.restore(original, "ADMISSION")


def test_first_checkpoint_after_an_expired_lease_preserves_the_existing_ordinal():
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    interrupted = task(lease_token="expired-host-recovery", attempts=2)
    checkpoint = persist_retry_failure(
        interrupted, stage="SCORING", code="GPU_DOWN", summary="worker unavailable",
        retry_after_ms=200, max_attempts=2, record=retries.record_failure,
    )
    assert checkpoint.attempts == 2 and checkpoint.blocked
    assert retries.restore(interrupted, "SCORING") == checkpoint


@pytest.mark.parametrize("stage", ["ADMISSION", "SCORING"])
@pytest.mark.parametrize("active_reset", [False, True])
def test_retry_cas_write_failure_then_expired_lease_recovers_from_original_authority(
    tmp_path, monkeypatch, stage, active_reset,
):
    from tests.validator.test_admission import settings as admission_settings
    from tests.validator.test_admission import submission_at
    from validator.admission import process_next_admission
    from validator.download import ModelDownloadError
    from validator.scoring import ScoringSettings, TemporaryScoringError, process_next_scoring
    from validator.store import ChallengeStore

    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")
    retries = PrivateRetryStore(objects, track_spec_digest="ef" * 32)
    original_put = client.put_object
    fail = [True]
    tasks = []
    calls = []

    def put(**kwargs):
        if fail:
            fail.pop()
            raise OSError("transport failed before PUT")
        return original_put(**kwargs)

    def restore(task, phase):
        tasks.append(task)
        return retries.restore(task, phase)

    def unavailable(*_args, **_kwargs):
        calls.append(True)
        if stage == "ADMISSION":
            raise ModelDownloadError("DOWNLOAD_TEMPORARY_FAILURE", "HF 500", temporary=True)
        raise TemporaryScoringError("GPU_DOWN", "worker unavailable")

    monkeypatch.setattr(client, "put_object", put)
    with ChallengeStore(tmp_path / "write-failure.db") as store:
        original = submission_at(100, "hk1")
        identifier, _ = store.add_submission(original, occurred_at_ms=1)
        if stage == "SCORING":
            for state in (ChallengeState.FETCHING, ChallengeState.VALIDATING,
                          ChallengeState.SCORING):
                store.transition(identifier, state, occurred_at_ms=2)
        configured = (
            replace(admission_settings(tmp_path), lease_ms=300_000, max_attempts=2,
                    restore_retry=restore, record_retry=retries.record_failure)
            if stage == "ADMISSION" else
            ScoringSettings(lease_ms=300_000, max_attempts=2, restore_retry=restore,
                            record_retry=retries.record_failure)
        )

        def run(now):
            common = dict(worker_id="worker", current_finalized_block=200,
                          settings=configured, now_ms=now)
            return (process_next_admission(store, downloader=unavailable, **common)
                    if stage == "ADMISSION" else
                    process_next_scoring(store, evaluator=unavailable, **common))

        def attempt_counts():
            return tuple(store._connection.execute(
                "SELECT attempts, total_attempts FROM challenges WHERE challenge_id = ?",
                (identifier,),
            ).fetchone())

        with pytest.raises(OSError, match="before PUT"):
            run(10)
        assert not client.objects
        assert attempt_counts() == (1, 1)
        if active_reset:
            retries.record_failure(tasks[0], stage, code="RESTORED_TRANSPORT",
                                   summary="original failure", retry_after_ms=100,
                                   max_attempts=2)
            retries.reset(identifier, stage)
        assert run(300_011) is not None
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_DEFERRED
        assert attempt_counts() == (1, 2)
        assert run(600_012) is not None
        row = store.public_challenge(identifier)
        assert row["state"] == ChallengeState.SYSTEM_BLOCKED
        assert row["submission_id"] == original.submission_id
        assert row["block_height"] == original.position.block
        assert attempt_counts() == (2, 3)
        document = json.loads(objects.get(f"retry_state/{identifier}/{stage.lower()}.json").body)
        assert document["attempts"] == 2 and document["state"] == "BLOCKED"
        assert all(key.endswith(f"retry_state/{identifier}/{stage.lower()}.json")
                   for _bucket, key in client.objects)
        assert run(10**9) is None and len(calls) == 3
        assert attempt_counts() == (2, 3)
