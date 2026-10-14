from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from validator.queue import ChainPosition, ChallengeState, submission_id
from validator.store import ChallengeStore, SubmissionRecord
from validator.tournament import resolve_tournament


def make_submission(*, hotkey: str = "hk1") -> SubmissionRecord:
    position = ChainPosition(100, 2, 1)
    block_hash = "0x" + "ab" * 32
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


def submission_at(block: int, *, hotkey: str = "hk1") -> SubmissionRecord:
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


def test_submission_replay_is_idempotent(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, inserted = store.add_submission(make_submission(), occurred_at_ms=10)
        replay_identifier, replay_inserted = store.add_submission(
            make_submission(), occurred_at_ms=20
        )

        assert inserted is True
        assert replay_inserted is False
        assert replay_identifier == identifier
        assert len(store.challenge_events(identifier)) == 1


def test_transition_updates_projection_and_appends_history(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(make_submission(), occurred_at_ms=10)
        store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=20)
        store.transition(
            identifier,
            ChallengeState.PARTICIPANT_FAILED,
            occurred_at_ms=30,
            error_code="REPOSITORY_UNAVAILABLE",
            error_summary="repository is not publicly readable",
            details={"http_status": 404},
        )

        challenge = store.get_challenge(identifier)
        events = store.challenge_events(identifier)

    assert challenge.state is ChallengeState.PARTICIPANT_FAILED
    assert challenge.error_code == "REPOSITORY_UNAVAILABLE"
    assert [event.sequence for event in events] == [0, 1, 2]
    assert events[-1].previous_state is ChallengeState.FETCHING
    assert events[-1].details == {"http_status": 404}


def test_invalid_transition_writes_nothing(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(make_submission(), occurred_at_ms=10)

        with pytest.raises(ValueError):
            store.transition(identifier, ChallengeState.SUCCEEDED, occurred_at_ms=20)

        assert store.get_challenge(identifier).state is ChallengeState.QUEUED
        assert len(store.challenge_events(identifier)) == 1


def test_state_survives_reopen(tmp_path) -> None:
    path = tmp_path / "validator.db"
    with ChallengeStore(path) as store:
        identifier, _ = store.add_submission(make_submission(), occurred_at_ms=10)
        store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=20)

    with ChallengeStore(path) as reopened:
        assert reopened.get_challenge(identifier).state is ChallengeState.FETCHING
        assert len(reopened.challenge_events(identifier)) == 2


def test_concurrent_first_open_initializes_database_once(tmp_path) -> None:
    path = tmp_path / "validator.db"
    start = Barrier(8)

    def open_store(_worker: int) -> None:
        start.wait()
        with ChallengeStore(path):
            pass

    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(open_store, range(8)))

    with ChallengeStore(path) as store:
        assert store.chain_cursor() is None


def test_conflicting_replay_is_rejected(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        original = make_submission()
        store.add_submission(original, occurred_at_ms=10)
        conflicting = SubmissionRecord(
            submission_id=original.submission_id,
            hotkey="different-hotkey",
            netuid=original.netuid,
            block_hash=original.block_hash,
            position=original.position,
            payload=original.payload,
            repo_id=original.repo_id,
            revision=original.revision,
        )

        with pytest.raises(RuntimeError, match="collision"):
            store.add_submission(conflicting, occurred_at_ms=20)


def test_submission_record_rejects_mismatched_identifier() -> None:
    original = make_submission()

    with pytest.raises(ValueError, match="identifier"):
        SubmissionRecord(
            submission_id="00" * 32,
            hotkey=original.hotkey,
            netuid=original.netuid,
            block_hash=original.block_hash,
            position=original.position,
            payload=original.payload,
            repo_id=original.repo_id,
            revision=original.revision,
        )


def test_database_binds_challenges_to_one_track_spec(tmp_path) -> None:
    path = tmp_path / "validator.db"
    digest = "ab" * 32
    with ChallengeStore(path, track_spec_digest=digest.upper()) as store:
        identifier, _ = store.add_submission(make_submission(), occurred_at_ms=10)
        assert store.track_spec_digest() == digest
        assert store.public_challenge(identifier)["track_spec_digest"] == digest

    with ChallengeStore(path, track_spec_digest=digest):
        pass
    with pytest.raises(RuntimeError, match="different track"):
        ChallengeStore(path, track_spec_digest="cd" * 32)


def test_concurrent_workers_bind_the_same_track_idempotently(tmp_path) -> None:
    path = tmp_path / "validator.db"
    digest = "ab" * 32
    with ChallengeStore(path):
        pass
    start = Barrier(8)

    def bind_track(_worker: int) -> None:
        start.wait()
        with ChallengeStore(path, track_spec_digest=digest):
            pass

    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(bind_track, range(8)))

    with ChallengeStore(path) as store:
        assert store.track_spec_digest() == digest


def test_concurrent_conflicting_track_bind_has_one_winner(tmp_path) -> None:
    path = tmp_path / "validator.db"
    with ChallengeStore(path):
        pass
    start = Barrier(2)

    def bind_track(digest: str) -> str | None:
        start.wait()
        try:
            with ChallengeStore(path, track_spec_digest=digest):
                return None
        except RuntimeError as exc:
            return str(exc)

    digests = ("ab" * 32, "cd" * 32)
    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(bind_track, digests))

    assert outcomes.count(None) == 1
    assert sum("different track" in outcome for outcome in outcomes if outcome is not None) == 1
    with ChallengeStore(path) as store:
        assert store.track_spec_digest() in digests


def test_chain_cursor_is_contiguous_and_persistent(tmp_path) -> None:
    path = tmp_path / "validator.db"
    first_hash = "0x" + "ab" * 32
    second_hash = "0x" + "cd" * 32
    with ChallengeStore(path) as store:
        assert store.chain_cursor() is None
        assert store.advance_chain_cursor(100, first_hash) is True
        assert store.advance_chain_cursor(100, first_hash) is False
        with pytest.raises(ValueError, match="hexadecimal"):
            store.advance_chain_cursor(101, "0x" + "zz" * 32)
        with pytest.raises(ValueError, match="exactly one"):
            store.advance_chain_cursor(102, second_hash)

    with ChallengeStore(path) as reopened:
        assert reopened.chain_cursor() == (100, first_hash)
        assert reopened.advance_chain_cursor(101, second_hash) is True


def test_cooldown_is_based_on_last_accepted_submission(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        first, _ = store.add_submission(submission_at(100), cooldown_blocks=20, occurred_at_ms=10)
        rejected, _ = store.add_submission(
            submission_at(119), cooldown_blocks=20, occurred_at_ms=20
        )
        next_accepted, _ = store.add_submission(
            submission_at(120), cooldown_blocks=20, occurred_at_ms=30
        )

        assert store.get_challenge(first).state is ChallengeState.QUEUED
        assert store.get_challenge(rejected).state is ChallengeState.REJECTED_RATE_LIMIT
        assert store.get_challenge(next_accepted).state is ChallengeState.QUEUED


def test_malformed_submission_is_recorded_and_does_not_consume_cooldown(tmp_path) -> None:
    malformed = submission_at(100)
    malformed = SubmissionRecord(
        submission_id=malformed.submission_id,
        hotkey=malformed.hotkey,
        netuid=malformed.netuid,
        block_hash=malformed.block_hash,
        position=malformed.position,
        payload=b"VDA1bad",
        repo_id=None,
        revision=None,
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        malformed_id, _ = store.add_submission(malformed, cooldown_blocks=20, occurred_at_ms=10)
        accepted_id, _ = store.add_submission(
            submission_at(101), cooldown_blocks=20, occurred_at_ms=20
        )

        assert store.get_challenge(malformed_id).state is ChallengeState.MALFORMED_SUBMISSION
        assert store.get_challenge(accepted_id).state is ChallengeState.QUEUED


def advance_to_scoring(store: ChallengeStore, identifier: str, timestamp: int) -> None:
    store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=timestamp)
    store.transition(identifier, ChallengeState.VALIDATING, occurred_at_ms=timestamp + 1)
    store.transition(identifier, ChallengeState.SCORING, occurred_at_ms=timestamp + 2)


def test_leaderboard_keeps_hotkeys_best_authoritative_score(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        older, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        newer, _ = store.add_submission(submission_at(120), occurred_at_ms=20)
        advance_to_scoring(store, older, 30)
        advance_to_scoring(store, newer, 40)

        store.complete_challenge(
            newer,
            ChallengeState.SUCCEEDED,
            final_score=0.5,
            occurred_at_ms=50,
        )
        store.complete_challenge(
            older,
            ChallengeState.SUCCEEDED,
            final_score=0.9,
            occurred_at_ms=60,
        )

        entries = store.leaderboard()
        assert len(entries) == 1
        assert entries[0].challenge_id == older
        assert entries[0].final_score == 0.9


def test_later_participant_failure_preserves_previous_authoritative_score(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        older, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        newer, _ = store.add_submission(submission_at(120), occurred_at_ms=20)
        advance_to_scoring(store, older, 30)
        store.complete_challenge(
            older,
            ChallengeState.SUCCEEDED,
            final_score=0.9,
            occurred_at_ms=40,
        )
        store.transition(newer, ChallengeState.FETCHING, occurred_at_ms=50)
        store.complete_challenge(
            newer,
            ChallengeState.PARTICIPANT_FAILED,
            error_code="REPOSITORY_UNAVAILABLE",
            occurred_at_ms=51,
        )

        assert store.leaderboard()[0].challenge_id == older
        assert store.leaderboard()[0].final_score == 0.9


def test_system_failure_preserves_previous_success(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        older, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        newer, _ = store.add_submission(submission_at(120), occurred_at_ms=20)
        advance_to_scoring(store, older, 30)
        store.complete_challenge(
            older,
            ChallengeState.SUCCEEDED,
            final_score=0.8,
            occurred_at_ms=40,
        )
        advance_to_scoring(store, newer, 50)
        scoring = store.claim_next_scoring("scoring", lease_ms=100, now_ms=60)
        assert scoring is not None and scoring.challenge_id == newer
        store.fail_leased_system(
            newer,
            scoring.lease_token,
            score_block=121,
            error_code="SCORING_WORKER_FAILURE",
            error_summary="scoring worker unavailable",
            occurred_at_ms=63,
        )

        assert store.get_challenge(newer).state is ChallengeState.SYSTEM_FAILED
        assert store.leaderboard()[0].challenge_id == older


def test_verified_public_match_restores_disposable_local_state(tmp_path) -> None:
    digest = "ab" * 32
    with ChallengeStore(tmp_path / "validator.db", track_spec_digest=digest) as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        public = store.public_challenge(identifier)
        public.update(
            {
                "state": ChallengeState.SUCCEEDED,
                "error_code": None,
                "file_sha256": "cd" * 32,
                "model_digest": "ef" * 32,
                "model_size_bytes": 123,
                "final_score": 0.8,
                "score_block": 110,
                "stages": [],
            }
        )
        tournament = resolve_tournament(
            [public],
            track_id="track",
            track_spec_digest=digest,
            netuid=7,
            minimum_gain=0.001,
        )

        assert store.restore_public_matches(tournament.matches) == 1
        assert store.restore_public_matches(tournament.matches) == 1
        restored = store.get_challenge(identifier)
        assert restored.state is ChallengeState.SUCCEEDED
        assert restored.final_score == 0.8
        assert store.leaderboard()[0].challenge_id == identifier
        assert store.public_challenge(identifier)["stages"] == []


def test_stage_score_replay_must_be_identical(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=11)
        store.transition(identifier, ChallengeState.VALIDATING, occurred_at_ms=12)
        store.transition(identifier, ChallengeState.SCORING, occurred_at_ms=13)
        task = store.claim_next_scoring("worker", lease_ms=100, now_ms=14)
        assert task is not None
        assert store.record_stage_score(
            identifier,
            task.lease_token,
            ChallengeState.SCORING,
            next_state=ChallengeState.SUCCEEDED,
            score=0.7,
            score_block=100,
            metrics={"samples": 4},
            occurred_at_ms=20,
        )
        assert not store.record_stage_score(
            identifier,
            task.lease_token,
            ChallengeState.SCORING,
            next_state=ChallengeState.SUCCEEDED,
            score=0.7,
            score_block=100,
            metrics={"samples": 4},
            occurred_at_ms=30,
        )
        with pytest.raises(RuntimeError, match="different"):
            store.record_stage_score(
                identifier,
                task.lease_token,
                ChallengeState.SCORING,
                next_state=ChallengeState.SUCCEEDED,
                score=0.8,
                score_block=100,
                metrics={"samples": 4},
                occurred_at_ms=40,
            )


def test_stale_worker_cannot_write_stage_score(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=11)
        store.transition(identifier, ChallengeState.VALIDATING, occurred_at_ms=12)
        store.transition(identifier, ChallengeState.SCORING, occurred_at_ms=13)
        stale = store.claim_next_scoring("worker-a", lease_ms=10, now_ms=20)
        current = store.claim_next_scoring("worker-b", lease_ms=10, now_ms=30)
        assert stale is not None and current is not None

        with pytest.raises(RuntimeError, match="lease"):
            store.record_stage_score(
                identifier,
                stale.lease_token,
                ChallengeState.SCORING,
                next_state=ChallengeState.SUCCEEDED,
                score=0.7,
                score_block=100,
                metrics={},
                occurred_at_ms=31,
            )
        assert store.record_stage_score(
            identifier,
            current.lease_token,
            ChallengeState.SCORING,
            next_state=ChallengeState.SUCCEEDED,
            score=0.7,
            score_block=100,
            metrics={},
            occurred_at_ms=32,
        )


def test_expired_admission_lease_is_recovered(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        first = store.claim_next_admission(
            "worker-a", lease_ms=50, ready_through_block=10_000, now_ms=100
        )
        assert first is not None
        assert first.challenge_id == identifier
        assert first.state is ChallengeState.FETCHING
        assert first.attempts == 1

        assert (
            store.claim_next_admission(
                "worker-b", lease_ms=50, ready_through_block=10_000, now_ms=149
            )
            is None
        )
        reclaimed = store.claim_next_admission(
            "worker-b", lease_ms=50, ready_through_block=10_000, now_ms=150
        )
        assert reclaimed is not None
        assert reclaimed.challenge_id == identifier
        assert reclaimed.lease_token != first.lease_token
        assert reclaimed.attempts == 2

        with pytest.raises(RuntimeError, match="lease"):
            store.advance_leased(
                identifier,
                first.lease_token,
                ChallengeState.VALIDATING,
                release=False,
                occurred_at_ms=151,
            )
        store.advance_leased(
            identifier,
            reclaimed.lease_token,
            ChallengeState.VALIDATING,
            release=False,
            occurred_at_ms=151,
        )


def test_deferred_admission_does_not_block_fifo_until_retry_time(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        second_id, _ = store.add_submission(submission_at(101, hotkey="hk2"), occurred_at_ms=20)
        first = store.claim_next_admission(
            "worker", lease_ms=100, ready_through_block=10_000, now_ms=100
        )
        assert first is not None
        store.defer_leased(
            first_id,
            first.lease_token,
            retry_after_ms=200,
            error_code="DOWNLOAD_TEMPORARY_FAILURE",
            error_summary="temporary outage",
            occurred_at_ms=110,
        )

        second = store.claim_next_admission(
            "worker", lease_ms=100, ready_through_block=10_000, now_ms=150
        )
        assert second is not None
        assert second.challenge_id == second_id
        store.advance_leased(
            second_id,
            second.lease_token,
            ChallengeState.VALIDATING,
            release=False,
            occurred_at_ms=151,
        )
        store.advance_leased(
            second_id,
            second.lease_token,
            ChallengeState.SCORING,
            release=True,
            occurred_at_ms=152,
        )

        retry = store.claim_next_admission(
            "worker", lease_ms=100, ready_through_block=10_000, now_ms=200
        )
        assert retry is not None
        assert retry.challenge_id == first_id
        assert retry.state is ChallengeState.FETCHING


def test_scoring_claims_are_fifo_and_reject_stale_completion(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        second_id, _ = store.add_submission(submission_at(101, hotkey="hk2"), occurred_at_ms=20)
        for identifier in (first_id, second_id):
            store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=30)
            store.transition(identifier, ChallengeState.VALIDATING, occurred_at_ms=31)
            store.transition(identifier, ChallengeState.SCORING, occurred_at_ms=32)

        first = store.claim_next_scoring("score-a", lease_ms=50, now_ms=100)
        second = store.claim_next_scoring("score-b", lease_ms=50, now_ms=100)
        assert first is not None and first.challenge_id == first_id
        assert second is None

        reclaimed = store.claim_next_scoring("score-c", lease_ms=50, now_ms=150)
        assert reclaimed is not None and reclaimed.challenge_id == first_id
        with pytest.raises(RuntimeError, match="lease"):
            store.complete_challenge(
                first_id,
                ChallengeState.PARTICIPANT_FAILED,
                error_code="MODEL_RUNTIME_INVALID",
                occurred_at_ms=151,
                lease_token=first.lease_token,
            )
        store.complete_challenge(
            first_id,
            ChallengeState.PARTICIPANT_FAILED,
            error_code="MODEL_RUNTIME_INVALID",
            occurred_at_ms=151,
            lease_token=reclaimed.lease_token,
        )
        second = store.claim_next_scoring("score-b", lease_ms=50, now_ms=152)
        assert second is not None and second.challenge_id == second_id


def test_scoring_waits_for_an_earlier_deferred_challenge(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        first_id, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        second_id, _ = store.add_submission(submission_at(101, hotkey="hk2"), occurred_at_ms=20)

        first = store.claim_next_admission(
            "admit-a", lease_ms=100, ready_through_block=10_000, now_ms=100
        )
        assert first is not None and first.challenge_id == first_id
        store.defer_leased(
            first_id,
            first.lease_token,
            retry_after_ms=200,
            error_code="DOWNLOAD_TEMPORARY_FAILURE",
            error_summary="temporary outage",
            occurred_at_ms=110,
        )
        advance_to_scoring(store, second_id, 120)

        assert store.claim_next_scoring("score", lease_ms=50, now_ms=150) is None

        retry = store.claim_next_admission(
            "admit-b", lease_ms=100, ready_through_block=10_000, now_ms=200
        )
        assert retry is not None and retry.challenge_id == first_id
        store.fail_leased_system(
            first_id,
            retry.lease_token,
            score_block=101,
            error_code="DOWNLOAD_TEMPORARY_FAILURE",
            error_summary="retry limit reached",
            occurred_at_ms=201,
        )

        second = store.claim_next_scoring("score", lease_ms=50, now_ms=202)
        assert second is not None and second.challenge_id == second_id


def test_public_snapshot_uses_explicit_allowlist(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        lease = store.claim_next_admission(
            "secret-worker", lease_ms=1_000, ready_through_block=10_000, now_ms=20
        )
        assert lease is not None
        store.record_downloaded_model(
            identifier,
            lease.lease_token,
            model_path="/secret/cache/model.safetensors",
            file_sha256="ab" * 32,
            size_bytes=123,
            occurred_at_ms=21,
            details={"private": "must-not-export"},
        )
        store.register_validated_model(
            identifier,
            lease.lease_token,
            model_digest="cd" * 32,
            occurred_at_ms=22,
        )
        scoring = store.claim_next_scoring("score-worker", lease_ms=1_000, now_ms=23)
        assert scoring is not None
        store.record_stage_score(
            identifier,
            scoring.lease_token,
            ChallengeState.SCORING,
            next_state=ChallengeState.SUCCEEDED,
            score=0.7,
            score_block=100,
            metrics={
                "upper_bound": 0.8,
                "budget_scores": [0.6, 0.7, 0.8],
                "sample_count": 12,
                "private": "must-not-export",
            },
            occurred_at_ms=24,
        )
        store.defer_leased(
            identifier,
            scoring.lease_token,
            retry_after_ms=2_000,
            error_code="SCORING_CACHE_INVALID",
            error_summary="frozen cache is unavailable",
            occurred_at_ms=25,
        )

        challenge = store.public_challenge(identifier)
        serialized = str(challenge)
        assert "secret-worker" not in serialized
        assert "/secret/cache" not in serialized
        assert "must-not-export" not in serialized
        assert challenge["file_sha256"] == "ab" * 32
        assert challenge["state"] == ChallengeState.SYSTEM_DEFERRED
        assert challenge["error_summary"] == "frozen cache is unavailable"
        assert challenge["stages"] == [
            {
                "stage": ChallengeState.SCORING,
                "next_state": ChallengeState.SUCCEEDED,
                "score": 0.7,
                "score_block": 100,
                "created_at_ms": 24,
            }
        ]
        assert challenge["events"][-1]["error_summary"] == "frozen cache is unavailable"


def test_public_challenge_reads_state_and_history_from_one_snapshot(monkeypatch, tmp_path) -> None:
    database = tmp_path / "validator.db"
    with ChallengeStore(database) as store, ChallengeStore(database) as writer:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
        original = store._public_challenge_row

        def racing_read(row):
            writer.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=20)
            return original(row)

        monkeypatch.setattr(store, "_public_challenge_row", racing_read)
        challenge = store.public_challenge(identifier)

        assert challenge["state"] == ChallengeState.QUEUED
        assert [event["state"] for event in challenge["events"]] == [ChallengeState.QUEUED]
        assert writer.get_challenge(identifier).state is ChallengeState.FETCHING
