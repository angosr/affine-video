import hashlib
import time
from dataclasses import replace

import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.evaluate import reference_attention, trip_pool
from scoring.local import ExactCase, ScreenCase
from scoring.selector import SelectorArchitecture
from scoring.spec import (
    BaseModelSpec,
    BudgetSpec,
    CacheSpec,
    ModelSpec,
    RolloutSpec,
    ScoringSpec,
    TrackSpec,
)
from validator.queue import ChainPosition, ChallengeState
from validator.scoring import (
    IsolatedEvaluator,
    ParticipantScoringError,
    RolloutResult,
    ScoringSettings,
    StageOutcome,
    TemporaryScoringError,
    TrackEvaluator,
    evaluate_isolated,
    process_next_scoring,
)
from validator.store import ChallengeStore, LeasedChallenge

from .test_store import submission_at


def isolated_success(_task: LeasedChallenge) -> StageOutcome:
    return StageOutcome(
        ChallengeState.SCORING,
        score=0.75,
        attention_cost=0.1,
        metrics={"worker": "child"},
    )


def isolated_sleep(_task: LeasedChallenge) -> StageOutcome:
    time.sleep(1)
    return isolated_success(_task)


def isolated_participant_failure(_task: LeasedChallenge) -> StageOutcome:
    raise ParticipantScoringError("MODEL_RUNTIME_INVALID", "invalid output")


def isolated_memory_failure(_task: LeasedChallenge) -> StageOutcome:
    raise MemoryError


def sample_task() -> LeasedChallenge:
    return LeasedChallenge(
        challenge_id="challenge",
        submission_id="56" * 32,
        state=ChallengeState.SCREENING,
        hotkey="hotkey",
        repo_id="owner/model",
        revision="12" * 20,
        position=ChainPosition(100, 0, 0),
        lease_token="lease",
        lease_expires_at_ms=1000,
        attempts=1,
        model_path="/cache/model.safetensors",
        file_sha256="ab" * 32,
    )


def test_isolated_evaluator_returns_only_structured_result() -> None:
    evaluator = IsolatedEvaluator(isolated_success, timeout_seconds=5)

    result = evaluator(sample_task())

    assert result.score == 0.75
    assert result.metrics == {"worker": "child"}


def test_isolated_evaluator_terminates_timeout_and_preserves_error_class() -> None:
    with pytest.raises(TemporaryScoringError) as timeout:
        evaluate_isolated(isolated_sleep, sample_task(), timeout_seconds=0.1)
    assert timeout.value.code == "SCORING_TIMEOUT"

    with pytest.raises(ParticipantScoringError) as invalid:
        evaluate_isolated(
            isolated_participant_failure,
            sample_task(),
            timeout_seconds=5,
        )
    assert invalid.value.code == "MODEL_RUNTIME_INVALID"

    with pytest.raises(TemporaryScoringError) as memory:
        evaluate_isolated(
            isolated_memory_failure,
            sample_task(),
            timeout_seconds=5,
        )
    assert memory.value.code == "SCORING_MEMORY_LIMIT"


def ready_for_screening(store: ChallengeStore) -> str:
    identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
    store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=11)
    store.transition(identifier, ChallengeState.VALIDATING, occurred_at_ms=12)
    store.transition(identifier, ChallengeState.SCREENING, occurred_at_ms=13)
    return identifier


def test_scoring_stages_advance_and_finish_with_one_leaderboard_result(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_screening(store)
        settings = ScoringSettings(lease_ms=100, retry_base_ms=10, retry_max_ms=100)

        screening = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=lambda _task: StageOutcome(
                ChallengeState.SCORING,
                score=0.8,
                attention_cost=0.05,
                metrics={"upper_bound": 0.9},
            ),
            now_ms=20,
        )
        assert screening is not None and screening.state is ChallengeState.SCORING

        local = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=lambda _task: StageOutcome(
                ChallengeState.ROLLOUT,
                score=0.7,
                attention_cost=0.2,
                metrics={"lcb": 0.68},
            ),
            now_ms=30,
        )
        assert local is not None and local.state is ChallengeState.ROLLOUT
        assert store.leaderboard()[0].provisional is True
        assert store.get_challenge(identifier).state is ChallengeState.ROLLOUT

        final = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=250,
            settings=settings,
            evaluator=lambda _task: StageOutcome(
                ChallengeState.SUCCEEDED,
                score=0.6,
                attention_cost=0.25,
                metrics={"video_quality": 0.9},
            ),
            now_ms=40,
        )

        assert final is not None and final.state is ChallengeState.SUCCEEDED
        challenge = store.get_challenge(identifier)
        assert challenge.final_score == 0.6
        assert challenge.attempts == 3
        assert store.leaderboard()[0].challenge_id == identifier
        assert store.leaderboard()[0].score_block == 250
        assert store.leaderboard()[0].provisional is False
        assert [event.state for event in store.challenge_events(identifier)][-3:] == [
            ChallengeState.SCORING,
            ChallengeState.ROLLOUT,
            ChallengeState.SUCCEEDED,
        ]


def test_recorded_stage_result_resumes_without_repeating_evaluation(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_screening(store)
        store.transition(identifier, ChallengeState.SCORING, occurred_at_ms=14)
        interrupted = store.claim_next_scoring("first", lease_ms=10, now_ms=20)
        assert interrupted is not None
        store.record_stage_score(
            identifier,
            interrupted.lease_token,
            ChallengeState.SCORING,
            next_state=ChallengeState.PROVISIONAL,
            score=0.8,
            attention_cost=0.1,
            score_block=100,
            metrics={"lower_bound": 0.8},
            occurred_at_ms=21,
        )

        def must_not_repeat(_task):
            raise AssertionError("a durable stage result must not be evaluated again")

        resumed = process_next_scoring(
            store,
            worker_id="second",
            current_finalized_block=250,
            settings=ScoringSettings(lease_ms=10),
            evaluator=must_not_repeat,
            now_ms=30,
        )

        assert resumed is not None and resumed.state is ChallengeState.PROVISIONAL
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.PROVISIONAL
        assert challenge.score_block == 100
        assert challenge.attempts == 2


def test_local_result_can_wait_provisionally_before_rollout(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_screening(store)
        process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=ScoringSettings(),
            evaluator=lambda _task: StageOutcome(
                ChallengeState.SCORING,
                score=0.8,
                attention_cost=0.1,
                metrics={},
            ),
            now_ms=20,
        )
        result = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=110,
            settings=ScoringSettings(),
            evaluator=lambda _task: StageOutcome(
                ChallengeState.PROVISIONAL,
                score=0.7,
                attention_cost=0.2,
                metrics={"lcb": 0.68},
            ),
            now_ms=30,
        )

        assert result is not None and result.state is ChallengeState.PROVISIONAL
        assert store.leaderboard()[0].provisional is True
        assert (
            store.claim_next_scoring("other", lease_ms=100, now_ms=31)
            is None
        )

        assert store.schedule_qualified_rollouts(0.8, occurred_at_ms=39) == 0
        assert store.get_challenge(identifier).state is ChallengeState.PROVISIONAL
        assert store.schedule_qualified_rollouts(0.7, occurred_at_ms=40) == 1
        assert store.schedule_qualified_rollouts(0.7, occurred_at_ms=40) == 0
        rollout = store.claim_next_scoring("audit", lease_ms=100, now_ms=41)
        assert rollout is not None
        assert rollout.state is ChallengeState.ROLLOUT


def test_screening_out_records_zero_terminal_score(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_screening(store)
        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=ScoringSettings(),
            evaluator=lambda _task: StageOutcome(
                ChallengeState.SCREENED_OUT,
                score=0.1,
                attention_cost=0.01,
                metrics={"upper_bound": 0.2},
            ),
            now_ms=20,
        )

        assert outcome is not None and outcome.state is ChallengeState.SCREENED_OUT
        challenge = store.get_challenge(identifier)
        assert challenge.final_score == 0
        assert challenge.error_code == "SCREENING_THRESHOLD"


def test_participant_failure_is_terminal_but_temporary_failure_retries(tmp_path) -> None:
    settings = ScoringSettings(lease_ms=100, retry_base_ms=10, retry_max_ms=100)
    with ChallengeStore(tmp_path / "participant.db") as store:
        identifier = ready_for_screening(store)

        def invalid(_task):
            raise ParticipantScoringError("MODEL_RUNTIME_INVALID", "non-finite output")

        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=invalid,
            now_ms=20,
        )
        assert outcome is not None and outcome.state is ChallengeState.PARTICIPANT_FAILED
        assert store.get_challenge(identifier).error_code == "MODEL_RUNTIME_INVALID"

    with ChallengeStore(tmp_path / "temporary.db") as store:
        identifier = ready_for_screening(store)

        def unavailable(_task):
            raise TemporaryScoringError("GPU_UNAVAILABLE", "worker unavailable")

        deferred = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=unavailable,
            now_ms=20,
        )
        assert deferred is not None and deferred.state is ChallengeState.SYSTEM_DEFERRED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_DEFERRED

        retried = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=lambda _task: StageOutcome(
                ChallengeState.SCORING,
                score=0.8,
                attention_cost=0.1,
                metrics={},
            ),
            now_ms=30,
        )
        assert retried is not None and retried.state is ChallengeState.SCORING
        assert store.get_challenge(identifier).attempts == 2


def test_repeated_infrastructure_failure_stops_without_zeroing_rank(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_screening(store)

        def unavailable(_task):
            raise TemporaryScoringError("GPU_UNAVAILABLE", "worker unavailable")

        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=ScoringSettings(max_attempts=1),
            evaluator=unavailable,
            now_ms=20,
        )

        assert outcome is not None and outcome.state is ChallengeState.SYSTEM_FAILED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_FAILED
        assert store.leaderboard() == []


def test_track_evaluator_runs_seeded_screen_and_exact_local_stages(tmp_path) -> None:
    architecture = SelectorArchitecture(1, 1, 2, ("layer_0_head_0",), "float32")
    model_path = tmp_path / "selector.safetensors"
    save_file(
        {
            name: np.zeros(tensor.shape, dtype=np.float32)
            for name, tensor in architecture.tensor_schema.items()
        },
        model_path,
    )
    track = TrackSpec(
        track_id="test",
        netuid=7,
        track_start_block=1,
        submission_cooldown_blocks=1,
        model=ModelSpec("selector.safetensors", 10_000, architecture),
        base_model=BaseModelSpec("owner/base", "12" * 20),
        cache=CacheSpec("34" * 32, 100_000, "56" * 32, 1_000_000),
        rollout=RolloutSpec("78" * 32, "9a" * 32, 2_000_000),
        scoring=ScoringSpec(
            budgets=(BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0)),
            local_sample_count=2,
            tail_mean_weight=0.8,
            stratum_weights=(("all", 1.0),),
            screening_threshold=0.0,
            rollout_threshold=1.0,
            minimum_frontier_gain=0.001,
            bootstrap_confidence=0.95,
            bootstrap_resamples=20,
            bootstrap_seed=7,
        ),
        digest="ab" * 32,
    )

    screen_cases: list[ScreenCase] = []
    exact_cases: list[ExactCase] = []
    for seed in (1, 2, 3, 4):
        generator = np.random.default_rng(seed)
        q = generator.normal(size=(2, 2, 2)).astype(np.float32)
        k = generator.normal(size=(3, 2, 2)).astype(np.float32)
        v = generator.normal(size=(3, 2, 2)).astype(np.float32)
        valid = np.ones(3, dtype=np.bool_)
        dense = reference_attention(q, k, v, np.ones((2, 3), dtype=np.bool_))
        screen_cases.append(
            ScreenCase(
                "layer_0_head_0",
                "all",
                trip_pool(q),
                trip_pool(k),
                np.full((2, 3), 1 / 3, dtype=np.float32),
                valid,
            )
        )
        exact_cases.append(
            ExactCase("layer_0_head_0", "all", q, k, v, dense, valid, 1e-6)
        )

    evaluator = TrackEvaluator(track, screen_cases, exact_cases)
    screening_task = replace(
        sample_task(),
        model_path=str(model_path),
        file_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    first = evaluator(screening_task)
    second = evaluator(screening_task)
    local = evaluator(replace(screening_task, state=ChallengeState.SCORING))
    other_submission = evaluator(
        replace(
            screening_task,
            submission_id="78" * 32,
            state=ChallengeState.SCORING,
            position=ChainPosition(200, 1, 0),
        )
    )

    with pytest.raises(TemporaryScoringError) as changed:
        evaluator(replace(screening_task, file_sha256="00" * 32))

    assert first == second
    assert changed.value.code == "SCORING_MODEL_CHANGED"
    assert first.next_state is ChallengeState.SCORING
    assert first.metrics["sample_count"] == 4
    assert local.next_state is ChallengeState.PROVISIONAL
    assert local == other_submission
    assert 0 <= local.score <= 1
    assert local.metrics["sample_count"] == 2

    audited = TrackEvaluator(
        track,
        screen_cases,
        exact_cases,
        rollout=lambda _task, _selector, _track: RolloutResult(
            final_score=0.5,
            attention_cost=0.4,
            metrics={"video_quality": 0.8},
        ),
    )(replace(screening_task, state=ChallengeState.ROLLOUT))
    assert audited.next_state is ChallengeState.SUCCEEDED
    assert audited.score == pytest.approx(0.5)
    assert audited.metrics == {"video_quality": 0.8}
