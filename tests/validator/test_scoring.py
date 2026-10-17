import hashlib
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.evaluate import reference_attention
from scoring.exact import ExactCase
from scoring.generator import generate_sources, source_digest
from scoring.model import validate_model
from scoring.selector import SelectorArchitecture
from scoring.spec import (
    VEDA_EVALUATION_FILE_SHA256,
    VEDA_FILE_SHA256,
    VEDA_FILENAME,
    VEDA_MODEL_DIGEST,
    VEDA_REPO_ID,
    VEDA_REVISION,
    BaseModelSpec,
    BudgetSpec,
    CacheSpec,
    InitialChampionSpec,
    ModelSpec,
    ScoringSpec,
    TrackSpec,
)
from validator import scoring as scoring_module
from validator.queue import ChainPosition, ChallengeState
from validator.scoring import (
    IsolatedEvaluator,
    PairedTrackEvaluator,
    ParticipantScoringError,
    ScoringSettings,
    StageOutcome,
    TemporaryScoringError,
    TrackEvaluator,
    evaluate_isolated,
    process_next_scoring,
    verify_initial_champion,
)
from validator.store import ChallengeStore, LeasedChallenge

from .test_store import submission_at


def test_paired_evaluator_scores_both_models_on_post_submission_batch(tmp_path) -> None:
    architecture = SelectorArchitecture(1, 1, 2, "float32")
    tensors = {
        name: np.zeros(spec.shape, dtype=np.float32)
        for name, spec in architecture.tensor_schema.items()
    }
    baseline_path = tmp_path / "baseline.safetensors"
    challenger_path = tmp_path / "challenger.safetensors"
    save_file(tensors, baseline_path)
    save_file(tensors, challenger_path, metadata={"hotkey": "hk1"})
    baseline = validate_model(
        baseline_path,
        schema=architecture.tensor_schema,
        max_bytes=10_000,
    )
    challenger = validate_model(
        challenger_path,
        schema=architecture.tensor_schema,
        max_bytes=10_000,
        expected_hotkey="hk1",
    )
    track = TrackSpec(
        track_id="paired",
        netuid=7,
        mechid=0,
        track_start_block=1,
        publication_grace_blocks=1,
        initial_champion=InitialChampionSpec(
            0,
            "genesis-hotkey",
            VEDA_REPO_ID,
            VEDA_REVISION,
            VEDA_FILENAME,
            VEDA_FILE_SHA256,
            baseline.file_sha256,
            baseline.model_digest,
            0.5,
        ),
        model=ModelSpec("selector.safetensors", 10_000, architecture),
        base_model=BaseModelSpec("owner/base", "12" * 20),
        cache=CacheSpec(1_000_000, 1, 1, "veda-source-v1"),
        scoring=ScoringSpec(
            budgets=(BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0)),
            sample_count=1,
            tail_mean_weight=0.8,
            stratum_weights=(("all", 1.0),),
            minimum_frontier_gain=0.001,
            bootstrap_confidence=0.95,
            bootstrap_resamples=20,
            bootstrap_seed=7,
        ),
        digest="ab" * 32,
    )
    cache_dir = tmp_path / "batches"
    cache_dir.mkdir()
    arrays = {}
    cases = []
    generated = generate_sources(
        track_spec_digest=track.digest,
        seed_block_hash="0x" + "ab" * 32,
        strata=("all",),
        samples_per_stratum=1,
    )
    for index, seed in enumerate((1,)):
        generator = np.random.default_rng(seed)
        q = generator.normal(size=(2, 2, 2)).astype(np.float32)
        k = generator.normal(size=(3, 2, 2)).astype(np.float32)
        v = generator.normal(size=(3, 2, 2)).astype(np.float32)
        dense = reference_attention(q, k, v, np.ones((2, 3), dtype=np.bool_))
        prefix = f"case_{index:06d}"
        arrays.update(
            {
                f"{prefix}_q": q,
                f"{prefix}_k": k,
                f"{prefix}_v": v,
                f"{prefix}_dense": dense,
                f"{prefix}_valid_keys": np.ones(3, dtype=np.bool_),
                f"{prefix}_required_keys": np.array(
                    [[True, False, False], [False, True, False]], dtype=np.bool_
                ),
            }
        )
        cases.append(
            {
                "route": "layer_0_head_0",
                "stratum": "all",
                "sample_id": generated[0].sample_id,
                "epsilon": 1e-6,
            }
        )
    metadata = np.frombuffer(
        json.dumps(
            {
                "format": "veda-exact-cache",
                "generated_block": 101,
                "source_digest": source_digest(generated),
                "seed_block_hash": "0x" + "ab" * 32,
                "cases": cases,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        dtype=np.uint8,
    )
    temporary_batch = cache_dir / "temporary.npz"
    np.savez(temporary_batch, metadata=metadata, **arrays)
    batch_digest = hashlib.sha256(temporary_batch.read_bytes()).hexdigest()
    temporary_batch.rename(cache_dir / f"101-{batch_digest}.npz")

    with ChallengeStore(tmp_path / "validator.db", track_spec_digest=track.digest) as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=1)
        lease = store.claim_next_admission(
            "admission", lease_ms=1_000, ready_through_block=100, now_ms=2
        )
        assert lease is not None
        store.record_downloaded_model(
            identifier,
            lease.lease_token,
            model_path=challenger_path,
            file_sha256=challenger.file_sha256,
            size_bytes=challenger.size_bytes,
            occurred_at_ms=3,
        )
        store.register_validated_model(
            identifier,
            lease.lease_token,
            model_digest=challenger.model_digest,
            occurred_at_ms=4,
        )
        evaluator = PairedTrackEvaluator(
            track,
            store,
            cache_dir,
            tmp_path / "models",
            baseline_path,
            "https://public.example/models/paired",
            30,
        )
        scoring = process_next_scoring(
            store,
            worker_id="scoring",
            current_finalized_block=101,
            settings=ScoringSettings(lease_ms=60_000),
            evaluator=evaluator,
            now_ms=5,
        )
        assert scoring is not None and scoring.state is ChallengeState.SUCCEEDED
        evaluated = store.public_challenge(identifier)

    assert evaluated["final_score"] == pytest.approx(evaluated["incumbent_score"])
    assert evaluated["evaluation_batch"]["generated_block"] == 101
    assert evaluated["evaluation_batch"]["source_digest"] == source_digest(generated)


def test_initial_champion_file_and_digest_are_verified(monkeypatch, tmp_path) -> None:
    initial = SimpleNamespace(
        hotkey="burn-hotkey",
        repo_id=VEDA_REPO_ID,
        revision=VEDA_REVISION,
        evaluation_file_sha256=VEDA_EVALUATION_FILE_SHA256,
        model_digest=VEDA_MODEL_DIGEST,
        score=0.7,
    )
    track = SimpleNamespace(
        initial_champion=initial,
        model=SimpleNamespace(tensor_schema={}, max_bytes=1),
        track_start_block=100,
    )
    validated = SimpleNamespace(
        file_sha256=VEDA_EVALUATION_FILE_SHA256,
        model_digest=VEDA_MODEL_DIGEST,
    )
    monkeypatch.setattr(scoring_module, "validate_model", lambda *_args, **_kwargs: validated)

    outcome = verify_initial_champion(
        track,
        model_path=tmp_path / "veda.safetensors",
    )
    assert outcome.model_digest == VEDA_MODEL_DIGEST

    validated.model_digest = "00" * 32
    with pytest.raises(ValueError, match="digest"):
        verify_initial_champion(
            track,
            model_path=tmp_path / "veda.safetensors",
        )


def isolated_success(_task: LeasedChallenge) -> StageOutcome:
    return StageOutcome(
        score=0.75,
        metrics={"worker": "child"},
    )


def isolated_sleep(task: LeasedChallenge) -> StageOutcome:
    time.sleep(1)
    return isolated_success(task)


def isolated_participant_failure(_task: LeasedChallenge) -> StageOutcome:
    raise ParticipantScoringError("MODEL_RUNTIME_INVALID", "invalid output")


def isolated_memory_failure(_task: LeasedChallenge) -> StageOutcome:
    raise MemoryError


def sample_task() -> LeasedChallenge:
    return LeasedChallenge(
        challenge_id="challenge",
        submission_id="56" * 32,
        state=ChallengeState.SCORING,
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


def ready_for_scoring(store: ChallengeStore) -> str:
    identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=10)
    store.transition(identifier, ChallengeState.FETCHING, occurred_at_ms=11)
    store.transition(identifier, ChallengeState.VALIDATING, occurred_at_ms=12)
    store.transition(identifier, ChallengeState.SCORING, occurred_at_ms=13)
    return identifier


def test_isolated_evaluator_returns_only_structured_result() -> None:
    result = IsolatedEvaluator(isolated_success, timeout_seconds=5)(sample_task())

    assert result.score == 0.75
    assert result.metrics == {"worker": "child"}


def test_isolated_evaluator_terminates_timeout_and_preserves_error_class() -> None:
    with pytest.raises(TemporaryScoringError) as timeout:
        evaluate_isolated(isolated_sleep, sample_task(), timeout_seconds=0.1)
    assert timeout.value.code == "SCORING_TIMEOUT"

    with pytest.raises(ParticipantScoringError) as invalid:
        evaluate_isolated(isolated_participant_failure, sample_task(), timeout_seconds=5)
    assert invalid.value.code == "MODEL_RUNTIME_INVALID"

    with pytest.raises(TemporaryScoringError) as memory:
        evaluate_isolated(isolated_memory_failure, sample_task(), timeout_seconds=5)
    assert memory.value.code == "SCORING_MEMORY_LIMIT"


def test_authoritative_score_finishes_one_challenge(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        result = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=250,
            settings=ScoringSettings(),
            evaluator=lambda _task: StageOutcome(
                score=0.6,
                metrics={"budget_scores": [0.4, 0.6, 0.8]},
            ),
            now_ms=20,
        )

        assert result is not None and result.state is ChallengeState.SUCCEEDED
        challenge = store.get_challenge(identifier)
        assert challenge.final_score == pytest.approx(0.6)
        assert challenge.score_block == 250
        assert store.leaderboard()[0].challenge_id == identifier
        assert [event.state for event in store.challenge_events(identifier)][-1] is (
            ChallengeState.SUCCEEDED
        )


def test_recorded_score_resumes_without_repeating_evaluation(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        interrupted = store.claim_next_scoring("first", lease_ms=10, now_ms=20)
        assert interrupted is not None
        store.record_stage_score(
            identifier,
            interrupted.lease_token,
            ChallengeState.SCORING,
            score=0.8,
            score_block=100,
            metrics={"lower_bound": 0.8},
            occurred_at_ms=21,
        )

        def must_not_repeat(_task):
            raise AssertionError("a durable score must not be evaluated again")

        resumed = process_next_scoring(
            store,
            worker_id="second",
            current_finalized_block=250,
            settings=ScoringSettings(lease_ms=10),
            evaluator=must_not_repeat,
            now_ms=30,
        )

        assert resumed is not None and resumed.state is ChallengeState.SUCCEEDED
        assert store.get_challenge(identifier).final_score == pytest.approx(0.8)
        assert store.get_challenge(identifier).score_block == 100


def test_participant_failure_is_terminal_but_temporary_failure_retries(tmp_path) -> None:
    settings = ScoringSettings(lease_ms=100, retry_base_ms=10, retry_max_ms=100)
    with ChallengeStore(tmp_path / "participant.db") as store:
        identifier = ready_for_scoring(store)

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
        identifier = ready_for_scoring(store)

        def unavailable(_task):
            raise TemporaryScoringError("SCORING_UNAVAILABLE", "worker unavailable")

        deferred = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=unavailable,
            now_ms=20,
        )
        assert deferred is not None and deferred.state is ChallengeState.SYSTEM_DEFERRED

        retried = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=isolated_success,
            now_ms=30,
        )
        assert retried is not None and retried.state is ChallengeState.SUCCEEDED
        assert store.get_challenge(identifier).attempts == 2


def test_repeated_infrastructure_failure_remains_retryable(tmp_path) -> None:
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)

        def unavailable(_task):
            raise TemporaryScoringError("SCORING_UNAVAILABLE", "worker unavailable")

        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=ScoringSettings(),
            evaluator=unavailable,
            now_ms=20,
        )

        assert outcome is not None and outcome.state is ChallengeState.SYSTEM_DEFERRED
        assert store.get_challenge(identifier).score_block is None


def test_track_evaluator_produces_deterministic_authoritative_score(tmp_path) -> None:
    architecture = SelectorArchitecture(1, 1, 2, "float32")
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
        mechid=0,
        track_start_block=1,
        publication_grace_blocks=1,
        initial_champion=InitialChampionSpec(
            0,
            "genesis-hotkey",
            VEDA_REPO_ID,
            VEDA_REVISION,
            VEDA_FILENAME,
            VEDA_FILE_SHA256,
            VEDA_EVALUATION_FILE_SHA256,
            VEDA_MODEL_DIGEST,
            0.7,
        ),
        model=ModelSpec("selector.safetensors", 10_000, architecture),
        base_model=BaseModelSpec("owner/base", "12" * 20),
        cache=CacheSpec(1_000_000, 1, 1, "veda-source-v1"),
        scoring=ScoringSpec(
            budgets=(BudgetSpec(1, 1.0), BudgetSpec(2, 2.0), BudgetSpec(3, 4.0)),
            sample_count=4,
            tail_mean_weight=0.8,
            stratum_weights=(("all", 1.0),),
            minimum_frontier_gain=0.001,
            bootstrap_confidence=0.95,
            bootstrap_resamples=20,
            bootstrap_seed=7,
        ),
        digest="ab" * 32,
    )

    exact_cases = []
    for seed in (1, 2, 3, 4):
        generator = np.random.default_rng(seed)
        q = generator.normal(size=(2, 2, 2)).astype(np.float32)
        k = generator.normal(size=(3, 2, 2)).astype(np.float32)
        v = generator.normal(size=(3, 2, 2)).astype(np.float32)
        valid = np.ones(3, dtype=np.bool_)
        dense = reference_attention(q, k, v, np.ones((2, 3), dtype=np.bool_))
        required = np.array([[True, False, False], [False, True, False]], dtype=np.bool_)
        exact_cases.append(
            ExactCase("layer_0_head_0", "all", q, k, v, dense, valid, required, 1e-6, "sample")
        )

    evaluator = TrackEvaluator(track, exact_cases)
    task = replace(
        sample_task(),
        model_path=str(model_path),
        file_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    first = evaluator(task)
    second = evaluator(task)
    other_submission = evaluator(
        replace(task, submission_id="78" * 32, position=ChainPosition(200, 1, 0))
    )

    with pytest.raises(TemporaryScoringError) as changed:
        evaluator(replace(task, file_sha256="00" * 32))

    assert changed.value.code == "SCORING_MODEL_CHANGED"
    assert first == second == other_submission
    assert 0 <= first.score <= 1
    assert first.metrics["sample_count"] == 4
