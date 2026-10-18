import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from scoring.cache import write_exact_cache_batch
from scoring.exact import ExactCase, canonical_dense_reference
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
    FullVideoSpec,
    InitialChampionSpec,
    ModelSpec,
    ScoringSpec,
    SourceCommitmentSpec,
    StepWindowSpec,
    TrackSpec,
    TrajectoryScheduleSpec,
)
from validator import scoring as scoring_module
from validator.production.batches import ArchivedStage, ReadyBatchSource
from validator.production.retries import PrivateRetryStore
from validator.production.storage import PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState
from validator.scoring import (
    IsolatedEvaluator,
    PairedTrackEvaluator,
    ParticipantScoringError,
    ScoringSettings,
    StageOutcome,
    TemporaryScoringError,
    evaluate_isolated,
    process_next_scoring,
    verify_initial_champion,
)
from validator.store import ChallengeStore, LeasedChallenge

from .production.test_storage import MemoryS3
from .test_store import submission_at


def scoring_spec(*, routes: int, rollouts: int = 1) -> ScoringSpec:
    return ScoringSpec(
        budgets=(
            BudgetSpec(1, 0, 1.0),
            BudgetSpec(2, 0, 2.0),
            BudgetSpec(3, 0, 4.0),
        ),
        sampled_route_count=routes,
        rollouts_per_stratum=rollouts,
        sampled_step_count=1,
        query_tiles_per_case=2,
        step_selection_algorithm="sha256-window-without-replacement-v1",
        query_tile_selection_algorithm="sha256-segment-quota-v1",
        trajectory_schedule=TrajectoryScheduleSpec(
            "test",
            (1.0, 0.0),
            (StepWindowSpec("all", (0,), 1.0),),
        ),
        tail_mean_weight=0.8,
        stratum_weights=(("all", 1.0),),
        minimum_frontier_gain=0.001,
        season_false_promotion_alpha=0.05,
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )


def cache_spec() -> CacheSpec:
    return CacheSpec(
        1_000_000,
        1,
        1,
        "veda-private-rollout-v1",
        "veda-exact-cache-v2",
        "float16",
        "float32",
        "float32",
    )


def video_spec() -> FullVideoSpec:
    return FullVideoSpec(0.001, "test-v1", "ef" * 32, 0.4, 0.3, 0.3, 0.01, 2.0)


@pytest.mark.parametrize("cache_loss", ["none", "deleted", "corrupt"])
@pytest.mark.parametrize("stage", ["exact", "confirmation"])
def test_paired_evaluator_restores_challenger_for_both_stages(
    tmp_path, monkeypatch, cache_loss, stage
) -> None:
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
        maximum_generation_batches=10,
        maximum_effective_challenges=1,
        track_close_grace_blocks=10,
        source_pool_merkle_root="cd" * 32,
        source_commitments=(SourceCommitmentSpec("cd" * 32, "production", "all"),),
        artifact_manifest_sha256="ef" * 32,
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
        cache=cache_spec(),
        scoring=scoring_spec(routes=1, rollouts=4),
        full_video=video_spec(),
        digest="ab" * 32,
    )
    cache_dir = tmp_path / "batches"
    cache_dir.mkdir()
    generator = np.random.default_rng(1)
    q = generator.normal(size=(2, 128, 2)).astype(np.float16)
    k = generator.normal(size=(3, 128, 2)).astype(np.float16)
    v = generator.normal(size=(3, 128, 2)).astype(np.float16)
    query_valid_counts = np.full(2, 128, dtype=np.int16)
    key_valid_counts = np.full(3, 128, dtype=np.int16)
    dense = canonical_dense_reference(
        q,
        k,
        v,
        query_valid_counts,
        key_valid_counts,
    )
    case = ExactCase(
        route="layer_0_head_0",
        stratum="all",
        rollout_id="rollout-0",
        source_id="source-0",
        step_index=0,
        step_window="all",
        scheduler_timestep=1.0,
        sigma=1.0,
        tile_plan_hash="12" * 32,
        query_tile_ids=(0, 1),
        q=q,
        k=k,
        v=v,
        dense=dense,
        query_valid_counts=query_valid_counts,
        key_valid_counts=key_valid_counts,
        key_segments=np.zeros(3, dtype=np.int8),
        required_keys=np.array([[True, False, False], [False, True, False]], dtype=np.bool_),
        epsilon=1e-6,
    )
    cases = tuple(
        replace(
            case,
            rollout_id=f"rollout-{index}",
            source_id=f"source-{index}",
        )
        for index in range(4)
    )
    batch = write_exact_cache_batch(
        cache_dir,
        cases,
        generated_block=101,
        seed_block_hash="0x" + "ab" * 32,
        plan_digest="23" * 32,
        track_spec_digest=track.digest,
        artifact_manifest_sha256=track.artifact_manifest_sha256,
        source_pool_merkle_root=track.source_pool_merkle_root,
        source_proofs=tuple({"source_id": f"source-{index}"} for index in range(4)),
        dense_references=tuple({"video_sha256": f"{index + 1:02x}" * 32} for index in range(4)),
        sample_cache={},
    )

    with ChallengeStore(tmp_path / "validator.db", track_spec_digest=track.digest) as store:
        identifier, _ = store.add_submission(submission_at(100), occurred_at_ms=1)
        lease = store.claim_next_admission(
            "admission", lease_ms=1_000, ready_through_block=100, now_ms=2
        )
        assert lease is not None
        assert store.reserve_season_slot(
            identifier,
            lease.lease_token,
            maximum_effective_challenges=track.maximum_effective_challenges,
        )
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
            downloader=lambda *_args, **_kwargs: pytest.fail("archive recovery contacted HF"),
        )
        source = ReadyBatchSource(
            track=track,
            objects=PrivateObjectStore(MemoryS3(), bucket="private", prefix="test"),
            cache_dir=tmp_path / "restored",
        )
        source.archive_model(lease, challenger_path, challenger)
        if cache_loss == "deleted":
            challenger_path.unlink()
        elif cache_loss == "corrupt":
            challenger_path.write_bytes(b"broken")
        evaluator.batch_source = source
        monkeypatch.setattr(source, "is_ready", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(source, "exact_batch", lambda _identifier: batch)
        if stage == "confirmation":
            confirmation = SimpleNamespace(
                plan=SimpleNamespace(
                    generation_block=101, plan_digest=batch.plan_digest,
                    block_hash=batch.seed_block_hash,
                ),
                exact=batch,
            )
            monkeypatch.setattr(source, "confirmation_batch", lambda _identifier: confirmation)
            monkeypatch.setattr(
                store, "stage_score", lambda *_args: SimpleNamespace(score_block=100)
            )
            calls = []

            def run_confirmation(**kwargs):
                recovered = validate_model(
                    kwargs["challenger_path"], schema=architecture.tensor_schema,
                    max_bytes=10_000, expected_hotkey="hk1",
                )
                assert recovered == challenger
                calls.append(kwargs)
                return SimpleNamespace(
                    challenger_score=0.5, incumbent_score=0.5, gain_lower_bound=0.0,
                    challenger_cells=(1, 2, 3, 4),
                )

            evaluator.confirmation_pool = SimpleNamespace(run=run_confirmation)
            result = evaluator(replace(
                lease, state=ChallengeState.CONFIRMING, model_path=str(challenger_path),
                file_sha256=challenger.file_sha256,
            ))
            assert result.score == 0.5
            assert len(calls) == 1
            return
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
    assert evaluated["evaluation_batch"]["plan_digest"] == batch.plan_digest


def test_promoted_champion_corrupt_cache_restores_for_both_scoring_stages(
    tmp_path, monkeypatch
) -> None:
    architecture = SelectorArchitecture(1, 1, 2, "float32")
    tensors = {
        name: np.zeros(spec.shape, dtype=np.float32)
        for name, spec in architecture.tensor_schema.items()
    }
    archived_path = tmp_path / "archived.safetensors"
    local_path = tmp_path / "corrupt.safetensors"
    save_file(tensors, archived_path, metadata={"hotkey": "champion-hotkey"})
    validated = validate_model(
        archived_path,
        schema=architecture.tensor_schema,
        max_bytes=10_000,
        expected_hotkey="champion-hotkey",
    )
    local_path.write_bytes(b"broken")
    track = SimpleNamespace(
        digest="ab" * 32,
        model=SimpleNamespace(tensor_schema=architecture.tensor_schema, max_bytes=10_000),
    )
    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket="private", prefix="tracks/test")
    objects.put_content(
        archived_path,
        sha256=validated.file_sha256,
        object_name=f"models/{validated.file_sha256}.safetensors",
    )
    source = ReadyBatchSource(
        track=track,
        objects=objects,
        cache_dir=tmp_path / "restored",
    )
    evaluator = PairedTrackEvaluator(
        track=track,
        store=SimpleNamespace(model_path=lambda _challenge_id: str(local_path)),
        cache_dir=tmp_path / "batches",
        model_cache=tmp_path / "models",
        initial_model_path=archived_path,
        champion_archive_url="https://public.example/models/test",
        timeout_seconds=30,
        downloader=lambda *_args, **_kwargs: pytest.fail("private restore contacted HF"),
        batch_source=source,
    )
    incumbent = SimpleNamespace(
        challenge_id="12" * 32,
        hotkey="champion-hotkey",
        repo_id="owner/model",
        revision="34" * 20,
        position=ChainPosition(100, 0, 0),
    )
    challenges = (
        {
            "challenge_id": incumbent.challenge_id,
            "file_sha256": validated.file_sha256,
            "model_digest": validated.model_digest,
            "model_size_bytes": validated.size_bytes,
        },
    )
    monkeypatch.setattr(
        scoring_module,
        "download_chunked_file",
        lambda *_args, **_kwargs: pytest.fail("private restore contacted public storage"),
    )

    exact_incumbent = evaluator._incumbent_task(incumbent, challenges)
    client.objects.clear()
    confirmation_incumbent = evaluator._incumbent_task(incumbent, challenges)

    assert exact_incumbent.model_path == confirmation_incumbent.model_path
    assert exact_incumbent.model_path != str(local_path)
    assert exact_incumbent.file_sha256 == validated.file_sha256
    assert validate_model(
        exact_incumbent.model_path,
        schema=architecture.tensor_schema,
        max_bytes=10_000,
        expected_hotkey="champion-hotkey",
    ) == validated


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


def test_exact_winner_requires_a_new_full_video_batch_before_completion(tmp_path) -> None:
    batches = {
        ChallengeState.SCORING: (120, "12" * 32, True, 0.02),
        ChallengeState.CONFIRMING: (140, "34" * 32, False, 0.01),
    }

    def evaluate(task):
        block, digest, requires_confirmation, gain = batches[task.state]
        return StageOutcome(
            0.8,
            {
                "cache_generated_block": block,
                "cache_plan_digest": digest,
                "cache_seed_block_hash": "0x" + "ab" * 32,
                "cache_sha256": digest,
                "incumbent_score": 0.7,
                "paired_gain_lower_bound": gain,
                "requires_confirmation": requires_confirmation,
            },
        )

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        exact = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=120,
            settings=ScoringSettings(),
            evaluator=evaluate,
            now_ms=20,
        )
        confirmation = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=140,
            settings=ScoringSettings(),
            evaluator=evaluate,
            now_ms=21,
        )
        challenge = store.public_challenge(identifier)

    assert exact is not None and exact.state is ChallengeState.CONFIRMING
    assert confirmation is not None and confirmation.state is ChallengeState.SUCCEEDED
    assert [stage["stage"] for stage in challenge["stages"]] == [
        ChallengeState.SCORING,
        ChallengeState.CONFIRMING,
    ]
    assert [stage["evaluation_batch"]["generated_block"] for stage in challenge["stages"]] == [
        120,
        140,
    ]
    assert challenge["evaluation_batch"]["generated_block"] == 140


def test_zero_quality_confirmation_finishes_without_infrastructure_retry(tmp_path) -> None:
    def evaluate(task):
        if task.state is ChallengeState.SCORING:
            return StageOutcome(
                0.8,
                {
                    "cache_generated_block": 120,
                    "cache_plan_digest": "12" * 32,
                    "cache_seed_block_hash": "0x" + "ab" * 32,
                    "cache_sha256": "34" * 32,
                    "incumbent_score": 0.7,
                    "paired_gain_lower_bound": 0.02,
                    "requires_confirmation": True,
                },
            )
        return StageOutcome(
            0.0,
            {
                "cache_generated_block": 140,
                "cache_plan_digest": "56" * 32,
                "cache_seed_block_hash": "0x" + "cd" * 32,
                "cache_sha256": "78" * 32,
                "incumbent_score": 0.7,
                "paired_gain_lower_bound": -0.7,
            },
        )

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        exact = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=120,
            settings=ScoringSettings(),
            evaluator=evaluate,
            now_ms=20,
        )
        confirmation = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=140,
            settings=ScoringSettings(),
            evaluator=evaluate,
            now_ms=21,
        )
        challenge = store.get_challenge(identifier)

    assert exact is not None and exact.state is ChallengeState.CONFIRMING
    assert confirmation is not None and confirmation.state is ChallengeState.SUCCEEDED
    assert challenge.state is ChallengeState.SUCCEEDED
    assert challenge.final_score == 0.0
    assert challenge.attempts == 2


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


def test_private_stage_archive_restores_without_repeating_evaluation(tmp_path) -> None:
    archived = ArchivedStage(
        0.8,
        220,
        {
            "cache_generated_block": 120,
            "requires_confirmation": True,
        },
    )
    rewrites = []

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=300,
            settings=ScoringSettings(
                restore_stage=lambda _task: archived,
                archive_stage=lambda task, result, block: rewrites.append(
                    (task.challenge_id, result.score, block)
                ),
            ),
            evaluator=lambda _task: pytest.fail("archived stage was evaluated again"),
            now_ms=20,
        )

        assert outcome is not None and outcome.state is ChallengeState.CONFIRMING
        recorded = store.stage_score(identifier, ChallengeState.SCORING)
        assert recorded is not None and recorded.score_block == 220
        assert rewrites == [(identifier, 0.8, 220)]


def test_private_stage_archive_waits_for_a_chain_head_that_covers_its_decision(tmp_path) -> None:
    archived = ArchivedStage(
        0.8,
        220,
        {"cache_generated_block": 120, "requires_confirmation": True},
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=200,
            settings=ScoringSettings(
                lease_ms=10,
                retry_base_ms=10,
                retry_max_ms=10,
                restore_stage=lambda _task: archived,
            ),
            evaluator=lambda _task: pytest.fail("archived stage was evaluated again"),
            now_ms=20,
        )

        assert outcome is not None
        assert outcome.state is ChallengeState.SYSTEM_DEFERRED
        assert outcome.code == "SCORING_ARCHIVE_AHEAD"
        assert store.stage_score(identifier, ChallengeState.SCORING) is None


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
    alerts = []
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)

        def unavailable(_task):
            raise TemporaryScoringError("SCORING_UNAVAILABLE", "worker unavailable")

        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=ScoringSettings(alert=lambda event, details: alerts.append((event, details))),
            evaluator=unavailable,
            now_ms=20,
        )

        assert outcome is not None and outcome.state is ChallengeState.SYSTEM_DEFERRED
        assert store.get_challenge(identifier).score_block is None
        assert alerts == [
            (
                "SCORING_RETRY_STARTED",
                {
                    "challenge_id": identifier,
                    "attempts": 1,
                    "code": "SCORING_UNAVAILABLE",
                    "stage": "SCORING",
                },
            )
        ]


def test_future_generation_boundary_is_not_claimed_as_a_scoring_attempt(tmp_path) -> None:
    class FutureBatch:
        def ready_through_block(self, _head):
            return 99

        def __call__(self, _task):
            raise AssertionError("future evaluation data must not be requested")

    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)

        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=ScoringSettings(),
            evaluator=FutureBatch(),
            now_ms=20,
        )

        assert outcome is None
        challenge = store.get_challenge(identifier)
        assert challenge.state is ChallengeState.SCORING
        assert challenge.attempts == 0


def test_normal_producer_lag_waits_without_attempt_or_alert(tmp_path) -> None:
    alerts = []

    class ProducerBatch:
        ready = False

        def ready_through_block(self, _head):
            return 100

        def batch_ready(self, _task):
            return self.ready

        def __call__(self, _task):
            return StageOutcome(0.7, {"stage": "scoring"})

    evaluator = ProducerBatch()
    settings = ScoringSettings(
        retry_base_ms=10,
        retry_max_ms=10,
        alert=lambda event, details: alerts.append((event, details)),
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        waiting = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=evaluator,
            now_ms=20,
        )
        assert waiting is not None
        assert waiting.code == "EVALUATION_BATCH_PENDING"
        assert store.get_challenge(identifier).attempts == 0
        assert store.public_challenge(identifier)["attempts"] == 0
        assert alerts == []

        evaluator.ready = True
        completed = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=evaluator,
            now_ms=31,
        )
        assert completed is not None and completed.state is ChallengeState.SUCCEEDED


def test_overdue_batch_uses_bounded_retries_and_push_alerts(tmp_path) -> None:
    alerts = []

    class MissingBatch:
        def ready_through_block(self, _head):
            return 100

        def batch_ready(self, _task):
            return False

        def batch_deadline_block(self, _task):
            return 100

        def __call__(self, _task):
            raise AssertionError("an unavailable batch must not be evaluated")

    settings = ScoringSettings(
        retry_base_ms=10,
        retry_max_ms=10,
        max_attempts=2,
        alert=lambda event, details: alerts.append((event, details)),
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)
        first = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=101,
            settings=settings,
            evaluator=MissingBatch(),
            now_ms=20,
        )
        assert first is not None and first.state is ChallengeState.SYSTEM_DEFERRED
        assert first.code == "EVALUATION_BATCH_TIMEOUT"
        assert store.get_challenge(identifier).attempts == 1

        final = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=102,
            settings=settings,
            evaluator=MissingBatch(),
            now_ms=30,
        )
        assert final is not None and final.state is ChallengeState.SYSTEM_BLOCKED
        assert final.code == "EVALUATION_BATCH_TIMEOUT"
        assert [event for event, _details in alerts] == [
            "SCORING_RETRY_STARTED",
            "SCORING_RETRY_EXHAUSTED",
        ]


def test_scoring_retry_ceiling_blocks_and_pushes_an_alert(tmp_path) -> None:
    alerts = []
    settings = ScoringSettings(
        max_attempts=1,
        alert=lambda event, details: alerts.append((event, details)),
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        identifier = ready_for_scoring(store)

        def unavailable(_task):
            raise TemporaryScoringError("SCORING_UNAVAILABLE", "worker unavailable")

        outcome = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=unavailable,
            now_ms=20,
        )

        assert outcome is not None and outcome.state is ChallengeState.SYSTEM_BLOCKED
        assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_BLOCKED
        assert store.claim_next_scoring("other", lease_ms=100, now_ms=1_000_000) is None
        assert alerts == [
            (
                "SCORING_RETRY_EXHAUSTED",
                {
                    "challenge_id": identifier,
                    "attempts": 1,
                    "code": "SCORING_UNAVAILABLE",
                    "stage": "SCORING",
                },
            )
        ]

        assert store.resume_blocked(identifier, occurred_at_ms=30) is ChallengeState.SCORING
        resumed = store.get_challenge(identifier)
        assert resumed.state is ChallengeState.SCORING
        assert resumed.attempts == 1

        success = process_next_scoring(
            store,
            worker_id="worker",
            current_finalized_block=100,
            settings=settings,
            evaluator=lambda _task: StageOutcome(0.7, {"stage": "scoring"}),
            now_ms=40,
        )
        assert success is not None and success.state is ChallengeState.SUCCEEDED


def test_scoring_retry_ceiling_survives_total_local_state_loss(tmp_path) -> None:
    retries = PrivateRetryStore(
        PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc"),
        track_spec_digest="ab" * 32,
    )
    calls = 0

    def unavailable(_task):
        nonlocal calls
        calls += 1
        raise TemporaryScoringError("SCORING_UNAVAILABLE", "worker unavailable")

    settings = ScoringSettings(
        max_attempts=2,
        retry_base_ms=10,
        retry_max_ms=10,
        restore_retry=retries.restore,
        record_retry=retries.record_failure,
    )
    outcomes = []
    for index, now in enumerate((20, 30, 1_000), start=1):
        with ChallengeStore(tmp_path / f"validator-{index}.db") as store:
            identifier = ready_for_scoring(store)
            outcomes.append(
                process_next_scoring(
                    store,
                    worker_id="worker",
                    current_finalized_block=100,
                    settings=settings,
                    evaluator=unavailable,
                    now_ms=now,
                )
            )
            if index >= 2:
                assert store.get_challenge(identifier).state is ChallengeState.SYSTEM_BLOCKED

    assert [outcome.state for outcome in outcomes if outcome is not None] == [
        ChallengeState.SYSTEM_DEFERRED,
        ChallengeState.SYSTEM_BLOCKED,
        ChallengeState.SYSTEM_BLOCKED,
    ]
    assert calls == 2
