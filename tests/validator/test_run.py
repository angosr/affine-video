import asyncio
import hashlib

import numpy as np
import pytest
from safetensors.numpy import save_file

from miner.submit import ModelSubmission
from scoring.model import TensorSpec
from validator.admission import AdmissionSettings, AdmissionStatus
from validator.download import DownloadedModel
from validator.queue import ChallengeState
from validator.run import process_cycle, run_validator
from validator.scoring import ScoringSettings, StageOutcome
from validator.store import ChallengeStore
from validator.weights import plan_weights

from .test_chain import HOTKEY, FakeClient, chain_block


def empty_chain_block(block_height: int) -> tuple[str, dict, list[dict]]:
    return "0x" + f"{block_height:064x}", {"extrinsics": []}, []


def test_cycle_connects_finalized_scan_admission_and_scoring(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file({"selector.weight": np.ones((2, 3), dtype=np.float32)}, model_path)
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient(
        {100: chain_block(payload, 100), 101: empty_chain_block(101)}
    )

    def download(*_args, **_kwargs):
        return DownloadedModel(model_path, file_digest, model_path.stat().st_size, None, None)

    def evaluate(task):
        if task.state is ChallengeState.SCREENING:
            return StageOutcome(ChallengeState.SCORING, 0.8, 0.2, {"stage": "screen"})
        return StageOutcome(ChallengeState.PROVISIONAL, 0.7, 0.3, {"stage": "local"})

    with ChallengeStore(tmp_path / "validator.db", track_spec_digest="ab" * 32) as store:
        arguments = {
            "netuid": 7,
            "track_start_block": 100,
            "cooldown_blocks": 10,
            "admission_settings": AdmissionSettings(
                cache_dir=tmp_path / "cache",
                max_model_bytes=10_000,
                tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
                publication_grace_blocks=1,
            ),
            "scoring_settings": ScoringSettings(),
            "evaluator": evaluate,
            "worker_id": "validator-a",
            "downloader": download,
        }
        first = asyncio.run(process_cycle(client, store, **arguments))
        second = asyncio.run(process_cycle(client, store, **arguments))

        assert first.scan.submissions_found == 1
        assert first.admission is not None
        assert first.admission.status is AdmissionStatus.ACCEPTED
        assert first.scoring is not None
        assert first.scoring.state is ChallengeState.SCORING
        assert second.scan.blocks_scanned == 0
        assert second.admission is None
        assert second.scoring is not None
        assert second.scoring.state is ChallengeState.PROVISIONAL
        assert store.leaderboard()[0].final_score == 0.7
        assert store.leaderboard()[0].provisional is True


def test_full_challenge_lifecycle_survives_restart_and_produces_weight_plan(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file({"selector.weight": np.ones((2, 3), dtype=np.float32)}, model_path)
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient(
        {100: chain_block(payload, 100), 101: empty_chain_block(101)}
    )
    database = tmp_path / "validator.db"
    downloads = 0

    def download(*_args, **_kwargs):
        nonlocal downloads
        downloads += 1
        return DownloadedModel(model_path, file_digest, model_path.stat().st_size, None, None)

    def evaluate(task):
        outcomes = {
            ChallengeState.SCREENING: StageOutcome(
                ChallengeState.SCORING, 0.8, 0.2, {"stage": "screen"}
            ),
            ChallengeState.SCORING: StageOutcome(
                ChallengeState.ROLLOUT, 0.7, 0.3, {"stage": "local"}
            ),
            ChallengeState.ROLLOUT: StageOutcome(
                ChallengeState.SUCCEEDED, 0.9, 0.25, {"stage": "rollout"}
            ),
        }
        return outcomes[task.state]

    arguments = {
        "netuid": 7,
        "track_start_block": 100,
        "cooldown_blocks": 10,
        "admission_settings": AdmissionSettings(
            cache_dir=tmp_path / "cache",
            max_model_bytes=10_000,
            tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
            publication_grace_blocks=1,
        ),
        "scoring_settings": ScoringSettings(),
        "evaluator": evaluate,
        "worker_id": "validator-a",
        "downloader": download,
    }
    outcomes = []
    for _stage in range(3):
        with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
            cycle = asyncio.run(process_cycle(client, store, **arguments))
            assert cycle.scoring is not None
            outcomes.append(cycle.scoring.state)

    with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
        snapshot = store.public_snapshot()
        challenge = snapshot["challenges"][0]
        plan = plan_weights(
            store.reward_entries(),
            uid_by_hotkey={HOTKEY: 4},
            current_block=101,
            min_allowed_weights=1,
            max_weight=1.0,
            minimum_frontier_gain=0.001,
        )

    assert downloads == 1
    assert outcomes == [
        ChallengeState.SCORING,
        ChallengeState.ROLLOUT,
        ChallengeState.SUCCEEDED,
    ]
    assert challenge["state"] == ChallengeState.SUCCEEDED
    assert challenge["final_score"] == pytest.approx(0.9)
    assert challenge["provisional"] is False
    assert [stage["stage"] for stage in challenge["stages"]] == [
        ChallengeState.SCREENING,
        ChallengeState.SCORING,
        ChallengeState.ROLLOUT,
    ]
    assert [event["state"] for event in challenge["events"]] == [
        ChallengeState.QUEUED,
        ChallengeState.FETCHING,
        ChallengeState.VALIDATING,
        ChallengeState.SCREENING,
        ChallengeState.SCORING,
        ChallengeState.ROLLOUT,
        ChallengeState.SUCCEEDED,
    ]
    assert plan is not None
    assert plan.hotkeys == (HOTKEY,)
    assert plan.uids == (4,)
    assert plan.weights == (1.0,)


def test_weight_submission_requires_authoritative_rollout_configuration(tmp_path) -> None:
    arguments = {
        "track_spec": tmp_path / "missing-track.json",
        "database": tmp_path / "validator.db",
        "model_cache": tmp_path / "models",
        "screen_cache": tmp_path / "screen.npz",
        "exact_cache": tmp_path / "exact.npz",
        "rollout_runner": None,
        "rollout_panel": None,
        "network": "test",
        "archive_endpoints": None,
        "worker_id": "validator",
        "submit_chain_weights": True,
        "once": True,
    }

    with pytest.raises(ValueError, match="require an authoritative rollout"):
        asyncio.run(run_validator(**arguments))
