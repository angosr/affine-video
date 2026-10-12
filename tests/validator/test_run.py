import asyncio
import hashlib

import numpy as np
from safetensors.numpy import save_file

from miner.submit import ModelSubmission
from scoring.model import TensorSpec
from validator.admission import AdmissionSettings, AdmissionStatus
from validator.download import DownloadedModel
from validator.queue import ChallengeState
from validator.run import process_cycle
from validator.scoring import ScoringSettings, StageOutcome
from validator.store import ChallengeStore

from .test_chain import FakeClient, chain_block


def test_cycle_connects_finalized_scan_admission_and_scoring(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file({"selector.weight": np.ones((2, 3), dtype=np.float32)}, model_path)
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient({100: chain_block(payload, 100)})

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
            "seed_delay_blocks": 0,
            "admission_settings": AdmissionSettings(
                cache_dir=tmp_path / "cache",
                max_model_bytes=10_000,
                tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
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
