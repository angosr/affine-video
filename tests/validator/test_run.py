import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from safetensors.numpy import save_file

from miner.submit import ModelSubmission
from scoring.model import TensorSpec
from validator import run as run_module
from validator.admission import AdmissionSettings, AdmissionStatus
from validator.download import DownloadedModel
from validator.publication import decode_weight_artifact, encode_weight_artifact
from validator.queue import ChainPosition, ChallengeState
from validator.run import process_cycle, process_weight_cycle
from validator.scoring import ScoringSettings, StageOutcome
from validator.store import ChallengeStore
from validator.tournament import Champion, resolve_tournament
from validator.weights import (
    WeightArtifact,
    WeightChainState,
    WeightPlan,
    WeightReceipt,
    plan_champion_weight,
)

from .test_chain import HOTKEY, FakeClient, chain_block


def test_release_defaults_target_sn120_initial_state() -> None:
    assert run_module.DEFAULT_NETUID == 120
    assert "/tracks/sn120/weights.json" in run_module.DEFAULT_PUBLIC_WEIGHTS_URL
    assert run_module.DEFAULT_TRACK_SPEC_DIGEST != "0" * 64


def test_initial_publisher_emits_current_uid_zero_weight(monkeypatch) -> None:
    chain = WeightChainState(
        321,
        {run_module.INITIAL_CHAMPION_HOTKEY: 0},
        1,
        1.0,
        100,
        {},
    )
    published = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Publisher:
        async def publish_initial_state(self, **kwargs):
            published.update(kwargs)

    async def read_state(_client, *, netuid):
        assert netuid == 120
        return chain

    monkeypatch.setattr(run_module.bt, "Client", lambda _network: Client())
    monkeypatch.setattr(run_module, "read_weight_chain_state", read_state)
    asyncio.run(run_module.run_initial_publisher(publisher=Publisher(), once=True))

    weights = decode_weight_artifact(published["weights"])
    status = json.loads(published["status"])
    assert weights.source_block == 321
    assert weights.plan == WeightPlan(
        321,
        (run_module.INITIAL_CHAMPION_HOTKEY,),
        (0,),
        (1.0,),
    )
    assert status["participants"][0]["champion"] is True
    assert [match["challenger"] for match in status["system_matches"]] == [
        "MeanPool control",
        "Shuffled-heads control",
    ]
    assert len(published["system_matches"]) == 2


def empty_chain_block(block_height: int) -> tuple[str, dict, list[dict]]:
    return "0x" + f"{block_height:064x}", {"extrinsics": []}, []


def test_independent_validator_fetches_checks_and_submits_public_weights(monkeypatch) -> None:
    artifact = WeightArtifact(
        "track",
        "ab" * 32,
        7,
        2,
        9,
        120,
        WeightPlan(120, ("alice",), (4,), (1.0,)),
    )
    chain = WeightChainState(
        120,
        {"alice": 4, "validator": 8},
        1,
        1.0,
        10,
        {"alice": 0, "validator": 100},
    )
    submitted = {}

    async def read_state(_client, *, netuid: int):
        assert netuid == 7
        return chain

    async def submit(plan, **kwargs):
        submitted["plan"] = plan
        submitted.update(kwargs)
        return WeightReceipt("0xabc", "120-1", "finalized")

    monkeypatch.setattr(run_module, "read_weight_chain_state", read_state)
    monkeypatch.setattr(run_module, "submit_weights", submit)

    async def execute():
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, content=encode_weight_artifact(artifact))
        )
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await process_weight_cycle(
                object(),
                http_client,
                SimpleNamespace(name="wallet"),
                validator_hotkey="validator",
                public_weights_url="https://public.example/weights.json",
                netuid=7,
                track_spec_digest="ab" * 32,
                max_staleness_blocks=20,
            )

    result = asyncio.run(execute())
    assert result.status == "SUBMITTED"
    assert result.receipt is not None and result.receipt.extrinsic_id == "120-1"
    assert submitted["plan"] == artifact.plan
    assert submitted["mechid"] == 2
    assert submitted["version_key"] == 9


def test_cycle_connects_finalized_scan_admission_and_scoring(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file({"selector.weight": np.ones((2, 3), dtype=np.float32)}, model_path)
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient({100: chain_block(payload, 100), 101: empty_chain_block(101)})

    def download(*_args, **_kwargs):
        return DownloadedModel(model_path, file_digest, model_path.stat().st_size, None, None)

    def evaluate(_task):
        return StageOutcome(0.7, {"stage": "scoring"})

    with ChallengeStore(tmp_path / "validator.db", track_spec_digest="ab" * 32) as store:
        arguments = {
            "netuid": 7,
            "track_start_block": 100,
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
        paused = asyncio.run(process_cycle(client, store, minimum_finalized_block=102, **arguments))
        first = asyncio.run(process_cycle(client, store, **arguments))

        assert paused.scan.submissions_found == 1
        assert paused.admission is paused.scoring is None
        assert first.scan.blocks_scanned == 0
        assert first.admission is not None
        assert first.admission.status is AdmissionStatus.ACCEPTED
        assert first.scoring is not None
        assert first.scoring.state is ChallengeState.SUCCEEDED
        assert store.leaderboard()[0].final_score == pytest.approx(0.7)


def test_full_challenge_lifecycle_survives_restart_and_produces_weight_plan(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file({"selector.weight": np.ones((2, 3), dtype=np.float32)}, model_path)
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient({100: chain_block(payload, 100), 101: empty_chain_block(101)})
    database = tmp_path / "validator.db"
    downloads = 0

    def download(*_args, **_kwargs):
        nonlocal downloads
        downloads += 1
        return DownloadedModel(model_path, file_digest, model_path.stat().st_size, None, None)

    def evaluate(_task):
        return StageOutcome(0.9, {"stage": "scoring"})

    arguments = {
        "netuid": 7,
        "track_start_block": 100,
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
    with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
        cycle = asyncio.run(process_cycle(client, store, **arguments))
        assert cycle.scoring is not None

    with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
        snapshot = store.public_snapshot()
        challenge = snapshot["challenges"][0]
        tournament = resolve_tournament(
            snapshot["challenges"],
            track_id="track",
            track_spec_digest="ab" * 32,
            netuid=7,
            minimum_gain=0.001,
            initial_champion=Champion(
                "genesis",
                "genesis-hotkey",
                "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
                "76f202874608115408d73280be9531c4ab888242",
                0.5,
                ChainPosition(100, 0, 0),
                uid=0,
                source="VEDA_BASELINE",
            ),
        )
        plan = plan_champion_weight(
            tournament.champion,
            uid_by_hotkey={HOTKEY: 4},
            current_block=101,
            min_allowed_weights=1,
            max_weight=1.0,
        )

    assert downloads == 1
    assert cycle.scoring.state is ChallengeState.SUCCEEDED
    assert challenge["state"] == ChallengeState.SUCCEEDED
    assert challenge["final_score"] == pytest.approx(0.9)
    assert [stage["stage"] for stage in challenge["stages"]] == [ChallengeState.SCORING]
    assert [event["state"] for event in challenge["events"]] == [
        ChallengeState.QUEUED,
        ChallengeState.FETCHING,
        ChallengeState.VALIDATING,
        ChallengeState.SCORING,
        ChallengeState.SUCCEEDED,
    ]
    assert plan is not None
    assert plan.hotkeys == (HOTKEY,)
    assert plan.uids == (4,)
    assert plan.weights == (1.0,)
