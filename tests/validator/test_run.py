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
from scoring.spec import track_spec_dict
from validator import run as run_module
from validator.admission import AdmissionSettings, AdmissionStatus
from validator.chain import ScanResult
from validator.download import DownloadedModel
from validator.publication import encode_weight_artifact
from validator.queue import ChainPosition, ChallengeState
from validator.run import EvaluationCycleResult, process_cycle, process_weight_cycle
from validator.scoring import ScoringSettings, StageOutcome
from validator.store import ChallengeStore
from validator.tournament import Champion, MatchOutcome, resolve_tournament
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
    assert run_module.DEFAULT_PUBLIC_WEIGHTS_URL.startswith(
        "https://iblosevwkuiootbbgnsy.storage.supabase.co/"
    )
    assert "/tracks/main/weights.json" in run_module.DEFAULT_PUBLIC_WEIGHTS_URL
    assert "/tracks/main/assets/" in run_module.DEFAULT_BASELINE_MANIFEST_URL
    assert run_module.DEFAULT_MECHID == 1


def test_evaluator_reconnects_after_unexpected_cycle_failure(monkeypatch, tmp_path) -> None:
    class StopLoop(BaseException):
        pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def finalized_block(self):
            return 100

    class Publisher:
        async def load(self, _track):
            return SimpleNamespace(matches=(), required_finalized_block=0, aborted=False)

    track = SimpleNamespace(
        digest="ab" * 32,
        track_id="track",
        netuid=120,
        track_start_block=100,
        publication_grace_blocks=1,
        maximum_effective_challenges=10,
        maximum_generation_batches=10,
        model=SimpleNamespace(
            max_bytes=1_000,
            tensor_schema={},
            selector=SimpleNamespace(head_dim=1, routes=("route",)),
        ),
        cache=SimpleNamespace(
            exact_max_bytes=1_000,
            minimum_generation_delay_blocks=1,
            generation_interval_blocks=1,
        ),
        scoring=SimpleNamespace(
            budgets=[SimpleNamespace(target_blocks=1)],
            stratum_weight_map={"stratum": 1.0},
            case_count=1,
        ),
        initial_champion=SimpleNamespace(hotkey="uid0-hotkey"),
        last_submission_block=1_000,
        track_close_block=2_000,
    )
    attempts = 0
    clients = 0
    reports = []

    def client_factory(*_args, **_kwargs):
        nonlocal clients
        clients += 1
        return Client()

    async def process(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("chain disconnected")
        return EvaluationCycleResult(100, ScanResult(None, None, 0, 0), None, None)

    async def seed(*_args, **_kwargs):
        return None

    async def no_sleep(_seconds):
        return None

    def report(result):
        reports.append(result)
        if result.cycle_error is None:
            raise StopLoop

    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", client_factory)
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", seed)
    monkeypatch.setattr(run_module, "_refresh_seed_block_hashes", seed)
    monkeypatch.setattr(run_module, "process_cycle", process)
    monkeypatch.setattr(run_module.asyncio, "sleep", no_sleep)
    (tmp_path / "exact.npz").mkdir()

    with pytest.raises(StopLoop):
        asyncio.run(
            run_module.run_evaluator(
                track_spec="track.json",
                database=tmp_path / "validator.db",
                model_cache=tmp_path / "models",
                exact_cache=tmp_path / "exact.npz",
                initial_model=tmp_path / "baseline.safetensors",
                network="finney",
                archive_endpoints=None,
                worker_id="worker",
                publisher=Publisher(),
                alert_webhook_url="https://alerts.example/veda",
                report=report,
            )
        )

    assert clients == 2
    assert reports[0].cycle_error == "chain disconnected"
    assert reports[1].cycle_error is None


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
        assert netuid == 120
        return chain

    async def submit(plan, **kwargs):
        submitted["plan"] = plan
        submitted.update(kwargs)
        return WeightReceipt("0xabc", "120-1", "finalized")

    monkeypatch.setattr(run_module, "read_weight_chain_state", read_state)
    monkeypatch.setattr(run_module, "submit_weights", submit)

    async def execute():
        track = run_module.load_track_spec("track.json")
        artifact = WeightArtifact(
            track.track_id,
            track.digest,
            track.netuid,
            track.mechid,
            9,
            120,
            WeightPlan(120, ("alice",), (4,), (1.0,)),
        )
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=(
                    json.dumps(
                        {
                            "track_spec_digest": track.digest,
                            "spec": track_spec_dict(track),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    if request.url.path.endswith("track.json")
                    else encode_weight_artifact(artifact)
                ),
            )
        )
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await process_weight_cycle(
                object(),
                http_client,
                SimpleNamespace(name="wallet"),
                validator_hotkey="validator",
                public_weights_url="https://public.example/weights.json",
                max_staleness_blocks=20,
            )

    result = asyncio.run(execute())
    assert result.status == "SUBMITTED"
    assert result.receipt is not None and result.receipt.extrinsic_id == "120-1"
    assert submitted["plan"] == artifact.plan
    assert submitted["mechid"] == 1
    assert submitted["version_key"] == 9


def test_independent_validator_rejects_weights_for_another_track() -> None:
    track = run_module.load_track_spec("track.json")
    artifact = WeightArtifact(
        "another-track",
        track.digest,
        track.netuid,
        track.mechid,
        9,
        120,
        WeightPlan(120, ("alice",), (4,), (1.0,)),
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=(
                json.dumps(
                    {
                        "track_spec_digest": track.digest,
                        "spec": track_spec_dict(track),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                if request.url.path.endswith("track.json")
                else encode_weight_artifact(artifact)
            ),
        )
    )

    async def execute():
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await process_weight_cycle(
                object(),
                http_client,
                SimpleNamespace(name="wallet"),
                validator_hotkey="validator",
                public_weights_url="https://public.example/weights.json",
                max_staleness_blocks=20,
            )

    with pytest.raises(ValueError, match="different track"):
        asyncio.run(execute())


def test_cycle_connects_finalized_scan_admission_and_scoring(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file(
        {"selector.weight": np.ones((2, 3), dtype=np.float32)},
        model_path,
        metadata={"hotkey": HOTKEY},
    )
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
        closed = asyncio.run(process_cycle(client, store, track_close_block=100, **arguments))
        aborted = asyncio.run(process_cycle(client, store, track_aborted=True, **arguments))
        first = asyncio.run(process_cycle(client, store, **arguments))

        assert paused.scan.submissions_found == 1
        assert paused.admission is paused.scoring is None
        assert closed.admission is closed.scoring is None
        assert aborted.admission is aborted.scoring is None
        assert first.scan.blocks_scanned == 0
        assert first.admission is not None
        assert first.admission.status is AdmissionStatus.ACCEPTED
        assert first.scoring is not None
        assert first.scoring.state is ChallengeState.SUCCEEDED
        assert store.leaderboard()[0].final_score == pytest.approx(0.7)


def test_closed_track_rejects_scanned_late_work_and_ignores_new_commitments(tmp_path) -> None:
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    blocks = {height: empty_chain_block(height) for height in range(100, 122)}
    blocks[100] = chain_block(payload, 100)
    blocks[121] = chain_block(payload, 121)
    client = FakeClient(blocks)
    external_calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal external_calls
        external_calls += 1
        raise AssertionError("closed tracks cannot start admission or scoring")

    with ChallengeStore(tmp_path / "validator.db", track_spec_digest="ab" * 32) as store:
        cycle = asyncio.run(
            process_cycle(
                client,
                store,
                netuid=7,
                track_start_block=100,
                track_close_block=120,
                admission_settings=AdmissionSettings(
                    cache_dir=tmp_path / "cache",
                    max_model_bytes=10_000,
                    tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
                    publication_grace_blocks=1,
                    last_submission_block=99,
                ),
                scoring_settings=ScoringSettings(),
                evaluator=forbidden,
                worker_id="validator-a",
                downloader=forbidden,
            )
        )

        challenges = store.public_snapshot()["challenges"]
        assert cycle.finalized_block == 121
        assert cycle.admission is cycle.scoring is None
        assert external_calls == 0
        assert len(challenges) == 1
        assert challenges[0]["block_height"] == 100
        assert challenges[0]["state"] == ChallengeState.REJECTED_TRACK_CLOSED
        assert challenges[0]["error_code"] == "TRACK_CLOSED"


def test_cycle_does_not_commit_a_stage_after_the_track_closes(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file(
        {"selector.weight": np.ones((2, 3), dtype=np.float32)},
        model_path,
        metadata={"hotkey": HOTKEY},
    )
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient({100: chain_block(payload, 100), 101: empty_chain_block(101)})
    finalized_reads = 0

    async def advancing_finalized_block():
        nonlocal finalized_reads
        finalized_reads += 1
        return 150 if finalized_reads >= 3 else 101

    client.finalized_block = advancing_finalized_block
    with ChallengeStore(tmp_path / "validator.db", track_spec_digest="ab" * 32) as store:
        cycle = asyncio.run(
            process_cycle(
                client,
                store,
                netuid=7,
                track_start_block=100,
                track_close_block=120,
                admission_settings=AdmissionSettings(
                    cache_dir=tmp_path / "cache",
                    max_model_bytes=10_000,
                    tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
                    publication_grace_blocks=1,
                ),
                scoring_settings=ScoringSettings(),
                evaluator=lambda _task: StageOutcome(0.9, {"stage": "scoring"}),
                worker_id="validator-a",
                downloader=lambda *_args, **_kwargs: DownloadedModel(
                    model_path,
                    file_digest,
                    model_path.stat().st_size,
                    None,
                    None,
                ),
            )
        )

        assert cycle.finalized_block == 150
        assert cycle.scoring is not None
        assert cycle.scoring.state is ChallengeState.SYSTEM_DEFERRED
        assert cycle.scoring.code == "TRACK_CLOSE_REACHED"
        challenge = store.public_snapshot()["challenges"][0]
        assert challenge["state"] == ChallengeState.SYSTEM_DEFERRED
        assert challenge["final_score"] is None
        assert challenge["stages"] == []


def test_full_challenge_lifecycle_survives_restart_and_produces_weight_plan(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file(
        {"selector.weight": np.ones((2, 3), dtype=np.float32)},
        model_path,
        metadata={"hotkey": HOTKEY},
    )
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    blocks = {height: empty_chain_block(height) for height in range(101, 301)}
    blocks[100] = chain_block(payload, 100)
    client = FakeClient(blocks)
    finalized_reads = 0

    async def advancing_finalized_block():
        nonlocal finalized_reads
        finalized_reads += 1
        return (101, 101, 150, 250, 250, 300)[min(finalized_reads - 1, 5)]

    client.finalized_block = advancing_finalized_block
    database = tmp_path / "validator.db"
    downloads = 0

    def download(*_args, **_kwargs):
        nonlocal downloads
        downloads += 1
        return DownloadedModel(model_path, file_digest, model_path.stat().st_size, None, None)

    def evaluate(task):
        if task.state is ChallengeState.SCORING:
            return StageOutcome(
                0.9,
                {
                    "stage": "scoring",
                    "incumbent_score": 0.5,
                    "paired_gain_lower_bound": 0.4,
                    "cache_sha256": "de" * 32,
                    "cache_generated_block": 101,
                    "cache_plan_digest": "ef" * 32,
                    "cache_seed_block_hash": "0x" + "ab" * 32,
                    "requires_confirmation": True,
                },
            )
        assert task.state is ChallengeState.CONFIRMING
        return StageOutcome(
            0.8,
            {
                "stage": "confirming",
                "incumbent_score": 0.5,
                "paired_gain_lower_bound": 0.3,
                "cache_sha256": "12" * 32,
                "cache_generated_block": 200,
                "cache_plan_digest": "34" * 32,
                "cache_seed_block_hash": "0x" + "cd" * 32,
            },
        )

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
        "max_scan_blocks": 300,
    }
    with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
        exact_cycle = asyncio.run(process_cycle(client, store, **arguments))
        assert exact_cycle.scoring is not None
        assert exact_cycle.scoring.state is ChallengeState.CONFIRMING

    with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
        confirmation_cycle = asyncio.run(process_cycle(client, store, **arguments))
        assert confirmation_cycle.scoring is not None
        snapshot = store.public_snapshot()
        challenge = snapshot["challenges"][0]
        tournament = resolve_tournament(
            snapshot["challenges"],
            track_id="track",
            track_spec_digest="ab" * 32,
            netuid=7,
            minimum_gain=0.1,
            minimum_exact_gain=0.1,
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
    assert confirmation_cycle.scoring.state is ChallengeState.SUCCEEDED
    assert challenge["state"] == ChallengeState.SUCCEEDED
    assert challenge["final_score"] == pytest.approx(0.8)
    assert challenge["score_block"] == 300
    assert confirmation_cycle.finalized_block == 300
    assert [stage["stage"] for stage in challenge["stages"]] == [
        ChallengeState.SCORING,
        ChallengeState.CONFIRMING,
    ]
    assert [event["state"] for event in challenge["events"]] == [
        ChallengeState.QUEUED,
        ChallengeState.FETCHING,
        ChallengeState.VALIDATING,
        ChallengeState.SCORING,
        ChallengeState.CONFIRMING,
        ChallengeState.SUCCEEDED,
    ]
    assert tournament.matches[0].outcome is MatchOutcome.CHALLENGER_WON
    assert tournament.champion.hotkey == HOTKEY
    assert plan is not None
    assert plan.hotkeys == (HOTKEY,)
    assert plan.uids == (4,)
    assert plan.weights == (1.0,)


def test_cycle_releases_scoring_lease_when_fresh_finalized_head_is_unavailable(tmp_path) -> None:
    model_path = tmp_path / "selector.safetensors"
    save_file(
        {"selector.weight": np.ones((2, 3), dtype=np.float32)},
        model_path,
        metadata={"hotkey": HOTKEY},
    )
    file_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient({100: chain_block(payload, 100), 101: empty_chain_block(101)})
    finalized_reads = 0

    async def failing_finalized_block():
        nonlocal finalized_reads
        finalized_reads += 1
        if finalized_reads >= 3:
            raise RuntimeError("chain head unavailable")
        return 101

    client.finalized_block = failing_finalized_block

    with ChallengeStore(tmp_path / "validator.db", track_spec_digest="ab" * 32) as store:
        cycle = asyncio.run(
            process_cycle(
                client,
                store,
                netuid=7,
                track_start_block=100,
                admission_settings=AdmissionSettings(
                    cache_dir=tmp_path / "cache",
                    max_model_bytes=10_000,
                    tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
                    publication_grace_blocks=1,
                ),
                scoring_settings=ScoringSettings(retry_base_ms=10, retry_max_ms=10),
                evaluator=lambda _task: StageOutcome(0.7, {"stage": "scoring"}),
                worker_id="validator-a",
                downloader=lambda *_args, **_kwargs: DownloadedModel(
                    model_path,
                    file_digest,
                    model_path.stat().st_size,
                    None,
                    None,
                ),
            )
        )

        assert cycle.scoring is not None
        assert cycle.scoring.state is ChallengeState.SYSTEM_DEFERRED
        assert cycle.scoring.code == "FINALIZED_HEAD_UNAVAILABLE"
        challenge = store.get_challenge(cycle.scoring.challenge_id)
        assert challenge.state is ChallengeState.SYSTEM_DEFERRED
        assert store.claim_next_scoring("recovery", lease_ms=100, now_ms=10**18) is not None
