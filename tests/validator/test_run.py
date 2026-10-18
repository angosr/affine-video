import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from safetensors.numpy import save_file

from miner.submit import ModelSubmission
from scoring.model import TensorSpec
from scoring.spec import parse_track_spec, track_spec_dict
from tests.scoring.test_spec import valid_spec
from validator import run as run_module
from validator.admission import AdmissionSettings, AdmissionStatus
from validator.chain import ScanResult
from validator.download import DownloadedModel
from validator.publication import (
    PublicArtifactError,
    S3Repository,
    build_public_artifacts,
    decode_weight_artifact,
    encode_weight_artifact,
)
from validator.queue import ChainPosition, ChallengeState
from validator.run import EvaluationCycleResult, process_weight_cycle
from validator.scoring import ScoringSettings, StageOutcome
from validator.store import ChallengeStore
from validator.tournament import Champion, MatchOutcome, resolve_tournament
from validator.weights import (
    WeightArtifact,
    WeightChainState,
    WeightPlan,
    WeightReceipt,
    plan_champion_weight,
    validate_weight_artifact,
)

from .test_chain import HOTKEY, FakeClient, chain_block
from .test_publication import MemoryS3


async def _complete_cycle(
    client, store, *, admission_settings, scoring_settings, evaluator, worker_id,
    downloader=run_module.download_public_model, model_validator=run_module.validate_model,
    **kwargs,
):
    """Drive one real worker to completion for the bounded one-shot regressions."""

    database = store._connection.execute("PRAGMA database_list").fetchone()[2]
    work = run_module._EvaluationWorker(
        database=database, track_spec_digest=store.track_spec_digest(),
        evaluator_factory=lambda _store: evaluator,
        admission_settings=admission_settings, scoring_settings=scoring_settings,
        worker_id=worker_id, downloader=downloader, model_validator=model_validator,
    )
    try:
        return await run_module.process_cycle(
            client, store, work=work, wait_for_work=True, **kwargs,
        )
    finally:
        work.close()


async def _eventually(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.fixture
def pending_evaluation(tmp_path):
    """Real scan, admission, SQLite leases and publication; only inference is held."""

    track = parse_track_spec(valid_spec())
    path = tmp_path / "selector.safetensors"
    save_file(
        {name: np.ones(spec.shape, dtype=spec.dtype)
         for name, spec in track.model.tensor_schema.items()},
        path, metadata={"hotkey": HOTKEY},
    )
    downloaded = DownloadedModel(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, None, None,
    )
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    client = FakeClient({
        height: chain_block(payload, height) if height == 100 else empty_chain_block(height)
        for height in range(100, 131)
    })
    client._finalized = 101
    entered, release = threading.Event(), threading.Event()
    calls, connections, alerts = [], [], []
    owner_thread = threading.get_ident()
    with ChallengeStore(tmp_path / "validator.db", track_spec_digest=track.digest) as store:
        def factory(worker_store):
            assert worker_store is not store
            assert threading.get_ident() != owner_thread
            # A connection opened on the owner thread would raise ProgrammingError here.
            worker_store.public_snapshot()
            connections.append(worker_store)

            def evaluate(task):
                calls.append(task.challenge_id)
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test did not release inference")
                return StageOutcome(0.7, {"stage": "scoring"})

            return evaluate

        work = run_module._EvaluationWorker(
            database=tmp_path / "validator.db", track_spec_digest=track.digest,
            evaluator_factory=factory,
            admission_settings=AdmissionSettings(
                cache_dir=tmp_path / "cache", max_model_bytes=track.model.max_bytes,
                tensor_schema=track.model.tensor_schema, publication_grace_blocks=1,
            ),
            scoring_settings=ScoringSettings(alert=lambda *args: alerts.append(args)),
            worker_id="single-worker", downloader=lambda *_args, **_kwargs: downloaded,
        )
        transport = MemoryS3()
        publisher = S3Repository(transport, bucket="test", prefix="isolated/heartbeat")
        publisher.verify_conditional_writes()
        try:
            yield SimpleNamespace(
                track=track, store=store, client=client, work=work, entered=entered,
                release=release, calls=calls, connections=connections, alerts=alerts,
                transport=transport, publisher=publisher,
            )
        finally:
            release.set()
            work.close()


async def _pending_cycle(case, **kwargs):
    return await run_module.process_cycle(
        case.client, case.store, netuid=case.track.netuid,
        track_start_block=case.track.track_start_block, work=case.work, **kwargs,
    )


async def _publish_pending(case):
    chain = WeightChainState(
        finalized_block=case.client._finalized,
        uid_by_hotkey={case.track.initial_champion.hotkey: 0, HOTKEY: 1},
        min_allowed_weights=1, max_weight=1.0, weights_rate_limit=1,
        last_update_by_hotkey={},
    )
    artifacts = build_public_artifacts(case.store, case.track, chain, version_key=0)
    assert artifacts is not None
    await case.publisher.publish(artifacts)
    weights = decode_weight_artifact(case.transport.objects["isolated/heartbeat/weights.json"])
    plan = validate_weight_artifact(
        weights, chain, expected_netuid=case.track.netuid,
        expected_track_digest=case.track.digest, expected_mechid=case.track.mechid,
        max_staleness_blocks=1,
    )
    assert plan is not None and plan.uids == (0,)
    assert weights.source_block == chain.finalized_block
    return artifacts


@pytest.mark.parametrize("phase", ["download", "scoring"])
def test_pending_work_keeps_scanning_status_and_cas_weights_fresh(pending_evaluation, phase):
    case = pending_evaluation
    if phase == "download":
        download = case.work.downloader

        def held_download(*args, **kwargs):
            case.entered.set()
            if not case.release.wait(10):
                raise OSError("test did not release download")
            return download(*args, **kwargs)

        case.work.downloader = held_download

    async def execute():
        result = await _pending_cycle(case)
        assert result.scoring is None
        await _eventually(case.entered.is_set)
        first = await _publish_pending(case)
        for height in range(102, 106):
            case.client._finalized = height
            result = await _pending_cycle(case)
            assert result.scoring is None
            assert case.store.chain_cursor()[0] == height
            artifacts = await _publish_pending(case)
            participant = next(
                row for row in json.loads(artifacts.status)["participants"]
                if row["hotkey"] == HOTKEY
            )
            assert participant["state"] == ("FETCHING" if phase == "download" else "SCORING")
            assert participant["score"] is None
            assert case.calls == ([] if phase == "download" else [participant["challenge_id"]])
        # A stale concurrent publisher cannot undo fresh weights while inference runs.
        before = case.transport.objects["isolated/heartbeat/weights.json"]
        with pytest.raises(PublicArtifactError, match="older|rollback|stale"):
            await S3Repository(
                case.transport, bucket="test", prefix="isolated/heartbeat",
            ).publish(first)
        assert case.transport.objects["isolated/heartbeat/weights.json"] == before
        with pytest.raises(RuntimeError, match="already pending"):
            case.work.start(105)
        case.release.set()
        result = await _pending_cycle(case, wait_for_work=True)
        assert result.scoring.state is ChallengeState.SUCCEEDED
        assert case.store.public_snapshot()["challenges"][0]["score_block"] == 105
        assert len(case.calls) == len(case.connections) == 1

    asyncio.run(execute())


@pytest.mark.parametrize("fence", ["close", "abort", "publication"])
def test_inflight_result_obeys_close_abort_and_publication_fences(pending_evaluation, fence):
    case = pending_evaluation

    async def execute():
        await _pending_cycle(case)
        await _eventually(case.entered.is_set)
        case.client._finalized = 105
        if fence == "publication":
            for _ in range(run_module.MAX_PUBLICATION_FAILURES):
                case.store.record_publication_failure()
            options = {}
        else:
            options = {"track_close_block": 103} if fence == "close" else {"track_aborted": True}
        paused = await _pending_cycle(case, **options)
        assert paused.scoring is None
        case.release.set()
        await _eventually(lambda: case.work.done)
        result = await _pending_cycle(case, **options)
        challenge = case.store.public_snapshot()["challenges"][0]
        assert challenge["final_score"] is None
        assert challenge["stages"] == []
        assert len(case.calls) == 1
        if fence == "publication":
            assert result.scoring is None and case.work.busy
            await _publish_pending(case)
            case.store.publication_succeeded()
            result = await _pending_cycle(case)
            assert result.scoring.state is ChallengeState.SUCCEEDED
            assert not case.work.busy
        else:
            assert result.scoring.state is ChallengeState.SYSTEM_DEFERRED
            expected = "TRACK_CLOSE_REACHED" if fence == "close" else "TRACK_ABORTED"
            assert result.scoring.code == expected
            assert not case.work.busy

    asyncio.run(execute())


def test_worker_setup_failure_alerts_and_does_not_poison_next_task(pending_evaluation):
    case = pending_evaluation
    factory = case.work.evaluator_factory

    def fail(_store):
        raise OSError("worker setup failed")

    async def execute():
        case.work.evaluator_factory = fail
        with pytest.raises(OSError, match="worker setup failed"):
            await _pending_cycle(case, wait_for_work=True)
        assert not case.work.busy
        assert case.alerts == [("EVALUATOR_WORKER_FAILURE", {
            "error_type": "OSError", "error": "worker setup failed",
        })]
        await _publish_pending(case)
        case.work.evaluator_factory = factory
        case.release.set()
        result = await _pending_cycle(case, wait_for_work=True)
        assert result.scoring.state is ChallengeState.SUCCEEDED
        assert len(case.calls) == 1

    asyncio.run(execute())


def test_evaluator_loop_publishes_while_worker_owns_its_connection(
    pending_evaluation, monkeypatch, tmp_path,
):
    case = pending_evaluation
    entered, release = threading.Event(), threading.Event()
    admission_gate = threading.Event()
    calls, reports, failures = [], [], []
    snapshots = []
    original_admission = run_module.process_next_admission
    original_scoring = run_module.evaluate_next_scoring
    owner = threading.get_ident()

    class StopLoop(BaseException):
        pass

    class Client(FakeClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    client = Client(case.client._substrate.blocks)
    client._finalized = 102

    async def no_seed(*_args, **_kwargs):
        pass

    async def chain_state(_client, *, netuid, finalized_block):
        assert netuid == case.track.netuid
        return WeightChainState(
            finalized_block=finalized_block,
            uid_by_hotkey={case.track.initial_champion.hotkey: 0, HOTKEY: 1},
            min_allowed_weights=1, max_weight=1.0, weights_rate_limit=1,
            last_update_by_hotkey={},
        )

    def admission(store, **kwargs):
        assert threading.get_ident() != owner
        if not admission_gate.wait(5):
            raise OSError("test did not release admission")
        return original_admission(store, **{**kwargs, "downloader": case.work.downloader})

    def scoring(store, **kwargs):
        assert threading.get_ident() != owner
        assert kwargs["evaluator"].store is store

        def held(task):
            calls.append(task.challenge_id)
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test did not release inference")
            return StageOutcome(0.7, {})

        return original_scoring(store, **{**kwargs, "evaluator": held})

    def report(result):
        if result.cycle_error or result.publication_error:
            failures.append(result)
            release.set()
            raise StopLoop
        published_block = client._finalized
        snapshots.append((
            published_block, case.transport.objects["isolated/heartbeat/status.json"],
        ))
        if not admission_gate.is_set():
            admission_gate.set()
            return
        if not entered.is_set():
            client._finalized += 1
            return
        assert result.published_block == published_block
        status = json.loads(case.transport.objects["isolated/heartbeat/status.json"])
        if next(row for row in status["participants"] if row["hotkey"] == HOTKEY)[
            "state"
        ] != "SCORING":
            # The first snapshot is frozen before admission, then retried unchanged
            # at this height even when the worker has progressed to SCORING.
            client._finalized += 1
            return
        reports.append(result)
        weights = decode_weight_artifact(
            case.transport.objects["isolated/heartbeat/weights.json"],
        )
        assert weights.source_block == published_block
        assert weights.plan.uids == (0,)
        if len(reports) == 3:
            release.set()
            raise StopLoop
        client._finalized += 1

    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: case.track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", no_seed)
    monkeypatch.setattr(run_module, "_refresh_seed_block_hashes", no_seed)
    monkeypatch.setattr(run_module, "read_weight_chain_state", chain_state)
    monkeypatch.setattr(run_module, "process_next_admission", admission)
    monkeypatch.setattr(run_module, "evaluate_next_scoring", scoring)
    monkeypatch.setattr(
        run_module, "webhook_alerter",
        lambda *_args, **_kwargs: lambda *args: case.alerts.append(args),
    )
    (tmp_path / "batches").mkdir()

    async def execute():
        async with asyncio.timeout(5):
            await run_module.run_evaluator(
                track_spec="track.json", database=tmp_path / "loop.db",
                model_cache=tmp_path / "models", exact_cache=tmp_path / "batches",
                initial_model=tmp_path / "selector.safetensors", network="finney",
                archive_endpoints=None, worker_id="single-worker", publisher=case.publisher,
                report=report, poll_seconds=0.001,
                alert_webhook_url="https://alerts.example/veda",
            )

    try:
        with pytest.raises(StopLoop):
            asyncio.run(execute())
    finally:
        admission_gate.set()
        release.set()
    assert not failures
    assert len(calls) == 1
    assert len(reports) == 3
    first_height = [payload for height, payload in snapshots if height == 102]
    assert len(first_height) >= 2 and len(set(first_height)) == 1
    assert next(
        row for row in json.loads(first_height[0])["participants"] if row["hotkey"] == HOTKEY
    )["state"] == "QUEUED"
    # ExitStack joins the in-flight thread before closing the evaluator resources.
    assert not any(thread.name.startswith("veda-evaluation") for thread in threading.enumerate())


def test_release_defaults_target_sn120_initial_state() -> None:
    assert run_module.DEFAULT_NETUID == 120
    assert run_module.DEFAULT_PUBLIC_WEIGHTS_URL.startswith(
        "https://iblosevwkuiootbbgnsy.storage.supabase.co/"
    )
    assert "/tracks/main/weights.json" in run_module.DEFAULT_PUBLIC_WEIGHTS_URL
    assert "/tracks/main/assets/" in run_module.DEFAULT_BASELINE_MANIFEST_URL
    assert run_module.DEFAULT_MECHID == 1


def test_production_evaluator_checks_release_for_custom_repository(monkeypatch) -> None:
    track = SimpleNamespace(track_id="future-track", digest="ab" * 32)
    publisher = SimpleNamespace(release_approval=object())
    calls = []

    def require(candidate, manifest, approval):
        calls.append((candidate, manifest, approval))

    monkeypatch.setattr(run_module, "require_release_approval", require)
    run_module._require_production_release(track, b"manifest", publisher)
    assert calls == [(track, b"manifest", publisher.release_approval)]


def test_production_evaluator_rejects_missing_release_approval(monkeypatch) -> None:
    def reject(*_args):
        raise ValueError("formal publication requires approval")

    monkeypatch.setattr(run_module, "require_release_approval", reject)
    with pytest.raises(ValueError, match="formal evaluator release rejected"):
        run_module._require_production_release(
            SimpleNamespace(track_id="future-track", digest="ab" * 32),
            b"manifest",
            SimpleNamespace(),
        )


def test_ledger_outage_scanning_reserves_uid0_before_any_submission(monkeypatch, tmp_path):
    document = valid_spec()
    document["initial_champion"]["hotkey"] = HOTKEY
    track = parse_track_spec(document)
    blocks = {
        100: chain_block(ModelSubmission("owner/uid-zero", "12" * 20).encode(), 100),
        101: empty_chain_block(101),
    }
    reports = []

    class Client(FakeClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Publisher(S3Repository):
        first_load = True

        async def load(self, candidate):
            if self.first_load:
                self.first_load = False
                raise OSError("public ledger temporarily unavailable")
            return await super().load(candidate)

    async def no_seed(*_args, **_kwargs):
        pass

    async def chain_state(_client, *, netuid, finalized_block):
        return WeightChainState(
            finalized_block=finalized_block, uid_by_hotkey={HOTKEY: 0},
            min_allowed_weights=1, max_weight=1.0, weights_rate_limit=1,
            last_update_by_hotkey={},
        )

    def no_download(*_args, **_kwargs):
        pytest.fail("UID0 must never reach model I/O, including after public recovery")

    transport = MemoryS3()
    publisher = Publisher(transport, bucket="test", prefix="isolated/uid-zero")
    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_args, **_kwargs: Client(blocks))
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", no_seed)
    monkeypatch.setattr(run_module, "_refresh_seed_block_hashes", no_seed)
    monkeypatch.setattr(run_module, "read_weight_chain_state", chain_state)
    (tmp_path / "batches").mkdir()
    database = tmp_path / "validator.db"
    arguments = {
        "track_spec": "track.json", "database": database,
        "model_cache": tmp_path / "models", "exact_cache": tmp_path / "batches",
        "initial_model": tmp_path / "baseline.safetensors", "network": "finney",
        "archive_endpoints": None, "worker_id": "single-worker", "publisher": publisher,
        "once": True, "report": reports.append,
    }
    original_admission = run_module.process_next_admission

    def admission(store, **kwargs):
        return original_admission(store, **{**kwargs, "downloader": no_download})

    monkeypatch.setattr(run_module, "process_next_admission", admission)
    asyncio.run(run_module.run_evaluator(**arguments))
    with ChallengeStore(database, track_spec_digest=track.digest) as store:
        snapshot = store.public_snapshot()
        assert snapshot["challenges"][0]["state"] == ChallengeState.REJECTED_DUPLICATE_HOTKEY
        assert store.effective_challenge_count() == 0
        assert store.chain_cursor()[0] == 101
    assert reports[-1].publication_error == "public ledger temporarily unavailable"
    asyncio.run(run_module.run_evaluator(**arguments))
    assert reports[-1].published_block == 101
    weights = decode_weight_artifact(transport.objects["isolated/uid-zero/weights.json"])
    assert weights.plan.uids == (0,)
    assert weights.match_sequence == 1
    match = json.loads(transport.objects[
        f"isolated/uid-zero/matches/{track.track_id}/000000000001.json"
    ])
    assert match["outcome"] == MatchOutcome.CHALLENGE_FAILED
    assert match["evaluation"]["state"] == ChallengeState.REJECTED_DUPLICATE_HOTKEY


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
        paused = asyncio.run(
            _complete_cycle(client, store, minimum_finalized_block=102, **arguments)
        )
        closed = asyncio.run(_complete_cycle(client, store, track_close_block=100, **arguments))
        aborted = asyncio.run(_complete_cycle(client, store, track_aborted=True, **arguments))
        first = asyncio.run(_complete_cycle(client, store, **arguments))

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
            _complete_cycle(
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
            _complete_cycle(
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
        exact_cycle = asyncio.run(_complete_cycle(client, store, **arguments))
        assert exact_cycle.scoring is not None
        assert exact_cycle.scoring.state is ChallengeState.CONFIRMING

    with ChallengeStore(database, track_spec_digest="ab" * 32) as store:
        confirmation_cycle = asyncio.run(_complete_cycle(client, store, **arguments))
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
            _complete_cycle(
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


def test_publication_gate_survives_restarts_and_republishes_before_work(monkeypatch, tmp_path):
    from scoring.spec import parse_track_spec
    from tests.scoring.test_spec import valid_spec
    from validator.store import MAX_PUBLICATION_FAILURES

    track = parse_track_spec(valid_spec())
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    blocks = {100: chain_block(payload, 100), 101: empty_chain_block(101)}
    events = []
    alerts = []
    fail = True
    load_fail = False
    skip_snapshot = False
    prefix = (("matches/1.json", b"first"), ("matches/2.json", b"second"))

    class Client(FakeClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    class Publisher:
        async def load(self, _track):
            if load_fail:
                raise RuntimeError("public ledger unavailable on restart")
            return SimpleNamespace(matches=(), required_finalized_block=0, aborted=False)

        async def publish(self, artifacts):
            assert artifacts.matches == prefix
            events.append("publish-prefix")
            if fail:
                raise RuntimeError("public storage unavailable after partial upload")

    async def no_op(*_args, **_kwargs):
        pass

    async def chain_state(*_args, **_kwargs):
        return SimpleNamespace(finalized_block=101)

    def work(*_args, **_kwargs):
        events.append("work")
        return None

    def build(*_args, **_kwargs):
        if skip_snapshot:
            return None
        return SimpleNamespace(matches=prefix, source_block=101, batch_groups=(), abort=None)

    monkeypatch.setattr(run_module, "load_track_spec", lambda _: track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_a, **_k: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_a, **_k: Client(blocks))
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", no_op)
    monkeypatch.setattr(run_module, "_refresh_seed_block_hashes", no_op)
    monkeypatch.setattr(run_module, "read_weight_chain_state", chain_state)
    monkeypatch.setattr(run_module, "process_next_admission", work)
    monkeypatch.setattr(run_module, "evaluate_next_scoring", work)
    monkeypatch.setattr(run_module, "build_public_artifacts", build)
    monkeypatch.setattr(
        run_module, "webhook_alerter", lambda *_a, **_k: lambda *a: alerts.append(a)
    )
    cache = tmp_path / "exact"
    cache.mkdir()
    database = tmp_path / "validator.db"

    def restart():
        asyncio.run(run_module.run_evaluator(
            track_spec="track.json", database=database, model_cache=tmp_path / "models",
            exact_cache=cache, initial_model=tmp_path / "baseline", network="test",
            archive_endpoints=None, worker_id="test", publisher=Publisher(), once=True,
            alert_webhook_url="https://alerts.example/test",
        ))

    for count in range(1, MAX_PUBLICATION_FAILURES + 1):
        restart()
        with ChallengeStore(database) as store:
            assert store.publication_failures() == count
    assert events.count("work") == 2 * MAX_PUBLICATION_FAILURES
    events.clear()
    restart()
    assert events == ["publish-prefix"]
    assert [row[1]["consecutive_failures"] for row in alerts] == [1, MAX_PUBLICATION_FAILURES]
    assert alerts[-1][1]["blocked"] is True
    load_fail = True
    events.clear()
    restart()
    assert events == []
    with ChallengeStore(database) as store:
        assert store.chain_cursor()[0] == 101
        assert store.publication_failures() == MAX_PUBLICATION_FAILURES
    load_fail = False

    # A cycle with no publication is not a successful recovery.
    skip_snapshot = True
    restart()
    with ChallengeStore(database) as store:
        assert store.publication_failures() == MAX_PUBLICATION_FAILURES
        assert store.chain_cursor()[0] == 101
        assert len(store.public_snapshot()["challenges"]) == 1
    skip_snapshot = False
    fail = False
    events.clear()
    restart()
    assert events == ["publish-prefix"]
    with ChallengeStore(database) as store:
        assert store.publication_failures() == 0
    restart()
    assert events == ["publish-prefix", "work", "work", "publish-prefix"]
