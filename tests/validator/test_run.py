import asyncio
import hashlib
import json
import threading
from dataclasses import replace
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
from validator.production.commitment import Artifact, ArtifactManifest, SourceLeaf, SourcePool
from validator.publication import (
    HISTORICAL_MATCHES_PREFIX,
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
from validator.tournament import Champion, MatchOutcome, genesis_champion, resolve_tournament
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
from .test_store import submission_at
from .test_tournament import challenge as scored_challenge


@pytest.mark.parametrize("failure", ["protection", "pruning"])
def test_optional_cleanup_failure_does_not_block_durable_disclosure(failure, caplog):
    from tests.validator.production.test_storage import MemoryS3 as PrivateMemoryS3
    from validator.production.storage import PrivateObjectStore

    track = parse_track_spec(valid_spec())
    objects = PrivateObjectStore(PrivateMemoryS3(), bucket="test", prefix="test")
    digest = "ab" * 32
    calls = []

    def protection(*_args):
        calls.append("protection")
        if failure == "protection":
            raise OSError("chain Timestamp unavailable")
        return frozenset({"cd" * 32})

    def prune(protected):
        assert protected == frozenset({"cd" * 32})
        calls.append("prune")
        raise OSError("optional filesystem reclamation unavailable")

    source = SimpleNamespace(
        track=track, objects=objects, protected_plan_digests=protection, prune_disclosed=prune,
        due_plan_digests=lambda _now: (digest,),
        public_disclosure=lambda *_a, **_k: "original-bound-disclosure",
        record_disclosed=lambda key: calls.append(("record", key)),
    )

    class Publisher:
        async def publish_disclosure(self, disclosure, **kwargs):
            assert disclosure == "original-bound-disclosure" and kwargs["track"] == track
            calls.append("publish")

    assert asyncio.run(run_module._publish_due_batch(
        source, Publisher(), finalized_block=120, challenges=[],
    )) == digest
    assert calls == (["protection"] if failure == "protection" else ["protection", "prune"]) + [
        "publish", ("record", digest),
    ]
    assert "optional disclosed-data cleanup failed" in caplog.text
    checkpoint = objects.get_optional(f"retry_state/disclosures/{digest}.json")
    assert checkpoint is not None
    row = json.loads(checkpoint.body)
    assert row["attempt"] == 1 and row["retry_after_ms"] == 0 and row["error"] is None


@pytest.mark.parametrize("cold", [False, True])
def test_disclosure_success_cannot_hide_another_batch_failure(monkeypatch, cold):
    from tests.validator.production.test_storage import MemoryS3 as PrivateMemoryS3
    from validator.production.storage import PrivateObjectStore

    track = parse_track_spec(valid_spec())
    client = PrivateMemoryS3()
    clock = [1_000_000_000]
    monkeypatch.setattr(run_module.time, "time_ns", lambda: clock[0])
    first, second = "ab" * 32, "cd" * 32
    uploads = []

    def new_source():
        objects = PrivateObjectStore(client, bucket="test", prefix="test")
        source = SimpleNamespace(
            track=track, objects=objects, protected_plan_digests=lambda *_a: frozenset(),
            prune_disclosed=lambda _p: 0, public_disclosure=lambda digest, **_kw: digest,
            _plan_by_digest=lambda digest: SimpleNamespace(
                plan_digest=digest, exact=SimpleNamespace(sha256="ef" * 32),
            ),
        )
        source.due_plan_digests = lambda _now: tuple(
            digest for digest in (first, second)
            if objects.get_optional(f"disclosed/{digest}.json") is None
        )
        source.record_disclosed = lambda digest: run_module.ReadyBatchSource.record_disclosed(
            source, digest,
        )
        return source

    class Publisher:
        async def publish_disclosure(self, digest, **_kwargs):
            uploads.append(digest)
            if digest == first and uploads.count(first) == 1:
                raise OSError("private upload failure")

    source = new_source()
    publisher = Publisher()

    def execute():
        return asyncio.run(run_module._publish_due_batch(
            source, publisher, finalized_block=120, challenges=[],
        ))

    with pytest.raises(OSError, match="private upload failure"):
        execute()
    failed = source.objects.get(f"retry_state/disclosures/{first}.json")
    if cold:
        source = new_source()  # Only the existing S3 facts survive.
    with pytest.raises(RuntimeError, match="unresolved disclosure"):
        execute()  # B is published, but A is still failed and cooling down.
    assert source.objects.get_optional(f"disclosed/{second}.json") is not None
    assert source.objects.get(f"retry_state/disclosures/{first}.json") == failed
    with pytest.raises(RuntimeError, match="unresolved disclosure"):
        execute()
    assert uploads == [first, second]
    clock[0] += 61_000_000_000
    assert execute() == first
    assert source.due_plan_digests(0) == ()
    assert execute() is None
    assert uploads == [first, second, first]


def test_disclosure_cooldown_is_not_a_successful_observation():
    from tests.validator.production.test_storage import MemoryS3 as PrivateMemoryS3
    from validator.production.retries import PrivateRetryStore
    from validator.production.storage import PrivateObjectStore

    track = parse_track_spec(valid_spec())
    objects = PrivateObjectStore(PrivateMemoryS3(), bucket="test", prefix="test")
    digest = "ab" * 32
    retry = PrivateRetryStore(objects, track_spec_digest=track.digest)
    retry.disclosure_attempt(digest, now_ms=run_module.time.time_ns() // 1_000_000, lease_ms=60_000)
    before = objects.get(f"retry_state/disclosures/{digest}.json")
    source = SimpleNamespace(
        track=track, objects=objects, protected_plan_digests=lambda *_a: frozenset(),
        prune_disclosed=lambda _p: 0, due_plan_digests=lambda _now: (digest,),
    )
    assert asyncio.run(run_module._publish_due_batch(
        source, SimpleNamespace(), finalized_block=120, challenges=[],
    )) is False
    assert objects.get(f"retry_state/disclosures/{digest}.json") == before
    source.due_plan_digests = lambda _now: ()
    assert asyncio.run(run_module._publish_due_batch(
        source, SimpleNamespace(), finalized_block=120, challenges=[],
    )) is None


@pytest.mark.parametrize("already_scanned", [False, True])
@pytest.mark.parametrize("same_hotkey", [False, True])
def test_empty_new_track_recovers_verified_old_submission_before_fifo(
    tmp_path, already_scanned, same_hotkey,
):
    raw = valid_spec()
    raw.update(track_id="maintenance", track_start_block=150)
    track = parse_track_spec(raw)
    payload = ModelSubmission("owner/model", "12" * 20).encode()
    old_block = chain_block(payload, 99)
    new_block = chain_block(payload, 150)
    new_hotkey = HOTKEY if same_hotkey else "new-hotkey"
    new_block[1]["extrinsics"][0]["address"] = new_hotkey
    new_block[2][1]["attributes"]["who"] = new_hotkey
    client = FakeClient({99: old_block, 150: new_block, 151: empty_chain_block(151)})
    with ChallengeStore(tmp_path / "old.db", track_spec_digest="cd" * 32) as old:
        asyncio.run(run_module.scan_finalized_blocks(client, old, netuid=7,
                                                    track_start_block=99, max_blocks=1))
        facts = old.public_snapshot()["challenges"][0]
        scored = scored_challenge(1, HOTKEY, score=0.7, incumbent_score=0.7)
        for key in ("challenge_id", "submission_id", "hotkey", "repo_id", "revision",
                    "block_height", "extrinsic_index", "event_index"):
            scored[key] = facts[key]
        scored["score_block"] = scored["stages"][0]["score_block"] = 110
        scored["evaluation_batch"]["generated_block"] = 100
        history = resolve_tournament(
            [scored], track_id="previous", track_spec_digest="cd" * 32, netuid=7,
            minimum_gain=0.001,
            initial_champion=replace(genesis_champion(track), position=ChainPosition(90, 0, 0)),
        ).matches
    async def no_result(*, wait):
        assert wait is False
        return None
    work = SimpleNamespace(scoring_settings=ScoringSettings(), busy=True,
                           allowed=False, take=no_result)
    with ChallengeStore(tmp_path / "fresh.db", track_spec_digest=track.digest) as store:
        if already_scanned:
            asyncio.run(run_module.scan_finalized_blocks(client, store, netuid=7,
                                                        track_start_block=150))
        for _ in range(2):
            result = asyncio.run(run_module.process_cycle(
                client, store, netuid=7, track_start_block=150, work=work,
                recovery_matches=history,
            ))
            assert result.restored_matches == 1 and work.allowed is not same_hotkey
        restored = store.public_challenge(history[0].challenge_id)
        assert restored["state"] == "SUCCEEDED" and restored["track_spec_digest"] == "cd" * 32
        replayed = resolve_tournament(
            [restored], track_id="previous", track_spec_digest="cd" * 32, netuid=7,
            minimum_gain=0.001,
            initial_champion=replace(genesis_champion(track), position=ChainPosition(90, 0, 0)),
        )
        assert replayed.matches[0].document == history[0].document
        assert store.model_owner(scored["model_digest"]) == history[0].challenge_id
        assert store.chain_cursor()[0] == 151
        new = store.public_snapshot()["challenges"][1]
        assert new["hotkey"] == new_hotkey
        assert new["state"] == ("REJECTED_DUPLICATE_HOTKEY" if same_hotkey else "QUEUED")
        # Original hotkey's one opportunity remains spent after total DB loss.
        repeat, _ = store.add_submission(replace(submission_at(152), hotkey=HOTKEY))
        assert store.get_challenge(repeat).state is ChallengeState.REJECTED_DUPLICATE_HOTKEY


def test_missing_or_changed_historical_chain_input_never_opens_fifo(tmp_path):
    from validator.scan import extract_submissions

    payload = ModelSubmission("owner/model", "12" * 20).encode()
    block_hash, block, events = chain_block(payload, 99)
    submission = extract_submissions(
        block_height=99, block_hash=block_hash, block=block, events=events, netuid=7,
    )[0]
    with ChallengeStore(tmp_path / "old.db") as old:
        identifier, _ = old.add_submission(submission)
        old.transition(identifier, ChallengeState.FETCHING)
        old.transition(identifier, ChallengeState.PARTICIPANT_FAILED, error_code="NOT_PUBLIC")
        history = resolve_tournament(
            old.public_snapshot()["challenges"], track_id="previous", track_spec_digest="cd" * 32,
            netuid=7, minimum_gain=0.001,
            initial_champion=genesis_champion(parse_track_spec(valid_spec())),
        ).matches
    work = SimpleNamespace(scoring_settings=ScoringSettings(), allowed=False)
    client = FakeClient({99: empty_chain_block(99), 150: empty_chain_block(150)})
    with ChallengeStore(tmp_path / "fresh.db", track_spec_digest="ab" * 32) as store:
        with pytest.raises(RuntimeError, match="absent from the finalized block"):
            asyncio.run(run_module.process_cycle(client, store, netuid=7, track_start_block=150,
                                                work=work, recovery_matches=history))
        assert not work.allowed and store.public_snapshot()["challenges"] == []
        assert store.chain_cursor() is None


def test_historical_terminal_does_not_fence_new_fifo_or_consume_public_prefix(pending_evaluation):
    case = pending_evaluation
    old, _ = case.store.add_submission(replace(submission_at(99), hotkey="historical-hotkey"))
    case.store.transition(old, ChallengeState.FETCHING)
    case.store.transition(
        old, ChallengeState.PARTICIPANT_FAILED, error_code="NOT_PUBLIC",
    )
    # A retained terminal row in a copied migration checkpoint keeps its origin.
    case.store._connection.execute(
        "UPDATE challenges SET track_spec_digest=? WHERE challenge_id=?", ("cd" * 32, old),
    )
    case.store._connection.commit()
    historical = case.store.public_challenge(old)
    history = resolve_tournament(
        [historical], track_id="previous", track_spec_digest="cd" * 32,
        netuid=case.track.netuid, minimum_gain=case.track.full_video.minimum_gain,
        initial_champion=replace(genesis_champion(case.track), position=ChainPosition(90, 0, 0)),
    ).matches
    original = json.dumps(history[0].document, sort_keys=True, separators=(",", ":")).encode()
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + history[0].object_name,
        hashlib.sha256(original).hexdigest(), len(original), history[0].object_name,
    ),))
    raw = track_spec_dict(case.track)
    raw["artifact_manifest_sha256"] = manifest.digest
    case.track = parse_track_spec(raw)
    # This is the isolated target checkpoint's binding, not a live source DB edit.
    case.store._connection.execute(
        "UPDATE store_metadata SET value=? WHERE key='track_spec_digest'", (case.track.digest,),
    )
    case.store._connection.commit()
    case.publisher._put_immutable(history[0].object_name, original)

    def unavailable(*_args, **_kwargs):
        from validator.download import ModelDownloadError
        raise ModelDownloadError("REPOSITORY_UNAVAILABLE", "private", temporary=False)

    cycle = asyncio.run(_complete_cycle(
        case.client, case.store, netuid=case.track.netuid,
        track_start_block=case.track.track_start_block,
        admission_settings=case.work.admission_settings,
        scoring_settings=case.work.scoring_settings,
        evaluator=lambda _task: pytest.fail("rejected model was scored"), worker_id="new-track",
        downloader=unavailable, published_match_count=0,
    ))
    assert cycle.admission is not None
    assert case.store.public_challenge(old) == historical
    chain = WeightChainState(101, {case.track.initial_champion.hotkey: 0, HOTKEY: 1}, 1, 1.0, 1, {})
    arguments = {"version_key": 1, "runtime_manifest": manifest.encode(),
                 "historical_matches": history}
    genesis = build_public_artifacts(case.store, case.track, chain, match_limit=0, **arguments)
    assert genesis.matches == ()
    current = build_public_artifacts(case.store, case.track, chain, match_limit=1, **arguments)
    assert len(current.matches) == 1
    assert json.loads(current.matches[0][1])["challenge_id"] != old
    assert json.loads(current.summary)["match_count"] == 1
    asyncio.run(case.publisher.publish(current))
    cold = S3Repository(case.transport, bucket="test", prefix="isolated/heartbeat")
    assert asyncio.run(cold.load(case.track)).committed_match_count == 1
    assert case.store.public_challenge(old) == historical


def test_retained_success_allows_new_admission_two_stages_and_publication(pending_evaluation):
    """CPU lifecycle fixture, not GPU scoring or a replay of a real miner's result."""
    case = pending_evaluation
    old, _ = case.store.add_submission(replace(
        submission_at(99, hotkey="historical-hotkey"), repo_id="owner/historical-model",
    ))
    original = case.store.public_challenge(old)
    scored = scored_challenge(1, original["hotkey"], score=0.7, incumbent_score=0.7)
    for key in ("challenge_id", "submission_id", "hotkey", "repo_id", "revision",
                "block_height", "extrinsic_index", "event_index"):
        scored[key] = original[key]
    scored["score_block"] = scored["stages"][0]["score_block"] = 99
    scored["evaluation_batch"]["generated_block"] = 99
    history = resolve_tournament(
        [scored], track_id="previous", track_spec_digest="cd" * 32,
        netuid=case.track.netuid, minimum_gain=case.track.full_video.minimum_gain,
        initial_champion=replace(genesis_champion(case.track), position=ChainPosition(90, 0, 0)),
    ).matches
    body = json.dumps(history[0].document, sort_keys=True, separators=(",", ":")).encode()
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + history[0].object_name,
        hashlib.sha256(body).hexdigest(), len(body), history[0].object_name,
    ),))
    raw = track_spec_dict(case.track)
    raw["artifact_manifest_sha256"] = manifest.digest
    case.track = parse_track_spec(raw)
    case.store._connection.execute(
        "UPDATE store_metadata SET value=? WHERE key='track_spec_digest'", (case.track.digest,),
    )
    case.store._connection.commit()
    assert case.store.restore_public_matches(history) == 1
    historical = case.store.public_challenge(old)
    assert historical["state"] == "SUCCEEDED" and len(historical["stages"]) == 1
    case.publisher._put_immutable(history[0].object_name, body)
    arguments = {"version_key": 1, "runtime_manifest": manifest.encode(),
                 "historical_matches": history}

    def artifacts(limit):
        chain = WeightChainState(
            case.client._finalized, {case.track.initial_champion.hotkey: 0, HOTKEY: 1},
            1, 1.0, 1, {},
        )
        return build_public_artifacts(case.store, case.track, chain, match_limit=limit, **arguments)

    evaluated = []

    def evaluate(task):
        evaluated.append(task.state)
        exact = task.state is ChallengeState.SCORING
        return StageOutcome(0.9 if exact else 0.8, {
            "stage": "scoring" if exact else "confirming",
            "incumbent_score": 0.7, "paired_gain_lower_bound": 0.1,
            "cache_sha256": ("de" if exact else "12") * 32,
            "cache_generated_block": 100 if exact else 101,
            "cache_plan_digest": ("ef" if exact else "34") * 32,
            "cache_seed_block_hash": "0x" + "ab" * 32,
            "requires_confirmation": exact,
        })

    async def cycle():
        return await _complete_cycle(
            case.client, case.store, netuid=case.track.netuid,
            track_start_block=case.track.track_start_block,
            admission_settings=case.work.admission_settings,
            scoring_settings=case.work.scoring_settings, evaluator=evaluate,
            worker_id="maintenance", downloader=case.work.downloader,
            published_match_count=0, recovery_matches=history,
        )

    exact = asyncio.run(cycle())
    assert exact.admission is not None, "retained history must not fence the new FIFO"
    assert exact.admission.status is AdmissionStatus.ACCEPTED
    assert exact.scoring.state is ChallengeState.CONFIRMING
    prefix = artifacts(0)
    assert prefix.matches == () and decode_weight_artifact(prefix.weights).plan.uids == (0,)
    asyncio.run(case.publisher.publish(prefix))
    case.client._finalized = 102
    confirmation = asyncio.run(cycle())  # A fresh worker/SQLite connection resumes phase two.
    assert confirmation.scoring.state is ChallengeState.SUCCEEDED
    assert evaluated == [ChallengeState.SCORING, ChallengeState.CONFIRMING]
    new = case.store.public_challenge(confirmation.scoring.challenge_id)
    assert len(new["stages"]) == 2 and new["track_spec_digest"] == case.track.digest
    assert case.store.public_challenge(old) == historical

    # Only the new terminal decision fences work until its own seq1 is published.
    fenced = asyncio.run(_pending_cycle(case, published_match_count=0))
    assert not case.work.allowed and fenced.admission is None and fenced.scoring is None
    current = artifacts(1)
    assert len(current.matches) == 1
    document = json.loads(current.matches[0][1])
    assert document["sequence"] == 1 and document["challenge_id"] == new["challenge_id"]
    assert document["outcome"] == MatchOutcome.CHALLENGER_WON
    assert decode_weight_artifact(current.weights).plan.uids == (1,)
    asyncio.run(case.publisher.publish(current))
    # Fixture scores/cache hashes do not claim a real public payload/cold restore.
    assert json.loads(case.transport.objects["isolated/heartbeat/summary.json"])["match_count"] == 1
    name, payload = current.matches[0]
    assert case.transport.objects["isolated/heartbeat/" + name] == payload
    assert case.transport.objects["isolated/heartbeat/" + history[0].object_name] == body
    assert case.store.public_challenge(old) == historical
    assert case.store.model_owner(scored["model_digest"]) == old
    repeated, _ = case.store.add_submission(submission_at(103, hotkey=historical["hotkey"]))
    assert case.store.get_challenge(repeated).state is ChallengeState.REJECTED_DUPLICATE_HOTKEY


def test_continuous_evaluator_reaches_track_loading_without_external_webhook(monkeypatch):
    def load(_path):
        raise RuntimeError("reached track loading")

    monkeypatch.setattr(run_module, "load_track_spec", load)
    with pytest.raises(RuntimeError, match="reached track loading"):
        asyncio.run(run_module.run_evaluator(
            track_spec="track.json", database="unused.db", model_cache="unused-models",
            exact_cache="unused-batches", initial_model=None, network="finney",
            archive_endpoints=None, worker_id="test", publisher=SimpleNamespace(),
            alert_webhook_url=None,
        ))


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

    raw = valid_spec()
    pool = SourcePool(tuple(
        SourceLeaf(
            "production" if index < 20 else "audit", f"family-{index}", f"source-{index}",
            f"unit-test prompt {index}", (), "all", "internal-cleared-v1",
            hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
        ) for index in range(21)
    ))
    raw["source_pool_merkle_root"] = pool.root
    raw["source_commitments"] = {
        partition: {"all": [leaf.digest for leaf in pool.leaves if leaf.partition == partition]}
        for partition in ("production", "audit")
    }
    track = parse_track_spec(raw)
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
                tensor_schema=track.model.tensor_schema,
            ),
            scoring_settings=ScoringSettings(alert=lambda *args: alerts.append(args)),
            worker_id="single-worker", downloader=lambda *_args, **_kwargs: downloaded,
        )
        transport = MemoryS3()
        publisher = S3Repository(transport, bucket="test", prefix="isolated/heartbeat")
        publisher.verify_writes()
        try:
            yield SimpleNamespace(
                track=track, pool=pool, store=store, client=client, work=work, entered=entered,
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


def test_inflight_result_obeys_publication_failure_fence(pending_evaluation):
    case = pending_evaluation

    async def execute():
        await _pending_cycle(case)
        await _eventually(case.entered.is_set)
        case.client._finalized = 105
        for _ in range(run_module.MAX_PUBLICATION_FAILURES):
            case.store.record_publication_failure()
        paused = await _pending_cycle(case)
        assert paused.scoring is None
        case.release.set()
        await _eventually(lambda: case.work.done)
        result = await _pending_cycle(case)
        challenge = case.store.public_snapshot()["challenges"][0]
        assert challenge["final_score"] is None and challenge["stages"] == []
        assert len(case.calls) == 1
        assert result.scoring is None and case.work.busy
        await _publish_pending(case)
        case.store.publication_succeeded()
        result = await _pending_cycle(case)
        assert result.scoring.state is ChallengeState.SUCCEEDED
        assert not case.work.busy

    asyncio.run(execute())


def test_scan_backlog_is_fully_replayed_before_starting_model_io(pending_evaluation):
    case = pending_evaluation
    case.client._finalized = 105

    async def execute():
        for height in range(100, 105):
            result = await _pending_cycle(case, max_scan_blocks=1)
            assert case.store.chain_cursor()[0] == height
            assert result.admission is result.scoring is None
            assert not case.work.busy
            assert not case.entered.is_set()
        await _pending_cycle(case, max_scan_blocks=1)
        await _eventually(case.entered.is_set)
        assert case.store.chain_cursor()[0] == 105
        case.release.set()
        result = await _pending_cycle(case, max_scan_blocks=1, wait_for_work=True)
        assert result.scoring.state is ChallengeState.SUCCEEDED

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


@pytest.mark.parametrize("benchmark_refresh", [
    "normal", "slow", "error", "inventory_slow", "inventory_error",
    "disclosure_slow", "disclosure_error", "inventory_recovery", "disclosure_recovery",
    "disclosure_multibatch_recovery", "disclosure_cold_recovery",
])
def test_evaluator_loop_publishes_while_worker_owns_its_connection(
    pending_evaluation, monkeypatch, tmp_path, benchmark_refresh,
):
    case = pending_evaluation
    entered, release = threading.Event(), threading.Event()
    admission_gate = threading.Event()
    inventory_gate = threading.Event()
    calls, reports, failures = [], [], []
    snapshots = []
    benchmark_calls = []
    metadata_snapshots = []
    metadata_error_seen = threading.Event()
    recovering = benchmark_refresh.endswith("_recovery")
    multibatch = benchmark_refresh in {"disclosure_multibatch_recovery", "disclosure_cold_recovery"}
    published_disclosures, disclosure_completions, uploads = [], [], []
    disclosure_clock = [run_module.time.time_ns()]
    if multibatch:
        monkeypatch.setattr(run_module.time, "time_ns", lambda: disclosure_clock[0])
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

    def no_startup_bulk(*_args):
        pytest.fail("startup bulk auditing blocked the owner heartbeat")

    monkeypatch.setattr(case.publisher, "_verify_model_archives", no_startup_bulk)
    monkeypatch.setattr(case.publisher, "_load_batch_disclosures", no_startup_bulk)

    if benchmark_refresh.startswith(("inventory_", "disclosure_")):
        from tests.validator.production.test_storage import MemoryS3 as PrivateMemoryS3
        from validator.production.storage import PrivateObjectStore

        original_evaluator = run_module.PairedTrackEvaluator
        if recovering:
            monkeypatch.setattr(run_module, "_DATA_REFRESH_SECONDS", 0.001)

        def inventory_plans():
            assert threading.get_ident() != owner
            benchmark_calls.append(benchmark_refresh)
            if benchmark_refresh == "inventory_error" or (
                benchmark_refresh == "inventory_recovery" and len(benchmark_calls) == 1
            ):
                raise OSError("private inventory read unavailable")
            if recovering:
                assert metadata_error_seen.wait(5)
                return ()
            if not inventory_gate.wait(5):
                raise OSError("test did not release inventory read")
            return ()

        def evaluator_factory(**kwargs):
            evaluator = original_evaluator(**kwargs)
            evaluator.inventory_status = {
                "allocated_stages": [], "observed_at_ns": 100,
                "ready_batches": 4, "producing_batches": 0, "target_batches": 2,
                "disclosure_pending_batches": 1, "disclosure_observed_at_ns": 50,
            }
            evaluator.batch_source = SimpleNamespace(
                _plans=inventory_plans, inventory_status=lambda _plans: {
                    "allocated_stages": [], "observed_at_ns": 200, "ready_batches": 5,
                    "producing_batches": 0, "target_batches": 2, "disclosure_pending_batches": 1,
                },
                protected_plan_digests=lambda *_a: frozenset(), prune_disclosed=lambda _p: 0,
                due_plan_digests=lambda _now: (),
            )
            if benchmark_refresh.startswith("disclosure_"):
                source = evaluator.batch_source
                source._plans = lambda: ()
                source.track = case.track
                source.objects = PrivateObjectStore(PrivateMemoryS3(), bucket="test", prefix="test")
                source.due_plan_digests = lambda _now: ("ab" * 32,)
                source.public_disclosure = lambda *_a, **_k: object()
                source.record_disclosed = lambda _digest: None
                if multibatch:
                    from validator.production.retries import PrivateRetryStore

                    source.due_plan_digests = lambda _now: tuple(
                        digest for digest in ("ab" * 32, "cd" * 32)
                        if source.objects.get_optional(f"disclosed/{digest}.json") is None
                    )
                    source.public_disclosure = lambda digest, **_kw: digest
                    source._plan_by_digest = lambda digest: SimpleNamespace(
                        plan_digest=digest, exact=SimpleNamespace(sha256="ef" * 32),
                    )

                    def record_disclosed(digest):
                        run_module.ReadyBatchSource.record_disclosed(source, digest)
                        published_disclosures.append(digest)

                    source.record_disclosed = record_disclosed
                    if benchmark_refresh == "disclosure_cold_recovery":
                        retry = PrivateRetryStore(
                            source.objects, track_spec_digest=case.track.digest,
                        )
                        now_ms = disclosure_clock[0] // 1_000_000
                        retry.disclosure_attempt("ab" * 32, now_ms=now_ms, lease_ms=1000)
                        retry.disclosure_attempt("ab" * 32, now_ms=now_ms, lease_ms=1000,
                                                 completed_attempt=1, error="prior host failure")
            return evaluator

        monkeypatch.setattr(run_module, "PairedTrackEvaluator", evaluator_factory)
        if benchmark_refresh.startswith("disclosure_"):
            if multibatch:
                original_disclose = run_module._publish_due_batch

                async def observe_disclosure(*args, **kwargs):
                    try:
                        return await original_disclose(*args, **kwargs)
                    finally:
                        disclosure_completions.append((
                            client._finalized, tuple(published_disclosures),
                        ))

                monkeypatch.setattr(run_module, "_publish_due_batch", observe_disclosure)
            elif recovering:
                async def recover_disclosure(*_args, **_kwargs):
                    benchmark_calls.append(benchmark_refresh)
                    if len(benchmark_calls) == 1:
                        raise OSError("public bulk upload unavailable")
                    if len(benchmark_calls) == 2:
                        return False  # Durable cooldown is not a successful retry.
                    assert await asyncio.to_thread(metadata_error_seen.wait, 5)
                    return "ab" * 32

                monkeypatch.setattr(run_module, "_publish_due_batch", recover_disclosure)

            async def disclose(disclosure, *_args, **_kwargs):
                benchmark_calls.append(benchmark_refresh)
                if multibatch:
                    uploads.append(disclosure)
                    if (benchmark_refresh != "disclosure_cold_recovery"
                            and disclosure == "ab" * 32 and uploads.count(disclosure) == 1):
                        raise OSError("public bulk upload unavailable")
                    return
                if benchmark_refresh == "disclosure_error":
                    raise OSError("public bulk upload unavailable")
                if not await asyncio.to_thread(inventory_gate.wait, 5):
                    raise OSError("test did not release disclosure upload")

            monkeypatch.setattr(case.publisher, "publish_disclosure", disclose)
    elif benchmark_refresh != "normal":
        async def load_benchmarks(_track, _manifest):
            benchmark_calls.append(benchmark_refresh)
            if benchmark_refresh == "error":
                raise OSError("informational benchmark read unavailable")
            await asyncio.Event().wait()

        monkeypatch.setattr(case.publisher, "load_system_matches", load_benchmarks)

    async def no_seed(*_args, **_kwargs):
        pass

    async def chain_state(_client, *, netuid, mechid, finalized_block):
        assert netuid == case.track.netuid
        assert mechid == case.track.mechid
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
        metadata_snapshots.append(status.get("inventory", {}))
        weights = decode_weight_artifact(
            case.transport.objects["isolated/heartbeat/weights.json"],
        )
        assert weights.source_block == published_block
        assert weights.plan.uids == (0,)
        metadata_ready = True
        if benchmark_refresh in {"inventory_error", "inventory_recovery"}:
            errors = [row for row in metadata_snapshots if row.get("refresh_error")]
            metadata_ready = bool(errors)
            if status.get("inventory", {}).get("refresh_error"):
                metadata_error_seen.set()
            if recovering:
                metadata_ready = metadata_ready and (
                    status.get("inventory", {}).get("observed_at_ns") == 200
                    and status["inventory"].get("refresh_error") is None
                )
        elif multibatch:
            after_second = any(
                block < published_block and published == ("cd" * 32,)
                for block, published in disclosure_completions
            )
            if after_second and "ab" * 32 not in published_disclosures:
                assert status["inventory"].get("disclosure_error") == "DATA_DISCLOSURE_FAILURE"
                assert status["inventory"]["disclosure_observed_at_ns"] == 50
                if not metadata_error_seen.is_set():
                    metadata_error_seen.set()
                    disclosure_clock[0] += 61_000_000_000
            metadata_ready = (
                metadata_error_seen.is_set() and "ab" * 32 in published_disclosures
                and status["inventory"].get("disclosure_error") is None
                and status["inventory"].get("disclosure_observed_at_ns", 0) > 50
            )
        elif benchmark_refresh in {"disclosure_error", "disclosure_recovery"}:
            errors = [row for row in metadata_snapshots if row.get("disclosure_error")]
            metadata_ready = bool(errors)
            if len(benchmark_calls) >= 2 and status.get("inventory", {}).get("disclosure_error"):
                assert status["inventory"]["disclosure_observed_at_ns"] == 50
                metadata_error_seen.set()
            if recovering:
                metadata_ready = metadata_ready and (
                    status.get("inventory", {}).get("disclosure_observed_at_ns", 0) > 50
                    and status["inventory"].get("disclosure_error") is None
                )
        if len(reports) >= 3 and metadata_ready:
            inventory_gate.set()
            release.set()
            raise StopLoop
        client._finalized += 1

    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: case.track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", no_seed)
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
                report=report, poll_seconds=0.005 if multibatch else 0.001,
                alert_webhook_url="https://alerts.example/veda",
            )

    try:
        with pytest.raises(StopLoop):
            asyncio.run(execute())
    finally:
        metadata_error_seen.set()
        inventory_gate.set()
        admission_gate.set()
        release.set()
    assert not failures
    assert len(calls) == 1
    assert len(reports) >= 3
    if recovering:
        assert len(benchmark_calls) >= 2
    elif benchmark_refresh != "normal":
        assert benchmark_calls == [benchmark_refresh]
    if benchmark_refresh in {"inventory_error", "inventory_recovery"}:
        failed = next(row for row in metadata_snapshots if row.get("refresh_error"))
        assert failed["refresh_error"] == "INVENTORY_REFRESH_FAILED"
        assert failed["observed_at_ns"] == 100 and failed["ready_batches"] == 4
    if benchmark_refresh in {"disclosure_error", "disclosure_recovery"}:
        failed = next(row for row in metadata_snapshots if row.get("disclosure_error"))
        assert failed["disclosure_error"] == "DATA_DISCLOSURE_FAILURE"
        assert failed["disclosure_observed_at_ns"] == 50
    if multibatch:
        assert published_disclosures == ["cd" * 32, "ab" * 32]
        assert sum(bool(row.get("disclosure_error")) for row in metadata_snapshots) >= 1
        assert sum(code == "DATA_DISCLOSURE_FAILURE" for code, _detail in case.alerts) == 1
    assert "private inventory read unavailable" not in json.dumps(metadata_snapshots)
    assert "public bulk upload unavailable" not in json.dumps(metadata_snapshots)
    first_height = [payload for height, payload in snapshots if height == 102]
    assert len(first_height) >= 2 and len(set(first_height)) == 1
    assert next(
        row for row in json.loads(first_height[0])["participants"] if row["hotkey"] == HOTKEY
    )["state"] == "QUEUED"
    # ExitStack joins the in-flight thread before closing the evaluator resources.
    assert not any(thread.name.startswith("veda-evaluation") for thread in threading.enumerate())


@pytest.mark.parametrize("mode", [
    "success", "prepare_retry", "retry", "retry_exhaustion", "restart", "stage_restart",
    "success_lost_response",
])
def test_slow_model_archive_keeps_weights_fresh_until_immutable_match_commit(
    pending_evaluation, monkeypatch, tmp_path, mode,
):
    """CPU orchestration/real byte uploads; fixture scores are not quality evidence."""

    from tests.validator.production.test_storage import MemoryS3 as PrivateMemoryS3
    from validator.production.batches import ReadyBatchSource
    from validator.production.plan import (
        DenseReference,
        ReadyObject,
        ReadySampleCache,
        build_plan,
        generation_seed,
        select_source_proofs,
    )
    from validator.production.storage import PrivateObjectStore

    case = pending_evaluation
    lost_response = mode.endswith("_lost_response")
    mode = mode.removesuffix("_lost_response")
    blocks = {height: empty_chain_block(height) for height in range(100, 1201)}
    blocks[100] = chain_block(ModelSubmission("owner/model", "12" * 20).encode(), 100)
    second = chain_block(ModelSubmission("owner/second", "34" * 20).encode(), 402)
    second[1]["extrinsics"][0]["address"] = "second-hotkey"
    second[2][1]["attributes"]["who"] = "second-hotkey"
    blocks[402] = second
    replay = FakeClient(blocks)
    plans, paths, exact_paths, input_roots = {}, {}, {}, {}
    used_sources = set()

    def build_groups(challenge_id):
        for generation in (200, 400):
            block_hash = "0x" + f"{generation:064x}"
            proofs = select_source_proofs(
                case.pool, commitments=case.track.source_commitments, partition="production",
                strata=("all",), sources_per_stratum=case.track.scoring.rollouts_per_stratum,
                batch_seed=generation_seed(case.track.digest, block_hash, "a5" * 32),
                excluded_source_digests=frozenset(used_sources),
                inventory_slot=0 if generation == 200 else 1,
            )
            used_sources.update(proof.leaf.digest for proof in proofs)
            plan = build_plan(secret="a5" * 32, inventory_slot=0 if generation == 200 else 1,
                track_spec_digest=case.track.digest, generation_block=generation,
                block_hash=block_hash,
                source_proofs=proofs, scoring=case.track.scoring,
                available_routes=case.track.model.selector.routes,
            )
            directory = tmp_path / f"group-{generation}"
            directory.mkdir()

            def content(name, *, directory=directory, generation=generation):
                path = directory / name
                payload = f"fixture bytes {generation}/{name}".encode()
                path.write_bytes(payload)
                item = ReadyObject(
                    f"private/{generation}/{name}",
                    hashlib.sha256(payload).hexdigest(), len(payload),
                )
                paths[item.object_key] = path
                source.content.put(path, sha256=item.sha256, size_bytes=item.size_bytes)
                return item

            exact = content("batch.npz")
            inputs = ReadySampleCache(content("index.json"), content("text.safetensors"))
            references = tuple(
                DenseReference(rollout.rollout_id, content(f"video-{index}.mp4"),
                               content(f"latent-{index}.safetensors"))
                for index, rollout in enumerate(plan.rollouts)
            )
            ready = plan.ready(
                completed_at_ms=(plan.generation_block + 1) * 1000,
                completed_block=plan.generation_block + 1,
                exact=exact,
                dense_references=references,
                sample_cache=inputs,
            )
            plans[generation] = ready
            exact_paths[ready.plan_digest] = paths[exact.object_key]
            input_roots[ready.plan_digest] = directory

    source = ReadyBatchSource(
        track=case.track, cache_dir=tmp_path / "restored",
        objects=PrivateObjectStore(PrivateMemoryS3(), bucket="test", prefix="private-fixture"),
    )
    source._plan_by_digest = lambda digest: next(
        plan for plan in plans.values() if plan.plan_digest == digest
    )
    source._restore_exact = lambda plan: SimpleNamespace(path=exact_paths[plan.plan_digest])
    source._restore_sample_cache = lambda plan: input_roots[plan.plan_digest]

    def fixture_score(task):
        if not plans:
            build_groups(task.challenge_id)
        exact = task.state is ChallengeState.SCORING
        plan = plans[200 if exact else 400]
        return StageOutcome(0.9, {
            "stage": "scoring" if exact else "confirming",
            "incumbent_score": 0.7, "paired_gain_lower_bound": 0.2,
            "cache_sha256": plan.exact.sha256,
            "cache_generated_block": plan.generation_block,
            "cache_plan_digest": plan.plan_digest,
            "cache_seed_block_hash": plan.block_hash,
            "requires_confirmation": exact,
        })

    for height in (301, 401):
        replay._finalized = height
        asyncio.run(_complete_cycle(
            replay, case.store, netuid=7, track_start_block=100, max_scan_blocks=500,
            admission_settings=case.work.admission_settings,
            scoring_settings=case.work.scoring_settings, evaluator=fixture_score,
            worker_id="fixture", downloader=case.work.downloader,
        ))
    challenge = case.store.public_snapshot()["challenges"][0]
    assert challenge["state"] == ChallengeState.SUCCEEDED, challenge["error_summary"]
    prepare_entered, prepare_release = threading.Event(), threading.Event()
    upload_entered, upload_release = threading.Event(), threading.Event()
    preparation_heads, upload_heads, failures = [], [], []
    retry_observed = []
    upload_failures = 0
    prepare_failed = False
    stage_failure_enabled = mode == "stage_restart"
    height = 401
    owner_thread = threading.get_ident()
    original_stage = run_module._stage_publication

    class StopLoop(BaseException):
        pass

    class Client(FakeClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def finalized_block(self):
            return height

    def slow_prepare():
        nonlocal prepare_failed
        assert threading.get_ident() != owner_thread
        prepare_entered.set()
        if not prepare_release.wait(5):
            raise OSError("test did not release immutable metadata preparation")
        if mode == "prepare_retry" and not prepare_failed:
            prepare_failed = True
            raise OSError("injected immutable metadata preparation failure")

    async def slow_stage(artifacts, **kwargs):
        await asyncio.to_thread(slow_prepare)
        prepared = await original_stage(artifacts, **kwargs)
        if stage_failure_enabled:
            raise OSError("injected crash after immutable match staging")
        return prepared

    def slow_upload(arguments):
        nonlocal upload_failures
        if arguments["Key"].endswith("/weights.json"):
            weights = decode_weight_artifact(arguments["Body"])
            if weights.match_sequence:
                assert not any("/batch_groups/" in key for key in transport.objects)
                assert any("/matches/" in key for key in transport.objects)
        if "/models/" in arguments["Key"] and arguments["Key"].endswith("/part-00000"):
            upload_entered.set()
            if not upload_release.wait(5):
                raise OSError("test did not release immutable upload")
            limit = (
                run_module.MAX_PUBLICATION_FAILURES + 1 if mode == "retry_exhaustion"
                else 1 if mode == "retry" else 0
            )
            if upload_failures < limit:
                upload_failures += 1
                raise OSError("injected immutable upload failure")

    async def no_seed(*_args, **_kwargs):
        pass

    async def chain_state(_client, *, netuid, mechid, finalized_block):
        assert mechid == case.track.mechid
        return WeightChainState(
            finalized_block, {case.track.initial_champion.hotkey: 0, HOTKEY: 1,
                              "second-hotkey": 2}, 1, 1.0, 1, {},
        )

    def no_work(*_args, **_kwargs):
        pytest.fail("unpublished terminal facts must fence the next FIFO task")

    transport = MemoryS3()
    transport.before_put = slow_upload
    response_lost = False
    original_put = transport.put_object

    def put_with_lost_response(**arguments):
        nonlocal response_lost
        original_put(**arguments)
        if (
            lost_response and not response_lost and "/models/" in arguments["Key"]
            and arguments["Key"].endswith("/part-00000")
        ):
            response_lost = True
            raise OSError("response lost after durable model chunk PUT")

    monkeypatch.setattr(transport, "put_object", put_with_lost_response)
    publisher = S3Repository(transport, bucket="test", prefix="isolated/slow-disclosure")
    database = case.work.database

    def report(result):
        nonlocal height
        if result.cycle_error:
            failures.append(result.cycle_error)
            prepare_release.set()
            upload_release.set()
            raise StopLoop
        weights = decode_weight_artifact(transport.objects["isolated/slow-disclosure/weights.json"])
        assert weights.source_block == height
        if weights.match_sequence == 1:
            assert weights.plan.uids == (1,)
            manifest_name = f"isolated/slow-disclosure/models/{case.track.track_id}/" + (
                f"{challenge['file_sha256']}/manifest.json"
            )
            assert manifest_name in transport.objects
            assert upload_release.is_set()
            raise StopLoop
        assert weights.match_sequence == 0 and weights.plan.uids == (0,)
        if mode == "stage_restart" and result.publication_error:
            assert "/matches/" in next(
                key for key in transport.objects if key.endswith("000000000001.json")
            )
            assert not any("/batch_groups/" in key for key in transport.objects)
            height += 1
            raise StopLoop
        if prepare_entered.is_set() and not prepare_release.is_set():
            preparation_heads.append(height)
            if len(preparation_heads) == 3:
                prepare_release.set()
        if upload_entered.is_set() and not upload_release.is_set():
            upload_heads.append(height)
            if len(upload_heads) == 3:
                upload_release.set()
                if mode == "restart":
                    height += 1
                    raise StopLoop
        with ChallengeStore(database, track_spec_digest=case.track.digest) as observer:
            if observer.publication_failures():
                retry_observed.append(observer.publication_failures())
        height += 1

    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: case.track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_args, **_kwargs: Client(blocks))
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", no_seed)
    monkeypatch.setattr(run_module, "read_weight_chain_state", chain_state)
    monkeypatch.setattr(run_module, "_stage_publication", slow_stage)
    monkeypatch.setattr(run_module, "process_next_admission", no_work)
    monkeypatch.setattr(run_module, "evaluate_next_scoring", no_work)
    monkeypatch.setattr(run_module, "webhook_alerter", lambda *_args, **_kwargs: None)
    (tmp_path / "batches").mkdir()

    async def execute(*, once=False):
        async with asyncio.timeout(8):
            await run_module.run_evaluator(
                track_spec="track.json", database=database, model_cache=tmp_path / "models",
                exact_cache=tmp_path / "batches", initial_model=tmp_path / "baseline.safetensors",
                network="test", archive_endpoints=None, worker_id="single-owner",
                publisher=publisher, report=None if once else report, once=once,
                poll_seconds=0.001, max_scan_blocks=1200,
                alert_webhook_url="https://alerts.example/veda",
            )

    try:
        with pytest.raises(StopLoop):
            asyncio.run(execute())
    finally:
        prepare_release.set()
        upload_release.set()
    assert not failures
    assert response_lost is lost_response
    assert len(preparation_heads) == len(upload_heads) == 3
    assert len(set(preparation_heads)) == len(set(upload_heads)) == 3
    if mode == "prepare_retry":
        assert prepare_failed and retry_observed
    if mode == "retry":
        assert upload_failures == 1 and retry_observed
    if mode == "retry_exhaustion":
        assert upload_failures == run_module.MAX_PUBLICATION_FAILURES + 1
        assert max(retry_observed) == run_module.MAX_PUBLICATION_FAILURES
    if mode in {"restart", "stage_restart"}:
        # Keep all remote immutable bytes, but discard the complete local database.
        # Reconcile the recovered public prefix without HF/model inference.
        database = tmp_path / "recovered.db"
        publisher = S3Repository(transport, bucket="test", prefix="isolated/slow-disclosure")
        recovered_ledger = asyncio.run(publisher.load(case.track))
        assert len(recovered_ledger.matches) == 1
        assert recovered_ledger.committed_match_count == 0
        if mode == "stage_restart":
            assert recovered_ledger.disclosed_plan_digests == frozenset()
            stage_failure_enabled = False
        asyncio.run(execute(once=True))
        with ChallengeStore(database, track_spec_digest=case.track.digest) as recovered:
            assert recovered.public_snapshot()["challenges"][0]["state"] == ChallengeState.SUCCEEDED
    weights = decode_weight_artifact(transport.objects["isolated/slow-disclosure/weights.json"])
    assert weights.match_sequence == 1 and weights.plan.uids == (1,)
    with ChallengeStore(database, track_spec_digest=case.track.digest) as recovered:
        assert recovered.publication_failures() == 0
    ledger = asyncio.run(publisher.load(case.track))
    assert len(ledger.matches) == 1
    assert ledger.disclosed_plan_digests == frozenset()


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


@pytest.mark.parametrize("chain_outage", [False, True])
def test_ledger_outage_scanning_reserves_uid0_and_recovers_from_chain_failure(
    monkeypatch, tmp_path, chain_outage,
):
    document = valid_spec()
    document["initial_champion"]["hotkey"] = HOTKEY
    track = parse_track_spec(document)
    blocks = {height: empty_chain_block(height) for height in range(100, 151)}
    blocks[100] = chain_block(ModelSubmission("owner/uid-zero", "12" * 20).encode(), 100)
    reports = []
    connections = []
    unavailable = chain_outage
    finalized_height = 101

    class StopLoop(BaseException):
        pass

    class Client(FakeClient):
        async def __aenter__(self):
            connections.append("opened")
            return self

        async def __aexit__(self, *_args):
            connections.append("closed")
            return None

        async def finalized_block(self):
            nonlocal unavailable
            if unavailable:
                unavailable = False
                raise OSError("finalized head temporarily unavailable")
            return finalized_height

    class Publisher(S3Repository):
        remaining_load_failures = 2 if chain_outage else 1

        async def recover(self, candidate):
            if self.remaining_load_failures:
                self.remaining_load_failures -= 1
                raise OSError("public ledger temporarily unavailable")
            return await super().recover(candidate)

    async def no_seed(*_args, **_kwargs):
        pass

    async def chain_state(_client, *, netuid, mechid, finalized_block):
        assert mechid == track.mechid
        return WeightChainState(
            finalized_block=finalized_block, uid_by_hotkey={HOTKEY: 0},
            min_allowed_weights=1, max_weight=1.0, weights_rate_limit=1,
            last_update_by_hotkey={},
        )

    def no_download(*_args, **_kwargs):
        pytest.fail("UID0 must never reach model I/O, including after public recovery")

    def report(result):
        nonlocal finalized_height
        reports.append(result)
        if chain_outage and result.published_block is not None:
            current = decode_weight_artifact(transport.objects["isolated/uid-zero/weights.json"])
            if current.match_sequence == 1:
                raise StopLoop
            finalized_height += 1

    transport = MemoryS3()
    publisher = Publisher(transport, bucket="test", prefix="isolated/uid-zero")
    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_args, **_kwargs: Client(blocks))
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", no_seed)
    monkeypatch.setattr(run_module, "read_weight_chain_state", chain_state)
    monkeypatch.setattr(run_module, "webhook_alerter", lambda *_args, **_kwargs: None)
    (tmp_path / "batches").mkdir()
    database = tmp_path / "validator.db"
    arguments = {
        "track_spec": "track.json", "database": database,
        "model_cache": tmp_path / "models", "exact_cache": tmp_path / "batches",
        "initial_model": tmp_path / "baseline.safetensors", "network": "finney",
        "archive_endpoints": None, "worker_id": "single-worker", "publisher": publisher,
        "once": not chain_outage, "report": report, "poll_seconds": 0.001,
        "alert_webhook_url": "https://alerts.example/veda",
    }
    original_admission = run_module.process_next_admission

    def admission(store, **kwargs):
        return original_admission(store, **{**kwargs, "downloader": no_download})

    monkeypatch.setattr(run_module, "process_next_admission", admission)

    async def execute():
        async with asyncio.timeout(5):
            await run_module.run_evaluator(**arguments)

    if chain_outage:
        with pytest.raises(StopLoop):
            asyncio.run(execute())
        assert reports[0].cycle_error == "finalized head temporarily unavailable"
        assert reports[0].scan is None
        assert reports[1].scan.submissions_found == 1
    else:
        asyncio.run(execute())
    with ChallengeStore(database, track_spec_digest=track.digest) as store:
        snapshot = store.public_snapshot()
        assert snapshot["challenges"][0]["state"] == ChallengeState.REJECTED_DUPLICATE_HOTKEY
        assert store.chain_cursor()[0] == finalized_height
    assert reports[0].publication_error == "public ledger temporarily unavailable"
    if not chain_outage:
        asyncio.run(execute())
    assert connections == ["opened", "closed"] * (3 if chain_outage else 2)
    assert reports[-1].published_block == finalized_height
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
        async def recover(self, _track):
            return SimpleNamespace(
                matches=(), required_finalized_block=0, committed_match_count=0,
            )

    track = SimpleNamespace(
        digest="ab" * 32,
        track_id="track",
        netuid=120,
        track_start_block=100,
        model=SimpleNamespace(
            max_bytes=1_000,
            tensor_schema={},
            selector=SimpleNamespace(head_dim=1, routes=("route",)),
        ),
        cache=SimpleNamespace(
            exact_max_bytes=1_000,
        ),
        scoring=SimpleNamespace(
            budgets=[SimpleNamespace(target_blocks=1)],
            stratum_weight_map={"stratum": 1.0},
            case_count=1,
        ),
        initial_champion=SimpleNamespace(hotkey="uid0-hotkey"),
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

    async def read_state(_client, *, netuid: int, mechid: int):
        assert netuid == 120
        assert mechid == 1
        return chain

    async def submit(plan, **kwargs):
        submitted["plan"] = plan
        submitted.update(kwargs)
        return WeightReceipt("0xabc", "120-1", "finalized")

    monkeypatch.setattr(run_module, "read_weight_chain_state", read_state)
    monkeypatch.setattr(run_module, "submit_weights", submit)

    async def execute():
        track = parse_track_spec({**valid_spec(), "netuid": 120, "mechid": 1})
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
    track = parse_track_spec({**valid_spec(), "netuid": 120, "mechid": 1})
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
            ),
            "scoring_settings": ScoringSettings(),
            "evaluator": evaluate,
            "worker_id": "validator-a",
            "downloader": download,
        }
        paused = asyncio.run(
            _complete_cycle(client, store, minimum_finalized_block=102, **arguments)
        )
        first = asyncio.run(_complete_cycle(client, store, **arguments))

        assert paused.scan.submissions_found == 1
        assert paused.admission is paused.scoring is None
        assert first.scan.blocks_scanned == 0
        assert first.admission is not None
        assert first.admission.status is AdmissionStatus.ACCEPTED
        assert first.scoring is not None
        assert first.scoring.state is ChallengeState.SUCCEEDED
        assert store.leaderboard()[0].final_score == pytest.approx(0.7)


def test_cycle_commits_a_stage_even_when_the_chain_advances(tmp_path) -> None:
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
                admission_settings=AdmissionSettings(
                    cache_dir=tmp_path / "cache",
                    max_model_bytes=10_000,
                    tensor_schema={"selector.weight": TensorSpec((2, 3), "float32")},
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
        assert cycle.scoring.state is ChallengeState.SUCCEEDED
        challenge = store.public_snapshot()["challenges"][0]
        assert challenge["state"] == ChallengeState.SUCCEEDED
        assert challenge["final_score"] == 0.9
        assert len(challenge["stages"]) == 1


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
    prefix = ()

    class Client(FakeClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    class Publisher:
        async def recover(self, _track):
            if load_fail:
                raise RuntimeError("public ledger unavailable on restart")
            return SimpleNamespace(
                matches=(), required_finalized_block=0, committed_match_count=0,
            )

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
