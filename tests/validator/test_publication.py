import asyncio
import hashlib
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator import chunks
from validator import publication as publication_module
from validator.publication import (
    PublicArtifactError,
    PublicArtifacts,
    PublicFile,
    S3Repository,
    build_public_artifacts,
    decode_weight_artifact,
    encode_weight_artifact,
    fetch_weight_artifact,
    validate_public_weights_url,
)
from validator.weights import WeightArtifact, WeightChainState, WeightPlan


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class MemoryS3:
    def __init__(self) -> None:
        self.objects = {}
        self.metadata = {}
        self.calls = []

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.metadata[kwargs["Key"]] = kwargs.get("Metadata", {})

    def head_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata.get(Key, {}),
        }


def weight_artifact() -> WeightArtifact:
    return WeightArtifact(
        track_id="track",
        track_spec_digest="ab" * 32,
        netuid=7,
        mechid=2,
        version_key=9,
        source_block=120,
        plan=WeightPlan(120, ("alice", "bob"), (1, 4), (0.6, 0.4)),
    )


def two_stage_challenge(value: dict) -> dict:
    result = dict(value)
    if result.get("state") != "SUCCEEDED":
        return result
    final_batch = dict(result["evaluation_batch"])
    final_batch["generated_block"] = max(final_batch["generated_block"], result["block_height"] + 5)
    exact_batch = {
        "sha256": hashlib.sha256((result["challenge_id"] + ":exact").encode()).hexdigest(),
        "generated_block": result["block_height"] + 4,
        "plan_digest": hashlib.sha256(
            (result["challenge_id"] + ":exact-plan").encode()
        ).hexdigest(),
        "seed_block_hash": "0x" + "cd" * 32,
    }
    exact_decision = result["block_height"] + 4
    result["evaluation_batch"] = final_batch
    result["score_block"] = max(result["score_block"], final_batch["generated_block"])
    result["stages"] = [
        {
            "stage": "SCORING",
            "score": result["final_score"],
            "score_block": exact_decision,
            "incumbent_score": result["incumbent_score"],
            "paired_gain_lower_bound": result["paired_gain_lower_bound"],
            "evaluation_batch": exact_batch,
        },
        {
            "stage": "CONFIRMING",
            "score": result["final_score"],
            "score_block": result["score_block"],
            "incumbent_score": result["incumbent_score"],
            "paired_gain_lower_bound": result["paired_gain_lower_bound"],
            "evaluation_batch": final_batch,
        },
    ]
    return result


def test_weight_artifact_has_one_strict_canonical_representation() -> None:
    encoded = encode_weight_artifact(weight_artifact())
    decoded = decode_weight_artifact(encoded)

    assert decoded == weight_artifact()
    assert encoded == encode_weight_artifact(decoded)
    with pytest.raises(PublicArtifactError, match="duplicate"):
        decode_weight_artifact(
            b'{"track_id":"a","track_id":"b","track_spec_digest":"'
            + b"ab" * 32
            + b'","netuid":7,"mechid":0,"version_key":0,"source_block":1,"weights":null}'
        )


def test_public_repository_rejects_active_track_pointer_rollback() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})
    current = PublicArtifacts(
        source_block=10,
        track=track,
        summary=encoded({"source_block": 10, "match_count": 1, "marker": "current"}),
        weights=b"{}",
        matches=(),
    )
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/main")
    asyncio.run(repository.publish(current))

    with pytest.raises(PublicArtifactError, match="rollback"):
        asyncio.run(
            repository.publish(
                PublicArtifacts(
                    source_block=9,
                    track=track,
                    summary=encoded({"source_block": 9, "match_count": 1, "marker": "older"}),
                    weights=b"{}",
                    matches=(),
                )
            )
        )
    with pytest.raises(PublicArtifactError, match="two public snapshots"):
        asyncio.run(
            repository.publish(
                PublicArtifacts(
                    source_block=10,
                    track=track,
                    summary=encoded({"source_block": 10, "match_count": 1, "marker": "conflict"}),
                    weights=b"{}",
                    matches=(),
                )
            )
        )


def test_fetch_weight_artifact_is_https_only_and_size_bounded() -> None:
    payload = encode_weight_artifact(weight_artifact())

    async def run() -> WeightArtifact:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_weight_artifact("https://public.example/weights.json", client=client)

    assert asyncio.run(run()) == weight_artifact()

    async def oversized() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b"x" * 20))
        async with httpx.AsyncClient(transport=transport) as client:
            await fetch_weight_artifact(
                "https://public.example/weights.json", client=client, max_bytes=10
            )

    with pytest.raises(PublicArtifactError, match="byte limit"):
        asyncio.run(oversized())
    with pytest.raises(ValueError, match="HTTPS"):
        validate_public_weights_url("http://public.example/weights.json")


@pytest.mark.parametrize("prefix", ["../track", "track//main", "track name", "track\nmain"])
def test_storage_rejects_unsafe_object_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        S3Repository(object(), bucket="public-data", prefix=prefix)


def test_storage_factory_requires_complete_safe_environment(monkeypatch) -> None:
    for name in ("STORAGE_URL", "ACCESS_KEY", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="required"):
        S3Repository.from_environment(bucket="public-data", prefix="veda/track")

    monkeypatch.setenv("STORAGE_URL", "http://storage.example/s3")
    monkeypatch.setenv("ACCESS_KEY", "access")
    monkeypatch.setenv("SECRET_ACCESS_KEY", "secret")
    with pytest.raises(ValueError, match="HTTPS"):
        S3Repository.from_environment(bucket="public-data", prefix="veda/track")


def test_storage_publishes_immutable_facts_before_summary_and_weights() -> None:
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="veda/track")
    artifacts = PublicArtifacts(
        120,
        b"track",
        b"summary",
        b"weights",
        (("matches/track/000000000001.json", b"match"),),
    )
    asyncio.run(publisher.publish(artifacts))

    writes = [call for call in client.calls if "Key" in call and "Body" in call]
    assert [call["Key"] for call in writes] == [
        "veda/track/track.json",
        "veda/track/matches/track/000000000001.json",
        "veda/track/summary.json",
        "veda/track/weights.json",
    ]
    assert writes[0]["CacheControl"] == "no-store, max-age=0"
    assert writes[1]["CacheControl"].endswith("immutable")
    assert writes[2]["CacheControl"] == writes[3]["CacheControl"] == "no-store, max-age=0"

    client.calls.clear()
    asyncio.run(publisher.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == [
        "veda/track/track.json",
        "veda/track/summary.json",
        "veda/track/weights.json",
    ]

    conflicting = PublicArtifacts(
        120,
        b"track",
        b"summary",
        b"weights",
        (("matches/track/000000000001.json", b"different"),),
    )
    with pytest.raises(PublicArtifactError, match="conflicts"):
        asyncio.run(publisher.publish(conflicting))


def test_runtime_manifest_is_content_addressed_and_published_before_track() -> None:
    payload = b"runtime-manifest"
    digest = hashlib.sha256(payload).hexdigest()
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/main")
    artifacts = PublicArtifacts(
        120,
        b"track",
        b"summary",
        b"weights",
        (),
        runtime_manifest=(
            f"runtime/{digest}/manifest.json",
            payload,
        ),
    )

    asyncio.run(publisher.publish(artifacts))

    writes = [call["Key"] for call in client.calls if "Body" in call]
    assert writes[:2] == [
        f"tracks/main/runtime/{digest}/manifest.json",
        "tracks/main/track.json",
    ]


def test_storage_publishes_large_files_as_bounded_verified_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 4)
    source = tmp_path / "batch.npz"
    source.write_bytes(b"abcdefghij")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/main")
    artifacts = PublicArtifacts(
        120,
        b"track",
        b"summary",
        b"weights",
        (),
        files=(PublicFile(f"batches/{digest}/manifest.json", source, digest),),
    )

    asyncio.run(publisher.publish(artifacts))

    prefix = f"tracks/main/batches/{digest}/"
    assert sorted(key for key in client.objects if key.startswith(prefix)) == [
        prefix + "manifest.json",
        prefix + "part-00000",
        prefix + "part-00001",
        prefix + "part-00002",
    ]

    source.unlink()
    client.calls.clear()
    restarted = S3Repository(client, bucket="public-data", prefix="tracks/main")
    asyncio.run(restarted.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == [
        "tracks/main/track.json",
        "tracks/main/summary.json",
        "tracks/main/weights.json",
    ]

    del client.objects[prefix + "part-00001"]
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        asyncio.run(
            S3Repository(client, bucket="public-data", prefix="tracks/main").publish(artifacts)
        )


def test_storage_ledger_rebuilds_and_verifies_every_committed_match() -> None:
    track = parse_track_spec(valid_spec())
    snapshot = {
        "challenges": [
            two_stage_challenge(
                {
                    "challenge_id": "challenge",
                    "submission_id": "submission",
                    "hotkey": "alice",
                    "repo_id": "owner/model",
                    "revision": "12" * 20,
                    "block_height": 100,
                    "extrinsic_index": 0,
                    "event_index": 0,
                    "state": "SUCCEEDED",
                    "error_code": None,
                    "file_sha256": "ab" * 32,
                    "model_digest": "cd" * 32,
                    "model_size_bytes": 123,
                    "final_score": 0.8,
                    "incumbent_score": 0.5,
                    "paired_gain_lower_bound": 0.3,
                    "evaluation_batch": {
                        "sha256": "de" * 32,
                        "generated_block": 104,
                        "plan_digest": "ef" * 32,
                        "seed_block_hash": "0x" + "ab" * 32,
                    },
                    "score_block": 110,
                    "stages": [],
                }
            )
        ]
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
    )
    chain = WeightChainState(120, {"alice": 4}, 1, 1.0, 10, {"alice": 0})
    artifacts = build_public_artifacts(store, track, chain, version_key=0)
    assert artifacts is not None

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="veda/track")
    asyncio.run(repository.publish(artifacts))
    ledger = asyncio.run(repository.load(track))

    assert ledger.source_block == 120
    assert ledger.required_finalized_block == 120
    assert len(ledger.matches) == 1
    assert ledger.matches[0].challenge_id == "challenge"

    original_match = client.objects.pop("veda/track/matches/veda-test-v1/000000000001.json")
    with pytest.raises(PublicArtifactError, match="committed match is missing"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))
    client.objects["veda/track/matches/veda-test-v1/000000000001.json"] = original_match

    client.objects["veda/track/matches/veda-test-v1/000000000001.json"] = b"{}"
    with pytest.raises(PublicArtifactError, match="match is invalid"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))


def test_empty_track_publishes_uid_zero_veda_as_genesis_champion() -> None:
    track = parse_track_spec(valid_spec())
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": []},
        active_competitors=lambda _mapping: [],
    )
    chain = WeightChainState(
        120,
        {track.initial_champion.hotkey: 0},
        1,
        1.0,
        10,
        {track.initial_champion.hotkey: 0},
    )

    artifacts = build_public_artifacts(store, track, chain, version_key=0)

    assert artifacts is not None
    weights = decode_weight_artifact(artifacts.weights)
    summary = json.loads(artifacts.summary)
    status = json.loads(artifacts.status)
    assert weights.plan == WeightPlan(
        120,
        (track.initial_champion.hotkey,),
        (0,),
        (1.0,),
    )
    assert summary["match_count"] == 0
    assert summary["champion"]["uid"] == 0
    assert summary["champion"]["source"] == "VEDA_BASELINE"
    assert status["participants"][0]["challenge_id"] == "genesis"
    assert status["participants"][0]["champion"] is True
    assert status["track_phase"] == "ACCEPTING"
    assert status["last_submission_block"] == 996
    assert status["track_close_block"] == 1150


def test_track_abort_stops_weight_refresh_after_close_grace() -> None:
    track = parse_track_spec(valid_spec())
    snapshot = {
        "challenges": [
            {
                "challenge_id": "blocked-challenge",
                "submission_id": "blocked-submission",
                "state": "SYSTEM_BLOCKED",
            }
        ]
    }
    block = track.track_close_block + 1
    store = SimpleNamespace(
        chain_cursor=lambda: (block, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
    )
    chain = WeightChainState(
        block,
        {track.initial_champion.hotkey: 0},
        1,
        1.0,
        10,
        {track.initial_champion.hotkey: 0},
    )

    artifacts = build_public_artifacts(store, track, chain, version_key=0)

    assert artifacts is not None
    assert artifacts.weights is None
    status = json.loads(artifacts.status)
    assert status["track_phase"] == "TRACK_ABORTED"
    assert status["abort"] == {
        "reason": "UNFINISHED_SYSTEM_WORK_AFTER_CLOSE_GRACE",
        "abort_block": track.track_close_block + 1,
    }
    assert artifacts.abort is not None

    client = MemoryS3()
    existing = b"existing-weights"
    client.objects["tracks/main/weights.json"] = existing
    repository = S3Repository(client, bucket="public-data", prefix="tracks/main")
    asyncio.run(repository.publish(artifacts))
    assert client.objects["tracks/main/weights.json"] == existing

    ledger = asyncio.run(repository.load(track))
    assert ledger.aborted is True
    recovered_store = SimpleNamespace(
        chain_cursor=lambda: (block, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": []},
        active_competitors=lambda _mapping: [],
    )
    recovered = build_public_artifacts(
        recovered_store,
        track,
        chain,
        version_key=0,
        track_aborted=ledger.aborted,
    )
    assert recovered is not None and recovered.weights is None
    assert json.loads(recovered.status)["track_phase"] == "TRACK_ABORTED"


def test_storage_loader_recovers_match_uploaded_before_mutable_summary() -> None:
    track = parse_track_spec(valid_spec())
    first = two_stage_challenge(
        {
            "challenge_id": "challenge-1",
            "submission_id": "submission-1",
            "hotkey": "alice",
            "repo_id": "owner/model-1",
            "revision": "12" * 20,
            "block_height": 100,
            "extrinsic_index": 0,
            "event_index": 0,
            "state": "SUCCEEDED",
            "error_code": None,
            "file_sha256": "ab" * 32,
            "model_digest": "cd" * 32,
            "model_size_bytes": 123,
            "final_score": 0.8,
            "incumbent_score": 0.5,
            "paired_gain_lower_bound": 0.3,
            "evaluation_batch": {
                "sha256": "de" * 32,
                "generated_block": 104,
                "plan_digest": "ef" * 32,
                "seed_block_hash": "0x" + "ab" * 32,
            },
            "score_block": 110,
            "stages": [],
        }
    )
    snapshot = {"challenges": [first]}
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
    )
    chain = WeightChainState(120, {"alice": 4, "bob": 5}, 1, 1.0, 10, {})
    committed = build_public_artifacts(store, track, chain, version_key=0)
    assert committed is not None
    client = MemoryS3()
    asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").publish(committed))

    second = two_stage_challenge(
        {
            **first,
            "challenge_id": "challenge-2",
            "submission_id": "submission-2",
            "hotkey": "bob",
            "repo_id": "owner/model-2",
            "revision": "34" * 20,
            "block_height": 121,
            "model_digest": "ef" * 32,
            "final_score": 0.9,
            "incumbent_score": 0.8,
            "paired_gain_lower_bound": 0.1,
            "evaluation_batch": {
                "sha256": "fa" * 32,
                "generated_block": 125,
                "plan_digest": "fb" * 32,
                "seed_block_hash": "0x" + "ab" * 32,
            },
            "score_block": 125,
        }
    )
    snapshot["challenges"] = [first, second]
    store.chain_cursor = lambda: (130, "0x" + "13" * 32)
    newer = build_public_artifacts(
        store,
        track,
        WeightChainState(130, {"alice": 4, "bob": 5}, 1, 1.0, 10, {}),
        version_key=0,
    )
    assert newer is not None
    orphan_name, orphan_payload = newer.matches[1]
    client.objects[f"veda/track/{orphan_name}"] = orphan_payload

    recovered = asyncio.run(
        S3Repository(client, bucket="public-data", prefix="veda/track").load(track)
    )
    assert recovered.source_block == 120
    assert recovered.required_finalized_block == 126
    assert [match.challenge_id for match in recovered.matches] == [
        "challenge-1",
        "challenge-2",
    ]


def test_public_artifacts_bind_summary_matches_and_weights_to_one_snapshot(
    monkeypatch, tmp_path
) -> None:
    snapshot = {
        "challenges": [
            two_stage_challenge(
                {
                    "challenge_id": "challenge",
                    "submission_id": "submission",
                    "hotkey": "alice",
                    "repo_id": "owner/model",
                    "revision": "12" * 20,
                    "block_height": 100,
                    "extrinsic_index": 0,
                    "event_index": 0,
                    "state": "SUCCEEDED",
                    "error_code": None,
                    "file_sha256": "ab" * 32,
                    "model_digest": "cd" * 32,
                    "model_size_bytes": 123,
                    "final_score": 0.8,
                    "incumbent_score": 0.5,
                    "paired_gain_lower_bound": 0.3,
                    "evaluation_batch": {
                        "sha256": "de" * 32,
                        "generated_block": 104,
                        "plan_digest": "ef" * 32,
                        "seed_block_hash": "0x" + "ab" * 32,
                    },
                    "score_block": 110,
                    "stages": [],
                }
            )
        ],
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
        model_path=lambda _challenge_id: None,
    )
    track = SimpleNamespace(
        track_id="track",
        digest="ab" * 32,
        netuid=7,
        mechid=2,
        track_start_block=100,
        maximum_generation_batches=10,
        track_close_grace_blocks=50,
        last_submission_block=999,
        track_close_block=1150,
        initial_champion=SimpleNamespace(
            uid=0,
            hotkey="genesis-hotkey",
            repo_id="Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
            revision="76f202874608115408d73280be9531c4ab888242",
            score=0.5,
        ),
        scoring=SimpleNamespace(minimum_frontier_gain=0.001),
        full_video=SimpleNamespace(minimum_gain=0.001),
        cache=SimpleNamespace(
            minimum_generation_delay_blocks=1,
            generation_interval_blocks=100,
        ),
    )
    chain = WeightChainState(
        120,
        {"genesis-hotkey": 0, "alice": 4},
        1,
        1.0,
        10,
        {"alice": 0},
    )
    monkeypatch.setattr(publication_module, "track_spec_dict", lambda _track: {"track_id": "track"})

    artifacts = build_public_artifacts(store, track, chain, version_key=9)
    assert artifacts is not None
    public_weights = decode_weight_artifact(artifacts.weights)
    public_summary = json.loads(artifacts.summary)
    public_match = json.loads(artifacts.matches[0][1])
    assert public_weights.source_block == public_summary["source_block"] == 120
    assert public_weights.plan is not None
    assert public_weights.plan.hotkeys == ("alice",)
    assert public_summary["champion"]["hotkey"] == "alice"
    assert public_summary["match_count"] == 1
    assert public_summary["leaderboard_total"] == 2
    assert public_match["outcome"] == "CHALLENGER_WON"

    withheld = build_public_artifacts(
        store,
        track,
        chain,
        version_key=9,
        exact_cache_dir=tmp_path,
        batch_reference_resolver=lambda _digest: ("challenge", "pending"),
        model_resolver=lambda _digest, _size: tmp_path / "model.safetensors",
    )
    assert withheld is not None
    assert not any(item.object_name.startswith("batches/") for item in withheld.files)
    disclosed = build_public_artifacts(
        store,
        track,
        chain,
        version_key=9,
        exact_cache_dir=tmp_path,
        batch_reference_resolver=lambda _digest: ("challenge",),
        model_resolver=lambda _digest, _size: tmp_path / "model.safetensors",
    )
    assert disclosed is not None
    assert sum(item.object_name.startswith("batches/") for item in disclosed.files) == 2

    store.chain_cursor = lambda: (119, "0x" + "11" * 32)
    assert build_public_artifacts(store, track, chain, version_key=9) is None

    store.chain_cursor = lambda: (120, "0x" + "12" * 32)
    snapshot["challenges"][0]["score_block"] = 121
    with pytest.raises(PublicArtifactError, match="newer"):
        build_public_artifacts(store, track, chain, version_key=9)
