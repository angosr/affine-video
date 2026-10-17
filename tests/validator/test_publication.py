import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator import publication as publication_module
from validator.publication import (
    PublicArtifactError,
    PublicArtifacts,
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
        self.calls = []

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]


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
        (("matches/000000000001.json", b"match"),),
    )
    asyncio.run(publisher.publish(artifacts))

    writes = [call for call in client.calls if "Key" in call and "Body" in call]
    assert [call["Key"] for call in writes] == [
        "veda/track/track.json",
        "veda/track/matches/000000000001.json",
        "veda/track/summary.json",
        "veda/track/weights.json",
    ]
    assert writes[0]["CacheControl"].endswith("immutable")
    assert writes[1]["CacheControl"].endswith("immutable")
    assert writes[2]["CacheControl"] == writes[3]["CacheControl"] == "no-store, max-age=0"

    client.calls.clear()
    asyncio.run(publisher.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == [
        "veda/track/summary.json",
        "veda/track/weights.json",
    ]

    conflicting = PublicArtifacts(
        120,
        b"track",
        b"summary",
        b"weights",
        (("matches/000000000001.json", b"different"),),
    )
    with pytest.raises(PublicArtifactError, match="conflicts"):
        asyncio.run(publisher.publish(conflicting))


def test_initial_state_publisher_stops_after_competition_track_exists() -> None:
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/main")

    asyncio.run(
        publisher.publish_initial_state(
            status=b"status",
            weights=b"weights",
            system_matches=(("system_matches/000000000001.json", b"system-match"),),
        )
    )
    assert client.objects["tracks/main/status.json"] == b"status"
    assert client.objects["tracks/main/weights.json"] == b"weights"
    assert client.objects["tracks/main/system_matches/000000000001.json"] == b"system-match"

    client.objects["tracks/main/track.json"] = b"track"
    with pytest.raises(PublicArtifactError, match="taken ownership"):
        asyncio.run(publisher.publish_initial_state(status=b"new", weights=b"new"))


def test_storage_ledger_rebuilds_and_verifies_every_committed_match() -> None:
    track = parse_track_spec(valid_spec())
    snapshot = {
        "challenges": [
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
                "evaluation_batch": {
                    "sha256": "de" * 32,
                    "generated_block": 101,
                    "source_digest": "ef" * 32,
                },
                "score_block": 110,
                "stages": [],
            }
        ]
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
    )
    chain = WeightChainState(120, {"alice": 4}, 1, 1.0, 10, {"alice": 0})
    artifacts = build_public_artifacts(store, track, chain, mechid=0, version_key=0)
    assert artifacts is not None

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="veda/track")
    asyncio.run(repository.publish(artifacts))
    ledger = asyncio.run(repository.load(track))

    assert ledger.source_block == 120
    assert ledger.required_finalized_block == 120
    assert len(ledger.matches) == 1
    assert ledger.matches[0].challenge_id == "challenge"

    original_match = client.objects.pop("veda/track/matches/000000000001.json")
    with pytest.raises(PublicArtifactError, match="committed match is missing"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))
    client.objects["veda/track/matches/000000000001.json"] = original_match

    client.objects["veda/track/matches/000000000001.json"] = b"{}"
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

    artifacts = build_public_artifacts(store, track, chain, mechid=0, version_key=0)

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


def test_storage_loader_recovers_match_uploaded_before_mutable_summary() -> None:
    track = parse_track_spec(valid_spec())
    first = {
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
        "evaluation_batch": {
            "sha256": "de" * 32,
            "generated_block": 101,
            "source_digest": "ef" * 32,
        },
        "score_block": 110,
        "stages": [],
    }
    snapshot = {"challenges": [first]}
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
    )
    chain = WeightChainState(120, {"alice": 4, "bob": 5}, 1, 1.0, 10, {})
    committed = build_public_artifacts(store, track, chain, mechid=0, version_key=0)
    assert committed is not None
    client = MemoryS3()
    asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").publish(committed))

    second = {
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
        "evaluation_batch": {
            "sha256": "fa" * 32,
            "generated_block": 122,
            "source_digest": "fb" * 32,
        },
        "score_block": 125,
    }
    snapshot["challenges"] = [first, second]
    store.chain_cursor = lambda: (130, "0x" + "13" * 32)
    newer = build_public_artifacts(
        store,
        track,
        WeightChainState(130, {"alice": 4, "bob": 5}, 1, 1.0, 10, {}),
        mechid=0,
        version_key=0,
    )
    assert newer is not None
    orphan_name, orphan_payload = newer.matches[1]
    client.objects[f"veda/track/{orphan_name}"] = orphan_payload

    recovered = asyncio.run(
        S3Repository(client, bucket="public-data", prefix="veda/track").load(track)
    )
    assert recovered.source_block == 120
    assert recovered.required_finalized_block == 125
    assert [match.challenge_id for match in recovered.matches] == [
        "challenge-1",
        "challenge-2",
    ]


def test_public_artifacts_bind_summary_matches_and_weights_to_one_snapshot(monkeypatch) -> None:
    snapshot = {
        "challenges": [
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
                "evaluation_batch": {
                    "sha256": "de" * 32,
                    "generated_block": 101,
                    "source_digest": "ef" * 32,
                },
                "score_block": 110,
                "stages": [],
            }
        ],
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
    )
    track = SimpleNamespace(
        track_id="track",
        digest="ab" * 32,
        netuid=7,
        track_start_block=100,
        initial_champion=SimpleNamespace(
            uid=0,
            hotkey="genesis-hotkey",
            repo_id="Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
            revision="76f202874608115408d73280be9531c4ab888242",
            score=0.5,
        ),
        scoring=SimpleNamespace(minimum_frontier_gain=0.001),
        cache=SimpleNamespace(minimum_generation_delay_blocks=1),
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

    artifacts = build_public_artifacts(store, track, chain, mechid=2, version_key=9)
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

    store.chain_cursor = lambda: (119, "0x" + "11" * 32)
    assert build_public_artifacts(store, track, chain, mechid=2, version_key=9) is None

    store.chain_cursor = lambda: (120, "0x" + "12" * 32)
    snapshot["challenges"][0]["score_block"] = 121
    with pytest.raises(PublicArtifactError, match="newer"):
        build_public_artifacts(store, track, chain, mechid=2, version_key=9)
