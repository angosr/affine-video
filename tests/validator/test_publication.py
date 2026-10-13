import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from validator.publication import (
    PublicArtifactError,
    PublicArtifacts,
    R2Publisher,
    build_public_artifacts,
    decode_weight_artifact,
    encode_weight_artifact,
    fetch_weight_artifact,
    validate_public_weights_url,
)
from validator.queue import ChainPosition
from validator.store import LeaderboardEntry
from validator.weights import WeightArtifact, WeightChainState, WeightPlan


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
def test_r2_rejects_unsafe_object_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        R2Publisher(object(), bucket="public-data", prefix=prefix)


def test_r2_publishes_results_before_consumable_weights() -> None:
    calls = []

    class FakeS3:
        def put_object(self, **kwargs) -> None:
            calls.append(kwargs)

    publisher = R2Publisher(FakeS3(), bucket="public-data", prefix="veda/track")
    artifacts = PublicArtifacts(120, b"weights", b"results")
    asyncio.run(publisher.publish(artifacts))

    assert [call["Key"] for call in calls] == [
        "veda/track/results.json",
        "veda/track/weights.json",
    ]
    assert [call["Body"] for call in calls] == [b"results", b"weights"]
    assert all(call["CacheControl"] == "no-store, max-age=0" for call in calls)


def test_public_artifacts_bind_results_and_weights_to_one_chain_snapshot() -> None:
    reward = LeaderboardEntry(
        hotkey="alice",
        challenge_id="challenge",
        final_score=0.8,
        attention_cost=0.2,
        score_block=110,
        provisional=False,
        position=ChainPosition(100, 0, 0),
    )
    snapshot = {
        "leaderboard": [{"hotkey": "alice"}],
        "rewards": [{"hotkey": "alice", "challenge_id": "challenge"}],
        "challenges": [{"challenge_id": "challenge"}],
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot_inputs=lambda: (snapshot, [reward]),
    )
    track = SimpleNamespace(
        track_id="track",
        digest="ab" * 32,
        netuid=7,
        scoring=SimpleNamespace(minimum_frontier_gain=0.001),
    )
    chain = WeightChainState(
        120,
        {"alice": 4},
        1,
        1.0,
        10,
        {"alice": 0},
    )

    artifacts = build_public_artifacts(store, track, chain, mechid=2, version_key=9)
    assert artifacts is not None
    public_weights = decode_weight_artifact(artifacts.weights)
    public_results = json.loads(artifacts.results)
    assert public_weights.source_block == public_results["source_block"] == 120
    assert public_weights.plan is not None
    assert public_weights.plan.hotkeys == ("alice",)
    assert public_results["challenges"] == snapshot["challenges"]

    store.chain_cursor = lambda: (119, "0x" + "11" * 32)
    assert build_public_artifacts(store, track, chain, mechid=2, version_key=9) is None
