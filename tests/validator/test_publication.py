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
from validator.production.batches import BatchDisclosure, DisclosureFile
from validator.production.commitment import SourceLeaf, SourcePool, canonical_json
from validator.production.plan import (
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    build_plan,
    generation_seed,
    select_source_proofs,
)
from validator.publication import (
    PublicArtifactError,
    PublicArtifacts,
    PublicFile,
    S3Repository,
    build_public_artifacts,
    decode_weight_artifact,
    encode_weight_artifact,
    fetch_track_identity,
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
        self.before_put = None

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {
            "Body": io.BytesIO(self.objects[Key]),
            "ETag": f'"{hashlib.md5(self.objects[Key], usedforsecurity=False).hexdigest()}"',
        }

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.before_put is not None:
            self.before_put(kwargs)
        existing = self.objects.get(kwargs["Key"])
        etag = (
            None if existing is None else hashlib.md5(existing, usedforsecurity=False).hexdigest()
        )
        if kwargs.get("IfNoneMatch") == "*" and existing is not None:
            raise precondition_failed()
        if "IfMatch" in kwargs and kwargs["IfMatch"] != etag:
            raise precondition_failed()
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.metadata[kwargs["Key"]] = kwargs.get("Metadata", {})

    def head_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {
            "ContentLength": len(self.objects[Key]),
            "ETag": f'"{hashlib.md5(self.objects[Key], usedforsecurity=False).hexdigest()}"',
            "Metadata": self.metadata.get(Key, {}),
        }

    def delete_object(self, *, Bucket, Key):
        del Bucket
        self.objects.pop(Key, None)
        self.metadata.pop(Key, None)


def precondition_failed():
    error = RuntimeError("precondition failed")
    error.response = {
        "Error": {"Code": "PreconditionFailed"},
        "ResponseMetadata": {"HTTPStatusCode": 412},
    }
    return error


def weight_artifact() -> WeightArtifact:
    return WeightArtifact(
        track_id="track",
        track_spec_digest="ab" * 32,
        netuid=7,
        mechid=2,
        version_key=9,
        source_block=120,
        plan=WeightPlan(120, ("alice", "bob"), (1, 4), (0.6, 0.4)),
        match_sequence=2,
        terminal_match_sha256="cd" * 32,
    )


def test_only_pinned_legacy_weights_may_omit_abort_fence() -> None:
    legacy = {
        "track_id": "veda-sn120-v1",
        "track_spec_digest": (
            "0fcae991aef29f87b97fbb07cf79c49e5eb9cc2a5d9aecdd2a566443e2b81dd5"
        ),
        "netuid": 120,
        "mechid": 1,
        "version_key": 0,
        "source_block": 9305106,
        "weights": [{"hotkey": "hotkey", "uid": 0, "weight": 1.0}],
    }
    decoded = decode_weight_artifact(json.dumps(legacy).encode())
    assert decoded.track_aborted is False
    assert decoded.match_sequence == 0
    assert decoded.terminal_match_sha256 is None

    legacy["track_id"] = "another-track"
    with pytest.raises(PublicArtifactError, match="fields"):
        decode_weight_artifact(json.dumps(legacy).encode())


def pointer_track(track_id: str = "track") -> bytes:
    return json.dumps(
        {"spec": {"track_id": track_id}, "track_spec_digest": "ab" * 32},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def pointer_weights(
    block: int,
    track_id: str = "track",
    *,
    match_sequence: int = 0,
    terminal_match_sha256: str | None = None,
) -> bytes:
    return encode_weight_artifact(
        WeightArtifact(
            track_id,
            "ab" * 32,
            7,
            0,
            0,
            block,
            WeightPlan(block, ("hk",), (1,), (1.0,)),
            match_sequence=match_sequence,
            terminal_match_sha256=terminal_match_sha256,
        )
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
        summary=encoded(
            {"track_id": "track", "source_block": 10, "match_count": 1, "marker": "current"}
        ),
        weights=pointer_weights(10),
        matches=(),
    )
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(current))

    with pytest.raises(PublicArtifactError, match="rollback"):
        asyncio.run(
            repository.publish(
                PublicArtifacts(
                    source_block=9,
                    track=track,
                    summary=encoded(
                        {
                            "track_id": "track",
                            "source_block": 9,
                            "match_count": 1,
                            "marker": "older",
                        }
                    ),
                    weights=pointer_weights(9),
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
                    summary=encoded(
                        {
                            "track_id": "track",
                            "source_block": 10,
                            "match_count": 1,
                            "marker": "conflict",
                        }
                    ),
                    weights=pointer_weights(10),
                    matches=(),
                )
            )
        )


def test_weights_reject_a_higher_block_with_a_stale_match_prefix_before_mutation() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})
    current = PublicArtifacts(
        source_block=10,
        track=track,
        summary=encoded({"track_id": "track", "source_block": 10, "match_count": 2}),
        weights=pointer_weights(
            10,
            match_sequence=2,
            terminal_match_sha256="22" * 32,
        ),
        matches=(),
    )
    stale = PublicArtifacts(
        source_block=11,
        track=track,
        summary=encoded({"track_id": "track", "source_block": 11, "match_count": 1}),
        weights=pointer_weights(
            11,
            match_sequence=1,
            terminal_match_sha256="11" * 32,
        ),
        matches=(),
    )
    conflicting = PublicArtifacts(
        source_block=12,
        track=track,
        summary=encoded({"track_id": "track", "source_block": 12, "match_count": 2}),
        weights=pointer_weights(
            12,
            match_sequence=2,
            terminal_match_sha256="33" * 32,
        ),
        matches=(),
    )
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(current))
    weights_before = client.objects["tracks/test/weights.json"]
    summary_before = client.objects["tracks/test/summary.json"]

    with pytest.raises(PublicArtifactError, match="rollback"):
        asyncio.run(repository.publish(stale))
    assert client.objects["tracks/test/weights.json"] == weights_before
    assert client.objects["tracks/test/summary.json"] == summary_before

    with pytest.raises(PublicArtifactError, match="two histories"):
        asyncio.run(repository.publish(conflicting))
    assert client.objects["tracks/test/weights.json"] == weights_before
    assert client.objects["tracks/test/summary.json"] == summary_before


def test_public_repository_capability_probe_checks_and_cleans_conditions() -> None:
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")

    repository.verify_conditional_writes()

    assert client.objects == {}


def test_public_repository_rejects_backend_that_ignores_conditions() -> None:
    class UnsafeS3(MemoryS3):
        def put_object(self, **kwargs):
            kwargs.pop("IfMatch", None)
            kwargs.pop("IfNoneMatch", None)
            return super().put_object(**kwargs)

    repository = S3Repository(UnsafeS3(), bucket="public-data", prefix="tracks/test")

    with pytest.raises(PublicArtifactError, match="ignores If-None-Match"):
        repository.verify_conditional_writes()


def test_concurrent_stale_summary_write_is_fenced() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})

    def artifacts(block):
        return PublicArtifacts(
            source_block=block,
            track=track,
            summary=encoded(
                {
                    "track_id": "track",
                    "source_block": block,
                    "match_count": 1,
                }
            ),
            weights=pointer_weights(block),
            matches=(),
        )

    client = MemoryS3()
    stale = S3Repository(client, bucket="public-data", prefix="tracks/test")
    current = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(current.publish(artifacts(10)))

    def interleave(arguments):
        if arguments["Key"] != "tracks/test/summary.json" or "IfMatch" not in arguments:
            return
        client.before_put = None
        current._put_mutable("summary.json", artifacts(12).summary)

    client.before_put = interleave
    with pytest.raises(PublicArtifactError, match="fenced"):
        asyncio.run(stale.publish(artifacts(11)))

    assert json.loads(client.objects["tracks/test/summary.json"])["source_block"] == 12


def test_track_abort_fences_concurrent_stale_weight_refresh() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})

    def artifacts(block, *, aborted=False):
        plan = None if aborted else WeightPlan(block, ("hk",), (1,), (1.0,))
        weights = encode_weight_artifact(
            WeightArtifact("track", "ab" * 32, 7, 0, 0, block, plan, aborted)
        )
        return PublicArtifacts(
            source_block=block,
            track=track,
            summary=encoded({"track_id": "track", "source_block": block, "match_count": 0}),
            weights=weights,
            matches=(),
            abort=("events/track/track-aborted.json", encoded({"event": "TRACK_ABORTED"}))
            if aborted
            else None,
        )

    client = MemoryS3()
    stale = S3Repository(client, bucket="public-data", prefix="tracks/test")
    current = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(current.publish(artifacts(9)))

    def interleave(arguments):
        if arguments["Key"] != "tracks/test/weights.json" or "IfMatch" not in arguments:
            return
        client.before_put = None
        asyncio.run(current.publish(artifacts(11, aborted=True)))

    client.before_put = interleave
    with pytest.raises(PublicArtifactError, match="fenced"):
        asyncio.run(stale.publish(artifacts(10)))

    published = decode_weight_artifact(client.objects["tracks/test/weights.json"])
    assert published.source_block == 11
    assert published.plan is None
    assert published.track_aborted is True
    assert "tracks/test/events/track/track-aborted.json" in client.objects

    summary = client.objects["tracks/test/summary.json"]
    with pytest.raises(PublicArtifactError, match="cannot resume"):
        asyncio.run(current.publish(artifacts(12)))
    assert client.objects["tracks/test/summary.json"] == summary


def test_pointer_from_previous_track_cannot_overwrite_active_track() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    old_track = encoded(
        {
            "spec": {"track_id": "old", "track_start_block": 10},
            "track_spec_digest": "ab" * 32,
        }
    )
    new_track = encoded(
        {
            "spec": {"track_id": "new", "track_start_block": 20},
            "track_spec_digest": "cd" * 32,
        }
    )
    old_summary = encoded({"track_id": "old", "source_block": 11, "match_count": 0})
    new_summary = encoded({"track_id": "new", "source_block": 21, "match_count": 0})
    stale_summary = encoded({"track_id": "old", "source_block": 22, "match_count": 0})

    repository._put_mutable("track.json", old_track)
    repository._put_mutable("summary.json", old_summary)
    repository._put_mutable("track.json", new_track)
    repository._put_mutable("summary.json", new_summary)

    with pytest.raises(PublicArtifactError, match="inactive track"):
        repository._put_mutable("summary.json", stale_summary)


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


def test_weight_setter_accepts_a_digest_bound_legacy_track_identity() -> None:
    spec = {
        "track_id": "veda-sn120-v1",
        "netuid": 120,
        "mechid": 1,
        "legacy_scoring": {"format": "retired-v1"},
    }
    digest = hashlib.sha256(canonical_json(spec)).hexdigest()
    payload = canonical_json({"spec": spec, "track_spec_digest": digest})

    async def run():
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_track_identity(
                "https://public.example/track.json",
                client=client,
            )

    identity = asyncio.run(run())
    assert identity.track_id == "veda-sn120-v1"
    assert identity.track_spec_digest == digest
    assert identity.netuid == 120
    assert identity.mechid == 1


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
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (("matches/track/000000000001.json", b"match"),),
    )
    asyncio.run(publisher.publish(artifacts))

    writes = [call for call in client.calls if "Key" in call and "Body" in call]
    assert [call["Key"] for call in writes] == [
        "veda/track/track.json",
        "veda/track/matches/track/000000000001.json",
        "veda/track/weights.json",
        "veda/track/summary.json",
    ]
    assert writes[0]["CacheControl"] == "no-store, max-age=0"
    assert writes[1]["CacheControl"].endswith("immutable")
    assert writes[2]["CacheControl"] == writes[3]["CacheControl"] == "no-store, max-age=0"

    client.calls.clear()
    asyncio.run(publisher.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == []

    conflicting = PublicArtifacts(
        120,
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (("matches/track/000000000001.json", b"different"),),
    )
    with pytest.raises(PublicArtifactError, match="conflicts"):
        asyncio.run(publisher.publish(conflicting))


def test_runtime_manifest_is_content_addressed_and_published_before_track() -> None:
    payload = b"runtime-manifest"
    digest = hashlib.sha256(payload).hexdigest()
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    artifacts = PublicArtifacts(
        120,
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (),
        runtime_manifest=(
            f"runtime/{digest}/manifest.json",
            payload,
        ),
    )

    asyncio.run(publisher.publish(artifacts))

    writes = [call["Key"] for call in client.calls if "Body" in call]
    assert writes[:2] == [
        f"tracks/test/runtime/{digest}/manifest.json",
        "tracks/test/track.json",
    ]


def test_storage_publishes_large_files_as_bounded_verified_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 4)
    source = tmp_path / "batch.npz"
    source.write_bytes(b"abcdefghij")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    artifacts = PublicArtifacts(
        120,
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (),
        files=(PublicFile(f"batches/{digest}/manifest.json", source, digest),),
    )

    asyncio.run(publisher.publish(artifacts))

    prefix = f"tracks/test/batches/{digest}/"
    assert sorted(key for key in client.objects if key.startswith(prefix)) == [
        prefix + "manifest.json",
        prefix + "part-00000",
        prefix + "part-00001",
        prefix + "part-00002",
    ]

    source.unlink()
    client.calls.clear()
    restarted = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(restarted.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == []

    del client.objects[prefix + "part-00001"]
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        asyncio.run(
            S3Repository(client, bucket="public-data", prefix="tracks/test").publish(artifacts)
        )


def test_storage_ledger_rebuilds_and_verifies_every_committed_match(tmp_path) -> None:
    track = parse_track_spec(valid_spec())
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
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
                    "file_sha256": model_sha256,
                    "model_digest": "cd" * 32,
                    "model_size_bytes": model.stat().st_size,
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
        model_path=lambda _challenge_id: str(model),
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

    model_prefix = f"veda/track/models/{track.track_id}/{model_sha256}/"
    model_part = next(key for key in client.objects if key.startswith(model_prefix + "part-"))
    model_bytes = client.objects.pop(model_part)
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))
    client.objects[model_part] = model_bytes

    original_match = client.objects.pop("veda/track/matches/veda-test-v1/000000000001.json")
    with pytest.raises(PublicArtifactError, match="committed match is missing"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))
    client.objects["veda/track/matches/veda-test-v1/000000000001.json"] = original_match

    client.objects["veda/track/matches/veda-test-v1/000000000001.json"] = b"{}"
    with pytest.raises(PublicArtifactError, match="match is invalid"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))


def test_public_artifacts_archive_a_scored_challenger_that_does_not_win(tmp_path) -> None:
    track = parse_track_spec(valid_spec())
    model = tmp_path / "retained-challenger.safetensors"
    model.write_bytes(b"retained challenger")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    batch = {
        "sha256": "de" * 32,
        "generated_block": 104,
        "plan_digest": "ef" * 32,
        "seed_block_hash": "0x" + "ab" * 32,
    }
    challenge = {
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
        "file_sha256": model_sha256,
        "model_digest": "cd" * 32,
        "model_size_bytes": model.stat().st_size,
        "final_score": 0.49,
        "incumbent_score": 0.5,
        "paired_gain_lower_bound": -0.01,
        "evaluation_batch": batch,
        "score_block": 110,
        "stages": [
            {
                "stage": "SCORING",
                "score": 0.49,
                "score_block": 110,
                "incumbent_score": 0.5,
                "paired_gain_lower_bound": -0.01,
                "evaluation_batch": batch,
            }
        ],
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": [challenge]},
        active_competitors=lambda _mapping: [],
        model_path=lambda _challenge_id: str(model),
    )
    chain = WeightChainState(
        120,
        {track.initial_champion.hotkey: 0, "alice": 4},
        1,
        1.0,
        10,
        {},
    )

    artifacts = build_public_artifacts(store, track, chain, version_key=0)

    assert artifacts is not None
    assert json.loads(artifacts.matches[0][1])["outcome"] == "CHAMPION_RETAINED"
    assert len(artifacts.files) == 1
    assert artifacts.files[0].object_name == (
        f"models/{track.track_id}/{model_sha256}/manifest.json"
    )
    assert artifacts.files[0].path == model


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
    assert status["last_submission_block"] == 896
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
    assert decode_weight_artifact(artifacts.weights).plan is None
    status = json.loads(artifacts.status)
    assert status["track_phase"] == "TRACK_ABORTED"
    assert status["abort"] == {
        "reason": "UNFINISHED_SYSTEM_WORK_AFTER_CLOSE_GRACE",
        "abort_block": track.track_close_block + 1,
    }
    assert artifacts.abort is not None

    client = MemoryS3()
    existing = encode_weight_artifact(
        WeightArtifact(
            track.track_id,
            track.digest,
            track.netuid,
            track.mechid,
            0,
            block - 1,
            WeightPlan(
                block - 1,
                (track.initial_champion.hotkey,),
                (0,),
                (1.0,),
            ),
        )
    )
    client.objects["tracks/test/weights.json"] = existing
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(artifacts))
    stopped = decode_weight_artifact(client.objects["tracks/test/weights.json"])
    assert stopped.source_block == block
    assert stopped.plan is None
    assert stopped.track_aborted is True

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
    assert recovered is not None and decode_weight_artifact(recovered.weights).plan is None
    assert json.loads(recovered.status)["track_phase"] == "TRACK_ABORTED"


def test_storage_loader_recovers_match_uploaded_before_mutable_summary(tmp_path) -> None:
    track = parse_track_spec(valid_spec())
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
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
            "file_sha256": model_sha256,
            "model_digest": "cd" * 32,
            "model_size_bytes": model.stat().st_size,
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
        model_path=lambda _challenge_id: str(model),
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


def test_restart_verifies_disclosed_batches_without_restoring_private_files(tmp_path) -> None:
    leaves = tuple(
        SourceLeaf(
            "production",
            f"family-{index}",
            f"source-{index}",
            f"prompt {index}",
            (),
            "all",
            "internal-cleared-v1",
            hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
        )
        for index in range(20)
    )
    audit = SourceLeaf(
        "audit",
        "audit-family",
        "audit-source",
        "audit prompt",
        (),
        "all",
        "internal-cleared-v1",
        hashlib.sha256(b"audit-nonce").hexdigest(),
    )
    pool = SourcePool((*leaves, audit))
    raw_track = valid_spec()
    raw_track["source_pool_merkle_root"] = pool.root
    raw_track["source_commitments"] = {
        "production": {
            "all": [
                leaf.digest for leaf in pool.leaves if leaf.partition == "production"
            ]
        },
        "audit": {"all": [leaf.digest for leaf in pool.leaves if leaf.partition == "audit"]},
    }
    track = parse_track_spec(raw_track)
    block_hash = "0x" + "12" * 32
    proofs = select_source_proofs(
        pool,
        commitments=track.source_commitments,
        partition="production",
        strata=tuple(track.scoring.stratum_weight_map),
        sources_per_stratum=track.scoring.rollouts_per_stratum,
        batch_seed=generation_seed(track.digest, block_hash),
        used_source_digests=frozenset(),
    )
    plan = build_plan(
        track_spec_digest=track.digest,
        generation_block=200,
        block_hash=block_hash,
        track_start_block=track.track_start_block,
        minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
        generation_interval_blocks=track.cache.generation_interval_blocks,
        exact_challenge_ids=("challenge",),
        confirmation_challenge_ids=(),
        source_proofs=proofs,
        scoring=track.scoring,
        available_routes=track.model.selector.routes,
    )

    def ready_file(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return path, ReadyObject(f"private/{name}", digest, len(content))

    exact_path, exact = ready_file("batch.npz", b"exact-batch")
    index_path, input_index = ready_file("index.json", b'{"samples":[]}')
    text_path, text = ready_file("text.safetensors", b"encoded-inputs")
    sample_cache = ReadySampleCache(input_index, text)
    references = []
    local_files = [
        (f"batches/{exact.sha256}/manifest.json", exact_path, exact),
        (
            f"batch_inputs/{input_index.sha256}/manifest.json",
            index_path,
            input_index,
        ),
        (f"batch_inputs/{text.sha256}/manifest.json", text_path, text),
    ]
    for index, rollout in enumerate(plan.rollouts):
        video_path, video = ready_file(f"video-{index}.mp4", f"video-{index}".encode())
        latent_path, latent = ready_file(
            f"latent-{index}.safetensors", f"latent-{index}".encode()
        )
        references.append(DenseReference(rollout.rollout_id, video, latent))
        local_files.extend(
            (
                (f"batch_assets/{video.sha256}/manifest.json", video_path, video),
                (f"batch_assets/{latent.sha256}/manifest.json", latent_path, latent),
            )
        )
    ready = plan.ready(
        exact=exact,
        dense_references=tuple(references),
        sample_cache=sample_cache,
    )
    public_references = []
    for reference in ready.dense_references:
        public = {"rollout_id": reference.rollout_id}
        for kind, item in (("video", reference.video), ("latent", reference.latent)):
            public[kind] = {
                "object": f"batch_assets/{item.sha256}/manifest.json",
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
        public_references.append(public)
    record = canonical_json(
        {
            "format": "veda-batch-disclosure-v1",
            "plan": ready.document(),
            "public_exact_object": f"batches/{exact.sha256}/manifest.json",
            "public_dense_references": public_references,
            "public_selected_inputs": {
                "condition": None,
                "index": {
                    "object": f"batch_inputs/{input_index.sha256}/manifest.json",
                    "sha256": input_index.sha256,
                    "size_bytes": input_index.size_bytes,
                },
                "text": {
                    "object": f"batch_inputs/{text.sha256}/manifest.json",
                    "sha256": text.sha256,
                    "size_bytes": text.size_bytes,
                },
            },
        }
    )
    batch = {
        "sha256": exact.sha256,
        "generated_block": ready.generation_block,
        "plan_digest": ready.plan_digest,
        "seed_block_hash": ready.block_hash,
        "object": f"batches/{exact.sha256}/manifest.json",
    }
    match = SimpleNamespace(
        challenge_id="challenge",
        document={
            "submission": {"block_height": ready.submission_first_block},
            "evaluation": {
                "stages": [
                    {
                        "stage": "SCORING",
                        "score_block": ready.generation_block,
                        "evaluation_batch": batch,
                    }
                ],
                "evaluation_batch": batch,
            }
        },
    )

    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    for object_name, path, item in local_files:
        publisher._put_file_immutable(PublicFile(object_name, path, item.sha256))
    publisher._put_immutable(f"batch_groups/{ready.plan_digest}.json", record)

    restarted = S3Repository(client, bucket="public-data", prefix="tracks/test")
    assert restarted._load_batch_disclosures(track, (match,)) == frozenset(
        {ready.plan_digest}
    )

    wrong_boundary = SimpleNamespace(
        challenge_id="challenge",
        document={
            **match.document,
            "submission": {"block_height": ready.submission_first_block - 1},
        },
    )
    with pytest.raises(PublicArtifactError, match="wrong generation boundary"):
        S3Repository(client, bucket="public-data", prefix="tracks/test")._load_batch_disclosures(
            track, (wrong_boundary,)
        )

    wrong_assignment = SimpleNamespace(
        challenge_id="challenge",
        document={
            **match.document,
            "evaluation": {
                **match.document["evaluation"],
                "stages": [
                    {
                        **match.document["evaluation"]["stages"][0],
                        "stage": "CONFIRMING",
                    }
                ],
            },
        },
    )
    with pytest.raises(PublicArtifactError, match="challenge assignment"):
        S3Repository(client, bucket="public-data", prefix="tracks/test")._load_batch_disclosures(
            track, (wrong_assignment,)
        )

    confirmation_plan = build_plan(
        track_spec_digest=track.digest,
        generation_block=ready.generation_block,
        block_hash=ready.block_hash,
        track_start_block=track.track_start_block,
        minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
        generation_interval_blocks=track.cache.generation_interval_blocks,
        exact_challenge_ids=(),
        confirmation_challenge_ids=("challenge",),
        source_proofs=proofs,
        scoring=track.scoring,
        available_routes=track.model.selector.routes,
    )
    confirmation_ready = confirmation_plan.ready(
        exact=exact,
        dense_references=tuple(references),
        sample_cache=sample_cache,
    )
    confirmation_record = canonical_json(
        {
            "format": "veda-batch-disclosure-v1",
            "plan": confirmation_ready.document(),
            "public_exact_object": f"batches/{exact.sha256}/manifest.json",
            "public_dense_references": public_references,
            "public_selected_inputs": {
                "condition": None,
                "index": {
                    "object": f"batch_inputs/{input_index.sha256}/manifest.json",
                    "sha256": input_index.sha256,
                    "size_bytes": input_index.size_bytes,
                },
                "text": {
                    "object": f"batch_inputs/{text.sha256}/manifest.json",
                    "sha256": text.sha256,
                    "size_bytes": text.size_bytes,
                },
            },
        }
    )
    publisher._put_immutable(
        f"batch_groups/{confirmation_ready.plan_digest}.json",
        confirmation_record,
    )
    confirmation_batch = {
        **batch,
        "plan_digest": confirmation_ready.plan_digest,
    }
    earlier_batch = {
        **batch,
        "generated_block": ready.generation_block
        - track.cache.generation_interval_blocks,
        "plan_digest": "00" * 32,
    }
    exact_score_block = max(
        earlier_batch["generated_block"],
        confirmation_ready.submission_first_block,
    )
    gap_match = SimpleNamespace(
        challenge_id="challenge",
        document={
            "submission": {"block_height": track.track_start_block},
            "evaluation": {
                "stages": [
                    {
                        "stage": "SCORING",
                        "score_block": exact_score_block,
                        "evaluation_batch": earlier_batch,
                    },
                    {
                        "stage": "CONFIRMING",
                        "score_block": confirmation_ready.generation_block,
                        "evaluation_batch": confirmation_batch,
                    },
                ],
                "evaluation_batch": confirmation_batch,
            }
        },
    )
    assert (
        S3Repository(client, bucket="public-data", prefix="tracks/test")
        ._load_batch_disclosures(track, (gap_match,))
        == frozenset()
    )

    client.objects.pop(f"tracks/test/batches/{exact.sha256}/part-00000")
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        S3Repository(client, bucket="public-data", prefix="tracks/test")._load_batch_disclosures(
            track, (match,)
        )


def test_public_artifacts_bind_summary_matches_and_weights_to_one_snapshot(
    monkeypatch, tmp_path
) -> None:
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"model")
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
        model_path=lambda _challenge_id: str(model_path),
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
    assert public_weights.match_sequence == 1
    assert public_weights.terminal_match_sha256 == hashlib.sha256(
        artifacts.matches[0][1]
    ).hexdigest()
    assert public_weights.plan is not None
    assert public_weights.plan.hotkeys == ("alice",)
    assert public_summary["champion"]["hotkey"] == "alice"
    assert public_summary["match_count"] == 1
    assert public_summary["leaderboard_total"] == 2
    assert public_match["outcome"] == "CHALLENGER_WON"

    missing_champion = WeightChainState(
        121,
        {"genesis-hotkey": 0},
        1,
        1.0,
        10,
        {"genesis-hotkey": 0},
    )
    fallback_store = SimpleNamespace(
        chain_cursor=lambda: (121, "0x" + "34" * 32),
        public_snapshot=store.public_snapshot,
        active_competitors=store.active_competitors,
        model_path=store.model_path,
    )
    fallback = build_public_artifacts(fallback_store, track, missing_champion, version_key=9)
    assert fallback is not None
    fallback_weights = decode_weight_artifact(fallback.weights)
    assert fallback_weights.plan == WeightPlan(
        121,
        ("genesis-hotkey",),
        (0,),
        (1.0,),
    )
    assert json.loads(fallback.summary)["champion"]["hotkey"] == "alice"

    def disclosure(plan_digest, challenge_ids):
        batch = next(
            stage["evaluation_batch"]
            for stage in snapshot["challenges"][0]["stages"]
            if stage["evaluation_batch"]["plan_digest"] == plan_digest
        )
        return BatchDisclosure(
            challenge_ids,
            f"batch_groups/{plan_digest}.json",
            b"{}",
            (
                DisclosureFile(
                    f"batches/{batch['sha256']}/manifest.json",
                    tmp_path / "batch.npz",
                    batch["sha256"],
                ),
                DisclosureFile(
                    f"batch_assets/{plan_digest}/manifest.json",
                    tmp_path / "asset",
                    plan_digest,
                ),
            ),
        )

    withheld = build_public_artifacts(
        store,
        track,
        chain,
        version_key=9,
        batch_disclosure_resolver=lambda digest: disclosure(digest, ("challenge", "pending")),
    )
    assert withheld is not None
    assert not any(item.object_name.startswith("batches/") for item in withheld.files)
    disclosed = build_public_artifacts(
        store,
        track,
        chain,
        version_key=9,
        batch_disclosure_resolver=lambda digest: disclosure(digest, ("challenge",)),
    )
    assert disclosed is not None
    assert sum(item.object_name.startswith("batches/") for item in disclosed.files) == 2
    assert sum(item.object_name.startswith("batch_assets/") for item in disclosed.files) == 2
    assert len(disclosed.batch_groups) == 2

    resolved_digests = frozenset(
        name.removeprefix("batch_groups/").removesuffix(".json")
        for name, _ in disclosed.batch_groups
    )

    def unexpected_disclosure(_digest):
        raise AssertionError("an already public batch was restored from private storage")

    repeated = build_public_artifacts(
        store,
        track,
        chain,
        version_key=9,
        batch_disclosure_resolver=unexpected_disclosure,
        disclosed_plan_digests=resolved_digests,
    )
    assert repeated is not None
    assert not any(
        item.object_name.startswith(("batches/", "batch_assets/"))
        for item in repeated.files
    )
    assert repeated.batch_groups == ()

    store.chain_cursor = lambda: (119, "0x" + "11" * 32)
    assert build_public_artifacts(store, track, chain, version_key=9) is None

    store.chain_cursor = lambda: (120, "0x" + "12" * 32)
    snapshot["challenges"][0]["score_block"] = 121
    with pytest.raises(PublicArtifactError, match="newer"):
        build_public_artifacts(store, track, chain, version_key=9)
