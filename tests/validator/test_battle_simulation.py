import asyncio
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator.publication import (
    PublicArtifactError,
    S3Repository,
    build_public_artifacts,
    decode_weight_artifact,
)
from validator.weights import WeightChainState, validate_weight_artifact


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class MemoryS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict] = {}

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {
            "Body": io.BytesIO(self.objects[Key]),
            "ETag": f'"{hashlib.md5(self.objects[Key], usedforsecurity=False).hexdigest()}"',
        }

    def put_object(self, **kwargs) -> None:
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


def precondition_failed():
    error = RuntimeError("precondition failed")
    error.response = {
        "Error": {"Code": "PreconditionFailed"},
        "ResponseMetadata": {"HTTPStatusCode": 412},
    }
    return error


def challenge(
    sequence: int,
    hotkey: str,
    *,
    state: str,
    score: float | None = None,
    incumbent_score: float | None = None,
    paired_gain: float | None = None,
    error_code: str | None = None,
) -> dict:
    gain = (
        paired_gain
        if paired_gain is not None
        else score - incumbent_score
        if score is not None and incumbent_score is not None
        else None
    )
    exact_batch = {
        "sha256": f"{sequence + 100:064x}",
        "generated_block": 200 + sequence,
        "plan_digest": f"{sequence + 200:064x}",
        "seed_block_hash": "0x" + "ab" * 32,
    }
    confirmation_batch = {
        "sha256": f"{sequence + 300:064x}",
        "generated_block": 230 + sequence,
        "plan_digest": f"{sequence + 400:064x}",
        "seed_block_hash": "0x" + "cd" * 32,
    }
    stages = []
    final_batch = None
    final_block = 200 + sequence
    if score is not None:
        stages = [
            {
                "stage": "SCORING",
                "score": score,
                "score_block": 220 + sequence,
                "incumbent_score": incumbent_score,
                "paired_gain_lower_bound": gain,
                "evaluation_batch": exact_batch,
            }
        ]
        final_batch = exact_batch
        final_block = 220 + sequence
        if gain is not None and gain > 0.01:
            stages.append(
                {
                    "stage": "CONFIRMING",
                    "score": score,
                    "score_block": 240 + sequence,
                    "incumbent_score": incumbent_score,
                    "paired_gain_lower_bound": gain,
                    "evaluation_batch": confirmation_batch,
                }
            )
            final_batch = confirmation_batch
            final_block = 240 + sequence
    return {
        "challenge_id": f"challenge-{sequence}",
        "submission_id": f"submission-{sequence}",
        "hotkey": hotkey,
        "repo_id": f"owner/model-{sequence}",
        "revision": f"{sequence:040x}",
        "block_height": 100 + sequence,
        "extrinsic_index": 0,
        "event_index": 0,
        "state": state,
        "attempts": 1,
        "error_code": error_code,
        "error_summary": None,
        "file_sha256": "ab" * 32 if score is not None else None,
        "model_digest": f"{sequence:064x}" if score is not None else None,
        "model_size_bytes": 550_000_000 if score is not None else None,
        "track_spec_digest": None,
        "final_score": score,
        "incumbent_score": incumbent_score,
        "paired_gain_lower_bound": gain,
        "evaluation_batch": final_batch,
        "score_block": final_block,
        "created_at_ms": sequence,
        "updated_at_ms": sequence,
        "stages": stages,
        "events": [],
    }


def test_complete_fifo_battle_publication_recovery_and_weight_switch(tmp_path) -> None:
    raw = valid_spec()
    raw["track_id"] = "veda-sn120"
    raw["netuid"] = 120
    raw["initial_champion"]["hotkey"] = "uid0"
    raw["initial_champion"]["score"] = 0.5
    raw["scoring"]["minimum_frontier_gain"] = 0.01
    raw["full_video"]["minimum_gain"] = 0.01
    track = parse_track_spec(raw)
    challenges = [
        challenge(1, "alice", state="SUCCEEDED", score=0.6, incumbent_score=0.5),
        challenge(2, "bob", state="SUCCEEDED", score=0.605, incumbent_score=0.6),
        challenge(
            3,
            "carol",
            state="PARTICIPANT_FAILED",
            error_code="REPOSITORY_UNAVAILABLE",
        ),
        challenge(4, "dave", state="SUCCEEDED", score=0.8, incumbent_score=0.6),
        challenge(
            5,
            "alice",
            state="REJECTED_DUPLICATE_HOTKEY",
            error_code="REJECTED_DUPLICATE_HOTKEY",
        ),
    ]
    winning_model = tmp_path / "winning.safetensors"
    winning_model.write_bytes(b"winning selector")
    winning_digest = hashlib.sha256(winning_model.read_bytes()).hexdigest()
    for item in challenges:
        if item["state"] == "SUCCEEDED":
            item["file_sha256"] = winning_digest
            item["model_size_bytes"] = winning_model.stat().st_size
    competitors = [
        {
            "uid": uid,
            "hotkey": item["hotkey"],
            "submission_id": item["submission_id"],
            "repo_id": item["repo_id"],
            "revision": item["revision"],
            "block_height": item["block_height"],
            "extrinsic_index": 0,
            "event_index": 0,
        }
        for uid, item in zip((11, 12, 13, 14), challenges[:4], strict=True)
    ]
    store = SimpleNamespace(
        chain_cursor=lambda: (300, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": challenges},
        active_competitors=lambda _mapping: competitors,
        model_path=lambda _challenge_id: str(winning_model),
    )
    chain = WeightChainState(
        300,
        {"uid0": 0, "alice": 11, "bob": 12, "carol": 13, "dave": 14},
        1,
        1.0,
        100,
        {},
    )

    artifacts = build_public_artifacts(store, track, chain, version_key=1)

    assert artifacts is not None
    summary = json.loads(artifacts.summary)
    status = json.loads(artifacts.status)
    weights = decode_weight_artifact(artifacts.weights)
    matches = [json.loads(payload) for _, payload in artifacts.matches]
    assert [item["outcome"] for item in matches] == [
        "CHALLENGER_WON",
        "CHAMPION_RETAINED",
        "CHALLENGE_FAILED",
        "CHALLENGER_WON",
        "CHALLENGE_FAILED",
    ]
    assert [item["champion_after"]["hotkey"] for item in matches] == [
        "alice",
        "alice",
        "alice",
        "dave",
        "dave",
    ]
    assert summary["champion"]["hotkey"] == "dave"
    assert weights.plan is not None and weights.plan.uids == (14,)
    assert (
        validate_weight_artifact(
            weights,
            chain,
            expected_netuid=120,
            expected_track_digest=track.digest,
            expected_mechid=0,
            max_staleness_blocks=1,
        )
        == weights.plan
    )
    rows = {row["hotkey"]: row for row in status["participants"]}
    assert rows["uid0"]["state"] == "DISPLACED"
    assert rows["dave"]["champion"] is True
    assert rows["bob"]["outcome"] == "CHAMPION_RETAINED"
    assert rows["carol"]["outcome"] == "CHALLENGE_FAILED"

    restored = tmp_path / "restored.safetensors"
    resolved = []

    def restore_model(sha256, size_bytes):
        resolved.append((sha256, size_bytes))
        return restored

    store.model_path = lambda _challenge_id: None
    with_files = build_public_artifacts(
        store,
        track,
        chain,
        version_key=1,
        model_resolver=restore_model,
    )
    assert with_files is not None
    assert resolved == [(winning_digest, winning_model.stat().st_size)] * 3
    assert sum(item.object_name.startswith("models/") for item in with_files.files) == 1

    remote = MemoryS3()
    repository = S3Repository(remote, bucket="public-data", prefix="tracks/main")
    asyncio.run(repository.publish(artifacts))
    recovered = asyncio.run(repository.load(track))
    assert len(recovered.matches) == 5
    assert recovered.matches[-1].champion_after.hotkey == "dave"

    name = "tracks/main/matches/veda-sn120/000000000004.json"
    tampered = bytearray(remote.objects[name])
    tampered[-2] ^= 1
    remote.objects[name] = bytes(tampered)
    with pytest.raises(PublicArtifactError):
        asyncio.run(S3Repository(remote, bucket="public-data", prefix="tracks/main").load(track))
