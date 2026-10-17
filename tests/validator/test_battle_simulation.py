import asyncio
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

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs) -> None:
        self.objects[kwargs["Key"]] = kwargs["Body"]


def challenge(
    sequence: int,
    hotkey: str,
    *,
    state: str,
    score: float | None = None,
    incumbent_score: float | None = None,
    error_code: str | None = None,
) -> dict:
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
        "evaluation_batch": (
            {
                "sha256": f"{sequence + 100:064x}",
                "generated_block": 200 + sequence,
                "source_digest": f"{sequence + 200:064x}",
            }
            if score is not None
            else None
        ),
        "score_block": 200 + sequence if score is not None else 200 + sequence,
        "created_at_ms": sequence,
        "updated_at_ms": sequence,
        "stages": [],
        "events": [],
    }


def test_complete_fifo_battle_publication_recovery_and_weight_switch() -> None:
    raw = valid_spec()
    raw["track_id"] = "veda-sn120"
    raw["netuid"] = 120
    raw["initial_champion"]["hotkey"] = "uid0"
    raw["initial_champion"]["score"] = 0.5
    raw["scoring"]["minimum_frontier_gain"] = 0.01
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
    )
    chain = WeightChainState(
        300,
        {"uid0": 0, "alice": 11, "bob": 12, "carol": 13, "dave": 14},
        1,
        1.0,
        100,
        {},
    )

    artifacts = build_public_artifacts(store, track, chain, mechid=0, version_key=1)

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
    assert validate_weight_artifact(
        weights,
        chain,
        expected_netuid=120,
        expected_track_digest=track.digest,
        max_staleness_blocks=1,
    ) == weights.plan
    rows = {row["hotkey"]: row for row in status["participants"]}
    assert rows["uid0"]["state"] == "DISPLACED"
    assert rows["dave"]["champion"] is True
    assert rows["bob"]["outcome"] == "CHAMPION_RETAINED"
    assert rows["carol"]["outcome"] == "CHALLENGE_FAILED"

    remote = MemoryS3()
    repository = S3Repository(remote, bucket="public-data", prefix="tracks/main")
    asyncio.run(repository.publish(artifacts))
    recovered = asyncio.run(repository.load(track))
    assert len(recovered.matches) == 5
    assert recovered.matches[-1].champion_after.hotkey == "dave"

    tampered = bytearray(remote.objects["tracks/main/matches/000000000004.json"])
    tampered[-2] ^= 1
    remote.objects["tracks/main/matches/000000000004.json"] = bytes(tampered)
    with pytest.raises(PublicArtifactError):
        asyncio.run(S3Repository(remote, bucket="public-data", prefix="tracks/main").load(track))
