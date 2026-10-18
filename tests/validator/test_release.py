import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator import release
from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
from validator.publication import PublicArtifactError, S3Repository, build_genesis_artifacts
from validator.release import ReleaseApproval
from validator.weights import WeightChainState

from .test_publication import MemoryS3


def test_research_inventory_does_not_gate_startup_or_claim_acceptance():
    directory = Path(__file__).resolve().parents[2] / "evidence/system-panel/acceptance"
    assert sorted(path.name for path in directory.glob("inventory*.json")) == ["inventory.json"]
    inventory = json.loads((directory / "inventory.json").read_bytes())
    checks = inventory["experiments"]
    assert len({check["id"] for check in checks}) == len(checks)
    assert not any(check["release_blocker"] for check in checks)
    assert inventory["release_accepted"] is False
    assert inventory["scope"] == "requirements_and_gap_inventory_not_acceptance"


def approved_candidate():
    document = valid_spec()
    document["track_id"] = "veda-sn120-v2"
    manifest = ArtifactManifest((
        Artifact("runtime/source.zip", "ab" * 32, 100, "artifacts/source"),
    ))
    document["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(document)
    return track, manifest.encode(), ReleaseApproval(track.digest)


def chain(track):
    return WeightChainState(99, {track.initial_champion.hotkey: 0}, 1, 1.0, 0, {})


def test_uncoordinated_candidate_cannot_build_genesis():
    document = valid_spec()
    document["track_id"] = "veda-sn120-v2"
    track = parse_track_spec(document)
    with pytest.raises(PublicArtifactError, match="coordinated track digest"):
        build_genesis_artifacts(track, chain(track), version_key=0)


@pytest.mark.parametrize(
    "failure", ["absent", "uncoordinated", "runtime_absent", "runtime_mismatch"]
)
def test_formal_release_rejection_performs_zero_writes(failure):
    track, manifest, approval = approved_candidate()
    artifacts = build_genesis_artifacts(
        track, chain(track), version_key=0, runtime_manifest=manifest, release_approval=approval
    )
    if failure == "absent":
        approval = None
    elif failure == "uncoordinated":
        approval = replace(approval, coordinated_track_digest="aa" * 32)
    elif failure == "runtime_absent":
        artifacts = replace(artifacts, runtime_manifest=None)
    else:
        artifacts = replace(artifacts, runtime_manifest=canonical_json({"invalid": True}))
    client = MemoryS3()
    client.objects = {
        f"tracks/main/{name}.json": b"existing live v1"
        for name in ("track", "weights", "summary", "status")
    }
    original = dict(client.objects)
    publisher = S3Repository(
        client, bucket="public-data", prefix="tracks/main", release_approval=approval
    )
    with pytest.raises(PublicArtifactError, match="formal release rejected"):
        asyncio.run(publisher.publish(artifacts))
    assert client.calls == []
    assert client.objects == original


def test_coordinated_release_can_publish_without_a_research_acceptance_file():
    track, manifest, approval = approved_candidate()
    artifacts = build_genesis_artifacts(
        track, chain(track), version_key=0, runtime_manifest=manifest, release_approval=approval
    )
    client = MemoryS3()
    publisher = S3Repository(
        client, bucket="public-data", prefix="tracks/main", release_approval=approval
    )
    asyncio.run(publisher.publish(artifacts))
    assert client.objects["tracks/main/track.json"] == artifacts.track


@pytest.mark.parametrize("digest", ["", "ab" * 20, "AB" * 32, "g" * 64])
def test_coordinated_release_environment_rejects_invalid_digest(monkeypatch, digest):
    monkeypatch.setenv("VEDA_COORDINATED_RELEASE_DIGEST", digest)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ReleaseApproval.from_environment()


def test_coordinated_release_environment_requires_only_the_existing_digest_setting(monkeypatch):
    monkeypatch.delenv("VEDA_COORDINATED_RELEASE_DIGEST", raising=False)
    assert ReleaseApproval.from_environment() is None
    monkeypatch.setenv("VEDA_COORDINATED_RELEASE_DIGEST", "ab" * 32)
    assert ReleaseApproval.from_environment() == ReleaseApproval("ab" * 32)


def test_retired_public_identity_cannot_bypass_release_approval():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="coordinated track digest"):
        release.require_release_approval(
            SimpleNamespace(
                track_id="veda-sn120-v1",
                digest="0fcae991aef29f87b97fbb07cf79c49e5eb9cc2a5d9aecdd2a566443e2b81dd5",
            ),
            None,
            None,
        )
