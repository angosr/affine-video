import asyncio
import hashlib
from dataclasses import replace

import pytest

from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator import release
from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
from validator.publication import PublicArtifactError, S3Repository, build_genesis_artifacts
from validator.release import ReleaseApproval
from validator.weights import WeightChainState

from .test_publication import MemoryS3


def approved_candidate(monkeypatch, *, mutation=None):
    document = valid_spec()
    document["track_id"] = "veda-sn120-v2"
    provisional = parse_track_spec(document)
    runtime_digest = "ef" * 32
    monkeypatch.setattr(release, "_evidence_runtime_digest", lambda _manifest: runtime_digest)
    acceptance = {
        "format": "veda-release-acceptance-v1",
        "release_accepted": True,
        "evidence_contract_sha256": release._evidence_contract_digest(provisional),
        "runtime_contract_sha256": runtime_digest,
        "experiments": {},
    }
    artifacts = []
    for identifier in sorted(release._REQUIRED_EXPERIMENTS):
        references = []
        for kind in ("protocol", "raw"):
            name = f"evidence/system-panel/acceptance/{identifier}-{kind}.json"
            payload = canonical_json({"unit_test_fixture": identifier, "kind": kind})
            digest = hashlib.sha256(payload).hexdigest()
            artifacts.append(Artifact(name, digest, len(payload), f"artifacts/{digest}"))
            references.append({"logical_name": name, "sha256": digest})
        acceptance["experiments"][identifier] = {
            "pass": True,
            "method": "unit-test fixture, not GPU acceptance",
            "actual_values": {"fixture": 1},
            "acceptance_manifest": references[0],
            "raw_evidence": [references[1]],
        }
    if mutation is not None:
        mutation(acceptance)
    payload = canonical_json(acceptance)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(release, "PRODUCTION_ACCEPTANCE_SHA256", digest)
    artifacts.append(
        Artifact(release.ACCEPTANCE_LOGICAL_NAME, digest, len(payload), f"artifacts/{digest}")
    )
    manifest = ArtifactManifest(
        tuple(sorted(artifacts, key=lambda artifact: artifact.logical_name))
    )
    document["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(document)
    return track, manifest.encode(), ReleaseApproval(track.digest, payload)


def chain(track):
    return WeightChainState(99, {track.initial_champion.hotkey: 0}, 1, 1.0, 0, {})


def test_unaccepted_candidate_cannot_build_genesis():
    document = valid_spec()
    document["track_id"] = "veda-sn120-v2"
    track = parse_track_spec(document)
    with pytest.raises(PublicArtifactError, match="verified acceptance"):
        build_genesis_artifacts(track, chain(track), version_key=0)


@pytest.mark.parametrize(
    "failure", ["absent", "uncoordinated", "runtime_absent", "unverified_hash"]
)
def test_formal_release_rejection_performs_zero_writes(monkeypatch, failure):
    track, manifest, approval = approved_candidate(monkeypatch)
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
        monkeypatch.setattr(release, "PRODUCTION_ACCEPTANCE_SHA256", "0" * 64)
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


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update(release_accepted=False),
        lambda result: result.update(runtime_contract_sha256="00" * 32),
        lambda result: result.update(evidence_contract_sha256="00" * 32),
        lambda result: result["experiments"].pop("E17"),
        lambda result: result["experiments"]["E10"].update(**{"pass": False}),
        lambda result: result["experiments"]["E11"].update(raw_evidence=[]),
        lambda result: result["experiments"]["E12"]["raw_evidence"][0].update(sha256="00" * 32),
    ],
)
def test_acceptance_cannot_omit_failed_unbound_or_mismatched_evidence(monkeypatch, mutation):
    track, manifest, approval = approved_candidate(monkeypatch, mutation=mutation)
    with pytest.raises(ValueError):
        approval.verify(track, manifest)


def test_accepted_coordinated_release_can_publish(monkeypatch):
    track, manifest, approval = approved_candidate(monkeypatch)
    artifacts = build_genesis_artifacts(
        track, chain(track), version_key=0, runtime_manifest=manifest, release_approval=approval
    )
    client = MemoryS3()
    publisher = S3Repository(
        client, bucket="public-data", prefix="tracks/main", release_approval=approval
    )
    asyncio.run(publisher.publish(artifacts))
    assert client.objects["tracks/main/track.json"] == artifacts.track


def test_legacy_exception_is_only_the_pinned_public_identity():
    from types import SimpleNamespace

    release.require_release_approval(
        SimpleNamespace(
            track_id=release.LEGACY_PUBLIC_IDENTITY[0], digest=release.LEGACY_PUBLIC_IDENTITY[1]
        ),
        None,
        None,
    )
    with pytest.raises(ValueError):
        release.require_release_approval(
            SimpleNamespace(track_id=release.LEGACY_PUBLIC_IDENTITY[0], digest="ab" * 32),
            None,
            None,
        )
