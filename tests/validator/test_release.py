import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from scoring.spec import parse_track_spec, track_spec_dict
from tests.scoring.test_spec import valid_spec
from validator import release
from validator import system_matches as matches
from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
from validator.publication import PublicArtifactError, S3Repository, build_genesis_artifacts
from validator.release import ReleaseApproval
from validator.weights import WeightChainState

from .test_publication import MemoryS3


def test_inventory_records_launch_checks_without_claiming_runtime_authorization():
    directory = Path(__file__).resolve().parents[2] / "evidence/system-panel/acceptance"
    assert sorted(path.name for path in directory.glob("inventory*.json")) == ["inventory.json"]
    inventory = json.loads((directory / "inventory.json").read_bytes())
    checks = inventory["experiments"]
    assert len({check["id"] for check in checks}) == len(checks)
    assert {check["id"] for check in checks if check["release_blocker"]} == (
        release.REQUIRED_EXPERIMENTS
    )
    assert "release_accepted" not in inventory
    assert inventory["scope"] == "release_requirements_and_evidence_not_authorization"
    assert {check["id"] for check in checks} == release.REQUIRED_EXPERIMENTS
    assert set(inventory["known_limitations"]) == {"cost_prediction", "selector_overhead"}
    assert inventory["development_history"]["scope"].endswith(
        "not additional production release requirements."
    )


@pytest.fixture
def approved_candidate(monkeypatch):
    """Synthetic bound reports exercise authorization, not real release acceptance."""

    monkeypatch.setattr(matches, "evaluation_source_digest", lambda: "31" * 32)
    monkeypatch.setattr(matches, "evaluation_environment_digest", lambda: "32" * 32)
    matches._evidence_runtime_digest.cache_clear()
    document = valid_spec()
    document["track_id"] = "veda-sn120-v2"
    raw_name = "evidence/system-panel/acceptance/unit-test-raw.json"
    raw = b'{"synthetic":true}'
    manifest = ArtifactManifest((
        Artifact(raw_name, hashlib.sha256(raw).hexdigest(), len(raw), "artifacts/raw"),
        Artifact("runtime/source.zip", "ab" * 32, 100, "artifacts/source"),
        Artifact("weights/reference", "cd" * 32, 100, "artifacts/reference"),
    ))
    document["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(document)
    reference = {"logical_name": raw_name, "sha256": hashlib.sha256(raw).hexdigest()}
    acceptance = canonical_json({
        "format": "veda-release-acceptance-v1",
        "release_accepted": True,
        "evidence_contract_sha256": matches._evidence_contract_digest(track),
        "runtime_contract_sha256": matches._evidence_runtime_digest(manifest.encode()),
        "experiments": {
            identifier: {
                "pass": True,
                "method": "Synthetic unit-test fixture only",
                "actual_values": {"synthetic": True},
                "acceptance_manifest": reference,
                "raw_evidence": [reference],
            }
            for identifier in sorted(release.REQUIRED_EXPERIMENTS)
        },
    })
    manifest = ArtifactManifest(tuple(sorted((
        *manifest.artifacts,
        Artifact(
            release.ACCEPTANCE_LOGICAL_NAME, hashlib.sha256(acceptance).hexdigest(),
            len(acceptance), "artifacts/acceptance",
        ),
    ), key=lambda artifact: artifact.logical_name)))
    document["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(document)
    yield track, manifest.encode(), ReleaseApproval(track.digest, acceptance)
    matches._evidence_runtime_digest.cache_clear()


def rebind_acceptance(candidate, result):
    """Bind changed test evidence so semantic rejection cannot rely on a stale hash."""

    track, payload, _ = candidate
    acceptance = canonical_json(result)
    manifest = ArtifactManifest(tuple(
        replace(artifact, sha256=hashlib.sha256(acceptance).hexdigest(), size_bytes=len(acceptance))
        if artifact.logical_name == release.ACCEPTANCE_LOGICAL_NAME else artifact
        for artifact in ArtifactManifest.decode(payload).artifacts
    ))
    document = track_spec_dict(track)
    document["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(document)
    return track, manifest.encode(), ReleaseApproval(track.digest, acceptance)


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
def test_formal_release_rejection_performs_zero_writes(failure, approved_candidate):
    track, manifest, approval = approved_candidate
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


def test_formal_release_requires_all_bound_experiments_to_publish(approved_candidate):
    track, manifest, approval = approved_candidate
    artifacts = build_genesis_artifacts(
        track, chain(track), version_key=0, runtime_manifest=manifest, release_approval=approval
    )
    client = MemoryS3()
    publisher = S3Repository(
        client, bucket="public-data", prefix="tracks/main", release_approval=approval
    )
    asyncio.run(publisher.publish(artifacts))
    assert client.objects["tracks/main/track.json"] == artifacts.track


@pytest.mark.parametrize(("failure", "reason"), [
    ("not_accepted", "differs from active scoring or runtime contracts"),
    ("missing_experiment", "missing mandatory experiments"),
    ("failed_E2", "incomplete or failed: E2"),
    ("failed_E16", "incomplete or failed: E16"),
    ("non_boolean_pass", "incomplete or failed: E2"),
    ("no_method", "incomplete or failed: E2"),
    ("no_values", "incomplete or failed: E2"),
    ("no_raw", "incomplete or failed: E2"),
    ("no_report", "has no bound evidence: E2"),
    ("unbound_raw", "not bound by the manifest: E2"),
    ("self_reference", "not bound by the manifest: E2"),
    ("foreign_scoring", "differs from active scoring or runtime contracts"),
    ("foreign_runtime", "differs from active scoring or runtime contracts"),
])
def test_bound_but_incomplete_acceptance_cannot_write(failure, reason, approved_candidate):
    track, manifest, approval = approved_candidate
    artifacts = build_genesis_artifacts(
        track, chain(track), version_key=0, runtime_manifest=manifest, release_approval=approval
    )
    result = deepcopy(json.loads(approval.acceptance))
    experiment = result["experiments"]["E2"]
    if failure == "not_accepted":
        result["release_accepted"] = False
    elif failure == "missing_experiment":
        del result["experiments"]["E3"]
    elif failure.startswith("failed_"):
        result["experiments"][failure.removeprefix("failed_")]["pass"] = False
    elif failure == "non_boolean_pass":
        experiment["pass"] = 1
    elif failure == "no_method":
        experiment["method"] = " "
    elif failure == "no_values":
        experiment["actual_values"] = {}
    elif failure == "no_raw":
        experiment["raw_evidence"] = []
    elif failure == "no_report":
        del experiment["acceptance_manifest"]
    elif failure == "unbound_raw":
        experiment["raw_evidence"][0]["sha256"] = "ff" * 32
    elif failure == "self_reference":
        experiment["raw_evidence"][0]["logical_name"] = release.ACCEPTANCE_LOGICAL_NAME
    elif failure == "foreign_scoring":
        result["evidence_contract_sha256"] = "ff" * 32
    else:
        result["runtime_contract_sha256"] = "ff" * 32
    track, manifest, approval = rebind_acceptance(approved_candidate, result)
    document = json.loads(artifacts.track)
    document["spec"] = track_spec_dict(track)
    document["track_spec_digest"] = track.digest
    artifacts = replace(artifacts, track=canonical_json(document), runtime_manifest=(
        f"runtime/{track.artifact_manifest_sha256}.json", manifest,
    ))
    client = MemoryS3()
    publisher = S3Repository(
        client, bucket="public-data", prefix="tracks/main", release_approval=approval
    )
    with pytest.raises(PublicArtifactError, match=reason):
        asyncio.run(publisher.publish(artifacts))
    assert client.calls == []
    assert client.objects == {}


@pytest.mark.parametrize("acceptance", [b"", b"{}", b"x" * (2 * 1024 * 1024 + 1)])
def test_missing_or_unbound_acceptance_is_rejected(acceptance, approved_candidate):
    track, manifest, approval = approved_candidate
    with pytest.raises(ValueError, match="acceptance"):
        replace(approval, acceptance=acceptance).verify(track, manifest)


@pytest.mark.parametrize("digest", ["", "ab" * 20, "AB" * 32, "g" * 64])
def test_coordinated_release_environment_rejects_invalid_digest(monkeypatch, digest):
    monkeypatch.setenv("VEDA_COORDINATED_RELEASE_DIGEST", digest)
    monkeypatch.setenv("VEDA_RELEASE_ACCEPTANCE", "unused.json")
    with pytest.raises(ValueError, match="lowercase SHA-256|both required"):
        ReleaseApproval.from_environment()


def test_coordinated_release_environment_requires_digest_and_acceptance(
    monkeypatch, tmp_path, approved_candidate,
):
    monkeypatch.delenv("VEDA_COORDINATED_RELEASE_DIGEST", raising=False)
    monkeypatch.delenv("VEDA_RELEASE_ACCEPTANCE", raising=False)
    assert ReleaseApproval.from_environment() is None
    monkeypatch.setenv("VEDA_COORDINATED_RELEASE_DIGEST", "ab" * 32)
    with pytest.raises(ValueError, match="both required"):
        ReleaseApproval.from_environment()
    path = tmp_path / "acceptance.json"
    path.write_bytes(approved_candidate[2].acceptance)
    monkeypatch.setenv("VEDA_RELEASE_ACCEPTANCE", str(path))
    assert ReleaseApproval.from_environment() == ReleaseApproval("ab" * 32, path.read_bytes())
    monkeypatch.delenv("VEDA_COORDINATED_RELEASE_DIGEST")
    with pytest.raises(ValueError, match="both required"):
        ReleaseApproval.from_environment()


def test_environment_rejects_oversized_acceptance_before_decoding(monkeypatch, tmp_path):
    path = tmp_path / "acceptance.json"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    monkeypatch.setenv("VEDA_COORDINATED_RELEASE_DIGEST", "ab" * 32)
    monkeypatch.setenv("VEDA_RELEASE_ACCEPTANCE", str(path))
    with pytest.raises(ValueError, match="invalid size"):
        ReleaseApproval.from_environment()


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
