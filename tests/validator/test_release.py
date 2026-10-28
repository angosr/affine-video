import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scoring.spec import parse_track_spec, track_spec_dict
from tests.scoring.test_spec import valid_spec
from validator import release
from validator import system_matches as matches
from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
from validator.publication import (
    PublicArtifactError,
    S3Repository,
    build_genesis_artifacts,
    build_public_artifacts,
    decode_weight_artifact,
)
from validator.release import ReleaseApproval
from validator.store import ChallengeStore
from validator.weights import WeightChainState

from .test_publication import MemoryS3
from .test_store import submission_at


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


@pytest.mark.parametrize("damage", [None, "data_spec_sha256", "data_contract_sha256"])
def test_new_rules_approval_binds_original_data_identity(approved_candidate, damage):
    from scoring.spec import data_contract_digest, data_spec_logical_name
    from tests.scoring.test_spec import rules_for_data

    data, original, old_approval = approved_candidate
    rules = parse_track_spec(rules_for_data(data))
    origin = canonical_json(track_spec_dict(data))
    manifest = ArtifactManifest(tuple(sorted((*ArtifactManifest.decode(original).artifacts,
        Artifact(data_spec_logical_name(rules), data.digest, len(origin),
                 "artifacts/" + data.digest),
    ), key=lambda a: a.logical_name)))
    result = json.loads(old_approval.acceptance)
    result.update(evidence_contract_sha256=matches._evidence_contract_digest(rules),
                  runtime_contract_sha256=matches._evidence_runtime_digest(manifest.encode()),
                  data_spec_sha256=data.digest, data_contract_sha256=data_contract_digest(data))
    if damage is not None:
        result.pop(damage)
    rules, runtime, approval = rebind_acceptance((rules, manifest.encode(), old_approval), result)
    if damage is None:
        approval.verify(rules, runtime)
        assert approval.public_identity(rules, runtime)["data_spec_sha256"] == data.digest
        assert release.verify_data_binding(rules, runtime, origin, original) == data
        client = MemoryS3()
        client.objects["tracks/main/track.json"] = canonical_json({
            "track_spec_digest": data.digest, "spec": track_spec_dict(data),
        })
        client.objects[f"tracks/main/runtime/{data.artifact_manifest_sha256}/manifest.json"] = (
            original
        )
        publisher = S3Repository(client, bucket="test", prefix="tracks/main",
                                 release_approval=approval)
        artifacts = build_genesis_artifacts(rules, chain(rules), version_key=0,
                                            runtime_manifest=runtime, release_approval=approval)
        asyncio.run(publisher.publish(artifacts))
        assert client.objects["tracks/main/artifacts/" + data.digest] == origin
        # Once the old mutable pointer is replaced, cold recovery still resolves
        # exactly the same data origin through its immutable artifact.
        cold = S3Repository(client, bucket="test", prefix="tracks/main")
        assert cold._load_data_track(rules) == data
    else:
        with pytest.raises(ValueError, match="original data contract"):
            approval.verify(rules, runtime)


def chain(track):
    return WeightChainState(99, {track.initial_champion.hotkey: 0}, 1, 1.0, 0, {})


@pytest.fixture
def compatible_runtime(approved_candidate):
    track, original, approval = approved_candidate
    original_manifest = ArtifactManifest.decode(original)
    artifacts = tuple(
        replace(artifact, sha256="ef" * 32, size_bytes=101, object_key="artifacts/new-source")
        if artifact.logical_name == "runtime/source.zip" else artifact
        for artifact in original_manifest.artifacts
        if artifact.logical_name != release.ACCEPTANCE_LOGICAL_NAME
    ) + (Artifact("runtime/image.json", "aa" * 32, 100, "artifacts/image"),)
    manifest = ArtifactManifest(tuple(sorted(artifacts, key=lambda item: item.logical_name)))
    result = deepcopy(json.loads(approval.acceptance))
    result.update(
        track_spec_digest=track.digest, competition_manifest_sha256=original_manifest.digest,
        image_id="sha256:" + "ab" * 32, source_sha256="31" * 32,
        environment_sha256="32" * 32,
        runtime_contract_sha256=matches._evidence_runtime_digest(manifest.encode()),
    )
    acceptance = canonical_json(result)
    manifest = ArtifactManifest(tuple(sorted((*manifest.artifacts, Artifact(
        release.ACCEPTANCE_LOGICAL_NAME, hashlib.sha256(acceptance).hexdigest(),
        len(acceptance), "artifacts/new-acceptance",
    )), key=lambda item: item.logical_name)))
    return track, manifest.encode(), ReleaseApproval(track.digest, acceptance, original)


def test_compatible_release_binds_new_source_without_changing_the_competition(compatible_runtime):
    track, payload, approval = compatible_runtime
    original = canonical_json(track_spec_dict(track))
    approval.verify(track, payload)
    identity = approval.public_identity(track, payload)
    assert identity["track_spec_digest"] == track.digest
    assert identity["artifact_manifest_sha256"] == hashlib.sha256(payload).hexdigest()
    assert identity["competition_manifest_sha256"] == track.artifact_manifest_sha256
    assert identity["image_id"] == "sha256:" + "ab" * 32
    assert canonical_json(track_spec_dict(track)) == original
    with pytest.raises(ValueError, match="differs from candidate"):
        replace(approval, competition_manifest=None).verify(track, payload)
    with pytest.raises(ValueError, match="OCI descriptor"):
        approval.verify(track, payload, image_id="sha256:" + "ff" * 32)


def test_runtime_release_keeps_live_track_cursor_queue_and_original_history(
    compatible_runtime, approved_candidate, tmp_path,
):
    """Synthetic CPU publication, no chain/cloud/OCI execution or real miner."""
    track, payload, approval = compatible_runtime
    client = MemoryS3()
    publisher = S3Repository(client, bucket="test", prefix="tracks/main",
                             release_approval=approved_candidate[2])
    with ChallengeStore(tmp_path / "unchanged.db", track_spec_digest=track.digest) as store:
        store.add_submission(submission_at(110, hotkey="queued-miner"))
        store.advance_chain_cursor(150, "0x" + "12" * 32)
        snapshot = store.public_snapshot()
        cursor = store.chain_cursor()
        state = replace(chain(track), finalized_block=150)
        original = build_public_artifacts(
            store, track, state, version_key=0, runtime_manifest=approved_candidate[1],
            release_approval=approved_candidate[2],
        )
        asyncio.run(publisher.publish(original))
        pointer = client.objects["tracks/main/track.json"]
        publisher.release_approval = approval
        # The original one-height/one-snapshot fence remains in force. A normal
        # scan advances one block before the new runtime is publicly observed.
        store.advance_chain_cursor(151, "0x" + "13" * 32)
        cursor = store.chain_cursor()
        state = replace(state, finalized_block=151)
        updated = build_public_artifacts(
            store, track, state, version_key=0, runtime_manifest=payload, release_approval=approval,
        )
        asyncio.run(publisher.publish(updated))
        assert client.objects["tracks/main/track.json"] == pointer
        assert updated.matches == original.matches == ()
        assert json.loads(updated.summary)["champion"] == json.loads(original.summary)["champion"]
        before_weight = decode_weight_artifact(original.weights)
        after_weight = decode_weight_artifact(updated.weights)
        assert after_weight.match_sequence == before_weight.match_sequence
        assert after_weight.terminal_match_sha256 == before_weight.terminal_match_sha256
        assert after_weight.plan.hotkeys == before_weight.plan.hotkeys
        assert after_weight.plan.uids == before_weight.plan.uids
        assert after_weight.plan.weights == before_weight.plan.weights
        assert decode_weight_artifact(updated.weights).track_spec_digest == track.digest
        identity = json.loads(updated.status)["runtime_release"]
        assert identity == approval.public_identity(track, payload)
        assert updated.runtime_manifest[0] != original.runtime_manifest[0]
        # Recovery still authenticates the pinned competition manifest, not the
        # new runtime package, and never clones/relabels the pending submission.
        cold = S3Repository(client, bucket="test", prefix="tracks/main")
        ledger = asyncio.run(cold.recover(track))
        assert ledger.matches == () and ledger.historical_matches == ()
        assert store.chain_cursor() == cursor and store.public_snapshot() == snapshot
        assert store.track_spec_digest() == track.digest


def test_public_status_cannot_claim_a_different_runtime(compatible_runtime):
    track, payload, approval = compatible_runtime
    artifacts = build_genesis_artifacts(track, chain(track), version_key=0,
                                        runtime_manifest=payload, release_approval=approval)
    status = json.loads(artifacts.status)
    status["runtime_release"]["source_sha256"] = "00" * 32
    artifacts = replace(artifacts, status=canonical_json(status))
    client = MemoryS3()
    publisher = S3Repository(client, bucket="test", prefix="tracks/main", release_approval=approval)
    with pytest.raises(PublicArtifactError, match="runtime identity"):
        asyncio.run(publisher.publish(artifacts))
    assert client.calls == []


def test_producer_runtime_update_does_not_replace_competition_bootstrap(
    compatible_runtime, tmp_path, monkeypatch,
):
    """CPU assembly only; storage/pool/installed checks have isolated test doubles."""
    from validator.production import service
    from validator.production.storage import ARTIFACT_MANIFEST_OBJECT, SOURCE_POOL_OBJECT

    track, payload, approval = compatible_runtime
    track_path = tmp_path / "track.json"
    track_path.write_bytes(canonical_json(track_spec_dict(track)))
    runtime_path = tmp_path / "runtime-manifest.json"
    runtime_path.write_bytes(payload)
    competition_path = tmp_path / "competition-manifest.json"
    competition_path.write_bytes(approval.competition_manifest)
    acceptance_path = tmp_path / "accepted.json"
    acceptance_path.write_bytes(approval.acceptance)
    binding_path = tmp_path / "source-binding.json"
    binding_path.write_bytes(b"original-binding")
    bootstrap = {ARTIFACT_MANIFEST_OBJECT: approval.competition_manifest,
                 SOURCE_POOL_OBJECT: b"original-pool"}
    def original(name, _path):
        return bootstrap[name]
    objects = SimpleNamespace(verify_conditional_writes=lambda: None, ensure_bootstrap=original)
    monkeypatch.setattr(service.PrivateObjectStore, "from_environment",
                        lambda **kwargs: objects)
    seen = []
    def verify(manifest):
        assert manifest.encode() == payload
        return {"image_id": "sha256:" + "ab" * 32}
    def pool(source, binding, **kwargs):
        seen.append((source, binding, kwargs))
        return "original-pool-instance"
    monkeypatch.setattr(service, "load_bound_source_pool", pool)
    monkeypatch.setattr(service, "_validate_source_capacity", lambda *_args: None)
    config = SimpleNamespace(
        track_spec=track_path, storage_bucket="private", storage_prefix="evaluation",
        source_pool=tmp_path / "pool", competition_manifest=competition_path,
        competition_payload=lambda _track: approval.competition_manifest,
        data_binding=lambda _track, _runtime: (track, approval.competition_manifest),
        data_competition_manifest=None,
        artifact_manifest=runtime_path, verify_manifest=verify,
        artifact_path=lambda name: acceptance_path if name == release.ACCEPTANCE_LOGICAL_NAME
        else binding_path,
        runtime=SimpleNamespace(source_binding="binding", sample_cache="cache"),
        work_dir=tmp_path / "unchanged-work",
    )
    producer = service.ProducerService(config)
    assert producer.track.digest == track.digest and producer.manifest.encode() == payload
    assert producer.pool == "original-pool-instance"
    assert bootstrap[ARTIFACT_MANIFEST_OBJECT] == approval.competition_manifest
    assert seen[0][:2] == (b"original-pool", b"original-binding")
    assert seen[0][2]["expected_root"] == track.source_pool_merkle_root


@pytest.mark.parametrize("damage", [
    "asset", "target", "baseline", "source", "environment", "image",
])
def test_compatible_release_cannot_hide_changed_rules_or_runtime(compatible_runtime, damage):
    track, payload, approval = compatible_runtime
    manifest = ArtifactManifest.decode(payload)
    if damage == "asset":
        manifest = ArtifactManifest(tuple(
            replace(item, sha256="00" * 32) if item.logical_name == "weights/reference" else item
            for item in manifest.artifacts
        ))
    else:
        result = json.loads(approval.acceptance)
        key = {"target": "track_spec_digest", "baseline": "competition_manifest_sha256",
               "source": "source_sha256", "environment": "environment_sha256",
               "image": "image_id"}[damage]
        result[key] = "00" * 32
        acceptance = canonical_json(result)
        manifest = ArtifactManifest(tuple(
            replace(item, sha256=hashlib.sha256(acceptance).hexdigest(), size_bytes=len(acceptance))
            if item.logical_name == release.ACCEPTANCE_LOGICAL_NAME else item
            for item in manifest.artifacts
        ))
        approval = replace(approval, acceptance=acceptance)
    with pytest.raises(ValueError, match="competition|installation"):
        approval.verify(track, manifest.encode())


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
