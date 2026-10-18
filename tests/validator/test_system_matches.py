import json
from dataclasses import replace
from pathlib import Path

import pytest

from scoring.spec import load_track_spec
from validator import system_matches as system_matches_module
from validator.production.commitment import Artifact, ArtifactManifest
from validator.system_matches import (
    PRODUCTION_NETUID,
    PRODUCTION_TRACK_ID,
    system_match_artifacts,
    system_match_summaries,
)


def _runtime_manifest(
    *,
    asset_sha256: str = "ab" * 32,
    package_sha256: str = "cd" * 32,
    evidence_sha256: str | None = None,
    evidence_name: str = "evidence/system-panel/raw.json",
) -> bytes:
    artifacts = [
        Artifact("runtime/source.zip", package_sha256, 10, "artifacts/" + package_sha256),
        Artifact(
            "weights/predictor.safetensors",
            asset_sha256,
            20,
            "artifacts/" + asset_sha256,
        ),
    ]
    if evidence_sha256 is not None:
        artifacts.append(
            Artifact(
                evidence_name,
                evidence_sha256,
                30,
                "artifacts/" + evidence_sha256,
            )
        )
    return ArtifactManifest(tuple(sorted(artifacts, key=lambda item: item.logical_name))).encode()


def _bind_evidence(monkeypatch, track, runtime_manifest: bytes) -> None:
    system_matches_module._evidence_runtime_digest.cache_clear()
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", lambda: "12" * 32)
    monkeypatch.setattr(
        system_matches_module,
        "evaluation_environment_digest",
        lambda: "34" * 32,
    )
    monkeypatch.setattr(
        system_matches_module,
        "PRODUCTION_EVIDENCE_CONTRACT_DIGEST",
        system_matches_module._evidence_contract_digest(track),
    )
    monkeypatch.setattr(
        system_matches_module,
        "PRODUCTION_EVIDENCE_RUNTIME_DIGEST",
        system_matches_module._evidence_runtime_digest(runtime_manifest),
    )


def test_runtime_evidence_contract_is_computed_once(monkeypatch) -> None:
    runtime_manifest = _runtime_manifest()
    calls = {"environment": 0, "source": 0}

    def environment_digest() -> str:
        calls["environment"] += 1
        return "12" * 32

    def source_digest() -> str:
        calls["source"] += 1
        return "34" * 32

    system_matches_module._evidence_runtime_digest.cache_clear()
    monkeypatch.setattr(
        system_matches_module,
        "evaluation_environment_digest",
        environment_digest,
    )
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", source_digest)

    first = system_matches_module._evidence_runtime_digest(runtime_manifest)
    assert system_matches_module._evidence_runtime_digest(runtime_manifest) == first
    assert calls == {"environment": 1, "source": 1}
    system_matches_module._evidence_runtime_digest.cache_clear()


def test_production_system_matches_are_complete_and_separate_from_rewards(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    assert track.track_id == PRODUCTION_TRACK_ID
    assert system_match_artifacts(track, runtime_manifest) == ()
    _bind_evidence(monkeypatch, track, runtime_manifest)
    artifacts = system_match_artifacts(track, runtime_manifest)

    assert [name for name, _payload in artifacts] == [
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000001.json",
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000002.json",
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000003.json",
    ]
    documents = [json.loads(payload) for _name, payload in artifacts]
    for document in documents:
        assert document["record_type"] == "SYSTEM_BENCHMARK"
        assert document["affects_tournament"] is False
        assert document["reward_eligible"] is False
        assert document["track_spec_digest"] == track.digest
        assert document["evaluation"]["track_contract_sha256"] == (
            system_matches_module._evidence_contract_digest(track)
        )
        assert document["evaluation"]["runtime_contract_sha256"] == (
            system_matches_module._evidence_runtime_digest(runtime_manifest)
        )
        assert document["outcome"] in {"REFERENCE_RETAINED", "REFERENCE_WON"}
        assert len(document["evaluation"]["batches"]) == 10
        assert all(
            batch["challenger_gain_lower_bound"] < document["evaluation"]["minimum_gain"]
            for batch in document["evaluation"]["batches"]
        )

    assert documents[0]["scores"]["reference"] == pytest.approx(0.9451052532287543)
    assert documents[0]["scores"]["challenger"] == pytest.approx(0.9396349056816276)
    assert documents[1]["scores"]["challenger"] == pytest.approx(0.8218193040739775)
    assert all(
        batch["reference_gain_lower_bound"] > 0.005
        for batch in documents[1]["evaluation"]["batches"]
    )
    assert (
        len(
            system_match_summaries(track, runtime_manifest)
        )
        == 3
    )


def test_candidate_track_keeps_the_exact_runtime_schedule() -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")

    assert track.scoring.trajectory_schedule.sigmas == (
        1.0,
        0.9882352941176471,
        0.972972972972973,
        0.9523809523809523,
        0.9230769230769231,
        0.8780487804878049,
        0.8,
        0.631578947368421,
        0.0,
    )


def test_system_evidence_binding_does_not_depend_on_release_artifact_hash(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)
    rebuilt = replace(
        track,
        artifact_manifest_sha256="ef" * 32,
        digest="cd" * 32,
    )

    documents = [
        json.loads(payload)
        for _name, payload in system_match_artifacts(rebuilt, runtime_manifest)
    ]

    assert len(documents) == 3
    assert {document["track_spec_digest"] for document in documents} == {"cd" * 32}


def test_system_evidence_rejects_changed_scoring_assets_but_not_repackaging(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, _runtime_manifest(package_sha256="ef" * 32))
    assert system_match_artifacts(track, _runtime_manifest(evidence_sha256="ef" * 32))
    assert system_match_artifacts(track, _runtime_manifest(asset_sha256="ef" * 32)) == ()
    assert system_match_artifacts(track, None) == ()


def test_only_the_canonical_panel_namespace_is_excluded_from_runtime_semantics(
    monkeypatch,
) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest(evidence_sha256="56" * 32)
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, _runtime_manifest(evidence_sha256="ef" * 32))
    for name in (
        "release/evidence/system-panel/raw.json",
        "evidence/system-panelish/raw.json",
        "other/evidence/system-panel/raw.json",
    ):
        assert system_match_artifacts(
            track,
            _runtime_manifest(evidence_sha256="ef" * 32, evidence_name=name),
        ) == ()


def test_system_matches_are_not_attached_to_another_track(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, runtime_manifest)
    assert system_match_artifacts(replace(track, track_id="another-track"), runtime_manifest) == ()
    assert (
        system_match_artifacts(
            replace(track, netuid=PRODUCTION_NETUID + 1), runtime_manifest
        )
        == ()
    )
    assert system_match_artifacts(
        replace(
            track,
            scoring=replace(track.scoring, minimum_frontier_gain=0.006),
        ),
        runtime_manifest,
    ) == ()


def test_system_evidence_is_invalidated_by_source_pool_changes(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, runtime_manifest)
    assert system_match_artifacts(
        replace(track, source_pool_merkle_root="ef" * 32),
        runtime_manifest,
    ) == ()
    assert system_match_artifacts(
        replace(
            track,
            source_commitments=(
                replace(track.source_commitments[0], sha256="ef" * 32),
                *track.source_commitments[1:],
            ),
        ),
        runtime_manifest,
    ) == ()
