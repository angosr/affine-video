import json
from dataclasses import replace
from pathlib import Path

import pytest

from scoring.spec import load_track_spec
from validator import system_matches as system_matches_module
from validator.system_matches import (
    PRODUCTION_NETUID,
    PRODUCTION_TRACK_ID,
    system_match_artifacts,
    system_match_summaries,
)


def test_production_system_matches_are_complete_and_separate_from_rewards(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    assert track.track_id == PRODUCTION_TRACK_ID
    assert system_match_artifacts(track) == ()
    monkeypatch.setattr(
        system_matches_module,
        "PRODUCTION_EVIDENCE_CONTRACT_DIGEST",
        system_matches_module._evidence_contract_digest(track),
    )
    artifacts = system_match_artifacts(track)

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
            system_match_summaries(track)
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
    monkeypatch.setattr(
        system_matches_module,
        "PRODUCTION_EVIDENCE_CONTRACT_DIGEST",
        system_matches_module._evidence_contract_digest(track),
    )
    rebuilt = replace(
        track,
        artifact_manifest_sha256="ef" * 32,
        digest="cd" * 32,
    )

    documents = [json.loads(payload) for _name, payload in system_match_artifacts(rebuilt)]

    assert len(documents) == 3
    assert {document["track_spec_digest"] for document in documents} == {"cd" * 32}


def test_system_matches_are_not_attached_to_another_track(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    monkeypatch.setattr(
        system_matches_module,
        "PRODUCTION_EVIDENCE_CONTRACT_DIGEST",
        system_matches_module._evidence_contract_digest(track),
    )

    assert system_match_artifacts(track)
    assert system_match_artifacts(replace(track, track_id="another-track")) == ()
    assert system_match_artifacts(replace(track, netuid=PRODUCTION_NETUID + 1)) == ()
    assert system_match_artifacts(
        replace(
            track,
            scoring=replace(track.scoring, minimum_frontier_gain=0.006),
        )
    ) == ()
