import json
from pathlib import Path

import pytest

from scoring.spec import load_track_spec
from validator.system_matches import (
    PRODUCTION_NETUID,
    PRODUCTION_TRACK_ID,
    PRODUCTION_TRACK_SPEC_DIGEST,
    system_match_artifacts,
    system_match_summaries,
)


def test_production_system_matches_are_complete_and_separate_from_rewards() -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    assert track.track_id == PRODUCTION_TRACK_ID
    assert track.digest != PRODUCTION_TRACK_SPEC_DIGEST
    assert system_match_artifacts(track.track_id, track.digest, track.netuid) == ()
    artifacts = system_match_artifacts(
        PRODUCTION_TRACK_ID,
        PRODUCTION_TRACK_SPEC_DIGEST,
        PRODUCTION_NETUID,
    )

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
            system_match_summaries(
                PRODUCTION_TRACK_ID,
                PRODUCTION_TRACK_SPEC_DIGEST,
                PRODUCTION_NETUID,
            )
        )
        == 3
    )


def test_system_matches_are_not_attached_to_another_track() -> None:
    assert system_match_artifacts("another-track", "ab" * 32, 7) == ()
    assert system_match_artifacts(PRODUCTION_TRACK_ID, "ab" * 32, PRODUCTION_NETUID) == ()
