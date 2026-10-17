import json

import pytest

from validator.system_matches import (
    PRODUCTION_NETUID,
    PRODUCTION_TRACK_ID,
    system_match_artifacts,
    system_match_summaries,
)


def test_production_system_matches_are_complete_and_separate_from_rewards() -> None:
    artifacts = system_match_artifacts(
        PRODUCTION_TRACK_ID,
        "ab" * 32,
        PRODUCTION_NETUID,
    )

    assert [name for name, _payload in artifacts] == [
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000001.json",
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000002.json",
    ]
    documents = [json.loads(payload) for _name, payload in artifacts]
    for document in documents:
        assert document["record_type"] == "SYSTEM_BENCHMARK"
        assert document["affects_tournament"] is False
        assert document["reward_eligible"] is False
        assert document["outcome"] == "REFERENCE_WON"
        assert len(document["evaluation"]["panels"]) == 4
        assert all(
            panel["reference"]["heat_kept"] > panel["challenger"]["heat_kept"]
            for panel in document["evaluation"]["panels"]
        )
        assert (
            document["scores"]["reference"]["heat_kept"]
            > document["scores"]["challenger"]["heat_kept"]
        )

    assert documents[0]["scores"]["reference"]["heat_kept"] == pytest.approx(0.549221386869737)
    assert documents[0]["scores"]["challenger"]["heat_kept"] == pytest.approx(0.505509267860990)
    assert documents[1]["scores"]["challenger"]["heat_kept"] == pytest.approx(0.224767047868143)
    assert (
        len(
            system_match_summaries(
                PRODUCTION_TRACK_ID,
                "ab" * 32,
                PRODUCTION_NETUID,
            )
        )
        == 2
    )


def test_system_matches_are_not_attached_to_another_track() -> None:
    assert system_match_artifacts("another-track", "ab" * 32, 7) == ()
