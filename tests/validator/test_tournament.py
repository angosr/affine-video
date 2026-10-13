import pytest

from validator.queue import ChallengeState
from validator.tournament import MatchOutcome, resolve_tournament, validate_match_document


def challenge(
    sequence: int,
    hotkey: str,
    *,
    state: ChallengeState = ChallengeState.SUCCEEDED,
    score: float | None = None,
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
        "error_code": error_code,
        "file_sha256": "ab" * 32 if score is not None else None,
        "model_digest": "cd" * 32 if score is not None else None,
        "model_size_bytes": 123 if score is not None else None,
        "final_score": score,
        "attention_cost": 0.2 if score is not None else None,
        "score_block": 200 + sequence if score is not None else None,
        "stages": [],
    }


def resolve(items: list[dict]):
    return resolve_tournament(
        items,
        track_id="track",
        track_spec_digest="ab" * 32,
        netuid=7,
        minimum_gain=0.01,
    )


def test_fifo_matches_replace_champion_only_after_required_gain() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.6),
            challenge(2, "bob", score=0.605),
            challenge(3, "carol", score=0.8),
        ]
    )

    assert [match.outcome for match in tournament.matches] == [
        MatchOutcome.INITIAL_CHAMPION,
        MatchOutcome.CHAMPION_RETAINED,
        MatchOutcome.CHALLENGER_WON,
    ]
    assert tournament.champion is not None
    assert tournament.champion.hotkey == "carol"
    assert tournament.matches[1].document["incumbent_before"]["hotkey"] == "alice"
    assert tournament.matches[1].document["champion_after"]["hotkey"] == "alice"


def test_exact_decimal_minimum_gain_is_not_lost_to_binary_rounding() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.2),
            challenge(2, "bob", score=0.21),
        ]
    )

    assert tournament.matches[1].score_gain == 0.01
    assert tournament.matches[1].outcome is MatchOutcome.CHALLENGER_WON


def test_failed_match_keeps_champion_and_is_publicly_auditable() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.6),
            challenge(
                2,
                "bob",
                state=ChallengeState.PARTICIPANT_FAILED,
                error_code="REPOSITORY_UNAVAILABLE",
            ),
        ]
    )

    failed = tournament.matches[1]
    assert failed.outcome is MatchOutcome.CHALLENGE_FAILED
    assert failed.document["evaluation"]["error_code"] == "REPOSITORY_UNAVAILABLE"
    assert "error_summary" not in failed.document["evaluation"]
    assert tournament.champion is not None and tournament.champion.hotkey == "alice"


def test_unfinished_submission_stops_the_public_match_prefix() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.6),
            challenge(2, "bob", state=ChallengeState.ROLLOUT),
            challenge(3, "carol", score=0.9),
        ]
    )

    assert [match.challenge_id for match in tournament.matches] == ["challenge-1"]
    assert tournament.matches[0].object_name == "matches/000000000001.json"


def test_public_match_validation_rejects_nonterminal_records() -> None:
    public = resolve([challenge(1, "alice", score=0.6)]).matches[0].document
    tampered = {
        **public,
        "evaluation": {
            **public["evaluation"],
            "state": ChallengeState.ROLLOUT,
            "error_code": "TEMPORARY",
            "final_score": None,
            "attention_cost": None,
        },
        "score_gain": None,
        "outcome": MatchOutcome.CHALLENGE_FAILED,
        "champion_after": None,
    }

    with pytest.raises(ValueError, match="terminal"):
        validate_match_document(
            tampered,
            sequence=1,
            track_id="track",
            track_spec_digest="ab" * 32,
            netuid=7,
            minimum_gain=0.01,
            incumbent=None,
        )


def test_leaderboard_keeps_each_hotkeys_best_authoritative_score() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.7),
            challenge(2, "bob", score=0.8),
            challenge(3, "alice", score=0.6),
        ]
    )

    assert [(entry["hotkey"], entry["score"]) for entry in tournament.leaderboard] == [
        ("bob", 0.8),
        ("alice", 0.7),
    ]
    assert [entry.final_score for entry in tournament.reward_entries] == [0.7, 0.8, 0.6]
