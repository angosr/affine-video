import pytest

from validator.production.commitment import canonical_json
from validator.queue import ChainPosition, ChallengeState
from validator.tournament import (
    Champion,
    MatchOutcome,
    canonical_score_gain,
    resolve_tournament,
    validate_match_document,
)


def genesis() -> Champion:
    return Champion(
        "genesis",
        "genesis-hotkey",
        "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
        "76f202874608115408d73280be9531c4ab888242",
        0.1,
        ChainPosition(100, 0, 0),
        uid=0,
        source="VEDA_BASELINE",
    )


def challenge(
    sequence: int,
    hotkey: str,
    *,
    state: ChallengeState = ChallengeState.SUCCEEDED,
    score: float | None = None,
    incumbent_score: float | None = None,
    paired_gain: float | None = None,
    error_code: str | None = None,
) -> dict:
    gain = (
        paired_gain
        if paired_gain is not None
        else canonical_score_gain(score, incumbent_score)
        if score is not None and incumbent_score is not None
        else None
    )
    exact_batch = {
        "sha256": f"{sequence + 100:064x}",
        "generated_block": 200 + sequence,
        "plan_digest": f"{sequence + 200:064x}",
        "seed_block_hash": "0x" + "ab" * 32,
    }
    confirmation_batch = {
        "sha256": f"{sequence + 300:064x}",
        "generated_block": 230 + sequence,
        "plan_digest": f"{sequence + 400:064x}",
        "seed_block_hash": "0x" + "cd" * 32,
    }
    stages = []
    final_batch = None
    final_block = None
    if score is not None:
        stages.append(
            {
                "stage": ChallengeState.SCORING,
                "score": score,
                "score_block": 220 + sequence,
                "incumbent_score": incumbent_score,
                "paired_gain_lower_bound": gain,
                "evaluation_batch": exact_batch,
            }
        )
        final_batch = exact_batch
        final_block = 220 + sequence
        if gain is not None and gain > 0.01:
            stages.append(
                {
                    "stage": ChallengeState.CONFIRMING,
                    "score": score,
                    "score_block": 240 + sequence,
                    "incumbent_score": incumbent_score,
                    "paired_gain_lower_bound": gain,
                    "evaluation_batch": confirmation_batch,
                }
            )
            final_batch = confirmation_batch
            final_block = 240 + sequence
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
        "incumbent_score": incumbent_score,
        "paired_gain_lower_bound": gain,
        "evaluation_batch": final_batch,
        "score_block": final_block,
        "stages": stages,
    }


def resolve(items: list[dict], *, history=(), minimum_gain=0.01):
    return resolve_tournament(
        items,
        track_id="track",
        track_spec_digest="ab" * 32,
        netuid=7,
        minimum_gain=minimum_gain,
        initial_champion=genesis(),
        historical_matches=history,
    )


def test_preproduced_confirmation_still_requires_two_independent_decisions():
    item = challenge(1, "alice", score=0.8, incumbent_score=0.7)
    for stage in item["stages"]:
        stage["evaluation_batch"]["generated_block"] = 100
    item["evaluation_batch"] = item["stages"][-1]["evaluation_batch"]
    assert resolve([item]).matches[0].outcome is MatchOutcome.CHALLENGER_WON
    item["stages"][-1]["evaluation_batch"] = item["stages"][0]["evaluation_batch"]
    item["evaluation_batch"] = item["stages"][-1]["evaluation_batch"]
    with pytest.raises(ValueError, match="independent inventory"):
        resolve([item])


def test_system_non_admission_is_not_a_loss_or_lifetime_exclusion():
    first = challenge(1, "alice", state=ChallengeState.SYSTEM_NOT_ADMITTED,
                      error_code="NO_PRE_SUBMISSION_DATA")
    second = challenge(2, "alice", score=0.8, incumbent_score=0.7)
    result = resolve([first, second])
    assert result.matches[0].outcome is MatchOutcome.SYSTEM_NOT_ADMITTED
    assert result.matches[0].champion_after == genesis()
    assert result.matches[1].outcome is MatchOutcome.CHALLENGER_WON
    assert result.champion.hotkey == "alice"
    history = resolve([first]).matches
    next_release = resolve_tournament(
        [second], track_id="next", track_spec_digest="ef" * 32, netuid=7,
        minimum_gain=0.01, initial_champion=genesis(), historical_matches=history,
    )
    assert next_release.champion.hotkey == "alice"


def test_previous_completed_loss_keeps_lifetime_uniqueness_without_relabelling():
    old = challenge(1, "alice", score=0.7, incumbent_score=0.7)
    old["track_spec_digest"] = "cd" * 32
    history = resolve_tournament(
        [old], track_id="previous", track_spec_digest="cd" * 32, netuid=7,
        minimum_gain=0.01, initial_champion=genesis(),
    ).matches
    repeated = challenge(2, "alice", state=ChallengeState.REJECTED_DUPLICATE_HOTKEY,
                         error_code="DUPLICATE_HOTKEY")
    result = resolve([old, repeated], history=history)
    assert len(result.matches) == 1 and result.matches[0].sequence == 1
    assert result.matches[0].challenge_id == repeated["challenge_id"]
    assert old["track_spec_digest"] == "cd" * 32 and old["final_score"] == 0.7
    assert resolve([old], history=history).matches == ()


def test_previous_pending_or_promoted_challenge_forbids_silent_rebinding():
    pending = challenge(1, "alice", state=ChallengeState.SCORING)
    pending["track_spec_digest"] = "cd" * 32
    with pytest.raises(ValueError, match="unfinished previous-track"):
        resolve([pending])
    won = challenge(1, "alice", score=0.8, incumbent_score=0.7)
    won["track_spec_digest"] = "cd" * 32
    with pytest.raises(ValueError, match="verified immutable match"):
        resolve([won])


@pytest.mark.parametrize("old_video_gate", [0.003, 0.005, 0.02])
@pytest.mark.parametrize("new_gate", [0.003, 0.005, 0.02])
def test_historical_two_stage_outcome_requires_proof_not_current_thresholds(
    old_video_gate, new_gate,
):
    old = challenge(1, "alice", score=0.8, incumbent_score=0.7)
    old["track_spec_digest"] = "cd" * 32
    old["paired_gain_lower_bound"] = 0.01
    for stage in old["stages"]:
        stage["paired_gain_lower_bound"] = 0.01
    history = resolve_tournament(
        [old], track_id="previous", track_spec_digest="cd" * 32, netuid=7,
        minimum_gain=old_video_gate, minimum_exact_gain=0.005,
        initial_champion=genesis(),
    ).matches
    original_bytes = canonical_json(history[0].document)
    with pytest.raises(ValueError, match="verified immutable match"):
        resolve([old], minimum_gain=new_gate)
    result = resolve([old], history=history, minimum_gain=new_gate)
    won = old_video_gate < 0.01
    assert history[0].outcome is (
        MatchOutcome.CHALLENGER_WON if won else MatchOutcome.CHAMPION_RETAINED
    )
    assert result.champion == history[0].champion_after
    assert result.champion.hotkey == ("alice" if won else "genesis-hotkey")
    assert result.matches == ()
    assert canonical_json(history[0].document) == original_bytes


@pytest.mark.parametrize("exact_gain", [0.002999999999, 0.003, 0.003000000001])
@pytest.mark.parametrize("full_gain", [0.002999999999, 0.003, 0.003000000001])
def test_candidate_three_milli_gate_remains_strict_for_both_stages(exact_gain, full_gain):
    item = challenge(1, "alice", score=0.8, incumbent_score=0.7)
    item["stages"][0]["paired_gain_lower_bound"] = exact_gain
    item["stages"][1]["paired_gain_lower_bound"] = full_gain
    if exact_gain <= 0.003:
        item["stages"] = item["stages"][:1]
        item["evaluation_batch"] = item["stages"][0]["evaluation_batch"]
        item["score_block"] = item["stages"][0]["score_block"]
        item["paired_gain_lower_bound"] = exact_gain
    else:
        item["paired_gain_lower_bound"] = full_gain
    result = resolve_tournament(
        [item], track_id="candidate", track_spec_digest="ab" * 32, netuid=7,
        minimum_gain=0.003, minimum_exact_gain=0.003, initial_champion=genesis(),
    )
    winner = exact_gain > 0.003 and full_gain > 0.003
    assert result.matches[0].outcome is (
        MatchOutcome.CHALLENGER_WON if winner else MatchOutcome.CHAMPION_RETAINED
    )
    assert result.champion.hotkey == ("alice" if winner else "genesis-hotkey")


def test_lower_gate_does_not_reopen_a_historical_exact_loss_or_spent_opportunity():
    old = challenge(1, "alice", score=0.704, incumbent_score=0.7, paired_gain=0.004)
    old["track_spec_digest"] = "cd" * 32
    historical = resolve_tournament(
        [old], track_id="previous", track_spec_digest="cd" * 32, netuid=7,
        minimum_gain=0.005, minimum_exact_gain=0.005, initial_champion=genesis(),
    ).matches
    original = canonical_json(historical[0].document)
    assert len(historical[0].document["evaluation"]["stages"]) == 1
    repeated = challenge(2, "alice", state=ChallengeState.REJECTED_DUPLICATE_HOTKEY,
                         error_code="DUPLICATE_HOTKEY")
    carried = resolve_tournament(
        [old, repeated], track_id="candidate", track_spec_digest="ab" * 32, netuid=7,
        minimum_gain=0.003, minimum_exact_gain=0.003, initial_champion=genesis(),
        historical_matches=historical,
    )
    assert canonical_json(historical[0].document) == original
    assert carried.champion.hotkey == "genesis-hotkey"
    assert carried.matches[0].state is ChallengeState.REJECTED_DUPLICATE_HOTKEY
    assert len(carried.matches) == 1


def test_confirmed_historical_champion_is_not_rejudged_by_new_thresholds():
    old = challenge(1, "alice", score=0.8, incumbent_score=0.7)
    old["track_spec_digest"] = "cd" * 32
    history = resolve_tournament(
        [old], track_id="previous", track_spec_digest="cd" * 32, netuid=7,
        minimum_gain=0.01, initial_champion=genesis(),
    ).matches
    frozen = dict(history[0].document)
    # Its original 0.1 lower bound would fail the new 0.5 gate. It stays a win.
    carried = resolve([old], history=history, minimum_gain=0.5)
    assert carried.champion.hotkey == "alice" and carried.matches == ()
    assert carried.champion.score == 0.8
    lost = challenge(2, "bob", score=0.7, incumbent_score=0.7)
    kept = resolve([old, lost], history=history, minimum_gain=0.5)
    assert kept.champion.hotkey == "alice" and kept.champion.score == 0.7
    won = challenge(3, "carol", score=0.95, incumbent_score=0.1, paired_gain=0.8)
    replaced = resolve([old, lost, won], history=history, minimum_gain=0.5)
    assert replaced.champion.hotkey == "carol"
    assert history[0].document == frozen


def test_fifo_matches_replace_champion_only_after_required_gain() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.6, incumbent_score=0.1),
            challenge(2, "bob", score=0.605, incumbent_score=0.6),
            challenge(3, "carol", score=0.8, incumbent_score=0.6),
        ]
    )

    assert [match.outcome for match in tournament.matches] == [
        MatchOutcome.CHALLENGER_WON,
        MatchOutcome.CHAMPION_RETAINED,
        MatchOutcome.CHALLENGER_WON,
    ]
    assert tournament.champion is not None
    assert tournament.champion.hotkey == "carol"
    assert tournament.matches[1].document["incumbent_before"]["hotkey"] == "alice"
    assert tournament.matches[1].document["champion_after"]["hotkey"] == "alice"


def test_gain_equal_to_the_threshold_does_not_promote() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.2, incumbent_score=0.1),
            challenge(2, "bob", score=0.21, incumbent_score=0.2),
        ]
    )

    assert tournament.matches[1].score_gain == 0.01
    assert tournament.matches[1].outcome is MatchOutcome.CHAMPION_RETAINED


def test_match_uses_paired_incumbent_score_not_historical_score() -> None:
    tournament = resolve([challenge(1, "alice", score=0.05, incumbent_score=0.04)])

    assert genesis().score == 0.1
    assert tournament.matches[0].score_gain == pytest.approx(0.01)
    assert tournament.matches[0].outcome is MatchOutcome.CHAMPION_RETAINED
    assert tournament.champion.hotkey == "genesis-hotkey"


def test_match_decision_uses_paired_gain_bound_not_raw_score_gap() -> None:
    retained = resolve([challenge(1, "alice", score=0.8, incumbent_score=0.5, paired_gain=0.009)])
    won = resolve([challenge(1, "alice", score=0.511, incumbent_score=0.5, paired_gain=0.011)])

    assert retained.matches[0].outcome is MatchOutcome.CHAMPION_RETAINED
    assert won.matches[0].outcome is MatchOutcome.CHALLENGER_WON


def test_failed_match_keeps_champion_and_is_publicly_auditable() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.6, incumbent_score=0.1),
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
            challenge(1, "alice", score=0.6, incumbent_score=0.1),
            challenge(2, "bob", state=ChallengeState.SCORING),
            challenge(3, "carol", score=0.9, incumbent_score=0.6),
        ]
    )

    assert [match.challenge_id for match in tournament.matches] == ["challenge-1"]
    assert tournament.matches[0].object_name == "matches/track/000000000001.json"


def test_public_match_validation_rejects_nonterminal_records() -> None:
    public = resolve([challenge(1, "alice", score=0.6, incumbent_score=0.1)]).matches[0].document
    tampered = {
        **public,
        "evaluation": {
            **public["evaluation"],
            "state": ChallengeState.SCORING,
            "error_code": "TEMPORARY",
            "final_score": None,
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
            incumbent=genesis(),
        )


def test_duplicate_hotkey_is_terminally_rejected_and_excluded_from_rankings() -> None:
    tournament = resolve(
        [
            challenge(1, "alice", score=0.7, incumbent_score=0.1),
            challenge(2, "bob", score=0.8, incumbent_score=0.7),
            challenge(
                3,
                "alice",
                state=ChallengeState.REJECTED_DUPLICATE_HOTKEY,
                error_code="REJECTED_DUPLICATE_HOTKEY",
            ),
        ]
    )

    assert [(entry["hotkey"], entry["score"]) for entry in tournament.leaderboard] == [
        ("bob", 0.8),
        ("alice", 0.7),
        ("genesis-hotkey", 0.1),
    ]


def test_tournament_rejects_a_second_non_rejected_submission_for_one_hotkey() -> None:
    with pytest.raises(ValueError, match="later submissions"):
        resolve(
            [
                challenge(1, "alice", score=0.7, incumbent_score=0.1),
                challenge(2, "alice", score=0.8, incumbent_score=0.7),
            ]
        )
