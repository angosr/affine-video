from itertools import pairwise

import pytest

from validator.queue import (
    TERMINAL_STATES,
    ChainPosition,
    ChallengeState,
    require_transition,
    submission_id,
)


def test_chain_positions_sort_by_finalized_location() -> None:
    positions = [
        ChainPosition(11, 0, 0),
        ChainPosition(10, 2, 0),
        ChainPosition(10, 1, 1),
    ]

    assert sorted(positions) == [positions[2], positions[1], positions[0]]


def test_submission_id_is_stable_and_domain_bound() -> None:
    block_hash = "0x" + "ab" * 32
    position = ChainPosition(10, 2, 1)

    first = submission_id(7, block_hash, position)
    second = submission_id(7, block_hash, position)

    assert first == second
    assert first != submission_id(8, block_hash, position)
    assert len(first) == 64


def test_state_machine_accepts_scored_success() -> None:
    path = [
        ChallengeState.QUEUED,
        ChallengeState.FETCHING,
        ChallengeState.VALIDATING,
        ChallengeState.SCORING,
        ChallengeState.SUCCEEDED,
    ]

    for current, target in pairwise(path):
        require_transition(current, target)

    assert path[-1] in TERMINAL_STATES


def test_terminal_state_cannot_restart() -> None:
    with pytest.raises(ValueError):
        require_transition(ChallengeState.PARTICIPANT_FAILED, ChallengeState.FETCHING)
