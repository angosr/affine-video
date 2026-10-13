import asyncio
from types import SimpleNamespace

import pytest

from validator.queue import ChainPosition
from validator.store import LeaderboardEntry
from validator.weights import (
    WeightArtifact,
    WeightChainState,
    WeightPlan,
    WeightSubmissionError,
    frontier_gains,
    plan_weights,
    read_weight_chain_state,
    submit_weights,
    validate_weight_artifact,
    waterfill_weights,
    weight_update_due,
)


def entry(
    hotkey: str,
    score: float,
    block: int,
    *,
    provisional: bool = False,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        hotkey=hotkey,
        challenge_id=f"challenge-{hotkey}",
        final_score=score,
        attention_cost=0.1,
        score_block=block,
        provisional=provisional,
        position=ChainPosition(block, 0, 0),
    )


def chain_state(
    *,
    finalized: int = 120,
    mapping: dict[str, int] | None = None,
    minimum: int = 1,
    maximum: float = 1.0,
    rate_limit: int = 10,
    last_updates: dict[str, int] | None = None,
) -> WeightChainState:
    mapping = mapping or {"alice": 4, "validator": 9}
    return WeightChainState(
        finalized,
        mapping,
        minimum,
        maximum,
        rate_limit,
        last_updates or {hotkey: 0 for hotkey in mapping},
    )


def artifact(*, source: int = 120, plan: WeightPlan | None = None) -> WeightArtifact:
    return WeightArtifact(
        "track",
        "ab" * 32,
        7,
        0,
        9,
        source,
        plan,
    )


def test_frontier_gains_telescope_ignore_copies_and_accumulate_hotkey() -> None:
    gains = frontier_gains(
        [
            entry("alice", 0.6, 10),
            entry("copy", 0.6, 20),
            entry("bob", 0.8, 30),
            entry("alice", 0.9, 40),
        ],
        minimum_gain=0.001,
    )

    assert gains == pytest.approx({"alice": 0.7, "bob": 0.2})
    assert sum(gains.values()) == pytest.approx(0.9)


def test_frontier_gains_ignore_numerical_dust() -> None:
    gains = frontier_gains(
        [
            entry("first", 0.6, 10),
            entry("dust", 0.6005, 20),
            entry("material", 0.602, 30),
        ],
        minimum_gain=0.001,
    )
    assert gains == pytest.approx({"first": 0.6, "material": 0.002})


def test_weight_chain_inputs_come_from_one_finalized_snapshot() -> None:
    calls: list[object] = []

    class FakeSubnets:
        async def metagraph(self, netuid: int, *, commitments: bool):
            calls.append(("metagraph", netuid, commitments))
            return SimpleNamespace(
                neurons=[
                    SimpleNamespace(uid=0, hotkey="alice", last_update=100),
                    SimpleNamespace(uid=3, hotkey="bob", last_update=110),
                ]
            )

    class FakeSnapshot:
        subnets = FakeSubnets()

        async def read(self, name: str, **parameters):
            calls.append((name, parameters))
            return {
                "min_allowed_weights": 2,
                "max_weight_limit": 32768,
                "weights_rate_limit": 100,
            }[name]

    class FakeClient:
        async def finalized_block(self):
            raise AssertionError("an explicit finalized block must not be reread")

        async def at(self, block: int):
            calls.append(("at", block))
            return FakeSnapshot()

    state = asyncio.run(read_weight_chain_state(FakeClient(), netuid=7, finalized_block=123))

    assert state.finalized_block == 123
    assert state.uid_by_hotkey == {"alice": 0, "bob": 3}
    assert state.last_update_by_hotkey == {"alice": 100, "bob": 110}
    assert state.min_allowed_weights == 2
    assert state.max_weight == pytest.approx(32768 / 65535)
    assert state.weights_rate_limit == 100
    assert calls[0] == ("at", 123)


def test_plan_excludes_provisional_deregistered_and_future_entries() -> None:
    plan = plan_weights(
        [
            entry("provisional", 0.95, 5, provisional=True),
            entry("alice", 0.3, 10),
            entry("bob", 0.6, 20),
            entry("carol", 0.9, 30),
            entry("deregistered", 1.0, 40),
            LeaderboardEntry(
                hotkey="future",
                challenge_id="challenge-future",
                final_score=1.0,
                attention_cost=0.1,
                score_block=101,
                provisional=False,
                position=ChainPosition(50, 0, 0),
            ),
        ],
        uid_by_hotkey={"alice": 8, "bob": 2, "carol": 5, "future": 9},
        current_block=100,
        min_allowed_weights=3,
        max_weight=0.6,
        minimum_frontier_gain=0.001,
    )

    assert plan is not None
    assert plan.hotkeys == ("bob", "carol", "alice")
    assert plan.uids == (2, 5, 8)
    assert plan.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_plan_rejects_infeasible_or_ambiguous_vectors() -> None:
    entries = [entry("alice", 0.5, 10), entry("bob", 1.0, 20)]
    arguments = {
        "uid_by_hotkey": {"alice": 1, "bob": 2},
        "current_block": 100,
        "minimum_frontier_gain": 0.001,
    }
    assert plan_weights(entries, min_allowed_weights=3, max_weight=1.0, **arguments) is None
    assert plan_weights(entries, min_allowed_weights=2, max_weight=0.4, **arguments) is None

    with pytest.raises(ValueError, match="same UID"):
        plan_weights(
            entries,
            uid_by_hotkey={"alice": 3, "bob": 3},
            current_block=100,
            min_allowed_weights=2,
            max_weight=0.6,
            minimum_frontier_gain=0.001,
        )


def test_waterfill_is_deterministic_normalized_and_capped() -> None:
    result = waterfill_weights(
        {"alice": 8.0, "bob": 1.0, "carol": 1.0},
        caps={"alice": 0.5, "bob": 0.7, "carol": 0.7},
    )
    assert result == pytest.approx({"alice": 0.5, "bob": 0.25, "carol": 0.25})
    assert waterfill_weights({"bob": 2.0, "alice": 1.0}) == waterfill_weights(
        {"alice": 1.0, "bob": 2.0}
    )


def test_public_weights_are_checked_against_current_chain_state() -> None:
    plan = WeightPlan(120, ("alice",), (4,), (1.0,))
    state = chain_state()

    assert (
        validate_weight_artifact(
            artifact(plan=plan),
            state,
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            max_staleness_blocks=20,
        )
        == plan
    )

    with pytest.raises(ValueError, match="stale"):
        validate_weight_artifact(
            artifact(source=90, plan=WeightPlan(90, ("alice",), (4,), (1.0,))),
            state,
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            max_staleness_blocks=20,
        )
    with pytest.raises(ValueError, match="mapping"):
        validate_weight_artifact(
            artifact(plan=WeightPlan(120, ("alice",), (5,), (1.0,))),
            state,
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            max_staleness_blocks=20,
        )

    with pytest.raises(ValueError, match="quantization"):
        validate_weight_artifact(
            artifact(
                plan=WeightPlan(
                    120,
                    ("alice", "validator"),
                    (4, 9),
                    (1.0 - 1e-20, 1e-20),
                )
            ),
            chain_state(minimum=2),
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            max_staleness_blocks=20,
        )


def test_validator_skips_applied_and_rate_limited_artifacts() -> None:
    public = artifact(plan=WeightPlan(120, ("alice",), (4,), (1.0,)))
    due, reason = weight_update_due(public, chain_state(), validator_hotkey="validator")
    assert due and reason == "READY"

    already = chain_state(last_updates={"alice": 0, "validator": 120})
    assert weight_update_due(public, already, validator_hotkey="validator") == (
        False,
        "ALREADY_APPLIED",
    )
    limited = chain_state(last_updates={"alice": 0, "validator": 115})
    assert weight_update_due(public, limited, validator_hotkey="validator") == (
        False,
        "RATE_LIMIT",
    )


def test_submit_weights_uses_existing_client_and_wallet() -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        async def execute(self, intent, wallet, **kwargs):
            calls["intent"] = intent
            calls["wallet"] = wallet
            calls["execute"] = kwargs
            return SimpleNamespace(
                success=True,
                block_hash="0xabc",
                extrinsic_id="12-3",
                message="finalized",
            )

    wallet = object()
    receipt = asyncio.run(
        submit_weights(
            WeightPlan(100, ("alice", "bob"), (1, 2), (0.5, 0.5)),
            client=FakeClient(),
            wallet=wallet,
            netuid=7,
            version_key=9,
        )
    )

    intent = calls["intent"]
    assert intent.netuid == 7
    assert intent.uids == [1, 2]
    assert intent.weights == pytest.approx([0.5, 0.5])
    assert intent.version_key == 9
    assert calls["wallet"] is wallet
    assert calls["execute"] == {
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
    }
    assert receipt.extrinsic_id == "12-3"


def test_submit_weights_surfaces_chain_failure() -> None:
    class FakeClient:
        async def execute(self, *_args, **_kwargs):
            return SimpleNamespace(success=False, message="rate limited")

    with pytest.raises(WeightSubmissionError, match="rate limited"):
        asyncio.run(
            submit_weights(
                WeightPlan(100, ("alice",), (1,), (1.0,)),
                client=FakeClient(),
                wallet=object(),
                netuid=7,
            )
        )
