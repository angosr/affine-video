import asyncio
from types import SimpleNamespace

import pytest

from validator import weights as weights_module
from validator.queue import ChainPosition
from validator.store import ChallengeStore, LeaderboardEntry, WeightRunState
from validator.weights import (
    WeightReceipt,
    WeightSubmissionError,
    frontier_gains,
    plan_weights,
    read_weight_chain_state,
    submit_recorded_weights,
    submit_weights,
    update_weights,
    waterfill_weights,
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


def test_frontier_gains_telescope_and_give_copies_no_extra_utility() -> None:
    gains = frontier_gains(
        [entry("copy", 0.6, 30), entry("first", 0.6, 10), entry("better", 0.9, 20)],
        minimum_gain=0.001,
    )

    assert gains == pytest.approx({"first": 0.6, "better": 0.3})
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


def test_frontier_gains_accumulate_same_hotkey_improvements() -> None:
    gains = frontier_gains(
        [
            entry("alice", 0.6, 10),
            entry("bob", 0.8, 20),
            entry("alice", 0.9, 30),
        ],
        minimum_gain=0.001,
    )

    assert gains == pytest.approx({"alice": 0.7, "bob": 0.2})
    assert sum(gains.values()) == pytest.approx(0.9)


def test_weight_chain_inputs_come_from_one_finalized_snapshot() -> None:
    calls: list[object] = []

    class FakeSubnets:
        async def metagraph(self, netuid: int, *, commitments: bool):
            calls.append(("metagraph", netuid, commitments))
            return SimpleNamespace(
                neurons=[
                    SimpleNamespace(uid=0, hotkey="alice"),
                    SimpleNamespace(uid=3, hotkey="bob"),
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
            return 123

        async def at(self, block: int):
            calls.append(("at", block))
            return FakeSnapshot()

    state = asyncio.run(read_weight_chain_state(FakeClient(), netuid=7))

    assert state.finalized_block == 123
    assert state.uid_by_hotkey == {"alice": 0, "bob": 3}
    assert state.min_allowed_weights == 2
    assert state.max_weight == pytest.approx(32768 / 65535)
    assert state.weights_rate_limit == 100
    assert calls[0] == ("at", 123)


def test_weight_update_obeys_block_interval_and_submits_one_snapshot(monkeypatch) -> None:
    async def fake_state(_client, *, netuid: int):
        assert netuid == 7
        return weights_module.WeightChainState(120, {"alice": 4}, 1, 1.0, 10)

    submitted = []

    async def fake_submit(store, plan, **kwargs):
        submitted.append((store, plan, kwargs))
        return SimpleNamespace(source_block=plan.source_block), object()

    monkeypatch.setattr(weights_module, "read_weight_chain_state", fake_state)
    monkeypatch.setattr(weights_module, "submit_recorded_weights", fake_submit)
    store = SimpleNamespace(
        reward_entries=lambda: [entry("alice", 0.8, 100)],
        latest_weight_run=lambda: None,
        chain_cursor=lambda: (120, "0x" + "12" * 32),
    )

    result = asyncio.run(
        update_weights(
            object(),
            store,
            netuid=7,
            network="test",
            wallet_name="validator",
            hotkey_name="default",
            interval_blocks=10,
            minimum_frontier_gain=0.001,
        )
    )

    assert result.source_block == 120
    assert len(submitted) == 1

    store.latest_weight_run = lambda: SimpleNamespace(source_block=115)
    skipped = asyncio.run(
        update_weights(
            object(),
            store,
            netuid=7,
            network="test",
            wallet_name="validator",
            hotkey_name="default",
            interval_blocks=10,
            minimum_frontier_gain=0.001,
        )
    )
    assert skipped is None
    assert len(submitted) == 1


def test_weight_update_waits_for_submission_scan_to_reach_snapshot(monkeypatch) -> None:
    async def fake_state(_client, *, netuid: int):
        assert netuid == 7
        return weights_module.WeightChainState(120, {"alice": 4}, 1, 1.0, 10)

    submitted = []

    async def fake_submit(*args, **kwargs):
        submitted.append((args, kwargs))
        raise AssertionError("weights must not be submitted while the scanner is behind")

    monkeypatch.setattr(weights_module, "read_weight_chain_state", fake_state)
    monkeypatch.setattr(weights_module, "submit_recorded_weights", fake_submit)
    store = SimpleNamespace(
        chain_cursor=lambda: (119, "0x" + "11" * 32),
        reward_entries=lambda: [entry("alice", 0.8, 100)],
        latest_weight_run=lambda: None,
    )

    result = asyncio.run(
        update_weights(
            object(),
            store,
            netuid=7,
            network="test",
            wallet_name="validator",
            hotkey_name="default",
            interval_blocks=10,
            minimum_frontier_gain=0.001,
        )
    )

    assert result is None
    assert submitted == []


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


def test_old_score_remains_eligible_while_track_is_unchanged() -> None:
    delayed = LeaderboardEntry(
        hotkey="alice",
        challenge_id="challenge-alice",
        final_score=1.0,
        attention_cost=0.1,
        score_block=100,
        provisional=False,
        position=ChainPosition(1, 0, 0),
    )

    plan = plan_weights(
        [delayed],
        uid_by_hotkey={"alice": 7},
        current_block=110,
        min_allowed_weights=1,
        max_weight=1.0,
        minimum_frontier_gain=0.001,
    )

    assert plan is not None
    assert plan.uids == (7,)


def test_plan_returns_none_when_count_or_caps_are_infeasible() -> None:
    entries = [entry("alice", 0.5, 10), entry("bob", 1.0, 20)]
    arguments = {
        "uid_by_hotkey": {"alice": 1, "bob": 2},
        "current_block": 100,
        "minimum_frontier_gain": 0.001,
    }

    assert plan_weights(entries, min_allowed_weights=3, max_weight=1.0, **arguments) is None
    assert plan_weights(entries, min_allowed_weights=2, max_weight=0.4, **arguments) is None


def test_plan_rejects_weights_that_disappear_during_sdk_quantization() -> None:
    plan = plan_weights(
        [entry("alice", 0.5, 10), entry("bob", 0.5000000001, 20)],
        uid_by_hotkey={"alice": 1, "bob": 2},
        current_block=100,
        min_allowed_weights=2,
        max_weight=1.0,
        minimum_frontier_gain=1e-12,
    )

    assert plan is None


def test_plan_rejects_duplicate_uid_mapping() -> None:
    with pytest.raises(ValueError, match="same UID"):
        plan_weights(
            [entry("alice", 0.5, 10), entry("bob", 1.0, 20)],
            uid_by_hotkey={"alice": 3, "bob": 3},
            current_block=100,
            min_allowed_weights=2,
            max_weight=0.6,
            minimum_frontier_gain=0.001,
        )


def test_waterfill_is_normalized_and_respects_caps() -> None:
    result = waterfill_weights(
        {"alice": 8.0, "bob": 1.0, "carol": 1.0},
        caps={"alice": 0.5, "bob": 0.7, "carol": 0.7},
    )

    assert result == pytest.approx({"alice": 0.5, "bob": 0.25, "carol": 0.25})
    assert sum(result.values()) == pytest.approx(1.0)


def test_waterfill_is_independent_of_mapping_order() -> None:
    forward = waterfill_weights({"alice": 1.0, "bob": 2.0})
    reverse = waterfill_weights({"bob": 2.0, "alice": 1.0})

    assert forward == reverse


def test_infeasible_caps_are_rejected() -> None:
    with pytest.raises(ValueError):
        waterfill_weights({"alice": 1, "bob": 1}, caps={"alice": 0.4, "bob": 0.4})


def test_submit_weights_uses_intent_and_waits_for_finalization(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeWallet:
        def __init__(self, **kwargs) -> None:
            calls["wallet"] = kwargs

    class FakeClient:
        def __init__(self, network: str) -> None:
            calls["network"] = network

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def execute(self, intent, wallet, **kwargs):
            calls["intent"] = intent
            calls["execute"] = kwargs
            return SimpleNamespace(
                success=True,
                block_hash="0xabc",
                extrinsic_id="12-3",
                message="finalized",
            )

    monkeypatch.setattr(weights_module.bt, "Wallet", FakeWallet)
    monkeypatch.setattr(weights_module.bt, "Client", FakeClient)
    plan = plan_weights(
        [entry("alice", 0.5, 10), entry("bob", 1.0, 20)],
        uid_by_hotkey={"alice": 1, "bob": 2},
        current_block=100,
        min_allowed_weights=2,
        max_weight=0.6,
        minimum_frontier_gain=0.001,
    )
    assert plan is not None

    receipt = asyncio.run(
        submit_weights(
            plan,
            netuid=7,
            network="test",
            wallet_name="validator",
            hotkey_name="default",
            wallet_path="/wallets",
            version_key=9,
        )
    )

    intent = calls["intent"]
    assert intent.netuid == 7
    assert intent.uids == [1, 2]
    assert intent.weights == pytest.approx([0.5, 0.5])
    assert intent.version_key == 9
    assert calls["wallet"] == {
        "name": "validator",
        "hotkey": "default",
        "path": "/wallets",
    }
    assert calls["execute"] == {
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
    }
    assert receipt.extrinsic_id == "12-3"


def test_submit_weights_surfaces_chain_failure(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, _network: str) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def execute(self, *_args, **_kwargs):
            return SimpleNamespace(success=False, message="rate limited")

    monkeypatch.setattr(weights_module.bt, "Wallet", lambda **_kwargs: object())
    monkeypatch.setattr(weights_module.bt, "Client", FakeClient)
    plan = plan_weights(
        [entry("alice", 1.0, 100)],
        uid_by_hotkey={"alice": 1},
        current_block=100,
        min_allowed_weights=1,
        max_weight=1.0,
        minimum_frontier_gain=0.001,
    )
    assert plan is not None

    with pytest.raises(WeightSubmissionError, match="rate limited"):
        asyncio.run(
            submit_weights(
                plan,
                netuid=7,
                network="test",
                wallet_name="validator",
                hotkey_name="default",
            )
        )


def test_recorded_submission_persists_finalized_result(monkeypatch, tmp_path) -> None:
    async def fake_submit(*_args, **_kwargs):
        return WeightReceipt(
            block_hash="0xabc",
            extrinsic_id="12-3",
            message="finalized",
        )

    monkeypatch.setattr(weights_module, "submit_weights", fake_submit)
    plan = plan_weights(
        [entry("alice", 1.0, 100)],
        uid_by_hotkey={"alice": 1},
        current_block=100,
        min_allowed_weights=1,
        max_weight=1.0,
        minimum_frontier_gain=0.001,
    )
    assert plan is not None

    with ChallengeStore(tmp_path / "validator.db") as store:
        run, _receipt = asyncio.run(
            submit_recorded_weights(
                store,
                plan,
                netuid=7,
                network="test",
                wallet_name="validator",
                hotkey_name="default",
            )
        )

        assert run.state is WeightRunState.FINALIZED
        assert run.uids == (1,)
        assert store.weight_runs() == [run]
