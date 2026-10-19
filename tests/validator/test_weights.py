import asyncio
from types import SimpleNamespace

import pytest
from bittensor.settings import GLOBAL_MAX_SUBNET_COUNT

from validator.queue import ChainPosition
from validator.tournament import Champion
from validator.weights import (
    WeightArtifact,
    WeightChainState,
    WeightPlan,
    WeightSubmissionError,
    plan_champion_weight,
    read_weight_chain_state,
    submit_weights,
    validate_weight_artifact,
    weight_update_due,
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


@pytest.mark.parametrize("mechid", [0, 1, 2])
def test_weight_chain_inputs_come_from_one_finalized_snapshot(mechid: int) -> None:
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

        async def query(self, item, parameters):
            assert item.container == "SubtensorModule"
            assert item.name == "LastUpdate"
            assert parameters == [mechid * GLOBAL_MAX_SUBNET_COUNT + 7]
            assert mechid != 0
            calls.append(("mechanism_last_update", parameters))
            return [5, 0, 0, 6]

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

    state = asyncio.run(
        read_weight_chain_state(FakeClient(), netuid=7, mechid=mechid, finalized_block=123)
    )

    assert state.finalized_block == 123
    assert state.uid_by_hotkey == {"alice": 0, "bob": 3}
    assert state.last_update_by_hotkey == (
        {"alice": 100, "bob": 110} if mechid == 0 else {"alice": 5, "bob": 6}
    )
    assert state.min_allowed_weights == 2
    assert state.max_weight == pytest.approx(32768 / 65535)
    assert state.weights_rate_limit == 100
    assert calls[0] == ("at", 123)


@pytest.mark.parametrize("mechid", [-1, 256, True, 1.0])
def test_weight_chain_rejects_invalid_mechanism_before_chain_access(mechid) -> None:
    with pytest.raises(ValueError, match="mechid"):
        asyncio.run(read_weight_chain_state(object(), netuid=120, mechid=mechid))


@pytest.mark.parametrize("updates", [None, {}, [], [124], [-1], [True], [1.0]])
def test_weight_chain_rejects_invalid_mechanism_last_updates(updates) -> None:
    class Snapshot:
        subnets = SimpleNamespace()

        async def read(self, name, **parameters):
            return {"min_allowed_weights": 1, "max_weight_limit": 65535, "weights_rate_limit": 10}[
                name
            ]

        async def query(self, item, parameters):
            return updates

    async def metagraph(netuid, *, commitments):
        return SimpleNamespace(neurons=[SimpleNamespace(uid=0, hotkey="alice", last_update=1)])

    Snapshot.subnets.metagraph = metagraph

    class Client:
        async def at(self, block):
            assert block == 123
            return Snapshot()

    with pytest.raises(RuntimeError, match="last-update"):
        asyncio.run(read_weight_chain_state(Client(), netuid=120, mechid=1, finalized_block=123))


def test_champion_plan_assigns_all_weight_to_uid_zero_baseline() -> None:
    champion = Champion(
        "genesis",
        "burn-hotkey",
        "official/veda",
        "12" * 20,
        0.7,
        ChainPosition(100, 0, 0),
        uid=0,
        source="VEDA_BASELINE",
    )

    plan = plan_champion_weight(
        champion,
        uid_by_hotkey={"burn-hotkey": 0, "other": 3},
        current_block=120,
        min_allowed_weights=1,
        max_weight=1.0,
    )

    assert plan == WeightPlan(120, ("burn-hotkey",), (0,), (1.0,))


def test_champion_plan_resolves_current_uid_and_fails_closed_on_infeasible_rules() -> None:
    champion = Champion(
        "genesis",
        "burn-hotkey",
        "official/veda",
        "12" * 20,
        0.7,
        ChainPosition(100, 0, 0),
        uid=0,
        source="VEDA_BASELINE",
    )
    arguments = {
        "champion": champion,
        "uid_by_hotkey": {"burn-hotkey": 4},
        "current_block": 120,
    }

    assert plan_champion_weight(**arguments, min_allowed_weights=1, max_weight=1.0) == WeightPlan(
        120, ("burn-hotkey",), (4,), (1.0,)
    )
    assert plan_champion_weight(**arguments, min_allowed_weights=2, max_weight=1.0) is None


def test_public_weights_are_checked_against_current_chain_state() -> None:
    plan = WeightPlan(120, ("alice",), (4,), (1.0,))
    state = chain_state()

    assert (
        validate_weight_artifact(
            artifact(plan=plan),
            state,
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            expected_mechid=0,
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
            expected_mechid=0,
            max_staleness_blocks=20,
        )
    with pytest.raises(ValueError, match="mechanism"):
        validate_weight_artifact(
            artifact(plan=plan),
            state,
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            expected_mechid=1,
            max_staleness_blocks=20,
        )
    assert validate_weight_artifact(
        artifact(plan=WeightPlan(120, ("alice",), (5,), (1.0,))),
        state,
        expected_netuid=7,
        expected_track_digest="ab" * 32,
        expected_mechid=0,
        max_staleness_blocks=20,
    ) == WeightPlan(120, ("alice",), (4,), (1.0,))

    remapped = validate_weight_artifact(
        artifact(
            plan=WeightPlan(
                120,
                ("alice", "validator"),
                (4, 9),
                (0.6, 0.4),
            )
        ),
        chain_state(mapping={"alice": 12, "validator": 2}),
        expected_netuid=7,
        expected_track_digest="ab" * 32,
        expected_mechid=0,
        max_staleness_blocks=20,
    )
    assert remapped == WeightPlan(
        120,
        ("validator", "alice"),
        (2, 12),
        (0.4, 0.6),
    )

    with pytest.raises(ValueError, match="no longer registered"):
        validate_weight_artifact(
            artifact(plan=WeightPlan(120, ("alice",), (4,), (1.0,))),
            chain_state(mapping={"validator": 9}),
            expected_netuid=7,
            expected_track_digest="ab" * 32,
            expected_mechid=0,
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
            expected_mechid=0,
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
