import asyncio
from types import SimpleNamespace

import bittensor as bt
import pytest

from miner import chain
from miner.chain import ChainSubmissionError, commitment_info, submit_model
from miner.submit import ModelSubmission


def test_commitment_info_uses_runtime_raw_length_variant() -> None:
    payload = ModelSubmission("owner/model", "12" * 20).encode()

    info = commitment_info(payload)
    call = bt.calls.Commitments.set_commitment(netuid=7, info=info)

    assert info == {"fields": [{f"Raw{len(payload)}": payload}]}
    assert call.module == "Commitments"
    assert call.function == "set_commitment"
    assert call.params == {"netuid": 7, "info": info}


def test_commitment_info_enforces_runtime_limit() -> None:
    with pytest.raises(ValueError, match="empty"):
        commitment_info(b"")
    with pytest.raises(ValueError, match="128"):
        commitment_info(b"x" * 129)


def test_submit_model_signs_with_hotkey_and_waits_for_finalization(monkeypatch) -> None:
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

        async def submit_call(self, call_value, wallet, **kwargs):
            calls["call"] = call_value
            calls["submit"] = kwargs
            return SimpleNamespace(
                success=True,
                block_hash="0xabc",
                extrinsic_id="10-2",
                message="finalized",
            )

    monkeypatch.setattr(chain.bt, "Wallet", FakeWallet)
    monkeypatch.setattr(chain.bt, "Client", FakeClient)
    submission = ModelSubmission("owner/model", "12" * 20)

    receipt = asyncio.run(
        submit_model(
            submission,
            netuid=7,
            network="test",
            wallet_name="alice",
            hotkey_name="miner",
            wallet_path="/wallets",
        )
    )

    assert receipt.block_hash == "0xabc"
    assert calls["wallet"] == {"name": "alice", "hotkey": "miner", "path": "/wallets"}
    assert calls["network"] == "test"
    assert calls["submit"] == {
        "signer": "hotkey",
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
    }


def test_submit_model_surfaces_chain_failure(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, _network: str) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def submit_call(self, *_args, **_kwargs):
            return SimpleNamespace(success=False, message="not registered")

    monkeypatch.setattr(chain.bt, "Wallet", lambda **_kwargs: object())
    monkeypatch.setattr(chain.bt, "Client", FakeClient)

    with pytest.raises(ChainSubmissionError, match="not registered"):
        asyncio.run(
            submit_model(
                ModelSubmission("owner/model", "12" * 20),
                netuid=7,
                network="test",
                wallet_name="alice",
                hotkey_name="miner",
            )
        )


def test_submit_model_wraps_network_failure(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, _network: str) -> None:
            pass

        async def __aenter__(self):
            raise OSError("private endpoint detail")

        async def __aexit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(chain.bt, "Wallet", lambda **_kwargs: object())
    monkeypatch.setattr(chain.bt, "Client", FakeClient)

    with pytest.raises(ChainSubmissionError, match="did not complete"):
        asyncio.run(
            submit_model(
                ModelSubmission("owner/model", "12" * 20),
                netuid=7,
                network="test",
                wallet_name="alice",
                hotkey_name="miner",
            )
        )


@pytest.mark.parametrize(
    ("block_hash", "extrinsic_id"),
    [(None, "10-2"), ("0xabc", None), ("", "10-2"), ("0xabc", "")],
)
def test_submit_model_requires_traceable_finalized_receipt(
    monkeypatch, block_hash: str | None, extrinsic_id: str | None
) -> None:
    class FakeClient:
        def __init__(self, _network: str) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def submit_call(self, *_args, **_kwargs):
            return SimpleNamespace(
                success=True,
                block_hash=block_hash,
                extrinsic_id=extrinsic_id,
                message="finalized",
            )

    monkeypatch.setattr(chain.bt, "Wallet", lambda **_kwargs: object())
    monkeypatch.setattr(chain.bt, "Client", FakeClient)

    with pytest.raises(ChainSubmissionError, match="missing chain identifiers"):
        asyncio.run(
            submit_model(
                ModelSubmission("owner/model", "12" * 20),
                netuid=7,
                network="test",
                wallet_name="alice",
                hotkey_name="miner",
            )
        )
