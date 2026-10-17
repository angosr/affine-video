import base64

from validator.competition import (
    is_valid_legacy_activation,
    legacy_activation_message,
)


class FakeKeypair:
    signature = b"s" * 64

    def __init__(self, *, ss58_address: str, crypto_type: int) -> None:
        assert ss58_address == "miner"
        assert crypto_type == 0

    def verify(self, message: bytes, signature: bytes) -> bool:
        return message == legacy_activation_message("miner", 120) and signature == self.signature


def test_legacy_activation_requires_hotkey_bound_valid_signature() -> None:
    signature = base64.urlsafe_b64encode(FakeKeypair.signature).rstrip(b"=").decode()
    payload = f"affine2|activate|miner|{signature}".encode()

    assert is_valid_legacy_activation(
        payload,
        hotkey="miner",
        netuid=120,
        keypair_type=FakeKeypair,
    )
    assert not is_valid_legacy_activation(
        payload.replace(b"|miner|", b"|other|"),
        hotkey="miner",
        netuid=120,
        keypair_type=FakeKeypair,
    )
    assert not is_valid_legacy_activation(
        payload.replace(b"|c3", b"|d3"),
        hotkey="miner",
        netuid=120,
        keypair_type=FakeKeypair,
    )
