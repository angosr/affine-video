"""Competition selection rules for the shared SN120 commitment stream."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

LEGACY_ACTIVATION_PREFIX = b"affine2|activate|"


def legacy_activation_message(hotkey: str, netuid: int) -> bytes:
    """Return the exact signed message used by the existing SN120 contest."""

    body = json.dumps(
        {"hotkey": hotkey, "netuid": netuid},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    registration_id = hashlib.sha256(b"affine-registration-v1\0" + body).hexdigest()
    return f"affine-activate|v1|{netuid}|{hotkey}|{registration_id}".encode()


def is_valid_legacy_activation(
    payload: bytes,
    *,
    hotkey: str,
    netuid: int,
    keypair_type: Any | None = None,
) -> bool:
    """Verify an old-contest activation instead of trusting its text prefix."""

    if not payload.startswith(LEGACY_ACTIVATION_PREFIX):
        return False
    try:
        parts = payload.decode("ascii").strip().split("|")
        if len(parts) != 4 or parts[:2] != ["affine2", "activate"] or parts[2] != hotkey:
            return False
        signature = base64.urlsafe_b64decode(parts[3] + "=" * (-len(parts[3]) % 4))
        if len(signature) != 64:
            return False
        if keypair_type is None:
            from bittensor.wallet import Keypair

            keypair_type = Keypair
        keypair = keypair_type(ss58_address=hotkey, crypto_type=0)
        return bool(keypair.verify(legacy_activation_message(hotkey, netuid), signature))
    except (UnicodeDecodeError, ValueError, TypeError):
        return False
