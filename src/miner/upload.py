"""Chain activation, local Ed25519 mailbox decrypt, native R2 upload, then ready.

No private key or management credential leaves its owner. Upload is not the
formal competition submission; the separate finalized ready fixes S.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bittensor as bt
import httpx

from miner.chain import submit_model
from miner.submit import (
    PrivateModelSubmission,
    UploadActivation,
    activation_message,
    upload_registration_id,
)
from validator.chunks import build_chunk_manifest, read_part
from validator.production.commitment import canonical_json, decode_canonical_json
from validator.s3 import client_config
from validator.uploads import ed25519_verify, envelope_signing_bytes, manifest_signing_bytes


async def upload_model(
    model: str | Path, *, mailbox_url: str, validator_identity: str, network: str,
    wallet_name: str, hotkey_name: str, wallet_path: str | None = None,
) -> PrivateModelSubmission:
    """Activate using the EXISTING Ed25519 hotkey, then upload; never auto-ready."""
    if not network:
        raise ValueError("upload network cannot be empty")
    arguments = {"name": wallet_name, "hotkey": hotkey_name}
    if wallet_path is not None:
        arguments["path"] = wallet_path
    wallet = bt.Wallet(**arguments)
    if getattr(wallet.hotkey, "crypto_type", None) != 0:
        raise ValueError("private upload requires an existing registered Ed25519 hotkey")
    async with bt.Client(network) as chain:
        genesis = (await chain.block_info(0)).hash
    if not isinstance(genesis, str):
        raise RuntimeError("upload network genesis is unavailable")
    genesis = genesis.removeprefix("0x")
    activation = UploadActivation(wallet.hotkey.sign(
        activation_message(genesis, wallet.hotkey.ss58_address),
    ))
    receipt = await submit_model(activation, netuid=120, network=network, wallet_name=wallet_name,
                                 hotkey_name=hotkey_name, wallet_path=wallet_path)
    async with httpx.AsyncClient(follow_redirects=False, timeout=20) as client:
        return await upload_file(model, mailbox_url=mailbox_url,
                                 validator_identity=validator_identity, genesis_hash=genesis,
                                 hotkey=wallet.hotkey, client=client,
                                 activation_block_hash=receipt.block_hash.removeprefix("0x"))


def _temporary_client(envelope: dict):
    import boto3

    return boto3.client(
        "s3", endpoint_url=envelope["endpoint"], region_name="auto",
        aws_access_key_id=envelope["access_key_id"],
        aws_secret_access_key=envelope["secret_access_key"],
        aws_session_token=envelope["session_token"], config=client_config(),
    )


def decrypt_mailbox(
    blob: bytes, *, hotkey: Any, validator_identity: str, genesis_hash: str,
) -> dict:
    """Pin the validator BEFORE trusting any endpoint, prefix or credential."""
    if getattr(hotkey, "crypto_type", None) != 0:
        raise ValueError("private mailbox requires Ed25519; sr25519 is not decrypt-compatible")
    if not blob or len(blob) > 65536:
        raise ValueError("private mailbox blob is invalid")
    envelope = decode_canonical_json(hotkey.decrypt(blob))
    expected = {
        "format", "genesis_hash", "hotkey", "upload_id", "generation", "endpoint", "bucket",
        "prefix", "validator", "expires_at", "access_key_id", "secret_access_key",
        "session_token", "signature", "activation_block_hash",
    }
    upload_id = upload_registration_id(genesis_hash, hotkey.ss58_address)
    if (not isinstance(envelope, dict) or set(envelope) != expected
            or envelope["format"] != "veda-mailbox-v1"
            or envelope["genesis_hash"] != genesis_hash
            or envelope["hotkey"] != hotkey.ss58_address or envelope["upload_id"] != upload_id
            or envelope["validator"] != validator_identity
            or not ed25519_verify(validator_identity, envelope_signing_bytes(envelope),
                                  bytes.fromhex(envelope["signature"]))
            or type(envelope["generation"]) is not int or envelope["generation"] < 1
            or type(envelope["expires_at"]) is not int
            or re.fullmatch(r"[a-f0-9]{64}", envelope["activation_block_hash"]) is None
            or not re.fullmatch(
                r"https://[a-f0-9]{32}(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com/?",
                envelope["endpoint"],
            )
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", envelope["bucket"])
            or envelope["bucket"] == "affine_video"
            or envelope["prefix"] != f"uploads/v1/{upload_id}/"
            or not all(isinstance(envelope[k], str) and envelope[k]
                       for k in ("access_key_id", "secret_access_key", "session_token"))):
        raise ValueError("private mailbox identity, signature, scope or expiry is invalid")
    return envelope


async def upload_file(
    model: str | Path, *, mailbox_url: str, validator_identity: str, genesis_hash: str,
    hotkey: Any, client: httpx.AsyncClient, s3_factory: Any = _temporary_client,
    timeout_seconds: float = 300, activation_block_hash: str | None = None,
) -> PrivateModelSubmission:
    """Read encrypted PUBLIC mailbox; send only parts/manifest to PRIVATE R2."""
    parsed = urlparse(mailbox_url)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("mailbox requires the existing public HTTPS object prefix")
    if getattr(hotkey, "crypto_type", None) != 0:
        raise ValueError("private upload requires an existing Ed25519 hotkey")
    manifest = build_chunk_manifest(model)
    upload_id = upload_registration_id(genesis_hash, hotkey.ss58_address)
    url = f"{mailbox_url.rstrip('/')}/{upload_id}.bin"
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = await client.get(url, headers={"Cache-Control": "no-cache"})
        if response.status_code == 200:
            envelope = decrypt_mailbox(response.content, hotkey=hotkey,
                                       validator_identity=validator_identity,
                                       genesis_hash=genesis_hash)
            if (envelope["expires_at"] > time.time() + 30
                    and (activation_block_hash is None
                         or envelope["activation_block_hash"] == activation_block_hash)):
                break
        if response.status_code not in {200, 400, 404} or time.monotonic() >= deadline:
            raise RuntimeError("private mailbox is unavailable; no model was submitted")
        await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
    s3 = s3_factory(envelope)
    prefix, bucket = envelope["prefix"], envelope["bucket"]
    try:
        for index, part in enumerate(manifest.parts):
            if time.time() + 15 >= envelope["expires_at"]:
                raise RuntimeError("private upload credential expired; reactivate the same upload")
            body = read_part(model, index, part.size)
            if len(body) != part.size or hashlib.sha256(body).hexdigest() != part.sha256:
                raise ValueError("model changed during upload")
            await asyncio.to_thread(s3.put_object, Bucket=bucket, Key=prefix + part.name,
                                    Body=body, ContentLength=part.size)
        if build_chunk_manifest(model) != manifest:
            raise ValueError("model changed before its signed manifest")
        document = {
            "format": "veda-upload-manifest-v1", "genesis_hash": genesis_hash,
            "hotkey": hotkey.ss58_address, "upload_id": upload_id,
            "manifest": json.loads(manifest.encode()),
        }
        document["signature"] = hotkey.sign(manifest_signing_bytes(document)).hex()
        raw = canonical_json(document)
        await asyncio.to_thread(s3.put_object, Bucket=bucket, Key=prefix + "manifest.json",
                                Body=raw, ContentLength=len(raw))
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError("private R2 transfer unavailable; retry the same registration") from exc
    finally:
        s3.close()
    return PrivateModelSubmission(upload_id, manifest.sha256, hashlib.sha256(raw).hexdigest())
