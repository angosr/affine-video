"""Signed metadata calls and fixed-part PUTs; never submit a chain transaction."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bittensor as bt
import httpx

from miner.submit import PrivateModelSubmission, UploadIntent
from validator.chunks import build_chunk_manifest, read_part


async def upload_model(
    model: str | Path, *, audience: str, network: str, wallet_name: str,
    hotkey_name: str, wallet_path: str | None = None,
) -> PrivateModelSubmission:
    """Retrying this command uses the same owner/file/domain ID, not a new model.

    The API only seals the declaration. Original ADMISSION still verifies the
    real full SHA/schema/owner and persists the model before any scoring.
    """
    if not network:
        raise ValueError("upload network cannot be empty")
    UploadIntent(audience, "00" * 32, "configuration", 120, "upload", "00" * 32,
                 "00" * 32, 1, "00" * 32, 1)
    arguments = {"name": wallet_name, "hotkey": hotkey_name}
    if wallet_path is not None:
        arguments["path"] = wallet_path
    wallet = bt.Wallet(**arguments)
    async with bt.Client(network) as chain:
        genesis = (await chain.block_info(0)).hash
    if not isinstance(genesis, str):
        raise RuntimeError("upload network genesis is unavailable")
    async with httpx.AsyncClient(follow_redirects=False, timeout=60.0) as client:
        return await upload_file(
            model, audience=audience, genesis_hash=genesis.removeprefix("0x"),
            hotkey=wallet.hotkey, client=client,
        )


async def upload_file(
    model: str | Path, *, audience: str, genesis_hash: str, hotkey: Any,
    client: httpx.AsyncClient,
) -> PrivateModelSubmission:
    """Send original 40MiB parts, never a full file through the metadata API."""
    path = Path(model)
    manifest = build_chunk_manifest(path)
    owner = hotkey.ss58_address
    manifest_sha = hashlib.sha256(manifest.encode()).hexdigest()
    upload_id = hashlib.sha256(json.dumps(
        ["veda-upload-v1", audience, genesis_hash, owner, manifest.sha256, manifest_sha],
        separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    reference = PrivateModelSubmission(upload_id, manifest.sha256)

    async def request(action: str) -> dict[str, Any]:
        intent = UploadIntent(
            audience, genesis_hash, owner, 120, action, upload_id, manifest.sha256,
            manifest.size, manifest_sha, int(time.time()) + 300,
        )
        envelope = {"intent": asdict(intent),
                    "signature": hotkey.sign(intent.signing_message()).hex()}
        if action == "upload":
            envelope["manifest"] = json.loads(manifest.encode())
        response = await client.post(audience, json=envelope)
        if response.status_code != 200:
            raise RuntimeError("private upload metadata request was not accepted")
        result = response.json()
        if not isinstance(result, dict) or result.get("upload_id") != upload_id:
            raise ValueError("private upload response has a different identity")
        return result

    try:
        grant = await request("upload")
        for index, part in enumerate(manifest.parts):
            if type(grant.get("expires_at")) is not int:
                raise ValueError("private upload response omitted its expiry")
            if grant["expires_at"] <= int(time.time()) + 30:
                grant = await request("upload")
            parts = grant.get("parts")
            if not isinstance(parts, list) or len(parts) != len(manifest.parts):
                raise ValueError("private upload response changed its part manifest")
            capability = parts[index]
            if (not isinstance(capability, dict) or set(capability) != {
                "name", "size_bytes", "url",
            } or capability["name"] != part.name or capability["size_bytes"] != part.size
                    or not isinstance(capability["url"], str)):
                raise ValueError("private upload response changed its part identity")
            url = urlparse(capability["url"])
            if (url.scheme != "https" or not url.netloc or url.username is not None
                    or url.password is not None or url.fragment):
                raise ValueError("private upload capability requires HTTPS")
            content = read_part(path, index, part.size)
            if hashlib.sha256(content).hexdigest() != part.sha256:
                raise ValueError("local model changed after its upload identity was fixed")
            response = await client.put(
                capability["url"], content=content,
                headers={"Content-Length": str(part.size)},
            )
            if not 200 <= response.status_code < 300:
                raise RuntimeError("private upload part failed; retry the same file")
            del response, content
        sealed = await request("seal")
        if (sealed.get("file_sha256") != reference.file_sha256
                or sealed.get("sealed") is not True or sealed.get("accepted") is not False):
            raise ValueError("private upload seal response has a different identity")
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        # Do not expose presigned bearer URLs, response bodies or signatures.
        raise RuntimeError("private upload transport failed; retry the same file") from exc
    return reference
