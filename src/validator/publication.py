"""Canonical public artifacts and Cloudflare R2 transport."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from scoring.spec import TrackSpec
from validator.store import ChallengeStore
from validator.weights import WeightArtifact, WeightChainState, WeightPlan, plan_weights

_HEX_32 = re.compile(r"^[0-9a-f]{64}$")
_R2_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_R2_ACCOUNT_ID = re.compile(r"^[0-9a-fA-F]{32}$")
_R2_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_MAX_WEIGHT_ARTIFACT_BYTES = 1024 * 1024


class PublicArtifactError(RuntimeError):
    """Raised when a public artifact or its transport is invalid."""


@dataclass(frozen=True, slots=True)
class PublicArtifacts:
    source_block: int
    weights: bytes
    results: bytes


def build_public_artifacts(
    store: ChallengeStore,
    track: TrackSpec,
    chain: WeightChainState,
    *,
    mechid: int,
    version_key: int,
) -> PublicArtifacts | None:
    """Build one auditable public snapshot when chain scanning is caught up."""

    cursor = store.chain_cursor()
    if cursor is None or cursor[0] != chain.finalized_block:
        return None
    snapshot, reward_entries = store.public_snapshot_inputs()
    plan = plan_weights(
        reward_entries,
        uid_by_hotkey=chain.uid_by_hotkey,
        current_block=chain.finalized_block,
        min_allowed_weights=chain.min_allowed_weights,
        max_weight=chain.max_weight,
        minimum_frontier_gain=track.scoring.minimum_frontier_gain,
    )
    artifact = WeightArtifact(
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        mechid=mechid,
        version_key=version_key,
        source_block=chain.finalized_block,
        plan=plan,
    )
    common = {
        "track_id": track.track_id,
        "track_spec_digest": track.digest,
        "netuid": track.netuid,
        "source_block": chain.finalized_block,
    }
    results = {
        **common,
        "leaderboard": snapshot["leaderboard"],
        "rewards": snapshot["rewards"],
        "challenges": snapshot["challenges"],
    }
    return PublicArtifacts(
        source_block=chain.finalized_block,
        weights=encode_weight_artifact(artifact),
        results=_json_bytes(results),
    )


def encode_weight_artifact(artifact: WeightArtifact) -> bytes:
    weights = None
    if artifact.plan is not None:
        weights = [
            {"hotkey": hotkey, "uid": uid, "weight": weight}
            for hotkey, uid, weight in zip(
                artifact.plan.hotkeys,
                artifact.plan.uids,
                artifact.plan.weights,
                strict=True,
            )
        ]
    return _json_bytes(
        {
            "track_id": artifact.track_id,
            "track_spec_digest": artifact.track_spec_digest,
            "netuid": artifact.netuid,
            "mechid": artifact.mechid,
            "version_key": artifact.version_key,
            "source_block": artifact.source_block,
            "weights": weights,
        }
    )


def decode_weight_artifact(payload: bytes) -> WeightArtifact:
    """Strictly decode the small untrusted artifact consumed by Validators."""

    if not payload or len(payload) > _MAX_WEIGHT_ARTIFACT_BYTES:
        raise PublicArtifactError("weight artifact size is invalid")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicArtifactError("weight artifact is not valid JSON") from exc
    root = _object(
        raw,
        {
            "track_id",
            "track_spec_digest",
            "netuid",
            "mechid",
            "version_key",
            "source_block",
            "weights",
        },
        "weight artifact",
    )
    track_id = _string(root["track_id"], "track_id")
    digest = _string(root["track_spec_digest"], "track_spec_digest").lower()
    if not _HEX_32.fullmatch(digest):
        raise PublicArtifactError("track_spec_digest must be 32-byte lowercase hexadecimal")
    netuid = _integer(root["netuid"], "netuid", maximum=0xFFFF)
    mechid = _integer(root["mechid"], "mechid")
    version_key = _integer(root["version_key"], "version_key")
    source_block = _integer(root["source_block"], "source_block")
    items = root["weights"]
    plan = None
    if items is not None:
        if not isinstance(items, list) or not items:
            raise PublicArtifactError("weights must be null or a non-empty list")
        hotkeys: list[str] = []
        uids: list[int] = []
        weights: list[float] = []
        for value in items:
            item = _object(value, {"hotkey", "uid", "weight"}, "weight entry")
            hotkeys.append(_string(item["hotkey"], "weight hotkey"))
            uids.append(_integer(item["uid"], "weight uid", maximum=0xFFFF))
            weight = item["weight"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise PublicArtifactError("weight must be numeric")
            weights.append(float(weight))
        try:
            plan = WeightPlan(source_block, tuple(hotkeys), tuple(uids), tuple(weights))
        except ValueError as exc:
            raise PublicArtifactError(str(exc)) from exc
    try:
        return WeightArtifact(
            track_id,
            digest,
            netuid,
            mechid,
            version_key,
            source_block,
            plan,
        )
    except ValueError as exc:
        raise PublicArtifactError(str(exc)) from exc


async def fetch_weight_artifact(
    url: str,
    *,
    client: httpx.AsyncClient,
    max_bytes: int = _MAX_WEIGHT_ARTIFACT_BYTES,
) -> WeightArtifact:
    """Fetch one public R2 weight artifact with a hard response-size bound."""

    validate_public_weights_url(url)
    if max_bytes <= 0 or max_bytes > _MAX_WEIGHT_ARTIFACT_BYTES:
        raise ValueError("weight artifact byte limit is invalid")
    received = bytearray()
    try:
        async with client.stream(
            "GET", url, headers={"Accept": "application/json", "Cache-Control": "no-cache"}
        ) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as exc:
                    raise PublicArtifactError("weight artifact content length is invalid") from exc
                if declared < 0 or declared > max_bytes:
                    raise PublicArtifactError("weight artifact exceeds the byte limit")
            async for chunk in response.aiter_bytes():
                received.extend(chunk)
                if len(received) > max_bytes:
                    raise PublicArtifactError("weight artifact exceeds the byte limit")
    except httpx.HTTPError as exc:
        raise PublicArtifactError("public weight artifact could not be fetched") from exc
    return decode_weight_artifact(bytes(received))


def validate_public_weights_url(url: str) -> str:
    """Validate the immutable Validator-side location configured by an operator."""

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("public weight URL must be an HTTPS URL without credentials or fragments")
    return url


class R2Publisher:
    """Publish results first and the consumable weight artifact last."""

    def __init__(self, client: Any, *, bucket: str, prefix: str) -> None:
        if not _R2_NAME.fullmatch(bucket) or len(bucket) > 64:
            raise ValueError("R2 bucket name is invalid")
        normalized = prefix.strip("/")
        if (
            not _R2_PREFIX.fullmatch(normalized)
            or len(normalized) > 512
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        ):
            raise ValueError("R2 object prefix is invalid")
        self._client = client
        self._bucket = bucket
        self._prefix = normalized

    @classmethod
    def from_environment(cls, *, account_id: str, bucket: str, prefix: str) -> R2Publisher:
        if not _R2_ACCOUNT_ID.fullmatch(account_id):
            raise ValueError("R2 account ID is invalid")
        access_key = os.environ.get("VEDA_R2_ACCESS_KEY_ID")
        secret_key = os.environ.get("VEDA_R2_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            raise ValueError("VEDA_R2_ACCESS_KEY_ID and VEDA_R2_SECRET_ACCESS_KEY are required")
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("R2 publishing requires the 'r2' package extra") from exc
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        return cls(client, bucket=bucket, prefix=prefix)

    async def publish(self, artifacts: PublicArtifacts) -> None:
        await asyncio.to_thread(self._put, "results.json", artifacts.results)
        await asyncio.to_thread(self._put, "weights.json", artifacts.weights)

    def _put(self, name: str, payload: bytes) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=f"{self._prefix}/{name}",
            Body=payload,
            ContentType="application/json; charset=utf-8",
            CacheControl="no-store, max-age=0",
        )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicArtifactError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicArtifactError(f"{name} fields are invalid")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicArtifactError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise PublicArtifactError(f"{name} is invalid")
    return value
