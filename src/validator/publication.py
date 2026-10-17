"""Canonical public artifacts and S3-compatible object-storage transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from scoring.spec import TrackSpec, parse_track_spec, track_spec_dict
from validator.chunks import ChunkManifest, build_chunk_manifest, chunk_object_name, read_part
from validator.production.batches import BatchDisclosure
from validator.production.commitment import verify_source_proof
from validator.production.plan import (
    BatchPlan,
    build_plan,
    validate_source_selection_sequence,
)
from validator.queue import TERMINAL_STATES, ChallengeState
from validator.s3 import client_config
from validator.store import ChallengeStore
from validator.system_matches import system_match_artifacts, system_match_summaries
from validator.tournament import (
    Champion,
    MatchRecord,
    Tournament,
    challenge_from_match,
    genesis_champion,
    resolve_tournament,
    validate_match_document,
)
from validator.weights import (
    WeightArtifact,
    WeightChainState,
    WeightPlan,
    plan_champion_weight,
)

_HEX_32 = re.compile(r"^[0-9a-f]{64}$")
_STORAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STORAGE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_MAX_WEIGHT_ARTIFACT_BYTES = 1024 * 1024
_MAX_PUBLIC_OBJECT_BYTES = 1024 * 1024
_SUMMARY_LEADERBOARD_LIMIT = 100
_LEGACY_UNFENCED_WEIGHT_TRACK = (
    "veda-sn120-v1",
    "0fcae991aef29f87b97fbb07cf79c49e5eb9cc2a5d9aecdd2a566443e2b81dd5",
)


class PublicArtifactError(RuntimeError):
    """Raised when a public artifact or its transport is invalid."""


@dataclass(frozen=True, slots=True)
class PublicArtifacts:
    source_block: int
    track: bytes
    summary: bytes
    weights: bytes
    matches: tuple[tuple[str, bytes], ...]
    status: bytes | None = None
    system_matches: tuple[tuple[str, bytes], ...] = ()
    files: tuple[PublicFile, ...] = ()
    runtime_manifest: tuple[str, bytes] | None = None
    abort: tuple[str, bytes] | None = None
    batch_groups: tuple[tuple[str, bytes], ...] = ()


@dataclass(frozen=True, slots=True)
class PublicFile:
    object_name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicLedger:
    source_block: int | None
    matches: tuple[MatchRecord, ...]
    aborted: bool = False
    disclosed_plan_digests: frozenset[str] = frozenset()

    @property
    def required_finalized_block(self) -> int:
        committed = 0 if self.source_block is None else self.source_block
        decided = max(
            (int(match.document["decided_block"]) for match in self.matches),
            default=0,
        )
        return max(committed, decided)


@dataclass(frozen=True, slots=True)
class PublicTrackIdentity:
    """The small authenticated track boundary needed by a weight setter."""

    track_id: str
    track_spec_digest: str
    netuid: int
    mechid: int


def build_public_artifacts(
    store: ChallengeStore,
    track: TrackSpec,
    chain: WeightChainState,
    *,
    version_key: int,
    batch_disclosure_resolver: Callable[[str], BatchDisclosure] | None = None,
    disclosed_plan_digests: frozenset[str] = frozenset(),
    model_resolver: Callable[[str, int], Path] | None = None,
    runtime_manifest: bytes | None = None,
    track_aborted: bool = False,
) -> PublicArtifacts | None:
    """Build one auditable public snapshot when chain scanning is caught up."""

    cursor = store.chain_cursor()
    if cursor is None or cursor[0] != chain.finalized_block:
        return None
    snapshot = store.public_snapshot()
    unfinished = [
        challenge
        for challenge in snapshot["challenges"]
        if challenge["state"] not in {state.value for state in TERMINAL_STATES}
    ]
    aborted = track_aborted or (
        chain.finalized_block > track.track_close_block and bool(unfinished)
    )
    initial = genesis_champion(track)
    tournament = resolve_tournament(
        snapshot["challenges"],
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        minimum_gain=track.full_video.minimum_gain,
        minimum_exact_gain=track.scoring.minimum_frontier_gain,
        initial_champion=initial,
        minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
    )
    if any(match.document["decided_block"] > chain.finalized_block for match in tournament.matches):
        raise PublicArtifactError("a terminal decision is newer than the finalized chain snapshot")
    plan = None
    if not aborted:
        plan = plan_champion_weight(
            tournament.champion,
            uid_by_hotkey=chain.uid_by_hotkey,
            current_block=chain.finalized_block,
            min_allowed_weights=chain.min_allowed_weights,
            max_weight=chain.max_weight,
        )
        if plan is None and tournament.champion.challenge_id != initial.challenge_id:
            plan = plan_champion_weight(
                initial,
                uid_by_hotkey=chain.uid_by_hotkey,
                current_block=chain.finalized_block,
                min_allowed_weights=chain.min_allowed_weights,
                max_weight=chain.max_weight,
            )
    artifact = WeightArtifact(
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        mechid=track.mechid,
        version_key=version_key,
        source_block=chain.finalized_block,
        plan=plan,
        track_aborted=aborted,
    )
    summary = _summary_document(track, chain.finalized_block, tournament)
    status = _status_document(
        track,
        chain.finalized_block,
        tournament,
        store.active_competitors(chain.uid_by_hotkey),
        snapshot["challenges"],
        track_aborted=aborted,
    )
    files: dict[str, PublicFile] = {}
    batch_groups: dict[str, bytes] = {}
    terminal_ids = {
        challenge["challenge_id"]
        for challenge in snapshot["challenges"]
        if challenge["state"] in {state.value for state in TERMINAL_STATES}
    }
    disclosure: dict[str, bool] = {}
    for match in tournament.matches:
        evaluation = match.document["evaluation"]
        batch = evaluation["evaluation_batch"]
        stage_batches = [
            stage["evaluation_batch"]
            for stage in evaluation["stages"]
            if "evaluation_batch" in stage
        ]
        if batch is not None:
            stage_batches.append(batch)
        for used_batch in stage_batches:
            plan_digest = used_batch["plan_digest"]
            if plan_digest not in disclosure:
                if plan_digest in disclosed_plan_digests:
                    disclosure[plan_digest] = True
                else:
                    resolved = (
                        None
                        if batch_disclosure_resolver is None
                        else batch_disclosure_resolver(plan_digest)
                    )
                    disclosure[plan_digest] = bool(resolved) and set(
                        resolved.challenge_ids
                    ).issubset(terminal_ids)
                    if disclosure[plan_digest]:
                        assert resolved is not None
                        batch_groups[resolved.record_name] = resolved.record_payload
                        for item in resolved.files:
                            files[item.object_name] = PublicFile(
                                item.object_name,
                                item.path,
                                item.sha256,
                            )
            if (
                disclosure[plan_digest]
                and plan_digest not in disclosed_plan_digests
                and used_batch["object"] not in files
            ):
                raise PublicArtifactError(
                    "batch disclosure omitted its exact evaluation object"
                )
        object_name = evaluation["model_object"]
        if object_name is not None:
            path_value = store.model_path(match.challenge_id)
            path = None if path_value is None else Path(path_value)
            if path is None or not path.is_file():
                if model_resolver is None:
                    raise PublicArtifactError("scored model has no durable archive source")
                path = model_resolver(evaluation["file_sha256"], evaluation["model_size_bytes"])
            files[object_name] = PublicFile(
                object_name,
                path,
                evaluation["file_sha256"],
            )
    return PublicArtifacts(
        source_block=chain.finalized_block,
        track=_json_bytes(
            {
                "track_spec_digest": track.digest,
                "spec": track_spec_dict(track),
            }
        ),
        summary=_json_bytes(summary),
        weights=encode_weight_artifact(artifact),
        matches=tuple(
            (match.object_name, _json_bytes(match.document)) for match in tournament.matches
        ),
        status=_json_bytes(status),
        system_matches=system_match_artifacts(track.track_id, track.digest, track.netuid),
        files=tuple(files.values()),
        runtime_manifest=_runtime_manifest(track, runtime_manifest),
        abort=_track_abort_artifact(track) if aborted else None,
        batch_groups=tuple(sorted(batch_groups.items())),
    )


def build_genesis_artifacts(
    track: TrackSpec,
    chain: WeightChainState,
    *,
    version_key: int,
    runtime_manifest: bytes | None = None,
) -> PublicArtifacts:
    """Build the complete pre-launch snapshot with UID0 as champion."""

    tournament = resolve_tournament(
        (),
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        minimum_gain=track.full_video.minimum_gain,
        minimum_exact_gain=track.scoring.minimum_frontier_gain,
        initial_champion=genesis_champion(track),
        minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
    )
    plan = plan_champion_weight(
        tournament.champion,
        uid_by_hotkey=chain.uid_by_hotkey,
        current_block=chain.finalized_block,
        min_allowed_weights=chain.min_allowed_weights,
        max_weight=chain.max_weight,
    )
    if plan is None:
        raise PublicArtifactError("UID0 cannot receive the complete chain weight")
    weights = WeightArtifact(
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        mechid=track.mechid,
        version_key=version_key,
        source_block=chain.finalized_block,
        plan=plan,
    )
    return PublicArtifacts(
        source_block=chain.finalized_block,
        track=_json_bytes({"track_spec_digest": track.digest, "spec": track_spec_dict(track)}),
        summary=_json_bytes(_summary_document(track, chain.finalized_block, tournament)),
        weights=encode_weight_artifact(weights),
        matches=(),
        status=_json_bytes(_status_document(track, chain.finalized_block, tournament, [], [])),
        system_matches=system_match_artifacts(track.track_id, track.digest, track.netuid),
        runtime_manifest=_runtime_manifest(track, runtime_manifest),
    )


def _runtime_manifest(track: TrackSpec, payload: bytes | None) -> tuple[str, bytes] | None:
    if payload is None:
        return None
    if hashlib.sha256(payload).hexdigest() != track.artifact_manifest_sha256:
        raise PublicArtifactError("runtime artifact manifest digest is incorrect")
    return f"runtime/{track.artifact_manifest_sha256}/manifest.json", payload


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
            "track_aborted": artifact.track_aborted,
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
    if isinstance(raw, dict) and "track_aborted" not in raw and (
        raw.get("track_id"), raw.get("track_spec_digest")
    ) == _LEGACY_UNFENCED_WEIGHT_TRACK:
        raw["track_aborted"] = False
    root = _object(
        raw,
        {
            "track_id",
            "track_spec_digest",
            "netuid",
            "mechid",
            "version_key",
            "source_block",
            "track_aborted",
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
    track_aborted = root["track_aborted"]
    if not isinstance(track_aborted, bool):
        raise PublicArtifactError("track_aborted must be boolean")
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
            track_aborted,
        )
    except ValueError as exc:
        raise PublicArtifactError(str(exc)) from exc


async def fetch_weight_artifact(
    url: str,
    *,
    client: httpx.AsyncClient,
    max_bytes: int = _MAX_WEIGHT_ARTIFACT_BYTES,
) -> WeightArtifact:
    """Fetch one public weight artifact with a hard response-size bound."""

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


async def fetch_track_spec(
    url: str,
    *,
    client: httpx.AsyncClient,
) -> TrackSpec:
    """Fetch and verify the current track document from public storage."""

    document = await _fetch_track_document(url, client=client)
    try:
        track = parse_track_spec(document["spec"])
    except ValueError as exc:
        raise PublicArtifactError("public track specification is invalid") from exc
    if document["track_spec_digest"] != track.digest:
        raise PublicArtifactError("public track digest is invalid")
    return track


async def fetch_track_identity(
    url: str,
    *,
    client: httpx.AsyncClient,
) -> PublicTrackIdentity:
    """Fetch the version-independent track identity needed for chain weights."""

    document = await _fetch_track_document(url, client=client)
    spec = document["spec"]
    try:
        track_id = _string(spec["track_id"], "track_id")
        netuid = _integer(spec["netuid"], "netuid", maximum=0xFFFF)
        mechid = _integer(spec["mechid"], "mechid")
    except (KeyError, TypeError, PublicArtifactError) as exc:
        raise PublicArtifactError("public track identity is invalid") from exc
    return PublicTrackIdentity(
        track_id,
        document["track_spec_digest"],
        netuid,
        mechid,
    )


async def _fetch_track_document(
    url: str,
    *,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    validate_public_weights_url(url)
    received = bytearray()
    try:
        async with client.stream(
            "GET", url, headers={"Accept": "application/json", "Cache-Control": "no-cache"}
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                received.extend(chunk)
                if len(received) > _MAX_PUBLIC_OBJECT_BYTES:
                    raise PublicArtifactError("track artifact exceeds the byte limit")
    except httpx.HTTPError as exc:
        raise PublicArtifactError("track artifact could not be fetched") from exc
    document = _decode_canonical_object(bytes(received), "track")
    if set(document) != {"track_spec_digest", "spec"}:
        raise PublicArtifactError("public track fields are invalid")
    digest = document["track_spec_digest"]
    spec = document["spec"]
    if (
        not isinstance(digest, str)
        or _HEX_32.fullmatch(digest) is None
        or not isinstance(spec, dict)
        or hashlib.sha256(_json_bytes(spec)).hexdigest() != digest
    ):
        raise PublicArtifactError("public track digest is invalid")
    return document


def validate_public_weights_url(url: str) -> str:
    """Validate the public Validator-side weight URL configured by an operator."""

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


class S3Repository:
    """Persist versioned match facts and publish bounded current-state pointers."""

    def __init__(self, client: Any, *, bucket: str, prefix: str) -> None:
        if not _STORAGE_NAME.fullmatch(bucket) or len(bucket) > 64:
            raise ValueError("storage bucket name is invalid")
        normalized = prefix.strip("/")
        if (
            not _STORAGE_PREFIX.fullmatch(normalized)
            or len(normalized) > 512
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        ):
            raise ValueError("storage object prefix is invalid")
        self._client = client
        self._bucket = bucket
        self._prefix = normalized
        self._confirmed_immutable: dict[str, str] = {}

    @classmethod
    def from_environment(cls, *, bucket: str, prefix: str) -> S3Repository:
        endpoint = os.environ.get("STORAGE_URL")
        access_key = os.environ.get("ACCESS_KEY")
        secret_key = os.environ.get("SECRET_ACCESS_KEY")
        if not endpoint or not access_key or not secret_key:
            raise ValueError("STORAGE_URL, ACCESS_KEY, and SECRET_ACCESS_KEY are required")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("STORAGE_URL must be an HTTPS endpoint without credentials or query")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("publishing requires the 'storage' package extra") from exc
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=client_config(),
        )
        return cls(client, bucket=bucket, prefix=prefix)

    def verify_conditional_writes(self) -> None:
        """Refuse a public backend that cannot fence concurrent publishers."""

        name = f"capability-probes/{uuid.uuid4().hex}.json"
        key = f"{self._prefix}/{name}"
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=b'{"state":"first"}',
                IfNoneMatch="*",
            )
            first = self._get_stored_optional(name)
            if first is None:
                raise PublicArtifactError("conditional create was not durable")
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=b'{"state":"overwrite"}',
                    IfNoneMatch="*",
                )
            except Exception as exc:
                if not _is_precondition_failed(exc):
                    raise
            else:
                raise PublicArtifactError("public object store ignores If-None-Match")
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=b'{"state":"stale"}',
                    IfMatch="invalid-etag",
                )
            except Exception as exc:
                if not _is_precondition_failed(exc):
                    raise
            else:
                raise PublicArtifactError("public object store ignores If-Match")
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=b'{"state":"ready"}',
                IfMatch=first[1],
            )
            stored = self._get_stored_optional(name)
            if stored is None or stored[0] != b'{"state":"ready"}':
                raise PublicArtifactError("conditional replacement was not durable")
        finally:
            self._client.delete_object(Bucket=self._bucket, Key=key)

    async def publish(self, artifacts: PublicArtifacts) -> None:
        track_document = _decode_canonical_object(artifacts.track, "track")
        try:
            track_id = track_document["spec"]["track_id"]
        except (KeyError, TypeError) as exc:
            raise PublicArtifactError("public track pointer is invalid") from exc
        if not isinstance(track_id, str) or not track_id:
            raise PublicArtifactError("public track pointer is invalid")
        weight_fence = decode_weight_artifact(artifacts.weights)
        if weight_fence.track_id != track_id:
            raise PublicArtifactError("public weights target a different track")
        abort_name = f"events/{track_id}/track-aborted.json"
        if artifacts.abort is None:
            if weight_fence.track_aborted:
                raise PublicArtifactError("track-aborted weights require the immutable event")
            existing_abort = await asyncio.to_thread(self._get_optional, abort_name)
            if existing_abort is not None:
                raise PublicArtifactError("an aborted public track cannot resume")
        elif artifacts.abort[0] != abort_name or not weight_fence.track_aborted:
            raise PublicArtifactError("track abort requires its null weight fence")
        if artifacts.runtime_manifest is not None:
            await asyncio.to_thread(self._put_immutable, *artifacts.runtime_manifest)
        await asyncio.to_thread(self._put_mutable, "track.json", artifacts.track)
        for artifact in artifacts.files:
            await asyncio.to_thread(self._put_file_immutable, artifact)
        for name, payload in artifacts.system_matches:
            await asyncio.to_thread(self._put_immutable, name, payload)
        for name, payload in artifacts.matches:
            await asyncio.to_thread(self._put_immutable, name, payload)
        for name, payload in artifacts.batch_groups:
            await asyncio.to_thread(self._put_immutable, name, payload)
        await asyncio.to_thread(self._put_mutable, "weights.json", artifacts.weights)
        if artifacts.abort is not None:
            await asyncio.to_thread(self._put_immutable, *artifacts.abort)
        await asyncio.to_thread(self._put_mutable, "summary.json", artifacts.summary)
        if artifacts.status is not None:
            await asyncio.to_thread(self._put_mutable, "status.json", artifacts.status)

    def _put_file_immutable(self, artifact: PublicFile) -> None:
        if self._confirmed_immutable.get(artifact.object_name) == artifact.sha256:
            return
        existing = self._get_optional(artifact.object_name)
        if existing is not None:
            try:
                manifest = ChunkManifest.decode(existing)
            except ValueError as exc:
                raise PublicArtifactError("immutable public file manifest is invalid") from exc
            if manifest.sha256 != artifact.sha256:
                raise PublicArtifactError("immutable public file conflicts with committed digest")
            self._verify_public_chunks(artifact.object_name, manifest)
            self._confirmed_immutable[artifact.object_name] = artifact.sha256
            return

        if not artifact.path.is_file():
            raise PublicArtifactError(f"public file source is missing: {artifact.object_name}")
        manifest = build_chunk_manifest(artifact.path)
        if manifest.sha256 != artifact.sha256:
            raise PublicArtifactError(f"public file digest changed: {artifact.object_name}")
        for index, part in enumerate(manifest.parts):
            name = chunk_object_name(artifact.object_name, part.name)
            key = f"{self._prefix}/{name}"
            try:
                uploaded = self._client.head_object(Bucket=self._bucket, Key=key)
            except Exception as exc:
                response = getattr(exc, "response", None)
                code = (response or {}).get("Error", {}).get("Code")
                if str(code) not in {"404", "NoSuchKey", "NotFound"}:
                    raise
            else:
                if (
                    uploaded.get("Metadata", {}).get("sha256") != part.sha256
                    or uploaded.get("ContentLength") != part.size
                ):
                    raise PublicArtifactError("immutable public chunk conflicts with content")
                continue
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=read_part(artifact.path, index, part.size),
                ContentType="application/octet-stream",
                CacheControl="public, max-age=31536000, immutable",
                Metadata={"sha256": part.sha256},
                IfNoneMatch="*",
            )
            uploaded = self._client.head_object(Bucket=self._bucket, Key=key)
            if (
                uploaded.get("Metadata", {}).get("sha256") != part.sha256
                or uploaded.get("ContentLength") != part.size
            ):
                raise PublicArtifactError("public chunk upload verification failed")
        self._put(artifact.object_name, manifest.encode(), True, if_none_match="*")
        self._confirmed_immutable[artifact.object_name] = artifact.sha256

    def _verify_public_chunks(self, manifest_name: str, manifest: ChunkManifest) -> None:
        for part in manifest.parts:
            name = chunk_object_name(manifest_name, part.name)
            try:
                uploaded = self._client.head_object(
                    Bucket=self._bucket,
                    Key=f"{self._prefix}/{name}",
                )
            except Exception as exc:
                response = getattr(exc, "response", None)
                code = (response or {}).get("Error", {}).get("Code")
                if str(code) in {"404", "NoSuchKey", "NotFound"}:
                    raise PublicArtifactError(f"immutable public chunk is missing: {name}") from exc
                raise
            if (
                uploaded.get("Metadata", {}).get("sha256") != part.sha256
                or uploaded.get("ContentLength") != part.size
            ):
                raise PublicArtifactError("immutable public chunk conflicts with manifest")

    async def load(self, track: TrackSpec) -> PublicLedger:
        """Load and verify the committed public tournament ledger from object storage."""

        return await asyncio.to_thread(self._load, track)

    def _load(self, track: TrackSpec) -> PublicLedger:
        track_payload = self._get_optional("track.json")
        summary_payload = self._get_optional("summary.json")
        if track_payload is None:
            if summary_payload is not None:
                raise PublicArtifactError("public summary exists without its immutable track")
            return PublicLedger(None, ())

        track_document = _decode_canonical_object(track_payload, "track")
        if set(track_document) != {"track_spec_digest", "spec"}:
            raise PublicArtifactError("public track fields are invalid")
        try:
            stored_track = parse_track_spec(track_document["spec"])
        except ValueError as exc:
            raise PublicArtifactError("public track specification is invalid") from exc
        if track_document["track_spec_digest"] != stored_track.digest:
            raise PublicArtifactError("public track digest is invalid")
        if stored_track != track:
            if stored_track.track_id == track.track_id:
                raise PublicArtifactError("an active track ID cannot change its specification")
            return PublicLedger(None, ())

        aborted = self._load_track_abort(track)

        if summary_payload is None:
            matches = tuple(
                self._load_following_matches(track, 1, genesis_champion(track))
            )
            self._verify_model_archives(matches)
            return PublicLedger(
                None,
                matches,
                aborted,
                self._load_batch_disclosures(track, matches),
            )
        summary = _decode_canonical_object(summary_payload, "summary")
        expected_fields = {
            "track_id",
            "track_spec_digest",
            "netuid",
            "source_block",
            "match_count",
            "latest_match",
            "champion",
            "leaderboard",
            "leaderboard_total",
        }
        if set(summary) != expected_fields:
            raise PublicArtifactError("public summary fields are invalid")
        source_block = _integer(summary["source_block"], "source_block")
        match_count = _integer(summary["match_count"], "match_count")
        matches: list[MatchRecord] = []
        committed_challenges: list[dict[str, Any]] = []
        incumbent = genesis_champion(track)
        for sequence in range(1, match_count + 1):
            name = f"matches/{track.track_id}/{sequence:012d}.json"
            payload = self._get_optional(name)
            if payload is None:
                raise PublicArtifactError(f"committed match is missing: {name}")
            document = _decode_canonical_object(payload, name)
            try:
                record = validate_match_document(
                    document,
                    sequence=sequence,
                    track_id=track.track_id,
                    track_spec_digest=track.digest,
                    netuid=track.netuid,
                    minimum_gain=track.full_video.minimum_gain,
                    minimum_exact_gain=track.scoring.minimum_frontier_gain,
                    incumbent=incumbent,
                    minimum_generation_delay_blocks=(track.cache.minimum_generation_delay_blocks),
                )
            except ValueError as exc:
                raise PublicArtifactError(f"public match is invalid: {name}") from exc
            matches.append(record)
            committed_challenges.append(challenge_from_match(document))
            if document["decided_block"] > source_block:
                raise PublicArtifactError(f"public match is newer than its summary: {name}")
            incumbent = record.champion_after
            self._confirmed_immutable[name] = hashlib.sha256(payload).hexdigest()

        tournament = resolve_tournament(
            committed_challenges,
            track_id=track.track_id,
            track_spec_digest=track.digest,
            netuid=track.netuid,
            minimum_gain=track.full_video.minimum_gain,
            minimum_exact_gain=track.scoring.minimum_frontier_gain,
            initial_champion=genesis_champion(track),
            minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
        )
        if summary != _summary_document(track, source_block, tournament):
            raise PublicArtifactError("public summary does not match its immutable matches")

        matches.extend(self._load_following_matches(track, match_count + 1, incumbent))
        complete_matches = tuple(matches)
        self._verify_model_archives(complete_matches)
        return PublicLedger(
            source_block,
            complete_matches,
            aborted,
            self._load_batch_disclosures(track, complete_matches),
        )

    def _load_track_abort(self, track: TrackSpec) -> bool:
        name, expected = _track_abort_artifact(track)
        payload = self._get_optional(name)
        if payload is None:
            return False
        if payload != expected:
            raise PublicArtifactError("public track-abort fact is invalid")
        self._confirmed_immutable[name] = hashlib.sha256(payload).hexdigest()
        return True

    def _load_following_matches(
        self,
        track: TrackSpec,
        sequence: int,
        incumbent: Champion,
    ) -> list[MatchRecord]:
        matches: list[MatchRecord] = []
        while True:
            name = f"matches/{track.track_id}/{sequence:012d}.json"
            payload = self._get_optional(name)
            if payload is None:
                break
            document = _decode_canonical_object(payload, name)
            try:
                record = validate_match_document(
                    document,
                    sequence=sequence,
                    track_id=track.track_id,
                    track_spec_digest=track.digest,
                    netuid=track.netuid,
                    minimum_gain=track.full_video.minimum_gain,
                    minimum_exact_gain=track.scoring.minimum_frontier_gain,
                    incumbent=incumbent,
                    minimum_generation_delay_blocks=(track.cache.minimum_generation_delay_blocks),
                )
            except ValueError as exc:
                raise PublicArtifactError(f"public match is invalid: {name}") from exc
            matches.append(record)
            incumbent = record.champion_after
            self._confirmed_immutable[name] = hashlib.sha256(payload).hexdigest()
            sequence += 1
        return matches

    def _load_batch_disclosures(
        self,
        track: TrackSpec,
        matches: tuple[MatchRecord, ...],
    ) -> frozenset[str]:
        """Verify complete public batch groups once when recovering the ledger."""

        terminal_ids = {match.challenge_id for match in matches}
        batches: dict[str, list[Mapping[str, Any]]] = {}
        stage_uses: dict[str, list[tuple[MatchRecord, Mapping[str, Any]]]] = {}
        for match in matches:
            evaluation = match.document["evaluation"]
            used = []
            for stage in evaluation["stages"]:
                batch = stage.get("evaluation_batch")
                if batch is None:
                    continue
                used.append(batch)
                stage_uses.setdefault(batch["plan_digest"], []).append((match, stage))
            if evaluation["evaluation_batch"] is not None:
                used.append(evaluation["evaluation_batch"])
            for batch in used:
                batches.setdefault(batch["plan_digest"], []).append(batch)

        generation_blocks = {}
        for plan_digest, used_batches in batches.items():
            blocks = {batch["generated_block"] for batch in used_batches}
            if len(blocks) != 1:
                raise PublicArtifactError("public matches disagree on a batch generation block")
            generation_blocks[plan_digest] = blocks.pop()
        if len(set(generation_blocks.values())) != len(generation_blocks):
            raise PublicArtifactError("public matches identify multiple plans for one block")

        verified = {}
        for plan_digest, used_batches in batches.items():
            record_name = f"batch_groups/{plan_digest}.json"
            payload = self._get_optional(record_name)
            if payload is None:
                continue
            document = _decode_canonical_object(payload, record_name)
            if set(document) != {
                "format",
                "plan",
                "public_dense_references",
                "public_exact_object",
            } or document["format"] != "veda-batch-disclosure-v1":
                raise PublicArtifactError("public batch disclosure fields are invalid")
            try:
                plan = BatchPlan.decode(_json_bytes(document["plan"]))
            except (TypeError, ValueError) as exc:
                raise PublicArtifactError("public batch disclosure plan is invalid") from exc
            challenge_ids = set(plan.exact_challenge_ids + plan.confirmation_challenge_ids)
            if (
                plan.state != "READY"
                or plan.plan_digest != plan_digest
                or plan.track_spec_digest != track.digest
                or not challenge_ids
                or not challenge_ids.issubset(terminal_ids)
            ):
                raise PublicArtifactError("public batch disclosure conflicts with the ledger")
            uses = stage_uses.get(plan_digest, [])
            exact_ids = tuple(
                match.challenge_id
                for match, stage in uses
                if stage["stage"] == ChallengeState.SCORING
            )
            confirmation_ids = tuple(
                match.challenge_id
                for match, stage in uses
                if stage["stage"] == ChallengeState.CONFIRMING
            )
            if (
                exact_ids != plan.exact_challenge_ids
                or confirmation_ids != plan.confirmation_challenge_ids
            ):
                raise PublicArtifactError(
                    "public batch disclosure challenge assignment is invalid"
                )
            for match, stage in uses:
                if stage["stage"] == ChallengeState.SCORING:
                    scheduled_from = match.document["submission"]["block_height"]
                else:
                    exact_stages = [
                        item
                        for item in match.document["evaluation"]["stages"]
                        if item["stage"] == ChallengeState.SCORING
                    ]
                    if len(exact_stages) != 1:
                        raise PublicArtifactError(
                            "public confirmation is missing its exact decision"
                        )
                    scheduled_from = exact_stages[0]["score_block"]
                if not (
                    plan.submission_first_block
                    <= scheduled_from
                    <= plan.submission_last_block
                ):
                    raise PublicArtifactError(
                        "public batch disclosure uses the wrong generation boundary"
                    )
            commitment_digests = tuple(item.sha256 for item in track.source_commitments)
            if any(
                not verify_source_proof(
                    proof,
                    track.source_pool_merkle_root,
                    commitment_digests=commitment_digests,
                )
                for proof in plan.source_proofs
            ):
                raise PublicArtifactError("public batch disclosure has an invalid source proof")
            rebuilt = build_plan(
                track_spec_digest=track.digest,
                generation_block=plan.generation_block,
                block_hash=plan.block_hash,
                track_start_block=track.track_start_block,
                minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
                generation_interval_blocks=track.cache.generation_interval_blocks,
                exact_challenge_ids=plan.exact_challenge_ids,
                confirmation_challenge_ids=plan.confirmation_challenge_ids,
                source_proofs=plan.source_proofs,
                scoring=track.scoring,
                available_routes=track.model.selector.routes,
            )
            if rebuilt.planned_document() != plan.planned_document():
                raise PublicArtifactError(
                    "public batch disclosure conflicts with the deterministic schedule"
                )
            assert plan.exact is not None
            exact_name = f"batches/{plan.exact.sha256}/manifest.json"
            if any(
                batch["sha256"] != plan.exact.sha256
                or batch["object"] != exact_name
                or batch["generated_block"] != plan.generation_block
                or batch["seed_block_hash"] != plan.block_hash
                for batch in used_batches
            ):
                raise PublicArtifactError("public match conflicts with its batch disclosure")

            public_references = []
            files = [(exact_name, plan.exact.sha256, plan.exact.size_bytes)]
            for reference in plan.dense_references:
                public = {"rollout_id": reference.rollout_id}
                for kind, item in (
                    ("video", reference.video),
                    ("latent", reference.latent),
                ):
                    object_name = f"batch_assets/{item.sha256}/manifest.json"
                    public[kind] = {
                        "object": object_name,
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                    }
                    files.append((object_name, item.sha256, item.size_bytes))
                public_references.append(public)
            expected = {
                "format": "veda-batch-disclosure-v1",
                "plan": plan.document(),
                "public_exact_object": exact_name,
                "public_dense_references": public_references,
            }
            if document != expected:
                raise PublicArtifactError("public batch disclosure is not canonical")
            for object_name, sha256, size_bytes in files:
                self._verify_public_file(object_name, sha256, size_bytes)
            verified[plan_digest] = (plan, record_name, payload)

        prefix = []
        for plan_digest in sorted(generation_blocks, key=generation_blocks.__getitem__):
            row = verified.get(plan_digest)
            if row is None:
                break
            prefix.append((plan_digest, *row))
        try:
            validate_source_selection_sequence(
                tuple(plan for _digest, plan, _name, _payload in prefix),
                commitments=track.source_commitments,
                scoring=track.scoring,
            )
        except ValueError as exc:
            raise PublicArtifactError(
                "public batch disclosure source selection is invalid"
            ) from exc
        for _digest, _plan, record_name, payload in prefix:
            self._confirmed_immutable[record_name] = hashlib.sha256(payload).hexdigest()
        return frozenset(digest for digest, _plan, _name, _payload in prefix)

    def _verify_model_archives(self, matches: tuple[MatchRecord, ...]) -> None:
        for match in matches:
            evaluation = match.document["evaluation"]
            object_name = evaluation["model_object"]
            if object_name is None:
                continue
            self._verify_public_file(
                object_name,
                evaluation["file_sha256"],
                evaluation["model_size_bytes"],
            )

    def _verify_public_file(self, object_name: str, sha256: str, size_bytes: int) -> None:
        payload = self._get_optional(object_name)
        if payload is None:
            raise PublicArtifactError("public immutable file is missing")
        try:
            manifest = ChunkManifest.decode(payload)
        except ValueError as exc:
            raise PublicArtifactError("public immutable file manifest is invalid") from exc
        if manifest.sha256 != sha256 or manifest.size != size_bytes:
            raise PublicArtifactError("public immutable file identity is invalid")
        self._verify_public_chunks(object_name, manifest)
        self._confirmed_immutable[object_name] = sha256

    def _put_immutable(self, name: str, payload: bytes) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        if self._confirmed_immutable.get(name) == digest:
            return
        existing = self._get_optional(name)
        if existing is not None:
            if existing != payload:
                raise PublicArtifactError(
                    f"immutable public object conflicts with local data: {name}"
                )
            self._confirmed_immutable[name] = digest
            return
        self._put(name, payload, True, if_none_match="*")
        self._confirmed_immutable[name] = digest

    def _put_mutable(self, name: str, payload: bytes) -> None:
        stored = self._get_stored_optional(name)
        if stored is not None:
            current, etag = stored
            if current == payload:
                return
            self._guard_mutable(name, current, payload)
            try:
                self._put(name, payload, False, if_match=etag)
            except Exception as exc:
                if _is_precondition_failed(exc):
                    raise PublicArtifactError(
                        f"concurrent public pointer update was fenced: {name}"
                    ) from exc
                raise
            return
        try:
            self._put(name, payload, False, if_none_match="*")
        except Exception as exc:
            if _is_precondition_failed(exc):
                raise PublicArtifactError(
                    f"concurrent public pointer creation was fenced: {name}"
                ) from exc
            raise

    def _guard_mutable(self, name: str, current: bytes, proposed: bytes) -> None:
        if name == "track.json":
            before = _decode_canonical_object(current, "track")
            after = _decode_canonical_object(proposed, "track")
            try:
                before_id = before["spec"]["track_id"]
                after_id = after["spec"]["track_id"]
                before_start = _integer(before["spec"]["track_start_block"], "track_start_block")
                after_start = _integer(after["spec"]["track_start_block"], "track_start_block")
            except (KeyError, TypeError) as exc:
                raise PublicArtifactError("public track pointer is invalid") from exc
            if before_id == after_id or after_start <= before_start:
                raise PublicArtifactError("public track pointer rollback is forbidden")
            return
        if name == "weights.json":
            before_weight = decode_weight_artifact(current)
            after_weight = decode_weight_artifact(proposed)
            if before_weight.track_id != after_weight.track_id:
                if after_weight.track_id != self._active_track_id():
                    raise PublicArtifactError("public pointer targets an inactive track")
                return
            if before_weight.track_aborted and not after_weight.track_aborted:
                raise PublicArtifactError("public track abort is irreversible")
            before_block = before_weight.source_block
            after_block = after_weight.source_block
            before_count = after_count = 0
        else:
            before = _decode_canonical_object(current, name)
            after = _decode_canonical_object(proposed, name)
            try:
                if before["track_id"] != after["track_id"]:
                    if after["track_id"] != self._active_track_id():
                        raise PublicArtifactError("public pointer targets an inactive track")
                    return
                before_block = _integer(before["source_block"], "source_block")
                after_block = _integer(after["source_block"], "source_block")
                before_count = _integer(before.get("match_count", 0), "match_count")
                after_count = _integer(after.get("match_count", 0), "match_count")
            except KeyError as exc:
                raise PublicArtifactError(f"public pointer is invalid: {name}") from exc
        if after_block < before_block or after_count < before_count:
            raise PublicArtifactError("public snapshot rollback is forbidden")
        if after_block == before_block:
            raise PublicArtifactError("one source block cannot identify two public snapshots")

    def _active_track_id(self) -> str:
        payload = self._get_optional("track.json")
        if payload is None:
            raise PublicArtifactError("public mutable pointer exists without its track")
        document = _decode_canonical_object(payload, "track")
        try:
            track_id = document["spec"]["track_id"]
        except (KeyError, TypeError) as exc:
            raise PublicArtifactError("public track pointer is invalid") from exc
        if not isinstance(track_id, str) or not track_id:
            raise PublicArtifactError("public track pointer is invalid")
        return track_id

    def _put(
        self,
        name: str,
        payload: bytes,
        immutable: bool,
        *,
        if_match: str | None = None,
        if_none_match: str | None = None,
    ) -> None:
        if not payload or len(payload) > _MAX_PUBLIC_OBJECT_BYTES:
            raise PublicArtifactError(f"public object size is invalid: {name}")
        arguments = {
            "Bucket": self._bucket,
            "Key": f"{self._prefix}/{name}",
            "Body": payload,
            "ContentType": "application/json; charset=utf-8",
            "CacheControl": (
                "public, max-age=31536000, immutable" if immutable else "no-store, max-age=0"
            ),
        }
        if if_match is not None:
            arguments["IfMatch"] = if_match
        if if_none_match is not None:
            arguments["IfNoneMatch"] = if_none_match
        self._client.put_object(**arguments)

    def _get_stored_optional(self, name: str) -> tuple[bytes, str] | None:
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=f"{self._prefix}/{name}",
            )
        except Exception as exc:
            if _is_missing(exc):
                return None
            raise
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise PublicArtifactError(f"storage returned an invalid object body: {name}")
        payload = body.read(_MAX_PUBLIC_OBJECT_BYTES + 1)
        if not isinstance(payload, bytes):
            raise PublicArtifactError(f"storage returned a non-byte object body: {name}")
        if not payload or len(payload) > _MAX_PUBLIC_OBJECT_BYTES:
            raise PublicArtifactError(f"public object size is invalid: {name}")
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise PublicArtifactError(f"storage omitted object ETag: {name}")
        return payload, etag.strip('"')

    def _get_optional(self, name: str) -> bytes | None:
        stored = self._get_stored_optional(name)
        return None if stored is None else stored[0]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _summary_document(
    track: TrackSpec,
    source_block: int,
    tournament: Tournament,
) -> dict[str, Any]:
    champion = tournament.champion
    return {
        "track_id": track.track_id,
        "track_spec_digest": track.digest,
        "netuid": track.netuid,
        "source_block": source_block,
        "match_count": len(tournament.matches),
        "latest_match": tournament.matches[-1].object_name if tournament.matches else None,
        "champion": champion.public(),
        "leaderboard": list(tournament.leaderboard[:_SUMMARY_LEADERBOARD_LIMIT]),
        "leaderboard_total": len(tournament.leaderboard),
    }


def _status_document(
    track: TrackSpec,
    source_block: int,
    tournament: Tournament,
    competitors: list[dict[str, Any]],
    challenges: list[dict[str, Any]],
    *,
    track_aborted: bool = False,
) -> dict[str, Any]:
    """Build the bounded live view used by the operator monitor."""

    unfinished = [
        challenge
        for challenge in challenges
        if challenge["state"] not in {state.value for state in TERMINAL_STATES}
    ]
    if track_aborted or (source_block > track.track_close_block and unfinished):
        phase = "TRACK_ABORTED"
    elif source_block < track.track_start_block:
        phase = "SCHEDULED"
    elif source_block <= track.last_submission_block:
        phase = "ACCEPTING"
    elif source_block <= track.track_close_block:
        phase = "DRAINING"
    else:
        phase = "CLOSED"
    challenge_by_id = {item["submission_id"]: item for item in challenges}
    match_by_id = {item.challenge_id: item for item in tournament.matches}
    initial = genesis_champion(track)
    active_rows: list[dict[str, Any]] = [
        {
            "uid": initial.uid,
            "hotkey": initial.hotkey,
            "challenge_id": initial.challenge_id,
            "submission_block": track.track_start_block,
            "state": (
                "CHAMPION"
                if tournament.champion.challenge_id == initial.challenge_id
                else "DISPLACED"
            ),
            "queue_position": None,
            "match_sequence": 0,
            "score": (
                tournament.champion.score
                if tournament.champion.challenge_id == initial.challenge_id
                else initial.score
            ),
            "incumbent_score": None,
            "score_gain": None,
            "batch_generated_block": None,
            "outcome": "GENESIS",
            "champion": tournament.champion.challenge_id == initial.challenge_id,
            "attempts": 0,
            "error_code": None,
            "alert": False,
        }
    ]
    for competitor in competitors:
        challenge = challenge_by_id.get(competitor["submission_id"])
        if challenge is None:
            raise PublicArtifactError("active commitment has no local challenge")
        match = match_by_id.get(challenge["challenge_id"])
        is_champion = tournament.champion.challenge_id == challenge["challenge_id"]
        active_rows.append(
            {
                "uid": competitor["uid"],
                "hotkey": competitor["hotkey"],
                "challenge_id": challenge["challenge_id"],
                "submission_block": competitor["block_height"],
                "state": challenge["state"],
                "queue_position": None,
                "match_sequence": None if match is None else match.sequence,
                "score": (tournament.champion.score if is_champion else challenge["final_score"]),
                "incumbent_score": challenge["incumbent_score"],
                "score_gain": None if match is None else match.score_gain,
                "batch_generated_block": (
                    None
                    if challenge["evaluation_batch"] is None
                    else challenge["evaluation_batch"]["generated_block"]
                ),
                "outcome": None if match is None else match.outcome,
                "champion": is_champion,
                "attempts": challenge["attempts"],
                "error_code": challenge["error_code"],
                "alert": challenge["state"] == "SYSTEM_BLOCKED",
            }
        )
    waiting = [
        row
        for row in active_rows
        if row["match_sequence"] is None
        and row["state"] not in {str(state) for state in TERMINAL_STATES}
    ]
    waiting.sort(key=lambda row: (row["submission_block"], row["challenge_id"]))
    queue_positions = {
        row["challenge_id"]: position for position, row in enumerate(waiting, start=1)
    }
    for row in active_rows:
        row["queue_position"] = queue_positions.get(row["challenge_id"])
    active_rows.sort(key=lambda row: row["uid"])
    return {
        "track_id": track.track_id,
        "track_spec_digest": track.digest,
        "netuid": track.netuid,
        "source_block": source_block,
        "track_phase": phase,
        "last_submission_block": track.last_submission_block,
        "track_close_block": track.track_close_block,
        "abort": (
            None
            if phase != "TRACK_ABORTED"
            else {
                "reason": "UNFINISHED_SYSTEM_WORK_AFTER_CLOSE_GRACE",
                "abort_block": track.track_close_block + 1,
            }
        ),
        "champion": tournament.champion.public(),
        "system_matches": system_match_summaries(track.track_id, track.digest, track.netuid),
        "participants": active_rows,
    }


def _track_abort_artifact(track: TrackSpec) -> tuple[str, bytes]:
    return (
        f"events/{track.track_id}/track-aborted.json",
        _json_bytes(
            {
                "event": "TRACK_ABORTED",
                "reason": "UNFINISHED_SYSTEM_WORK_AFTER_CLOSE_GRACE",
                "track_id": track.track_id,
                "track_spec_digest": track.digest,
                "abort_block": track.track_close_block + 1,
            }
        ),
    )


def _decode_canonical_object(payload: bytes, name: str) -> dict[str, Any]:
    if not payload or len(payload) > _MAX_PUBLIC_OBJECT_BYTES:
        raise PublicArtifactError(f"public {name} size is invalid")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicArtifactError(f"public {name} is not valid JSON") from exc
    if not isinstance(raw, dict) or _json_bytes(raw) != payload:
        raise PublicArtifactError(f"public {name} is not a canonical JSON object")
    return raw


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


def _is_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = response.get("Error", {}).get("Code")
    return status == 404 or str(code) in {"404", "NoSuchKey", "NotFound"}


def _is_precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = response.get("Error", {}).get("Code")
    return status == 412 or str(code) in {"412", "PreconditionFailed"}
