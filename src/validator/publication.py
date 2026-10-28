"""Canonical public artifacts and S3-compatible object-storage transport."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from miner.submit import private_model_source
from scoring.spec import TrackSpec, data_spec_logical_name, parse_track_spec, track_spec_dict
from validator.chunks import (
    ChunkManifest,
    build_chunk_manifest,
    check_deadline,
    chunk_object_name,
    read_part,
    verify_chunked_content,
)
from validator.production.batches import BatchDisclosure, public_selected_inputs_document
from validator.production.commitment import Artifact, ArtifactManifest, verify_source_proof
from validator.production.inventory import Inventory
from validator.production.plan import (
    BatchPlan,
    build_plan,
    validate_source_selection_sequence,
)
from validator.queue import TERMINAL_STATES
from validator.release import (
    PRODUCTION_TRACK_ID,
    ReleaseApproval,
    require_release_approval,
    verify_data_binding,
)
from validator.s3 import client_config, is_missing, is_precondition_failed
from validator.store import ChallengeStore
from validator.system_matches import (
    MAX_SYSTEM_MATCHES,
    system_match_summaries,
    validate_system_match_artifact,
    validated_system_matches,
)
from validator.tournament import (
    Champion,
    MatchRecord,
    Tournament,
    challenge_from_match,
    genesis_champion,
    historical_champion,
    recorded_match,
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
HISTORICAL_MATCHES_PREFIX = "evidence/system-panel/history/"
_MAX_WEIGHT_ARTIFACT_BYTES = 1024 * 1024
_MAX_PUBLIC_OBJECT_BYTES = 1024 * 1024
_SUMMARY_LEADERBOARD_LIMIT = 100
_LOG = logging.getLogger(__name__)
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
    batch_groups: tuple[tuple[str, bytes], ...] = ()


@dataclass(frozen=True, slots=True)
class PublicFile:
    object_name: str
    path: Path | None
    sha256: str
    restore: Callable[[], Path] | None = None
    before_publish: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class PublicLedger:
    source_block: int | None
    matches: tuple[MatchRecord, ...]
    disclosed_plan_digests: frozenset[str] = frozenset()
    committed_match_count: int = 0
    historical_matches: tuple[MatchRecord, ...] = ()

    @property
    def required_finalized_block(self) -> int:
        committed = 0 if self.source_block is None else self.source_block
        decided = max(
            (int(match.document["decided_block"])
             for match in (*self.historical_matches, *self.matches)),
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


def _require_inventory_bindings(track: TrackSpec, document: Mapping[str, Any]) -> None:
    """Formal current matches need pre-submission seals, even before raw reveal.

    Historical records are authenticated separately against their original
    runtime manifest. This guard does not relabel them as this pool protocol.
    """
    submission = document["submission"]["block_height"]
    evaluation = document["evaluation"]
    for stage in evaluation["stages"]:
        batch = stage.get("evaluation_batch") or {}
        allocation = batch.get("allocation") or {}
        completion = _integer(batch.get("completed_block"), "batch completion block")
        completed_at = _integer(batch.get("completed_at_ms"), "batch completion time")
        submitted_at = _integer(allocation.get("submission_timestamp_ms"), "submission time")
        if (
            not batch["generated_block"] < completion < submission
            or not completed_at < submitted_at
            < completed_at + track.cache.disclosure_delay_seconds * 1000
            or not _HEX_32.fullmatch(str(batch.get("ready_sha256", "")))
            or allocation.get("ready_sha256") != batch["ready_sha256"]
            or allocation.get("plan_digest") != batch["plan_digest"]
            or allocation.get("challenge_id") != document["challenge_id"]
            or allocation.get("submission_block") != submission
            or allocation.get("stage") != stage["stage"]
            or allocation.get("file_sha256") != evaluation["file_sha256"]
            or allocation.get("model_digest") != evaluation["model_digest"]
        ):
            raise PublicArtifactError("current match lacks its original pre-submission allocation")


def build_public_artifacts(
    store: ChallengeStore,
    track: TrackSpec,
    chain: WeightChainState,
    *,
    version_key: int,
    match_limit: int | None = None,
    model_resolver: Callable[[str, int], Path] | None = None,
    private_winner_archive: Callable[[str, int], None] | None = None,
    runtime_manifest: bytes | None = None,
    release_approval: ReleaseApproval | None = None,
    system_matches: tuple[tuple[str, bytes], ...] = (),
    inventory_status: Mapping[str, Any] | None = None,
    evaluation_running: bool = False,
    historical_matches: tuple[MatchRecord, ...] = (),
) -> PublicArtifacts | None:
    """Build bounded metadata without restoring or uploading disclosure payloads."""

    _verify_candidate_release(track, runtime_manifest, release_approval)
    if any(match.document["decided_block"] > chain.finalized_block
           for match in historical_matches):
        raise PublicArtifactError("historical decision is newer than the finalized chain snapshot")
    cursor = store.chain_cursor()
    if cursor is None or cursor[0] != chain.finalized_block:
        return None
    snapshot = store.public_snapshot()
    initial = genesis_champion(track)
    if match_limit is not None and (type(match_limit) is not int or match_limit < 0):
        raise ValueError("public match limit must be a non-negative integer")
    challenges = snapshot["challenges"]
    if match_limit is not None:
        # Historical terminal rows retain lifetime uniqueness, but never consume
        # this track's independently numbered, immutable publication prefix.
        historical = [row for row in challenges
                      if row.get("track_spec_digest") not in (None, track.digest)]
        current = [row for row in challenges
                   if row.get("track_spec_digest") in (None, track.digest)]
        challenges = historical + current[:match_limit]
    tournament = resolve_tournament(
        challenges,
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        minimum_gain=track.full_video.minimum_gain,
        minimum_exact_gain=track.scoring.minimum_frontier_gain,
        promotion=track.promotion,
        initial_champion=initial,
        historical_matches=historical_matches,
    )
    if match_limit is not None and len(tournament.matches) != match_limit:
        raise PublicArtifactError("the committed public match prefix is missing locally")
    if any(match.document["decided_block"] > chain.finalized_block for match in tournament.matches):
        raise PublicArtifactError("a terminal decision is newer than the finalized chain snapshot")
    match_sequence, terminal_match_sha256 = _terminal_match_fence(tournament)
    # Contest switches withdraw rewards without rewriting historical matches.
    excluded = store.legacy_excluded_hotkeys()
    eligible_uids = {
        hotkey: uid
        for hotkey, uid in chain.uid_by_hotkey.items()
        if hotkey not in excluded or hotkey == initial.hotkey
    }
    plan = plan_champion_weight(
        tournament.champion,
        uid_by_hotkey=eligible_uids,
        current_block=chain.finalized_block,
        min_allowed_weights=chain.min_allowed_weights,
        max_weight=chain.max_weight,
    )
    if plan is None and tournament.champion.challenge_id != initial.challenge_id:
        plan = plan_champion_weight(
            initial,
            uid_by_hotkey=eligible_uids,
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
        match_sequence=match_sequence,
        terminal_match_sha256=terminal_match_sha256,
    )
    summary = _summary_document(track, chain.finalized_block, tournament)
    system_matches = validated_system_matches(track, runtime_manifest, system_matches)
    status = _status_document(
        track,
        chain.finalized_block,
        tournament,
        store.active_competitors(chain.uid_by_hotkey),
        snapshot["challenges"],
        system_matches=system_match_summaries(system_matches),
        historical_matches=historical_matches,
    )
    if release_approval is not None and runtime_manifest is not None:
        status["runtime_release"] = release_approval.public_identity(track, runtime_manifest)
    if inventory_status is not None:
        status["inventory"] = {
            key: value for key, value in inventory_status.items() if key != "allocated_stages"
        }
        allocated = set(inventory_status["allocated_stages"])
        challenges_by_id = {item["challenge_id"]: item for item in snapshot["challenges"]}
        for row in status["participants"]:
            if row["queue_position"] != 1:
                continue
            state = row["state"]
            stage = state
            if state == "SYSTEM_DEFERRED":
                events = challenges_by_id[row["challenge_id"]].get("events", [])
                stage = events[-1].get("previous_state") if events else None
            if stage in {"SCORING", "CONFIRMING"}:
                row["activity"] = (
                    "WAIT_INVENTORY" if (row["challenge_id"], stage) not in allocated
                    else "COMPUTING" if evaluation_running
                    else "RETRY_WAIT" if state == "SYSTEM_DEFERRED" else "READY"
                )
    files: dict[str, PublicFile] = {}
    for match in tournament.matches:
        evaluation = match.document["evaluation"]
        object_name = evaluation["model_object"]
        if object_name is not None:
            path_value = store.model_path(match.challenge_id)
            path = None if path_value is None else Path(path_value)
            files[object_name] = PublicFile(
                object_name,
                path,
                evaluation["file_sha256"],
                None if model_resolver is None else partial(
                    model_resolver,
                    evaluation["file_sha256"],
                    evaluation["model_size_bytes"],
                ),
                None if (private_winner_archive is None
                         or private_model_source(match.document["submission"]) is None
                         or match.document["outcome"] != "CHALLENGER_WON") else partial(
                    private_winner_archive, evaluation["file_sha256"],
                    evaluation["model_size_bytes"],
                ),
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
        system_matches=system_matches,
        files=tuple(files.values()),
        runtime_manifest=_runtime_manifest(track, runtime_manifest, release_approval),
    )


def build_genesis_artifacts(
    track: TrackSpec,
    chain: WeightChainState,
    *,
    version_key: int,
    runtime_manifest: bytes | None = None,
    release_approval: ReleaseApproval | None = None,
    system_matches: tuple[tuple[str, bytes], ...] = (),
    historical_matches: tuple[MatchRecord, ...] = (),
    excluded_hotkeys: frozenset[str] = frozenset(),
) -> PublicArtifacts:
    """Build the pre-launch snapshot, retaining any confirmed historical champion."""

    _verify_candidate_release(track, runtime_manifest, release_approval)
    if any(match.document["decided_block"] > chain.finalized_block
           for match in historical_matches):
        raise PublicArtifactError("historical decision is newer than the finalized chain snapshot")
    tournament = resolve_tournament(
        (),
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        minimum_gain=track.full_video.minimum_gain,
        minimum_exact_gain=track.scoring.minimum_frontier_gain,
        promotion=track.promotion,
        initial_champion=genesis_champion(track),
        historical_matches=historical_matches,
    )
    eligible_uids = {hotkey: uid for hotkey, uid in chain.uid_by_hotkey.items()
                     if hotkey not in excluded_hotkeys or hotkey == track.initial_champion.hotkey}
    plan = plan_champion_weight(
        tournament.champion,
        uid_by_hotkey=eligible_uids,
        current_block=chain.finalized_block,
        min_allowed_weights=chain.min_allowed_weights,
        max_weight=chain.max_weight,
    )
    if plan is None and tournament.champion.challenge_id != "genesis":
        plan = plan_champion_weight(
            genesis_champion(track), uid_by_hotkey=eligible_uids,
            current_block=chain.finalized_block,
            min_allowed_weights=chain.min_allowed_weights, max_weight=chain.max_weight,
        )
    if plan is None and tournament.champion.challenge_id == "genesis":
        raise PublicArtifactError("UID0 cannot receive the complete chain weight")
    weights = WeightArtifact(
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        mechid=track.mechid,
        version_key=version_key,
        source_block=chain.finalized_block,
        plan=plan,
        match_sequence=0,
        terminal_match_sha256=None,
    )
    system_matches = validated_system_matches(track, runtime_manifest, system_matches)
    status = _status_document(
        track, chain.finalized_block, tournament, [], [],
        system_matches=system_match_summaries(system_matches),
        historical_matches=historical_matches,
    )
    if release_approval is not None and runtime_manifest is not None:
        status["runtime_release"] = release_approval.public_identity(track, runtime_manifest)
    return PublicArtifacts(
        source_block=chain.finalized_block,
        track=_json_bytes({"track_spec_digest": track.digest, "spec": track_spec_dict(track)}),
        summary=_json_bytes(_summary_document(track, chain.finalized_block, tournament)),
        weights=encode_weight_artifact(weights),
        matches=(),
        status=_json_bytes(status),
        system_matches=system_matches,
        runtime_manifest=_runtime_manifest(track, runtime_manifest, release_approval),
    )


def _verify_candidate_release(
    track: TrackSpec, runtime_manifest: bytes | None, approval: ReleaseApproval | None
) -> None:
    if track.track_id != PRODUCTION_TRACK_ID and approval is None:
        return
    try:
        require_release_approval(track, runtime_manifest, approval)
    except ValueError as exc:
        raise PublicArtifactError(str(exc)) from exc


def _runtime_manifest(
    track: TrackSpec, payload: bytes | None, approval: ReleaseApproval | None = None,
) -> tuple[str, bytes] | None:
    if payload is None:
        return None
    if hashlib.sha256(payload).hexdigest() != track.artifact_manifest_sha256:
        try:
            require_release_approval(track, payload, approval)
        except ValueError as exc:
            raise PublicArtifactError("runtime artifact manifest digest is incorrect") from exc
    return f"runtime/{hashlib.sha256(payload).hexdigest()}/manifest.json", payload


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
            "match_sequence": artifact.match_sequence,
            "terminal_match_sha256": artifact.terminal_match_sha256,
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
    if isinstance(raw, dict) and (
        raw.get("track_id"), raw.get("track_spec_digest")
    ) == _LEGACY_UNFENCED_WEIGHT_TRACK:
        raw.setdefault("track_aborted", False)
        raw.setdefault("match_sequence", 0)
        raw.setdefault("terminal_match_sha256", None)
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
            "match_sequence",
            "terminal_match_sha256",
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
    match_sequence = _integer(root["match_sequence"], "match_sequence")
    terminal_match_sha256 = root["terminal_match_sha256"]
    if terminal_match_sha256 is not None:
        terminal_match_sha256 = _string(
            terminal_match_sha256,
            "terminal_match_sha256",
        )
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
            match_sequence,
            terminal_match_sha256,
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
    return _decode_track_document(bytes(received))


def _decode_track_document(payload: bytes) -> dict[str, Any]:
    """Authenticate the version-independent envelope, not an older scoring schema."""

    document = _decode_canonical_object(payload, "track")
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

    def __init__(
        self, client: Any, *, bucket: str, prefix: str,
        release_approval: ReleaseApproval | None = None,
    ) -> None:
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
        self.release_approval = release_approval
        self._confirmed_immutable: dict[str, str] = {}
        self._writer_lock: Any | None = None
        self._system_match_cache: dict[tuple[str, str], dict[str, bytes]] = {}
        self._history_cache: dict[tuple[str, str], tuple[MatchRecord, ...]] = {}

    def acquire_writer_lock(self, database: str | Path) -> None:
        """Hold the one evaluator deployment's filesystem lock until close or process exit."""

        if self._writer_lock is not None:
            return
        path = Path(database).parent / ".public-writer.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise PublicArtifactError("another evaluator holds the public writer lock") from exc
        self._writer_lock = handle

    def close(self) -> None:
        if self._writer_lock is not None:
            self._writer_lock.close()
            self._writer_lock = None

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
            config=client_config(),
        )
        return cls(
            client, bucket=bucket, prefix=prefix,
            release_approval=ReleaseApproval.from_environment(),
        )

    def verify_writes(self) -> None:
        """Check write/read visibility for the single managed public publisher."""

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
            if first is None or first[0] != b'{"state":"first"}':
                raise PublicArtifactError("public object creation was not durable")
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=b'{"state":"ready"}',
                IfMatch=first[1],
            )
            stored = self._get_stored_optional(name)
            if stored is None or stored[0] != b'{"state":"ready"}':
                raise PublicArtifactError("public object replacement was not durable")
        finally:
            self._client.delete_object(Bucket=self._bucket, Key=key)

    async def stage(self, artifacts: PublicArtifacts) -> None:
        """Verify/upload immutable facts without changing any mutable pointer."""

        track_document = _decode_canonical_object(artifacts.track, "track")
        if self._prefix == "tracks/main":
            try:
                document = track_document
                if hashlib.sha256(_json_bytes(document["spec"])).hexdigest() != document[
                    "track_spec_digest"
                ]:
                    raise ValueError("public track digest is invalid")
                track = parse_track_spec(document["spec"])
                for name, payload in artifacts.matches:
                    _require_inventory_bindings(track, _decode_canonical_object(payload, name))
                require_release_approval(
                    track,
                    (
                        None if artifacts.runtime_manifest is None
                        else artifacts.runtime_manifest[1]
                    ),
                    self.release_approval,
                )
                runtime = (None if artifacts.runtime_manifest is None
                           else artifacts.runtime_manifest[1])
                status_release = _decode_canonical_object(artifacts.status, "status").get(
                    "runtime_release",
                )
                expected_release = (None if runtime is None or self.release_approval is None
                                    else self.release_approval.public_identity(track, runtime))
                if runtime is not None and status_release != expected_release:
                    raise ValueError("public runtime identity differs from release approval")
            except (KeyError, TypeError, ValueError) as exc:
                raise PublicArtifactError(f"formal release rejected: {exc}") from exc
        try:
            track_id = track_document["spec"]["track_id"]
        except (KeyError, TypeError) as exc:
            raise PublicArtifactError("public track pointer is invalid") from exc
        if not isinstance(track_id, str) or not track_id:
            raise PublicArtifactError("public track pointer is invalid")
        weight_fence = decode_weight_artifact(artifacts.weights)
        if weight_fence.track_id != track_id:
            raise PublicArtifactError("public weights target a different track")
        # Check before any payload PUT: private losses must not carry either an
        # archive reference or a stray model file. Full track snapshots use the
        # same canonical resolver as public recovery, including both strict gates.
        if "scoring" in track_document["spec"] or any(
            item.object_name.startswith("models/") for item in artifacts.files
        ):
            try:
                track = parse_track_spec(track_document["spec"])
                model_objects = {}
                incumbent = genesis_champion(track)
                if artifacts.runtime_manifest is not None:
                    competition = artifacts.runtime_manifest[1]
                    if (self.release_approval is not None
                            and self.release_approval.competition_manifest is not None):
                        competition = self.release_approval.competition_manifest
                    history = await asyncio.to_thread(self._load_history, track, competition)
                    incumbent = historical_champion(history, incumbent)
                for sequence, (name, payload) in enumerate(artifacts.matches, start=1):
                    document = _decode_canonical_object(payload, name)
                    match = validate_match_document(
                        document, sequence=sequence, track_id=track.track_id,
                        track_spec_digest=track.digest, netuid=track.netuid,
                        minimum_gain=track.full_video.minimum_gain,
                        minimum_exact_gain=track.scoring.minimum_frontier_gain,
                        promotion=track.promotion,
                        incumbent=incumbent,
                    )
                    incumbent = match.champion_after
                    if name != match.object_name:
                        raise ValueError("public model match name is invalid")
                    evaluation = match.document["evaluation"]
                    if evaluation["model_object"] is not None:
                        model_objects[evaluation["model_object"]] = evaluation["file_sha256"]
                if any(item.object_name.startswith("models/")
                       and model_objects.get(item.object_name) != item.sha256
                       for item in artifacts.files):
                    raise ValueError("unreferenced public model file")
            except (KeyError, TypeError, ValueError) as exc:
                raise PublicArtifactError("public model publication violates its match") from exc
        if artifacts.runtime_manifest is not None:
            track = parse_track_spec(track_document["spec"])
            if track.data_spec_sha256 is not None:
                await asyncio.to_thread(
                    self._stage_data_origin, track, artifacts.runtime_manifest[1],
                )
            await asyncio.to_thread(self._put_immutable, *artifacts.runtime_manifest)
        for artifact in artifacts.files:
            if artifact.before_publish is not None:
                await asyncio.to_thread(artifact.before_publish)
            await asyncio.to_thread(self._put_file_immutable, artifact)
        await self._stage_system_matches(artifacts)
        for name, payload in artifacts.matches:
            await asyncio.to_thread(self._put_immutable, name, payload)
        for name, payload in artifacts.batch_groups:
            await asyncio.to_thread(self._put_immutable, name, payload)
        if artifacts.runtime_manifest is not None:
            track = parse_track_spec(track_document["spec"])
            competition_manifest = artifacts.runtime_manifest[1]
            if (self.release_approval is not None
                    and self.release_approval.competition_manifest is not None):
                competition_manifest = self.release_approval.competition_manifest
            history = await asyncio.to_thread(
                self._load_history, track, competition_manifest,
            )
            if history:
                tournament = resolve_tournament(
                    [challenge_from_match(_decode_canonical_object(body, name))
                     for name, body in artifacts.matches],
                    track_id=track.track_id, track_spec_digest=track.digest, netuid=track.netuid,
                    minimum_gain=track.full_video.minimum_gain,
                    minimum_exact_gain=track.scoring.minimum_frontier_gain,
                    promotion=track.promotion,
                    initial_champion=genesis_champion(track), historical_matches=history,
                )
                if artifacts.summary != _json_bytes(
                    _summary_document(track, artifacts.source_block, tournament)
                ):
                    raise PublicArtifactError("public snapshot would reset its historical champion")

    async def publish_disclosure(
        self, disclosure: BatchDisclosure, *, track: TrackSpec, now_ms: int,
        deadline: float | None = None,
    ) -> None:
        """One age-triggered bulk job; never write match or mutable pointer keys."""
        document = _decode_canonical_object(disclosure.record_payload, disclosure.record_name)
        try:
            plan = BatchPlan.decode(_json_bytes(document["plan"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicArtifactError("public disclosure READY is invalid") from exc
        if (
            document.get("format") != "veda-batch-disclosure-v3"
            or plan.track_spec_digest != track.digest or plan.state != "READY"
            or plan.completed_block is None
            or plan.completed_at_ms + track.cache.disclosure_delay_seconds * 1000 > now_ms
            or disclosure.record_name != f"batch_groups/{plan.plan_digest}.json"
        ):
            raise PublicArtifactError("public disclosure is premature or belongs to another track")
        for artifact in disclosure.files:
            check_deadline(deadline)
            if artifact.object_name not in {
                f"{namespace}/{artifact.sha256}/manifest.json"
                for namespace in ("batches", "batch_assets", "batch_inputs")
            }:
                raise PublicArtifactError("disclosure cannot write models or mutable pointers")
            await asyncio.to_thread(
                self._put_file_immutable,
                PublicFile(artifact.object_name, artifact.path, artifact.sha256), deadline=deadline,
            )
        check_deadline(deadline)
        await asyncio.to_thread(self._put_immutable,
                                disclosure.record_name, disclosure.record_payload)

    async def publish(self, artifacts: PublicArtifacts) -> None:
        await self.stage(artifacts)
        await asyncio.to_thread(self._put_mutable, "track.json", artifacts.track)
        await asyncio.to_thread(self._put_mutable, "weights.json", artifacts.weights)
        await asyncio.to_thread(self._put_mutable, "summary.json", artifacts.summary)
        status = artifacts.status
        if status is not None and (artifacts.system_matches or self._system_match_cache):
            try:
                document = _decode_canonical_object(artifacts.track, "track")
                manifest = (
                    None if artifacts.runtime_manifest is None else artifacts.runtime_manifest[1]
                )
                key = (document["track_spec_digest"], hashlib.sha256(manifest or b"").hexdigest())
                visible = tuple(sorted(self._system_match_cache.get(key, {}).items()))
                if artifacts.system_matches or visible:
                    status_document = _decode_canonical_object(status, "status")
                    status_document["system_matches"] = system_match_summaries(visible)
                    status = _json_bytes(status_document)
            except Exception as exc:
                _LOG.warning("system benchmark status skipped: %s", exc)
                status = None
        if status is not None:
            try:
                await asyncio.to_thread(self._put_mutable, "status.json", status)
            except PublicArtifactError:
                current = await asyncio.to_thread(self._get_optional, "status.json")
                if current is None:
                    raise
                before = _decode_canonical_object(current, "status")
                after = _decode_canonical_object(status, "status")
                if before.pop("system_matches", None) == after.pop("system_matches", None):
                    raise
                if before != after:
                    raise
                # Never rewrite a frozen source-block snapshot for new information.
                _LOG.warning("system benchmark status deferred until a new block")

    def publish_mailbox(self, name: str, ciphertext: bytes) -> None:
        """The existing public prefix carries ONLY the sealed credential blob.

        This deliberately cannot write formal track/weights/match objects.
        Caller is the control-side mailbox signer, not a miner/S3 credential.
        """
        if (re.fullmatch(r"mailbox/v1/[a-f0-9]{64}\.bin", name) is None
                or not isinstance(ciphertext, bytes) or not 48 <= len(ciphertext) <= 65536):
            raise PublicArtifactError("invalid encrypted mailbox object")
        self._put(name, ciphertext, False, content_type="application/octet-stream")
        if self._get_optional(name) != ciphertext:
            raise PublicArtifactError("encrypted mailbox readback failed")

    async def load_system_matches(
        self, track: TrackSpec, runtime_manifest: bytes | None,
    ) -> tuple[tuple[str, bytes], ...]:
        """Read a bounded immutable sequence; retain this runtime's last valid facts."""

        if runtime_manifest is None:
            return ()
        key = (track.digest, hashlib.sha256(runtime_manifest).hexdigest())
        known = self._system_match_cache.setdefault(key, {})
        if known:
            valid = await asyncio.to_thread(
                validated_system_matches, track, runtime_manifest, tuple(known.items()),
            )
            known.clear()
            known.update(valid)
        for sequence in range(1, MAX_SYSTEM_MATCHES + 1):
            name = f"system_matches/{track.track_id}/{sequence:012d}.json"
            try:
                payload = await asyncio.to_thread(self._get_optional, name)
            except Exception as exc:
                _LOG.warning("system benchmark read failed: %s (%s)", name, exc)
                break
            if payload is None:
                if name in known:
                    _LOG.warning("system benchmark missing; retaining last valid record: %s", name)
                break
            try:
                await asyncio.to_thread(
                    validate_system_match_artifact, track, runtime_manifest, name, payload,
                )
                if name in known and known[name] != payload:
                    raise ValueError("immutable system benchmark changed")
                known[name] = payload
                self._confirmed_immutable[name] = hashlib.sha256(payload).hexdigest()
            except Exception as exc:
                _LOG.warning("system benchmark skipped: %s (%s)", name, exc)
        return tuple(sorted(known.items()))

    async def _stage_system_matches(self, artifacts: PublicArtifacts) -> None:
        if not artifacts.system_matches:
            return
        try:
            document = _decode_track_document(artifacts.track)
            track = parse_track_spec(document["spec"])
            manifest = None if artifacts.runtime_manifest is None else artifacts.runtime_manifest[1]
            key = (track.digest, hashlib.sha256(manifest or b"").hexdigest())
            known = self._system_match_cache.setdefault(key, {})
        except Exception as exc:
            _LOG.warning("system benchmark publication skipped: %s", exc)
            return
        for name, payload in artifacts.system_matches:
            try:
                await asyncio.to_thread(
                    validate_system_match_artifact, track, manifest, name, payload,
                )
                if name in known and known[name] != payload:
                    raise ValueError("immutable system benchmark changed")
                await asyncio.to_thread(self._put_immutable, name, payload)
                known[name] = payload
            except Exception as exc:
                _LOG.warning("system benchmark publication skipped: %s (%s)", name, exc)

    def _put_file_immutable(self, artifact: PublicFile, *, deadline: float | None = None) -> None:
        check_deadline(deadline)
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
            self._verify_public_chunks(artifact.object_name, manifest, deadline=deadline)
            self._confirmed_immutable[artifact.object_name] = artifact.sha256
            return

        path = artifact.path
        try:
            manifest = None if path is None else build_chunk_manifest(path, deadline=deadline)
        except TimeoutError:
            raise
        except (OSError, ValueError):
            manifest = None
        if manifest is None or manifest.sha256 != artifact.sha256:
            if artifact.restore is None:
                raise PublicArtifactError(f"public file source is invalid: {artifact.object_name}")
            # Only unpublished models need local recovery. Already verified public
            # archives return above without re-reading historical local models.
            if path is not None:
                path.unlink(missing_ok=True)
            path = artifact.restore()
            manifest = build_chunk_manifest(path)
            if manifest.sha256 != artifact.sha256:
                raise PublicArtifactError(
                    f"restored public file digest changed: {artifact.object_name}"
                )
        for index, part in enumerate(manifest.parts):
            check_deadline(deadline)
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
                Body=read_part(path, index, part.size),
                ContentType="application/octet-stream",
                CacheControl="public, max-age=31536000, immutable",
                Metadata={"sha256": part.sha256},
                IfNoneMatch="*",
            )
        self._verify_public_chunks(artifact.object_name, manifest, deadline=deadline)
        self._put(artifact.object_name, manifest.encode(), True, if_none_match="*")
        self._confirmed_immutable[artifact.object_name] = artifact.sha256

    def _verify_public_chunks(
        self, manifest_name: str, manifest: ChunkManifest, *, deadline: float | None = None,
    ) -> None:
        def open_part(part_name: str):
            name = chunk_object_name(manifest_name, part_name)
            try:
                response = self._client.get_object(
                    Bucket=self._bucket,
                    Key=f"{self._prefix}/{name}",
                )
            except Exception as exc:
                response = getattr(exc, "response", None)
                code = (response or {}).get("Error", {}).get("Code")
                if str(code) in {"404", "NoSuchKey", "NotFound"}:
                    raise PublicArtifactError(f"immutable public chunk is missing: {name}") from exc
                raise
            return response.get("Body")

        try:
            verify_chunked_content(manifest, open_part, deadline=deadline)
        except ValueError as exc:
            raise PublicArtifactError(f"immutable public {exc}") from exc

    async def load(
        self, track: TrackSpec, *, runtime_manifest: bytes | None = None,
    ) -> PublicLedger:
        """Load and verify the committed public tournament ledger from object storage."""

        return await asyncio.to_thread(self._load, track, runtime_manifest)

    async def recover(
        self, track: TrackSpec, *, runtime_manifest: bytes | None = None,
    ) -> PublicLedger:
        """Recover authoritative facts without auditing previously published bulk."""
        return await asyncio.to_thread(self._recover, track, runtime_manifest)

    async def collect_history_artifacts(self) -> tuple[Artifact, ...]:
        """Collect original references for a rules migration, never a second ledger.

        Reuse normal authenticated recovery (including staged immutable tails).
        A changing track or weight fence requires recollection before binding.
        This reads only small public facts, not model or disclosure payloads.
        """
        return await asyncio.to_thread(self._collect_history_artifacts)

    def _collect_history_artifacts(self) -> tuple[Artifact, ...]:
        pointer = self._get_optional("track.json")
        fence = self._get_optional("weights.json")
        if pointer is None or fence is None:
            raise PublicArtifactError("maintenance requires a published track and weight fence")
        document = _decode_track_document(pointer)
        track = parse_track_spec(document["spec"])
        weight = decode_weight_artifact(fence)
        if (weight.track_id != track.track_id or weight.track_spec_digest != track.digest
                or weight.netuid != track.netuid or weight.mechid != track.mechid):
            raise PublicArtifactError("maintenance weight fence targets a different track")
        ledger = self._recover(track)
        artifacts = tuple(sorted((
            Artifact(
                HISTORICAL_MATCHES_PREFIX + match.object_name,
                hashlib.sha256(_json_bytes(match.document)).hexdigest(),
                len(_json_bytes(match.document)), match.object_name,
            )
            for match in (*ledger.historical_matches, *ledger.matches)
        ), key=lambda artifact: artifact.logical_name))
        if (self._get_optional("track.json") != pointer
                or self._get_optional("weights.json") != fence):
            raise PublicArtifactError("maintenance public fence changed during collection")
        return artifacts

    def _load_history(
        self, track: TrackSpec, runtime_manifest: bytes | None = None,
    ) -> tuple[MatchRecord, ...]:
        """Reuse original public match objects pinned by the existing release manifest."""
        if (runtime_manifest is not None
                and hashlib.sha256(runtime_manifest).hexdigest() != track.artifact_manifest_sha256):
            raise PublicArtifactError("historical ledger runtime manifest is invalid")
        key = (track.digest, track.artifact_manifest_sha256)
        if key in self._history_cache:
            return self._history_cache[key]
        if runtime_manifest is None:
            runtime_manifest = self._get_optional(
                f"runtime/{track.artifact_manifest_sha256}/manifest.json",
            )
        if runtime_manifest is None:
            if self._prefix == "tracks/main":
                raise PublicArtifactError("formal ledger is missing its runtime manifest")
            return ()
        if hashlib.sha256(runtime_manifest).hexdigest() != track.artifact_manifest_sha256:
            raise PublicArtifactError("historical ledger runtime manifest is invalid")
        manifest = ArtifactManifest.decode(runtime_manifest)
        matches = []
        for artifact in manifest.artifacts:
            if not artifact.logical_name.startswith(HISTORICAL_MATCHES_PREFIX):
                continue
            name = artifact.logical_name.removeprefix(HISTORICAL_MATCHES_PREFIX)
            if (not re.fullmatch(r"matches/[a-z0-9][a-z0-9-]{0,63}/[0-9]{12}\.json", name)
                    or artifact.object_key != name):
                raise PublicArtifactError("historical match must reference its original public key")
            payload = self._get_optional(name)
            if (payload is None or len(payload) != artifact.size_bytes
                    or hashlib.sha256(payload).hexdigest() != artifact.sha256):
                raise PublicArtifactError("historical immutable match is missing or changed")
            try:
                record = recorded_match(
                    _decode_canonical_object(payload, name), netuid=track.netuid,
                )
                if (record.object_name != name or record.document["track_id"] == track.track_id
                        or record.document["track_spec_digest"] == track.digest):
                    raise ValueError("historical match was rebound to the current track")
            except ValueError as exc:
                raise PublicArtifactError(f"historical immutable match is invalid: {name}") from exc
            matches.append(record)
        matches.sort(key=lambda match: (
            match.document["submission"]["block_height"],
            match.document["submission"]["extrinsic_index"],
            match.document["submission"]["event_index"],
        ))
        result = tuple(matches)
        try:
            historical_champion(result, genesis_champion(track))
        except ValueError as exc:
            raise PublicArtifactError(f"historical champion chain is invalid: {exc}") from exc
        self._history_cache[key] = result
        return result

    def _load(self, track: TrackSpec, runtime_manifest: bytes | None = None) -> PublicLedger:
        ledger = self._recover(track, runtime_manifest)
        self._verify_model_archives(ledger.matches)
        data_track = self._load_data_track(track, runtime_manifest)
        disclosures = self._load_batch_disclosures(data_track, ledger.matches)
        return replace(ledger, disclosed_plan_digests=disclosures)

    def _stage_data_origin(self, track: TrackSpec, runtime_manifest: bytes) -> None:
        """Preserve the pinned original public spec before changing public pointers.

        Only the small original specification is materialized as a bound artifact;
        no plan, READY, claim, model or disclosure record is relabelled or copied.
        """
        manifest = ArtifactManifest.decode(runtime_manifest)
        origin = next((a for a in manifest.artifacts
                       if a.logical_name == data_spec_logical_name(track)), None)
        if origin is None or origin.object_key != f"artifacts/{track.data_spec_sha256}":
            raise PublicArtifactError("data origin artifact is not bound to its fixed key")
        body = self._get_optional(origin.object_key)
        if body is None:
            pointer = self._get_optional("track.json")
            if pointer is None:
                raise PublicArtifactError("original data specification must exist before cutover")
            previous = _decode_track_document(pointer)
            data = parse_track_spec(previous["spec"])
            if (previous["track_spec_digest"] != data.digest
                    or data.digest != track.data_spec_sha256):
                raise PublicArtifactError("previous public pointer is not the pinned data origin")
            body = _json_bytes(previous["spec"])
        data = parse_track_spec(_decode_canonical_object(body, origin.logical_name))
        original = self._get_optional(f"runtime/{data.artifact_manifest_sha256}/manifest.json")
        if original is None or len(body) != origin.size_bytes:
            raise PublicArtifactError("original data manifest is unavailable")
        try:
            verify_data_binding(track, runtime_manifest, body, original)
        except ValueError as exc:
            raise PublicArtifactError("published data origin differs from its contract") from exc
        self._put_immutable(origin.object_key, body)

    def _load_data_track(
        self, track: TrackSpec, runtime_manifest: bytes | None = None,
    ) -> TrackSpec:
        """Read the single manifest-pinned original data contract, never a fallback pool."""
        if track.data_spec_sha256 is None:
            return track
        if runtime_manifest is None:
            runtime_manifest = self._get_optional(
                f"runtime/{track.artifact_manifest_sha256}/manifest.json",
            )
        if runtime_manifest is None:
            raise PublicArtifactError("data origin requires its verified competition manifest")
        if hashlib.sha256(runtime_manifest).hexdigest() != track.artifact_manifest_sha256:
            raise PublicArtifactError("data origin manifest differs from its rule commitment")
        manifest = ArtifactManifest.decode(runtime_manifest)
        origin = next((a for a in manifest.artifacts
                       if a.logical_name == data_spec_logical_name(track)), None)
        if (origin is None or origin.sha256 != track.data_spec_sha256
                or origin.object_key != f"artifacts/{track.data_spec_sha256}"):
            raise PublicArtifactError("data origin is missing from the committed manifest")
        body = self._get_optional(origin.object_key)
        if (body is None or hashlib.sha256(body).hexdigest() != origin.sha256
                or len(body) != origin.size_bytes):
            raise PublicArtifactError("original data specification is unavailable or changed")
        try:
            data = parse_track_spec(_decode_canonical_object(body, origin.logical_name))
            original = self._get_optional(f"runtime/{data.artifact_manifest_sha256}/manifest.json")
            if original is None:
                raise ValueError("original data manifest is unavailable")
            return verify_data_binding(track, runtime_manifest, body, original)
        except ValueError as exc:
            raise PublicArtifactError("public data contract is incompatible") from exc

    def _recover(self, track: TrackSpec, runtime_manifest: bytes | None = None) -> PublicLedger:
        history = self._load_history(track, runtime_manifest)
        track_payload = self._get_optional("track.json")
        summary_payload = self._get_optional("summary.json")
        if track_payload is None:
            if summary_payload is not None:
                raise PublicArtifactError("public summary exists without its immutable track")
            return self._uncommitted_ledger(track, history)

        track_document = _decode_track_document(track_payload)
        try:
            stored_id = _string(track_document["spec"]["track_id"], "track_id")
        except (KeyError, TypeError) as exc:
            raise PublicArtifactError("public track identity is invalid") from exc
        if stored_id != track.track_id:
            return self._uncommitted_ledger(track, history)
        if track_document["track_spec_digest"] != track.digest:
            raise PublicArtifactError("an active track ID cannot change its specification")

        summary = (
            None if summary_payload is None
            else _decode_canonical_object(summary_payload, "summary")
        )
        if summary is None or summary.get("track_id") != track.track_id:
            return self._uncommitted_ledger(track, history)
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
        incumbent = historical_champion(history, genesis_champion(track))
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
                    promotion=track.promotion,
                    incumbent=incumbent,
                )
            except ValueError as exc:
                raise PublicArtifactError(f"public match is invalid: {name}") from exc
            matches.append(record)
            committed_challenges.append(challenge_from_match(document))
            if self._prefix == "tracks/main":
                _require_inventory_bindings(track, document)
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
            promotion=track.promotion,
            initial_champion=genesis_champion(track),
            historical_matches=history,
        )
        if summary != _summary_document(track, source_block, tournament):
            raise PublicArtifactError("public summary does not match its immutable matches")

        matches.extend(self._load_following_matches(track, match_count + 1, incumbent))
        complete_matches = tuple(matches)
        return self._recovered_ledger(
            track, complete_matches, source_block=source_block,
            committed_count=match_count, history=history,
        )

    def _uncommitted_ledger(
        self, track: TrackSpec, history: tuple[MatchRecord, ...] = (),
    ) -> PublicLedger:
        matches = tuple(self._load_following_matches(
            track, 1, historical_champion(history, genesis_champion(track)),
        ))
        return self._recovered_ledger(
            track, matches, source_block=None, committed_count=0, history=history,
        )

    def _recovered_ledger(
        self, track: TrackSpec, matches: tuple[MatchRecord, ...], *,
        source_block: int | None, committed_count: int,
        history: tuple[MatchRecord, ...] = (),
    ) -> PublicLedger:
        payload = self._get_optional("weights.json")
        if payload is not None and (
            _decode_canonical_object(payload, "weights").get("track_id") == track.track_id
        ):
            weights = decode_weight_artifact(payload)
            if (
                weights.track_id != track.track_id or weights.track_spec_digest != track.digest
                or weights.netuid != track.netuid or weights.mechid != track.mechid
            ):
                raise PublicArtifactError("public weight fence targets a different track")
            count = weights.match_sequence
            if count > len(matches):
                raise PublicArtifactError("public weight fence is missing immutable matches")
            terminal_sha256 = (
                None if count == 0
                else hashlib.sha256(_json_bytes(matches[count - 1].document)).hexdigest()
            )
            if (
                weights.terminal_match_sha256 != terminal_sha256
                or any(match.document["decided_block"] > weights.source_block
                       for match in matches[:count])
            ):
                raise PublicArtifactError("public weight fence does not bind its immutable prefix")
            committed_count = max(committed_count, count)
            source_block = max(0 if source_block is None else source_block, weights.source_block)
        return PublicLedger(
            source_block, matches, frozenset(), committed_count, history,
        )

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
                    promotion=track.promotion,
                    incumbent=incumbent,
                )
            except ValueError as exc:
                raise PublicArtifactError(f"public match is invalid: {name}") from exc
            matches.append(record)
            incumbent = record.champion_after
            if self._prefix == "tracks/main":
                _require_inventory_bindings(track, document)
            self._confirmed_immutable[name] = hashlib.sha256(payload).hexdigest()
            sequence += 1
        return matches

    def _load_batch_disclosures(
        self,
        track: TrackSpec,
        matches: tuple[MatchRecord, ...],
    ) -> frozenset[str]:
        """Verify complete public batch groups once when recovering the ledger."""

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
            seals = {tuple(batch.get(key) for key in (
                "completed_block", "completed_at_ms", "ready_sha256",
            )) for batch in used_batches}
            if len(seals) != 1:
                raise PublicArtifactError("public matches disagree on a READY seal")
            generation_blocks[plan_digest] = blocks.pop()
        if len(set(generation_blocks.values())) != len(generation_blocks):
            raise PublicArtifactError("public matches identify multiple plans for one block")

        verified = {}
        allocations = {}
        for plan_digest, used_batches in batches.items():
            record_name = f"batch_groups/{plan_digest}.json"
            payload = self._get_optional(record_name)
            if payload is None:
                completed_at = used_batches[0].get("completed_at_ms")
                if completed_at is not None and time.time_ns() // 1_000_000 >= (
                    completed_at + track.cache.disclosure_delay_seconds * 1000
                ):
                    _LOG.error(
                        "DATA_DISCLOSURE_OVERDUE plan_digest=%s completed_at_ms=%d",
                        plan_digest, completed_at,
                    )
                continue
            document = _decode_canonical_object(payload, record_name)
            if set(document) != {
                "format",
                "plan",
                "public_dense_references",
                "public_exact_object",
                "public_selected_inputs",
            } or document["format"] != "veda-batch-disclosure-v3":
                raise PublicArtifactError("public batch disclosure fields are invalid")
            try:
                plan = BatchPlan.decode(_json_bytes(document["plan"]))
            except (TypeError, ValueError) as exc:
                raise PublicArtifactError("public batch disclosure plan is invalid") from exc
            uses = stage_uses.get(plan_digest, [])
            if (
                plan.plan_digest != plan_digest
                or plan.track_spec_digest != track.digest
                or not uses
            ):
                raise PublicArtifactError("public inventory conflicts with the ledger")
            for match, stage in uses:
                allocation = stage["evaluation_batch"].get("allocation")
                try:
                    Inventory.validate_claim(
                        allocation, plan,
                        disclosure_delay_seconds=track.cache.disclosure_delay_seconds,
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    raise PublicArtifactError("public inventory allocation is invalid") from exc
                if (
                    allocation["challenge_id"] != match.challenge_id
                    or allocation["stage"] != stage["stage"]
                    or allocation["file_sha256"] != match.document["evaluation"]["file_sha256"]
                    or allocation["model_digest"] != match.document["evaluation"]["model_digest"]
                    or allocation["submission_block"]
                    != match.document["submission"]["block_height"]
                ):
                    raise PublicArtifactError("public inventory claimant differs from its match")
                identity = (match.challenge_id, stage["stage"])
                if any(row["ordinal"] == allocation["ordinal"] and key != identity
                       for key, row in allocations.items()):
                    raise PublicArtifactError("public inventory allocation ordinal is reused")
                allocations[identity] = allocation
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
                secret=plan.secret,
                inventory_slot=plan.inventory_slot,
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
                or batch.get("completed_block") != plan.completed_block
                or batch.get("completed_at_ms") != plan.completed_at_ms
                or batch.get("ready_sha256") != hashlib.sha256(plan.encode()).hexdigest()
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
            if plan.sample_cache is None:
                raise PublicArtifactError("public READY plan omitted selected source inputs")
            public_inputs = public_selected_inputs_document(plan.sample_cache)
            for kind, item in (
                ("index", plan.sample_cache.index),
                ("text", plan.sample_cache.text),
                ("condition", plan.sample_cache.condition),
            ):
                if item is None:
                    continue
                public = public_inputs[kind]
                assert public is not None
                object_name = public["object"]
                assert isinstance(object_name, str)
                files.append((object_name, item.sha256, item.size_bytes))
            expected = {
                "format": "veda-batch-disclosure-v3",
                "plan": plan.document(),
                "public_exact_object": exact_name,
                "public_dense_references": public_references,
                "public_selected_inputs": public_inputs,
            }
            if document != expected:
                raise PublicArtifactError("public batch disclosure is not canonical")
            for object_name, sha256, size_bytes in files:
                self._verify_public_file(object_name, sha256, size_bytes)
            verified[plan_digest] = (plan, record_name, payload)

        # The stage ledger identifies both allocations even when confirmation was
        # generated before the exact decision. Prove independent source families.
        for match in matches:
            stages = match.document["evaluation"]["stages"]
            plans = [
                verified[stage["evaluation_batch"]["plan_digest"]][0]
                for stage in stages if stage.get("evaluation_batch", {}).get("plan_digest")
                in verified
            ]
            if len(plans) == 2 and (
                plans[0].inventory_slot == plans[1].inventory_slot or {
                    proof.leaf.source_family_id for proof in plans[0].source_proofs
                }.intersection(proof.leaf.source_family_id for proof in plans[1].source_proofs)
            ):
                raise PublicArtifactError("confirmation disclosure reuses exact sources")
            if len(plans) == 2 and (
                allocations[(match.challenge_id, "SCORING")]["ordinal"]
                >= allocations[(match.challenge_id, "CONFIRMING")]["ordinal"]
            ):
                raise PublicArtifactError("confirmation allocation predates exact allocation")
        prefix = []
        for plan_digest in sorted(generation_blocks, key=generation_blocks.__getitem__):
            row = verified.get(plan_digest)
            if row is not None:
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
                if is_precondition_failed(exc):
                    raise PublicArtifactError(
                        f"concurrent public pointer update was fenced: {name}"
                    ) from exc
                raise
        else:
            try:
                self._put(name, payload, False, if_none_match="*")
            except Exception as exc:
                if is_precondition_failed(exc):
                    raise PublicArtifactError(
                        f"concurrent public pointer creation was fenced: {name}"
                    ) from exc
                raise
        if self._get_optional(name) != payload:
            raise PublicArtifactError(f"public pointer failed write verification: {name}")

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
            self._guard_track_history(before, after)
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
            before_count = before_weight.match_sequence
            after_count = after_weight.match_sequence
            if (
                after_count == before_count
                and after_weight.terminal_match_sha256
                != before_weight.terminal_match_sha256
            ):
                raise PublicArtifactError("one match sequence cannot identify two histories")
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

    def _guard_track_history(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
        """A maintenance release must retain every original immutable match reference."""

        def references(pointer: Mapping[str, Any]) -> set[tuple[str, str, int]]:
            digest = pointer["spec"].get("artifact_manifest_sha256")
            payload = None if digest is None else self._get_optional(
                f"runtime/{digest}/manifest.json",
            )
            if payload is None:
                return set()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise PublicArtifactError("maintenance history manifest is invalid")
            manifest = ArtifactManifest.decode(payload)
            return {(artifact.object_key, artifact.sha256, artifact.size_bytes)
                    for artifact in manifest.artifacts
                    if artifact.logical_name.startswith(HISTORICAL_MATCHES_PREFIX)}

        required = references(before)
        track_id = before["spec"]["track_id"]
        count = 0
        summary = self._get_optional("summary.json")
        if summary is not None:
            document = _decode_canonical_object(summary, "summary")
            if document.get("track_id") == track_id:
                count = _integer(document.get("match_count"), "match_count")
        weights = self._get_optional("weights.json")
        weight = None
        if weights is not None:
            weight = decode_weight_artifact(weights)
            if weight.track_id == track_id:
                count = max(count, weight.match_sequence)
        sequence = 1
        while True:
            name = f"matches/{track_id}/{sequence:012d}.json"
            payload = self._get_optional(name)
            if payload is None:
                if sequence <= count:
                    raise PublicArtifactError("maintenance cannot drop a missing original match")
                break
            digest = hashlib.sha256(payload).hexdigest()
            if (weight is not None and weight.track_id == track_id
                    and sequence == weight.match_sequence
                    and digest != weight.terminal_match_sha256):
                raise PublicArtifactError("maintenance history differs from the weight fence")
            required.add((name, digest, len(payload)))
            sequence += 1
        if references(after) != required:
            raise PublicArtifactError(
                "maintenance cannot drop, replace or invent historical matches",
            )

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
        content_type: str = "application/json; charset=utf-8",
    ) -> None:
        if not payload or len(payload) > _MAX_PUBLIC_OBJECT_BYTES:
            raise PublicArtifactError(f"public object size is invalid: {name}")
        arguments = {
            "Bucket": self._bucket,
            "Key": f"{self._prefix}/{name}",
            "Body": payload,
            "ContentType": content_type,
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
            if is_missing(exc):
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


def _terminal_match_fence(tournament: Tournament) -> tuple[int, str | None]:
    """Bind mutable weights to the immutable terminal match in their snapshot."""

    if not tournament.matches:
        return 0, None
    terminal = tournament.matches[-1]
    if terminal.sequence != len(tournament.matches):
        raise PublicArtifactError("tournament match sequence is not contiguous")
    return terminal.sequence, hashlib.sha256(_json_bytes(terminal.document)).hexdigest()


def _status_document(
    track: TrackSpec,
    source_block: int,
    tournament: Tournament,
    competitors: list[dict[str, Any]],
    challenges: list[dict[str, Any]],
    *,
    system_matches: list[dict[str, Any]],
    historical_matches: tuple[MatchRecord, ...] = (),
) -> dict[str, Any]:
    """Build the bounded live view used by the operator monitor."""

    phase = "SCHEDULED" if source_block < track.track_start_block else "ACCEPTING"
    challenge_by_id = {item["submission_id"]: item for item in challenges}
    match_by_id = {item.challenge_id: item for item in (*historical_matches, *tournament.matches)}
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
    chain_positions: dict[str, tuple[int, int, int]] = {}
    for competitor in competitors:
        challenge = challenge_by_id.get(competitor["submission_id"])
        if challenge is None:
            raise PublicArtifactError("active commitment has no local challenge")
        chain_positions[challenge["challenge_id"]] = (
            competitor["block_height"], competitor["extrinsic_index"], competitor["event_index"],
        )
        match = match_by_id.get(challenge["challenge_id"])
        historical = (
            challenge.get("track_spec_digest") is not None
            and challenge["track_spec_digest"] != track.digest
        )
        if historical and match is not None:
            # Old results come from the authenticated immutable decision, not
            # a synthetic "historical" outcome or a newer champion observation.
            challenge = {**challenge, **challenge_from_match(match.document)}
        is_champion = tournament.champion.challenge_id == challenge["challenge_id"]
        # Stage aggregates are safe before raw reveal. Project only these
        # commitments, never arbitrary metric payloads (prompts/seeds/cases).
        stages = [{key: stage[key] for key in (
            "stage", "score", "score_block", "incumbent_score", "paired_gain_lower_bound",
            "evaluation_batch", "promotion",
        ) if key in stage} for stage in challenge.get("stages", [])]
        latest_score = stages[-1]["score"] if stages else None
        active_rows.append(
            {
                "uid": competitor["uid"],
                "hotkey": competitor["hotkey"],
                "challenge_id": challenge["challenge_id"],
                "track_spec_digest": challenge.get("track_spec_digest", track.digest),
                "submission_block": competitor["block_height"],
                "state": challenge["state"],
                "queue_position": None,
                "match_sequence": None if match is None else match.sequence,
                "score": (match.final_score if historical and match is not None else
                          tournament.champion.score if is_champion else
                          challenge["final_score"] if challenge["final_score"] is not None
                          else latest_score),
                "file_sha256": challenge.get("file_sha256"),
                "model_digest": challenge.get("model_digest"),
                "stages": stages,
                "incumbent_score": challenge["incumbent_score"],
                "score_gain": (challenge.get("paired_gain_lower_bound") if match is None
                               else match.score_gain),
                "batch_generated_block": (
                    None
                    if challenge["evaluation_batch"] is None
                    else challenge["evaluation_batch"]["generated_block"]
                ),
                "outcome": None if match is None else match.outcome,
                "champion": is_champion,
                "attempts": (
                    None if historical and challenge["created_at_ms"] == 0
                    else challenge["attempts"]
                ),
                "error_code": challenge["error_code"],
                "alert": challenge["state"] == "SYSTEM_BLOCKED",
            }
        )
    champion = tournament.champion
    original = next((match for match in historical_matches
                     if match.challenge_id == champion.challenge_id), None)
    if original is not None and not any(
        row["challenge_id"] == champion.challenge_id for row in active_rows
    ):
        evaluation = original.document["evaluation"]
        batch = evaluation["evaluation_batch"]
        active_rows.append({
            "uid": champion.uid,
            "hotkey": champion.hotkey, "challenge_id": champion.challenge_id,
            "track_spec_digest": original.document["track_spec_digest"],
            "submission_block": champion.position.block, "state": original.state,
            "queue_position": None, "match_sequence": original.sequence,
            "score": original.final_score, "incumbent_score": evaluation["incumbent_score"],
            "score_gain": original.score_gain,
            "batch_generated_block": None if batch is None else batch["generated_block"],
            "outcome": original.outcome, "champion": True, "attempts": None,
            "error_code": evaluation["error_code"], "alert": False,
        })
    waiting = [
        row
        for row in active_rows
        if row["match_sequence"] is None
        and row["state"] not in {str(state) for state in TERMINAL_STATES} | {"CHAMPION"}
    ]
    waiting.sort(key=lambda row: chain_positions[row["challenge_id"]])
    queue_positions = {
        row["challenge_id"]: position for position, row in enumerate(waiting, start=1)
    }
    for row in active_rows:
        row["queue_position"] = queue_positions.get(row["challenge_id"])
    active_rows.sort(key=lambda row: (row["uid"] is None, row["uid"] or 0))
    return {
        "track_id": track.track_id,
        "track_spec_digest": track.digest,
        "netuid": track.netuid,
        "source_block": source_block,
        "track_phase": phase,
        "champion": tournament.champion.public(),
        "system_matches": system_matches,
        "participants": active_rows,
    }


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
