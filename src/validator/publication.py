"""Canonical public artifacts and S3-compatible object-storage transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from scoring.spec import TrackSpec, parse_track_spec, track_spec_dict
from validator.queue import TERMINAL_STATES
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


@dataclass(frozen=True, slots=True)
class PublicFile:
    object_name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicLedger:
    source_block: int | None
    matches: tuple[MatchRecord, ...]

    @property
    def required_finalized_block(self) -> int:
        committed = 0 if self.source_block is None else self.source_block
        decided = max(
            (int(match.document["decided_block"]) for match in self.matches),
            default=0,
        )
        return max(committed, decided)


def build_public_artifacts(
    store: ChallengeStore,
    track: TrackSpec,
    chain: WeightChainState,
    *,
    version_key: int,
    exact_cache_dir: str | Path | None = None,
) -> PublicArtifacts | None:
    """Build one auditable public snapshot when chain scanning is caught up."""

    cursor = store.chain_cursor()
    if cursor is None or cursor[0] != chain.finalized_block:
        return None
    snapshot = store.public_snapshot()
    initial = genesis_champion(track)
    tournament = resolve_tournament(
        snapshot["challenges"],
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        minimum_gain=track.scoring.minimum_frontier_gain,
        initial_champion=initial,
        minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
    )
    if any(match.document["decided_block"] > chain.finalized_block for match in tournament.matches):
        raise PublicArtifactError("a terminal decision is newer than the finalized chain snapshot")
    plan = plan_champion_weight(
        tournament.champion,
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
    )
    summary = _summary_document(track, chain.finalized_block, tournament)
    status = _status_document(
        track,
        chain.finalized_block,
        tournament,
        store.active_competitors(chain.uid_by_hotkey),
        snapshot["challenges"],
    )
    files: dict[str, PublicFile] = {}
    for match in tournament.matches:
        evaluation = match.document["evaluation"]
        batch = evaluation["evaluation_batch"]
        if batch is not None and exact_cache_dir is not None:
            path = Path(exact_cache_dir) / f"{batch['generated_block']}-{batch['sha256']}.npz"
            files[batch["object"]] = PublicFile(batch["object"], path, batch["sha256"])
        if exact_cache_dir is not None and match.outcome.value == "CHALLENGER_WON":
            path_value = store.model_path(match.challenge_id)
            if path_value is None:
                raise PublicArtifactError("winning model has no local archive source")
            object_name = evaluation["model_object"]
            files[object_name] = PublicFile(
                object_name,
                Path(path_value),
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
    )


def build_genesis_artifacts(
    track: TrackSpec,
    chain: WeightChainState,
    *,
    version_key: int,
) -> PublicArtifacts:
    """Build the complete pre-launch snapshot with UID0 as champion."""

    tournament = resolve_tournament(
        (),
        track_id=track.track_id,
        track_spec_digest=track.digest,
        netuid=track.netuid,
        minimum_gain=track.scoring.minimum_frontier_gain,
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
    try:
        track = parse_track_spec(document["spec"])
    except ValueError as exc:
        raise PublicArtifactError("public track specification is invalid") from exc
    if document["track_spec_digest"] != track.digest:
        raise PublicArtifactError("public track digest is invalid")
    return track


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
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("publishing requires the 'storage' package extra") from exc
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        return cls(client, bucket=bucket, prefix=prefix)

    async def publish(self, artifacts: PublicArtifacts) -> None:
        await asyncio.to_thread(self._put, "track.json", artifacts.track, False)
        for artifact in artifacts.files:
            await asyncio.to_thread(self._put_file_immutable, artifact)
        for name, payload in artifacts.system_matches:
            await asyncio.to_thread(self._put_immutable, name, payload)
        for name, payload in artifacts.matches:
            await asyncio.to_thread(self._put_immutable, name, payload)
        await asyncio.to_thread(self._put, "summary.json", artifacts.summary, False)
        if artifacts.status is not None:
            await asyncio.to_thread(self._put, "status.json", artifacts.status, False)
        await asyncio.to_thread(self._put, "weights.json", artifacts.weights, False)

    def _put_file_immutable(self, artifact: PublicFile) -> None:
        if not artifact.path.is_file():
            raise PublicArtifactError(f"public file source is missing: {artifact.object_name}")
        if _file_sha256(artifact.path) != artifact.sha256:
            raise PublicArtifactError(f"public file digest changed: {artifact.object_name}")
        key = f"{self._prefix}/{artifact.object_name}"
        try:
            existing = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = (response or {}).get("Error", {}).get("Code")
            if str(code) not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            metadata = existing.get("Metadata", {})
            if (
                metadata.get("sha256") != artifact.sha256
                or existing.get("ContentLength") != artifact.path.stat().st_size
            ):
                raise PublicArtifactError(
                    f"immutable public object conflicts with local content: {artifact.object_name}"
                )
            return
        self._client.upload_file(
            str(artifact.path),
            self._bucket,
            key,
            ExtraArgs={
                "ContentType": "application/octet-stream",
                "CacheControl": "public, max-age=31536000, immutable",
                "Metadata": {"sha256": artifact.sha256},
            },
        )
        uploaded = self._client.head_object(Bucket=self._bucket, Key=key)
        if (
            uploaded.get("Metadata", {}).get("sha256") != artifact.sha256
            or uploaded.get("ContentLength") != artifact.path.stat().st_size
        ):
            raise PublicArtifactError(
                f"public file upload verification failed: {artifact.object_name}"
            )

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

        if summary_payload is None:
            return PublicLedger(
                None,
                tuple(self._load_following_matches(track, 1, genesis_champion(track))),
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
                    minimum_gain=track.scoring.minimum_frontier_gain,
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
            minimum_gain=track.scoring.minimum_frontier_gain,
            initial_champion=genesis_champion(track),
            minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
        )
        if summary != _summary_document(track, source_block, tournament):
            raise PublicArtifactError("public summary does not match its immutable matches")

        matches.extend(self._load_following_matches(track, match_count + 1, incumbent))
        return PublicLedger(source_block, tuple(matches))

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
                    minimum_gain=track.scoring.minimum_frontier_gain,
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
        self._put(name, payload, True)
        self._confirmed_immutable[name] = digest

    def _put(self, name: str, payload: bytes, immutable: bool) -> None:
        if not payload or len(payload) > _MAX_PUBLIC_OBJECT_BYTES:
            raise PublicArtifactError(f"public object size is invalid: {name}")
        self._client.put_object(
            Bucket=self._bucket,
            Key=f"{self._prefix}/{name}",
            Body=payload,
            ContentType="application/json; charset=utf-8",
            CacheControl=(
                "public, max-age=31536000, immutable" if immutable else "no-store, max-age=0"
            ),
        )

    def _get_optional(self, name: str) -> bytes | None:
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=f"{self._prefix}/{name}",
            )
        except Exception as exc:
            response_data = getattr(exc, "response", None)
            code = (
                response_data.get("Error", {}).get("Code")
                if isinstance(response_data, dict)
                else None
            )
            status = (
                response_data.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if isinstance(response_data, dict)
                else None
            )
            if status == 404 or code in {"NoSuchKey", "404", "NotFound"}:
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
        return payload


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> dict[str, Any]:
    """Build the bounded live view used by the operator monitor."""

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
                "alert": (challenge["state"] == "SYSTEM_DEFERRED" and challenge["attempts"] >= 3),
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
        "champion": tournament.champion.public(),
        "system_matches": system_match_summaries(track.track_id, track.digest, track.netuid),
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
