"""Private upload metadata admission and the original ADMISSION source adapter.

The thin HTTP adapter uses only authorize/seal. It never handles model bodies
or writes the models archive. Real byte verification happens in download,
called by the existing admission worker before its ordinary schema/owner gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import ceil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from miner.submit import PrivateModelSubmission, UploadIntent
from scoring.model import model_file_sha256
from validator.chunks import ChunkManifest, verify_chunked_content
from validator.download import DownloadedModel, ModelDownloadError
from validator.production.commitment import canonical_json
from validator.production.storage import (
    PrivateObjectStore,
    StoredObject,
    _environment_client,
    _validated_endpoint,
)
from validator.store import ChallengeStore


@dataclass(frozen=True, slots=True)
class UploadPartGrant:
    """A retryable write to one original-format part; never log its bearer URL."""

    name: str
    size_bytes: int
    url: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class UploadGrant:
    """Only the fixed parts from the authenticated original chunk manifest."""

    upload_id: str
    expires_at: int
    parts: tuple[UploadPartGrant, ...]


class PrivateUploads:
    """One CAS upload record plus a hotkey quota pointer, NOT an opportunity ledger.

    The host must supply restricted staging/authority clients and
    a finalized SN120 registration lookup, never a client-provided UID.
    """

    def __init__(
        self, *, objects: PrivateObjectStore, staging_client: Any,
        staging_bucket: str, staging_prefix: str, audience: str, genesis_hash: str,
        store: ChallengeStore, registration: Callable[[str], tuple[int | None, int]] | None,
        max_bytes: int,
    ) -> None:
        if (not staging_bucket or "/" in staging_bucket or staging_bucket == "affine_video"
                or not staging_prefix or any(
                    part in {"", ".", ".."} for part in staging_prefix.split("/")
                ) or max_bytes <= 0):
            raise ValueError("private staging scope or byte limit is invalid")
        self.objects = objects
        self.staging = staging_client
        self.bucket = staging_bucket
        self.prefix = staging_prefix
        self.audience = audience
        self.genesis_hash = genesis_hash
        self.store = store
        self.registration = registration
        self.max_bytes = max_bytes
        self._gc_after = ""

    @classmethod
    def from_environment(
        cls, *, store: ChallengeStore, max_bytes: int,
        registration: Callable[[str], tuple[int | None, int]] | None = None,
        required: bool = False,
    ) -> PrivateUploads | None:
        """Use explicit restricted credentials, never publisher/admin fallbacks.

        Each process receives its own role values under these same names.
        HTTP needs staging PUT/HEAD; evaluator needs GET/exact DELETE. Neither
        changes the original models writer or per-track private backend.
        Staging is R2 only; its explicit bucket and auto signing region do not
        change the authority/public clients or AWS_DEFAULT_REGION.
        """
        names = (
            "AUDIENCE", "GENESIS_HASH", "AUTHORITY_BUCKET", "STAGING_BUCKET",
            "AUTHORITY_STORAGE_URL", "AUTHORITY_ACCESS_KEY", "AUTHORITY_SECRET_ACCESS_KEY",
            "STAGING_STORAGE_URL", "STAGING_ACCESS_KEY", "STAGING_SECRET_ACCESS_KEY",
        )
        values = tuple(os.environ.get(f"VEDA_UPLOAD_{name}") for name in names)
        if not any(values) and not required:
            return None
        if not all(values):
            raise ValueError("all VEDA_UPLOAD domain and restricted storage settings are required")
        (audience, genesis, bucket, staging_bucket, authority_url, authority_key, authority_secret,
         staging_url, staging_key, staging_secret) = values
        if bucket == "affine_video":
            raise ValueError("upload authority cannot use the public bucket")
        # Validate before constructing any clients; do not log settings/credentials.
        UploadIntent(audience, genesis, "configuration", 120, "upload", "00" * 32,
                     "00" * 32, 1, "00" * 32, 1)
        authority_url = _validated_endpoint(authority_url)
        if not re.fullmatch(
            r"https://[a-f0-9]{32}(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com/?",
            staging_url,
        ):
            raise ValueError("upload staging requires an R2 S3 API endpoint")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", staging_bucket):
            raise ValueError("R2 staging bucket name is invalid")
        return cls(
            objects=PrivateObjectStore(
                _environment_client(authority_url, authority_key, authority_secret),
                bucket=bucket, prefix="evaluation",
            ),
            staging_client=_environment_client(
                staging_url, staging_key, staging_secret, region_name="auto",
            ),
            staging_bucket=staging_bucket, staging_prefix="uploads/v1",
            audience=audience, genesis_hash=genesis, store=store,
            registration=registration, max_bytes=max_bytes,
        )

    def _authenticate(self, intent: UploadIntent, signature: bytes, *, now: int) -> None:
        if (intent.audience != self.audience or intent.genesis_hash != self.genesis_hash
                or not 0 < intent.expires_at - now <= 300
                or intent.size_bytes > self.max_bytes or len(signature) != 64):
            raise ValueError("upload signature domain, expiry or size is invalid")
        from bittensor.wallet import Keypair

        try:
            verified = Keypair(ss58_address=intent.hotkey, crypto_type=0).verify(
                intent.signing_message(), signature,
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("upload hotkey signature is invalid") from exc
        if not verified:
            raise ValueError("upload hotkey signature is invalid")

    def _eligible(
        self, intent: UploadIntent, *, reference: PrivateModelSubmission | None = None,
    ) -> None:
        if self.registration is None:
            raise RuntimeError("upload authorization requires a finalized registration lookup")
        uid, finalized_block = self.registration(intent.hotkey)
        self.store.require_upload_eligible(
            intent.hotkey, uid=uid, finalized_block=finalized_block, reference=reference,
        )

    @staticmethod
    def _identity(intent: UploadIntent) -> dict[str, object]:
        return {
            "hotkey": intent.hotkey, "upload_id": intent.upload_id,
            "file_sha256": intent.file_sha256, "size_bytes": intent.size_bytes,
            "manifest_sha256": intent.manifest_sha256,
        }

    @staticmethod
    def _record_name(upload_id: str) -> str:
        return f"uploads/{upload_id}.json"

    @staticmethod
    def _quota_name(hotkey: str) -> str:
        return f"uploads/hotkeys/{hashlib.sha256(hotkey.encode()).hexdigest()}.json"

    def _key(self, intent: UploadIntent) -> str:
        owner = hashlib.sha256(intent.hotkey.encode()).hexdigest()
        return f"{self.prefix}/{owner}/{intent.upload_id}"

    def _read(self, upload_id: str) -> tuple[StoredObject | None, dict[str, Any] | None]:
        stored = self.objects.get_optional(self._record_name(upload_id))
        if stored is None:
            return None, None
        value = json.loads(stored.body)
        fields = {
            "format", "hotkey", "upload_id", "file_sha256", "size_bytes", "expires_at", "state",
            "manifest", "manifest_sha256",
        }
        if (not isinstance(value, dict) or set(value) not in (fields, fields | {"cleanup"})
                or ("cleanup" in value and value["cleanup"] not in {"pending", "done"})
                or value["format"] != "veda-upload-v1" or value["upload_id"] != upload_id
                or not isinstance(value["hotkey"], str) or not value["hotkey"]
                or value["state"] not in {"UPLOADING", "SEALED"}
                or type(value["expires_at"]) is not int
                or type(value["size_bytes"]) is not int
                or not 0 < value["size_bytes"] <= self.max_bytes):
            raise ValueError("private upload record is invalid")
        PrivateModelSubmission(value["upload_id"], value["file_sha256"])
        manifest = ChunkManifest.decode(canonical_json(value["manifest"]))
        if (manifest.sha256 != value["file_sha256"] or manifest.size != value["size_bytes"]
                or hashlib.sha256(manifest.encode()).hexdigest() != value["manifest_sha256"]):
            raise ValueError("private upload manifest conflicts with its identity")
        return stored, value

    def _claim_quota(self, intent: UploadIntent, *, now: int) -> None:
        name = self._quota_name(intent.hotkey)
        stored = self.objects.get_optional(name)
        expires = intent.expires_at
        if stored is not None:
            value = json.loads(stored.body)
            if (not isinstance(value, dict) or set(value) != {
                "hotkey", "upload_id", "expires_at",
            } or value["hotkey"] != intent.hotkey or type(value["expires_at"]) is not int):
                raise ValueError("private upload quota pointer is invalid")
            PrivateModelSubmission(value["upload_id"], intent.file_sha256)
            if value["upload_id"] != intent.upload_id and value["expires_at"] > now:
                raise ValueError("another upload is active")
            if value["upload_id"] == intent.upload_id and value["expires_at"] >= expires:
                return
        document = canonical_json({
            "hotkey": intent.hotkey, "upload_id": intent.upload_id, "expires_at": expires,
        })
        if stored is None:
            self.objects.create_plan(name, document)
        else:
            self.objects.replace_plan(name, document, etag=stored.etag)

    def authorize(
        self, intent: UploadIntent, signature: bytes, *, manifest: ChunkManifest, now: int,
    ) -> UploadGrant:
        started = time.monotonic()
        if intent.action != "upload":
            raise ValueError("expected an upload intent")
        self._authenticate(intent, signature, now=now)
        manifest = ChunkManifest.decode(manifest.encode())
        if (manifest.sha256 != intent.file_sha256 or manifest.size != intent.size_bytes
                or hashlib.sha256(manifest.encode()).hexdigest() != intent.manifest_sha256):
            raise ValueError("upload manifest differs from signed intent")
        stored, previous = self._read(intent.upload_id)
        if previous is not None and previous.get("cleanup") == "pending":
            raise RuntimeError("original upload cleanup must finish before issuing permissions")
        identity = self._identity(intent)
        if previous is not None and any(
            previous[key] != value for key, value in identity.items()
        ):
            raise ValueError("upload identity is already fixed")
        self._eligible(intent, reference=(
            PrivateModelSubmission(intent.upload_id, intent.file_sha256)
            if previous is not None and previous["state"] == "SEALED" else None
        ))
        self._claim_quota(intent, now=now)
        if previous is None or previous["expires_at"] < intent.expires_at:
            document = canonical_json({
                "format": "veda-upload-v1", **identity,
                # Sealing fixes the file/parts, not permission to repair those SAME bytes.
                "state": "UPLOADING" if previous is None else previous["state"],
                "expires_at": intent.expires_at,
                "manifest": json.loads(manifest.encode()),
            })
            if stored is None:
                self.objects.create_plan(self._record_name(intent.upload_id), document)
            else:
                self.objects.replace_plan(
                    self._record_name(intent.upload_id), document, etag=stored.etag,
                )
        parts = []
        for part in manifest.parts:
            remaining = intent.expires_at - now - ceil(time.monotonic() - started)
            if remaining <= 0:
                raise ValueError("upload intent expired before capability issuance")
            url = self.staging.generate_presigned_url(
                "put_object", Params={
                    "Bucket": self.bucket, "Key": f"{self._key(intent)}/{part.name}",
                    "ContentLength": part.size,
                }, ExpiresIn=remaining, HttpMethod="PUT",
            )
            parsed = urlparse(url)
            if (parsed.scheme != "https" or not parsed.netloc
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError("private upload capability requires HTTPS")
            parts.append(UploadPartGrant(part.name, part.size, url))
        return UploadGrant(intent.upload_id, intent.expires_at, tuple(parts))

    def seal(
        self, intent: UploadIntent, signature: bytes, *, now: int,
    ) -> PrivateModelSubmission:
        """Freeze metadata, not acceptance. HEAD confirms only transport completion."""
        if intent.action != "seal":
            raise ValueError("expected a seal intent")
        self._authenticate(intent, signature, now=now)
        stored, record = self._read(intent.upload_id)
        if (stored is None or record is None or any(
            record[key] != value for key, value in self._identity(intent).items()
        )):
            raise ValueError("upload has no matching authorization")
        if record.get("cleanup") == "pending":
            raise RuntimeError("original upload cleanup is still pending")
        reference = PrivateModelSubmission(intent.upload_id, intent.file_sha256)
        if record["state"] == "SEALED":
            return reference
        self._eligible(intent)
        manifest = ChunkManifest.decode(canonical_json(record["manifest"]))
        for part in manifest.parts:
            observed = self.staging.head_object(
                Bucket=self.bucket, Key=f"{self._key(intent)}/{part.name}",
            )
            if observed.get("ContentLength") != part.size:
                raise ValueError("upload part is not complete")
        self.objects.replace_plan(
            self._record_name(intent.upload_id), canonical_json({**record, "state": "SEALED"}),
            etag=stored.etag,
        )
        return reference

    def download(
        self, reference: PrivateModelSubmission, *, hotkey: str, cache_dir: str | Path,
        max_bytes: int,
    ) -> DownloadedModel:
        """ADMISSION source adapter: verify actual bytes once, then normal validation."""
        try:
            _, record = self._read(reference.upload_id)
        except Exception as exc:
            raise ModelDownloadError(
                "PRIVATE_UPLOAD_AUTHORITY_UNAVAILABLE", "private source cannot be verified",
                temporary=True,
            ) from exc
        if (record is None or record["state"] != "SEALED"
                or record.get("cleanup") == "pending"):
            raise ModelDownloadError(
                "PRIVATE_UPLOAD_UNAVAILABLE", "private source is not available",
                temporary=True,
            )
        if record["hotkey"] != hotkey or record["file_sha256"] != reference.file_sha256:
            raise ModelDownloadError(
                "PRIVATE_UPLOAD_BINDING_MISMATCH", "private source is not sealed for this hotkey",
            )
        if record["size_bytes"] > max_bytes:
            raise ModelDownloadError("MODEL_TOO_LARGE", "private source exceeds the byte limit")
        manifest = ChunkManifest.decode(canonical_json(record["manifest"]))
        target = Path(cache_dir) / f"{reference.file_sha256}.safetensors"
        intent_key = (
            f"{self.prefix}/{hashlib.sha256(hotkey.encode()).hexdigest()}/{reference.upload_id}"
        )

        def open_part(name: str):
            response = self.staging.get_object(
                Bucket=self.bucket, Key=f"{intent_key}/{name}",
            )
            return response["Body"]

        temporary: Path | None = None
        try:
            if (target.is_file() and target.stat().st_size == manifest.size
                    and model_file_sha256(target) == reference.file_sha256):
                return DownloadedModel(target, reference.file_sha256, manifest.size, None, None)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".part", delete=False,
            ) as out:
                temporary = Path(out.name)
                verify_chunked_content(manifest, open_part, write=out.write)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, target)
            temporary = None
        except ValueError as exc:
            raise ModelDownloadError(
                "PRIVATE_UPLOAD_HASH_MISMATCH", "private source failed byte verification",
                temporary=True,
            ) from exc
        except Exception as exc:
            raise ModelDownloadError(
                "PRIVATE_UPLOAD_UNAVAILABLE", "private source is temporarily unavailable",
                temporary=True,
            ) from exc
        finally:
            active_error = sys.exc_info()[0] is not None
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    if not active_error:
                        raise ModelDownloadError(
                            "PRIVATE_UPLOAD_UNAVAILABLE", "private source cleanup failed",
                            temporary=True,
                        ) from exc
        return DownloadedModel(target, reference.file_sha256, manifest.size, None, None)

    def reclaim_expired(
        self, *, store: ChallengeStore, finalized_block: int,
        finalized_timestamp: int, now: int,
    ) -> int:
        """One upload per call; original facts and cap expiry protect live references.

        Only evaluator calls this, after a real finalized timestamp/scan fence.
        The pending marker in the SAME CAS record excludes permission issuance
        during DELETE, and resumes exact-key cleanup after a crash. Metadata,
        SEALED identity and original chain/opportunity facts are never deleted.
        """
        def page(after: str | None) -> tuple[str, ...]:
            return tuple(name for name in self.objects.list_names("uploads/", limit=16, after=after)
                         if name.startswith("uploads/") and name.count("/") == 1
                         and name.endswith(".json"))

        names = page(self._gc_after or None)
        if not names and self._gc_after:
            self._gc_after = ""
            names = page(None)
        for name in names:
            self._gc_after = name
            upload_id = name.removeprefix("uploads/").removesuffix(".json")
            stored, record = self._read(upload_id)
            if (record is None or stored is None or record.get("cleanup") == "done"
                    or record["expires_at"] >= min(now, finalized_timestamp)):
                continue
            reference = PrivateModelSubmission(upload_id, record["file_sha256"])
            if store.upload_is_referenced(
                record["hotkey"], reference=reference, finalized_block=finalized_block,
            ):
                if record.get("cleanup") == "pending":
                    original = {key: value for key, value in record.items() if key != "cleanup"}
                    self.objects.replace_plan(name, canonical_json(original), etag=stored.etag)
                continue
            if record.get("cleanup") != "pending":
                record = {**record, "cleanup": "pending"}
                stored = self.objects.replace_plan(name, canonical_json(record), etag=stored.etag)
            # Recheck after the CAS fence; a newly observed in-flight S keeps its bytes.
            if store.upload_is_referenced(
                record["hotkey"], reference=reference, finalized_block=finalized_block,
            ):
                original = {key: value for key, value in record.items() if key != "cleanup"}
                self.objects.replace_plan(name, canonical_json(original), etag=stored.etag)
                return 0
            root = (f"{self.prefix}/{hashlib.sha256(record['hotkey'].encode()).hexdigest()}"
                    f"/{upload_id}")
            manifest = ChunkManifest.decode(canonical_json(record["manifest"]))
            for part in manifest.parts:
                self.staging.delete_object(Bucket=self.bucket, Key=f"{root}/{part.name}")
            self.objects.replace_plan(
                name, canonical_json({**record, "cleanup": "done"}), etag=stored.etag,
            )
            return len(manifest.parts)
        return 0


def serve_uploads(*, database: str | Path, track_spec: str | Path, network: str, port: int) -> None:
    """Sole CPU metadata entry, same-host read-only DB; TLS remains the managed proxy."""
    import bittensor as bt

    from scoring.spec import load_track_spec

    track = load_track_spec(track_spec)
    if track.netuid != 120:
        raise ValueError("private upload service is restricted to SN120")

    async def registration(hotkey: str) -> tuple[int | None, int]:
        async with asyncio.timeout(10), bt.Client(network) as client:
            genesis = (await client.block_info(0)).hash
            if not isinstance(genesis, str) or genesis.removeprefix("0x") != uploads.genesis_hash:
                raise RuntimeError("upload registration network differs from its signature domain")
            finalized = await client.finalized_block()
            snapshot = await client.at(finalized)
            uid = await snapshot.read("uid", hotkey_ss58=hotkey, netuid=120)
            return uid, finalized

    with ChallengeStore.readonly(database) as store:
        uploads = PrivateUploads.from_environment(
            store=store, max_bytes=track.model.max_bytes, required=True,
            registration=lambda hotkey: asyncio.run(registration(hotkey)),
        )
        assert uploads is not None
        with upload_http_server(uploads, port=port) as server:
            server.serve_forever()


def upload_http_server(
    uploads: PrivateUploads, *, port: int, clock: Callable[[], float] = time.time,
) -> HTTPServer:
    """Build the sole bounded metadata adapter, bound ONLY to IPv4 loopback.

    The owner constructs this and its read-only SQLite connection in the same
    serving thread. TLS belongs to the existing external HTTPS reverse proxy.
    Creating this object does not start a serving loop or another worker.
    """
    class Handler(BaseHTTPRequestHandler):
        # StreamRequestHandler.setup applies this before any request-line/header read.
        timeout = 10

        def log_message(self, *_args: object) -> None:
            # Do not put request bodies, signed URLs or signature tokens in logs.
            pass

        def do_POST(self) -> None:
            self.close_connection = True
            code = 400
            document: dict[str, object] = {"error": "INVALID_UPLOAD_REQUEST"}
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if (self.path != (urlparse(uploads.audience).path or "/")
                        or self.headers.get("Transfer-Encoding") is not None
                        or len(lengths) != 1 or not lengths[0].isdigit()
                        or not 0 < int(lengths[0]) <= 64 * 1024
                        or self.headers.get_content_type() != "application/json"):
                    raise TypeError("invalid metadata envelope")
                payload = self.rfile.read(int(lengths[0]))
                if len(payload) != int(lengths[0]):
                    raise TypeError("incomplete metadata envelope")
                request = json.loads(payload)
                if not isinstance(request, dict) or not isinstance(request.get("intent"), dict):
                    raise TypeError("invalid metadata envelope")
                intent = UploadIntent(**request["intent"])
                expected = {"intent", "signature"} | (
                    {"manifest"} if intent.action == "upload" else set()
                )
                if set(request) != expected or not isinstance(request["signature"], str):
                    raise TypeError("invalid metadata envelope")
                signature = bytes.fromhex(request["signature"])
                if intent.action == "upload":
                    manifest = ChunkManifest.decode(canonical_json(request["manifest"]))
                    grant = uploads.authorize(
                        intent, signature, manifest=manifest, now=int(clock()),
                    )
                    # Bearer URLs go only in the authenticated HTTP response, not logs/disk.
                    document = {
                        "upload_id": grant.upload_id, "expires_at": grant.expires_at,
                        "parts": [{"name": part.name, "size_bytes": part.size_bytes,
                                   "url": part.url} for part in grant.parts],
                    }
                else:
                    reference = uploads.seal(intent, signature, now=int(clock()))
                    document = {"upload_id": reference.upload_id,
                                "file_sha256": reference.file_sha256,
                                "sealed": True, "accepted": False}
                code = 200
            except (TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
                pass
            except ValueError:
                code, document = 403, {"error": "UPLOAD_DENIED"}
            except Exception:
                code, document = 503, {"error": "UPLOAD_UNAVAILABLE"}
            encoded = canonical_json(document)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

    return HTTPServer(("127.0.0.1", port), Handler)
