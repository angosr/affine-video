"""Private upload metadata admission and the original ADMISSION source adapter.

The thin HTTP adapter uses only authorize/seal. It never handles model bodies
or writes the models archive. Real byte verification happens in download,
called by the existing admission worker before its ordinary schema/owner gate.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from validator.production.storage import PrivateObjectStore, StoredObject
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

    This preparatory API is not exposed as a listener or wired into the live
    evaluator. The host must supply restricted staging/authority clients and
    a finalized SN120 registration lookup, never a client-provided UID.
    """

    def __init__(
        self, *, objects: PrivateObjectStore, staging_client: Any,
        staging_bucket: str, staging_prefix: str, audience: str, genesis_hash: str,
        store: ChallengeStore, registration: Callable[[str], tuple[int | None, int]],
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

    def _eligible(self, intent: UploadIntent) -> None:
        uid, finalized_block = self.registration(intent.hotkey)
        self.store.require_upload_eligible(
            intent.hotkey, uid=uid, finalized_block=finalized_block,
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
        if (not isinstance(value, dict) or set(value) != {
            "format", "hotkey", "upload_id", "file_sha256", "size_bytes", "expires_at", "state",
            "manifest", "manifest_sha256",
        } or value["format"] != "veda-upload-v1" or value["upload_id"] != upload_id
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
        self._eligible(intent)
        stored, previous = self._read(intent.upload_id)
        identity = self._identity(intent)
        if previous is not None:
            if any(previous[key] != value for key, value in identity.items()):
                raise ValueError("upload identity is already fixed")
            if previous["state"] == "SEALED":
                raise ValueError("upload is already sealed")
        self._claim_quota(intent, now=now)
        if previous is None or previous["expires_at"] < intent.expires_at:
            document = canonical_json({
                "format": "veda-upload-v1", **identity,
                "state": "UPLOADING", "expires_at": intent.expires_at,
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
        if record is None or record["state"] != "SEALED":
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


def upload_http_server(
    uploads: PrivateUploads, *, port: int, clock: Callable[[], float] = time.time,
) -> HTTPServer:
    """Build the sole bounded metadata adapter, bound ONLY to IPv4 loopback.

    The owner constructs this and its read-only SQLite connection in the same
    serving thread. TLS belongs to the existing external HTTPS reverse proxy.
    Creating this object does not start a serving loop or another worker.
    """
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            # Do not put request bodies, signed URLs or signature tokens in logs.
            pass

        def do_POST(self) -> None:
            self.close_connection = True
            self.connection.settimeout(10)
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
