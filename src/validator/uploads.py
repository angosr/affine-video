"""Private upload admission primitives, without an HTTP server or a second queue.

The hosting adapter supplies a fresh finalized registration lookup and the
existing scanned ChallengeStore. Temporary S3 writes are separated from the
existing immutable model archive. Only the latter can be used after sealing.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from miner.submit import PrivateModelSubmission, UploadIntent
from validator.chunks import ChunkManifest, verify_chunked_content
from validator.download import DownloadedModel
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
    """CAS upload quota and immutable sealing; NOT a participation ledger.

    This preparatory API is not wired into the live evaluator or exposed as a
    listener. Deployment must supply restricted staging and authority clients.
    """

    def __init__(
        self, *, objects: PrivateObjectStore, staging_client: Any,
        staging_bucket: str, staging_prefix: str, audience: str, genesis_hash: str,
        store: ChallengeStore, registration: Callable[[str], tuple[int | None, int]],
        max_bytes: int, cache_dir: str | Path,
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
        self.cache_dir = Path(cache_dir)

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
        # A lookup error propagates: no capability is issued on unknown state.
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
    def _lease_name(hotkey: str) -> str:
        return f"uploads/leases/{hashlib.sha256(hotkey.encode()).hexdigest()}.json"

    def _key(self, intent: UploadIntent) -> str:
        owner = hashlib.sha256(intent.hotkey.encode()).hexdigest()
        return f"{self.prefix}/{owner}/{intent.upload_id}"

    def _read(self, hotkey: str) -> tuple[StoredObject | None, dict[str, Any] | None]:
        stored = self.objects.get_optional(self._lease_name(hotkey))
        if stored is None:
            return None, None
        value = json.loads(stored.body)
        if (not isinstance(value, dict) or set(value) != {
            "format", "hotkey", "upload_id", "file_sha256", "size_bytes", "expires_at", "state",
            "manifest", "manifest_sha256",
        } or value["format"] != "veda-upload-lease-v1" or value["hotkey"] != hotkey
                or value["state"] not in {"UPLOADING", "SEALED"}
                or type(value["expires_at"]) is not int
                or type(value["size_bytes"]) is not int
                or not 0 < value["size_bytes"] <= self.max_bytes):
            raise ValueError("private upload quota record is invalid")
        PrivateModelSubmission(value["upload_id"], value["file_sha256"])
        manifest = ChunkManifest.decode(canonical_json(value["manifest"]))
        if (manifest.sha256 != value["file_sha256"] or manifest.size != value["size_bytes"]
                or hashlib.sha256(manifest.encode()).hexdigest() != value["manifest_sha256"]):
            raise ValueError("private upload quota manifest conflicts with its identity")
        return stored, value

    def authorize(
        self, intent: UploadIntent, signature: bytes, *, manifest: ChunkManifest, now: int,
    ) -> UploadGrant:
        """Authorize only fixed original-format chunks, checking original eligibility."""
        if intent.action != "upload":
            raise ValueError("expected an upload intent")
        self._authenticate(intent, signature, now=now)
        # The hosting adapter must bound the small JSON body before decoding it.
        manifest = ChunkManifest.decode(manifest.encode())
        if (manifest.sha256 != intent.file_sha256 or manifest.size != intent.size_bytes
                or hashlib.sha256(manifest.encode()).hexdigest() != intent.manifest_sha256):
            raise ValueError("upload manifest differs from signed intent")
        self._eligible(intent)
        stored, previous = self._read(intent.hotkey)
        identity = self._identity(intent)
        if previous is not None:
            same = all(previous[key] == value for key, value in identity.items())
            if (same and previous["state"] == "SEALED") or (
                not same and previous["expires_at"] > now
            ):
                raise ValueError("another upload is active or already sealed")
            if same and previous["expires_at"] >= intent.expires_at:
                # Exact replay may recover a lost response, never extend its lifetime.
                return self._grant(intent, manifest, now=now)
        document = canonical_json({
            "format": "veda-upload-lease-v1", **identity,
            "state": "UPLOADING", "expires_at": intent.expires_at,
            "manifest": json.loads(manifest.encode()),
        })
        if stored is None:
            self.objects.create_plan(self._lease_name(intent.hotkey), document)
        else:
            self.objects.replace_plan(self._lease_name(intent.hotkey), document, etag=stored.etag)
        return self._grant(intent, manifest, now=now)

    def _grant(self, intent: UploadIntent, manifest: ChunkManifest, *, now: int) -> UploadGrant:
        parts = []
        for part in manifest.parts:
            url = self.staging.generate_presigned_url(
                "put_object", Params={
                    "Bucket": self.bucket, "Key": f"{self._key(intent)}/{part.name}",
                    "ContentLength": part.size,
                }, ExpiresIn=intent.expires_at - now, HttpMethod="PUT",
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
        """Hash real bytes, archive once, then CAS seal. Do not trust S3 metadata.

        An old presigned URL can still overwrite staging, but can never write
        the content-addressed archive or change the sealed reference.
        """
        if intent.action != "seal":
            raise ValueError("expected a seal intent")
        self._authenticate(intent, signature, now=now)
        stored, record = self._read(intent.hotkey)
        if (stored is None or record is None or any(
            record[key] != value for key, value in self._identity(intent).items()
        )):
            raise ValueError("upload has no matching authorization")
        if record["state"] == "SEALED":
            return self._receipt(record)
        self._eligible(intent)
        manifest = ChunkManifest.decode(canonical_json(record["manifest"]))

        def open_part(name: str):
            response = self.staging.get_object(
                Bucket=self.bucket, Key=f"{self._key(intent)}/{name}",
            )
            body = response["Body"]
            expected_size = next(part.size for part in manifest.parts if part.name == name)
            if response.get("ContentLength") != expected_size:
                body.close()
                raise ValueError("private upload part length differs from signed manifest")
            return body

        # Run this verifier on the existing owner, not as a body-proxying HTTPS
        # function. Upload bytes go directly from the miner into staging S3.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".upload") as output:
            verify_chunked_content(manifest, open_part, write=output.write)
            output.flush()
            self._eligible(intent)
            self.objects.put_content(
                output.name, sha256=intent.file_sha256,
                object_name=f"models/{intent.file_sha256}.safetensors",
            )
        sealed = {**record, "state": "SEALED"}
        self.objects.replace_plan(
            self._lease_name(intent.hotkey), canonical_json(sealed), etag=stored.etag,
        )
        return self._receipt(sealed)

    def _receipt(self, record: dict[str, Any]) -> PrivateModelSubmission:
        reference = PrivateModelSubmission(record["upload_id"], record["file_sha256"])
        # Recover this exact immutable receipt if the previous response was lost.
        self.objects.create_plan(
            f"uploads/sealed/{reference.upload_id}.json", canonical_json({
                "format": "veda-sealed-upload-v1",
                **{key: record[key] for key in (
                    "hotkey", "upload_id", "file_sha256", "size_bytes",
                )},
            }),
        )
        return reference

    def download(
        self, reference: PrivateModelSubmission, *, hotkey: str, cache_dir: str | Path,
        max_bytes: int,
    ) -> DownloadedModel:
        """Source adapter to the original model cache/archive; no HF impersonation."""
        stored = self.objects.get(f"uploads/sealed/{reference.upload_id}.json")
        value = json.loads(stored.body)
        if (not isinstance(value, dict) or set(value) != {
            "format", "hotkey", "upload_id", "file_sha256", "size_bytes",
        } or value["format"] != "veda-sealed-upload-v1"
                or value["hotkey"] != hotkey or value["upload_id"] != reference.upload_id
                or value["file_sha256"] != reference.file_sha256
                or type(value["size_bytes"]) is not int
                or not 0 < value["size_bytes"] <= min(max_bytes, self.max_bytes)):
            raise ValueError("sealed upload does not match the submitting identity")
        path = self.objects.download_content(
            f"models/{reference.file_sha256}.safetensors",
            Path(cache_dir) / f"{reference.file_sha256}.safetensors",
            sha256=reference.file_sha256, size_bytes=value["size_bytes"],
        )
        return DownloadedModel(path, reference.file_sha256, value["size_bytes"], None, None)
