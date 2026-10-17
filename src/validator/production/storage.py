"""Private content-addressed storage for the single evaluation-data writer."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from validator.chunks import (
    ChunkManifest,
    build_chunk_manifest,
    chunk_object_name,
    read_part,
)
from validator.production.commitment import ArtifactManifest
from validator.s3 import client_config


class StorageError(RuntimeError):
    """The private store failed or violated the required semantics."""


class PreconditionFailed(StorageError):
    """A plan changed since the single writer read it."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    body: bytes
    etag: str


class PrivateObjectStore:
    """Minimal S3 transport used by one plan writer and immutable artifacts."""

    def __init__(
        self,
        client: Any,
        *,
        bucket: str,
        prefix: str,
        reconnect_from_environment: bool = False,
    ) -> None:
        if not bucket or "/" in bucket:
            raise ValueError("private storage bucket is invalid")
        normalized = prefix.strip("/")
        if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("private storage prefix is invalid")
        self._client = client
        self._bucket = bucket
        self._prefix = normalized
        self._reconnect_from_environment = reconnect_from_environment

    @classmethod
    def from_environment(cls, *, bucket: str, prefix: str) -> PrivateObjectStore:
        private_values = (
            os.environ.get("PRIVATE_STORAGE_URL"),
            os.environ.get("PRIVATE_ACCESS_KEY"),
            os.environ.get("PRIVATE_SECRET_ACCESS_KEY"),
        )
        if any(private_values) and not all(private_values):
            raise ValueError("private storage credentials must be configured together")
        endpoint, access_key, secret_key = (
            private_values
            if all(private_values)
            else (
                os.environ.get("STORAGE_URL"),
                os.environ.get("ACCESS_KEY"),
                os.environ.get("SECRET_ACCESS_KEY"),
            )
        )
        if not endpoint or not access_key or not secret_key:
            raise ValueError(
                "PRIVATE_STORAGE_URL, PRIVATE_ACCESS_KEY, and "
                "PRIVATE_SECRET_ACCESS_KEY are required"
            )
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
        return cls(
            _environment_client(endpoint, access_key, secret_key),
            bucket=bucket,
            prefix=prefix,
            reconnect_from_environment=True,
        )

    def __getstate__(self) -> dict[str, Any]:
        if not self._reconnect_from_environment:
            raise TypeError("an injected private object-store client cannot cross processes")
        return {
            "_bucket": self._bucket,
            "_prefix": self._prefix,
            "_reconnect_from_environment": True,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        restored = type(self).from_environment(
            bucket=state["_bucket"],
            prefix=state["_prefix"],
        )
        self.__dict__.update(restored.__dict__)

    def create_plan(self, name: str, payload: bytes) -> StoredObject:
        try:
            self._put(name, payload, if_none_match="*")
        except Exception as exc:
            if not _is_precondition_failed(exc):
                raise
            existing = self.get(name)
            if existing.body != payload:
                raise PreconditionFailed("plan already exists with different bytes") from exc
            return existing
        stored = self.get(name)
        if stored.body != payload:
            raise StorageError("created plan does not match the submitted bytes")
        return stored

    def replace_plan(self, name: str, payload: bytes, *, etag: str) -> StoredObject:
        try:
            self._put(name, payload, if_match=etag)
        except Exception as exc:
            if _is_precondition_failed(exc):
                raise PreconditionFailed("plan changed since it was read") from exc
            raise
        stored = self.get(name)
        if stored.body != payload:
            raise StorageError("updated plan does not match the submitted bytes")
        return stored

    def verify_conditional_writes(self) -> None:
        """Refuse a backend that silently ignores the plan CAS contract."""

        name = f"capability-probes/{uuid.uuid4().hex}.json"
        key = self._key(name)
        try:
            self._put(name, b'{"state":"first"}', if_none_match="*")
            first = self.get(name)
            try:
                self._put(name, b'{"state":"overwrite"}', if_none_match="*")
            except Exception as exc:
                if not _is_precondition_failed(exc):
                    raise
            else:
                raise StorageError("object store ignores If-None-Match")
            if self.get(name).body != first.body:
                raise StorageError("failed conditional create changed the stored object")
            try:
                self._put(name, b'{"state":"stale"}', if_match="invalid-etag")
            except Exception as exc:
                if not _is_precondition_failed(exc):
                    raise
            else:
                raise StorageError("object store ignores If-Match")
            self._put(name, b'{"state":"ready"}', if_match=first.etag)
            if self.get(name).body != b'{"state":"ready"}':
                raise StorageError("conditional replacement was not durable")
        finally:
            self._client.delete_object(Bucket=self._bucket, Key=key)

    def get(self, name: str) -> StoredObject:
        response = self._client.get_object(Bucket=self._bucket, Key=self._key(name))
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise StorageError("object store returned an invalid body")
        payload = body.read()
        if not isinstance(payload, bytes):
            raise StorageError("object store returned a non-byte body")
        etag = _etag(response.get("ETag"))
        return StoredObject(payload, etag)

    def get_optional(self, name: str) -> StoredObject | None:
        """Read a small object, returning None only for a definite missing key."""

        try:
            return self.get(name)
        except Exception as exc:
            if _is_missing(exc) or isinstance(exc, KeyError):
                return None
            raise

    def ensure_bootstrap(self, name: str, local_path: str | Path) -> bytes:
        """Restore a small authoritative bootstrap object or seed it once."""

        stored = self.get_optional(name)
        if stored is not None:
            return stored.body
        path = Path(local_path)
        if not path.is_file():
            raise StorageError(f"private bootstrap object is unavailable: {name}")
        payload = path.read_bytes()
        if not payload:
            raise StorageError(f"private bootstrap object is empty: {name}")
        return self.create_plan(name, payload).body

    def ensure_artifacts(self, manifest: ArtifactManifest, root: str | Path) -> None:
        """Mirror committed runtime bytes and restore any missing local artifact."""

        base = Path(root)
        base.mkdir(parents=True, exist_ok=True)
        for artifact in manifest.artifacts:
            path = base / artifact.logical_name
            local_valid = (
                path.is_file()
                and path.stat().st_size == artifact.size_bytes
                and _file_sha256(path) == artifact.sha256
            )
            if local_valid:
                self.put_content(
                    path,
                    sha256=artifact.sha256,
                    object_name=artifact.object_key,
                )
            else:
                self.download_content(
                    artifact.object_key,
                    path,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                )
        manifest.verify_local(base)

    def put_content(self, path: str | Path, *, sha256: str, object_name: str) -> None:
        """Upload one immutable file as storage-limit-safe verified chunks."""

        source = Path(path)
        if not source.is_file():
            raise StorageError("content source is missing or its SHA-256 is wrong")
        manifest = build_chunk_manifest(source)
        if manifest.sha256 != sha256:
            raise StorageError("content source is missing or its SHA-256 is wrong")
        existing_manifest = self.get_optional(object_name)
        manifest_exists = existing_manifest is not None
        if existing_manifest is not None:
            try:
                stored = ChunkManifest.decode(existing_manifest.body)
            except ValueError as exc:
                raise StorageError("immutable object manifest is invalid") from exc
            if stored != manifest:
                raise StorageError("immutable object conflicts with existing content")
        for index, part in enumerate(manifest.parts):
            name = chunk_object_name(object_name, part.name)
            key = self._key(name)
            try:
                existing = self._client.head_object(Bucket=self._bucket, Key=key)
            except Exception as exc:
                if not _is_missing(exc):
                    raise
            else:
                if (
                    existing.get("Metadata", {}).get("sha256") != part.sha256
                    or existing.get("ContentLength") != part.size
                ):
                    raise StorageError("immutable chunk conflicts with existing content")
                continue
            content = read_part(source, index, part.size)
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=content,
                ContentType="application/octet-stream",
                Metadata={"sha256": part.sha256},
            )
            stored = self._client.head_object(Bucket=self._bucket, Key=key)
            if (
                stored.get("Metadata", {}).get("sha256") != part.sha256
                or stored.get("ContentLength") != part.size
            ):
                raise StorageError("immutable chunk upload verification failed")
        if not manifest_exists:
            self.create_plan(object_name, manifest.encode())

    def list_names(self, prefix: str) -> tuple[str, ...]:
        """List a small private index with strong-consistency assumptions."""

        relative = prefix.strip("/")
        query = self._key(relative)
        token: str | None = None
        names: list[str] = []
        while True:
            arguments: dict[str, Any] = {"Bucket": self._bucket, "Prefix": query}
            if token is not None:
                arguments["ContinuationToken"] = token
            response = self._client.list_objects_v2(**arguments)
            for item in response.get("Contents", []):
                key = item.get("Key")
                root = f"{self._prefix}/"
                if not isinstance(key, str) or not key.startswith(root):
                    raise StorageError("object store returned a key outside the private prefix")
                names.append(key[len(root) :])
            if not response.get("IsTruncated", False):
                break
            token = response.get("NextContinuationToken")
            if not isinstance(token, str) or not token:
                raise StorageError("truncated object listing omitted its continuation token")
        if len(set(names)) != len(names):
            raise StorageError("object store listing contains duplicate keys")
        return tuple(sorted(names))

    def download_content(
        self,
        object_name: str,
        destination: str | Path,
        *,
        sha256: str,
        size_bytes: int,
    ) -> Path:
        """Restore immutable content through a checked atomic local rename."""

        if size_bytes <= 0:
            raise ValueError("content size must be positive")
        target = Path(destination)
        if (
            target.is_file()
            and target.stat().st_size == size_bytes
            and _file_sha256(target) == sha256
        ):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            stored = self.get(object_name)
            try:
                manifest = ChunkManifest.decode(stored.body)
            except ValueError as exc:
                raise StorageError("immutable content manifest is invalid") from exc
            if manifest.sha256 != sha256 or manifest.size != size_bytes:
                raise StorageError("immutable content manifest conflicts with expected identity")
            complete = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".part", delete=False
            ) as out:
                temporary = Path(out.name)
                for part in manifest.parts:
                    response = self._client.get_object(
                        Bucket=self._bucket,
                        Key=self._key(chunk_object_name(object_name, part.name)),
                    )
                    body = response.get("Body")
                    content = body.read(part.size + 1) if hasattr(body, "read") else None
                    if (
                        not isinstance(content, bytes)
                        or len(content) != part.size
                        or hashlib.sha256(content).hexdigest() != part.sha256
                    ):
                        raise StorageError("restored immutable chunk is invalid")
                    complete.update(content)
                    out.write(content)
                out.flush()
                os.fsync(out.fileno())
            if temporary.stat().st_size != size_bytes or complete.hexdigest() != sha256:
                raise StorageError("restored content failed size or SHA-256 verification")
            os.replace(temporary, target)
            return target
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _put(
        self,
        name: str,
        payload: bytes,
        *,
        if_none_match: str | None = None,
        if_match: str | None = None,
    ) -> None:
        if not payload:
            raise ValueError("private object payload cannot be empty")
        arguments: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": self._key(name),
            "Body": payload,
            "ContentType": "application/json; charset=utf-8",
        }
        if if_none_match is not None:
            arguments["IfNoneMatch"] = if_none_match
        if if_match is not None:
            arguments["IfMatch"] = if_match
        self._client.put_object(**arguments)

    def _key(self, name: str) -> str:
        normalized = name.strip("/")
        if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("private object name is invalid")
        return f"{self._prefix}/{normalized}"


def _environment_client(endpoint: str, access_key: str, secret_key: str) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("private storage requires the 'storage' package extra") from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=client_config(),
    )


def _etag(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise StorageError("object store response omitted its ETag")
    return value.strip('"')


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
