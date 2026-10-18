"""Private content-addressed storage for the single evaluation-data writer."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any
from urllib.parse import urlparse

from validator.chunks import (
    CHUNK_BYTES,
    ChunkManifest,
    build_chunk_manifest,
    chunk_object_name,
    read_part,
    verify_chunked_content,
)
from validator.production.commitment import Artifact
from validator.s3 import client_config

ADMISSIONS_NAMESPACE = "admissions"
ARTIFACTS_NAMESPACE = "artifacts"
CAPABILITY_PROBES_NAMESPACE = "capability-probes"
EXACT_NAMESPACE = "exact"
INPUTS_NAMESPACE = "inputs"
LATENTS_NAMESPACE = "latents"
MODELS_NAMESPACE = "models"
PLANS_NAMESPACE = "plans"
RETRY_STATE_NAMESPACE = "retry_state"
STAGES_NAMESPACE = "stages"
VIDEOS_NAMESPACE = "videos"
ARTIFACT_MANIFEST_OBJECT = "bootstrap/artifact_manifest.json"
SOURCE_POOL_OBJECT = "bootstrap/source_pool.json"
_CONTENT_TIMESTAMP_SETTLE_NS = 1_000_000_000

_EVALUATOR_LIST_NAMESPACES = (PLANS_NAMESPACE,)
_EVALUATOR_READ_NAMESPACES = (
    ADMISSIONS_NAMESPACE,
    CAPABILITY_PROBES_NAMESPACE,
    "disclosed",
    "interruptions",
    MODELS_NAMESPACE,
    PLANS_NAMESPACE,
    RETRY_STATE_NAMESPACE,
    STAGES_NAMESPACE,
)
_EVALUATOR_WRITE_NAMESPACES = (
    ADMISSIONS_NAMESPACE,
    CAPABILITY_PROBES_NAMESPACE,
    "disclosed",
    MODELS_NAMESPACE,
    RETRY_STATE_NAMESPACE,
    STAGES_NAMESPACE,
)
_EVALUATOR_DELETE_NAMESPACES = (CAPABILITY_PROBES_NAMESPACE,)
_EVALUATOR_READ_OBJECTS = (ARTIFACT_MANIFEST_OBJECT,)
_EVALUATOR_DENIED_OBJECTS = (SOURCE_POOL_OBJECT,)
_EVALUATOR_DENIED_NAMESPACES = ("audit", "source-pool")


def evaluator_iam_policy(*, bucket: str, prefix: str) -> dict[str, object]:
    """Build the exact least-privilege policy from runtime object namespaces."""

    if not bucket or "/" in bucket:
        raise ValueError("private storage bucket is invalid")
    root = _normalize_prefix(prefix)
    bucket_arn = f"arn:aws:s3:::{bucket}"

    def object_arn(name: str) -> str:
        return f"{bucket_arn}/{root}/{name}"

    def namespace_arn(namespace: str) -> str:
        return object_arn(f"{namespace}/*")

    def list_patterns(namespaces: tuple[str, ...]) -> list[str]:
        return [
            pattern
            for namespace in namespaces
            for pattern in (f"{root}/{namespace}", f"{root}/{namespace}/*")
        ]

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [bucket_arn],
                "Condition": {
                    "StringLike": {
                        "s3:prefix": list_patterns(_EVALUATOR_LIST_NAMESPACES),
                    }
                },
            },
            {
                "Effect": "Deny",
                "Action": ["s3:ListBucket"],
                "Resource": [bucket_arn],
                "Condition": {
                    "StringLike": {
                        "s3:prefix": list_patterns(_EVALUATOR_DENIED_NAMESPACES),
                    }
                },
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    *(namespace_arn(name) for name in _EVALUATOR_READ_NAMESPACES),
                    *(object_arn(name) for name in _EVALUATOR_READ_OBJECTS),
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject"],
                "Resource": [namespace_arn(name) for name in _EVALUATOR_WRITE_NAMESPACES],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:DeleteObject"],
                "Resource": [namespace_arn(name) for name in _EVALUATOR_DELETE_NAMESPACES],
            },
            {
                "Effect": "Deny",
                "Action": ["s3:GetObject"],
                "Resource": [
                    *(namespace_arn(name) for name in _EVALUATOR_DENIED_NAMESPACES),
                    *(object_arn(name) for name in _EVALUATOR_DENIED_OBJECTS),
                ],
            },
        ],
    }


def verify_evaluator_iam_policy(
    policy: object,
    *,
    bucket: str,
    prefix: str,
) -> None:
    """Reject missing or surplus policy paths, including retired namespace aliases."""

    if policy != evaluator_iam_policy(bucket=bucket, prefix=prefix):
        raise ValueError("evaluator IAM policy differs from the runtime object contract")


class StorageError(RuntimeError):
    """The private store failed or violated the required semantics."""


class PreconditionFailed(StorageError):
    """A plan changed since the single writer read it."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    body: bytes
    etag: str


class LocalBatchStore:
    """One GPU-local immutable content store shared by producer and evaluator.

    Object keys remain disclosure identifiers, not private S3 locations. Runtime
    assets and the full hidden source pool must not be placed in this directory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._verified: dict[str, tuple[int, int, int, int]] = {}

    def path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("local batch digest must be lowercase SHA-256")
        return self.root / sha256

    def get(self, *, sha256: str, size_bytes: int) -> Path:
        path = self.path(sha256)
        try:
            stat = path.stat()
            identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if not path.is_file() or stat.st_size != size_bytes:
                raise StorageError("local batch content is missing or has the wrong size")
            # Linux can give consecutive writes identical mtime/ctime values.
            # A hash of a freshly changed inode must not become a reusable hit,
            # even if a later lookup happens after the timestamp tick has ended.
            stable = time_ns() - max(stat.st_mtime_ns, stat.st_ctime_ns) >= (
                _CONTENT_TIMESTAMP_SETTLE_NS
            )
            if not stable:
                self._verified.pop(sha256, None)
            if self._verified.get(sha256) != identity:
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                after = path.stat()
                if digest != sha256 or identity != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns,
                ):
                    raise StorageError("local batch content failed SHA-256 verification")
                if stable:
                    self._verified[sha256] = identity
            return path
        except OSError as exc:
            raise StorageError(
                "GPU batch content is unavailable; original plan needs regeneration"
            ) from exc

    def put(self, source: str | Path, *, sha256: str, size_bytes: int) -> Path:
        """Seal a completed output without uploading or duplicating its large bytes."""

        source = Path(source)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path(sha256)
        with source.open("rb") as handle:
            if (
                os.fstat(handle.fileno()).st_size != size_bytes
                or hashlib.file_digest(handle, "sha256").hexdigest() != sha256
            ):
                raise StorageError("completed batch content disagrees with its descriptor")
            os.fsync(handle.fileno())
        try:
            os.link(source, destination)
        except FileExistsError:
            try:
                self.get(sha256=sha256, size_bytes=size_bytes)
            except StorageError:
                temporary = destination.with_name(f".{sha256}.{uuid.uuid4().hex}")
                try:
                    os.link(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError:
            # Cross-device deployment is valid, but still uses a single final CAS file.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as output:
                    temporary = Path(output.name)
                    with source.open("rb") as handle:
                        shutil.copyfileobj(handle, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                with suppress(FileExistsError):
                    os.link(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return self.get(sha256=sha256, size_bytes=size_bytes)

    def materialize(self, destination: Path, *, sha256: str, size_bytes: int) -> Path:
        source = self.get(sha256=sha256, size_bytes=size_bytes)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return destination
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}")
        try:
            temporary.symlink_to(source.resolve())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class PrivateObjectStore:
    """Minimal S3 transport used by one plan writer and immutable artifacts."""

    def __init__(
        self,
        client: Any,
        *,
        bucket: str,
        prefix: str,
        reconnect_from_environment: bool = False,
        credential_role: str = "producer",
    ) -> None:
        if not bucket or "/" in bucket:
            raise ValueError("private storage bucket is invalid")
        normalized = _normalize_prefix(prefix)
        self._client = client
        self._bucket = bucket
        self._prefix = normalized
        self._reconnect_from_environment = reconnect_from_environment
        if credential_role not in {"producer", "evaluator"}:
            raise ValueError("private storage credential role is invalid")
        self._credential_role = credential_role

    @classmethod
    def from_environment(cls, *, bucket: str, prefix: str) -> PrivateObjectStore:
        endpoint, access_key, secret_key = _storage_credentials("producer")
        return cls(
            _environment_client(endpoint, access_key, secret_key),
            bucket=bucket,
            prefix=prefix,
            reconnect_from_environment=True,
            credential_role="producer",
        )

    @classmethod
    def from_evaluator_environment(cls, *, bucket: str, prefix: str) -> PrivateObjectStore:
        """Use a distinct IAM identity which cannot read the hidden source pool."""

        endpoint, access_key, secret_key = _storage_credentials("evaluator")
        return cls(
            _environment_client(endpoint, access_key, secret_key),
            bucket=bucket,
            prefix=prefix,
            reconnect_from_environment=True,
            credential_role="evaluator",
        )

    def verify_evaluator_boundary(self, hidden_artifacts: tuple[Artifact, ...]) -> None:
        """Fail unless evaluator IAM denies hidden manifests, chunks, and listings."""

        if self._credential_role != "evaluator" or not hidden_artifacts:
            raise ValueError("evaluator boundary requires hidden objects and evaluator credentials")
        denied_objects = [SOURCE_POOL_OBJECT]
        denied_prefixes = ["source-pool/"]
        for artifact in hidden_artifacts:
            name = artifact.object_key
            part_count = (artifact.size_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES
            denied_objects.append(name)
            denied_objects.extend(
                chunk_object_name(name, f"part-{index:05d}") for index in range(part_count)
            )
            denied_prefixes.append(
                chunk_object_name(name, "part-00000").rsplit("/", 1)[0] + "/"
            )
        for name in denied_objects:
            try:
                self.get(name)
            except Exception as exc:
                if _is_access_denied(exc):
                    continue
                raise StorageError(
                    f"evaluator hidden-object denial could not be verified: {name}"
                ) from exc
            raise StorageError(f"evaluator credentials can read a hidden object: {name}")
        for prefix in denied_prefixes:
            try:
                self.list_names(prefix)
            except Exception as exc:
                if not _is_access_denied(exc):
                    raise StorageError(
                        f"evaluator hidden-prefix denial could not be verified: {prefix}"
                    ) from exc
            else:
                raise StorageError(f"evaluator credentials can list a hidden prefix: {prefix}")

    def __getstate__(self) -> dict[str, Any]:
        if not self._reconnect_from_environment:
            raise TypeError("an injected private object-store client cannot cross processes")
        return {
            "_bucket": self._bucket,
            "_prefix": self._prefix,
            "_reconnect_from_environment": True,
            "_credential_role": self._credential_role,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        constructor = (
            type(self).from_evaluator_environment
            if state["_credential_role"] == "evaluator"
            else type(self).from_environment
        )
        restored = constructor(bucket=state["_bucket"], prefix=state["_prefix"])
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

        name = f"{CAPABILITY_PROBES_NAMESPACE}/{uuid.uuid4().hex}.json"
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

    def put_content(self, path: str | Path, *, sha256: str, object_name: str) -> None:
        """Upload one immutable file as storage-limit-safe verified chunks."""

        _reject_hidden_batch_transport(object_name)
        source = Path(path)
        if not source.is_file():
            raise StorageError("content source is missing or its SHA-256 is wrong")
        manifest = build_chunk_manifest(source)
        if manifest.sha256 != sha256:
            raise StorageError("content source is missing or its SHA-256 is wrong")
        self._put_content(source, manifest, object_name=object_name)

    def _put_content(
        self,
        source: Path,
        manifest: ChunkManifest,
        *,
        object_name: str,
    ) -> None:
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
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=content,
                    ContentType="application/octet-stream",
                    Metadata={"sha256": part.sha256},
                    IfNoneMatch="*",
                )
            except Exception as exc:
                if not _is_precondition_failed(exc):
                    raise
            stored = self._client.head_object(Bucket=self._bucket, Key=key)
            if (
                stored.get("Metadata", {}).get("sha256") != part.sha256
                or stored.get("ContentLength") != part.size
            ):
                raise StorageError("immutable chunk upload verification failed")
        try:
            verify_chunked_content(
                manifest,
                lambda part_name: self._open_content_part(object_name, part_name),
            )
        except ValueError as exc:
            raise StorageError(f"immutable {exc}") from exc
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

        _reject_hidden_batch_transport(object_name)
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
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".part", delete=False
            ) as out:
                temporary = Path(out.name)
                try:
                    verify_chunked_content(
                        manifest,
                        lambda part_name: self._open_content_part(object_name, part_name),
                        write=out.write,
                    )
                except ValueError as exc:
                    raise StorageError(f"restored immutable content is invalid: {exc}") from exc
                out.flush()
                os.fsync(out.fileno())
            if temporary.stat().st_size != size_bytes:
                raise StorageError("restored content failed size or SHA-256 verification")
            os.replace(temporary, target)
            return target
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _open_content_part(self, object_name: str, part_name: str):
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key(chunk_object_name(object_name, part_name)),
        )
        return response.get("Body")

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


def _reject_hidden_batch_transport(object_name: str) -> None:
    if object_name.split("/", 1)[0] in {
        EXACT_NAMESPACE, INPUTS_NAMESPACE, LATENTS_NAMESPACE, VIDEOS_NAMESPACE,
    }:
        raise StorageError("hidden batch payloads are GPU-local, not private S3 objects")


def _storage_credentials(role: str) -> tuple[str, str, str]:
    if role == "evaluator":
        names = (
            "EVALUATOR_PRIVATE_STORAGE_URL",
            "EVALUATOR_PRIVATE_ACCESS_KEY",
            "EVALUATOR_PRIVATE_SECRET_ACCESS_KEY",
        )
        values = tuple(os.environ.get(name) for name in names)
        if not all(values):
            raise ValueError("evaluator private storage credentials are required")
        endpoint, access_key, secret_key = values
    else:
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
    assert endpoint is not None and access_key is not None and secret_key is not None
    return _validated_endpoint(endpoint), access_key, secret_key


def _validated_endpoint(endpoint: str) -> str:
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
    return endpoint


def _normalize_prefix(prefix: str) -> str:
    normalized = prefix.strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("private storage prefix is invalid")
    return normalized


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


def _is_access_denied(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = response.get("Error", {}).get("Code")
    return status == 403 or str(code) in {"403", "AccessDenied", "Forbidden"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
