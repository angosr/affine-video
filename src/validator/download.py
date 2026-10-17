"""Anonymous, bounded download of a fixed Hugging Face model file."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.errors import (
    HTTPError as HubTransportError,
)

from validator.chunks import ChunkManifest

MODEL_FILENAME = "selector.safetensors"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PART_NAME = re.compile(r"^part-[0-9]{3}$")
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_PART_BYTES = 48 * 1024 * 1024


class ModelDownloadError(RuntimeError):
    def __init__(self, code: str, message: str, *, temporary: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.temporary = temporary


@dataclass(frozen=True, slots=True)
class DownloadedModel:
    path: Path
    file_sha256: str
    size_bytes: int
    etag: str | None
    download_host: str | None


def download_chunked_file(
    manifest_url: str,
    *,
    cache_dir: str | Path,
    expected_sha256: str,
    max_bytes: int,
    suffix: str,
    timeout_seconds: float = 120,
    http_client: httpx.Client | None = None,
) -> Path:
    """Restore one generic public chunk manifest with bounded memory."""

    parsed_url = urlparse(manifest_url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
        or not parsed_url.path.endswith("/manifest.json")
        or not _DIGEST.fullmatch(expected_sha256)
        or not suffix.startswith(".")
        or "/" in suffix
    ):
        raise ValueError("chunked public file request is invalid")
    if max_bytes <= 0 or timeout_seconds <= 0:
        raise ValueError("chunked public file limits must be positive")
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{expected_sha256}{suffix}"
    if (
        target.is_file()
        and target.stat().st_size <= max_bytes
        and _sha256(target) == expected_sha256
    ):
        return target

    owned_client = http_client is None
    client = http_client or httpx.Client(follow_redirects=False, timeout=timeout_seconds)
    temporary: Path | None = None
    try:
        response = client.get(manifest_url, headers={"Accept": "application/json"})
        response.raise_for_status()
        if len(response.content) > _MAX_MANIFEST_BYTES:
            raise ValueError("chunk manifest exceeds the byte limit")
        manifest = ChunkManifest.decode(response.content)
        if manifest.sha256 != expected_sha256 or manifest.size > max_bytes:
            raise ValueError("chunk manifest conflicts with the expected file")
        descriptor, name = tempfile.mkstemp(prefix=".partial-chunk-", dir=cache)
        temporary = Path(name)
        complete = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output:
            for part in manifest.parts:
                part_url = urljoin(manifest_url, part.name)
                if urlparse(part_url).netloc != parsed_url.netloc:
                    raise ValueError("chunk URL escaped the manifest origin")
                part_response = client.get(part_url)
                part_response.raise_for_status()
                content = part_response.content
                if len(content) != part.size or hashlib.sha256(content).hexdigest() != part.sha256:
                    raise ValueError("public chunk failed size or digest verification")
                complete.update(content)
                output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != manifest.size or complete.hexdigest() != expected_sha256:
            raise ValueError("reassembled public file failed verification")
        os.replace(temporary, target)
        temporary = None
        return target
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise RuntimeError("public chunked file could not be restored") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if owned_client:
            client.close()


def download_chunked_asset(
    manifest_url: str,
    *,
    cache_dir: str | Path,
    expected_sha256: str,
    expected_model_digest: str,
    max_bytes: int,
    timeout_seconds: float = 120,
    http_client: httpx.Client | None = None,
) -> Path:
    """Reassemble one immutable public safetensors asset from bounded objects."""

    parsed_url = urlparse(manifest_url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
        or not parsed_url.path.endswith("/manifest.json")
    ):
        raise ValueError("asset manifest must be a credential-free HTTPS manifest URL")
    if not _DIGEST.fullmatch(expected_sha256) or not _DIGEST.fullmatch(expected_model_digest):
        raise ValueError("expected asset digests must be lowercase SHA-256 values")
    if max_bytes <= 0 or timeout_seconds <= 0:
        raise ValueError("asset download limits must be positive")

    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{expected_sha256}.safetensors"
    if (
        target.is_file()
        and target.stat().st_size <= max_bytes
        and _sha256(target) == expected_sha256
    ):
        return target

    owned_client = http_client is None
    client = http_client or httpx.Client(follow_redirects=False, timeout=timeout_seconds)
    temporary: Path | None = None
    try:
        manifest_bytes = bytearray()
        with client.stream("GET", manifest_url, headers={"Accept": "application/json"}) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared is not None and int(declared) > _MAX_MANIFEST_BYTES:
                raise ValueError("asset manifest exceeds the byte limit")
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                manifest_bytes.extend(chunk)
                if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
                    raise ValueError("asset manifest exceeds the byte limit")
        manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_json_object)
        parts = _validate_asset_manifest(
            manifest,
            expected_sha256=expected_sha256,
            expected_model_digest=expected_model_digest,
            max_bytes=max_bytes,
        )

        descriptor, temporary_name = tempfile.mkstemp(prefix=".partial-baseline-", dir=cache)
        temporary = Path(temporary_name)
        complete_digest = hashlib.sha256()
        received = 0
        with os.fdopen(descriptor, "wb") as output:
            for part in parts:
                part_digest = hashlib.sha256()
                part_received = 0
                part_url = urljoin(manifest_url, part["name"])
                if urlparse(part_url).netloc != parsed_url.netloc:
                    raise ValueError("asset part escaped the manifest origin")
                with client.stream("GET", part_url) as part_response:
                    part_response.raise_for_status()
                    declared = part_response.headers.get("content-length")
                    if declared is not None and int(declared) != part["size"]:
                        raise ValueError("asset part content length is incorrect")
                    for chunk in part_response.iter_bytes(chunk_size=1024 * 1024):
                        part_received += len(chunk)
                        received += len(chunk)
                        if part_received > part["size"] or received > max_bytes:
                            raise ValueError("asset download exceeds its byte limit")
                        part_digest.update(chunk)
                        complete_digest.update(chunk)
                        output.write(chunk)
                if part_received != part["size"] or part_digest.hexdigest() != part["sha256"]:
                    raise ValueError("asset part digest is incorrect")
            output.flush()
            os.fsync(output.fileno())
        if received != manifest["size"] or complete_digest.hexdigest() != expected_sha256:
            raise ValueError("reassembled asset digest is incorrect")
        os.replace(temporary, target)
        temporary = None
        return target
    except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise RuntimeError("public baseline asset could not be restored") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if owned_client:
            client.close()


def _validate_asset_manifest(
    value: object,
    *,
    expected_sha256: str,
    expected_model_digest: str,
    max_bytes: int,
) -> list[dict[str, object]]:
    fields = {"format", "filename", "size", "sha256", "model_digest", "source", "parts"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("asset manifest fields are invalid")
    if value["format"] != "chunked-safetensors" or value["filename"] != MODEL_FILENAME:
        raise ValueError("asset manifest format is invalid")
    if value["sha256"] != expected_sha256 or value["model_digest"] != expected_model_digest:
        raise ValueError("asset manifest identifies different weights")
    size = value["size"]
    if type(size) is not int or not 0 < size <= max_bytes:
        raise ValueError("asset manifest size is invalid")
    parts = value["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("asset manifest parts are invalid")
    validated: list[dict[str, object]] = []
    total = 0
    for index, item in enumerate(parts):
        if not isinstance(item, dict) or set(item) != {"name", "size", "sha256"}:
            raise ValueError("asset part fields are invalid")
        name = item["name"]
        part_size = item["size"]
        digest = item["sha256"]
        if (
            name != f"part-{index:03d}"
            or not isinstance(name, str)
            or not _PART_NAME.fullmatch(name)
        ):
            raise ValueError("asset part name is invalid")
        if type(part_size) is not int or not 0 < part_size <= _MAX_PART_BYTES:
            raise ValueError("asset part size is invalid")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("asset part digest is invalid")
        total += part_size
        validated.append(item)
    if total != size:
        raise ValueError("asset part sizes do not match the complete asset")
    return validated


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("asset manifest contains duplicate fields")
        value[key] = item
    return value


def download_public_model(
    repo_id: str,
    revision: str,
    *,
    cache_dir: str | Path,
    max_bytes: int,
    timeout_seconds: float = 60,
    api: HfApi | None = None,
    http_client: httpx.Client | None = None,
) -> DownloadedModel:
    """Download ``selector.safetensors`` anonymously into a content-addressed cache."""

    if max_bytes <= 0:
        raise ValueError("max bytes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")

    api = api or HfApi(token=False)
    try:
        info = api.model_info(
            repo_id,
            revision=revision,
            files_metadata=True,
            timeout=timeout_seconds,
            token=False,
        )
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise ModelDownloadError(
            "REPOSITORY_UNAVAILABLE", "repository is not anonymously readable"
        ) from exc
    except RevisionNotFoundError as exc:
        raise ModelDownloadError("REVISION_NOT_FOUND", "revision does not exist") from exc
    except HfHubHTTPError as exc:
        raise _http_failure(exc.response.status_code) from exc
    except HubTransportError as exc:
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE",
            "Hugging Face metadata request did not complete",
            temporary=True,
        ) from exc

    resolved_revision = getattr(info, "sha", None)
    if not isinstance(resolved_revision, str) or resolved_revision.lower() != revision.lower():
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE",
            "Hugging Face did not resolve the requested immutable revision",
            temporary=True,
        )

    try:
        sibling = next(
            (item for item in (info.siblings or ()) if item.rfilename == MODEL_FILENAME),
            None,
        )
    except (AttributeError, TypeError) as exc:
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE",
            "Hugging Face returned invalid repository metadata",
            temporary=True,
        ) from exc
    if sibling is None:
        raise ModelDownloadError("MODEL_FILE_MISSING", f"{MODEL_FILENAME} is missing")
    try:
        declared_size = sibling.size
        lfs = sibling.lfs
    except AttributeError as exc:
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE",
            "Hugging Face returned invalid model file metadata",
            temporary=True,
        ) from exc
    if declared_size is None and lfs is not None:
        if not isinstance(lfs, Mapping):
            raise ModelDownloadError(
                "DOWNLOAD_TEMPORARY_FAILURE",
                "Hugging Face returned invalid model file metadata",
                temporary=True,
            )
        declared_size = lfs.get("size")
    if declared_size is not None and (type(declared_size) is not int or declared_size < 0):
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE",
            "Hugging Face returned invalid model size metadata",
            temporary=True,
        )
    if declared_size is not None and declared_size > max_bytes:
        raise ModelDownloadError("MODEL_TOO_LARGE", "model exceeds the byte limit")

    owned_client = http_client is None
    client: httpx.Client | None = http_client
    descriptor: int | None = None
    temporary: Path | None = None
    digest = hashlib.sha256()
    received = 0
    etag: str | None = None
    host: str | None = None

    try:
        cache = Path(cache_dir).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".partial-", dir=cache)
        temporary = Path(temporary_name)
        client = client or httpx.Client(follow_redirects=True, timeout=timeout_seconds)
        raw_descriptor = descriptor
        descriptor = None
        with os.fdopen(raw_descriptor, "wb") as output:
            url = hf_hub_url(repo_id, MODEL_FILENAME, revision=revision)
            with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise _http_failure(response.status_code)
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise ModelDownloadError("MODEL_TOO_LARGE", "model exceeds the byte limit")
                etag = response.headers.get("etag")
                host = urlparse(str(response.url)).hostname
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    received += len(chunk)
                    if received > max_bytes:
                        raise ModelDownloadError("MODEL_TOO_LARGE", "model exceeds the byte limit")
                    digest.update(chunk)
                    output.write(chunk)
            if declared_size is not None and received != declared_size:
                raise ModelDownloadError(
                    "DOWNLOAD_TEMPORARY_FAILURE",
                    "downloaded model size does not match repository metadata",
                    temporary=True,
                )
            output.flush()
            os.fsync(output.fileno())

        file_sha256 = digest.hexdigest()
        target = cache / f"{file_sha256}.safetensors"
        if target.exists() and _sha256(target) == file_sha256:
            temporary.unlink()
        else:
            os.replace(temporary, target)
        temporary = None
        return DownloadedModel(target, file_sha256, received, etag, host)
    except ModelDownloadError:
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE", "model download did not complete", temporary=True
        ) from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        if owned_client and client is not None:
            with suppress(OSError):
                client.close()


def _http_failure(status_code: int) -> ModelDownloadError:
    if status_code in (401, 403):
        return ModelDownloadError(
            "REPOSITORY_UNAVAILABLE", "repository is not anonymously readable"
        )
    if status_code == 404:
        return ModelDownloadError("MODEL_FILE_MISSING", f"{MODEL_FILENAME} is missing")
    if status_code == 429 or status_code >= 500 or status_code in (408, 425):
        return ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE", "Hugging Face is temporarily unavailable", temporary=True
        )
    return ModelDownloadError(
        "REPOSITORY_UNAVAILABLE", f"Hugging Face rejected the download ({status_code})"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
