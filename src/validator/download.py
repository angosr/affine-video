"""Anonymous, bounded download of a fixed Hugging Face model file."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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

MODEL_FILENAME = "selector.safetensors"


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
            (
                item
                for item in (info.siblings or ())
                if item.rfilename == MODEL_FILENAME
            ),
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
    if declared_size is not None and (
        type(declared_size) is not int or declared_size < 0
    ):
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE",
            "Hugging Face returned invalid model size metadata",
            temporary=True,
        )
    if declared_size is not None and declared_size > max_bytes:
        raise ModelDownloadError("MODEL_TOO_LARGE", "model exceeds the byte limit")

    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".partial-", dir=cache)
    temporary = Path(temporary_name)
    owned_client = http_client is None
    client = http_client or httpx.Client(follow_redirects=True, timeout=timeout_seconds)
    digest = hashlib.sha256()
    received = 0
    etag: str | None = None
    host: str | None = None

    try:
        with os.fdopen(descriptor, "wb") as output:
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
        return DownloadedModel(target, file_sha256, received, etag, host)
    except ModelDownloadError:
        temporary.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            "DOWNLOAD_TEMPORARY_FAILURE", "model download did not complete", temporary=True
        ) from exc
    finally:
        if owned_client:
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
