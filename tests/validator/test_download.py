import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import BlobLfsInfo

from validator import chunks
from validator import download as download_module
from validator.chunks import ChunkManifest, ChunkPart
from validator.download import (
    ModelDownloadError,
    download_chunked_asset,
    download_chunked_file,
    download_public_model,
)


class FakeApi:
    def __init__(self, siblings) -> None:
        self.siblings = siblings

    def model_info(self, *_args, **_kwargs):
        return SimpleNamespace(siblings=self.siblings, sha="12" * 20)


class FakeResponse:
    status_code = 200
    headers = {"content-length": "6", "etag": '"etag"'}
    url = "https://cdn.example/model"

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def iter_bytes(self, *, chunk_size: int):
        assert chunk_size == 1024 * 1024
        yield b"abc"
        yield b"def"


class FakeClient:
    def __init__(self) -> None:
        self.called = False

    def stream(self, method: str, url: str):
        assert method == "GET"
        assert "/resolve/" in url
        self.called = True
        return FakeResponse()


class AssetResponse:
    status_code = 200

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.headers = {"content-length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, *, chunk_size: int):
        assert chunk_size in {64 * 1024, 1024 * 1024}
        yield self.content


class AssetClient:
    def __init__(self, manifest: bytes, parts: dict[str, bytes]) -> None:
        self.manifest = manifest
        self.parts = parts

    def stream(self, method: str, url: str, *, headers: dict[str, str] | None = None):
        assert method == "GET"
        if url.endswith("/manifest.json"):
            assert headers == {"Accept": "application/json", "Accept-Encoding": "identity"}
            return AssetResponse(self.manifest)
        assert headers == {"Accept-Encoding": "identity"}
        return AssetResponse(self.parts[url.rsplit("/", 1)[-1]])


def sibling(*, size: int | None):
    return SimpleNamespace(
        rfilename="selector.safetensors", size=size, lfs=None,
        blob_id=hashlib.sha1(b"blob 6\0abcdef", usedforsecurity=False).hexdigest(),
    )


def test_chunked_public_asset_is_reassembled_and_fully_verified(tmp_path) -> None:
    parts = {"part-000": b"abc", "part-001": b"def"}
    complete = b"".join(parts.values())
    complete_digest = hashlib.sha256(complete).hexdigest()
    model_digest = "ab" * 32
    manifest = json.dumps(
        {
            "format": "chunked-safetensors",
            "filename": "selector.safetensors",
            "size": len(complete),
            "sha256": complete_digest,
            "model_digest": model_digest,
            "source": {},
            "parts": [
                {
                    "name": name,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in parts.items()
            ],
        }
    ).encode()

    path = download_chunked_asset(
        "https://public.example/assets/manifest.json",
        cache_dir=tmp_path,
        expected_sha256=complete_digest,
        expected_model_digest=model_digest,
        max_bytes=10,
        http_client=AssetClient(manifest, parts),
    )

    assert path.read_bytes() == complete
    assert path.name == f"{complete_digest}.safetensors"
    assert not list(tmp_path.glob(".partial-baseline-*"))


def test_chunked_public_asset_rejects_a_corrupt_part(tmp_path) -> None:
    expected = hashlib.sha256(b"abcdef").hexdigest()
    manifest = json.dumps(
        {
            "format": "chunked-safetensors",
            "filename": "selector.safetensors",
            "size": 6,
            "sha256": expected,
            "model_digest": "ab" * 32,
            "source": {},
            "parts": [
                {"name": "part-000", "size": 6, "sha256": expected},
            ],
        }
    ).encode()

    with pytest.raises(RuntimeError, match="could not be restored"):
        download_chunked_asset(
            "https://public.example/assets/manifest.json",
            cache_dir=tmp_path,
            expected_sha256=expected,
            expected_model_digest="ab" * 32,
            max_bytes=10,
            http_client=AssetClient(manifest, {"part-000": b"xxxxxx"}),
        )
    assert not list(tmp_path.glob(".partial-baseline-*"))


def test_generic_chunked_file_is_reassembled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 3)
    parts = {"part-00000": b"abc", "part-00001": b"def"}
    complete = b"abcdef"
    digest = hashlib.sha256(complete).hexdigest()
    manifest = ChunkManifest(
        digest,
        len(complete),
        tuple(
            ChunkPart(name, hashlib.sha256(content).hexdigest(), len(content))
            for name, content in parts.items()
        ),
    ).encode()

    path = download_chunked_file(
        "https://public.example/models/digest/manifest.json",
        cache_dir=tmp_path,
        expected_sha256=digest,
        max_bytes=10,
        suffix=".safetensors",
        http_client=AssetClient(manifest, parts),
    )

    assert path.read_bytes() == complete


class _CountingStream(httpx.SyncByteStream):
    def __init__(self, content: bytes, repeats: int = 1) -> None:
        self.content = content
        self.repeats = repeats
        self.chunks_read = 0
        self.closed = False

    def __iter__(self):
        for _ in range(self.repeats):
            self.chunks_read += 1
            yield self.content

    def close(self) -> None:
        self.closed = True


@pytest.fixture(params=["generic", "asset"])
def public_chunked_download(request):
    content = b"abcdef"
    digest = hashlib.sha256(content).hexdigest()
    part_name = "part-00000" if request.param == "generic" else "part-000"
    if request.param == "generic":
        manifest = ChunkManifest(digest, len(content), (
            ChunkPart(part_name, digest, len(content)),
        )).encode()
    else:
        manifest = json.dumps({
            "format": "chunked-safetensors", "filename": "selector.safetensors",
            "size": len(content), "sha256": digest, "model_digest": "ab" * 32,
            "source": {}, "parts": [{"name": part_name, "size": len(content), "sha256": digest}],
        }).encode()

    def load(cache_dir, http_client=None):
        common = {
            "cache_dir": cache_dir, "expected_sha256": digest,
            "max_bytes": 10, "http_client": http_client,
        }
        url = "https://public.example/assets/manifest.json"
        if request.param == "generic":
            return download_chunked_file(url, suffix=".safetensors", **common)
        return download_chunked_asset(url, expected_model_digest="ab" * 32, **common)

    return SimpleNamespace(manifest=manifest, part_name=part_name, digest=digest, load=load)


@pytest.mark.parametrize("location", ["manifest", "part"])
def test_public_chunked_download_rejects_encoding_before_decompression(
    tmp_path, monkeypatch, public_chunked_download, location,
) -> None:
    case = public_chunked_download
    body = _CountingStream(gzip.compress(b"x" * 8192))
    requests = []

    def decode_was_called(*_args):
        pytest.fail("compressed response reached the automatic decoder")

    monkeypatch.setattr("httpx._decoders.GZipDecoder.decode", decode_was_called)

    def respond(request):
        requests.append(request)
        if location == "part" and request.url.path.endswith("/manifest.json"):
            return httpx.Response(200, content=case.manifest)
        return httpx.Response(200, stream=body, headers={"Content-Encoding": "gzip"})

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(RuntimeError) as caught,
    ):
        case.load(tmp_path, client)
    assert isinstance(caught.value.__cause__, ValueError)
    assert body.chunks_read == 0 and body.closed
    assert all(r.headers["accept-encoding"] == "identity" for r in requests)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ["http_close", "part_unlink_and_close", "success_close"])
def test_public_chunked_download_cleanup_preserves_fallback_errors(
    tmp_path, monkeypatch, public_chunked_download, fault,
) -> None:
    case = public_chunked_download
    calls = []
    close_failure = httpx.CloseError("secondary client cleanup failure")

    def respond(request):
        if request.url.path.endswith("/manifest.json"):
            return httpx.Response(200, content=case.manifest)
        return httpx.Response(200 if fault == "success_close" else 503, content=b"abcdef")

    client = httpx.Client(transport=httpx.MockTransport(respond))
    real_close, real_unlink = client.close, Path.unlink

    def fail_close():
        calls.append("close")
        real_close()
        raise close_failure

    def fail_unlink(path, *args, **kwargs):
        if path.name.startswith(".partial-"):
            calls.append("unlink")
            raise OSError(errno.EIO, "secondary unlink failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(download_module.httpx, "Client", lambda **_kwargs: client)
        patch.setattr(client, "close", fail_close)
        if fault == "part_unlink_and_close":
            patch.setattr(Path, "unlink", fail_unlink)
        with pytest.raises(RuntimeError) as caught:
            case.load(tmp_path)
    if fault == "success_close":
        assert caught.value.__cause__ is close_failure
        assert case.load(tmp_path, client).read_bytes() == b"abcdef"
    else:
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    assert calls == (["unlink", "close"] if fault == "part_unlink_and_close" else ["close"])
    for partial in tmp_path.glob(".partial-*"):
        partial.unlink()


@pytest.mark.parametrize("declared", [True, False])
def test_generic_chunked_file_limits_manifest_before_buffering(
    tmp_path, monkeypatch, declared,
) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(download_module, "_MAX_MANIFEST_BYTES", limit)
    body = _CountingStream(b"x" * limit, repeats=64)
    headers = {"Content-Length": str(limit + 1)} if declared else {}
    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=body, headers=headers),
        )) as client,
        pytest.raises(RuntimeError, match="could not be restored"),
    ):
        download_chunked_file(
            "https://public.example/models/digest/manifest.json",
            cache_dir=tmp_path,
            expected_sha256="ab" * 32,
            max_bytes=10,
            suffix=".safetensors",
            http_client=client,
        )
    assert body.chunks_read <= (0 if declared else 2)
    assert body.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", [
    "oversized_header", "short_header", "oversized_body", "lying_header",
    "short_body", "corrupt_body", "http_error",
])
def test_generic_chunked_file_limits_parts_and_preserves_existing_cache(tmp_path, fault) -> None:
    content = b"a" * (64 * 1024)
    digest = hashlib.sha256(content).hexdigest()
    manifest = ChunkManifest(
        digest, len(content), (ChunkPart("part-00000", digest, len(content)),),
    ).encode()
    headers = {}
    if fault in {"oversized_header", "short_header", "lying_header"}:
        delta = {"oversized_header": 1, "short_header": -1, "lying_header": 0}[fault]
        headers["Content-Length"] = str(len(content) + delta)
    body = _CountingStream(
        b"b" * len(content) if fault == "corrupt_body" else
        b"a" if fault == "short_body" else content,
        repeats=1 if fault in {"short_body", "corrupt_body"} else 64,
    )
    target = tmp_path / f"{digest}.safetensors"
    target.write_bytes(b"previous cache")

    def respond(request):
        if request.url.path.endswith("/manifest.json"):
            return httpx.Response(200, content=manifest)
        return httpx.Response(503 if fault == "http_error" else 200, stream=body, headers=headers)

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(RuntimeError, match="could not be restored"),
    ):
        download_chunked_file(
            "https://public.example/models/digest/manifest.json",
            cache_dir=tmp_path,
            expected_sha256=digest,
            max_bytes=len(content),
            suffix=".safetensors",
            http_client=client,
        )
    assert body.chunks_read <= (0 if fault in {
        "oversized_header", "short_header", "http_error",
    } else 2)
    assert body.closed
    assert target.read_bytes() == b"previous cache"
    assert list(tmp_path.iterdir()) == [target]


def test_downloads_anonymously_into_content_addressed_cache(tmp_path) -> None:
    client = FakeClient()
    downloaded = download_public_model(
        "owner/model",
        "12" * 20,
        cache_dir=tmp_path,
        max_bytes=10,
        api=FakeApi([sibling(size=6)]),
        http_client=client,
    )

    assert client.called is True
    assert downloaded.path.read_bytes() == b"abcdef"
    assert downloaded.path.name == f"{downloaded.file_sha256}.safetensors"
    assert downloaded.size_bytes == 6
    assert downloaded.download_host == "cdn.example"
    assert not list(tmp_path.glob(".partial-*"))


def test_download_persists_an_absolute_cache_path(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    downloaded = download_public_model(
        "owner/model",
        "12" * 20,
        cache_dir="relative-cache",
        max_bytes=10,
        api=FakeApi([sibling(size=6)]),
        http_client=FakeClient(),
    )

    assert downloaded.path.is_absolute()
    assert downloaded.path.parent == (tmp_path / "relative-cache").resolve()


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EIO])
@pytest.mark.parametrize(
    "operation",
    ["mkdir", "mkstemp", "fdopen", "write", "flush", "close", "fsync", "replace", "unlink"],
)
def test_local_download_io_failure_is_temporary(
    monkeypatch, tmp_path, operation, error_number
) -> None:
    failure = OSError(error_number, "injected local I/O failure")

    def disk_full(*_args, **_kwargs):
        raise failure

    targets = {
        "mkdir": "validator.download.Path.mkdir",
        "mkstemp": "validator.download.tempfile.mkstemp",
        "fdopen": "validator.download.os.fdopen",
        "fsync": "validator.download.os.fsync",
        "replace": "validator.download.os.replace",
        "unlink": "validator.download.Path.unlink",
    }
    if operation == "unlink":
        (tmp_path / f"{hashlib.sha256(b'abcdef').hexdigest()}.safetensors").write_bytes(
            b"abcdef"
        )
    if operation in targets:
        monkeypatch.setattr(targets[operation], disk_full)
    else:
        real_fdopen = os.fdopen

        class FailingOutput:
            def __init__(self, *args, **kwargs):
                self.output = real_fdopen(*args, **kwargs)

            def close(self):
                self.output.close()
                if operation == "close":
                    raise failure

            def __getattr__(self, name):
                if name == operation:
                    return disk_full
                return getattr(self.output, name)

        monkeypatch.setattr(download_module.os, "fdopen", FailingOutput)

    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=10,
            api=FakeApi([sibling(size=6)]),
            http_client=FakeClient(),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True
    assert caught.value.__cause__ is failure


def test_fdopen_failure_closes_the_unclaimed_descriptor(monkeypatch, tmp_path) -> None:
    descriptors = []
    real_close = os.close

    def fail_fdopen(descriptor, _mode):
        descriptors.append(descriptor)
        raise OSError(errno.ENOSPC, "fdopen failed")

    monkeypatch.setattr(download_module.os, "fdopen", fail_fdopen)
    with pytest.raises(ModelDownloadError):
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([sibling(size=6)]), http_client=FakeClient(),
        )

    assert len(descriptors) == 1
    with pytest.raises(OSError) as caught:
        real_close(descriptors[0])
    assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 200])
def test_cleanup_failures_preserve_the_original_download_error(
    monkeypatch, tmp_path, status
) -> None:
    primary_failure = OSError(errno.ENOSPC, "original disk full")
    cleanup_calls = []
    real_fdopen = os.fdopen

    def failing_fdopen(*args, **kwargs):
        output = real_fdopen(*args, **kwargs)
        real_close = output.close

        def fail_close():
            real_close()
            cleanup_calls.append("file")
            raise OSError(errno.EIO, "secondary file close failure")

        monkeypatch.setattr(output, "close", fail_close)
        return output

    class CleanupFailingClient(FakeClient):
        def stream(self, method, url):
            response = super().stream(method, url)
            response.status_code = status
            return response

        def close(self):
            cleanup_calls.append("client")
            raise httpx.CloseError("secondary close failure")

    def fail_unlink(*_args, **_kwargs):
        cleanup_calls.append("unlink")
        raise OSError(errno.EIO, "secondary unlink failure")

    def fail_fsync(*_args):
        raise primary_failure

    monkeypatch.setattr(download_module.httpx, "Client", lambda **_kwargs: CleanupFailingClient())
    monkeypatch.setattr(download_module.os, "fdopen", failing_fdopen)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(download_module.os, "fsync", fail_fsync)
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([sibling(size=6)]),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True
    assert cleanup_calls == ["file", "unlink", "client"]
    if status == 200:
        assert caught.value.__cause__ is primary_failure


@pytest.mark.parametrize("status", [401, 403, 404])
@pytest.mark.parametrize("becomes_private", [False, True])
def test_payload_access_failure_rechecks_metadata_before_retry(
    tmp_path, status, becomes_private
) -> None:
    class ChangingApi(FakeApi):
        calls = 0

        def model_info(self, *args, **kwargs):
            self.calls += 1
            assert kwargs["token"] is False
            if becomes_private and self.calls > 1:
                raise RepositoryNotFoundError(
                    "repository is no longer public",
                    response=httpx.Response(
                        401, request=httpx.Request("GET", "https://hf.test"),
                    ),
                )
            return super().model_info(*args, **kwargs)

    class RecoveringClient(FakeClient):
        requests = 0

        def stream(self, *args):
            self.requests += 1
            response = super().stream(*args)
            response.status_code = status if self.requests == 1 else 200
            return response

    api = ChangingApi([sibling(size=6)])
    client = RecoveringClient()
    arguments = dict(
        cache_dir=tmp_path, max_bytes=10, api=api, http_client=client,
    )
    with pytest.raises(ModelDownloadError) as first:
        download_public_model("owner/model", "12" * 20, **arguments)
    assert first.value.temporary is True
    assert first.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert not list(tmp_path.iterdir())
    if becomes_private:
        with pytest.raises(ModelDownloadError) as second:
            download_public_model("owner/model", "12" * 20, **arguments)
        assert second.value.code == "REPOSITORY_UNAVAILABLE"
        assert second.value.temporary is False
        assert client.requests == 1
    else:
        result = download_public_model("owner/model", "12" * 20, **arguments)
        assert result.path.read_bytes() == b"abcdef"
        assert client.requests == 2
    assert api.calls == 2


@pytest.mark.parametrize(
    ("status", "code", "temporary"),
    [(401, "REPOSITORY_UNAVAILABLE", False), (403, "REPOSITORY_UNAVAILABLE", False),
     (404, "MODEL_FILE_MISSING", False), (429, "DOWNLOAD_TEMPORARY_FAILURE", True),
     (500, "DOWNLOAD_TEMPORARY_FAILURE", True)],
)
def test_anonymous_metadata_failure_keeps_its_admission_classification(
    tmp_path, status, code, temporary
) -> None:
    class DeniedApi:
        def model_info(self, *_args, **kwargs):
            assert kwargs["token"] is False
            raise HfHubHTTPError(
                "metadata rejected",
                response=httpx.Response(status, request=httpx.Request("GET", "https://hf.test")),
            )

    client = FakeClient()
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=DeniedApi(), http_client=client,
        )
    assert caught.value.code == code
    assert caught.value.temporary is temporary
    assert client.called is False


@pytest.mark.parametrize(
    "failure", [OSError(errno.EIO, "close failed"), httpx.CloseError("close failed")]
)
def test_client_close_io_failure_after_download_is_temporary(
    monkeypatch, tmp_path, failure
) -> None:
    class CleanupFailingClient(FakeClient):
        def close(self):
            raise failure

    monkeypatch.setattr(download_module.httpx, "Client", lambda **_kwargs: CleanupFailingClient())
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([sibling(size=6)]),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True
    assert caught.value.__cause__ is failure
    assert list(tmp_path.glob("*.safetensors"))


def test_metadata_size_rejects_before_download(tmp_path) -> None:
    client = FakeClient()
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=5,
            api=FakeApi([sibling(size=6)]),
            http_client=client,
        )

    assert caught.value.code == "MODEL_TOO_LARGE"
    assert caught.value.temporary is False
    assert client.called is False


@pytest.mark.parametrize("declared_size", (None, 6))
@pytest.mark.parametrize("failure", ("header", "body"))
def test_oversized_response_is_bounded_and_retryable_after_pinned_metadata(
    tmp_path, declared_size, failure,
) -> None:
    class OversizedResponse(FakeResponse):
        headers = {"content-length": "11"} if failure == "header" else {}

        def iter_bytes(self, *, chunk_size):
            if failure == "header":
                pytest.fail("oversized response header must reject before reading the body")
            yield b"x" * 11
            pytest.fail("body must stop immediately at the byte limit")

    class OversizedClient(FakeClient):
        def stream(self, *_args):
            return OversizedResponse()

    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([sibling(size=declared_size)]), http_client=OversizedClient(),
        )
    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True
    assert not list(tmp_path.iterdir())


def test_missing_fixed_filename_is_participant_failure(tmp_path) -> None:
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=10,
            api=FakeApi([]),
            http_client=FakeClient(),
        )

    assert caught.value.code == "MODEL_FILE_MISSING"
    assert caught.value.temporary is False


def test_revision_resolution_must_match_submitted_commit(tmp_path) -> None:
    api = FakeApi([sibling(size=6)])

    def mismatched(*_args, **_kwargs):
        return SimpleNamespace(siblings=api.siblings, sha="34" * 20)

    api.model_info = mismatched
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=10,
            api=api,
            http_client=FakeClient(),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True


@pytest.mark.parametrize("metadata_type", ("lfs-dict", "lfs-sdk", "git", "git-unknown-size"))
@pytest.mark.parametrize("matching", (True, False))
def test_downloaded_bytes_must_match_pinned_revision_file_digest(
    tmp_path, metadata_type, matching,
) -> None:
    payload = b"abcdef" if matching else b"ABCDEF"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    item = sibling(size=None if metadata_type == "git-unknown-size" else 6)
    if metadata_type.startswith("lfs"):
        values = {"size": 6, "sha256": expected_sha256, "pointer_size": 128}
        item.lfs = BlobLfsInfo(**values) if metadata_type == "lfs-sdk" else values
    else:
        item.blob_id = hashlib.sha1(b"blob 6\0" + payload, usedforsecurity=False).hexdigest()
    arguments = dict(
        cache_dir=tmp_path, max_bytes=10,
        api=FakeApi([item]), http_client=FakeClient(),
    )
    if matching:
        downloaded = download_public_model("owner/model", "12" * 20, **arguments)
        assert downloaded.path.read_bytes() == payload
        assert downloaded.file_sha256 == expected_sha256
    else:
        with pytest.raises(ModelDownloadError) as caught:
            download_public_model("owner/model", "12" * 20, **arguments)
        assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
        assert caught.value.temporary is True
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("declared_size", (None, 6))
@pytest.mark.parametrize(
    "lfs",
    (
        "bad", {}, {"size": 6}, {"size": 6, "sha256": "bad"},
        {"size": True, "sha256": "ab" * 32},
        {"size": -1, "sha256": "ab" * 32},
        {"size": "6", "sha256": "ab" * 32},
    ),
)
def test_invalid_lfs_binding_is_temporary_before_download(tmp_path, declared_size, lfs) -> None:
    item = sibling(size=declared_size)
    item.lfs = lfs
    client = FakeClient()
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([item]), http_client=client,
        )
    assert caught.value.temporary is True
    assert client.called is False
    assert not list(tmp_path.iterdir())


def test_conflicting_lfs_size_is_temporary_before_download(tmp_path) -> None:
    item = sibling(size=7)
    item.lfs = {"size": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest()}
    client = FakeClient()
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([item]), http_client=client,
        )
    assert caught.value.temporary is True
    assert client.called is False


@pytest.mark.parametrize("blob_id", (None, 123, "invalid", "ab" * 19, "ab" * 21))
def test_invalid_git_blob_binding_is_temporary_before_download(tmp_path, blob_id) -> None:
    item = sibling(size=6)
    item.blob_id = blob_id
    client = FakeClient()
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model", "12" * 20, cache_dir=tmp_path, max_bytes=10,
            api=FakeApi([item]), http_client=client,
        )
    assert caught.value.temporary is True
    assert client.called is False


def test_truncated_download_is_temporary_instead_of_participant_failure(tmp_path) -> None:
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=10,
            api=FakeApi([sibling(size=7)]),
            http_client=FakeClient(),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True
    assert not list(tmp_path.glob(".partial-*"))


def test_malformed_repository_metadata_is_temporary(tmp_path) -> None:
    malformed = SimpleNamespace(rfilename="selector.safetensors", size="6", lfs=None)
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=10,
            api=FakeApi([malformed]),
            http_client=FakeClient(),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True


def test_malformed_lfs_metadata_is_temporary(tmp_path) -> None:
    malformed = SimpleNamespace(rfilename="selector.safetensors", size=None, lfs="bad")
    with pytest.raises(ModelDownloadError) as caught:
        download_public_model(
            "owner/model",
            "12" * 20,
            cache_dir=tmp_path,
            max_bytes=10,
            api=FakeApi([malformed]),
            http_client=FakeClient(),
        )

    assert caught.value.code == "DOWNLOAD_TEMPORARY_FAILURE"
    assert caught.value.temporary is True
