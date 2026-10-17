import errno
import hashlib
import json
from types import SimpleNamespace

import pytest

from validator import chunks
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
            assert headers == {"Accept": "application/json"}
            return AssetResponse(self.manifest)
        assert headers is None
        return AssetResponse(self.parts[url.rsplit("/", 1)[-1]])

    def get(self, url: str, *, headers: dict[str, str] | None = None):
        if url.endswith("/manifest.json"):
            assert headers == {"Accept": "application/json"}
            return AssetResponse(self.manifest)
        assert headers is None
        return AssetResponse(self.parts[url.rsplit("/", 1)[-1]])


def sibling(*, size: int | None):
    return SimpleNamespace(rfilename="selector.safetensors", size=size, lfs=None)


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


def test_local_download_preparation_failure_is_temporary(monkeypatch, tmp_path) -> None:
    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr("validator.download.tempfile.mkstemp", disk_full)
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
    assert client.called is False


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
