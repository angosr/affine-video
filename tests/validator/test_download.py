from types import SimpleNamespace

import pytest

from validator.download import ModelDownloadError, download_public_model


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


def sibling(*, size: int | None):
    return SimpleNamespace(rfilename="selector.safetensors", size=size, lfs=None)


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
