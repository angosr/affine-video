import hashlib
import io
import pickle

import pytest

from validator import chunks
from validator.production.commitment import Artifact, ArtifactManifest
from validator.production.storage import PreconditionFailed, PrivateObjectStore
from validator.s3 import client_config


class MemoryS3:
    def __init__(self) -> None:
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, **_kwargs):
        payload = bytes(Body)
        existing = self.objects.get((Bucket, Key))
        if _kwargs.get("IfNoneMatch") == "*" and existing is not None:
            raise precondition_failed()
        if "IfMatch" in _kwargs and (existing is None or existing[1] != _kwargs["IfMatch"]):
            raise precondition_failed()
        etag = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        self.objects[(Bucket, Key)] = (payload, etag, _kwargs.get("Metadata", {}))
        return {"ETag": f'"{etag}"'}

    def get_object(self, *, Bucket, Key):
        payload, etag, _metadata = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(payload), "ETag": f'"{etag}"'}

    def head_object(self, *, Bucket, Key):
        try:
            payload, etag, metadata = self.objects[(Bucket, Key)]
        except KeyError as exc:
            error = RuntimeError("missing")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error from exc
        return {
            "ContentLength": len(payload),
            "ETag": f'"{etag}"',
            "Metadata": metadata,
        }

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def precondition_failed():
    error = RuntimeError("precondition failed")
    error.response = {
        "Error": {"Code": "PreconditionFailed"},
        "ResponseMetadata": {"HTTPStatusCode": 412},
    }
    return error


def test_single_writer_plan_updates_are_idempotent_and_detect_stale_reads() -> None:
    store = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")

    created = store.create_plan("plans/100.json", b"planned")
    assert store.create_plan("plans/100.json", b"planned") == created
    with pytest.raises(PreconditionFailed, match="different bytes"):
        store.create_plan("plans/100.json", b"other")
    with pytest.raises(PreconditionFailed, match="changed"):
        store.replace_plan("plans/100.json", b"ready", etag="stale")
    ready = store.replace_plan("plans/100.json", b"ready", etag=created.etag)

    assert ready.body == b"ready"


def test_plan_backend_capability_probe_checks_both_conditions_and_cleans_up() -> None:
    client = MemoryS3()
    store = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")

    store.verify_conditional_writes()

    assert client.objects == {}


def test_environment_store_reconnects_after_cross_process_serialization(
    monkeypatch,
) -> None:
    clients = []

    def client(_endpoint, _access_key, _secret_key):
        result = MemoryS3()
        clients.append(result)
        return result

    monkeypatch.setenv("STORAGE_URL", "https://storage.example/s3")
    monkeypatch.setenv("ACCESS_KEY", "access")
    monkeypatch.setenv("SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr("validator.production.storage._environment_client", client)
    store = PrivateObjectStore.from_environment(bucket="private", prefix="tracks/abc")

    restored = pickle.loads(pickle.dumps(store))

    assert restored._bucket == "private"
    assert restored._prefix == "tracks/abc"
    assert restored._client is clients[-1]


def test_private_storage_credentials_take_precedence_and_must_be_complete(monkeypatch) -> None:
    captured = []

    def client(endpoint, access_key, secret_key):
        captured.append((endpoint, access_key, secret_key))
        return MemoryS3()

    monkeypatch.setenv("STORAGE_URL", "https://public.example/s3")
    monkeypatch.setenv("ACCESS_KEY", "public-access")
    monkeypatch.setenv("SECRET_ACCESS_KEY", "public-secret")
    monkeypatch.setenv("PRIVATE_STORAGE_URL", "https://private.example/s3")
    monkeypatch.setenv("PRIVATE_ACCESS_KEY", "private-access")
    monkeypatch.setenv("PRIVATE_SECRET_ACCESS_KEY", "private-secret")
    monkeypatch.setattr("validator.production.storage._environment_client", client)

    PrivateObjectStore.from_environment(bucket="private", prefix="tracks/abc")
    assert captured == [("https://private.example/s3", "private-access", "private-secret")]

    monkeypatch.delenv("PRIVATE_SECRET_ACCESS_KEY")
    with pytest.raises(ValueError, match="configured together"):
        PrivateObjectStore.from_environment(bucket="private", prefix="tracks/abc")


def test_injected_store_client_cannot_be_serialized() -> None:
    store = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")

    with pytest.raises(TypeError, match="cannot cross processes"):
        pickle.dumps(store)


def test_supabase_transport_disables_unsupported_optional_checksums() -> None:
    client = client_config()

    assert client.request_checksum_calculation == "when_required"
    assert client.response_checksum_validation == "when_required"


def test_content_is_chunked_and_restored_with_complete_verification(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 4)
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefghij")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    client = MemoryS3()
    store = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")

    store.put_content(source, sha256=digest, object_name="objects/file")
    restored = store.download_content(
        "objects/file",
        tmp_path / "restored.bin",
        sha256=digest,
        size_bytes=10,
    )

    assert restored.read_bytes() == b"abcdefghij"
    keys = sorted(key for bucket, key in client.objects if bucket == "private")
    assert keys == [
        "tracks/abc/objects/file",
        "tracks/abc/objects/file.parts/part-00000",
        "tracks/abc/objects/file.parts/part-00001",
        "tracks/abc/objects/file.parts/part-00002",
    ]


def test_bootstrap_and_manifest_restore_survive_loss_of_local_artifacts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 4)
    root = tmp_path / "runtime"
    root.mkdir()
    path = root / "model.bin"
    path.write_bytes(b"abcdefghij")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = ArtifactManifest((Artifact("model.bin", digest, 10, f"artifacts/{digest}"),))
    bootstrap = tmp_path / "manifest.json"
    bootstrap.write_bytes(manifest.encode())
    client = MemoryS3()
    store = PrivateObjectStore(client, bucket="private", prefix="tracks/abc")

    payload = store.ensure_bootstrap("bootstrap/artifact_manifest.json", bootstrap)
    store.ensure_artifacts(ArtifactManifest.decode(payload), root)
    path.unlink()
    bootstrap.unlink()

    restored_payload = store.ensure_bootstrap("bootstrap/artifact_manifest.json", bootstrap)
    store.ensure_artifacts(ArtifactManifest.decode(restored_payload), root)

    assert restored_payload == manifest.encode()
    assert path.read_bytes() == b"abcdefghij"
