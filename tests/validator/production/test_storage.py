import hashlib
import io
import json
import pickle
from copy import deepcopy

import pytest

from validator import chunks
from validator.production.commitment import Artifact, ArtifactManifest
from validator.production.storage import (
    RETRY_STATE_NAMESPACE,
    PreconditionFailed,
    PrivateObjectStore,
    StorageError,
    evaluator_iam_policy,
    verify_evaluator_iam_policy,
)
from validator.publication import S3Repository
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


def test_evaluator_iam_policy_is_derived_from_runtime_namespaces() -> None:
    policy = evaluator_iam_policy(bucket="private", prefix="evaluation/tracks/test")

    verify_evaluator_iam_policy(
        policy,
        bucket="private",
        prefix="evaluation/tracks/test",
    )
    encoded = json.dumps(policy, sort_keys=True)
    assert f"/tracks/test/{RETRY_STATE_NAMESPACE}/*" in encoded
    assert "/tracks/test/retries/*" not in encoded
    assert "/tracks/test/videos/*" in encoded
    assert "/tracks/test/bootstrap/artifact_manifest.json" in encoded


@pytest.mark.parametrize("change", ("retired-prefix", "missing-runtime-prefix"))
def test_evaluator_iam_policy_rejects_namespace_drift(change: str) -> None:
    policy = evaluator_iam_policy(bucket="private", prefix="tracks/test")
    changed = deepcopy(policy)
    get_resources = changed["Statement"][2]["Resource"]
    assert isinstance(get_resources, list)
    if change == "retired-prefix":
        get_resources.append("arn:aws:s3:::private/tracks/test/retries/*")
    else:
        get_resources.remove(
            f"arn:aws:s3:::private/tracks/test/{RETRY_STATE_NAMESPACE}/*"
        )

    with pytest.raises(ValueError, match="differs from the runtime object contract"):
        verify_evaluator_iam_policy(
            changed,
            bucket="private",
            prefix="tracks/test",
        )


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


def test_evaluator_credentials_are_separate_and_do_not_fallback(monkeypatch) -> None:
    captured = []

    def client(endpoint, access_key, secret_key):
        captured.append((endpoint, access_key, secret_key))
        return MemoryS3()

    monkeypatch.setenv("PRIVATE_STORAGE_URL", "https://producer.example/s3")
    monkeypatch.setenv("PRIVATE_ACCESS_KEY", "producer-access")
    monkeypatch.setenv("PRIVATE_SECRET_ACCESS_KEY", "producer-secret")
    monkeypatch.setattr("validator.production.storage._environment_client", client)

    with pytest.raises(ValueError, match="evaluator private storage credentials"):
        PrivateObjectStore.from_evaluator_environment(bucket="private", prefix="tracks/abc")

    monkeypatch.setenv("EVALUATOR_PRIVATE_STORAGE_URL", "https://evaluator.example/s3")
    monkeypatch.setenv("EVALUATOR_PRIVATE_ACCESS_KEY", "evaluator-access")
    monkeypatch.setenv("EVALUATOR_PRIVATE_SECRET_ACCESS_KEY", "evaluator-secret")
    store = PrivateObjectStore.from_evaluator_environment(
        bucket="private", prefix="tracks/abc"
    )

    assert store._credential_role == "evaluator"
    assert captured == [("https://evaluator.example/s3", "evaluator-access", "evaluator-secret")]


def test_evaluator_boundary_requires_real_access_denials() -> None:
    hidden = Artifact("hidden.bin", "0" * 64, 1, "source-pool/assets/hidden")

    class DeniedS3(MemoryS3):
        def get_object(self, *, Bucket, Key):
            if "/source-pool/" in Key or Key.endswith("bootstrap/source_pool.json"):
                error = RuntimeError("denied")
                error.response = {
                    "Error": {"Code": "AccessDenied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                }
                raise error
            return super().get_object(Bucket=Bucket, Key=Key)

        def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
            if "/source-pool" in Prefix:
                error = RuntimeError("denied")
                error.response = {
                    "Error": {"Code": "AccessDenied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                }
                raise error
            return {"Contents": [], "IsTruncated": False}

    client = DeniedS3()
    store = PrivateObjectStore(
        client,
        bucket="private",
        prefix="tracks/abc",
        credential_role="evaluator",
    )

    store.verify_evaluator_boundary((hidden,))

    permissive = PrivateObjectStore(
        MemoryS3(),
        bucket="private",
        prefix="tracks/abc",
        credential_role="evaluator",
    )
    with pytest.raises(StorageError, match="denial could not be verified"):
        permissive.verify_evaluator_boundary((hidden,))


def test_evaluator_boundary_rejects_readable_noninitial_hidden_chunk() -> None:
    hidden = "source-pool/assets/hidden"
    artifact = Artifact("hidden.bin", "0" * 64, chunks.CHUNK_BYTES + 1, hidden)

    class IncompleteChunkDenialS3(MemoryS3):
        def get_object(self, *, Bucket, Key):
            if (
                Key.endswith("bootstrap/source_pool.json")
                or Key.endswith(hidden)
                or Key.endswith(f"{hidden}.parts/part-00000")
            ):
                error = RuntimeError("denied")
                error.response = {
                    "Error": {"Code": "AccessDenied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                }
                raise error
            return super().get_object(Bucket=Bucket, Key=Key)

        def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
            if Prefix.endswith("source-pool"):
                error = RuntimeError("denied")
                error.response = {
                    "Error": {"Code": "AccessDenied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                }
                raise error
            return {"Contents": [], "IsTruncated": False}

    client = IncompleteChunkDenialS3()
    part_key = "tracks/abc/source-pool/assets/hidden.parts/part-00001"
    client.objects[("private", part_key)] = (b"hidden", "etag", {})
    store = PrivateObjectStore(
        client,
        bucket="private",
        prefix="tracks/abc",
        credential_role="evaluator",
    )

    with pytest.raises(StorageError, match="can read a hidden object"):
        store.verify_evaluator_boundary((artifact,))


def test_injected_store_client_cannot_be_serialized() -> None:
    store = PrivateObjectStore(MemoryS3(), bucket="private", prefix="tracks/abc")

    with pytest.raises(TypeError, match="cannot cross processes"):
        pickle.dumps(store)


def test_s3_transport_disables_unsupported_optional_checksums() -> None:
    client = client_config()

    assert client.request_checksum_calculation == "when_required"
    assert client.response_checksum_validation == "when_required"


@pytest.mark.parametrize("region", [None, "", "us-east-1", "eu-central-1", "auto"])
def test_s3_transport_uses_standard_region_configuration(monkeypatch, region) -> None:
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    if region is not None:
        monkeypatch.setenv("AWS_DEFAULT_REGION", region)

    assert client_config().region_name == (region or "auto")


@pytest.mark.parametrize("role", ["public", "producer", "evaluator"])
def test_environment_clients_share_s3_region_without_changing_object_paths(monkeypatch, role):
    import boto3

    captured = []

    def client(service, **arguments):
        assert service == "s3"
        captured.append(arguments)
        return MemoryS3()

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    for prefix in ("", "PRIVATE_", "EVALUATOR_PRIVATE_"):
        monkeypatch.setenv(f"{prefix}STORAGE_URL", "https://s3.eu-central-1.amazonaws.com")
        monkeypatch.setenv(f"{prefix}ACCESS_KEY", "access")
        monkeypatch.setenv(f"{prefix}SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("VEDA_RELEASE_ACCEPTANCE", raising=False)
    monkeypatch.delenv("VEDA_COORDINATED_RELEASE_DIGEST", raising=False)
    if role == "public":
        store = S3Repository.from_environment(bucket="public", prefix="tracks/main")
    elif role == "evaluator":
        store = PrivateObjectStore.from_evaluator_environment(
            bucket="private", prefix="tracks/main"
        )
    else:
        store = PrivateObjectStore.from_environment(bucket="private", prefix="tracks/main")

    assert len(captured) == 1
    arguments = captured[0]
    assert arguments.get("region_name", arguments["config"].region_name) == "eu-central-1"
    assert arguments["endpoint_url"] == "https://s3.eu-central-1.amazonaws.com"
    assert store._prefix == "tracks/main"


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


def test_private_upload_verifies_bytes_before_committing_manifest(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()

    class CorruptingS3(MemoryS3):
        def put_object(self, **kwargs):
            if kwargs["Key"].endswith("part-00000"):
                kwargs["Body"] = b"other"
            return super().put_object(**kwargs)

    client = CorruptingS3()
    store = PrivateObjectStore(client, bucket="private", prefix="root")
    with pytest.raises(StorageError, match="conflicts with manifest"):
        store.put_content(source, sha256=digest, object_name="model/manifest.json")
    assert ("private", "root/model/manifest.json") not in client.objects
    assert client.head_object(Bucket="private", Key="root/model/part-00000")["Metadata"] == {
        "sha256": digest
    }


def test_private_upload_rechecks_existing_chunk_bytes_not_just_metadata(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    client = MemoryS3()
    store = PrivateObjectStore(client, bucket="private", prefix="root")
    store.put_content(source, sha256=digest, object_name="model/manifest.json")
    key = ("private", "root/model/part-00000")
    _, etag, metadata = client.objects[key]
    client.objects[key] = (b"other", etag, metadata)

    with pytest.raises(StorageError, match="conflicts with manifest"):
        PrivateObjectStore(client, bucket="private", prefix="root").put_content(
            source, sha256=digest, object_name="model/manifest.json"
        )
    assert client.objects[key][0] == b"other"


@pytest.mark.parametrize("winner", [b"model", b"other"])
def test_private_chunk_creation_race_preserves_and_verifies_the_winner(tmp_path, winner):
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    conditions = []

    class RacingS3(MemoryS3):
        def put_object(self, **kwargs):
            if kwargs["Key"].endswith("part-00000"):
                conditions.append(kwargs.get("IfNoneMatch"))
                super().put_object(**{**kwargs, "Body": winner})
            return super().put_object(**kwargs)

    client = RacingS3()
    store = PrivateObjectStore(client, bucket="private", prefix="root")
    if winner == b"other":
        with pytest.raises(StorageError, match="conflicts with manifest"):
            store.put_content(source, sha256=digest, object_name="model/manifest.json")
        assert ("private", "root/model/manifest.json") not in client.objects
    else:
        store.put_content(source, sha256=digest, object_name="model/manifest.json")
    assert conditions == ["*"]
    assert client.objects[("private", "root/model/part-00000")][0] == winner


def test_private_upload_and_restore_stream_and_close_actual_chunk_bodies(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"x" * 150_000)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    bodies = []

    class BoundedBody(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 64 * 1024
            return super().read(size)

    class StreamingS3(MemoryS3):
        def get_object(self, **kwargs):
            response = super().get_object(**kwargs)
            if kwargs["Key"].endswith("part-00000"):
                payload = self.objects[(kwargs["Bucket"], kwargs["Key"])][0]
                response["Body"] = BoundedBody(payload)
                bodies.append(response["Body"])
            return response

    store = PrivateObjectStore(StreamingS3(), bucket="private", prefix="root")
    store.put_content(source, sha256=digest, object_name="model/manifest.json")
    restored = store.download_content(
        "model/manifest.json", tmp_path / "restored.bin", sha256=digest, size_bytes=150_000,
    )
    assert restored.read_bytes() == source.read_bytes()
    assert len(bodies) == 2
    assert all(body.closed for body in bodies)


@pytest.mark.parametrize("mirror", [False, True])
def test_warm_artifact_sync_reads_local_bytes_once_before_final_verification(
    tmp_path, monkeypatch, mirror,
):
    source = tmp_path / "model.bin"
    data = b"runtime bytes"
    source.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = ArtifactManifest((Artifact("model.bin", digest, len(data), f"artifacts/{digest}"),))
    store = PrivateObjectStore(MemoryS3(), bucket="private", prefix="root")
    store.put_content(source, sha256=digest, object_name=f"artifacts/{digest}")
    read_bytes = 0
    original_open = type(source).open

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, *args, **kwargs):
            nonlocal read_bytes
            content = self.stream.read(*args, **kwargs)
            read_bytes += len(content)
            return content

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

    def tracked_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return Reader(stream) if path == source else stream

    monkeypatch.setattr(type(source), "open", tracked_open)
    if mirror:
        store.ensure_artifacts(manifest, tmp_path)
    else:
        store.restore_artifacts(manifest, tmp_path, logical_names=("model.bin",))
    assert read_bytes == len(data)
    manifest.verify_local(tmp_path)
    assert read_bytes == 2 * len(data)


@pytest.mark.parametrize("mirror", [False, True])
def test_artifact_sync_restores_same_size_corrupted_local_bytes(tmp_path, mirror):
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    manifest = ArtifactManifest((Artifact("model.bin", digest, 5, f"artifacts/{digest}"),))
    store = PrivateObjectStore(MemoryS3(), bucket="private", prefix="root")
    store.put_content(source, sha256=digest, object_name=f"artifacts/{digest}")
    source.write_bytes(b"other")

    if mirror:
        store.ensure_artifacts(manifest, tmp_path)
    else:
        store.restore_artifacts(manifest, tmp_path, logical_names=("model.bin",))

    assert source.read_bytes() == b"model"
    manifest.verify_local(tmp_path)


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
