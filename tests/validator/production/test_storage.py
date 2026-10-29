import hashlib
import io
import json
import os
import pickle
import stat
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from validator import chunks
from validator.production import storage as storage_module
from validator.production.commitment import Artifact, ArtifactManifest
from validator.production.storage import (
    RETRY_STATE_NAMESPACE,
    LocalBatchStore,
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

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=None, StartAfter=None):
        contents = [
            {"Key": key} for bucket, key in sorted(self.objects)
            if bucket == Bucket and key.startswith(Prefix)
            and (StartAfter is None or key > StartAfter)
        ]
        return {"Contents": contents if MaxKeys is None else contents[:MaxKeys],
                "IsTruncated": MaxKeys is not None and len(contents) > MaxKeys}


def test_private_listing_supports_bounded_original_index_pages():
    client = MemoryS3()
    store = PrivateObjectStore(client, bucket="private", prefix="evaluation")
    for index in range(40):
        store.create_plan(f"uploads/{index:064x}.json", b"metadata")
    first = store.list_names("uploads/", limit=16)
    second = store.list_names("uploads/", limit=16, after=first[-1])
    last = store.list_names("uploads/", limit=16, after=second[-1])
    assert len(first) == len(second) == 16 and len(last) == 8
    assert (*first, *second, *last) == store.list_names("uploads/")
    assert store.list_names("uploads/", limit=16, after=last[-1]) == ()
    with pytest.raises(ValueError, match="outside"):
        store.list_names("uploads/", limit=16, after="plans/1.json")
    with pytest.raises(ValueError, match="positive"):
        store.list_names("uploads/", limit=0)


@pytest.mark.parametrize("directory", [
    "plans", "plans/", "uploads/", "attempts/digest/", "source-pool/hidden.parts/",
])
def test_private_directory_listing_preserves_scoped_prefix(directory):
    root = "evaluation/tracks/data/"
    relative = directory.strip("/")
    calls = []

    class ScopedS3(MemoryS3):
        def list_objects_v2(self, *, Bucket, Prefix, **kwargs):
            calls.append(Prefix)
            if Prefix != root + relative + "/":
                raise PermissionError("directory capability requires trailing slash")
            return super().list_objects_v2(Bucket=Bucket, Prefix=Prefix, **kwargs)

    client = ScopedS3()
    store = PrivateObjectStore(client, bucket="private", prefix=root)
    store.create_plan(relative + "/one.json", b"original")
    store.create_plan(relative + "-sibling/two.json", b"unrelated")
    store.create_plan(relative + ".json", b"directory marker is not an entry")
    assert store.list_names(directory) == (relative + "/one.json",)
    assert calls == [root + relative + "/"]


def test_private_directory_listing_resumes_seventeenth_upload_across_service_pages():
    root = "evaluation/"
    calls = []

    class PagedS3(MemoryS3):
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=None, StartAfter=None,
                            ContinuationToken=None):
            assert Prefix == root + "uploads/"
            assert StartAfter is None or ContinuationToken is None
            calls.append((MaxKeys, StartAfter, ContinuationToken))
            contents = super().list_objects_v2(
                Bucket=Bucket, Prefix=Prefix, StartAfter=StartAfter,
            )["Contents"]
            # Opaque service pages can be shorter than the requested limit.
            if ContinuationToken is not None:
                contents = [item for item in contents if item["Key"] > ContinuationToken]
            page = contents[:min(3, MaxKeys) if MaxKeys is not None else 3]
            response = {"Contents": page, "IsTruncated": len(page) < len(contents)}
            if response["IsTruncated"]:
                response["NextContinuationToken"] = page[-1]["Key"]
            return response

    store = PrivateObjectStore(PagedS3(), bucket="private", prefix=root)
    expected = tuple(f"uploads/{index:064x}.json" for index in range(17))
    for name in reversed(expected):
        store.create_plan(name, b"metadata")
    store.create_plan("uploads-sibling/other.json", b"unrelated")
    first = store.list_names("uploads/", limit=16)
    assert first == expected[:16]
    assert [limit for limit, _, _ in calls] == [16, 13, 10, 7, 4, 1]
    assert all(after is None for _, after, _ in calls)
    assert all(token is not None for _, _, token in calls[1:])
    assert store.list_names("uploads/", limit=16, after=first[-1]) == expected[16:]
    assert calls[-1] == (16, root + expected[15], None)
    assert store.list_names("uploads/", limit=16, after=expected[-1]) == ()
    assert store.list_names("uploads/") == expected
    with pytest.raises(ValueError, match="outside"):
        store.list_names("uploads/", limit=16, after="uploads-sibling/other.json")


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
    assert "/tracks/test/disclosed/*" in encoded
    for namespace in ("videos", "latents", "inputs", "exact", "artifacts"):
        assert f"/tracks/test/{namespace}/*" not in encoded
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


@pytest.mark.parametrize(("leak", "shared_client"), [
    ("bootstrap/source_pool.json", False),
    ("source-pool/assets/hidden", False),
    ("source-pool/assets/hidden.parts/part-00000", False),
    ("source-pool/assets/hidden.parts/part-00001", False),
    ("source-pool/", False),
    ("source-pool/assets/hidden.parts/", False),
    (None, False),
    (None, True),
])
def test_evaluator_boundary_checks_each_distinct_capability(leak, shared_client) -> None:
    root = "tracks/abc/"
    object_names = [
        "bootstrap/source_pool.json",
        "source-pool/assets/hidden",
        "source-pool/assets/hidden.parts/part-00000",
        "source-pool/assets/hidden.parts/part-00001",
    ]
    prefixes = ["source-pool/", "source-pool/assets/hidden.parts/"]

    class CapabilityS3(MemoryS3):
        def __init__(self, readable):
            super().__init__()
            self.readable = readable
            self.calls = []

        def deny(self):
            error = RuntimeError("denied")
            error.response = {
                "Error": {"Code": "AccessDenied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            }
            raise error

        def get_object(self, *, Bucket, Key):
            self.calls.append(("get", Bucket, Key))
            assert store._client is read_client
            assert store._write_client is write_client
            if self.readable is not None and Key == root + self.readable:
                return {"Body": io.BytesIO(b"hidden"), "ETag": '"etag"'}
            self.deny()

        def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
            self.calls.append(("list", Bucket, Prefix))
            assert store._client is read_client
            assert store._write_client is write_client
            if self.readable is not None and Prefix == root + self.readable:
                return {"Contents": [], "IsTruncated": False}
            self.deny()

    read_client = CapabilityS3(None)
    write_client = read_client if shared_client else CapabilityS3(leak)
    store = PrivateObjectStore(
        read_client, write_client=write_client, bucket="private", prefix="tracks/abc",
        credential_role="evaluator",
    )
    artifact = Artifact("hidden.bin", "0" * 64, chunks.CHUNK_BYTES + 1,
                        "source-pool/assets/hidden")
    if leak is not None:
        kind = "list a hidden prefix" if leak in prefixes else "read a hidden object"
        with pytest.raises(StorageError, match=kind):
            store.verify_evaluator_boundary((artifact,))
    else:
        store.verify_evaluator_boundary((artifact,))
    expected = [("get", "private", root + name) for name in object_names]
    expected += [("list", "private", root + prefix) for prefix in prefixes]
    assert read_client.calls == expected
    if not shared_client:
        if leak is not None:
            operation = "list" if leak in prefixes else "get"
            expected = expected[:expected.index((operation, "private", root + leak)) + 1]
        assert write_client.calls == expected


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


def test_split_private_clients_keep_reads_and_conditional_writes_separate(tmp_path):
    calls = []
    backend = MemoryS3()

    class Read:
        def __getattr__(self, operation):
            assert operation in {"get_object", "head_object", "list_objects_v2"}

            def read(**kwargs):
                calls.append(("read", operation))
                if "source_pool" in kwargs.get("Key", ""):
                    raise PermissionError("hidden source denied by fixture capability")
                return getattr(backend, operation)(**kwargs)
            return read

    class Write:
        def __getattr__(self, operation):
            assert operation in {"put_object", "delete_object"}

            def write(**kwargs):
                calls.append(("write", operation))
                key = kwargs["Key"]
                if "/plans/" in key and not key.endswith("/plans/allocations.json"):
                    raise PermissionError("producer plans are not writable")
                return getattr(backend, operation)(**kwargs)
            return write

    backend.put_object(Bucket="affinemodel", Key="evaluation/plans/100.json", Body=b"original")
    store = PrivateObjectStore(Read(), write_client=Write(), bucket="affinemodel",
                               prefix="evaluation", credential_role="evaluator")
    assert store.get("plans/100.json").body == b"original"
    assert store.list_names("plans") == ("plans/100.json",)
    with pytest.raises(PermissionError, match="not writable"):
        store.create_plan("plans/100.json", b"changed")
    allocation = store.create_plan("plans/allocations.json", b"one")
    store.replace_plan("plans/allocations.json", b"two", etag=allocation.etag)
    with pytest.raises(PreconditionFailed):
        store.replace_plan("plans/allocations.json", b"stale", etag=allocation.etag)
    with pytest.raises(PermissionError, match="hidden"):
        store.get("bootstrap/source_pool.json")
    store.verify_conditional_writes()  # Including cleanup Delete, never via the RO client.
    model = tmp_path / "small-fixture"
    model.write_bytes(b"synthetic legacy archive fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    store.put_content(model, sha256=digest, object_name=f"models/{digest}")
    assert store.download_content(f"models/{digest}", tmp_path / "restored",
                                   sha256=digest, size_bytes=model.stat().st_size).read_bytes() == (
        model.read_bytes()
    )
    upload = store.with_prefix("evaluation/uploads")
    assert upload._client is store._client and upload._write_client is store._write_client
    assert backend.objects[("affinemodel", "evaluation/plans/100.json")][0] == b"original"
    assert {kind for kind, _op in calls} == {"read", "write"}


@pytest.mark.parametrize("role", ["producer", "evaluator"])
def test_named_profiles_reconstruct_by_role_without_pickling_credentials(monkeypatch, role):
    calls = []

    def client(endpoint, *args, profile_name=None):
        assert endpoint == "https://private.example" and not args
        calls.append(profile_name)
        return MemoryS3()

    monkeypatch.setenv("PRIVATE_STORAGE_URL", "https://private.example")
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    monkeypatch.setenv("PRIVATE_AWS_PROFILE", "producer")
    monkeypatch.setenv("EVALUATOR_PRIVATE_READ_PROFILE", "read")
    monkeypatch.setenv("EVALUATOR_PRIVATE_WRITE_PROFILE", "write")
    monkeypatch.setattr(storage_module, "_environment_client", client)
    factory = (PrivateObjectStore.from_environment if role == "producer" else
               PrivateObjectStore.from_evaluator_environment)
    store = factory(bucket="affinemodel", prefix="evaluation")
    restored = pickle.loads(pickle.dumps(store))
    assert restored._credential_role == role
    assert set(restored.__getstate__()) == {
        "_bucket", "_prefix", "_credential_role", "_reconnect_from_environment",
    }
    assert calls == (["producer"] * 2 if role == "producer" else ["read", "write"] * 2)
    assert (restored._client is restored._write_client) == (role == "producer")


@pytest.mark.parametrize("bad", ["missing-write", "empty", "keys", "duplicate-url", "bucket"])
def test_profile_role_configuration_never_falls_back_to_explicit_keys(monkeypatch, bad):
    monkeypatch.setenv("PRIVATE_STORAGE_URL", "https://private.example")
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    monkeypatch.setenv("EVALUATOR_PRIVATE_READ_PROFILE", "read")
    monkeypatch.setenv("EVALUATOR_PRIVATE_WRITE_PROFILE", "write")
    if bad == "missing-write":
        monkeypatch.delenv("EVALUATOR_PRIVATE_WRITE_PROFILE")
    elif bad == "empty":
        monkeypatch.setenv("EVALUATOR_PRIVATE_READ_PROFILE", " ")
    elif bad == "keys":
        monkeypatch.setenv("EVALUATOR_PRIVATE_ACCESS_KEY", "fixture-old-key")
    elif bad == "duplicate-url":
        monkeypatch.setenv("EVALUATOR_PRIVATE_STORAGE_URL", "https://another-authority.example")
    else:
        monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "another-bucket")
    monkeypatch.setattr(storage_module, "_environment_client",
                        lambda *_a, **_k: pytest.fail("invalid profile must not create a client"))
    with pytest.raises(ValueError):
        PrivateObjectStore.from_evaluator_environment(bucket="affinemodel", prefix="evaluation")


@pytest.mark.parametrize("bad", ["missing-profile", "missing-private-address", "legacy-mix"])
def test_private_producer_never_uses_public_keys_for_profile_configuration(monkeypatch, bad):
    monkeypatch.setenv("STORAGE_URL", "https://public.example")
    monkeypatch.setenv("ACCESS_KEY", "fixture-public-key")
    monkeypatch.setenv("SECRET_ACCESS_KEY", "fixture-public-secret")
    monkeypatch.setenv("PRIVATE_STORAGE_URL", "https://private.example")
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    if bad != "missing-profile":
        monkeypatch.setenv("PRIVATE_AWS_PROFILE", "producer")
    if bad == "missing-private-address":
        monkeypatch.delenv("PRIVATE_STORAGE_URL")
    elif bad == "legacy-mix":
        monkeypatch.setenv("PRIVATE_ACCESS_KEY", "fixture-private-key")
    monkeypatch.setattr(storage_module, "_environment_client",
                        lambda *_a, **_k: pytest.fail("must not fall back to public credentials"))
    with pytest.raises((KeyError, ValueError)):
        PrivateObjectStore.from_environment(bucket="affinemodel", prefix="evaluation")


def _native_process_profile(tmp_path, monkeypatch):
    import shlex
    import sys

    provider = tmp_path / "fixture_credentials.py"
    counter = tmp_path / "invocations"
    provider.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[1]); n=int(p.read_text())+1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "print(json.dumps(dict(Version=1, AccessKeyId='fixture-access', "
        "SecretAccessKey='fixture-secret', SessionToken='fixture-session-'+str(n), "
        "Expiration='2050-01-01T%02d:00:00Z'%n)))\n",
    )
    profile = tmp_path / "config"
    command = shlex.join([sys.executable, str(provider), str(counter)])
    profile.write_text("[profile fixture]\nregion = auto\ncredential_process = " + command + "\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(profile))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-static-keys"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-key-must-not-win")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret-must-not-win")
    return counter


def test_sdk_process_profile_refreshes_session_on_same_client_without_network(
    tmp_path, monkeypatch,
):
    from datetime import UTC, datetime

    from botocore.exceptions import ProfileNotFound

    counter = _native_process_profile(tmp_path, monkeypatch)
    client = storage_module._environment_client("https://private.example", profile_name="fixture")
    try:
        credentials = client._request_signer._credentials
        assert credentials.method == "custom-process"
        assert credentials.get_frozen_credentials().token == "fixture-session-1"
        credentials._time_fetcher = lambda: datetime(2050, 1, 1, 0, 59, 59, tzinfo=UTC)
        assert credentials.get_frozen_credentials().token == "fixture-session-2"
        assert counter.read_text() == "2"  # SDK invokes the real fixture process on refresh.
        with pytest.raises(ProfileNotFound):
            storage_module._environment_client("https://private.example", profile_name="missing")
        with pytest.raises(ValueError, match="mixed"):
            storage_module._environment_client("https://private.example", "old", "keys",
                                               profile_name="fixture")
    finally:
        client.close()


def _inspect_spawned_profile_store(store, output):
    output.put((store._credential_role, store._prefix, store._client.meta.endpoint_url,
                store._client._request_signer._credentials.method,
                store._write_client._request_signer._credentials.method))


def test_named_profiles_rebuild_in_real_spawn_child(tmp_path, monkeypatch):
    import multiprocessing

    counter = _native_process_profile(tmp_path, monkeypatch)
    monkeypatch.setenv("PRIVATE_STORAGE_URL", "https://private.example")
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    monkeypatch.setenv("EVALUATOR_PRIVATE_READ_PROFILE", "fixture")
    monkeypatch.setenv("EVALUATOR_PRIVATE_WRITE_PROFILE", "fixture")
    store = PrivateObjectStore.from_evaluator_environment(bucket="affinemodel", prefix="evaluation")
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    child = context.Process(target=_inspect_spawned_profile_store, args=(store, output))
    try:
        child.start()
        assert output.get(timeout=20) == ("evaluator", "evaluation", "https://private.example",
                                         "custom-process", "custom-process")
        child.join(timeout=20)
        assert child.exitcode == 0
        assert counter.read_text() == "4"  # Two native providers in each process.
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        output.close()
        output.join_thread()
        store._client.close()
        store._write_client.close()


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


def test_local_batch_content_is_shared_without_s3_and_detects_corruption(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    digest = hashlib.sha256(b"video").hexdigest()
    producer = LocalBatchStore(tmp_path / "content")
    sealed = producer.put(source, sha256=digest, size_bytes=5)
    assert sealed.stat().st_ino == source.stat().st_ino
    evaluator = LocalBatchStore(tmp_path / "content")
    view = evaluator.materialize(tmp_path / "view/video.mp4", sha256=digest, size_bytes=5)
    assert view.is_symlink() and view.read_bytes() == b"video"
    source.unlink()
    assert evaluator.get(sha256=digest, size_bytes=5) == sealed
    sealed.write_bytes(b"other")
    with pytest.raises(StorageError, match="SHA-256"):
        evaluator.get(sha256=digest, size_bytes=5)


def test_local_batch_seals_actual_0600_cache_as_group_readable_without_copy(tmp_path, monkeypatch):
    from tests.scoring.test_cache import write_batch

    previous = os.umask(0o077)
    try:
        batch = write_batch(tmp_path / "batch")
        before = batch.path.stat()
        assert stat.S_IMODE(before.st_mode) == 0o600  # Actual NamedTemporaryFile chain.
        monkeypatch.setattr(storage_module.shutil, "copyfileobj",
                            lambda *_a, **_k: pytest.fail("same-device seal must not copy bytes"))
        sealed = LocalBatchStore(tmp_path / "content").put(
            batch.path, sha256=batch.sha256, size_bytes=before.st_size,
        )
    finally:
        os.umask(previous)
    after = sealed.stat()
    assert stat.S_IMODE(after.st_mode) == 0o640
    assert (after.st_ino, after.st_gid, after.st_uid) == (
        before.st_ino, before.st_gid, before.st_uid,
    )
    assert hashlib.sha256(sealed.read_bytes()).hexdigest() == batch.sha256
    view = LocalBatchStore(tmp_path / "content").materialize(
        tmp_path / "eval/batch.npz", sha256=batch.sha256, size_bytes=before.st_size,
    )
    assert view.is_symlink() and view.resolve() == sealed.resolve()


@pytest.mark.parametrize("mode", [0o644, 0o660])
def test_local_batch_new_seal_has_no_group_write_or_world_access(tmp_path, mode):
    source = tmp_path / "complete"
    source.write_bytes(b"body")
    source.chmod(mode)
    sealed = LocalBatchStore(tmp_path / "content").put(
        source, sha256=hashlib.sha256(b"body").hexdigest(), size_bytes=4,
    )
    assert sealed.stat().st_ino == source.stat().st_ino
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o640


def test_local_batch_existing_cross_device_fallback_also_seals_0640(tmp_path, monkeypatch):
    import errno

    source = tmp_path / "complete"
    source.write_bytes(b"body")
    source.chmod(0o600)
    real_link = os.link

    def cross_device(path, destination):
        if Path(path) == source:
            raise OSError(errno.EXDEV, "fixture cross-device link")
        return real_link(path, destination)

    monkeypatch.setattr(os, "link", cross_device)
    sealed = LocalBatchStore(tmp_path / "content").put(
        source, sha256=hashlib.sha256(b"body").hexdigest(), size_bytes=4,
    )
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o640
    assert sealed.read_bytes() == b"body"


def test_local_batch_already_shared_source_needs_no_permission_mutation(tmp_path, monkeypatch):
    source = tmp_path / "complete"
    source.write_bytes(b"body")
    source.chmod(0o640)
    store = LocalBatchStore(tmp_path / "content")
    digest = hashlib.sha256(b"body").hexdigest()
    store.put(source, sha256=digest, size_bytes=4)
    monkeypatch.setattr(os, "fchmod", lambda *_a: pytest.fail("already shared, no owner chmod"))
    assert store.put(source, sha256=digest, size_bytes=4).read_bytes() == b"body"


def test_local_batch_rejected_source_is_not_made_group_readable(tmp_path):
    source = tmp_path / "wrong"
    source.write_bytes(b"body")
    source.chmod(0o600)
    with pytest.raises(StorageError, match="descriptor"):
        LocalBatchStore(tmp_path / "content").put(source, sha256="00" * 32, size_bytes=4)
    assert stat.S_IMODE(source.stat().st_mode) == 0o600


@pytest.mark.parametrize("age_after_verification_ns", [0, 2_000_000_000])
def test_local_batch_fresh_hash_is_not_reused_across_equal_timestamp_writes(
    tmp_path, monkeypatch, age_after_verification_ns,
):
    source = tmp_path / "source"
    source.write_bytes(b"video")
    digest = hashlib.sha256(b"video").hexdigest()
    sealed = LocalBatchStore(tmp_path / "content").put(source, sha256=digest, size_bytes=5)
    timestamp = sealed.stat()
    now = [max(timestamp.st_mtime_ns, timestamp.st_ctime_ns) + 100]
    monkeypatch.setattr(storage_module, "time_ns", lambda: now[0], raising=False)
    real_stat = Path.stat

    def coarse_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path != sealed:
            return result
        # Reproduce observed same-tick writes without timing-sensitive sleeps.
        return SimpleNamespace(
            st_mode=result.st_mode, st_ino=result.st_ino, st_size=result.st_size,
            st_mtime_ns=timestamp.st_mtime_ns, st_ctime_ns=timestamp.st_ctime_ns,
        )

    monkeypatch.setattr(Path, "stat", coarse_stat)
    evaluator = LocalBatchStore(tmp_path / "content")
    assert evaluator.get(sha256=digest, size_bytes=5).read_bytes() == b"video"
    sealed.write_bytes(b"other")
    now[0] += age_after_verification_ns
    with pytest.raises(StorageError, match="SHA-256"):
        evaluator.get(sha256=digest, size_bytes=5)


def test_local_batch_stable_content_reuses_hash_but_rechecks_changed_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"video")
    digest = hashlib.sha256(b"video").hexdigest()
    sealed = LocalBatchStore(tmp_path / "content").put(source, sha256=digest, size_bytes=5)
    timestamp = sealed.stat()
    monkeypatch.setattr(
        storage_module, "time_ns",
        lambda: max(timestamp.st_mtime_ns, timestamp.st_ctime_ns) + 2_000_000_000,
        raising=False,
    )
    calls = []
    file_digest = hashlib.file_digest

    def tracked_hash(*args, **kwargs):
        calls.append(True)
        return file_digest(*args, **kwargs)

    monkeypatch.setattr(storage_module.hashlib, "file_digest", tracked_hash)
    evaluator = LocalBatchStore(tmp_path / "content")
    for _ in range(10):
        assert evaluator.get(sha256=digest, size_bytes=5) == sealed
    assert len(calls) == 1
    # Change inode/bytes, even if a filesystem supplies the same timestamps.
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"other")
    replacement.replace(sealed)
    with pytest.raises(StorageError, match="SHA-256"):
        evaluator.get(sha256=digest, size_bytes=5)
    assert len(calls) == 2


def test_local_batch_loss_requires_regeneration_and_never_reads_private_s3(tmp_path):
    store = LocalBatchStore(tmp_path / "content")
    digest = hashlib.sha256(b"input").hexdigest()
    with pytest.raises(StorageError, match="regeneration"):
        store.get(sha256=digest, size_bytes=5)
    path = tmp_path / "regenerated"
    path.write_bytes(b"input")
    store.put(path, sha256=digest, size_bytes=5)
    assert store.get(sha256=digest, size_bytes=5).read_bytes() == b"input"
    with pytest.raises(StorageError, match="descriptor"):
        store.put(path, sha256="00" * 32, size_bytes=5)


@pytest.mark.parametrize("namespace", ["exact", "inputs", "latents", "videos"])
def test_private_store_rejects_retired_hidden_bulk_transport(tmp_path, namespace):
    client = MemoryS3()
    store = PrivateObjectStore(client, bucket="private", prefix="root")
    with pytest.raises(StorageError, match="GPU-local"):
        store.put_content(tmp_path / "absent", sha256="ab" * 32, object_name=f"{namespace}/a")
    with pytest.raises(StorageError, match="GPU-local"):
        store.download_content(
            f"{namespace}/a", tmp_path / "absent", sha256="ab" * 32, size_bytes=1,
        )
    assert client.objects == {}


def test_small_bootstrap_restore_does_not_upload_runtime_artifacts(
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
    path.unlink()
    bootstrap.unlink()

    restored_payload = store.ensure_bootstrap("bootstrap/artifact_manifest.json", bootstrap)

    assert restored_payload == manifest.encode()
    assert not path.exists()
    assert len(client.objects) == 1
    assert payload == restored_payload
