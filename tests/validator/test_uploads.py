import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock

import numpy as np
import pytest
from bittensor.wallet import Keypair
from safetensors.numpy import save

from miner.submit import ModelSubmission, PrivateModelSubmission, UploadIntent
from scoring.model import TensorSpec, validate_model
from validator.chunks import CHUNK_BYTES, ChunkManifest, ChunkPart, build_chunk_manifest, read_part
from validator.production.storage import PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState, submission_id
from validator.store import ChallengeStore, SubmissionRecord
from validator.uploads import PrivateUploads

from .production.test_storage import MemoryS3

NOW = 1_000
KEY = Keypair.create_from_seed(bytes.fromhex("11" * 32), crypto_type=0)
SCHEMA = {"selector.weight": TensorSpec((2, 2), "float32")}


class StagingS3(MemoryS3):
    def __init__(self):
        super().__init__()
        self.grants = []

    def generate_presigned_url(self, operation, **kwargs):
        assert operation == "put_object"
        self.grants.append(kwargs)
        return "https://private.example/staging?secret=bearer"

    def get_object(self, **kwargs):
        result = super().get_object(**kwargs)
        result["ContentLength"] = len(self.objects[(kwargs["Bucket"], kwargs["Key"])][0])
        return result


@pytest.fixture
def setup(tmp_path):
    payload = save({"selector.weight": np.ones((2, 2), dtype=np.float32)},
                   metadata={"hotkey": KEY.ss58_address})
    reference = PrivateModelSubmission("ab" * 32, hashlib.sha256(payload).hexdigest())
    manifest = ChunkManifest(reference.file_sha256, len(payload), (
        ChunkPart("part-00000", reference.file_sha256, len(payload)),
    ))
    intent = UploadIntent(
        "https://auth.example/veda", "01" * 32, KEY.ss58_address, 120, "upload",
        reference.upload_id, reference.file_sha256, len(payload),
        hashlib.sha256(manifest.encode()).hexdigest(), NOW + 300,
    )
    staging = StagingS3()
    authority = MemoryS3()
    with ChallengeStore(tmp_path / "state.db") as store:
        store.advance_chain_cursor(100, "ab" * 32)
        uploads = PrivateUploads(
            objects=PrivateObjectStore(authority, bucket="private", prefix="test"),
            staging_client=staging, staging_bucket="uploads-private", staging_prefix="temporary",
            audience=intent.audience, genesis_hash=intent.genesis_hash, store=store,
            registration=lambda _hotkey: (9, 100), max_bytes=10_000, cache_dir=tmp_path / "seal",
        )
        yield uploads, intent, payload, store


def sign(intent):
    return KEY.sign(intent.signing_message())


def authorize(uploads, intent, signature=None, *, now):
    manifest = ChunkManifest(intent.file_sha256, intent.size_bytes, (
        ChunkPart("part-00000", intent.file_sha256, intent.size_bytes),
    ))
    return uploads.authorize(intent, sign(intent) if signature is None else signature,
                             manifest=manifest, now=now)


def upload(uploads, intent, payload):
    grant = authorize(uploads, intent, now=NOW)
    params = uploads.staging.grants[-1]["Params"]
    uploads.staging.put_object(
        Bucket=params["Bucket"], Key=params["Key"], Body=payload,
        # Deliberately untrustworthy metadata must not decide the file's hash.
        Metadata={"sha256": intent.file_sha256},
    )
    return grant, params


def formal_submission(store, hotkey=KEY.ss58_address, block=100):
    position = ChainPosition(block, 1, 1)
    block_hash = "ab" * 32
    model = ModelSubmission("owner/model", "12" * 20)
    identifier, _ = store.add_submission(SubmissionRecord(
        submission_id(120, block_hash, position), hotkey, 120, block_hash,
        position, model.encode(), model.repo_id, model.revision,
    ))
    return identifier


def test_real_hotkey_signature_retry_seal_and_original_archive_adapter(setup, tmp_path):
    uploads, intent, payload, store = setup
    grant, params = upload(uploads, intent, payload)
    before = dict(uploads.objects._client.objects)
    retried = authorize(uploads, intent, now=NOW + 1)
    assert before == uploads.objects._client.objects
    assert retried.expires_at == intent.expires_at
    assert uploads.staging.grants[-1]["ExpiresIn"] == 299
    assert "bearer" not in repr(grant)
    assert params == {
        "Bucket": "uploads-private", "Key": f"{uploads._key(intent)}/part-00000",
        "ContentLength": len(payload),
    }
    assert store.public_snapshot()["challenges"] == []  # retries consume no opportunity

    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    # An unexpired presigned URL is reusable; post-seal overwrites never affect source bytes.
    uploads.staging.put_object(Bucket=params["Bucket"], Key=params["Key"], Body=b"replaced")
    restored = uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cold",
                                max_bytes=10_000)
    assert restored.path.read_bytes() == payload
    assert restored.download_host is None
    assert validate_model(restored.path, schema=SCHEMA, max_bytes=10_000,
                          expected_hotkey=intent.hotkey).file_sha256 == reference.file_sha256
    assert uploads.seal(seal, sign(seal), now=NOW + 2) == reference
    with pytest.raises(ValueError, match="already sealed"):
        authorize(uploads, intent, now=NOW + 2)
    assert store.public_snapshot()["challenges"] == []


@pytest.mark.parametrize("mutation", [
    {"audience": "https://other.example/veda"}, {"genesis_hash": "02" * 32},
    {"expires_at": NOW}, {"expires_at": NOW + 301}, {"size_bytes": 10_001},
    {"file_sha256": "bb" * 32}, {"upload_id": "cd" * 32},
    {"hotkey": Keypair.create_from_seed(bytes.fromhex("22" * 32), crypto_type=0).ss58_address},
])
def test_tampering_expiry_cross_origin_chain_and_hotkey_fail_before_capability(setup, mutation):
    uploads, intent, _, _ = setup
    changed = replace(intent, **mutation)
    with pytest.raises(ValueError):
        authorize(uploads, changed, sign(intent), now=NOW)
    assert uploads.staging.grants == []
    assert uploads.objects._client.objects == {}


@pytest.mark.parametrize("uid", [None, 0, -1, True, 65536])
def test_registration_and_uid_zero_gate_new_permissions(setup, uid):
    uploads, intent, _, _ = setup
    uploads.registration = lambda _hotkey: (uid, 100)
    with pytest.raises(ValueError, match="registered"):
        authorize(uploads, intent, now=NOW)
    assert uploads.staging.grants == []


def test_unknown_or_unscanned_chain_state_cannot_issue_upload_permission(setup):
    uploads, intent, _, _ = setup
    uploads.registration = lambda _hotkey: (9, 101)
    with pytest.raises(RuntimeError, match="behind"):
        authorize(uploads, intent, now=NOW)
    def failed(_hotkey):
        raise RuntimeError("RPC unavailable")
    uploads.registration = failed
    with pytest.raises(RuntimeError, match="unavailable"):
        authorize(uploads, intent, now=NOW)
    assert uploads.staging.grants == []


def test_signed_legacy_exclusion_and_prior_hf_share_original_opportunity_gate(setup):
    uploads, intent, _, store = setup
    identifier = formal_submission(store)
    with pytest.raises(ValueError, match="formal submission"):
        authorize(uploads, intent, now=NOW)
    # Only the original, explicit system non-admission exception retains the chance.
    store.transition(identifier, ChallengeState.FETCHING)
    store.transition(identifier, ChallengeState.SYSTEM_NOT_ADMITTED,
                     error_code="NO_PRE_SUBMISSION_DATA")
    authorize(uploads, intent, now=NOW)
    store.seed_legacy_exclusions({intent.hotkey: 100})
    with pytest.raises(ValueError, match="excluded"):
        authorize(uploads, intent, now=NOW)


def test_hf_arriving_after_grant_prevents_sealing_and_never_resets_its_attempt(setup):
    uploads, intent, payload, store = setup
    upload(uploads, intent, payload)
    identifier = formal_submission(store)
    seal = replace(intent, action="seal")
    with pytest.raises(ValueError, match="formal submission"):
        uploads.seal(seal, sign(seal), now=NOW + 1)
    assert store.get_challenge(identifier).state is ChallengeState.QUEUED
    assert store.get_challenge(identifier).attempts == 0
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)


def test_registration_loss_after_upload_prevents_seal(setup):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    uploads.registration = lambda _hotkey: (None, 100)
    seal = replace(intent, action="seal")
    with pytest.raises(ValueError, match="registered"):
        uploads.seal(seal, sign(seal), now=NOW + 1)


def test_client_metadata_cannot_replace_real_full_sha_verification(setup):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, b"!" * len(payload))
    seal = replace(intent, action="seal")
    with pytest.raises(ValueError, match="chunk conflicts"):
        uploads.seal(seal, sign(seal), now=NOW + 1)
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)


def test_s3_stream_is_bounded_even_with_forged_content_length(setup, monkeypatch):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    body = io.BytesIO(payload + b"over limit")
    monkeypatch.setattr(uploads.staging, "get_object", lambda **_kwargs: {
        "ContentLength": len(payload), "Body": body,
    })
    with pytest.raises(ValueError, match="exceeds manifest size"):
        uploads.seal(seal, sign(seal), now=NOW + 1)
    assert body.closed


@pytest.mark.parametrize("lost_on", ["lease", "seal", "receipt"])
def test_committed_response_loss_recovers_exact_original_bytes_without_new_chance(setup, lost_on):
    uploads, intent, payload, store = setup
    client = uploads.objects._client
    original_put = client.put_object
    lost = False

    def put(**kwargs):
        nonlocal lost
        response = original_put(**kwargs)
        key = kwargs["Key"]
        document = json.loads(kwargs["Body"]) if key.endswith(".json") else {}
        matches = (
            lost_on == "lease" and document.get("state") == "UPLOADING"
            or lost_on == "seal" and document.get("state") == "SEALED"
            or lost_on == "receipt" and "/sealed/" in key
        )
        if matches and not lost:
            lost = True
            raise RuntimeError("response lost after server committed")
        return response

    client.put_object = put
    if lost_on == "lease":
        with pytest.raises(RuntimeError, match="response lost"):
            authorize(uploads, intent, now=NOW)
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    if lost_on != "lease":
        with pytest.raises(RuntimeError, match="response lost"):
            uploads.seal(seal, sign(seal), now=NOW + 1)
    reference = uploads.seal(seal, sign(seal), now=NOW + 2)
    receipt = uploads.objects.get(f"uploads/sealed/{reference.upload_id}.json")
    assert uploads.seal(seal, sign(seal), now=NOW + 3) == reference
    assert uploads.objects.get(f"uploads/sealed/{reference.upload_id}.json") == receipt
    assert store.public_snapshot()["challenges"] == []


def test_source_adapter_rejects_wrong_owner_digest_and_oversize(setup, tmp_path):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    for wrong_ref, owner, max_bytes in (
        (reference, "another-hotkey", 10_000),
        (replace(reference, file_sha256="bb" * 32), intent.hotkey, 10_000),
        (reference, intent.hotkey, len(payload) - 1),
    ):
        with pytest.raises(ValueError, match="submitting identity"):
            uploads.download(wrong_ref, hotkey=owner, cache_dir=tmp_path / "wrong",
                             max_bytes=max_bytes)
    assert not (tmp_path / "wrong").exists()


def test_action_signature_cannot_be_replayed_as_seal_and_expiry_cannot_extend(setup):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    with pytest.raises(ValueError, match="signature"):
        uploads.seal(seal, sign(intent), now=NOW + 1)
    with pytest.raises(ValueError, match="expiry"):
        authorize(uploads, intent, now=intent.expires_at)
    assert len(uploads.staging.grants) == 1


def test_concurrent_different_uploads_for_one_hotkey_only_issue_one_capability(setup):
    uploads, intent, _, store = setup
    barrier = Barrier(2)
    lock = Lock()
    client = uploads.objects._client
    original_get, original_put = client.get_object, client.put_object
    database = store._connection.execute("PRAGMA database_list").fetchone()[2]

    def get(**kwargs):
        if "/leases/" not in kwargs["Key"]:
            return original_get(**kwargs)
        try:
            result = original_get(**kwargs)
        except KeyError:
            barrier.wait(timeout=5)
            raise
        return result

    def put(**kwargs):
        with lock:
            return original_put(**kwargs)

    client.get_object, client.put_object = get, put

    def issue(candidate):
        with ChallengeStore(database) as connection:
            worker = PrivateUploads(
                objects=uploads.objects, staging_client=uploads.staging,
                staging_bucket=uploads.bucket, staging_prefix=uploads.prefix,
                audience=uploads.audience, genesis_hash=uploads.genesis_hash,
                store=connection, registration=uploads.registration,
                max_bytes=uploads.max_bytes, cache_dir=uploads.cache_dir,
            )
            try:
                return authorize(worker, candidate, now=NOW)
            except RuntimeError:
                return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(issue, (intent, replace(intent, upload_id="cd" * 32))))
    assert sum(result is not None for result in results) == 1
    assert len(uploads.staging.grants) == 1
    assert store.public_snapshot()["challenges"] == []


def test_sealed_response_and_bytes_recover_with_fresh_adapter_without_staging(setup, tmp_path):
    uploads, intent, payload, _ = setup
    _, params = upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    uploads.staging.delete_object(Bucket=params["Bucket"], Key=params["Key"])
    fresh = PrivateUploads(
        objects=uploads.objects, staging_client=uploads.staging,
        staging_bucket=uploads.bucket, staging_prefix=uploads.prefix,
        audience=uploads.audience, genesis_hash=uploads.genesis_hash,
        store=uploads.store, registration=uploads.registration,
        max_bytes=uploads.max_bytes, cache_dir=tmp_path / "fresh-seal",
    )
    assert fresh.seal(seal, sign(seal), now=NOW + 2) == reference
    restored = fresh.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "fresh-model",
                              max_bytes=10_000)
    assert restored.path.read_bytes() == payload


def test_reregistration_and_expired_upload_quota_do_not_reset_formal_opportunity(setup):
    uploads, intent, payload, store = setup
    upload(uploads, intent, payload)
    formal_submission(store)
    uploads.registration = lambda _hotkey: (123, 100)  # new UID, same hotkey and spent chance
    renewed = replace(intent, upload_id="cd" * 32, expires_at=NOW + 600)
    with pytest.raises(ValueError, match="formal submission"):
        authorize(uploads, renewed, now=NOW + 301)
    assert len(uploads.staging.grants) == 1


def test_original_40mib_chunks_resume_without_reupload_and_verify_complete_file(setup, tmp_path):
    uploads, original, _, store = setup
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"x" * CHUNK_BYTES + b"tail")
    manifest = build_chunk_manifest(source)
    uploads.max_bytes = manifest.size
    intent = replace(original, size_bytes=manifest.size, file_sha256=manifest.sha256,
                     manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    grant = uploads.authorize(intent, sign(intent), manifest=manifest, now=NOW)
    assert [part.name for part in grant.parts] == ["part-00000", "part-00001"]
    assert [part.size_bytes for part in grant.parts] == [CHUNK_BYTES, 4]
    first, second = [request["Params"] for request in uploads.staging.grants]
    uploads.staging.put_object(Bucket=first["Bucket"], Key=first["Key"],
                               Body=read_part(source, 0, CHUNK_BYTES))
    original_etag = uploads.staging.objects[(first["Bucket"], first["Key"])][1]
    seal = replace(intent, action="seal")
    with pytest.raises(KeyError):
        uploads.seal(seal, sign(seal), now=NOW + 1)
    assert list(uploads.cache_dir.iterdir()) == []
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)
    # Resume the missing original part, not another upload or competition opportunity.
    uploads.staging.put_object(Bucket=second["Bucket"], Key=second["Key"],
                               Body=read_part(source, 1, 4))
    reference = uploads.seal(seal, sign(seal), now=NOW + 2)
    restored = uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "restored",
                                max_bytes=manifest.size)
    assert hashlib.sha256(restored.path.read_bytes()).hexdigest() == manifest.sha256
    assert uploads.staging.objects[(first["Bucket"], first["Key"])][1] == original_etag
    assert store.public_snapshot()["challenges"] == []


def test_valid_part_sha_cannot_substitute_for_signed_complete_file_sha(setup):
    uploads, original, payload, _ = setup
    manifest = ChunkManifest("bb" * 32, len(payload), (
        ChunkPart("part-00000", hashlib.sha256(payload).hexdigest(), len(payload)),
    ))
    intent = replace(original, file_sha256=manifest.sha256,
                     manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
    uploads.authorize(intent, sign(intent), manifest=manifest, now=NOW)
    params = uploads.staging.grants[-1]["Params"]
    uploads.staging.put_object(Bucket=params["Bucket"], Key=params["Key"], Body=payload)
    seal = replace(intent, action="seal")
    with pytest.raises(ValueError, match="complete digest"):
        uploads.seal(seal, sign(seal), now=NOW + 1)
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)
