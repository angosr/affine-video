import hashlib
import io
import json
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Barrier, Lock
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from bittensor.wallet import Keypair
from safetensors.numpy import save

from miner.submit import ModelSubmission, PrivateModelSubmission, UploadIntent
from scoring.model import TensorSpec, validate_model
from validator.admission import AdmissionSettings, AdmissionStatus, process_next_admission
from validator.chunks import CHUNK_BYTES, ChunkManifest, ChunkPart, build_chunk_manifest, read_part
from validator.download import DownloadedModel, ModelDownloadError
from validator.production.batches import ReadyBatchSource
from validator.production.retries import PrivateRetryStore
from validator.production.storage import PrivateObjectStore
from validator.queue import ChainPosition, ChallengeState, submission_id
from validator.scan import ChainCommitment, submission_from_commitment
from validator.store import ChallengeStore, SubmissionRecord
from validator.uploads import PrivateUploads, upload_http_server

from .production.test_storage import MemoryS3

NOW = 1_000
KEY = Keypair.create_from_seed(bytes.fromhex("11" * 32), crypto_type=0)
SCHEMA = {"selector.weight": TensorSpec((2, 2), "float32")}


class StagingS3(MemoryS3):
    def __init__(self):
        super().__init__()
        self.grants = []
        self.reads = []

    def generate_presigned_url(self, operation, **kwargs):
        assert operation == "put_object"
        self.grants.append(kwargs)
        return "https://private.example/staging?secret=bearer"

    def get_object(self, **kwargs):
        self.reads.append(kwargs["Key"])
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
            registration=lambda _hotkey: (9, 100), max_bytes=10_000,
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


def private_formal_submission(store, reference, block=100):
    record = submission_from_commitment(ChainCommitment(
        KEY.ss58_address, 120, "ab" * 32, ChainPosition(block, 1, 1), reference.encode(),
    ))
    assert record is not None and record.private_submission == reference
    identifier, _ = store.add_submission(record)
    store.observe_commitment(
        hotkey=record.hotkey, netuid=record.netuid, block_hash=record.block_hash,
        position=record.position, payload=record.payload,
        submission_id=record.submission_id, legacy_excluded=False,
    )
    assert store.get_challenge(identifier).state is ChallengeState.QUEUED
    return identifier, record


def test_private_inflight_repair_uses_original_admission_archive_and_cold_restore(
    setup, tmp_path, monkeypatch,
):
    uploads, intent, payload, store = setup
    _, params = upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    identifier, record = private_formal_submission(store, reference)
    projected = store.public_challenge(identifier)
    assert projected["repo_id"] is projected["revision"] is None
    assert projected["private_upload"] == {
        "upload_id": reference.upload_id, "file_sha256": reference.file_sha256,
    }
    assert store.active_competitors({intent.hotkey: 9})[0]["private_upload"] == projected[
        "private_upload"
    ]
    del uploads.staging.objects[(params["Bucket"], params["Key"])]
    archive = ReadyBatchSource(
        track=SimpleNamespace(digest="cc" * 32), objects=PrivateObjectStore(
            MemoryS3(), bucket="archive", prefix="original"), cache_dir=tmp_path / "archive",
    )
    retries = PrivateRetryStore(archive.objects, track_spec_digest="cc" * 32)
    config = AdmissionSettings(
        cache_dir=tmp_path / "admission", max_model_bytes=10_000, tensor_schema=SCHEMA,
        retry_base_ms=10, retry_max_ms=100, private_download=uploads.download,
        archive=archive.archive_model, restore=archive.restore_admission,
        restore_retry=retries.restore, record_retry=retries.record_failure,
    )

    def no_hf(*_args, **_kwargs):
        pytest.fail("VDS1 must not be routed to an HF repository or invented revision")

    first = process_next_admission(
        store, worker_id="original", settings=config, current_finalized_block=100,
        now_ms=10, downloader=no_hf,
    )
    assert first.status is AdmissionStatus.DEFERRED
    assert first.code == "PRIVATE_UPLOAD_UNAVAILABLE"
    unchanged = store.public_challenge(identifier)
    renewed = replace(intent, expires_at=NOW + 600)
    with ChallengeStore.readonly(tmp_path / "state.db") as readonly:
        uploads.store = readonly
        try:
            grant = authorize(uploads, renewed, now=NOW + 301)
        finally:
            uploads.store = store
    assert grant.upload_id == reference.upload_id
    assert store.public_challenge(identifier) == unchanged  # Repair never rewrites S/attempts.
    latest_cap = uploads.objects.get(uploads._record_name(reference.upload_id))
    authorize(uploads, replace(intent, expires_at=NOW + 500), now=NOW + 302)
    assert uploads.objects.get(uploads._record_name(reference.upload_id)) == latest_cap
    uploads.staging.put_object(**uploads.staging.grants[-1]["Params"], Body=payload)
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms", lambda _b: 1000)

    def archive_then_crash(task, path, validated):
        assert task.private_submission == reference
        archive.archive_model(task, path, validated)
        raise SystemExit("host lost after original models/admission archival")

    with pytest.raises(SystemExit, match="host lost"):
        process_next_admission(
            store, worker_id="repair", settings=replace(config, archive=archive_then_crash),
            current_finalized_block=100, now_ms=30, downloader=no_hf,
        )
    # A fresh host must use the original archive even if all staging bytes are gone.
    uploads.staging.objects.clear()
    (config.cache_dir / f"{reference.file_sha256}.safetensors").unlink()
    with ChallengeStore(tmp_path / "cold.db") as cold:
        cold.add_submission(record)
        cold.advance_chain_cursor(100, "ab" * 32)
        def no_staging(*_args, **_kwargs):
            pytest.fail("confirmed original models archive must precede staging re-upload")
        result = process_next_admission(
            cold, worker_id="cold", settings=replace(
                config, cache_dir=tmp_path / "cold-cache", private_download=no_staging,
            ), current_finalized_block=100, now_ms=1000, downloader=no_hf,
        )
        assert result.status is AdmissionStatus.ACCEPTED
        restored = cold.public_challenge(identifier)
        assert restored["state"] == ChallengeState.SCORING
        assert restored["block_height"] == record.position.block
        assert restored["file_sha256"] == reference.file_sha256
        assert restored["private_upload"] == projected["private_upload"]
        before = len(uploads.staging.grants)
        uploads.store = cold
        try:
            with pytest.raises(ValueError, match="formal submission"):
                authorize(uploads, renewed, now=NOW + 302)
        finally:
            uploads.store = store
        assert len(uploads.staging.grants) == before
        later_hf = formal_submission(cold, block=101)
        assert cold.get_challenge(later_hf).state is ChallengeState.REJECTED_DUPLICATE_HOTKEY


def test_an_earlier_hf_submission_owns_the_same_opportunity_as_private_s3(setup, tmp_path):
    uploads, intent, payload, store = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    hf = formal_submission(store)
    private, _ = private_formal_submission(store, reference, block=101)
    path = tmp_path / "hf.safetensors"
    path.write_bytes(payload)
    def no_private(*_args, **_kwargs):
        pytest.fail("an earlier original HF opportunity must not be replaced by S3")
    config = AdmissionSettings(cache_dir=tmp_path / "admit", max_model_bytes=10_000,
                               tensor_schema=SCHEMA, private_download=no_private)
    first = process_next_admission(
        store, worker_id="hf", settings=config, current_finalized_block=110, now_ms=10,
        downloader=lambda *_a, **_k: DownloadedModel(
            path, intent.file_sha256, len(payload), None, None,
        ),
    )
    assert first.challenge_id == hf and first.status is AdmissionStatus.ACCEPTED
    second = process_next_admission(store, worker_id="private", settings=config,
                                    current_finalized_block=110, now_ms=20)
    assert second.challenge_id == private and second.status is AdmissionStatus.FAILED
    assert store.get_challenge(private).state is ChallengeState.REJECTED_DUPLICATE_HOTKEY


def test_private_source_without_runtime_wiring_defers_instead_of_falling_back_to_hf(
    setup, tmp_path,
):
    uploads, intent, _, store = setup
    reference = PrivateModelSubmission(intent.upload_id, intent.file_sha256)
    identifier, _ = private_formal_submission(store, reference)
    config = AdmissionSettings(cache_dir=tmp_path / "no-private", max_model_bytes=10_000,
                               tensor_schema=SCHEMA)
    def no_hf(*_args, **_kwargs):
        pytest.fail("an unconfigured private source must never use HF")
    result = process_next_admission(store, worker_id="unconfigured", settings=config,
                                    current_finalized_block=100, now_ms=10, downloader=no_hf)
    assert result.status is AdmissionStatus.DEFERRED
    assert result.code == "PRIVATE_UPLOAD_UNAVAILABLE"
    row = store.public_challenge(identifier)
    assert row["state"] == ChallengeState.SYSTEM_DEFERRED
    assert row["model_digest"] is row["file_sha256"] is None
    assert uploads.staging.grants == []


@pytest.mark.parametrize("fault", [
    "uid0", "deregistered", "excluded", "cursor", "rpc", "other_id", "other_file",
    "terminal", "scoring",
])
def test_private_inflight_repair_cannot_bypass_original_identity_or_eligibility(setup, fault):
    uploads, intent, payload, store = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    identifier, _ = private_formal_submission(store, reference)
    renewed = replace(intent, expires_at=NOW + 600)
    if fault == "uid0":
        uploads.registration = lambda _hotkey: (0, 100)
    elif fault == "deregistered":
        uploads.registration = lambda _hotkey: (None, 100)
    elif fault == "excluded":
        store.seed_legacy_exclusions({intent.hotkey: 100})
    elif fault == "cursor":
        uploads.registration = lambda _hotkey: (9, 101)
    elif fault == "rpc":
        def unavailable(_hotkey):
            raise RuntimeError("registration unavailable")
        uploads.registration = unavailable
    elif fault == "other_id":
        renewed = replace(renewed, upload_id="ef" * 32)
    elif fault == "other_file":
        changed = ChunkManifest("ef" * 32, intent.size_bytes, (
            ChunkPart("part-00000", "ef" * 32, intent.size_bytes),
        ))
        renewed = replace(renewed, file_sha256="ef" * 32,
                          manifest_sha256=hashlib.sha256(changed.encode()).hexdigest())
    elif fault == "terminal":
        store.transition(identifier, ChallengeState.FETCHING)
        store.complete_challenge(identifier, ChallengeState.PARTICIPANT_FAILED,
                                 error_code="INVALID_MODEL", score_block=100)
    elif fault == "scoring":
        store.transition(identifier, ChallengeState.FETCHING)
        store.transition(identifier, ChallengeState.VALIDATING)
        store.transition(identifier, ChallengeState.SCORING)
    before = store.public_challenge(identifier)
    with pytest.raises((ValueError, RuntimeError)):
        authorize(uploads, renewed, now=NOW + 301)
    assert store.public_challenge(identifier) == before
    assert len(uploads.staging.grants) == 1


def test_seal_freezes_metadata_and_admission_verifies_the_real_source_once(setup, tmp_path):
    uploads, intent, payload, store = setup
    grant, params = upload(uploads, intent, payload)
    before = dict(uploads.objects._client.objects)
    retried = authorize(uploads, intent, now=NOW + 1)
    assert before == uploads.objects._client.objects
    assert retried.expires_at == intent.expires_at
    assert 0 < uploads.staging.grants[-1]["ExpiresIn"] <= 299
    assert "bearer" not in repr(grant)
    assert params == {
        "Bucket": "uploads-private", "Key": f"{uploads._key(intent)}/part-00000",
        "ContentLength": len(payload),
    }
    assert store.public_snapshot()["challenges"] == []  # retries consume no opportunity

    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    assert uploads.staging.reads == []  # HTTP seal never proxies a model body
    # Seal success is NOT file acceptance. An unexpired URL can still overwrite staging.
    uploads.staging.put_object(Bucket=params["Bucket"], Key=params["Key"], Body=b"replaced")
    with pytest.raises(ModelDownloadError) as invalid:
        uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cold",
                         max_bytes=10_000)
    assert invalid.value.code == "PRIVATE_UPLOAD_HASH_MISMATCH"
    assert invalid.value.temporary  # unknown backend corruption must not become a miner loss
    uploads.staging.put_object(Bucket=params["Bucket"], Key=params["Key"], Body=payload)
    restored = uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cold",
                                max_bytes=10_000)
    assert restored.path.read_bytes() == payload
    assert restored.download_host is None
    assert validate_model(restored.path, schema=SCHEMA, max_bytes=10_000,
                          expected_hotkey=intent.hotkey).file_sha256 == reference.file_sha256
    assert uploads.seal(seal, sign(seal), now=NOW + 2) == reference
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)
    assert all("/uploads/" in key for _, key in uploads.objects._client.objects)
    sealed_facts = dict(uploads.objects._client.objects)
    authorize(uploads, intent, now=NOW + 2)  # Renew only the same immutable file/parts.
    assert uploads.objects._client.objects == sealed_facts
    assert store.public_snapshot()["challenges"] == []


def test_unsubmitted_sealed_missing_part_can_get_a_new_signed_grant_after_expiry(setup, tmp_path):
    uploads, intent, payload, store = setup
    _, params = upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    _, original = uploads._read(intent.upload_id)
    del uploads.staging.objects[(params["Bucket"], params["Key"])]
    with pytest.raises(ModelDownloadError) as missing:
        uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cold",
                         max_bytes=10_000)
    assert missing.value.code == "PRIVATE_UPLOAD_UNAVAILABLE" and missing.value.temporary

    renewed = replace(intent, expires_at=NOW + 600)
    manifest = ChunkManifest(intent.file_sha256, intent.size_bytes, (
        ChunkPart("part-00000", intent.file_sha256, intent.size_bytes),
    ))
    with upload_http_server(uploads, port=0, clock=lambda: NOW + 301) as server:
        status, _, grant = http_request(server, {
            "intent": asdict(renewed), "signature": sign(renewed).hex(),
            "manifest": json.loads(manifest.encode()),
        })
    assert status == 200 and grant["upload_id"] == intent.upload_id
    assert len(uploads.staging.grants) == 2  # No direct backend repair without a fresh grant.
    permission = uploads.staging.grants[-1]
    assert permission["Params"] == params and 0 < permission["ExpiresIn"] <= 299
    assert grant["parts"][0]["name"] == "part-00000"
    uploads.staging.put_object(**permission["Params"], Body=payload)
    _, repaired = uploads._read(intent.upload_id)
    assert {**original, "expires_at": NOW + 600} == repaired
    assert repaired["state"] == "SEALED"  # Never reopen metadata or a competition opportunity.
    downloaded = uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cold",
                                  max_bytes=10_000)
    assert downloaded.file_sha256 == intent.file_sha256 and downloaded.path.read_bytes() == payload
    assert store.public_snapshot()["challenges"] == []


def test_sealed_resume_cannot_change_its_file_or_bypass_a_formal_hf_submission(setup):
    uploads, intent, payload, store = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    uploads.seal(seal, sign(seal), now=NOW + 1)
    changed = replace(intent, file_sha256="ee" * 32, expires_at=NOW + 600)
    changed_manifest = ChunkManifest(changed.file_sha256, changed.size_bytes, (
        ChunkPart("part-00000", changed.file_sha256, changed.size_bytes),
    ))
    changed = replace(
        changed, manifest_sha256=hashlib.sha256(changed_manifest.encode()).hexdigest(),
    )
    with pytest.raises(ValueError, match="identity is already fixed"):
        authorize(uploads, changed, now=NOW + 301)
    identifier = formal_submission(store)
    renewed = replace(intent, expires_at=NOW + 600)
    with pytest.raises(ValueError, match="formal submission"):
        authorize(uploads, renewed, now=NOW + 301)
    assert store.get_challenge(identifier).state is ChallengeState.QUEUED
    assert store.get_challenge(identifier).attempts == 0
    assert len(uploads.staging.grants) == 1


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


def test_seal_checks_transport_only_metadata_cannot_replace_admission_full_sha(setup, tmp_path):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, b"!" * len(payload))
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    with pytest.raises(ModelDownloadError) as invalid:
        uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cache",
                         max_bytes=10_000)
    assert invalid.value.code == "PRIVATE_UPLOAD_HASH_MISMATCH"
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)


def test_s3_stream_is_bounded_even_with_forged_content_length(setup, monkeypatch, tmp_path):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    body = io.BytesIO(payload + b"over limit")
    monkeypatch.setattr(uploads.staging, "get_object", lambda **_kwargs: {
        "ContentLength": len(payload), "Body": body,
    })
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    with pytest.raises(ModelDownloadError) as invalid:
        uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cache",
                         max_bytes=10_000)
    assert invalid.value.code == "PRIVATE_UPLOAD_HASH_MISMATCH"
    assert body.closed


@pytest.mark.parametrize("lost_on", ["quota", "upload", "seal"])
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
            lost_on == "quota" and "/hotkeys/" in key
            or lost_on == "upload" and document.get("state") == "UPLOADING"
            or lost_on == "seal" and document.get("state") == "SEALED"
        )
        if matches and not lost:
            lost = True
            raise RuntimeError("response lost after server committed")
        return response

    client.put_object = put
    if lost_on != "seal":
        with pytest.raises(RuntimeError, match="response lost"):
            authorize(uploads, intent, now=NOW)
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    if lost_on == "seal":
        with pytest.raises(RuntimeError, match="response lost"):
            uploads.seal(seal, sign(seal), now=NOW + 1)
    reference = uploads.seal(seal, sign(seal), now=NOW + 2)
    receipt = uploads.objects.get(uploads._record_name(reference.upload_id))
    assert uploads.seal(seal, sign(seal), now=NOW + 3) == reference
    assert uploads.objects.get(uploads._record_name(reference.upload_id)) == receipt
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
        with pytest.raises(ModelDownloadError):
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
        if "/hotkeys/" not in kwargs["Key"]:
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
        with ChallengeStore.readonly(database) as connection:
            worker = PrivateUploads(
                objects=uploads.objects, staging_client=uploads.staging,
                staging_bucket=uploads.bucket, staging_prefix=uploads.prefix,
                audience=uploads.audience, genesis_hash=uploads.genesis_hash,
                store=connection, registration=uploads.registration,
                max_bytes=uploads.max_bytes,
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


def test_fresh_adapter_recovers_the_same_sealed_record_and_verifies_staging(setup, tmp_path):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    fresh = PrivateUploads(
        objects=uploads.objects, staging_client=uploads.staging,
        staging_bucket=uploads.bucket, staging_prefix=uploads.prefix,
        audience=uploads.audience, genesis_hash=uploads.genesis_hash,
        store=uploads.store, registration=uploads.registration,
        max_bytes=uploads.max_bytes,
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
    with pytest.raises(RuntimeError, match="missing"):
        uploads.seal(seal, sign(seal), now=NOW + 1)
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


def test_valid_part_sha_cannot_substitute_for_signed_complete_file_sha(setup, tmp_path):
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
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)
    with pytest.raises(ModelDownloadError) as invalid:
        uploads.download(reference, hotkey=intent.hotkey, cache_dir=tmp_path / "cache",
                         max_bytes=10_000)
    assert invalid.value.code == "PRIVATE_UPLOAD_HASH_MISMATCH"
    assert not any("/models/" in key for _, key in uploads.objects._client.objects)


def test_readonly_eligibility_uses_existing_schema_and_cannot_write(setup, monkeypatch, tmp_path):
    _, intent, _, writer = setup
    path = writer._connection.execute("PRAGMA database_list").fetchone()[2]

    def no_initialize(_self):
        pytest.fail("read-only upload admission attempted DDL/WAL initialization")

    monkeypatch.setattr(ChallengeStore, "_initialize", no_initialize)
    with ChallengeStore.readonly(path) as reader:
        reader.require_upload_eligible(intent.hotkey, uid=9, finalized_block=100)
        assert reader._connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._connection.execute("DELETE FROM submissions")
    missing = tmp_path / "must-not-create.db"
    with pytest.raises(sqlite3.OperationalError):
        ChallengeStore.readonly(missing)
    assert not missing.exists()
    assert writer.public_snapshot()["challenges"] == []


def test_eligibility_cursor_exclusions_and_opportunity_share_one_read_snapshot(setup):
    _, intent, _, writer = setup
    path = writer._connection.execute("PRAGMA database_list").fetchone()[2]
    with ChallengeStore.readonly(path) as reader:
        queries = []
        inserted = False

        def trace(query):
            nonlocal inserted
            queries.append(query)
            if "FROM competition_exclusions" in query and not inserted:
                inserted = True
                formal_submission(writer, block=101)

        reader._connection.set_trace_callback(trace)
        # This snapshot precedes the new commitment; mixing snapshots would
        # observe a half-old/half-new qualification. ADMISSION still arbitrates.
        reader.require_upload_eligible(intent.hotkey, uid=9, finalized_block=100)
        assert queries[0] == "BEGIN" and queries[-1] == "ROLLBACK"
        reader._connection.set_trace_callback(None)
        with pytest.raises(ValueError, match="formal submission"):
            reader.require_upload_eligible(intent.hotkey, uid=9, finalized_block=100)


def test_private_source_cache_io_failure_is_temporary_not_a_participant_loss(
    setup, monkeypatch, tmp_path,
):
    uploads, intent, payload, _ = setup
    upload(uploads, intent, payload)
    seal = replace(intent, action="seal")
    reference = uploads.seal(seal, sign(seal), now=NOW + 1)

    def denied(_path):
        raise PermissionError("cache access denied")

    monkeypatch.setattr("validator.uploads.model_file_sha256", denied)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{reference.file_sha256}.safetensors").write_bytes(payload)
    with pytest.raises(ModelDownloadError) as unavailable:
        uploads.download(reference, hotkey=intent.hotkey, cache_dir=cache, max_bytes=10_000)
    assert unavailable.value.temporary
    assert unavailable.value.code == "PRIVATE_UPLOAD_UNAVAILABLE"


def test_authority_delay_cannot_extend_a_signed_upload_capability(setup, monkeypatch):
    uploads, intent, _, _ = setup
    instants = iter((0.0, 301.0))
    monkeypatch.setattr("validator.uploads.time.monotonic", lambda: next(instants))
    with pytest.raises(ValueError, match="expired before"):
        authorize(uploads, intent, now=NOW)
    assert uploads.staging.grants == []


def http_request(server, payload, *, content_length=None):
    encoded = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    request = Request(
        f"http://127.0.0.1:{server.server_port}/veda", data=encoded,
        headers={"Content-Type": "application/json", "Content-Length": str(
            len(encoded) if content_length is None else content_length,
        )}, method="POST",
    )

    def client():
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, dict(response.headers), json.loads(response.read())

    # SQLite stays in its owning main thread; only the client runs elsewhere.
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(client)
        server.handle_request()
        return result.result(timeout=5)


def test_single_loopback_http_adapter_handles_only_small_signed_metadata(setup, capsys):
    uploads, intent, payload, store = setup
    manifest = ChunkManifest(intent.file_sha256, intent.size_bytes, (
        ChunkPart("part-00000", intent.file_sha256, intent.size_bytes),
    ))
    with upload_http_server(uploads, port=0, clock=lambda: NOW) as server:
        assert server.server_address[0] == "127.0.0.1"
        status, headers, grant = http_request(server, {
            "intent": asdict(intent), "signature": sign(intent).hex(),
            "manifest": json.loads(manifest.encode()),
        })
        assert status == 200 and headers["Cache-Control"] == "no-store"
        assert len(grant["parts"]) == 1 and grant["upload_id"] == intent.upload_id
        params = uploads.staging.grants[-1]["Params"]
        uploads.staging.put_object(Bucket=params["Bucket"], Key=params["Key"], Body=payload)
        seal = replace(intent, action="seal")
        status, _, sealed = http_request(server, {
            "intent": asdict(seal), "signature": sign(seal).hex(),
        })
        assert status == 200 and sealed["sealed"] is True and sealed["accepted"] is False
        assert uploads.staging.reads == []  # no model-body proxy or redundant model admission
        assert store.public_snapshot()["challenges"] == []
    assert "bearer" not in capsys.readouterr().err


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_http_rejects_body_over_limit_and_does_not_expose_private_errors(
    setup, monkeypatch, failure,
):
    uploads, intent, _, _ = setup
    with upload_http_server(uploads, port=0, clock=lambda: NOW) as server:
        status, _, response = http_request(server, b"{}", content_length=64 * 1024 + 1)
        assert status == 400 and response == {"error": "INVALID_UPLOAD_REQUEST"}
        assert uploads.staging.grants == []

        def unavailable(*_args, **_kwargs):
            raise failure("PRIVATE_ENDPOINT_AND_CREDENTIALS")

        monkeypatch.setattr(uploads, "authorize", unavailable)
        manifest = ChunkManifest(intent.file_sha256, intent.size_bytes, (
            ChunkPart("part-00000", intent.file_sha256, intent.size_bytes),
        ))
        status, _, response = http_request(server, {
            "intent": asdict(intent), "signature": sign(intent).hex(),
            "manifest": json.loads(manifest.encode()),
        })
        assert status == 503 and response == {"error": "UPLOAD_UNAVAILABLE"}


@pytest.mark.parametrize("partial_request", [
    b"POST /ve",
    b"POST /veda HTTP/1.1\r\nHost: auth.example\r\nX-Incomplete: ",
], ids=["request-line", "headers"])
def test_http_times_out_before_parsing_and_can_serve_the_next_request(
    setup, monkeypatch, partial_request,
):
    uploads, _, _, _ = setup
    with upload_http_server(uploads, port=0, clock=lambda: NOW) as server:
        handler = server.RequestHandlerClass
        assert handler.timeout == 10
        monkeypatch.setattr(handler, "timeout", 0.05)
        original_setup = handler.setup
        observed_timeouts = []

        def setup_connection(self):
            original_setup(self)
            timeout = self.connection.gettimeout()
            observed_timeouts.append(timeout)

        monkeypatch.setattr(handler, "setup", setup_connection)

        def stalled_client():
            with socket.create_connection(server.server_address, timeout=1) as connection:
                connection.settimeout(0.5)
                connection.sendall(partial_request)
                try:
                    return connection.recv(1) == b""
                except TimeoutError:
                    return False  # Closing releases the unfixed handler, not the assertion.

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(stalled_client)
            server.handle_request()
            assert result.result(timeout=2), "partial metadata request held the only handler"
        assert observed_timeouts == [0.05]
        assert uploads.staging.grants == []
        status, _, response = http_request(server, {})
        assert status == 400 and response == {"error": "INVALID_UPLOAD_REQUEST"}
        assert observed_timeouts == [0.05, 0.05]
