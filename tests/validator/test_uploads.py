"""Native R2 mailbox replacement; fake chain/storage, real Ed25519 crypto."""
import asyncio
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from bittensor.wallet import Keypair
from safetensors.numpy import save

from miner.submit import (
    ModelSubmission,
    PrivateModelSubmission,
    UploadActivation,
    activation_message,
    upload_registration_id,
)
from miner.upload import decrypt_mailbox, upload_file
from scoring.model import TensorSpec
from validator.admission import AdmissionSettings, AdmissionStatus, process_next_admission
from validator.download import ModelDownloadError
from validator.production.batches import ReadyBatchSource
from validator.production.storage import PrivateObjectStore
from validator.publication import PublicArtifactError, S3Repository
from validator.queue import ChainPosition, ChallengeState
from validator.scan import ChainCommitment, submission_from_commitment
from validator.store import ChallengeStore
from validator.uploads import (
    CREDENTIAL_TTL,
    MODEL_RETENTION_MS,
    CloudflareR2Tokens,
    MailboxControl,
    OwnerMailboxControl,
    PrivateUploads,
    attach_evaluator,
    attach_from_environment,
)

from .production.test_storage import MemoryS3

KEY = Keypair.create_from_seed(bytes.fromhex("11" * 32), crypto_type=0)
SIGNER = Keypair.create_from_seed(bytes.fromhex("22" * 32), crypto_type=0)
GENESIS = "01" * 32
ENDPOINT = "https://" + "ab" * 16 + ".r2.cloudflarestorage.com"
SCHEMA = {"selector.weight": TensorSpec((2, 2), "float32")}
NOW = 1000


class StagingS3(MemoryS3):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.meta = SimpleNamespace(endpoint_url=ENDPOINT)

    def get_object(self, **kwargs):
        self.events.append("get")
        return super().get_object(**kwargs)

    def close(self):
        pass

    def copy_object(self, *, Bucket, Key, CopySource):
        self.events.append("champion-copy")
        return self.put_object(Bucket=Bucket, Key=Key,
                               Body=self.objects[(CopySource["Bucket"], CopySource["Key"])][0])


class Tokens:
    account = "ab" * 16
    jurisdiction = "default"

    def __init__(self, events):
        self.events = events
        self.parents = {}
        self.revoked = set()

    def create_parent(self, name, bucket, ttl):
        self.events.append("parent")
        assert bucket == "affinemodel" and ttl == CREDENTIAL_TTL
        return self.parents.setdefault(name, name)

    def mint(self, parent, bucket, prefix, ttl):
        self.events.append("mint")
        assert parent not in self.revoked and prefix.startswith("uploads/v1/")
        return {"access_key_id": "fixture-access", "secret_access_key": "fixture-secret",
                "session_token": "fixture-session"}

    def revoke(self, parent, *, name, bucket):
        assert parent == name and bucket == "affinemodel"
        self.events.append("revoke")
        self.revoked.add(parent)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.uploads.time.time", lambda: NOW)
    events, mailbox = [], {}
    staging = StagingS3(events)
    objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="evaluation")
    with ChallengeStore(tmp_path / "state.db") as store:
        store.advance_chain_cursor(200, "ab" * 32)
        uploads = PrivateUploads(
            objects=objects,
            staging_client=staging, staging_bucket="affinemodel", staging_prefix="uploads/v1",
            genesis_hash=GENESIS, store=store, registration=lambda _h: (9, 200),
            max_bytes=10000, control=MailboxControl(
                tokens=Tokens(events), signer=SIGNER, endpoint=ENDPOINT,
                bucket="affinemodel", genesis_hash=GENESIS, objects=objects,
                publish_mailbox=lambda name, body: mailbox.__setitem__(name, body),
            ),
        )
        uploads.submission_timestamp = lambda _block: NOW * 1000
        model = tmp_path / "prepared.safetensors"
        model.write_bytes(save({"selector.weight": np.ones((2, 2), dtype=np.float32)},
                               metadata={"hotkey": KEY.ss58_address}))
        yield SimpleNamespace(uploads=uploads, store=store, model=model, events=events,
                              mailbox=mailbox, root=tmp_path)


def activate(ctx, *, block=150, key=KEY):
    message = activation_message(GENESIS, key.ss58_address)
    ctx.uploads.observe_activation(UploadActivation(key.sign(message)), hotkey=key.ss58_address,
                                   block=block, extrinsic_index=2, event_index=3,
                                   block_hash=f"{block:064x}")


def transfer(ctx):
    activate(ctx)
    assert ctx.uploads.issue_pending(now=NOW)
    assert ctx.store.public_snapshot()["challenges"] == []

    def transport(request):
        assert request.method == "GET"  # NO HTTP auth POST / presigned PUT.
        key = "mailbox/v1/" + request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=ctx.mailbox[key])

    async def upload():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            return await upload_file(
                ctx.model, mailbox_url="https://public.example/tracks/main/mailbox/v1",
                validator_identity=SIGNER.ss58_address, genesis_hash=GENESIS, hotkey=KEY,
                client=client, s3_factory=lambda _envelope: ctx.uploads.staging,
            )
    return asyncio.run(upload())


def submit(ctx, reference, block=200):
    record = submission_from_commitment(ChainCommitment(
        KEY.ss58_address, 120, "ab" * 32, ChainPosition(block, 1, 1), reference.encode(),
    ))
    if isinstance(reference, PrivateModelSubmission):
        assert record.private_submission == reference
    identifier, _ = ctx.store.add_submission(record)
    return identifier, record


def test_automatic_mailbox_r2_ready_original_admission_archive_and_cold_restore(setup, monkeypatch):
    ctx = setup
    reference = transfer(ctx)
    assert len(reference.encode()) == 100 and reference.encode().startswith(b"VDR1")
    assert ctx.events == ["parent", "mint"]
    identifier, record = submit(ctx, reference)
    assert record.position.block == 200  # Activation was at 150; ready determines S.
    archive = ReadyBatchSource(
        track=SimpleNamespace(digest="cc" * 32),
        objects=PrivateObjectStore(MemoryS3(), bucket="archive", prefix="fixture"),
        cache_dir=ctx.root / "archive",
    )
    archive.private_uploads = ctx.uploads
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms",
                        lambda _b: NOW * 1000)
    settings = AdmissionSettings(
        cache_dir=ctx.root / "cache", max_model_bytes=10000, tensor_schema=SCHEMA,
        private_download=ctx.uploads.download, archive=archive.archive_model,
        restore=archive.restore_admission,
    )
    result = process_next_admission(ctx.store, worker_id="original", settings=settings,
                                    current_finalized_block=200, now_ms=10)
    assert result.status is AdmissionStatus.ACCEPTED
    assert ctx.events.index("revoke") < ctx.events.index("get")
    before = ctx.store.public_challenge(identifier)
    assert before["state"] == ChallengeState.SCORING
    assert before["block_height"] == 200
    assert before["private_upload"]["manifest_sha256"] == reference.manifest_sha256
    # Only original small references reach MinIO; R2 remains the formal model body.
    assert not any(".parts/" in name for _bucket, name in archive.objects._client.objects)
    ctx.model.unlink()
    for path in (ctx.root / "cache").glob("*.safetensors"):
        path.unlink()
    def forbidden(*_args, **_kwargs):
        pytest.fail("cold restoration must not redownload/reupload/re-evaluate")
    with ChallengeStore(ctx.root / "cold.db") as cold:
        cold.add_submission(record)
        cold.advance_chain_cursor(200, "ab" * 32)
        result = process_next_admission(
            cold, worker_id="cold", current_finalized_block=200, now_ms=1000,
            settings=replace(settings, cache_dir=ctx.root / "fresh-cache",
                             private_download=forbidden),
        )
        assert result.status is AdmissionStatus.ACCEPTED
        assert cold.public_challenge(identifier)["private_upload"] == before["private_upload"]
        assert cold.public_challenge(identifier)["file_sha256"] == reference.file_sha256


def admit_retained(ctx, monkeypatch):
    reference = transfer(ctx)
    identifier, record = submit(ctx, reference)
    archive = ReadyBatchSource(
        track=SimpleNamespace(digest="cc" * 32),
        objects=PrivateObjectStore(MemoryS3(), bucket="archive", prefix="fixture"),
        cache_dir=ctx.root / "archive",
    )
    archive.private_uploads = ctx.uploads
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms",
                        lambda _b: NOW * 1000)
    result = process_next_admission(
        ctx.store, worker_id="original", current_finalized_block=200, now_ms=10,
        settings=AdmissionSettings(cache_dir=ctx.root / "cache", max_model_bytes=10000,
                                   tensor_schema=SCHEMA, private_download=ctx.uploads.download,
                                   archive=archive.archive_model,
                                   restore=archive.restore_admission),
    )
    assert result.status is AdmissionStatus.ACCEPTED
    ctx.uploads.current_champion = lambda _store: None
    return reference, identifier, record, archive


def test_retained_r2_never_copies_model_payload_to_minio_and_cold_restores(setup, monkeypatch):
    ctx = setup
    # Any MinIO payload I/O, including a readback, is an application failure.
    monkeypatch.setattr(PrivateObjectStore, "put_content",
                        lambda *_a, **_k: pytest.fail("private MinIO payload copy"))
    monkeypatch.setattr(PrivateObjectStore, "download_content",
                        lambda *_a, **_k: pytest.fail("private MinIO payload readback"))
    reference, identifier, original, archive = admit_retained(ctx, monkeypatch)
    before = ctx.store.public_challenge(identifier)
    _, record = ctx.uploads._read(reference.upload_id)
    model = record["model_reference"]
    assert model["retain_until_ms"] == NOW * 1000 + MODEL_RETENTION_MS
    assert model["submission_block"] == original.position.block == 200
    assert model["file_sha256"] == reference.file_sha256
    assert len(archive.objects._client.objects) == 2  # model reference + admission, no parts
    cached = ctx.root / "cache" / f"{reference.file_sha256}.safetensors"
    cached.unlink()
    restored = archive.restore_model(reference.file_sha256, ctx.model.stat().st_size)
    assert restored.read_bytes() == ctx.model.read_bytes()
    assert ctx.store.public_challenge(identifier) == before
    assert ctx.uploads._read(reference.upload_id)[1]["model_reference"] == model


def test_r2_retention_is_from_original_s_and_pins_admitted_work_after_thirty_days(
    setup, monkeypatch,
):
    ctx = setup
    reference, identifier, _original, archive = admit_retained(ctx, monkeypatch)
    deadline = NOW + MODEL_RETENTION_MS // 1000
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=deadline - 1, now=deadline - 1) == 0
    # model_digest is populated, but an admitted in-flight task still pins its body.
    assert ctx.store.public_challenge(identifier)["model_digest"] is not None
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=deadline + 1, now=deadline + 1) == 0
    ctx.store.transition(identifier, ChallengeState.PARTICIPANT_FAILED,
                         error_code="MODEL_RUNTIME_INVALID", occurred_at_ms=20)
    model = json.loads(archive.objects.get(f"models/{reference.file_sha256}.safetensors").body)
    assert model["retain_until_ms"] == deadline * 1000
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=deadline, now=deadline) == 2
    assert ctx.uploads._read(reference.upload_id)[1]["cleanup"] == "done"


def test_r2_current_champion_is_pinned_and_permanent_archive_cold_restores(
    setup, monkeypatch,
):
    ctx = setup
    reference, identifier, _original, archive = admit_retained(ctx, monkeypatch)
    ctx.store.transition(identifier, ChallengeState.PARTICIPANT_FAILED,
                         error_code="MODEL_RUNTIME_INVALID", occurred_at_ms=20)
    # GC consumes the already-resolved champion reference, not this synthetic stage state.
    ctx.uploads.current_champion = lambda _store: reference
    after = NOW + MODEL_RETENTION_MS // 1000 + 1
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=after, now=after) == 0
    archive.archive_champion(reference.file_sha256, ctx.model.stat().st_size)
    assert "champion-copy" in ctx.events
    count = ctx.events.count("champion-copy")
    archive.archive_champion(reference.file_sha256, ctx.model.stat().st_size)
    assert ctx.events.count("champion-copy") == count
    ctx.uploads.current_champion = lambda _store: None
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=after, now=after) == 2
    restored = archive.restore_model(reference.file_sha256, ctx.model.stat().st_size)
    assert restored.read_bytes() == ctx.model.read_bytes()
    assert all(not key.startswith("uploads/v1/")
               for _bucket, key in ctx.uploads.staging.objects)
    assert any(key.startswith("champions/v1/")
               for _bucket, key in ctx.uploads.staging.objects)


def test_r2_formal_retention_survives_cold_db_and_refuses_new_write_permission(
    setup, monkeypatch,
):
    from copy import copy

    ctx = setup
    reference, _identifier, _record, _archive = admit_retained(ctx, monkeypatch)
    ctx.store.advance_chain_cursor(201, "cd" * 32)
    ctx.uploads.registration = lambda _h: (9, 201)
    activate(ctx, block=201)
    count = ctx.events.count("mint")
    with pytest.raises(ValueError, match="write permissions"):
        ctx.uploads.issue_pending(now=NOW + 1)
    assert ctx.events.count("mint") == count
    with ChallengeStore(ctx.root / "empty.db") as store:
        store.advance_chain_cursor(201, "cd" * 32)
        cold = copy(ctx.uploads)
        cold.store = store
        after = NOW + MODEL_RETENTION_MS // 1000 + 1
        assert cold.reclaim_expired(store=store, finalized_block=201,
                                    finalized_timestamp=after, now=after) == 0
    assert ctx.uploads._read(reference.upload_id)[1]["model_reference"] is not None


def test_r2_reference_and_original_retention_clock_cannot_be_rebound(setup, monkeypatch):
    from validator.production.commitment import canonical_json

    ctx = setup
    reference, _identifier, _record, archive = admit_retained(ctx, monkeypatch)
    with pytest.raises(ValueError, match="original S"):
        ctx.uploads.bind_submission(reference, hotkey=KEY.ss58_address, submission_block=201,
                                    submission_timestamp_ms=(NOW + 1) * 1000)
    name = f"models/{reference.file_sha256}.safetensors"
    stored = archive.objects.get(name)
    altered = json.loads(stored.body)
    altered["model_digest"] = "ef" * 32
    archive.objects.replace_plan(name, canonical_json(altered), etag=stored.etag)
    with pytest.raises(RuntimeError, match="original retention"):
        archive.restore_model(reference.file_sha256, ctx.model.stat().st_size)


def test_r2_formal_schema_rejection_still_retains_bytes_for_thirty_days(setup, monkeypatch):
    from scoring.model import ModelValidationError

    ctx = setup
    reference = transfer(ctx)
    identifier, _record = submit(ctx, reference)

    def invalid(*_args, **_kwargs):
        raise ModelValidationError("INVALID_MODEL", "synthetic schema rejection")

    result = process_next_admission(
        ctx.store, worker_id="original", current_finalized_block=200, now_ms=10,
        settings=AdmissionSettings(cache_dir=ctx.root / "cache", max_model_bytes=10000,
                                   tensor_schema=SCHEMA, private_download=ctx.uploads.download),
        model_validator=invalid,
    )
    assert result.status is AdmissionStatus.FAILED
    row = ctx.store.public_challenge(identifier)
    assert row["state"] == ChallengeState.PARTICIPANT_FAILED and row["model_digest"] is None
    before = NOW + MODEL_RETENTION_MS // 1000 - 1
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=before, now=before) == 0
    assert ctx.uploads._read(reference.upload_id)[1]["formal_submission"][
        "submission_block"
    ] == 200


def test_failed_r2_champion_copy_never_marks_a_durable_archive(setup, monkeypatch):
    ctx = setup
    reference, _identifier, _record, archive = admit_retained(ctx, monkeypatch)

    def interrupted(**_kwargs):
        raise OSError("copy interrupted")

    copy_object = ctx.uploads.staging.copy_object
    monkeypatch.setattr(ctx.uploads.staging, "copy_object", interrupted)
    with pytest.raises(OSError, match="copy interrupted"):
        archive.archive_champion(reference.file_sha256, ctx.model.stat().st_size)
    assert ctx.uploads._read(reference.upload_id)[1].get("champion_archive") is None
    monkeypatch.setattr(ctx.uploads.staging, "copy_object", copy_object)
    archive.archive_champion(reference.file_sha256, ctx.model.stat().st_size)
    assert ctx.uploads._read(reference.upload_id)[1]["champion_archive"].startswith("champions/")


def test_existing_private_minio_archive_is_preserved_without_recopy(setup, monkeypatch):
    from scoring.model import validate_model

    ctx = setup
    reference = transfer(ctx)
    identifier, _record = submit(ctx, reference)
    archive = ReadyBatchSource(track=SimpleNamespace(digest="cc" * 32),
                               objects=PrivateObjectStore(MemoryS3(), bucket="a", prefix="old"),
                               cache_dir=ctx.root / "archive")
    archive.objects.put_content(ctx.model, sha256=reference.file_sha256,
                                 object_name=f"models/{reference.file_sha256}.safetensors")
    task = SimpleNamespace(challenge_id=identifier, submission_id="ab" * 32,
                           repo_id=None, revision=None, private_submission=reference,
                           position=ChainPosition(200, 1, 1), hotkey=KEY.ss58_address)
    before = dict(archive.objects._client.objects)
    monkeypatch.setattr(archive.objects, "put_content",
                        lambda *_a, **_k: pytest.fail("old private body recopied"))
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms", lambda _b: NOW)
    archive.archive_model(task, ctx.model, validate_model(ctx.model, schema=SCHEMA,
                          max_bytes=10000, expected_hotkey=KEY.ss58_address))
    assert all(archive.objects._client.objects[key] == value for key, value in before.items())
    assert ctx.uploads._read(reference.upload_id)[1].get("model_reference") is None
    staging_before = dict(ctx.uploads.staging.objects)
    monkeypatch.setattr(archive, "restore_model",
                        lambda *_a: pytest.fail("legacy champion must not be recopied"))
    archive.archive_champion(reference.file_sha256, ctx.model.stat().st_size)
    assert ctx.uploads.staging.objects == staging_before
    assert ctx.uploads._read(reference.upload_id)[1].get("model_reference") is None


@pytest.mark.parametrize("crypto_type", [1])
def test_non_ed25519_activation_never_gets_credentials(setup, crypto_type):
    ctx = setup
    key = Keypair.create_from_seed(bytes.fromhex("33" * 32), crypto_type=crypto_type)
    with pytest.raises(ValueError, match="Ed25519"):
        activate(ctx, key=key)
    assert not ctx.events and not ctx.mailbox


@pytest.mark.parametrize("kind", ["stale", "uid0", "unregistered", "legacy", "hf"])
def test_original_qualification_is_the_only_upload_gate(setup, kind):
    ctx = setup
    activate(ctx)
    if kind == "stale":
        ctx.uploads.registration = lambda _h: (9, 201)
    elif kind == "uid0":
        ctx.uploads.registration = lambda _h: (0, 200)
    elif kind == "unregistered":
        ctx.uploads.registration = lambda _h: (None, 200)
    elif kind == "legacy":
        ctx.store.seed_legacy_exclusions({KEY.ss58_address: 100})
    else:
        model = ModelSubmission("owner/model", "12" * 20)
        submit(ctx, model)
    with pytest.raises((RuntimeError, ValueError)):
        ctx.uploads.issue_pending(now=NOW)
    assert not ctx.events and not ctx.mailbox


def test_activation_and_mailbox_replay_do_not_duplicate_parent_or_opportunity(setup):
    ctx = setup
    activate(ctx)
    first = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))[0]
    activate(ctx)
    assert ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))[0] == first
    assert ctx.uploads.issue_pending(now=NOW)
    blob = next(iter(ctx.mailbox.values()))
    assert b"fixture-secret" not in blob and b"fixture-session" not in blob
    assert not ctx.uploads.issue_pending(now=NOW)
    ctx.mailbox.clear()
    ctx.uploads.republish_mailbox(upload_registration_id(GENESIS, KEY.ss58_address))
    assert next(iter(ctx.mailbox.values())) == blob
    assert ctx.events.count("parent") == 1 and ctx.store.public_snapshot()["challenges"] == []


@pytest.mark.parametrize("cold_owner", [False, True])
def test_seventeenth_activation_is_reachable_after_full_page_and_owner_restart(
    setup, cold_owner, monkeypatch,
):
    from contextlib import nullcontext

    ctx = setup
    list_objects = ctx.uploads.objects._client.list_objects_v2

    def scoped_list(*, Prefix, **kwargs):
        assert Prefix == "evaluation/uploads/"
        return list_objects(Prefix=Prefix, **kwargs)

    monkeypatch.setattr(ctx.uploads.objects._client, "list_objects_v2", scoped_list)
    keys = [Keypair.create_from_seed(index.to_bytes(32, "big"), crypto_type=0)
            for index in range(3, 20)]
    identities = sorted((upload_registration_id(GENESIS, key.ss58_address), key) for key in keys)
    uid_by_hotkey = {key.ss58_address: uid for uid, key in enumerate(keys, start=1)}
    ctx.uploads.registration = lambda hotkey: (uid_by_hotkey[hotkey], 200)
    for key in keys:
        activate(ctx, key=key)
    for _ in range(16):
        assert ctx.uploads.issue_pending(now=NOW)
    for upload_id, _ in identities[:16]:
        assert ctx.uploads._read(upload_id)[1]["mailbox"]["published"]
    last_id = identities[-1][0]
    assert ctx.uploads._read(last_id)[1]["state"] == "ISSUING"
    connection = (ChallengeStore.readonly(ctx.root / "state.db")
                  if cold_owner else nullcontext(ctx.store))
    with connection as store:
        owner = ctx.uploads
        if cold_owner:
            # New owner and SQLite connection; only original CAS/provider facts survive.
            owner = PrivateUploads(
                objects=ctx.uploads.objects, staging_client=ctx.uploads.staging,
                staging_bucket="affinemodel", staging_prefix="uploads/v1",
                genesis_hash=GENESIS, store=store, max_bytes=10000,
                registration=ctx.uploads.registration, control=ctx.uploads.control,
            )
            assert owner._issue_after == ""
            assert not owner.issue_pending(now=NOW)  # First full page is already complete.
        assert owner.issue_pending(now=NOW)
        assert owner._read(last_id)[1]["mailbox"]["published"]
        assert len(ctx.mailbox) == 17 and ctx.events.count("parent") == 17
        # A later request before the cursor remains reachable after empty-page wrap.
        first_id, first_key = identities[0]
        ctx.store.advance_chain_cursor(201, "cd" * 32)
        owner.registration = lambda hotkey: (uid_by_hotkey[hotkey], 201)
        owner.observe_activation(
            UploadActivation(first_key.sign(activation_message(GENESIS, first_key.ss58_address))),
            hotkey=first_key.ss58_address, block=201, extrinsic_index=2, event_index=3,
            block_hash="cd" * 32,
        )
        assert owner.issue_pending(now=NOW)
        assert owner._read(first_id)[1]["generation"] == 2
        assert ctx.events.count("parent") == 18
        assert store.public_snapshot()["challenges"] == []


def test_mailbox_lost_publish_replays_exact_ciphertext_after_cold_restart(setup, monkeypatch):
    ctx = setup
    activate(ctx)
    original = ctx.uploads.control.publish_mailbox
    def lost(name, body):
        original(name, body)
        raise OSError("publish response lost")
    monkeypatch.setattr(ctx.uploads.control, "publish_mailbox", lost)
    with pytest.raises(OSError):
        ctx.uploads.issue_pending(now=NOW)
    blob = next(iter(ctx.mailbox.values()))
    monkeypatch.setattr(ctx.uploads.control, "publish_mailbox", original)
    assert not ctx.uploads.issue_pending(now=NOW)
    assert ctx.uploads.maintenance_status()["private_upload_error"] == (
        "PRIVATE_UPLOAD_ISSUANCE_FAILED"
    )
    assert ctx.uploads.issue_pending(now=NOW + 30)
    assert next(iter(ctx.mailbox.values())) == blob
    assert ctx.uploads.maintenance_status()["private_upload_error"] is None


@pytest.mark.parametrize("alter", ["other-validator", "other-owner", "other-chain", "tamper"])
def test_mailbox_requires_pinned_validator_owner_network_and_authentic_ciphertext(setup, alter):
    ctx = setup
    activate(ctx)
    ctx.uploads.issue_pending(now=NOW)
    blob = next(iter(ctx.mailbox.values()))
    kwargs = dict(hotkey=KEY, validator_identity=SIGNER.ss58_address, genesis_hash=GENESIS)
    if alter == "other-validator":
        kwargs["validator_identity"] = KEY.ss58_address
    elif alter == "other-owner":
        kwargs["hotkey"] = SIGNER
    elif alter == "other-chain":
        kwargs["genesis_hash"] = "02" * 32
    else:
        blob = bytes([blob[0] ^ 1]) + blob[1:]
    with pytest.raises((ValueError, TypeError)):
        decrypt_mailbox(blob, **kwargs)


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_ready_revoke_failure_defers_without_reading_or_burning_opportunity(
    setup, monkeypatch, failure,
):
    ctx = setup
    reference = transfer(ctx)
    identifier, _ = submit(ctx, reference)
    def unavailable(_parent, **_scope):
        raise failure("control unavailable")
    monkeypatch.setattr(ctx.uploads.control.tokens, "revoke", unavailable)
    result = process_next_admission(
        ctx.store, worker_id="original", current_finalized_block=200, now_ms=10,
        settings=AdmissionSettings(cache_dir=ctx.root / "cache", max_model_bytes=10000,
                                   tensor_schema=SCHEMA, private_download=ctx.uploads.download),
    )
    assert result.status is AdmissionStatus.DEFERRED and "get" not in ctx.events
    assert ctx.store.public_challenge(identifier)["state"] == ChallengeState.SYSTEM_DEFERRED


def test_ready_payload_and_manifest_cannot_change_owner_or_bytes(setup):
    ctx = setup
    reference = transfer(ctx)
    assert ctx.uploads.download(reference, hotkey=KEY.ss58_address,
                                cache_dir=ctx.root / "good", max_bytes=10000).file_sha256 == (
        reference.file_sha256
    )
    wrong = PrivateModelSubmission(reference.upload_id, "33" * 32, reference.manifest_sha256)
    with pytest.raises(ModelDownloadError, match="binding"):
        ctx.uploads.download(wrong, hotkey=KEY.ss58_address, cache_dir=ctx.root / "bad",
                             max_bytes=10000)


def test_gc_pins_original_ready_until_archive_then_exactly_cleans_only_its_prefix(setup):
    ctx = setup
    reference = transfer(ctx)
    identifier, _ = submit(ctx, reference)
    now = NOW + CREDENTIAL_TTL + 1
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=now, now=now) == 0
    result = process_next_admission(
        ctx.store, worker_id="original", current_finalized_block=200, now_ms=10,
        settings=AdmissionSettings(cache_dir=ctx.root / "cache", max_model_bytes=10000,
                                   tensor_schema=SCHEMA, private_download=ctx.uploads.download,
                                   archive=lambda *_args: None),
    )
    assert result.status is AdmissionStatus.ACCEPTED
    original = ctx.store.public_challenge(identifier)
    ctx.uploads.staging.put_object(
        Bucket="affinemodel", Key="uploads/v1/another/part", Body=b"keep",
    )
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=NOW + MODEL_RETENTION_MS // 1000,
                                      now=NOW + MODEL_RETENTION_MS // 1000) == 2
    assert ctx.store.public_challenge(identifier) == original
    assert ctx.uploads.staging.objects[("affinemodel", "uploads/v1/another/part")][0] == b"keep"


@pytest.mark.parametrize("count,interrupt_after", [
    (1000, None), (1001, None), (2001, None), (1001, 17),
])
def test_gc_deletes_one_page_and_cold_resumes_original_pending_cas(
    setup, monkeypatch, count, interrupt_after,
):
    ctx = setup
    reference = transfer(ctx)
    staging = ctx.uploads.staging
    prefix = f"uploads/v1/{reference.upload_id}/"
    for index in range(count - 2):  # The original upload has a part and manifest.
        staging.put_object(Bucket="affinemodel", Key=f"{prefix}extra-{index:04d}", Body=b"small")
    staging.put_object(Bucket="affinemodel", Key="uploads/v1/another/keep", Body=b"keep")
    original = ctx.store.public_snapshot()
    deleted, listings = [], []
    list_objects, delete_object = staging.list_objects_v2, staging.delete_object
    fail = interrupt_after is not None

    def list_page(**kwargs):
        assert kwargs == {"Bucket": "affinemodel", "Prefix": prefix, "MaxKeys": 1000}
        listings.append(kwargs)
        return list_objects(**kwargs)

    def delete(**kwargs):
        assert kwargs["Key"].startswith(prefix) and "revoke" in ctx.events
        if fail and len(deleted) == interrupt_after:
            raise OSError("interrupted page deletion")
        delete_object(**kwargs)
        deleted.append(kwargs["Key"])

    monkeypatch.setattr(staging, "list_objects_v2", list_page)
    monkeypatch.setattr(staging, "delete_object", delete)
    now = NOW + CREDENTIAL_TTL + 1
    if fail:
        with pytest.raises(OSError, match="interrupted page"):
            ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                        finalized_timestamp=now, now=now)
        assert len(deleted) == interrupt_after and len(listings) == 1
        assert ctx.uploads._read(reference.upload_id)[1]["cleanup"] == "pending"
        fail = False

    remaining = count - len(deleted)
    while remaining:
        # A new owner reads only the same CAS and existing SQLite facts. No page cursor.
        with ChallengeStore.readonly(ctx.root / "state.db") as store:
            cold = PrivateUploads(
                objects=ctx.uploads.objects, staging_client=staging,
                staging_bucket="affinemodel", staging_prefix="uploads/v1",
                genesis_hash=GENESIS, store=store, max_bytes=10000, control=ctx.uploads.control,
            )
            before = len(listings)
            removed = cold.reclaim_expired(store=store, finalized_block=200,
                                           finalized_timestamp=now, now=now)
        assert len(listings) == before + 1
        assert removed == min(1000, remaining)
        remaining -= removed
        record = cold._read(reference.upload_id)[1]
        assert record["cleanup"] == ("pending" if remaining else "done")
        assert record["parent_token_id"] is None
    assert len(deleted) == len(set(deleted)) == count
    assert ctx.events.count("revoke") == 1
    assert ctx.store.public_snapshot() == original
    assert staging.objects == {
        ("affinemodel", "uploads/v1/another/keep"):
            staging.objects[("affinemodel", "uploads/v1/another/keep")],
    }
    before = len(listings)
    assert cold.reclaim_expired(store=ctx.store, finalized_block=200,
                                finalized_timestamp=now, now=now) == 0
    assert len(listings) == before


@pytest.mark.parametrize("endpoint", [
    "https://old.supabase.co/storage/v1/s3", "https://public.r2.dev", ENDPOINT + "/path",
])
def test_production_staging_has_no_old_supabase_or_public_fallback(setup, monkeypatch, endpoint):
    monkeypatch.setenv("PRIVATE_STORAGE_URL", endpoint)
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    monkeypatch.setenv("VEDA_UPLOAD_GENESIS_HASH", GENESIS)
    monkeypatch.setenv("VEDA_UPLOAD_STAGING_PROFILE", "fixture-body-profile")
    with pytest.raises(ValueError, match="R2 S3 API"):
        PrivateUploads.from_environment(store=setup.store, max_bytes=10000)


def test_profile_uploads_reuse_original_authority_and_one_private_address(setup, monkeypatch):
    ctx = setup
    authority = ctx.uploads.objects
    authority._bucket, authority._credential_role = "affinemodel", "evaluator"
    authority._prefix = "evaluation/tracks/" + "ab" * 32
    monkeypatch.setenv("PRIVATE_STORAGE_URL", ENDPOINT)
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    monkeypatch.setenv("VEDA_UPLOAD_GENESIS_HASH", GENESIS)
    monkeypatch.setenv("VEDA_UPLOAD_STAGING_PROFILE", "model-body")
    calls = []

    def client(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return ctx.uploads.staging

    monkeypatch.setattr("validator.uploads._environment_client", client)
    uploads = PrivateUploads.from_environment(store=ctx.store, max_bytes=10000, authority=authority)
    assert calls == [(ENDPOINT, {"profile_name": "model-body", "region_name": "auto"})]
    assert uploads.objects._client is authority._client
    assert uploads.objects._write_client is authority._write_client
    assert uploads.objects._prefix == "evaluation" and uploads.bucket == authority._bucket
    assert authority._prefix.endswith("ab" * 32)  # Original data scope untouched.


@pytest.mark.parametrize("bad", ["missing-profile", "empty-profile", "legacy", "bucket", "role"])
def test_upload_profiles_refuse_missing_or_mixed_configuration(setup, monkeypatch, bad):
    ctx = setup
    authority = ctx.uploads.objects
    authority._bucket, authority._credential_role = "affinemodel", "evaluator"
    monkeypatch.setenv("PRIVATE_STORAGE_URL", ENDPOINT)
    monkeypatch.setenv("PRIVATE_STORAGE_BUCKET", "affinemodel")
    monkeypatch.setenv("VEDA_UPLOAD_GENESIS_HASH", GENESIS)
    monkeypatch.setenv("VEDA_UPLOAD_STAGING_PROFILE", "model-body")
    if bad == "missing-profile":
        monkeypatch.delenv("VEDA_UPLOAD_STAGING_PROFILE")
    elif bad == "empty-profile":
        monkeypatch.setenv("VEDA_UPLOAD_STAGING_PROFILE", " ")
    elif bad == "legacy":
        monkeypatch.setenv("VEDA_UPLOAD_AUTHORITY_STORAGE_URL", "https://old-authority.example")
    elif bad == "bucket":
        authority._bucket = "other"
    else:
        authority._credential_role = "producer"
    monkeypatch.setattr("validator.uploads._environment_client",
                        lambda *_a, **_k: pytest.fail("invalid upload settings must not fall back"))
    with pytest.raises((KeyError, ValueError)):
        PrivateUploads.from_environment(store=ctx.store, max_bytes=10000, authority=authority)


def test_cloudflare_transport_mints_prefix_session_not_shared_main_key(monkeypatch):
    admin = CloudflareR2Tokens("ab" * 16, "fixture-management-token")
    calls = []
    parent_id, name = "12" * 16, "veda-" + "cd" * 32 + "-g1"
    resource = "com.cloudflare.edge.r2.bucket." + "ab" * 16 + "_default_affinemodel"
    def call(method, path, body=None):
        calls.append((method, path, body))
        if path.startswith("/tokens?"):
            return []
        if path == "/tokens/permission_groups":
            return [{"name": "Workers R2 Storage Bucket Item Write", "id": "write"}]
        if path == "/tokens":
            return {"id": parent_id, "value": "parent-value-not-persisted"}
        if path == f"/tokens/{parent_id}":
            return {"id": parent_id, "name": name,
                    "policies": [{"effect": "allow", "resources": {resource: "*"}}]}
        return {"accessKeyId": "temporary", "secretAccessKey": "temporary-secret",
                "sessionToken": "temporary-session"}
    monkeypatch.setattr(admin, "_call", call)
    parent = admin.create_parent(name, "affinemodel", 3600)
    cred = admin.mint(parent, "affinemodel", "uploads/v1/fixture/", 3600)
    assert cred["session_token"] == "temporary-session"
    assert calls[-1][2] == {"bucket": "affinemodel", "parentAccessKeyId": parent_id,
                           "permission": "object-read-write", "ttlSeconds": 3600,
                           "prefixes": ["uploads/v1/fixture/"]}
    admin.revoke(parent, name=name, bucket="affinemodel")
    assert calls[-2][:2] == ("GET", f"/tokens/{parent_id}")
    assert calls[-1][:2] == ("DELETE", f"/tokens/{parent_id}")


@pytest.mark.parametrize("operation,tamper", [
    ("revoke", "parent"), ("issue", "generation"), ("publish", "ciphertext"),
])
def test_control_uses_original_cas_not_owner_supplied_management_targets(setup, operation, tamper):
    ctx = setup
    reference = transfer(ctx)
    _, record = ctx.uploads._read(reference.upload_id)
    forged = {**record}
    if tamper == "parent":
        forged["parent_token_id"] = "shared-or-unrelated-parent"
    elif tamper == "generation":
        forged["generation"] += 1
        forged["state"] = "ISSUING"
    else:
        forged["mailbox"] = {**record["mailbox"], "ciphertext": (b"untrusted" * 8).hex()}
    before = list(ctx.events), dict(ctx.mailbox)
    with pytest.raises(ValueError, match="CAS"):
        ctx.uploads.control.handle(operation, forged)
    assert (ctx.events, ctx.mailbox) == before


def test_control_never_issues_during_original_cleanup_even_with_matching_cas(setup):
    ctx = setup
    activate(ctx)
    stored, record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))
    record = {**record, "cleanup": "pending"}
    ctx.uploads._replace(stored, record)
    with pytest.raises(ValueError, match="phase"):
        ctx.uploads.control.handle("issue", record)
    assert ctx.events == [] and ctx.mailbox == {}


@pytest.mark.parametrize("failure", ["missing", "unavailable", "corrupt"])
def test_control_cas_read_failure_never_falls_back_to_request_record(setup, monkeypatch, failure):
    ctx = setup
    activate(ctx)
    _, record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))

    def failed(_name):
        if failure == "missing":
            return None
        if failure == "corrupt":
            return SimpleNamespace(body=b"not-canonical-json")
        raise OSError("private authority unavailable")

    monkeypatch.setattr(ctx.uploads.objects, "get_optional", failed)
    with pytest.raises((ValueError, OSError)):
        ctx.uploads.control.handle("issue", record)
    assert ctx.events == [] and ctx.mailbox == {}


@pytest.mark.parametrize("alter", ["name", "bucket", "account", "id", "effect"])
def test_parent_revoke_checks_account_token_metadata_before_exact_delete(monkeypatch, alter):
    admin = CloudflareR2Tokens("ab" * 16, "fixture-management-token")
    parent, name = "12" * 16, "veda-" + "cd" * 32 + "-g1"
    resource = "com.cloudflare.edge.r2.bucket." + "ab" * 16 + "_default_affinemodel"
    policy = {"effect": "allow", "resources": {resource: "*"}}
    token = {"id": parent, "name": name, "policies": [policy]}
    if alter == "name":
        token["name"] = "another-registration-or-shared-key"
    elif alter in {"bucket", "account"}:
        resource = resource.replace("affinemodel", "otherbucket") if alter == "bucket" else (
            resource.replace("ab" * 16, "ef" * 16)
        )
        policy["resources"] = {resource: "*"}
    elif alter == "id":
        token["id"] = "34" * 16
    else:
        policy["effect"] = "deny"
    calls = []

    def read(method, path, body=None):
        calls.append((method, path))
        return token

    monkeypatch.setattr(admin, "_call", read)
    with pytest.raises(ValueError, match="registration/bucket"):
        admin.revoke(parent, name=name, bucket="affinemodel")
    assert calls == [("GET", f"/tokens/{parent}")]


def test_parent_revoke_lost_delete_response_is_idempotent_without_new_target(monkeypatch):
    admin = CloudflareR2Tokens("ab" * 16, "fixture-management-token")
    calls = []

    def deleted(method, path, body=None):
        calls.append((method, path))
        return None

    monkeypatch.setattr(admin, "_call", deleted)
    admin.revoke("12" * 16, name="veda-" + "cd" * 32 + "-g1", bucket="affinemodel")
    assert calls == [("GET", "/tokens/" + "12" * 16)]
    with pytest.raises(ValueError, match="identity"):
        admin.revoke("../arbitrary-url", name="other", bucket="affinemodel")
    assert len(calls) == 1


def test_new_generation_reconciles_prior_parent_whose_issue_response_was_lost(monkeypatch):
    admin = CloudflareR2Tokens("ab" * 16, "fixture-management-token")
    prefix = "veda-" + "cd" * 32 + "-g"
    old_id, new_id = "12" * 16, "34" * 16
    resource = "com.cloudflare.edge.r2.bucket." + "ab" * 16 + "_default_affinemodel"
    # g1 committed before its response disappeared; original CAS has no returned ID.
    old = {"id": old_id, "name": prefix + "1",
           "policies": [{"effect": "allow", "resources": {resource: "*"}}]}
    calls = []

    def call(method, path, body=None):
        calls.append((method, path))
        if path.startswith("/tokens?"):
            return [old]
        if path == "/tokens/" + old_id:
            return old
        if path == "/tokens/permission_groups":
            return [{"name": "Workers R2 Storage Bucket Item Write", "id": "write"}]
        return {"id": new_id, "value": "fixture-parent-value"}

    monkeypatch.setattr(admin, "_call", call)
    assert admin.create_parent(prefix + "2", "affinemodel", 3600) == new_id
    assert calls.index(("DELETE", "/tokens/" + old_id)) < calls.index(("POST", "/tokens"))


@pytest.mark.parametrize("alter", ["account", "jurisdiction", "public-bucket", "cas-prefix"])
def test_control_scope_is_fixed_by_configuration_not_incoming_messages(setup, alter):
    ctx = setup
    endpoint, bucket, objects = ENDPOINT, "affinemodel", ctx.uploads.objects
    if alter == "account":
        endpoint = ENDPOINT.replace("ab" * 16, "ef" * 16)
    elif alter == "jurisdiction":
        endpoint = ENDPOINT.replace(".r2.", ".eu.r2.")
    elif alter == "public-bucket":
        bucket = "affine_video"
    else:
        objects = PrivateObjectStore(MemoryS3(), bucket="private", prefix="not-evaluation")
    with pytest.raises(ValueError, match="scope"):
        MailboxControl(tokens=Tokens(ctx.events), signer=SIGNER, endpoint=endpoint, bucket=bucket,
                       genesis_hash=GENESIS, objects=objects, publish_mailbox=lambda *_a: None)
    assert ctx.events == []


@pytest.mark.parametrize("bad_seed", [
    None, "mode", "length", "symlink", "identity", "missing-manager",
])
def test_control_environment_keeps_seed_and_management_local_and_cas_readonly(
    setup, tmp_path, monkeypatch, bad_seed,
):
    ctx = setup
    seed = tmp_path / "signing.seed"
    seed.write_bytes(bytes.fromhex("22" * 32) + (b"x" if bad_seed == "length" else b""))
    seed.chmod(0o644 if bad_seed == "mode" else 0o600)
    if bad_seed == "symlink":
        link = tmp_path / "seed-link"
        link.symlink_to(seed)
        seed = link
    environment = {
        "PRIVATE_STORAGE_URL": ENDPOINT, "PRIVATE_STORAGE_BUCKET": "affinemodel",
        "VEDA_UPLOAD_GENESIS_HASH": GENESIS, "VEDA_UPLOAD_SIGNING_SEED_FILE": str(seed),
        "VEDA_UPLOAD_VALIDATOR_IDENTITY": (KEY if bad_seed == "identity" else SIGNER).ss58_address,
        "CLOUDFLARE_CONTROL_API_TOKEN": "fixture-control-token",
        "STORAGE_URL": "https://public.example", "ACCESS_KEY": "fixture-public-access",
        "SECRET_ACCESS_KEY": "fixture-public-secret",
        "EVALUATOR_PRIVATE_READ_PROFILE": "fixture-read-profile",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("VEDA_UPLOAD_MANAGEMENT_TOKEN", "fixture-obsolete-manager")
    monkeypatch.setenv("Token_value", "fixture-storage-token-not-manager")
    if bad_seed == "missing-manager":
        monkeypatch.delenv("CLOUDFLARE_CONTROL_API_TOKEN")
    closed, clients, attached, launched = [], [], [], []
    ctx.uploads.objects._bucket = "affinemodel"
    authority = ctx.uploads.objects._client
    monkeypatch.setattr(authority, "close", lambda: closed.append("authority"), raising=False)
    public = SimpleNamespace(close=lambda: closed.append("public"))

    def client(endpoint, access=None, secret=None, **kwargs):
        clients.append(endpoint)
        if endpoint == ENDPOINT:
            assert (access, secret) == (None, None)
            assert kwargs == {"profile_name": "fixture-read-profile"}
            return authority
        assert (access, secret) == ("fixture-public-access", "fixture-public-secret")
        return public

    def start():
        launched.append(True)
        return "fixture-managed-process"

    def attach(*, start_evaluator, control):
        assert start_evaluator() == "fixture-managed-process"
        assert control.tokens._token == "fixture-control-token"
        assert control.signer.ss58_address == SIGNER.ss58_address
        assert control.objects._client is authority
        activate(ctx)
        _, record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))
        # Same original decoder, no authority write or remote management call.
        with pytest.raises(ValueError, match="CAS"):
            control.handle("issue", {**record, "generation": record["generation"] + 1})
        attached.append(True)
        return 0

    monkeypatch.setattr("validator.uploads._environment_client", client)
    monkeypatch.setattr("validator.uploads.attach_evaluator", attach)
    if bad_seed == "missing-manager":
        with pytest.raises(KeyError) as failure:
            attach_from_environment(start_evaluator=start)
        assert failure.value.args == ("CLOUDFLARE_CONTROL_API_TOKEN",)
        assert clients == [] and attached == [] and launched == []
    elif bad_seed:
        with pytest.raises((ValueError, OSError)):
            attach_from_environment(start_evaluator=start)
        assert clients == [] and attached == [] and launched == []
    else:
        assert attach_from_environment(start_evaluator=start) == 0
        assert clients == ["https://public.example", ENDPOINT]
        assert closed == ["authority", "public"] and attached == [True] and launched == [True]


def test_existing_owner_attach_never_sends_temporary_plaintext_back_to_gpu(
    setup, monkeypatch, capsys,
):
    ctx = setup
    activate(ctx)
    _, record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))
    request = {"id": "ab" * 16, "operation": "issue", "record": record}

    class Input(io.StringIO):
        def close(self):
            self.saved = self.getvalue()
            super().close()

    incoming = Input()
    outgoing = io.StringIO("VEDA_MAILBOX_REQUEST " + json.dumps(request) + "\n")
    launched = []
    monkeypatch.setenv("CLOUDFLARE_CONTROL_API_TOKEN", "fixture-control-secret")
    monkeypatch.setenv("Token_value", "fixture-storage-token")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/fixture/owner-only-config")
    monkeypatch.setenv("EVALUATOR_PRIVATE_READ_PROFILE", "fixture-owner-role")
    monkeypatch.setattr("os.getpgid", lambda _pid: 1234)
    monkeypatch.setattr("os.getsid", lambda _pid: 1234)
    monkeypatch.setattr("os.killpg", lambda *_a: None)

    def start():
        launched.append(True)
        return SimpleNamespace(pid=1234, stdin=incoming, stdout=outgoing,
                               wait=lambda **_k: 0, poll=lambda: 0)

    assert attach_evaluator(start_evaluator=start,
                            control=ctx.uploads.control) == 0
    response = json.loads(incoming.saved)
    assert response["id"] == request["id"]
    assert "fixture-secret" not in incoming.saved and "fixture-session" not in incoming.saved
    assert "access_key_id" not in incoming.saved
    assert set(response["result"]) == {"parent_token_id", "retire_parent", "expires_at", "mailbox"}
    assert launched == [True]
    captured = capsys.readouterr()
    assert not captured.out and not captured.err  # No request/response frames in journal output.


def test_owner_actual_subprocess_pipes_without_listener(setup, monkeypatch, capsys):
    import subprocess
    import sys

    ctx = setup
    activate(ctx)
    _, record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))
    program = (
        "import json,os,sys; from validator.uploads import OwnerMailboxControl; "
        "assert 'CLOUDFLARE_CONTROL_API_TOKEN' not in os.environ; "
        "assert 'VEDA_UPLOAD_SIGNING_SEED_FILE' not in os.environ; "
        "r=OwnerMailboxControl().handle('issue',json.loads(sys.argv[1])); "
        "assert set(r)=={'parent_token_id','retire_parent','expires_at','mailbox'}; "
        "assert 'secret_access_key' not in json.dumps(r); print('OWNER_PIPE_OK')"
    )

    monkeypatch.setenv("CLOUDFLARE_CONTROL_API_TOKEN", "fixture-management-secret")
    monkeypatch.setenv("VEDA_UPLOAD_SIGNING_SEED_FILE", "/owner-only/seed")
    launched = []

    def child():
        process = subprocess.Popen(
            [sys.executable, "-c", program, json.dumps(record)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            env={"PATH": "/usr/bin:/bin"}, close_fds=True, start_new_session=True,
        )
        launched.append(process)
        return process

    assert attach_evaluator(start_evaluator=child,
                            control=ctx.uploads.control) == 0
    assert len(launched) == 1 and launched[0].returncode == 0
    captured = capsys.readouterr()
    assert not captured.out  # Protocol frames are consumed, business output is stderr only.
    output = captured.err
    assert "OWNER_PIPE_OK" in output
    assert "fixture-secret" not in output and "fixture-session" not in output


def test_native_scanner_activation_is_not_a_submission_and_ready_keeps_actual_s(setup, monkeypatch):
    from copy import copy

    from validator import chain

    ctx = setup
    signature = KEY.sign(activation_message(GENESIS, KEY.ss58_address))
    reference = PrivateModelSubmission(upload_registration_id(GENESIS, KEY.ss58_address),
                                       "ab" * 32, "cd" * 32)

    async def commitments(_client, *, netuid, block_height):
        payload = (UploadActivation(signature).encode() if block_height == 150
                   else reference.encode())
        block_hash = f"{block_height:064x}"
        return block_hash, [ChainCommitment(KEY.ss58_address, netuid, block_hash,
                                           ChainPosition(block_height, 2, 3), payload)]

    async def finalized():
        return 151

    monkeypatch.setattr(chain, "finalized_commitments", commitments)
    with ChallengeStore(ctx.root / "scanner.db") as store:
        uploads = copy(ctx.uploads)
        uploads.store = store
        result = asyncio.run(chain.scan_finalized_blocks(
            SimpleNamespace(finalized_block=finalized), store, netuid=120,
            track_start_block=150, private_uploads=uploads,
        ))
        assert result.submissions_found == 1
        assert store.chain_cursor()[0] == 151
        challenges = store.public_snapshot()["challenges"]
        assert len(challenges) == 1 and challenges[0]["block_height"] == 151
        assert challenges[0]["attempts"] == 0
        assert uploads._read(reference.upload_id)[1]["activation_position"] == [150, 2, 3]


def test_failed_cleanup_reactivation_can_scan_then_clean_before_reissue(setup, monkeypatch):
    from validator import chain

    ctx = setup
    reference = transfer(ctx)
    now = NOW + CREDENTIAL_TTL + 1
    delete = ctx.uploads.staging.delete_object
    monkeypatch.setattr(ctx.uploads.staging, "delete_object",
                        lambda **_kw: (_ for _ in ()).throw(OSError("delete unavailable")))
    with pytest.raises(OSError):
        ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                    finalized_timestamp=now, now=now)
    assert ctx.uploads._read(reference.upload_id)[1]["cleanup"] == "pending"
    signature = KEY.sign(activation_message(GENESIS, KEY.ss58_address))

    async def commitments(_client, *, netuid, block_height):
        block_hash = f"{block_height:064x}"
        return block_hash, ([ChainCommitment(
            KEY.ss58_address, netuid, block_hash, ChainPosition(block_height, 1, 1),
            UploadActivation(signature).encode(),
        )] if block_height == 201 else [])

    async def finalized():
        return 202

    monkeypatch.setattr(chain, "finalized_commitments", commitments)
    result = asyncio.run(chain.scan_finalized_blocks(
        SimpleNamespace(finalized_block=finalized), ctx.store, netuid=120,
        track_start_block=150, private_uploads=ctx.uploads,
    ))
    assert result.submissions_found == 0 and ctx.store.chain_cursor()[0] == 202
    record = ctx.uploads._read(reference.upload_id)[1]
    assert record["activation_request"]["position"] == [201, 1, 1]
    assert record["generation"] == 1 and record["cleanup"] == "pending"
    assert not ctx.uploads.issue_pending(now=now)  # No credentials during deletion.
    with pytest.raises(RuntimeError, match="scan"):
        ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=203,
                                    finalized_timestamp=now, now=now)
    monkeypatch.setattr(ctx.uploads.staging, "delete_object", delete)
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=202,
                                      finalized_timestamp=now, now=now) == 2
    ctx.uploads.registration = lambda _h: (9, 202)
    assert ctx.uploads.issue_pending(now=now)
    record = ctx.uploads._read(reference.upload_id)[1]
    assert record["generation"] == 2 and record["activation_position"] == [201, 1, 1]
    assert "cleanup" not in record and ctx.store.public_snapshot()["challenges"] == []


def test_mailbox_issue_retry_survives_cold_owner_and_clears_only_after_recovery(setup, monkeypatch):
    from copy import copy

    ctx = setup
    activate(ctx)
    control = ctx.uploads.control.handle
    calls = []

    def denied(operation, record):
        calls.append(operation)
        raise RuntimeError("fixture-secret endpoint authorization detail")

    monkeypatch.setattr(ctx.uploads.control, "handle", denied)
    upload_id = upload_registration_id(GENESIS, KEY.ss58_address)
    now = NOW
    for attempt in range(1, 7):
        with pytest.raises(RuntimeError):
            ctx.uploads.issue_pending(now=now)
        retry = ctx.uploads._read(upload_id)[1]["issue_retry"]
        assert retry["attempts"] == attempt
        assert 30 <= retry["retry_after"] - now <= 300
        assert retry["error"] == "PRIVATE_UPLOAD_ISSUANCE_FAILED"
        # Fresh process projection and cooldown both come from the SAME original CAS.
        cold = copy(ctx.uploads)
        cold._issue_after = ""
        assert cold.maintenance_status()["private_upload_error"] == "PRIVATE_UPLOAD_ISSUANCE_FAILED"
        assert not cold.issue_pending(now=retry["retry_after"] - 1)
        assert len(calls) == attempt
        now = retry["retry_after"]
    assert b"fixture-secret" not in ctx.uploads._read(upload_id)[0].body
    monkeypatch.setattr(ctx.uploads.control, "handle", control)
    assert cold.issue_pending(now=now)
    assert cold.maintenance_status()["private_upload_error"] is None
    assert "issue_retry" not in cold._read(upload_id)[1]
    assert ctx.store.public_snapshot()["challenges"] == []


def test_another_mailbox_success_does_not_clear_original_durable_issue_error(setup, monkeypatch):
    ctx = setup
    activate(ctx)
    control = ctx.uploads.control.handle

    def denied(operation, record):
        if record["hotkey"] == KEY.ss58_address:
            raise RuntimeError("management unavailable")
        return control(operation, record)

    monkeypatch.setattr(ctx.uploads.control, "handle", denied)
    with pytest.raises(RuntimeError):
        ctx.uploads.issue_pending(now=NOW)
    activate(ctx, key=SIGNER)
    assert ctx.uploads.issue_pending(now=NOW)
    assert ctx.uploads.maintenance_status()["private_upload_error"] == (
        "PRIVATE_UPLOAD_ISSUANCE_FAILED"
    )


def test_final_mailbox_cas_response_loss_uses_original_completed_fact(setup, monkeypatch):
    ctx = setup
    activate(ctx)
    replace_record = ctx.uploads._replace

    def lost(stored, record):
        result = replace_record(stored, record)
        if (record.get("mailbox") or {}).get("published"):
            raise OSError("final CAS response lost")
        return result

    monkeypatch.setattr(ctx.uploads, "_replace", lost)
    assert ctx.uploads.issue_pending(now=NOW)
    assert not ctx.uploads.issue_pending(now=NOW)
    assert ctx.events == ["parent", "mint"] and len(ctx.mailbox) == 1
    assert ctx.uploads.maintenance_status() == {
        "private_upload_error": None, "private_upload_cleanup_error": None,
    }


def test_pending_cleanup_error_survives_other_cleanup_and_cold_status(setup, monkeypatch):
    from copy import copy

    ctx = setup
    reference = transfer(ctx)
    activate(ctx, key=SIGNER)
    ctx.uploads.issue_pending(now=NOW)
    other_id = upload_registration_id(GENESIS, SIGNER.ss58_address)
    ctx.uploads.staging.put_object(Bucket="affinemodel",
                                   Key=f"uploads/v1/{other_id}/part-00000", Body=b"fixture")
    delete = ctx.uploads.staging.delete_object

    def failure(**kwargs):
        if reference.upload_id in kwargs["Key"]:
            raise OSError("original prefix delete failed")
        return delete(**kwargs)

    monkeypatch.setattr(ctx.uploads.staging, "delete_object", failure)
    now = NOW + CREDENTIAL_TTL + 1
    # Start immediately after B so the original bounded/wrapping iterator visits A first.
    ctx.uploads._gc_after = ctx.uploads._record_name(other_id)
    with pytest.raises(OSError):
        ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                    finalized_timestamp=now, now=now)
    assert ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                      finalized_timestamp=now, now=now) in (0, 1)
    assert copy(ctx.uploads).maintenance_status()["private_upload_cleanup_error"] == (
        "PRIVATE_UPLOAD_CLEANUP_FAILED"
    )
    monkeypatch.setattr(ctx.uploads.staging, "delete_object", delete)
    ctx.uploads.reclaim_expired(store=ctx.store, finalized_block=200,
                                finalized_timestamp=now, now=now)
    assert copy(ctx.uploads).maintenance_status()["private_upload_cleanup_error"] is None


@pytest.mark.parametrize("failure", [None, "block_hash", "uid"])
def test_owner_issuance_uses_one_authenticated_scanned_snapshot(setup, monkeypatch, failure):
    from datetime import UTC, datetime

    from validator import run

    ctx = setup
    activate(ctx)
    upload_id = upload_registration_id(GENESIS, KEY.ss58_address)
    original_record = ctx.uploads._read(upload_id)[1]
    original_cursor = ctx.store.chain_cursor()
    original_matches = ctx.store.public_snapshot()
    calls = []
    clock = [NOW]
    broken = [failure]
    monkeypatch.setattr("validator.uploads.time.time", lambda: clock[0])

    class Client:
        def __init__(self):
            self.height = None
            self._substrate = SimpleNamespace(block_hash=self.block_hash)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def block_info(self, height):
            assert height == 0
            return SimpleNamespace(hash="0x" + GENESIS)

        async def finalized_block(self):
            head = 201 + sum(call[0] == "head" for call in calls)
            calls.append(("head", head))
            return head

        async def block_hash(self, height):
            calls.append(("hash", height))
            if broken[0] == "block_hash":
                raise OSError("fixture RPC unavailable")
            assert height == original_cursor[0]
            return "0x" + original_cursor[1].removeprefix("0x")

        async def at(self, height):
            self.height = height
            return self

        async def timestamp(self):
            return datetime.fromtimestamp(clock[0], UTC)

        async def read(self, field, **kwargs):
            calls.append(("uid", self.height))
            assert field == "uid" and kwargs == {"hotkey_ss58": KEY.ss58_address,
                                                  "netuid": 120}
            if broken[0] == "uid":
                raise OSError("fixture UID unavailable")
            return 9

    monkeypatch.setattr(run.bt, "Client", lambda *_a, **_kw: Client())

    def maintain():
        return asyncio.run(run._reclaim_private_uploads(
            ctx.uploads, database=ctx.root / "state.db", network="finney", archive_endpoints=None,
        ))

    if failure:
        assert maintain()["private_upload_error"] == "PRIVATE_UPLOAD_ISSUANCE_FAILED"
        failed = ctx.uploads._read(upload_id)[1]
        assert failed["state"] == "ISSUING" and failed["issue_retry"]["attempts"] == 1
        assert not ctx.uploads.control.tokens.parents
        broken[0] = None
        ctx.uploads._issue_after = ""  # Cold owner starts from the original CAS.
        assert maintain()["private_upload_error"] == "PRIVATE_UPLOAD_ISSUANCE_FAILED"
        assert not ctx.uploads.control.tokens.parents  # Durable cooldown is not bypassed.
        clock[0] += 30
    assert maintain() == {"private_upload_error": None, "private_upload_cleanup_error": None}
    record = ctx.uploads._read(upload_id)[1]
    assert record["state"] == "UPLOADING" and record["mailbox"]["published"]
    assert "issue_retry" not in record
    for name in ("upload_id", "hotkey", "generation", "activation_position",
                 "activation_block_hash"):
        assert record[name] == original_record[name]
    assert len(ctx.uploads.control.tokens.parents) == 1
    assert ctx.store.chain_cursor() == original_cursor
    assert ctx.store.public_snapshot() == original_matches
    assert [call for call in calls if call[0] == "uid"]
    assert all(height == 200 for kind, height in calls if kind in {"uid", "hash"})


@pytest.mark.parametrize("failure", ["missing_cursor", "not_finalized", "hash_mismatch"])
def test_owner_issuance_refuses_unauthenticated_scanned_snapshot(setup, monkeypatch, failure):
    from datetime import UTC, datetime

    from validator import run

    ctx = setup
    activate(ctx)
    database = ctx.root / "state.db"
    if failure == "missing_cursor":
        database = ctx.root / "unscanned.db"
        with ChallengeStore(database):
            pass

    class Client:
        def __init__(self):
            self._substrate = SimpleNamespace(block_hash=self.block_hash)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def block_info(self, _height):
            return SimpleNamespace(hash="0x" + GENESIS)

        async def finalized_block(self):
            return 199 if failure == "not_finalized" else 201

        async def block_hash(self, _height):
            return "0x" + "cd" * 32

        async def at(self, _height):
            return self

        async def timestamp(self):
            return datetime.fromtimestamp(NOW, UTC)

        async def read(self, *_args, **_kwargs):
            pytest.fail("UID lookup must not use an unauthenticated cursor")

    monkeypatch.setattr(run.bt, "Client", lambda *_a, **_kw: Client())
    assert asyncio.run(run._reclaim_private_uploads(
        ctx.uploads, database=database, network="finney", archive_endpoints=None,
    ))["private_upload_error"] == "PRIVATE_UPLOAD_ISSUANCE_FAILED"
    record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))[1]
    assert record["state"] == "ISSUING" and record["issue_retry"]["attempts"] == 1
    assert record["issue_retry"]["error"] == "PRIVATE_UPLOAD_ISSUANCE_FAILED"
    assert not ctx.uploads.control.tokens.parents
    assert ctx.store.chain_cursor() == (200, "0x" + "ab" * 32)


def test_owner_maintenance_keeps_gc_and_durable_issue_status_independent(setup, monkeypatch):
    from datetime import UTC, datetime

    from scoring.spec import parse_track_spec
    from tests.scoring.test_spec import valid_spec
    from tests.validator.test_publication import MemoryS3 as PublicMemoryS3
    from validator import run
    from validator.publication import build_public_artifacts, decode_weight_artifact
    from validator.weights import WeightChainState

    ctx = setup
    activate(ctx)
    calls = []
    control = ctx.uploads.control.handle

    def denied(_operation, _record):
        calls.append("issue")
        raise RuntimeError("private backend endpoint and fixture-secret")

    monkeypatch.setattr(ctx.uploads.control, "handle", denied)
    reclaim = PrivateUploads.reclaim_expired

    def observed_gc(worker, **kwargs):
        calls.append("gc")
        return reclaim(worker, **kwargs)

    monkeypatch.setattr(PrivateUploads, "reclaim_expired", observed_gc)

    class Client:
        def __init__(self):
            self._substrate = SimpleNamespace(block_hash=self.block_hash)

        async def block_hash(self, _height):
            return "0x" + ("ab" * 32 if _height == 200 else f"{_height:064x}")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def block_info(self, _height):
            return SimpleNamespace(hash="0x" + GENESIS)

        async def finalized_block(self):
            return 201

        async def at(self, _height):
            return self

        async def timestamp(self):
            return datetime.fromtimestamp(NOW, UTC)

        async def read(self, _field, **_kwargs):
            return 9

    monkeypatch.setattr(run.bt, "Client", lambda *_a, **_kw: Client())
    track = parse_track_spec(valid_spec())
    public = PublicMemoryS3()
    publisher = S3Repository(public, bucket="fixture", prefix="test")

    def publish(status, block):
        if block > 200:
            ctx.store.advance_chain_cursor(block, f"{block:064x}")
        chain = WeightChainState(
            finalized_block=block, uid_by_hotkey={track.initial_champion.hotkey: 0},
            min_allowed_weights=1, max_weight=1.0, weights_rate_limit=1,
            last_update_by_hotkey={},
        )
        artifacts = build_public_artifacts(
            ctx.store, track, chain, version_key=0,
            inventory_status={"allocated_stages": [], **status},
        )
        assert artifacts is not None
        asyncio.run(publisher.publish(artifacts))
        weights = decode_weight_artifact(public.objects["test/weights.json"])
        assert weights.source_block == block and weights.plan.uids == (0,)
        assert json.loads(public.objects["test/status.json"])["inventory"] == status

    for block in (200, 201):
        status = asyncio.run(run._reclaim_private_uploads(
            ctx.uploads, database=ctx.root / "state.db", network="finney", archive_endpoints=None,
        ))
        assert status == {"private_upload_error": "PRIVATE_UPLOAD_ISSUANCE_FAILED",
                          "private_upload_cleanup_error": None}
        publish(status, block)
    assert calls == ["issue", "gc", "gc"]
    monkeypatch.setattr(ctx.uploads.control, "handle", control)
    monkeypatch.setattr("validator.uploads.time.time", lambda: NOW + 30)
    status = asyncio.run(run._reclaim_private_uploads(
        ctx.uploads, database=ctx.root / "state.db", network="finney", archive_endpoints=None,
    ))
    assert status == {"private_upload_error": None, "private_upload_cleanup_error": None}
    publish(status, 202)
    assert ctx.store.public_snapshot()["challenges"] == []


def test_missing_original_part_can_be_repaired_without_replacing_ready_or_s(setup):
    ctx = setup
    reference = transfer(ctx)
    identifier, _ = submit(ctx, reference)
    part = ("affinemodel", f"uploads/v1/{reference.upload_id}/part-00000")
    original = ctx.uploads.staging.objects.pop(part)
    with pytest.raises(ModelDownloadError) as failed:
        ctx.uploads.download(reference, hotkey=KEY.ss58_address,
                             cache_dir=ctx.root / "first", max_bytes=10000)
    assert failed.value.temporary
    ctx.store.advance_chain_cursor(201, "cd" * 32)
    ctx.uploads.registration = lambda _h: (9, 201)
    activate(ctx, block=201)
    assert ctx.uploads.issue_pending(now=NOW)
    ctx.uploads.staging.objects[part] = original
    result = ctx.uploads.download(reference, hotkey=KEY.ss58_address,
                                  cache_dir=ctx.root / "fixed", max_bytes=10000)
    assert result.file_sha256 == reference.file_sha256
    assert ctx.store.public_challenge(identifier)["block_height"] == 200


def test_refused_reactivation_never_blocks_original_ready_admission(setup):
    ctx = setup
    reference = transfer(ctx)
    submit(ctx, reference)
    ctx.store.advance_chain_cursor(201, "cd" * 32)
    activate(ctx, block=201)
    ctx.uploads.registration = lambda _h: (None, 201)
    with pytest.raises(ValueError, match="registered"):
        ctx.uploads.issue_pending(now=NOW)
    record = ctx.uploads._read(reference.upload_id)[1]
    assert record["state"] == "UPLOADING" and "activation_request" not in record
    assert ctx.uploads.download(reference, hotkey=KEY.ss58_address,
                                cache_dir=ctx.root / "original", max_bytes=10000).file_sha256 == (
        reference.file_sha256
    )


def test_only_explicit_system_not_admitted_can_release_original_file_binding(setup):
    ctx = setup
    reference = transfer(ctx)
    identifier, _ = submit(ctx, reference)
    ctx.uploads.download(reference, hotkey=KEY.ss58_address,
                         cache_dir=ctx.root / "original", max_bytes=10000)
    ctx.store.transition(identifier, ChallengeState.FETCHING)
    ctx.store.transition(identifier, ChallengeState.SYSTEM_NOT_ADMITTED,
                         error_code="NO_PRE_SUBMISSION_DATA")
    original = ctx.store.public_challenge(identifier)
    ctx.store.advance_chain_cursor(201, "cd" * 32)
    ctx.uploads.registration = lambda _h: (9, 201)
    activate(ctx, block=201)
    assert ctx.uploads.issue_pending(now=NOW)
    record = ctx.uploads._read(reference.upload_id)[1]
    assert "file_sha256" not in record and "manifest_sha256" not in record
    assert ctx.store.public_challenge(identifier) == original


def test_existing_public_repository_only_adds_encrypted_mailbox_not_four_pointers():
    client = MemoryS3()
    publisher = S3Repository(client, bucket="affine_video", prefix="tracks/main")
    try:
        key = "mailbox/v1/" + "ab" * 32 + ".bin"
        publisher.publish_mailbox(key, b"encrypted-fixture" * 4)
        assert set(client.objects) == {("affine_video", "tracks/main/" + key)}
        for forbidden in ("track.json", "weights.json", "status.json", "summary.json"):
            with pytest.raises(PublicArtifactError):
                publisher.publish_mailbox(forbidden, b"encrypted-fixture" * 4)
    finally:
        publisher.close()


def test_malformed_owner_response_is_infrastructure_not_a_miner_binding_error(monkeypatch):
    import sys

    reader = SimpleNamespace(buffer=SimpleNamespace(raw=io.BytesIO(b"partial-response\n")))
    monkeypatch.setattr(sys, "stdin", reader)
    monkeypatch.setattr("select.select", lambda *_args: ([reader], [], []))
    with pytest.raises(RuntimeError, match="response is invalid"):
        OwnerMailboxControl().handle("revoke", {})


@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_owner_returns_child_exit_without_relaunch(setup, capsys, exit_code):
    import subprocess
    import sys

    launched = []

    def start():
        process = subprocess.Popen(
            [sys.executable, "-c", f"print('BUSINESS_LOG'); raise SystemExit({exit_code})"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={"PATH": "/usr/bin:/bin"}, close_fds=True, start_new_session=True,
        )
        launched.append(process)
        return process

    assert attach_evaluator(start_evaluator=start, control=setup.uploads.control) == exit_code
    assert len(launched) == 1
    assert launched[0].stdin.closed and launched[0].stdout.closed
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "BUSINESS_LOG\n"


def test_native_owner_launch_failure_does_not_retry_or_touch_control(setup):
    launched = []

    def start():
        launched.append(True)
        raise PermissionError("fixture child credential drop failed")

    with pytest.raises(PermissionError):
        attach_evaluator(start_evaluator=start, control=setup.uploads.control)
    assert launched == [True] and setup.events == []


def test_native_owner_closes_group_even_when_direct_child_already_exited(setup, monkeypatch):
    import signal

    signals = []
    process = SimpleNamespace(pid=1234, stdin=io.StringIO(), stdout=io.StringIO(),
                              poll=lambda: 0, wait=lambda **_kw: 0)
    monkeypatch.setattr("os.getpgid", lambda _pid: 1234)
    monkeypatch.setattr("os.getsid", lambda _pid: 1234)
    monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))
    assert attach_evaluator(start_evaluator=lambda: process, control=setup.uploads.control) == 0
    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("fault", ["oversize", "broken-pipe", "interrupt", "missing-pipe"])
def test_native_owner_failure_reaps_owned_process_group(setup, monkeypatch, fault):
    import signal
    import subprocess

    events = []

    class Output(io.StringIO):
        def readline(self, *_args):
            if fault == "interrupt":
                raise KeyboardInterrupt
            return super().readline(*_args)

    class Input(io.StringIO):
        def write(self, text):
            if fault == "broken-pipe":
                raise BrokenPipeError
            return super().write(text)

    output = Output("x" * 262145 if fault == "oversize" else "VEDA_MAILBOX_REQUEST {}\n")
    process = SimpleNamespace(pid=1234, stdin=Input(), stdout=output, poll=lambda: None)
    if fault == "missing-pipe":
        process.stdout = None

    def wait(**kwargs):
        events.append(("wait", kwargs))
        if kwargs.get("timeout"):
            raise subprocess.TimeoutExpired("fixture child", kwargs["timeout"])
        return -signal.SIGKILL

    process.wait = wait
    monkeypatch.setattr("os.getpgid", lambda _pid: 1234)
    monkeypatch.setattr("os.getsid", lambda _pid: 1234)
    monkeypatch.setattr("os.killpg", lambda pid, sig: events.append((pid, sig)))
    error = {"oversize": RuntimeError, "broken-pipe": BrokenPipeError,
             "interrupt": KeyboardInterrupt, "missing-pipe": ValueError}[fault]
    with pytest.raises(error):
        attach_evaluator(start_evaluator=lambda: process, control=setup.uploads.control)
    assert events == [(1234, signal.SIGTERM), ("wait", {"timeout": 10}),
                      (1234, signal.SIGKILL), ("wait", {})]
    assert process.stdin.closed and output.closed == (fault != "missing-pipe")


def test_native_owner_rejects_shared_session_without_signalling_owners_group(setup, monkeypatch):
    events = []
    process = SimpleNamespace(
        pid=1234, stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None,
        terminate=lambda: events.append("terminate"), wait=lambda **_kw: events.append("wait"),
    )
    monkeypatch.setattr("os.getpgid", lambda _pid: 999)
    monkeypatch.setattr("os.getsid", lambda _pid: 999)
    monkeypatch.setattr("os.killpg", lambda *_a: pytest.fail("never signal an unowned group"))
    with pytest.raises(ValueError, match="session"):
        attach_evaluator(start_evaluator=lambda: process, control=setup.uploads.control)
    assert events == ["terminate", "wait"]
