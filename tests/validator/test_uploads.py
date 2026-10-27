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
    CloudflareR2Tokens,
    MailboxControl,
    OwnerMailboxControl,
    PrivateUploads,
    attach_evaluator,
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


class Tokens:
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

    def revoke(self, parent):
        self.events.append("revoke")
        self.revoked.add(parent)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("validator.uploads.time.time", lambda: NOW)
    events, mailbox = [], {}
    staging = StagingS3(events)
    with ChallengeStore(tmp_path / "state.db") as store:
        store.advance_chain_cursor(200, "ab" * 32)
        uploads = PrivateUploads(
            objects=PrivateObjectStore(MemoryS3(), bucket="private", prefix="evaluation"),
            staging_client=staging, staging_bucket="affinemodel", staging_prefix="uploads/v1",
            genesis_hash=GENESIS, store=store, registration=lambda _h: (9, 200),
            max_bytes=10000, control=MailboxControl(
                tokens=Tokens(events), signer=SIGNER, endpoint=ENDPOINT,
                bucket="affinemodel", genesis_hash=GENESIS,
                publish_mailbox=lambda name, body: mailbox.__setitem__(name, body),
            ),
        )
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
    monkeypatch.setattr("validator.production.batches._submission_timestamp_ms", lambda _b: 1000)
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
    # Original archive is sufficient even after all R2 bytes disappear.
    ctx.uploads.staging.objects.clear()
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
    def unavailable(_parent):
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
                                      finalized_timestamp=now, now=now) == 2
    assert ctx.store.public_challenge(identifier) == original
    assert ctx.uploads.staging.objects[("affinemodel", "uploads/v1/another/part")][0] == b"keep"


@pytest.mark.parametrize("endpoint", [
    "https://old.supabase.co/storage/v1/s3", "https://public.r2.dev", ENDPOINT + "/path",
])
def test_production_staging_has_no_old_supabase_or_public_fallback(setup, monkeypatch, endpoint):
    values = {
        "GENESIS_HASH": GENESIS, "AUTHORITY_BUCKET": "private",
        "STAGING_BUCKET": "affinemodel", "AUTHORITY_STORAGE_URL": "https://private.example",
        "AUTHORITY_ACCESS_KEY": "fixture", "AUTHORITY_SECRET_ACCESS_KEY": "fixture",
        "STAGING_STORAGE_URL": endpoint, "STAGING_ACCESS_KEY": "fixture",
        "STAGING_SECRET_ACCESS_KEY": "fixture",
    }
    for name, value in values.items():
        monkeypatch.setenv("VEDA_UPLOAD_" + name, value)
    with pytest.raises(ValueError, match="R2 S3 API"):
        PrivateUploads.from_environment(store=setup.store, max_bytes=10000)


def test_cloudflare_transport_mints_prefix_session_not_shared_main_key(monkeypatch):
    admin = CloudflareR2Tokens("ab" * 16, "fixture-management-token")
    calls = []
    def call(method, path, body=None):
        calls.append((method, path, body))
        if path.startswith("/tokens?"):
            return []
        if path == "/tokens/permission_groups":
            return [{"name": "Workers R2 Storage Bucket Item Write", "id": "write"}]
        if path == "/tokens":
            return {"id": "parent", "value": "parent-value-not-persisted"}
        return {"accessKeyId": "temporary", "secretAccessKey": "temporary-secret",
                "sessionToken": "temporary-session"}
    monkeypatch.setattr(admin, "_call", call)
    parent = admin.create_parent("veda-registration", "affinemodel", 3600)
    cred = admin.mint(parent, "affinemodel", "uploads/v1/fixture/", 3600)
    assert cred["session_token"] == "temporary-session"
    assert calls[-1][2] == {"bucket": "affinemodel", "parentAccessKeyId": "parent",
                           "permission": "object-read-write", "ttlSeconds": 3600,
                           "prefixes": ["uploads/v1/fixture/"]}
    admin.revoke(parent)
    assert calls[-1][:2] == ("DELETE", "/tokens/parent")


def test_existing_owner_attach_never_sends_temporary_plaintext_back_to_gpu(setup, monkeypatch):
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
    observed = []
    environments = []
    monkeypatch.setenv("VEDA_UPLOAD_MANAGEMENT_TOKEN", "fixture-control-secret")

    def process(argv, **kwargs):
        observed.append(argv)
        environments.append(kwargs["env"])
        return SimpleNamespace(stdin=incoming, stdout=outgoing, wait=lambda: 0, poll=lambda: 0)

    monkeypatch.setattr("subprocess.Popen", process)
    assert attach_evaluator(host="gpu-veda", container="veda-evaluator",
                            control=ctx.uploads.control) == 0
    response = json.loads(incoming.saved)
    assert response["id"] == request["id"]
    assert "fixture-secret" not in incoming.saved and "fixture-session" not in incoming.saved
    assert "access_key_id" not in incoming.saved
    assert set(response["result"]) == {"parent_token_id", "retire_parent", "expires_at", "mailbox"}
    assert "docker attach --sig-proxy=false veda-evaluator" in observed[0][-1]
    assert "docker start --attach --interactive veda-evaluator" in observed[0][-1]
    assert "VEDA_UPLOAD_MANAGEMENT_TOKEN" not in environments[0]


def test_owner_actual_subprocess_pipes_without_listener(setup, monkeypatch, capsys):
    import subprocess
    import sys

    ctx = setup
    activate(ctx)
    _, record = ctx.uploads._read(upload_registration_id(GENESIS, KEY.ss58_address))
    real_popen = subprocess.Popen
    program = (
        "import json,sys; from validator.uploads import OwnerMailboxControl; "
        "r=OwnerMailboxControl().handle('issue',json.loads(sys.argv[1])); "
        "assert set(r)=={'parent_token_id','retire_parent','expires_at','mailbox'}; "
        "assert 'secret_access_key' not in json.dumps(r); print('OWNER_PIPE_OK')"
    )

    def child(_argv, **kwargs):
        return real_popen([sys.executable, "-c", program, json.dumps(record)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", child)
    assert attach_evaluator(host="gpu-veda", container="veda-evaluator",
                            control=ctx.uploads.control) == 0
    output = capsys.readouterr().out
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
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def block_info(self, _height):
            return SimpleNamespace(hash="0x" + GENESIS)

        async def finalized_block(self):
            return 200

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


@pytest.mark.parametrize("running", [False, True])
def test_owner_reconnect_attaches_without_restarting_running_writer(
    setup, tmp_path, monkeypatch, capsys, running,
):
    import subprocess

    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "$1" = inspect ]; then echo '
        + ("true" if running else "false") + '; else printf "%s\\n" "$*"; fi\n',
    )
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + ":/usr/bin:/bin")
    real_popen = subprocess.Popen

    def isolated_shell(argv, **kwargs):
        # Execute the real fixed remote command with a fixture Docker binary,
        # never SSH/connect to a daemon/start a business container.
        return real_popen(["/bin/sh", "-c", argv[-1]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", isolated_shell)
    assert attach_evaluator(host="gpu-veda", container="veda-evaluator",
                            control=setup.uploads.control) == 0
    expected = ("attach --sig-proxy=false veda-evaluator" if running
                else "start --attach --interactive veda-evaluator")
    assert capsys.readouterr().out.strip() == expected
