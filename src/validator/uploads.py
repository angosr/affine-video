"""Private R2 activation/mailbox/ready, sharing original CAS and ADMISSION.

No HTTP listener or presigned authorization. Management operations and the
mailbox signer are injected by the control owner, never sent to miners/GPU.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from miner.submit import (
    PrivateModelSubmission,
    UploadActivation,
    activation_message,
    private_model_source,
    upload_registration_id,
)
from scoring.model import model_file_sha256
from validator.chunks import ChunkManifest, verify_chunked_content
from validator.download import DownloadedModel, ModelDownloadError
from validator.production.commitment import canonical_json, decode_canonical_json
from validator.production.storage import (
    PreconditionFailed,
    PrivateObjectStore,
    _environment_client,
    _validated_endpoint,
    is_precondition_failed,
)
from validator.store import ChallengeStore

CREDENTIAL_TTL = 3600
MAX_MANIFEST_BYTES = 64 * 1024
MODEL_RETENTION_MS = 30 * 86400 * 1000
MODEL_REFERENCE_FORMAT = "veda-r2-model-reference-v1"


def decode_model_reference(payload: bytes) -> dict:
    """The original model slot may hold a fixed R2 body reference, never a URL/key."""
    row = decode_canonical_json(payload)
    if (not isinstance(row, dict) or set(row) != {
        "format", "hotkey", "private_upload", "file_sha256", "model_digest", "size_bytes",
        "submission_block", "submission_timestamp_ms", "retain_until_ms",
    } or row["format"] != MODEL_REFERENCE_FORMAT):
        raise ValueError("private model reference is invalid")
    reference = private_model_source(row)
    if (reference is None or reference.manifest_sha256 is None
            or reference.file_sha256 != row["file_sha256"]
            or not isinstance(row["hotkey"], str) or not row["hotkey"]
            or not re.fullmatch(r"[a-f0-9]{64}", str(row["model_digest"]))
            or any(type(row[key]) is not int or row[key] <= 0
                   for key in ("size_bytes", "submission_block", "submission_timestamp_ms"))
            or row["retain_until_ms"] != row["submission_timestamp_ms"] + MODEL_RETENTION_MS):
        raise ValueError("private model reference identity is invalid")
    return row


def ed25519_verify(hotkey: str, message: bytes, signature: bytes) -> bool:
    from bittensor.wallet import Keypair

    try:
        return len(signature) == 64 and bool(
            Keypair(ss58_address=hotkey, crypto_type=0).verify(message, signature),
        )
    except (ValueError, TypeError):
        return False


def manifest_signing_bytes(document: dict) -> bytes:
    return b"veda-upload-manifest-v1\0" + canonical_json(
        {key: value for key, value in document.items() if key != "signature"},
    )


def envelope_signing_bytes(document: dict) -> bytes:
    return b"veda-mailbox-v1\0" + canonical_json(
        {key: value for key, value in document.items() if key != "signature"},
    )


def _read_upload_record(objects: PrivateObjectStore, genesis_hash: str, upload_id: str):
    """One decoder for the SAME CAS on the GPU owner and read-only control side."""
    PrivateModelSubmission(upload_id, "00" * 32)
    stored = objects.get_optional(f"uploads/{upload_id}.json")
    if stored is None:
        return None, None
    record = decode_canonical_json(stored.body)
    if (not isinstance(record, dict) or record.get("format") != "veda-upload-v2"
            or record.get("upload_id") != upload_id
            or record.get("state") not in {"ISSUING", "UPLOADING", "SEALED"}
            or upload_registration_id(genesis_hash, record.get("hotkey", "")) != upload_id
            or type(record.get("expires_at")) is not int):
        raise ValueError("private upload record is invalid")
    if "model_reference" in record:
        model = decode_model_reference(canonical_json(record["model_reference"]))
        if (model["hotkey"] != record["hotkey"]
                or model["private_upload"]["upload_id"] != upload_id
                or model["file_sha256"] != record.get("file_sha256")
                or model["private_upload"]["manifest_sha256"] != record.get("manifest_sha256")
                or model["size_bytes"] != record.get("size_bytes")):
            raise ValueError("retained model differs from its sealed upload")
    formal = record.get("formal_submission")
    if formal is not None and (
        not isinstance(formal, dict) or set(formal) != {
            "submission_block", "submission_timestamp_ms", "retain_until_ms",
        } or any(type(formal[key]) is not int or formal[key] <= 0 for key in formal)
        or formal["retain_until_ms"] != formal["submission_timestamp_ms"] + MODEL_RETENTION_MS
        or ("model_reference" in record and any(
            record["model_reference"][key] != formal[key] for key in formal
        ))
    ):
        raise ValueError("private retention differs from original formal submission")
    if record.get("champion_archive") is not None and (
        "model_reference" not in record
        or record["champion_archive"] != f"champions/v1/{record.get('file_sha256')}/"
    ):
        raise ValueError("private champion archive is invalid")
    retry = record.get("issue_retry")
    if retry is not None and (
        not isinstance(retry, dict) or set(retry) != {"attempts", "retry_after", "error"}
        or type(retry["attempts"]) is not int or retry["attempts"] <= 0
        or type(retry["retry_after"]) is not int or retry["retry_after"] < 0
        or retry["error"] not in {None, "PRIVATE_UPLOAD_ISSUANCE_FAILED"}
    ):
        raise ValueError("private upload issuance retry is invalid")
    return stored, record


class CloudflareR2Tokens:
    """Control-only parent-token mint/revoke; no bucket creation or shared miner key.

    Adapted from upstream Affine f4203d8 r2.py. Raw API errors/response bodies
    never become public errors or logs. This client is NOT constructed on GPU.
    """

    def __init__(self, account: str, token: str, jurisdiction: str = "default") -> None:
        if re.fullmatch(r"[a-f0-9]{32}", account) is None or not token:
            raise ValueError("Cloudflare control credentials are required")
        if jurisdiction not in {"default", "eu", "us", "fedramp"}:
            raise ValueError("unsupported R2 jurisdiction")
        self.account, self._token = account, token
        self.jurisdiction = jurisdiction

    def _call(self, method: str, path: str, body: dict | None = None):
        import httpx

        try:
            response = httpx.request(
                method, f"https://api.cloudflare.com/client/v4/accounts/{self.account}{path}",
                headers={"Authorization": f"Bearer {self._token}"}, json=body, timeout=10,
            )
            if response.status_code == 404 and (method == "DELETE" or (
                method == "GET" and re.fullmatch(r"/tokens/[a-f0-9]{32}", path)
            )):
                return None
            data = response.json()
            if response.status_code >= 400 or not data.get("success"):
                raise RuntimeError("Cloudflare control operation was not confirmed")
            return data["result"]
        except Exception as exc:
            raise RuntimeError("Cloudflare control operation unavailable") from exc

    def create_parent(self, name: str, bucket: str, ttl: int) -> str:
        # Recover a committed create whose response was lost, without another parent.
        identity = re.fullmatch(r"(veda-[a-f0-9]{64}-g)([1-9][0-9]{0,9})", name)
        if identity is None:
            raise ValueError("private parent token name is invalid")
        inventory = []
        for page in range(1, 11):
            rows = self._call("GET", f"/tokens?page={page}&per_page=50")
            inventory.extend(rows)
            if len(rows) < 50:
                break
        else:
            raise RuntimeError("Cloudflare token inventory exceeds bounded lookup")
        matches = [row for row in inventory if row.get("name") == name]
        if len(matches) > 1:
            raise RuntimeError("duplicate registration parent tokens require reconciliation")
        # An interrupted issue can leave a parent ID absent from the CAS. Before
        # granting a NEW generation, reconcile only this registration's older
        # generations using the same bounded provider inventory and scope check.
        # Gather pages before deleting, so pagination cannot skip a shifted row.
        for row in inventory:
            older = re.fullmatch(re.escape(identity[1]) + r"([1-9][0-9]{0,9})",
                                 row.get("name") or "")
            if older is not None and int(older[1]) < int(identity[2]):
                self.revoke(row["id"], name=row["name"], bucket=bucket)
        if matches:
            return matches[0]["id"]
        groups = self._call("GET", "/tokens/permission_groups")
        permission = next(row["id"] for row in groups
                          if row["name"] == "Workers R2 Storage Bucket Item Write")
        result = self._call("POST", "/tokens", {
            "name": name,
            "expires_on": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(time.time() + ttl + 600)),
            "policies": [{"effect": "allow", "permission_groups": [{"id": permission}],
                          "resources": {f"com.cloudflare.edge.r2.bucket."
                                        f"{self.account}_{self.jurisdiction}_{bucket}": "*"}}],
        })
        return result["id"]

    def mint(self, parent: str, bucket: str, prefix: str, ttl: int) -> dict[str, str]:
        result = self._call("POST", "/r2/temp-access-credentials", {
            "bucket": bucket, "parentAccessKeyId": parent,
            "permission": "object-read-write", "prefixes": [prefix], "ttlSeconds": ttl,
        })
        return {"access_key_id": result["accessKeyId"],
                "secret_access_key": result["secretAccessKey"],
                "session_token": result["sessionToken"]}

    def revoke(self, parent: str, *, name: str, bucket: str) -> None:
        # The CAS ID alone is not authority to delete an arbitrary account token.
        if (not isinstance(parent, str) or re.fullmatch(r"[a-f0-9]{32}", parent) is None
                or re.fullmatch(r"veda-[a-f0-9]{64}-g[1-9][0-9]{0,9}", name) is None):
            raise ValueError("private parent token identity is invalid")
        record = self._call("GET", f"/tokens/{parent}")
        if record is None:  # A previously committed deletion lost its response.
            return
        policies = record.get("policies", [])
        resource = f"com.cloudflare.edge.r2.bucket.{self.account}_{self.jurisdiction}_{bucket}"
        if (record.get("id") != parent or record.get("name") != name
                or len(policies) != 1 or policies[0].get("effect") != "allow"
                or policies[0].get("resources") != {resource: "*"}):
            raise ValueError("private parent token does not belong to this registration/bucket")
        self._call("DELETE", f"/tokens/{parent}")


class MailboxControl:
    """Control-side operations only; responses contain NO plaintext credentials."""

    def __init__(self, *, tokens, signer, publish_mailbox, endpoint: str,
                 bucket: str, genesis_hash: str, objects: PrivateObjectStore):
        if getattr(signer, "crypto_type", None) != 0:
            raise ValueError("mailbox signer must be Ed25519")
        upload_registration_id(genesis_hash, "configuration")
        match = re.fullmatch(
            r"https://([a-f0-9]{32})(?:\.(eu|us|fedramp))?\.r2\.cloudflarestorage\.com/?",
            endpoint,
        )
        if (match is None or tokens.account != match[1]
                or tokens.jurisdiction != (match[2] or "default")
                or re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket) is None
                or bucket == "affine_video" or objects._prefix != "evaluation"
                or objects._bucket == "affine_video"):
            raise ValueError("control requires the configured private R2 account/bucket/CAS scope")
        self.tokens, self.signer, self.publish_mailbox = tokens, signer, publish_mailbox
        self.endpoint, self.bucket, self.genesis_hash = endpoint, bucket, genesis_hash
        self.objects = objects

    def handle(self, operation: str, record: dict) -> dict:
        if operation not in {"issue", "revoke", "publish"}:
            raise ValueError("unsupported mailbox control operation")
        upload_id = upload_registration_id(self.genesis_hash, record["hotkey"])
        if (record["upload_id"] != upload_id or type(record["generation"]) is not int
                or not 0 < record["generation"] < 1 << 32):
            raise ValueError("mailbox request does not identify a private registration")
        _, authoritative = _read_upload_record(self.objects, self.genesis_hash, upload_id)
        if authoritative is None or record != authoritative:
            raise ValueError("mailbox request differs from its original CAS")
        record = authoritative
        if operation == "revoke":
            if not (record.get("cleanup") == "pending"
                    or (record.get("file_sha256") and record.get("manifest_sha256"))):
                raise ValueError("private revoke does not match the original lifecycle phase")
            if record.get("parent_token_id"):
                self._revoke_parent(record)
            return {}
        if operation == "publish":
            if record["state"] != "UPLOADING" or not record.get("mailbox"):
                raise ValueError("private publish does not match the original lifecycle phase")
            mailbox = record["mailbox"]
            if mailbox["key"] != f"mailbox/v1/{upload_id}.bin":
                raise ValueError("mailbox publish escaped its registration")
            self.publish_mailbox(mailbox["key"], bytes.fromhex(mailbox["ciphertext"]))
            return {}
        if record["state"] != "ISSUING" or record.get("cleanup") == "pending":
            raise ValueError("private issue does not match the original lifecycle phase")
        from bittensor.wallet import Keypair

        if record.get("retire_parent"):
            self._revoke_parent(record)
        parent = (None if record.get("retire_parent") else record.get("parent_token_id"))
        parent = parent or self.tokens.create_parent(
            f"veda-{upload_id}-g{record['generation']}", self.bucket, CREDENTIAL_TTL,
        )
        credential = self.tokens.mint(parent, self.bucket,
                                      f"uploads/v1/{upload_id}/", CREDENTIAL_TTL)
        if (set(credential) != {"access_key_id", "secret_access_key", "session_token"}
                or not all(isinstance(v, str) and v for v in credential.values())):
            raise RuntimeError("temporary R2 credential response is incomplete")
        expires_at = int(time.time()) + CREDENTIAL_TTL
        envelope = {
            "format": "veda-mailbox-v1", "genesis_hash": self.genesis_hash,
            "hotkey": record["hotkey"], "upload_id": upload_id,
            "generation": record["generation"], "endpoint": self.endpoint,
            "activation_block_hash": record["activation_block_hash"],
            "bucket": self.bucket, "prefix": f"uploads/v1/{upload_id}/",
            "validator": self.signer.ss58_address, "expires_at": expires_at, **credential,
        }
        envelope["signature"] = self.signer.sign(envelope_signing_bytes(envelope)).hex()
        blob = Keypair.encrypt_for(record["hotkey"], canonical_json(envelope), crypto_type=0)
        return {"parent_token_id": parent, "retire_parent": False, "expires_at": expires_at,
                "mailbox": {"key": f"mailbox/v1/{upload_id}.bin", "ciphertext": blob.hex(),
                            "published": False}}

    def _revoke_parent(self, record: dict) -> None:
        generation = record["generation"] - int(bool(record.get("retire_parent")))
        self.tokens.revoke(record["parent_token_id"],
                           name=f"veda-{record['upload_id']}-g{generation}", bucket=self.bucket)


class OwnerMailboxControl:
    """Fixed RPC over the EXISTING attached owner's SSH stdin/stdout, not a listener.

    Control's attach wrapper answers only issue/revoke/publish. A lock keeps
    ADMISSION and bounded cleanup responses on the same single owner channel.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def handle(self, operation: str, record: dict) -> dict:
        import json
        import select
        import uuid

        with self._lock:
            identifier = uuid.uuid4().hex
            sys.stdout.write("VEDA_MAILBOX_REQUEST " + json.dumps({
                "id": identifier, "operation": operation, "record": record,
            }, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            deadline = time.monotonic() + 45
            while True:
                remaining = max(0, deadline - time.monotonic())
                if not select.select([sys.stdin], [], [], remaining)[0]:
                    raise RuntimeError("control owner mailbox response timed out")
                raw = sys.stdin.buffer.raw.readline(262145)
                if not raw or len(raw) > 262144:
                    raise RuntimeError("control owner mailbox response is invalid")
                try:
                    response = json.loads(raw)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError("control owner mailbox response is invalid") from exc
                if response.get("id") == identifier:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("control owner mailbox response timed out")
            if response.get("error"):
                raise RuntimeError("control owner mailbox operation unavailable")
            return response["result"]


def attach_evaluator(*, container: str, control: MailboxControl) -> int:
    """Production owner attaches ONE local container using the same stdio protocol."""
    import json
    import subprocess

    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", container) is None:
        raise ValueError("owner container name is invalid")
    environment = {name: value for name, value in os.environ.items()
                   if name in {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL"}}
    # Pin the local daemon, ignoring any inherited remote Docker context/host.
    docker = ["docker", "--host=unix:///var/run/docker.sock"]
    try:
        state = subprocess.run(
            [*docker, "inspect", "--format", "{{.State.Running}}", container],
            capture_output=True, text=True, check=True, timeout=15, env=environment,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("owner cannot inspect its local evaluator") from exc
    if state not in {"true", "false"}:
        raise RuntimeError("owner received an invalid local container state")
    # Reconnect never restarts a running writer. No shell, SSH, or new container.
    command = ([*docker, "attach", "--sig-proxy=false", container] if state == "true"
               else [*docker, "start", "--attach", "--interactive", container])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=environment,
    )
    try:
        while line := process.stdout.readline(262145):
            if len(line) > 262144:
                raise RuntimeError("owner attach response exceeds its bound")
            if not line.startswith("VEDA_MAILBOX_REQUEST "):
                print(line, end="", file=sys.stderr, flush=True)
                continue
            identifier = None
            try:
                request = json.loads(line.removeprefix("VEDA_MAILBOX_REQUEST "))
                identifier = request["id"]
                if re.fullmatch(r"[a-f0-9]{32}", identifier) is None:
                    raise ValueError("invalid owner request identifier")
                result = control.handle(request["operation"], request["record"])
                response = {"id": identifier, "result": result}
            except Exception:
                # Never print control API bodies, temporary credentials or seed material.
                response = {"id": identifier, "error": "MAILBOX_CONTROL_UNAVAILABLE"}
            process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
            process.stdin.flush()
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdin.close()
        process.stdout.close()


def attach_from_environment(*, container: str) -> int:
    """Production owner only; secrets/config paths never enter Docker argv/env."""
    import stat

    from bittensor.wallet import Keypair

    from validator.publication import S3Repository

    management_token = os.environ["CLOUDFLARE_CONTROL_API_TOKEN"]
    endpoint = os.environ["VEDA_UPLOAD_STAGING_STORAGE_URL"]
    match = re.fullmatch(
        r"https://([a-f0-9]{32})(?:\.(eu|us|fedramp))?\.r2\.cloudflarestorage\.com/?",
        endpoint,
    )
    if match is None:
        raise ValueError("control requires the R2 S3 API endpoint")
    # Opening with O_NOFOLLOW avoids copying/generating/replacing any wallet/key file.
    fd = os.open(os.environ["VEDA_UPLOAD_SIGNING_SEED_FILE"], os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_uid != os.geteuid()):
            raise ValueError("control mailbox seed must be an owner-only 0600 regular file")
        seed = stream.read(33)
    if len(seed) != 32:
        raise ValueError("control mailbox signing seed must contain exactly 32 bytes")
    signer = Keypair.create_from_seed(seed, crypto_type=0)
    if signer.ss58_address != os.environ["VEDA_UPLOAD_VALIDATOR_IDENTITY"]:
        raise ValueError("control mailbox signer differs from the configured public identity")
    client = _environment_client(
        _validated_endpoint(os.environ["STORAGE_URL"]),
        os.environ["ACCESS_KEY"], os.environ["SECRET_ACCESS_KEY"],
    )
    publisher = S3Repository(client, bucket="affine_video", prefix="tracks/main")
    authority = None
    try:
        authority = _environment_client(
            _validated_endpoint(os.environ["VEDA_UPLOAD_AUTHORITY_STORAGE_URL"]),
            os.environ["VEDA_UPLOAD_AUTHORITY_ACCESS_KEY"],
            os.environ["VEDA_UPLOAD_AUTHORITY_SECRET_ACCESS_KEY"],
        )
        return attach_evaluator(container=container, control=MailboxControl(
            tokens=CloudflareR2Tokens(match[1], management_token, match[2] or "default"),
            signer=signer, publish_mailbox=publisher.publish_mailbox, endpoint=endpoint,
            bucket=os.environ["VEDA_UPLOAD_STAGING_BUCKET"],
            genesis_hash=os.environ["VEDA_UPLOAD_GENESIS_HASH"],
            objects=PrivateObjectStore(authority,
                                       bucket=os.environ["VEDA_UPLOAD_AUTHORITY_BUCKET"],
                                       prefix="evaluation"),
        ))
    finally:
        if authority is not None:
            authority.close()
        publisher.close()
        client.close()


class PrivateUploads:
    """The SAME upload CAS, not a second participation database."""

    def __init__(
        self, *, objects: PrivateObjectStore, staging_client: Any,
        staging_bucket: str, staging_prefix: str, genesis_hash: str,
        store: ChallengeStore, max_bytes: int,
        registration: Callable[[str], tuple[int | None, int]] | None = None,
        control: MailboxControl | OwnerMailboxControl | None = None,
    ) -> None:
        if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", staging_bucket)
                or staging_bucket == "affine_video" or staging_prefix != "uploads/v1"
                or max_bytes <= 0):
            raise ValueError("private R2 staging scope or limit is invalid")
        upload_registration_id(genesis_hash, "configuration")
        self.objects, self.staging = objects, staging_client
        self.bucket, self.prefix = staging_bucket, staging_prefix
        self.genesis_hash, self.store, self.max_bytes = genesis_hash, store, max_bytes
        self.registration, self.control = registration, control
        self._gc_after = ""
        self._issue_after = ""
        self.current_champion: (
            Callable[[ChallengeStore], PrivateModelSubmission | None] | None
        ) = None
        self.submission_timestamp: Callable[[int], int] | None = None

    @classmethod
    def from_environment(cls, *, store: ChallengeStore, max_bytes: int) -> PrivateUploads | None:
        names = (
            "GENESIS_HASH", "AUTHORITY_BUCKET", "STAGING_BUCKET",
            "AUTHORITY_STORAGE_URL", "AUTHORITY_ACCESS_KEY", "AUTHORITY_SECRET_ACCESS_KEY",
            "STAGING_STORAGE_URL", "STAGING_ACCESS_KEY", "STAGING_SECRET_ACCESS_KEY",
        )
        values = tuple(os.environ.get(f"VEDA_UPLOAD_{name}") for name in names)
        if not any(values):
            return None
        if not all(values):
            raise ValueError("all private R2 and authority settings are required")
        (genesis, bucket, staging_bucket, authority_url, authority_key, authority_secret,
         staging_url, staging_key, staging_secret) = values
        if bucket == "affine_video":
            raise ValueError("upload authority cannot use the public bucket")
        if not re.fullmatch(
            r"https://[a-f0-9]{32}(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com/?",
            staging_url,
        ):
            raise ValueError("upload staging requires an R2 S3 API endpoint")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", staging_bucket):
            raise ValueError("R2 staging bucket name is invalid")
        return cls(
            objects=PrivateObjectStore(
                _environment_client(_validated_endpoint(authority_url), authority_key,
                                    authority_secret), bucket=bucket, prefix="evaluation",
            ),
            staging_client=_environment_client(
                staging_url, staging_key, staging_secret, region_name="auto",
            ), staging_bucket=staging_bucket, staging_prefix="uploads/v1",
            genesis_hash=genesis, store=store, max_bytes=max_bytes,
            control=OwnerMailboxControl(),
        )

    @staticmethod
    def _record_name(upload_id: str) -> str:
        return f"uploads/{upload_id}.json"

    def _read(self, upload_id: str):
        return _read_upload_record(self.objects, self.genesis_hash, upload_id)

    def _replace(self, stored, record: dict):
        name, payload = self._record_name(record["upload_id"]), canonical_json(record)
        return (self.objects.create_plan(name, payload) if stored is None
                else self.objects.replace_plan(name, payload, etag=stored.etag))

    def observe_activation(
        self, activation: UploadActivation, *, hotkey: str, block: int,
        extrinsic_index: int, event_index: int, block_hash: str,
    ) -> None:
        """Record the signed chain request before the original scanner advances.

        This is NOT a submission/attempt. Credential issuance waits for the
        original finalized DB qualification gate, including concurrent HF.
        """
        if not ed25519_verify(hotkey, activation_message(self.genesis_hash, hotkey),
                              activation.signature):
            raise ValueError("activation requires an Ed25519 hotkey signature")
        upload_id = upload_registration_id(self.genesis_hash, hotkey)
        block_hash = block_hash.removeprefix("0x")
        if re.fullmatch(r"[a-f0-9]{64}", block_hash) is None:
            raise ValueError("private activation requires its finalized block hash")
        stored, record = self._read(upload_id)
        position = [block, extrinsic_index, event_index]
        previous_position = ((record or {}).get("activation_request") or {}).get(
            "position", (record or {}).get("activation_position", []),
        )
        if record is not None and position <= previous_position:
            return
        if record is not None:
            # Do NOT put an existing ready into ISSUING merely because another
            # request appeared. Qualification happens first; original ADMISSION
            # must remain usable if reactivation is refused or temporarily fails.
            self._replace(stored, {**record, "activation_request": {
                "position": position, "block_hash": block_hash,
            }})
            return
        self._replace(stored, {
            **(record or {}), "format": "veda-upload-v2", "hotkey": hotkey,
            "upload_id": upload_id, "activation_position": position,
            "activation_block_hash": block_hash,
            "generation": (record or {}).get("generation", 0) + 1,
            "state": "ISSUING", "expires_at": (record or {}).get("expires_at", 0),
            "parent_token_id": (record or {}).get("parent_token_id"),
            "retire_parent": bool((record or {}).get("parent_token_id")),
            "mailbox": None,
        })

    def issue_pending(self, *, now: int) -> bool:
        """One control operation; live qualification remains the original SQLite gate."""
        if self.control is None or self.registration is None:
            raise RuntimeError("private mailbox control owner is not configured")
        names = self.objects.list_names("uploads/", limit=16, after=self._issue_after or None)
        if not names and self._issue_after:
            self._issue_after = ""
            names = self.objects.list_names("uploads/", limit=16)
        for name in names:
            self._issue_after = name
            if name.count("/") != 1 or not name.endswith(".json"):
                continue
            upload_id = name[8:-5]
            stored, record = self._read(upload_id)
            if record.get("cleanup") == "pending":
                continue
            unpublished = (record["state"] == "UPLOADING" and record.get("mailbox")
                           and not record["mailbox"].get("published"))
            if not unpublished and record["state"] != "ISSUING" and not record.get(
                "activation_request",
            ):
                continue
            retry = record.get("issue_retry", {})
            if retry.get("retry_after", 0) > now:
                continue
            attempts = retry.get("attempts", 0) + 1
            # Persist the cooldown BEFORE external I/O, including a lost response
            # or owner crash. Only this original upload record owns retry state.
            record = {**record, "issue_retry": {
                "attempts": attempts,
                "retry_after": now + min(300, 30 * 2 ** min(attempts - 1, 4)),
                "error": retry.get("error"),
            }}
            stored = self._replace(stored, record)
            try:
                self._issue_record(stored, record, unpublished=bool(unpublished))
            except ValueError:
                # An original eligibility refusal is not a Cloudflare failure.
                current, latest = self._read(upload_id)
                self._replace(current, {k: v for k, v in latest.items() if k != "issue_retry"})
                raise
            except Exception:
                current, latest = self._read(upload_id)
                if (latest["state"] == "UPLOADING" and not latest.get("activation_request")
                        and (latest.get("mailbox") or {}).get("published")):
                    # The final CAS committed but its response was lost. Durable
                    # completion, not a second mint/retry, decides success.
                    return True
                self._replace(current, {**latest, "issue_retry": {
                    **record["issue_retry"], "error": "PRIVATE_UPLOAD_ISSUANCE_FAILED",
                }})
                raise
            return True
        return False

    def maintenance_status(self) -> dict[str, str | None]:
        """Rebuild public errors from original small CAS records, also after restart.

        A different mailbox's success cannot erase an outstanding failure. This
        is a read-only projection, not another failure list/queue or authority.
        """
        status = {"private_upload_error": None, "private_upload_cleanup_error": None}
        for name in self.objects.list_names("uploads/"):
            if name.count("/") != 1 or not name.endswith(".json"):
                continue
            _, record = self._read(name[8:-5])
            if (record.get("issue_retry") or {}).get("error"):
                status["private_upload_error"] = "PRIVATE_UPLOAD_ISSUANCE_FAILED"
            if record.get("cleanup") == "pending":
                status["private_upload_cleanup_error"] = "PRIVATE_UPLOAD_CLEANUP_FAILED"
        return status

    def _issue_record(self, stored, record: dict, *, unpublished: bool) -> None:
        upload_id = record["upload_id"]
        if unpublished:
            self.republish_mailbox(upload_id)
            return
        if record.get("model_reference") is not None:
            if record.get("activation_request") is not None:
                self._replace(stored, {k: v for k, v in record.items()
                                       if k != "activation_request"})
            raise ValueError("admitted R2 body cannot receive new miner write permissions")
        pending = record.get("activation_request")
        rows = [row for row in self.store.public_snapshot()["challenges"]
                if row["hotkey"] == record["hotkey"]]
        exempt = (any(row["state"] == "SYSTEM_NOT_ADMITTED" for row in rows)
                  and not any(row["state"] != "SYSTEM_NOT_ADMITTED" for row in rows))
        reference = (PrivateModelSubmission(upload_id, record["file_sha256"],
                                            record["manifest_sha256"])
                     if record.get("file_sha256") and not exempt else None)
        if reference is None:
            # Read the ORIGINAL formal submission, not a new eligibility ledger.
            for row in rows:
                private = private_model_source(row)
                if (private is not None and private.upload_id == upload_id
                        and row["state"] != "SYSTEM_NOT_ADMITTED"):
                    reference = private
                    break
        uid, finalized = self.registration(record["hotkey"])
        try:
            self.store.require_upload_eligible(record["hotkey"], uid=uid,
                                               finalized_block=finalized, reference=reference)
        except ValueError:
            if pending is not None:
                self._replace(stored, {k: v for k, v in record.items()
                                       if k != "activation_request"})
            raise
        if pending is not None:
            cleared = {"activation_request", "cleanup"}
            if exempt:
                cleared |= {"file_sha256", "manifest_sha256", "manifest", "size_bytes"}
            record = {**{k: v for k, v in record.items() if k not in cleared},
                      "activation_position": pending["position"],
                      "activation_block_hash": pending["block_hash"],
                      "generation": record["generation"] + 1, "state": "ISSUING",
                      "retire_parent": bool(record["parent_token_id"]), "mailbox": None}
            stored = self._replace(stored, record)
        if reference is not None and record.get("file_sha256") is None:
            record = {**record, "file_sha256": reference.file_sha256,
                      "manifest_sha256": reference.manifest_sha256}
            stored = self._replace(stored, record)
        # Management, credentials, signing and encryption all stay on control.
        result = self.control.handle("issue", record)
        if set(result) != {"parent_token_id", "retire_parent", "expires_at", "mailbox"}:
            raise RuntimeError("private control response has an invalid shape")
        self._replace(stored, {**record, **result, "state": "UPLOADING"})
        self.republish_mailbox(upload_id)

    def republish_mailbox(self, upload_id: str) -> None:
        stored, record = self._read(upload_id)
        if record is None or record["state"] != "UPLOADING" or self.control is None:
            raise ValueError("private mailbox is not available")
        self.control.handle("publish", record)
        self._replace(stored, {**{k: v for k, v in record.items() if k != "issue_retry"},
                               "mailbox": {**record["mailbox"], "published": True}})

    def _seal_ready(self, reference: PrivateModelSubmission, hotkey: str):
        try:
            stored, record = self._read(reference.upload_id)
        except Exception as exc:
            raise RuntimeError("private upload authority is unavailable") from exc
        if record is None or record["hotkey"] != hotkey or reference.manifest_sha256 is None:
            raise ValueError("ready does not identify an original private registration")
        if ((record.get("cleanup") == "pending" and not record.get("champion_archive"))
                or record["state"] == "ISSUING"):
            raise RuntimeError("private upload lifecycle is not ready for admission")
        if record.get("file_sha256") is not None and (
            record["file_sha256"] != reference.file_sha256
            or record["manifest_sha256"] != reference.manifest_sha256
        ):
            raise ValueError("ready cannot replace its original model or manifest")
        if record.get("file_sha256") is None:
            record = {**record, "file_sha256": reference.file_sha256,
                      "manifest_sha256": reference.manifest_sha256}
            stored = self._replace(stored, record)
        if record["parent_token_id"]:
            if self.control is None:
                raise RuntimeError("ready awaits control-owner write revocation")
            try:
                self.control.handle("revoke", record)
            except Exception as exc:
                raise RuntimeError("private ready write revocation is unconfirmed") from exc
            record = {**record, "parent_token_id": None}
            stored = self._replace(stored, record)
        if record["state"] == "SEALED":
            return record
        try:
            response = self.staging.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}/{reference.upload_id}/manifest.json",
            )
            body = response["Body"]
            try:
                raw = body.read(MAX_MANIFEST_BYTES + 1)
            finally:
                body.close()
            if (len(raw) <= MAX_MANIFEST_BYTES
                    and response.get("ContentLength", len(raw)) != len(raw)):
                raise RuntimeError("private manifest transport was incomplete")
        except Exception as exc:
            raise RuntimeError("private manifest transport is unavailable") from exc
        if (len(raw) > MAX_MANIFEST_BYTES
                or hashlib.sha256(raw).hexdigest() != reference.manifest_sha256):
            raise ValueError("ready manifest failed its original SHA")
        document = decode_canonical_json(raw)
        if (not isinstance(document, dict)
                or set(document) != {"format", "genesis_hash", "hotkey", "upload_id", "manifest",
                                    "signature"}
                or document["format"] != "veda-upload-manifest-v1"
                or document["genesis_hash"] != self.genesis_hash or document["hotkey"] != hotkey
                or document["upload_id"] != reference.upload_id
                or not ed25519_verify(hotkey, manifest_signing_bytes(document),
                                      bytes.fromhex(document["signature"]))):
            raise ValueError("private manifest identity or Ed25519 signature is invalid")
        manifest = ChunkManifest.decode(canonical_json(document["manifest"]))
        if manifest.sha256 != reference.file_sha256 or manifest.size > self.max_bytes:
            raise ValueError("private ready model identity or length differs")
        record = {**record, "state": "SEALED", "manifest": document["manifest"],
                  "file_sha256": reference.file_sha256,
                  "manifest_sha256": reference.manifest_sha256, "size_bytes": manifest.size}
        self._replace(stored, record)
        return record

    def download(
        self, reference: PrivateModelSubmission, *, hotkey: str, cache_dir: str | Path,
        max_bytes: int, submission_block: int | None = None,
    ) -> DownloadedModel:
        """Revoke first, verify bytes, then original ADMISSION/schema and small binding."""
        try:
            record = self._seal_ready(reference, hotkey)
            if submission_block is not None:
                if self.submission_timestamp is None:
                    raise RuntimeError("original submission clock is unavailable")
                self.bind_submission(
                    reference, hotkey=hotkey, submission_block=submission_block,
                    submission_timestamp_ms=self.submission_timestamp(submission_block),
                )
        except ValueError as exc:
            raise ModelDownloadError("PRIVATE_UPLOAD_BINDING_MISMATCH",
                                     "private ready binding is invalid") from exc
        except Exception as exc:
            raise ModelDownloadError("PRIVATE_UPLOAD_UNAVAILABLE",
                                     "private source is temporarily unavailable",
                                     temporary=True) from exc
        if record["size_bytes"] > max_bytes:
            raise ModelDownloadError("MODEL_TOO_LARGE", "private source exceeds byte limit")
        manifest = ChunkManifest.decode(canonical_json(record["manifest"]))
        target = Path(cache_dir) / f"{reference.file_sha256}.safetensors"
        prefix = record.get("champion_archive") or f"{self.prefix}/{reference.upload_id}/"

        def open_part(name: str):
            return self.staging.get_object(
                Bucket=self.bucket, Key=prefix + name,
            )["Body"]

        temporary = None
        try:
            if (target.is_file() and target.stat().st_size == manifest.size
                    and model_file_sha256(target) == reference.file_sha256):
                return DownloadedModel(target, reference.file_sha256, manifest.size, None, None)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".part", delete=False,
            ) as out:
                temporary = Path(out.name)
                verify_chunked_content(manifest, open_part, write=out.write)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, target)
            temporary = None
        except ValueError as exc:
            raise ModelDownloadError("PRIVATE_UPLOAD_HASH_MISMATCH",
                                     "private source failed byte verification",
                                     temporary=True) from exc
        except Exception as exc:
            raise ModelDownloadError("PRIVATE_UPLOAD_UNAVAILABLE", "private source unavailable",
                                     temporary=True) from exc
        finally:
            active_error = sys.exc_info()[0] is not None
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    if not active_error:
                        raise ModelDownloadError("PRIVATE_UPLOAD_UNAVAILABLE",
                                                 "private source cleanup failed",
                                                 temporary=True) from exc
        return DownloadedModel(target, reference.file_sha256, manifest.size, None, None)

    def bind_submission(self, reference, *, hotkey, submission_block, submission_timestamp_ms):
        """Retain even rejected formal bytes for 30 days from the original chain S."""
        self._seal_ready(reference, hotkey)
        formal = {"submission_block": submission_block,
                  "submission_timestamp_ms": submission_timestamp_ms,
                  "retain_until_ms": submission_timestamp_ms + MODEL_RETENTION_MS}
        if any(type(value) is not int or value <= 0 for value in formal.values()):
            raise ValueError("original submission retention clock is invalid")
        for _ in range(16):
            stored, record = self._read(reference.upload_id)
            if (record.get("cleanup") in {"pending", "done"}
                    or record["state"] != "SEALED" or record["parent_token_id"] is not None):
                raise RuntimeError("formal model retention awaits original sealed fence")
            previous = record.get("formal_submission")
            if previous is not None:
                if previous != formal:
                    raise ValueError("formal model cannot change original S or retention deadline")
                return
            try:
                self._replace(stored, {**record, "formal_submission": formal})
                return
            except PreconditionFailed:
                continue
        raise RuntimeError("formal model retention CAS contention")

    def retain_model(
        self, reference: PrivateModelSubmission, *, hotkey: str, model_digest: str,
        size_bytes: int, submission_block: int, submission_timestamp_ms: int,
    ) -> dict:
        """Pin the verified body in the existing upload CAS before admission commits."""
        record = self._seal_ready(reference, hotkey)
        self.bind_submission(reference, hotkey=hotkey, submission_block=submission_block,
                             submission_timestamp_ms=submission_timestamp_ms)
        model = decode_model_reference(canonical_json({
            "format": MODEL_REFERENCE_FORMAT, "hotkey": hotkey,
            "private_upload": {"upload_id": reference.upload_id,
                               "file_sha256": reference.file_sha256,
                               "manifest_sha256": reference.manifest_sha256},
            "file_sha256": reference.file_sha256, "model_digest": model_digest,
            "size_bytes": size_bytes, "submission_block": submission_block,
            "submission_timestamp_ms": submission_timestamp_ms,
            "retain_until_ms": submission_timestamp_ms + MODEL_RETENTION_MS,
        }))
        if size_bytes != record["size_bytes"]:
            raise ValueError("retained model length differs from sealed bytes")
        for _ in range(16):
            stored, record = self._read(reference.upload_id)
            if record.get("cleanup") in {"pending", "done"}:
                raise RuntimeError("private model body is being reclaimed")
            if (record["state"] != "SEALED" or record["parent_token_id"] is not None
                    or record.get("file_sha256") != reference.file_sha256
                    or record.get("manifest_sha256") != reference.manifest_sha256):
                raise RuntimeError("private model retention awaits original sealed write fence")
            if record.get("model_reference") is not None:
                if record["model_reference"] != model:
                    raise ValueError("retained model cannot change its original submission")
                return model
            try:
                self._replace(stored, {**record, "model_reference": model})
                return model
            except PreconditionFailed:
                continue
        raise RuntimeError("private model retention CAS contention")

    def archive_champion(self, model: dict) -> None:
        """Only a resolved private winner calls this before public payload/match staging."""
        model = decode_model_reference(canonical_json(model))
        reference = private_model_source(model)
        record = self._seal_ready(reference, model["hotkey"])
        if record.get("model_reference") != model:
            raise ValueError("champion differs from its admitted model")
        manifest = ChunkManifest.decode(canonical_json(record["manifest"]))
        prefix = f"champions/v1/{reference.file_sha256}/"
        if record.get("champion_archive") == prefix:
            existing = self.staging.get_object(Bucket=self.bucket, Key=prefix + "manifest.json")
            with existing["Body"] as body:
                if body.read(MAX_MANIFEST_BYTES + 1) != manifest.encode():
                    raise ValueError("private champion manifest changed")
            return
        try:
            with self.staging.get_object(Bucket=self.bucket, Key=prefix + "manifest.json")[
                "Body"
            ] as body:
                if body.read(MAX_MANIFEST_BYTES + 1) != manifest.encode():
                    raise ValueError("private champion manifest conflicts")
        except Exception as exc:
            from validator.s3 import is_missing

            if not isinstance(exc, KeyError) and not is_missing(exc):
                raise
            for part in manifest.parts:
                self.staging.copy_object(
                    Bucket=self.bucket, Key=prefix + part.name,
                    CopySource={"Bucket": self.bucket,
                                "Key": f"{self.prefix}/{reference.upload_id}/{part.name}"},
                )
        verify_chunked_content(
            manifest, lambda name: self.staging.get_object(
                Bucket=self.bucket, Key=prefix + name,
            )["Body"],
        )
        try:
            self.staging.put_object(Bucket=self.bucket, Key=prefix + "manifest.json",
                                    Body=manifest.encode(), IfNoneMatch="*")
        except Exception as exc:
            if not is_precondition_failed(exc):
                raise
            with self.staging.get_object(Bucket=self.bucket, Key=prefix + "manifest.json")[
                "Body"
            ] as body:
                if body.read(MAX_MANIFEST_BYTES + 1) != manifest.encode():
                    raise ValueError("private champion manifest conflicts") from exc
        for _ in range(16):
            stored, record = self._read(reference.upload_id)
            if record.get("model_reference") != model:
                raise ValueError("champion model binding changed")
            if record.get("champion_archive") == prefix:
                return
            try:
                self._replace(stored, {**record, "champion_archive": prefix})
                return
            except PreconditionFailed:
                continue
        raise RuntimeError("private champion CAS contention")

    def restore_model_reference(self, model: dict, *, cache_dir: str | Path) -> Path:
        model = decode_model_reference(canonical_json(model))
        reference = private_model_source(model)
        _, record = self._read(reference.upload_id)
        if record is None or record.get("model_reference") != model:
            raise RuntimeError("R2 model reference differs from original retention")
        return self.download(reference, hotkey=model["hotkey"], cache_dir=cache_dir,
                             max_bytes=model["size_bytes"]).path

    def reclaim_expired(
        self, *, store: ChallengeStore, finalized_block: int,
        finalized_timestamp: int, now: int,
    ) -> int:
        """One page of one prefix per call; scan, reference and cap expiry still apply."""
        names = self.objects.list_names("uploads/", limit=16, after=self._gc_after or None)
        if not names:
            self._gc_after = ""
            names = self.objects.list_names("uploads/", limit=16)
        for name in names:
            self._gc_after = name
            if name.count("/") != 1 or not name.endswith(".json"):
                continue
            stored, record = self._read(name[8:-5])
            if (record.get("cleanup") == "done" or record["state"] == "ISSUING"
                    or record["expires_at"] >= min(now, finalized_timestamp)):
                continue
            references = [private for row in store.public_snapshot()["challenges"]
                          if row["hotkey"] == record["hotkey"]
                          and (private := row.get("private_upload")) is not None
                          and private["upload_id"] == record["upload_id"]]
            model = record.get("model_reference")
            formal = record.get("formal_submission")
            if (formal is not None
                    and min(now, finalized_timestamp) * 1000 < formal["retain_until_ms"]):
                continue
            if model is not None:
                if (min(now, finalized_timestamp) * 1000 < model["retain_until_ms"]
                        or not references or self.current_champion is None):
                    continue
                champion = self.current_champion(store)
                if champion is not None and champion.upload_id == record["upload_id"]:
                    continue
            if any(store.upload_is_referenced(
                record["hotkey"], reference=PrivateModelSubmission(**private),
                finalized_block=finalized_block,
                include_admitted=model is not None,
            ) for private in references):
                continue
            if not references:
                store.upload_is_referenced(
                    record["hotkey"],
                    reference=PrivateModelSubmission(record["upload_id"], "00" * 32),
                    finalized_block=finalized_block,
                )
            if record.get("cleanup") != "pending":
                record = {**record, "cleanup": "pending"}
                stored = self._replace(stored, record)
            if record["parent_token_id"]:
                if self.control is None:
                    continue
                self.control.handle("revoke", record)
                record = {**record, "parent_token_id": None}
                stored = self._replace(stored, record)
            prefix = f"{self.prefix}/{record['upload_id']}/"
            response = self.staging.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1000)
            keys = [row["Key"] for row in response.get("Contents", [])]
            if any(not key.startswith(prefix) for key in keys):
                raise ValueError("private cleanup escaped registration prefix")
            for key in keys:
                self.staging.delete_object(Bucket=self.bucket, Key=key)
            # Re-list the remaining first page next round, including after a crash
            # partway through deletion. The original CAS stays pending until empty.
            if not response.get("IsTruncated"):
                self._replace(stored, {**record, "cleanup": "done"})
            return len(keys)
        return 0
