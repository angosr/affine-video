"""Canonical model submission payload encoding.

The miner package is a local, one-shot submission tool. This module contains
the chain payload contract without importing the Bittensor SDK, so the exact
bytes can be tested independently of a network connection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

MAGIC = b"VDA1"
PRIVATE_MAGIC = b"VDS1"
REVISION_BYTES = 20
REVISION_HEX_LENGTH = REVISION_BYTES * 2
MAX_REPO_ID_BYTES = 96
MAX_PAYLOAD_BYTES = len(MAGIC) + 1 + MAX_REPO_ID_BYTES + REVISION_BYTES

_REPO_PART = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


class SubmissionFormatError(ValueError):
    """Raised when a model submission cannot be represented canonically."""


def _validate_repo_id(repo_id: str) -> bytes:
    if repo_id.count("/") != 1:
        raise SubmissionFormatError("repo_id must use the owner/repository form")

    owner, repository = repo_id.split("/", 1)
    if not _REPO_PART.fullmatch(owner) or not _REPO_PART.fullmatch(repository):
        raise SubmissionFormatError("repo_id contains unsupported characters")

    encoded = repo_id.encode("ascii")
    if len(encoded) > MAX_REPO_ID_BYTES:
        raise SubmissionFormatError(
            f"repo_id exceeds the {MAX_REPO_ID_BYTES}-byte competition limit"
        )
    return encoded


def _validate_revision(revision: str) -> str:
    if not _REVISION.fullmatch(revision):
        raise SubmissionFormatError("revision must be a complete 40-character commit SHA")
    return revision.lower()


@dataclass(frozen=True, slots=True)
class ModelSubmission:
    """The two participant-provided fields stored in one chain commitment."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        _validate_repo_id(self.repo_id)
        object.__setattr__(self, "revision", _validate_revision(self.revision))

    def encode(self) -> bytes:
        repo = _validate_repo_id(self.repo_id)
        revision = bytes.fromhex(self.revision)
        payload = MAGIC + bytes((len(repo),)) + repo + revision
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise AssertionError("validated submission exceeded the payload bound")
        return payload

    @classmethod
    def decode(cls, payload: bytes) -> ModelSubmission:
        minimum = len(MAGIC) + 1 + REVISION_BYTES
        if len(payload) < minimum:
            raise SubmissionFormatError("payload is truncated")
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise SubmissionFormatError("payload exceeds the competition limit")
        if payload[: len(MAGIC)] != MAGIC:
            raise SubmissionFormatError("payload has an unknown magic value")

        repo_length = payload[len(MAGIC)]
        expected = minimum + repo_length
        if len(payload) != expected:
            raise SubmissionFormatError("payload length does not match repo_id length")

        repo_start = len(MAGIC) + 1
        repo_end = repo_start + repo_length
        try:
            repo_id = payload[repo_start:repo_end].decode("ascii")
        except UnicodeDecodeError as exc:
            raise SubmissionFormatError("repo_id is not ASCII") from exc

        revision = payload[repo_end:].hex()
        return cls(repo_id=repo_id, revision=revision)


@dataclass(frozen=True, slots=True)
class PrivateModelSubmission:
    """A sealed private upload, never a URL, credential, or HF revision."""

    upload_id: str
    file_sha256: str

    def __post_init__(self) -> None:
        for name in ("upload_id", "file_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise SubmissionFormatError(f"{name} must be a full lowercase SHA-256 value")

    def encode(self) -> bytes:
        return PRIVATE_MAGIC + bytes.fromhex(self.upload_id) + bytes.fromhex(self.file_sha256)

    @classmethod
    def decode(cls, payload: bytes) -> PrivateModelSubmission:
        if len(payload) != 68 or not payload.startswith(PRIVATE_MAGIC):
            raise SubmissionFormatError("private submission must be exactly 68 VDS1 bytes")
        return cls(payload[4:36].hex(), payload[36:].hex())


def model_source_fields(
    repo_id: str | None, revision: str | None,
    private: PrivateModelSubmission | None = None,
) -> dict[str, object]:
    """Keep HF documents byte-compatible; private identities never invent HF fields."""
    result: dict[str, object] = {"repo_id": repo_id, "revision": revision}
    if private is not None:
        if repo_id is not None or revision is not None:
            raise ValueError("private sources must not masquerade as HF repositories")
        result["private_upload"] = {
            "upload_id": private.upload_id, "file_sha256": private.file_sha256,
        }
    return result


def private_model_source(value: Mapping[str, object]) -> PrivateModelSubmission | None:
    """Decode the same optional private identity in existing public/archive documents."""
    if "private_upload" not in value:
        return None
    private = value["private_upload"]
    if (not isinstance(private, Mapping) or set(private) != {"upload_id", "file_sha256"}
            or value.get("repo_id") is not None or value.get("revision") is not None):
        raise SubmissionFormatError("private source identity conflicts with HF fields")
    return PrivateModelSubmission(private["upload_id"], private["file_sha256"])


@dataclass(frozen=True, slots=True)
class UploadIntent:
    """Short-lived hotkey-signed intent, shared by CLI and upload admission.

    Retrying the exact intent is idempotent, not another competition attempt.
    The audience is a configured HTTPS authority; genesis separates forks.
    """

    audience: str
    genesis_hash: str
    hotkey: str
    netuid: int
    action: str
    upload_id: str
    file_sha256: str
    size_bytes: int
    manifest_sha256: str
    expires_at: int

    def __post_init__(self) -> None:
        from urllib.parse import urlparse

        PrivateModelSubmission(self.upload_id, self.file_sha256)
        PrivateModelSubmission(self.upload_id, self.manifest_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", self.genesis_hash) is None:
            raise ValueError("upload genesis hash must be a full lowercase SHA-256 value")
        url = urlparse(self.audience)
        if (url.scheme != "https" or not url.netloc or url.username is not None
                or url.password is not None or url.query or url.fragment):
            raise ValueError("upload audience must be a credential-free HTTPS endpoint")
        if (not self.hotkey or len(self.hotkey) > 64 or type(self.netuid) is not int
                or self.netuid != 120 or self.action not in {"upload", "seal"}
                or type(self.size_bytes) is not int or self.size_bytes <= 0
                or type(self.expires_at) is not int or self.expires_at <= 0):
            raise ValueError("upload intent fields are invalid")

    def signing_message(self) -> bytes:
        from dataclasses import asdict

        return b"veda-private-upload-v1\0" + json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii")
