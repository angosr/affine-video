"""Canonical model submission payload encoding.

The miner package is a local, one-shot submission tool. This module contains
the chain payload contract without importing the Bittensor SDK, so the exact
bytes can be tested independently of a network connection.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

MAGIC = b"VDA1"
PRIVATE_MAGIC = b"VDS1"
READY_MAGIC = b"VDR1"
ACTIVATE_MAGIC = b"VDU1"
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
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("upload_id", "file_sha256", "manifest_sha256"):
            value = getattr(self, name)
            if name == "manifest_sha256" and value is None:
                continue
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise SubmissionFormatError(f"{name} must be a full lowercase SHA-256 value")

    def encode(self) -> bytes:
        prefix = READY_MAGIC if self.manifest_sha256 is not None else PRIVATE_MAGIC
        return (prefix + bytes.fromhex(self.upload_id) + bytes.fromhex(self.file_sha256)
                + (bytes.fromhex(self.manifest_sha256) if self.manifest_sha256 else b""))

    @classmethod
    def decode(cls, payload: bytes) -> PrivateModelSubmission:
        if len(payload) == 100 and payload.startswith(READY_MAGIC):
            return cls(payload[4:36].hex(), payload[36:68].hex(), payload[68:].hex())
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
        if private.manifest_sha256 is not None:
            result["private_upload"]["manifest_sha256"] = private.manifest_sha256
    return result


def private_model_source(value: Mapping[str, object]) -> PrivateModelSubmission | None:
    """Decode the same optional private identity in existing public/archive documents."""
    if "private_upload" not in value:
        return None
    private = value["private_upload"]
    if (not isinstance(private, Mapping) or set(private) not in (
        {"upload_id", "file_sha256"}, {"upload_id", "file_sha256", "manifest_sha256"},
    )
            or value.get("repo_id") is not None or value.get("revision") is not None):
        raise SubmissionFormatError("private source identity conflicts with HF fields")
    return PrivateModelSubmission(
        private["upload_id"], private["file_sha256"], private.get("manifest_sha256"),
    )


def upload_registration_id(genesis_hash: str, hotkey: str) -> str:
    """One private registration per SN120 hotkey and chain, not a second opportunity."""
    if re.fullmatch(r"[0-9a-f]{64}", genesis_hash) is None or not hotkey:
        raise ValueError("private registration requires genesis hash and hotkey")
    return hashlib.sha256(
        b"veda-registration-v1\0" + bytes.fromhex(genesis_hash) + b"120\0" + hotkey.encode(),
    ).hexdigest()


def activation_message(genesis_hash: str, hotkey: str) -> bytes:
    return b"veda-activate-v1\0" + bytes.fromhex(upload_registration_id(genesis_hash, hotkey))


@dataclass(frozen=True, slots=True)
class UploadActivation:
    """Ed25519 proof carried by the registered hotkey; NOT a model submission."""

    signature: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.signature, bytes) or len(self.signature) != 64:
            raise SubmissionFormatError("activation requires a 64-byte Ed25519 signature")

    def encode(self) -> bytes:
        return ACTIVATE_MAGIC + self.signature

    @classmethod
    def decode(cls, payload: bytes) -> UploadActivation:
        if len(payload) != 68 or not payload.startswith(ACTIVATE_MAGIC):
            raise SubmissionFormatError("private activation must be exactly 68 VDU1 bytes")
        return cls(payload[4:])
