"""Canonical model submission payload encoding.

The miner package is a local, one-shot submission tool. This module contains
the chain payload contract without importing the Bittensor SDK, so the exact
bytes can be tested independently of a network connection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAGIC = b"VDA1"
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
