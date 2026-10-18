"""Canonical manifests for files stored as bounded object-storage chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_BYTES = 40 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PART = re.compile(r"^part-[0-9]{5}$")


@dataclass(frozen=True, slots=True)
class ChunkPart:
    name: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if _PART.fullmatch(self.name) is None or _DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("chunk identity is invalid")
        if not 0 < self.size <= CHUNK_BYTES:
            raise ValueError("chunk size is invalid")

    def document(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class ChunkManifest:
    sha256: str
    size: int
    parts: tuple[ChunkPart, ...]

    def __post_init__(self) -> None:
        if _DIGEST.fullmatch(self.sha256) is None or self.size <= 0 or not self.parts:
            raise ValueError("chunked file identity is invalid")
        if tuple(part.name for part in self.parts) != tuple(
            f"part-{index:05d}" for index in range(len(self.parts))
        ):
            raise ValueError("chunk sequence is not contiguous")
        if any(part.size != CHUNK_BYTES for part in self.parts[:-1]):
            raise ValueError("non-final chunks must use the canonical chunk size")
        if sum(part.size for part in self.parts) != self.size:
            raise ValueError("chunk sizes do not equal the complete file size")

    def encode(self) -> bytes:
        return json.dumps(
            {
                "format": "veda-chunked-object-v1",
                "parts": [part.document() for part in self.parts],
                "sha256": self.sha256,
                "size": self.size,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()

    @classmethod
    def decode(cls, payload: bytes) -> ChunkManifest:
        try:
            value = json.loads(payload, object_pairs_hook=_unique)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("chunk manifest is not valid JSON") from exc
        if not isinstance(value, dict) or set(value) != {"format", "parts", "sha256", "size"}:
            raise ValueError("chunk manifest fields are invalid")
        if value["format"] != "veda-chunked-object-v1" or not isinstance(value["parts"], list):
            raise ValueError("chunk manifest format is invalid")
        parts = []
        for raw in value["parts"]:
            if not isinstance(raw, dict) or set(raw) != {"name", "sha256", "size"}:
                raise ValueError("chunk part fields are invalid")
            parts.append(ChunkPart(raw["name"], raw["sha256"], raw["size"]))
        result = cls(value["sha256"], value["size"], tuple(parts))
        if result.encode() != payload:
            raise ValueError("chunk manifest is not canonical")
        return result


def build_chunk_manifest(path: str | Path) -> ChunkManifest:
    source = Path(path)
    complete = hashlib.sha256()
    parts = []
    size = 0
    with source.open("rb") as stream:
        for index, content in enumerate(iter(lambda: stream.read(CHUNK_BYTES), b"")):
            complete.update(content)
            size += len(content)
            parts.append(
                ChunkPart(
                    f"part-{index:05d}",
                    hashlib.sha256(content).hexdigest(),
                    len(content),
                )
            )
    return ChunkManifest(complete.hexdigest(), size, tuple(parts))


def read_part(path: str | Path, index: int, expected_size: int) -> bytes:
    with Path(path).open("rb") as stream:
        stream.seek(index * CHUNK_BYTES)
        content = stream.read(expected_size)
        if len(content) != expected_size or stream.read(1) and expected_size < CHUNK_BYTES:
            raise ValueError("source file changed while it was being uploaded")
        return content


def chunk_object_name(manifest_name: str, part_name: str) -> str:
    if manifest_name.endswith("/manifest.json"):
        return f"{manifest_name.rsplit('/', 1)[0]}/{part_name}"
    return f"{manifest_name}.parts/{part_name}"


def verify_chunked_content(
    manifest: ChunkManifest,
    open_part: Callable[[str], Any],
    *,
    write: Callable[[bytes], Any] | None = None,
) -> None:
    """Stream and verify actual part and complete bytes, optionally copying them."""

    complete = hashlib.sha256()
    for part in manifest.parts:
        body = open_part(part.name)
        received = 0
        digest = hashlib.sha256()
        try:
            if not callable(getattr(body, "read", None)):
                raise ValueError("chunk response has no readable body")
            while True:
                content = body.read(min(64 * 1024, part.size - received + 1))
                if not isinstance(content, bytes):
                    raise ValueError("chunk response contains non-byte content")
                if not content:
                    break
                if len(content) > part.size - received:
                    raise ValueError("chunk exceeds manifest size")
                received += len(content)
                digest.update(content)
                complete.update(content)
                if write is not None:
                    write(content)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if received != part.size or digest.hexdigest() != part.sha256:
            raise ValueError("chunk conflicts with manifest")
    if complete.hexdigest() != manifest.sha256:
        raise ValueError("file conflicts with its complete digest")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate chunk manifest field: {key}")
        result[key] = value
    return result
