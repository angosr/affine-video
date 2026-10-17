"""Canonical source-pool and artifact commitments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PARTITIONS = frozenset({"production", "audit"})
_PARTITION_ORDER = {"production": 0, "audit": 1}


def canonical_json(value: Any) -> bytes:
    """Encode the single JSON representation used by private commitments."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def decode_canonical_json(payload: bytes) -> Any:
    """Decode canonical JSON while rejecting duplicate keys and aliases."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("canonical JSON payload must be non-empty bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical JSON payload is invalid") from exc
    if canonical_json(value) != payload:
        raise ValueError("JSON payload does not use the canonical encoding")
    return value


@dataclass(frozen=True, slots=True)
class SourceLeaf:
    partition: str
    source_family_id: str
    source_id: str
    prompt: str
    condition_asset_sha256: tuple[str, ...]
    generation_geometry: str
    redistribution_license_id: str
    commitment_nonce: str

    def __post_init__(self) -> None:
        if self.partition not in _PARTITIONS:
            raise ValueError("source partition must be production or audit")
        for name, value in (
            ("source_family_id", self.source_family_id),
            ("source_id", self.source_id),
            ("prompt", self.prompt),
            ("generation_geometry", self.generation_geometry),
            ("redistribution_license_id", self.redistribution_license_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"source {name} must be a non-empty string")
        if len(set(self.condition_asset_sha256)) != len(self.condition_asset_sha256):
            raise ValueError("source condition assets must be unique")
        for digest in self.condition_asset_sha256:
            _require_digest(digest, "condition asset SHA-256")
        _require_digest(self.commitment_nonce, "source commitment nonce")

    def document(self) -> dict[str, Any]:
        return {
            "condition_asset_sha256": list(self.condition_asset_sha256),
            "commitment_nonce": self.commitment_nonce,
            "generation_geometry": self.generation_geometry,
            "partition": self.partition,
            "prompt": self.prompt,
            "redistribution_license_id": self.redistribution_license_id,
            "source_family_id": self.source_family_id,
            "source_id": self.source_id,
        }

    @classmethod
    def from_document(cls, value: Any) -> SourceLeaf:
        row = _object(
            value,
            {
                "condition_asset_sha256",
                "commitment_nonce",
                "generation_geometry",
                "partition",
                "prompt",
                "redistribution_license_id",
                "source_family_id",
                "source_id",
            },
            "source leaf",
        )
        assets = row["condition_asset_sha256"]
        if not isinstance(assets, list) or any(not isinstance(item, str) for item in assets):
            raise ValueError("source condition assets must be a string list")
        return cls(
            row["partition"],
            row["source_family_id"],
            row["source_id"],
            row["prompt"],
            tuple(assets),
            row["generation_geometry"],
            row["redistribution_license_id"],
            row["commitment_nonce"],
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(b"\x00" + canonical_json(self.document())).hexdigest()


@dataclass(frozen=True, slots=True)
class MerkleStep:
    side: str
    sha256: str

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Merkle sibling side must be left or right")
        _require_digest(self.sha256, "Merkle sibling SHA-256")


@dataclass(frozen=True, slots=True)
class SourceProof:
    leaf: SourceLeaf
    index: int
    siblings: tuple[MerkleStep, ...]

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Merkle leaf index must be a non-negative integer")

    def document(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leaf": self.leaf.document(),
            "siblings": [
                {"sha256": sibling.sha256, "side": sibling.side} for sibling in self.siblings
            ],
        }

    @classmethod
    def from_document(cls, value: Any) -> SourceProof:
        row = _object(value, {"index", "leaf", "siblings"}, "source proof")
        siblings = row["siblings"]
        if not isinstance(siblings, list):
            raise ValueError("source proof siblings must be a list")
        parsed = []
        for value in siblings:
            sibling = _object(value, {"sha256", "side"}, "Merkle sibling")
            parsed.append(MerkleStep(sibling["side"], sibling["sha256"]))
        return cls(SourceLeaf.from_document(row["leaf"]), row["index"], tuple(parsed))


class SourcePool:
    """Sorted Merkle tree over the frozen source identities."""

    def __init__(self, leaves: tuple[SourceLeaf, ...]) -> None:
        ordered = tuple(
            sorted(
                leaves,
                key=lambda leaf: (
                    _PARTITION_ORDER[leaf.partition],
                    leaf.generation_geometry,
                    leaf.source_id,
                ),
            )
        )
        if not ordered:
            raise ValueError("source pool cannot be empty")
        if len({leaf.source_id for leaf in ordered}) != len(ordered):
            raise ValueError("source IDs must be unique")
        families: set[str] = set()
        for leaf in ordered:
            if leaf.source_family_id in families:
                raise ValueError("source families must be unique across the committed pool")
            families.add(leaf.source_family_id)
        self.leaves = ordered
        self._levels = _merkle_levels(tuple(bytes.fromhex(leaf.digest) for leaf in ordered))

    @property
    def root(self) -> str:
        return self._levels[-1][0].hex()

    def encode(self) -> bytes:
        return canonical_json(
            {
                "format": "veda-source-pool-v1",
                "leaves": [leaf.document() for leaf in self.leaves],
            }
        )

    def public_commitments(self) -> tuple[dict[str, str], ...]:
        """Return the salted, prompt-hiding index used for public source selection."""

        return tuple(
            {
                "generation_geometry": leaf.generation_geometry,
                "partition": leaf.partition,
                "sha256": leaf.digest,
            }
            for leaf in self.leaves
        )

    @classmethod
    def decode(cls, payload: bytes) -> SourcePool:
        root = _object(decode_canonical_json(payload), {"format", "leaves"}, "source pool")
        if root["format"] != "veda-source-pool-v1":
            raise ValueError("source pool format is unsupported")
        rows = root["leaves"]
        if not isinstance(rows, list):
            raise ValueError("source pool leaves must be a list")
        return cls(tuple(SourceLeaf.from_document(row) for row in rows))

    def proof(self, source_id: str) -> SourceProof:
        try:
            index = next(i for i, leaf in enumerate(self.leaves) if leaf.source_id == source_id)
        except StopIteration as exc:
            raise KeyError(source_id) from exc
        position = index
        siblings: list[MerkleStep] = []
        for level in self._levels[:-1]:
            sibling_position = position - 1 if position % 2 else min(position + 1, len(level) - 1)
            siblings.append(
                MerkleStep("left" if position % 2 else "right", level[sibling_position].hex())
            )
            position //= 2
        return SourceProof(self.leaves[index], index, tuple(siblings))


@dataclass(frozen=True, slots=True)
class SourceCacheBinding:
    """Commit one source pool to the exact encoded SampleCache bytes it uses."""

    source_pool_merkle_root: str
    index_sha256: str
    text_sha256: str
    condition_sha256: str | None

    def __post_init__(self) -> None:
        _require_digest(self.source_pool_merkle_root, "source pool Merkle root")
        _require_digest(self.index_sha256, "sample index SHA-256")
        _require_digest(self.text_sha256, "sample text SHA-256")
        if self.condition_sha256 is not None:
            _require_digest(self.condition_sha256, "sample condition SHA-256")

    def document(self) -> dict[str, Any]:
        return {
            "condition_sha256": self.condition_sha256,
            "format": "veda-source-cache-binding-v1",
            "index_sha256": self.index_sha256,
            "source_pool_merkle_root": self.source_pool_merkle_root,
            "text_sha256": self.text_sha256,
        }

    def encode(self) -> bytes:
        return canonical_json(self.document())

    @classmethod
    def decode(cls, payload: bytes) -> SourceCacheBinding:
        root = _object(
            decode_canonical_json(payload),
            {
                "condition_sha256",
                "format",
                "index_sha256",
                "source_pool_merkle_root",
                "text_sha256",
            },
            "source cache binding",
        )
        if root["format"] != "veda-source-cache-binding-v1":
            raise ValueError("source cache binding format is unsupported")
        condition = root["condition_sha256"]
        if condition is not None and not isinstance(condition, str):
            raise ValueError("source condition cache digest must be a string or null")
        return cls(
            root["source_pool_merkle_root"],
            root["index_sha256"],
            root["text_sha256"],
            condition,
        )

    def verify_local(self, pool: SourcePool, sample_cache: str | Path) -> None:
        """Reject an ID-preserving substitution of any encoded source input."""

        if pool.root != self.source_pool_merkle_root:
            raise ValueError("source cache binding targets another source pool")
        root = Path(sample_cache)
        index = root / "index.json"
        text = root / "text.safetensors"
        condition = root / "cond.safetensors"
        if _file_sha256(index) != self.index_sha256:
            raise ValueError("encoded sample index differs from its source binding")
        if _file_sha256(text) != self.text_sha256:
            raise ValueError("encoded sample text differs from its source binding")
        if self.condition_sha256 is None:
            if condition.exists():
                raise ValueError("unexpected encoded sample conditions are present")
        elif _file_sha256(condition) != self.condition_sha256:
            raise ValueError("encoded sample conditions differ from their source binding")

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate encoded sample index field: {key}")
                result[key] = value
            return result

        try:
            document = json.loads(index.read_text(encoding="utf-8"), object_pairs_hook=unique)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("encoded sample index is invalid") from exc
        if not isinstance(document, dict) or set(document) != {"samples"}:
            raise ValueError("encoded sample index fields are invalid")
        samples = document["samples"]
        if not isinstance(samples, list) or any(not isinstance(row, dict) for row in samples):
            raise ValueError("encoded sample index rows are invalid")
        identifiers = [row.get("id") for row in samples]
        if (
            any(not isinstance(source_id, str) or not source_id for source_id in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or set(identifiers) != {leaf.source_id for leaf in pool.leaves}
        ):
            raise ValueError("encoded sample IDs differ from the committed source pool")


def load_bound_source_pool(
    source_pool: bytes,
    source_binding: bytes,
    *,
    expected_root: str,
    expected_commitments: tuple[dict[str, str], ...] | None = None,
    sample_cache: str | Path,
) -> SourcePool:
    """Load the only source pool whose committed inputs may enter runtime."""

    pool = SourcePool.decode(source_pool)
    if pool.root != expected_root:
        raise ValueError("source pool does not match the track commitment")
    if expected_commitments is not None and pool.public_commitments() != expected_commitments:
        raise ValueError("source pool differs from the public selection index")
    binding = SourceCacheBinding.decode(source_binding)
    binding.verify_local(pool, sample_cache)
    return pool


def verify_source_proof(proof: SourceProof, expected_root: str) -> bool:
    """Verify one disclosed source without revealing the remaining pool."""

    _require_digest(expected_root, "source pool Merkle root")
    value = bytes.fromhex(proof.leaf.digest)
    for sibling in proof.siblings:
        other = bytes.fromhex(sibling.sha256)
        value = _node_hash(other, value) if sibling.side == "left" else _node_hash(value, other)
    return value.hex() == expected_root


@dataclass(frozen=True, slots=True)
class Artifact:
    logical_name: str
    sha256: str
    size_bytes: int
    object_key: str

    def __post_init__(self) -> None:
        if not self.logical_name or not self.object_key:
            raise ValueError("artifact name and object key cannot be empty")
        for value, name in (
            (self.logical_name, "logical name"),
            (self.object_key, "object key"),
        ):
            if value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
                raise ValueError(f"artifact {name} is invalid")
        _require_digest(self.sha256, "artifact SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("artifact size must be a positive integer")

    def document(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "object_key": self.object_key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifacts: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        if not self.artifacts:
            raise ValueError("artifact manifest cannot be empty")
        names = [artifact.logical_name for artifact in self.artifacts]
        if len(set(names)) != len(names):
            raise ValueError("artifact logical names must be unique")
        objects: dict[str, tuple[str, int]] = {}
        for artifact in self.artifacts:
            identity = (artifact.sha256, artifact.size_bytes)
            previous = objects.setdefault(artifact.object_key, identity)
            if previous != identity:
                raise ValueError("one artifact object key identifies different content")
        if tuple(names) != tuple(sorted(names)):
            raise ValueError("artifacts must be ordered by logical name")

    def document(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.document() for artifact in self.artifacts],
            "format": "veda-artifact-manifest-v1",
        }

    def encode(self) -> bytes:
        return canonical_json(self.document())

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.document())).hexdigest()

    def verify_local(self, root: str | Path) -> None:
        base = Path(root)
        for artifact in self.artifacts:
            path = base / artifact.logical_name
            if not path.is_file() or path.stat().st_size != artifact.size_bytes:
                raise ValueError(
                    f"artifact is missing or has the wrong size: {artifact.logical_name}"
                )
            if _file_sha256(path) != artifact.sha256:
                raise ValueError(f"artifact digest does not match: {artifact.logical_name}")

    def verify_runtime_coverage(
        self,
        root: str | Path,
        *,
        files: tuple[str, ...],
        directories: tuple[str, ...],
    ) -> None:
        """Require every configured runtime byte to be covered by the commitment."""

        base = Path(root).resolve()
        committed = {artifact.logical_name for artifact in self.artifacts}
        required = set(files)
        for name in directories:
            directory = (base / name).resolve()
            if directory != base and base not in directory.parents:
                raise ValueError("runtime directory escapes the artifact root")
            if not directory.is_dir():
                raise ValueError(f"runtime artifact directory is missing: {name}")
            required.update(
                path.relative_to(base).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and _is_runtime_file(path.relative_to(directory))
            )
        missing = required - committed
        if missing:
            raise ValueError(
                f"artifact manifest omits runtime bytes: {', '.join(sorted(missing)[:5])}"
            )

    @classmethod
    def decode(cls, payload: bytes) -> ArtifactManifest:
        root = _object(decode_canonical_json(payload), {"artifacts", "format"}, "artifact manifest")
        if root["format"] != "veda-artifact-manifest-v1":
            raise ValueError("artifact manifest format is unsupported")
        rows = root["artifacts"]
        if not isinstance(rows, list):
            raise ValueError("artifact manifest artifacts must be a list")
        artifacts = []
        for value in rows:
            row = _object(
                value,
                {"logical_name", "object_key", "sha256", "size_bytes"},
                "artifact",
            )
            artifacts.append(
                Artifact(
                    row["logical_name"],
                    row["sha256"],
                    row["size_bytes"],
                    row["object_key"],
                )
            )
        return cls(tuple(artifacts))


def _is_runtime_file(relative: Path) -> bool:
    return (
        ".cache" not in relative.parts
        and "__pycache__" not in relative.parts
        and relative.suffix not in {".lock", ".pyc"}
    )


def _merkle_levels(values: tuple[bytes, ...]) -> tuple[tuple[bytes, ...], ...]:
    levels = [values]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(
            tuple(
                _node_hash(current[index], current[min(index + 1, len(current) - 1)])
                for index in range(0, len(current), 2)
            )
        )
    return tuple(levels)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase hexadecimal containing 32 bytes")


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
