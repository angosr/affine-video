"""Verified, one-case-at-a-time access to exact-cache v2 NPZ batches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from scoring.exact import ExactCase, validate_dense_reference
from scoring.spec import EXACT_CACHE_FORMAT
from validator.production.plan import scheduled_generation_block


class CacheLoadError(ValueError):
    """Raised when a frozen scoring cache is missing, changed, or malformed."""


_BATCH_NAME = re.compile(r"^(0|[1-9][0-9]*)-([0-9a-f]{64})[.]npz$")
_ARRAY_FIELDS = (
    "q",
    "k",
    "v",
    "dense",
    "query_valid_counts",
    "key_valid_counts",
    "key_segments",
    "required_keys",
)
_CASE_FIELDS = {
    "epsilon",
    "query_tile_ids",
    "rollout_id",
    "route",
    "scheduler_timestep",
    "sigma",
    "source_id",
    "step_index",
    "step_window",
    "stratum",
    "tile_plan_hash",
}
_ROOT_FIELDS = {
    "artifact_manifest_sha256",
    "cases",
    "format",
    "generated_block",
    "plan_digest",
    "scorer_schema",
    "seed_block_hash",
    "source_pool_merkle_root",
    "source_proofs",
    "storage_dtype",
    "track_spec_digest",
    "dense_references",
    "selected_inputs",
}


@dataclass(frozen=True, slots=True)
class ExactCacheBatch:
    path: Path
    sha256: str
    generated_block: int
    plan_digest: str
    track_spec_digest: str
    seed_block_hash: str
    cases: tuple[tuple[str, str, str, int], ...]
    artifact_manifest_sha256: str
    source_pool_merkle_root: str
    source_proofs: tuple[Mapping[str, Any], ...]
    dense_references: tuple[Mapping[str, Any], ...]
    selected_inputs: Mapping[str, Any]


def write_exact_cache_batch(
    directory: str | Path,
    cases: tuple[ExactCase, ...],
    *,
    generated_block: int,
    seed_block_hash: str,
    plan_digest: str,
    track_spec_digest: str,
    artifact_manifest_sha256: str,
    source_pool_merkle_root: str,
    source_proofs: tuple[Mapping[str, Any], ...],
    dense_references: tuple[Mapping[str, Any], ...],
    selected_inputs: Mapping[str, Any],
    storage_dtype: str = "float16",
) -> ExactCacheBatch:
    """Atomically serialize a plan-bound cache in canonical case order."""

    if type(generated_block) is not int or generated_block < 0 or not cases:
        raise ValueError("cache generation block and cases are invalid")
    if storage_dtype != "float16":
        raise ValueError("exact-cache v2 currently requires float16 storage")
    seed_block_hash = _block_hash(seed_block_hash, ValueError)
    for digest, name in (
        (plan_digest, "plan digest"),
        (track_spec_digest, "track specification digest"),
        (artifact_manifest_sha256, "artifact manifest SHA-256"),
        (source_pool_merkle_root, "source pool Merkle root"),
    ):
        _digest(digest, name, ValueError)
    ordered = tuple(
        sorted(
            cases,
            key=lambda case: (
                case.route,
                case.stratum,
                case.rollout_id,
                case.step_index,
            ),
        )
    )
    if len(
        {(case.route, case.stratum, case.rollout_id, case.step_index) for case in ordered}
    ) != len(ordered):
        raise ValueError("cache case coordinates must be unique")
    metadata_cases = [_case_document(case) for case in ordered]
    metadata_document = {
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "cases": metadata_cases,
        "format": EXACT_CACHE_FORMAT,
        "generated_block": generated_block,
        "plan_digest": plan_digest,
        "scorer_schema": "veda-selector-segmented-v1",
        "seed_block_hash": seed_block_hash,
        "source_pool_merkle_root": source_pool_merkle_root,
        "source_proofs": list(source_proofs),
        "storage_dtype": storage_dtype,
        "track_spec_digest": track_spec_digest,
        "dense_references": list(dense_references),
        "selected_inputs": dict(selected_inputs),
    }
    metadata = np.frombuffer(_canonical_json(metadata_document), dtype=np.uint8)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with zipfile.ZipFile(
                temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                _write_array(archive, "metadata.npy", metadata)
                for index, case in enumerate(ordered):
                    prefix = f"case_{index:06d}"
                    for field in _ARRAY_FIELDS:
                        value = np.asarray(getattr(case, field))
                        if value.dtype.hasobject:
                            raise ValueError("cache arrays cannot contain objects")
                        if field in {"q", "k", "v", "dense"}:
                            value = value.astype(np.float16, copy=False)
                        _write_array(archive, f"{prefix}_{field}.npy", value)
            temporary.flush()
            os.fsync(temporary.fileno())
        sha256 = _file_sha256(temporary_path)
        target = root / f"{generated_block}-{sha256}.npz"
        if target.exists():
            if _file_sha256(target) != sha256:
                raise CacheLoadError("existing cache batch content is invalid")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, target)
        return _batch_from_metadata(target, sha256, metadata_document)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def assemble_exact_cache_batch(
    directory: str | Path,
    shards: tuple[str | Path, ...],
    *,
    generated_block: int,
    seed_block_hash: str,
    plan_digest: str,
    track_spec_digest: str,
    artifact_manifest_sha256: str,
    source_pool_merkle_root: str,
    source_proofs: tuple[Mapping[str, Any], ...],
    dense_references: tuple[Mapping[str, Any], ...],
    selected_inputs: Mapping[str, Any],
) -> ExactCacheBatch:
    """Assemble rollout shards without materializing any Q/K/V array."""

    if not shards:
        raise ValueError("cache assembly requires rollout shards")
    locations: list[tuple[dict[str, Any], Path, int]] = []
    for shard_value in shards:
        shard = Path(shard_value)
        metadata = _read_metadata(shard)
        for index, case in enumerate(metadata["cases"]):
            locations.append((_case_metadata(case), shard, index))
    ordered = sorted(
        locations,
        key=lambda item: (
            item[0]["route"],
            item[0]["stratum"],
            item[0]["rollout_id"],
            item[0]["step_index"],
        ),
    )
    coordinates = [
        (item[0]["route"], item[0]["stratum"], item[0]["rollout_id"], item[0]["step_index"])
        for item in ordered
    ]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("cache shard coordinates must be unique")
    seed_block_hash = _block_hash(seed_block_hash, ValueError)
    for digest, name in (
        (plan_digest, "plan digest"),
        (track_spec_digest, "track specification digest"),
        (artifact_manifest_sha256, "artifact manifest SHA-256"),
        (source_pool_merkle_root, "source pool Merkle root"),
    ):
        _digest(digest, name, ValueError)
    document = {
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "cases": [item[0] for item in ordered],
        "format": EXACT_CACHE_FORMAT,
        "generated_block": generated_block,
        "plan_digest": plan_digest,
        "scorer_schema": "veda-selector-segmented-v1",
        "seed_block_hash": seed_block_hash,
        "source_pool_merkle_root": source_pool_merkle_root,
        "source_proofs": list(source_proofs),
        "storage_dtype": "float16",
        "track_spec_digest": track_spec_digest,
        "dense_references": list(dense_references),
        "selected_inputs": dict(selected_inputs),
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with zipfile.ZipFile(
                temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as target:
                _write_array(
                    target,
                    "metadata.npy",
                    np.frombuffer(_canonical_json(document), dtype=np.uint8),
                )
                for target_index, (_case, shard, source_index) in enumerate(ordered):
                    with zipfile.ZipFile(shard) as source:
                        for field in _ARRAY_FIELDS:
                            source_name = f"case_{source_index:06d}_{field}.npy"
                            target_name = f"case_{target_index:06d}_{field}.npy"
                            with (
                                source.open(source_name) as incoming,
                                target.open(target_name, "w", force_zip64=True) as outgoing,
                            ):
                                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        sha256 = _file_sha256(temporary_path)
        target_path = root / f"{generated_block}-{sha256}.npz"
        if target_path.exists():
            if _file_sha256(target_path) != sha256:
                raise CacheLoadError("existing assembled cache content is invalid")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, target_path)
        return _batch_from_metadata(target_path, sha256, document)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def inspect_exact_cache_batch(path: str | Path, *, max_bytes: int) -> ExactCacheBatch:
    """Read bounded provenance without materializing any case array."""

    cache_path = Path(path)
    sha256 = _file_sha256(cache_path)
    _verify_archive(cache_path, sha256, max_bytes)
    return _batch_from_metadata(cache_path, sha256, _read_metadata(cache_path))


def select_exact_cache_batch(
    directory: str | Path,
    *,
    submission_block: int,
    minimum_delay_blocks: int,
    track_start_block: int,
    generation_interval_blocks: int,
    expected_track_spec_digest: str,
    max_bytes: int,
    head_dim: int,
    tile_size: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
    sampled_route_count: int,
    rollouts_per_stratum: int,
    sampled_step_count: int,
    query_tiles_per_case: int,
) -> ExactCacheBatch:
    expected_block = scheduled_generation_block(
        submission_block,
        track_start_block=track_start_block,
        minimum_generation_delay_blocks=minimum_delay_blocks,
        generation_interval_blocks=generation_interval_blocks,
    )
    root = Path(directory)
    candidates = []
    try:
        paths = sorted(root.glob("*.npz"))
    except OSError as exc:
        raise CacheLoadError("cache batch directory cannot be read") from exc
    for path in paths:
        match = _BATCH_NAME.fullmatch(path.name)
        if match is None:
            raise CacheLoadError("cache batch filename is invalid")
        if int(match.group(1)) == expected_block:
            candidates.append((match.group(2), path))
    if len(candidates) != 1:
        raise CacheLoadError("the scheduled exact cache batch is unavailable or ambiguous")
    named_sha256, path = candidates[0]
    batch = inspect_exact_cache_batch(path, max_bytes=max_bytes)
    if batch.generated_block != expected_block or batch.sha256 != named_sha256:
        raise CacheLoadError("cache batch filename does not match its content")
    validate_exact_cache(
        path,
        expected_sha256=batch.sha256,
        expected_track_spec_digest=expected_track_spec_digest,
        max_bytes=max_bytes,
        head_dim=head_dim,
        tile_size=tile_size,
        routes=routes,
        strata=strata,
        sampled_route_count=sampled_route_count,
        rollouts_per_stratum=rollouts_per_stratum,
        sampled_step_count=sampled_step_count,
        query_tiles_per_case=query_tiles_per_case,
    )
    return batch


def validate_exact_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_track_spec_digest: str,
    max_bytes: int,
    head_dim: int,
    tile_size: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
    sampled_route_count: int,
    rollouts_per_stratum: int,
    sampled_step_count: int,
    query_tiles_per_case: int,
    dense_device: str | None = None,
) -> None:
    """Validate the complete cross design and every stored dense reference."""

    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    metadata = _read_metadata(cache_path)
    if metadata["track_spec_digest"] != expected_track_spec_digest:
        raise CacheLoadError("cache belongs to a different track specification")
    cases = [_case_metadata(item) for item in metadata["cases"]]
    expected_count = sampled_route_count * len(strata) * rollouts_per_stratum * sampled_step_count
    if len(cases) != expected_count:
        raise CacheLoadError("cache case count does not match the cross design")
    route_set = {case["route"] for case in cases}
    if len(route_set) != sampled_route_count or not route_set <= set(routes):
        raise CacheLoadError("cache route sample does not match the track")
    if {case["stratum"] for case in cases} != set(strata):
        raise CacheLoadError("cache does not cover every scoring stratum")
    steps = {case["step_index"] for case in cases}
    if len(steps) != sampled_step_count:
        raise CacheLoadError("cache step sample count does not match the track")
    for stratum in strata:
        rollouts = {case["rollout_id"] for case in cases if case["stratum"] == stratum}
        if len(rollouts) != rollouts_per_stratum:
            raise CacheLoadError("cache rollout count does not match the track")
        expected = {
            (route, rollout, step) for route in route_set for rollout in rollouts for step in steps
        }
        actual = {
            (case["route"], case["rollout_id"], case["step_index"])
            for case in cases
            if case["stratum"] == stratum
        }
        if actual != expected:
            raise CacheLoadError("cache cases do not form the complete cross design")
    expected_names = {"metadata.npy"}
    with zipfile.ZipFile(cache_path) as archive:
        for index, _case in enumerate(cases):
            prefix = f"case_{index:06d}"
            expected_names.update(f"{prefix}_{field}.npy" for field in _ARRAY_FIELDS)
            shapes = {
                field: _npy_header(archive, f"{prefix}_{field}.npy") for field in _ARRAY_FIELDS
            }
            _validate_headers(
                shapes,
                query_count=query_tiles_per_case,
                tile_size=tile_size,
                head_dim=head_dim,
            )
        if set(archive.namelist()) != expected_names:
            raise CacheLoadError("cache archive members do not match its metadata")
    _validate_dense_references(cache_path, cases, device=dense_device)


def validate_exact_cache_references(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    device: str | None = None,
) -> None:
    """Stream and verify Q/K/V→dense consistency for a cache or rollout shard."""

    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    metadata = _read_metadata(cache_path)
    cases = [_case_metadata(item) for item in metadata["cases"]]
    _validate_dense_references(cache_path, cases, device=device)


class ExactCacheReader:
    """Context-managed iterator that materializes only the current case."""

    def __init__(self, path: str | Path, *, expected_sha256: str, max_bytes: int) -> None:
        self.path = _verify_archive(path, expected_sha256, max_bytes)
        self.metadata = _read_metadata(self.path)
        self._cases = tuple(_case_metadata(item) for item in self.metadata["cases"])
        self._archive: Any | None = None

    def __enter__(self) -> ExactCacheReader:
        self._archive = np.load(self.path, allow_pickle=False)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __iter__(self) -> Iterator[ExactCase]:
        if self._archive is None:
            raise RuntimeError("exact cache reader must be used as a context manager")
        for index, case in enumerate(self._cases):
            yield _load_case(self._archive, index, case)


def _validate_dense_references(
    path: Path,
    cases: list[dict[str, Any]],
    *,
    device: str | None,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            for index, metadata in enumerate(cases):
                case = _load_case(archive, index, metadata)
                validate_dense_reference(case, device=device)
                del case
    except CacheLoadError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CacheLoadError("cache dense reference is inconsistent with its Q/K/V") from exc


def _load_case(archive: Any, index: int, case: Mapping[str, Any]) -> ExactCase:
    prefix = f"case_{index:06d}"
    arrays = {field: np.asarray(archive[f"{prefix}_{field}"]) for field in _ARRAY_FIELDS}
    return ExactCase(
        route=case["route"],
        stratum=case["stratum"],
        rollout_id=case["rollout_id"],
        source_id=case["source_id"],
        step_index=case["step_index"],
        step_window=case["step_window"],
        scheduler_timestep=float(case["scheduler_timestep"]),
        sigma=float(case["sigma"]),
        tile_plan_hash=case["tile_plan_hash"],
        query_tile_ids=tuple(case["query_tile_ids"]),
        epsilon=float(case["epsilon"]),
        **arrays,
    )


def _case_document(case: ExactCase) -> dict[str, Any]:
    if not isfinite(case.epsilon) or case.epsilon <= 0:
        raise ValueError("cache case epsilon must be finite and positive")
    return {
        "epsilon": case.epsilon,
        "query_tile_ids": list(case.query_tile_ids),
        "rollout_id": case.rollout_id,
        "route": case.route,
        "scheduler_timestep": case.scheduler_timestep,
        "sigma": case.sigma,
        "source_id": case.source_id,
        "step_index": case.step_index,
        "step_window": case.step_window,
        "stratum": case.stratum,
        "tile_plan_hash": case.tile_plan_hash,
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive["metadata"])
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise CacheLoadError("cache metadata must be a byte vector")
        value = json.loads(raw.tobytes().decode("utf-8"), object_pairs_hook=_unique_object)
        root = _object(value, _ROOT_FIELDS, "cache metadata")
        if root["format"] != EXACT_CACHE_FORMAT:
            raise CacheLoadError("cache format identifier is invalid")
        if root["storage_dtype"] != "float16":
            raise CacheLoadError("cache storage dtype is unsupported")
        if root["scorer_schema"] != "veda-selector-segmented-v1":
            raise CacheLoadError("cache scorer schema is unsupported")
        if not isinstance(root["cases"], list) or not root["cases"]:
            raise CacheLoadError("cache must contain at least one case")
        for digest, name in (
            (root["plan_digest"], "plan digest"),
            (root["track_spec_digest"], "track digest"),
            (root["artifact_manifest_sha256"], "artifact manifest digest"),
            (root["source_pool_merkle_root"], "source pool root"),
        ):
            _digest(digest, name, CacheLoadError)
        _block_hash(root["seed_block_hash"], CacheLoadError)
        if type(root["generated_block"]) is not int or root["generated_block"] < 0:
            raise CacheLoadError("cache generated block is invalid")
        if (
            not isinstance(root["source_proofs"], list)
            or not isinstance(root["dense_references"], list)
            or not isinstance(root["selected_inputs"], dict)
        ):
            raise CacheLoadError("cache disclosure metadata is invalid")
        return root
    except CacheLoadError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CacheLoadError("cache metadata cannot be read") from exc


def _case_metadata(value: Any) -> dict[str, Any]:
    case = _object(value, _CASE_FIELDS, "exact case")
    for field in (
        "rollout_id",
        "route",
        "source_id",
        "step_window",
        "stratum",
        "tile_plan_hash",
    ):
        if not isinstance(case[field], str) or not case[field]:
            raise CacheLoadError(f"cache case {field} is invalid")
    if type(case["step_index"]) is not int or case["step_index"] < 0:
        raise CacheLoadError("cache case step index is invalid")
    if (
        isinstance(case["epsilon"], bool)
        or not isinstance(case["epsilon"], (int, float))
        or not isfinite(float(case["epsilon"]))
        or case["epsilon"] <= 0
    ):
        raise CacheLoadError("cache case epsilon is invalid")
    query_ids = case["query_tile_ids"]
    if (
        not isinstance(query_ids, list)
        or not query_ids
        or any(type(item) is not int or item < 0 for item in query_ids)
        or len(set(query_ids)) != len(query_ids)
    ):
        raise CacheLoadError("cache query tile identifiers are invalid")
    return case


def _batch_from_metadata(path: Path, sha256: str, metadata: dict[str, Any]) -> ExactCacheBatch:
    cases = tuple(
        (
            case["route"],
            case["stratum"],
            case["rollout_id"],
            case["step_index"],
        )
        for case in (_case_metadata(item) for item in metadata["cases"])
    )
    return ExactCacheBatch(
        path,
        sha256,
        metadata["generated_block"],
        metadata["plan_digest"],
        metadata["track_spec_digest"],
        _block_hash(metadata["seed_block_hash"], CacheLoadError),
        cases,
        metadata["artifact_manifest_sha256"],
        metadata["source_pool_merkle_root"],
        tuple(metadata["source_proofs"]),
        tuple(metadata["dense_references"]),
        metadata["selected_inputs"],
    )


def _validate_headers(
    headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
    *,
    query_count: int,
    tile_size: int,
    head_dim: int,
) -> None:
    q_shape, q_dtype = headers["q"]
    k_shape, k_dtype = headers["k"]
    if q_shape != (query_count, tile_size, head_dim) or len(k_shape) != 3:
        raise CacheLoadError("cache Q/K array headers are invalid")
    if k_shape[1:] != (tile_size, head_dim) or k_shape[0] == 0:
        raise CacheLoadError("cache K array header is invalid")
    if q_dtype != np.dtype(np.float16) or k_dtype != np.dtype(np.float16):
        raise CacheLoadError("cache Q/K storage dtype is invalid")
    if headers["v"] != (k_shape, k_dtype) or headers["dense"] != (q_shape, q_dtype):
        raise CacheLoadError("cache V/dense array headers are invalid")
    expected = {
        "query_valid_counts": ((query_count,), np.dtype(np.int16)),
        "key_valid_counts": ((k_shape[0],), np.dtype(np.int16)),
        "key_segments": ((k_shape[0],), np.dtype(np.int8)),
        "required_keys": ((query_count, k_shape[0]), np.dtype(np.bool_)),
    }
    if any(headers[name] != value for name, value in expected.items()):
        raise CacheLoadError("cache mask array headers are invalid")


def _npy_header(archive: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
    try:
        with archive.open(name) as member:
            major, minor = read_magic(member)
            if (major, minor) == (1, 0):
                shape, fortran, dtype = read_array_header_1_0(member)
            elif major in {2, 3}:
                shape, fortran, dtype = read_array_header_2_0(member)
            else:
                raise CacheLoadError("cache array uses an unsupported NPY version")
    except KeyError as exc:
        raise CacheLoadError(f"cache array is missing: {name}") from exc
    if fortran or dtype.hasobject:
        raise CacheLoadError("cache arrays must be C-order and cannot contain objects")
    return tuple(shape), dtype


def _write_array(archive: zipfile.ZipFile, name: str, value: np.ndarray) -> None:
    with archive.open(name, "w", force_zip64=True) as member:
        np.lib.format.write_array(member, value, allow_pickle=False)


def _verify_archive(path: str | Path, expected_sha256: str, max_bytes: int) -> Path:
    cache_path = Path(path)
    if max_bytes <= 0:
        raise ValueError("cache byte limit must be positive")
    _digest(expected_sha256, "cache digest", ValueError)
    if _file_sha256(cache_path) != expected_sha256:
        raise CacheLoadError("cache SHA-256 does not match")
    try:
        with zipfile.ZipFile(cache_path) as archive:
            members = archive.infolist()
            if len({item.filename for item in members}) != len(members):
                raise CacheLoadError("cache archive contains duplicate members")
            if sum(item.file_size for item in members) > max_bytes:
                raise CacheLoadError("cache decoded size exceeds the track limit")
    except CacheLoadError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise CacheLoadError("cache is not a readable NPZ archive") from exc
    return cache_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CacheLoadError("cache is not readable") from exc
    return digest.hexdigest()


def _digest(value: Any, name: str, error: type[Exception]) -> str:
    if not isinstance(value, str):
        raise error(f"{name} must be hexadecimal")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise error(f"{name} must be hexadecimal") from exc
    if len(decoded) != 32:
        raise error(f"{name} must contain 32 bytes")
    return value.lower()


def _block_hash(value: Any, error: type[Exception]) -> str:
    if not isinstance(value, str):
        raise error("cache seed block hash must be hexadecimal")
    normalized = value.removeprefix("0x").lower()
    _digest(normalized, "cache seed block hash", error)
    return "0x" + normalized


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CacheLoadError(f"duplicate cache JSON field: {key}")
        result[key] = value
    return result


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CacheLoadError(f"{name} fields are invalid")
    return value
