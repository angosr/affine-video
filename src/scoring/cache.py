"""Verified loading of the frozen exact scoring cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, prod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from scoring.exact import ExactCase


class CacheLoadError(ValueError):
    """Raised when a frozen scoring cache is missing, changed, or malformed."""


_BATCH_NAME = re.compile(r"^(0|[1-9][0-9]*)-([0-9a-f]{64})[.]npz$")


@dataclass(frozen=True, slots=True)
class ExactCacheBatch:
    path: Path
    sha256: str
    generated_block: int
    source_digest: str


def write_exact_cache_batch(
    directory: str | Path,
    cases: tuple[ExactCase, ...],
    *,
    generated_block: int,
    source_digest: str,
) -> ExactCacheBatch:
    """Atomically serialize one private exact-cache batch for the evaluator."""

    if type(generated_block) is not int or generated_block < 0:
        raise ValueError("cache generated block must be a non-negative integer")
    if not cases:
        raise ValueError("cache batch requires at least one case")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError("cache source digest must contain 32 bytes")
    try:
        bytes.fromhex(source_digest)
    except ValueError as exc:
        raise ValueError("cache source digest must be hexadecimal") from exc
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    metadata_cases = []
    arrays: dict[str, np.ndarray] = {}
    for index, case in enumerate(cases):
        if not case.route or not case.stratum or not isfinite(case.epsilon) or case.epsilon <= 0:
            raise ValueError("cache case metadata is invalid")
        metadata_cases.append(
            {"route": case.route, "stratum": case.stratum, "epsilon": case.epsilon}
        )
        prefix = f"case_{index:06d}"
        for field in ("q", "k", "v", "dense", "valid_keys", "required_keys"):
            value = np.asarray(getattr(case, field))
            if value.dtype.hasobject:
                raise ValueError("cache arrays cannot contain objects")
            arrays[f"{prefix}_{field}"] = value
    metadata = np.frombuffer(
        json.dumps(
            {
                "format": "veda-exact-cache",
                "generated_block": generated_block,
                "source_digest": source_digest.lower(),
                "cases": metadata_cases,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode(),
        dtype=np.uint8,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, metadata=metadata, **arrays)
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
        return ExactCacheBatch(target, sha256, generated_block, source_digest.lower())
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def select_exact_cache_batch(
    directory: str | Path,
    *,
    submission_block: int,
    minimum_delay_blocks: int,
    max_bytes: int,
    min_blocks: int,
    max_blocks: int,
    head_dim: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
    sample_count: int,
) -> ExactCacheBatch:
    """Select the earliest valid batch generated after one submission became immutable."""

    if submission_block < 0 or minimum_delay_blocks <= 0:
        raise ValueError("cache generation boundary is invalid")
    root = Path(directory)
    candidates: list[tuple[int, str, Path]] = []
    try:
        paths = sorted(root.glob("*.npz"))
    except OSError as exc:
        raise CacheLoadError("cache batch directory cannot be read") from exc
    for path in paths:
        match = _BATCH_NAME.fullmatch(path.name)
        if match is None:
            raise CacheLoadError("cache batch filename is invalid")
        generated_block = int(match.group(1))
        if generated_block >= submission_block + minimum_delay_blocks:
            candidates.append((generated_block, match.group(2), path))
    if not candidates:
        raise CacheLoadError("no exact cache batch was generated after the submission")
    named_block, named_sha256, path = min(candidates)
    batch = inspect_exact_cache_batch(path, max_bytes=max_bytes)
    if batch.generated_block != named_block or batch.sha256 != named_sha256:
        raise CacheLoadError("cache batch filename does not match its content")
    validate_exact_cache(
        path,
        expected_sha256=batch.sha256,
        max_bytes=max_bytes,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        head_dim=head_dim,
        routes=routes,
        strata=strata,
        sample_count=sample_count,
    )
    return batch


def inspect_exact_cache_batch(path: str | Path, *, max_bytes: int) -> ExactCacheBatch:
    """Read immutable provenance from one size-bounded exact cache batch."""

    cache_path = Path(path)
    sha256 = _file_sha256(cache_path)
    _verify_archive(cache_path, sha256, max_bytes)
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = _metadata(archive, expected_format="veda-exact-cache")
    except CacheLoadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CacheLoadError("cache metadata cannot be read") from exc
    generated_block = metadata["generated_block"]
    if type(generated_block) is not int or generated_block < 0:
        raise CacheLoadError("cache generated block must be a non-negative integer")
    source_digest = metadata["source_digest"]
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise CacheLoadError("cache source digest must contain 32 bytes")
    try:
        bytes.fromhex(source_digest)
    except ValueError as exc:
        raise CacheLoadError("cache source digest must be hexadecimal") from exc
    return ExactCacheBatch(cache_path, sha256, generated_block, source_digest.lower())


def validate_exact_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    min_blocks: int,
    max_blocks: int,
    head_dim: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
    sample_count: int | None,
) -> None:
    """Validate the exact cache index without materializing case arrays."""

    _validate_cache_index(
        path,
        expected_sha256=expected_sha256,
        max_bytes=max_bytes,
        expected_format="veda-exact-cache",
        case_fields={"route", "stratum", "epsilon"},
        array_fields=("q", "k", "v", "dense", "valid_keys", "required_keys"),
        routes=routes,
        strata=strata,
        sample_count=sample_count,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        head_dim=head_dim,
    )


def load_exact_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[ExactCase, ...]:
    """Load a hash-pinned exact QKV cache without pickle support."""

    return _load_exact_cache(path, expected_sha256, max_bytes)


def _load_exact_cache(
    path: str | Path,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[ExactCase, ...]:
    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = _metadata(archive, expected_format="veda-exact-cache")
            cases = [
                _object(raw_case, {"route", "stratum", "epsilon"}, "exact case")
                for raw_case in metadata["cases"]
            ]
            expected = {"metadata"}
            loaded: list[ExactCase] = []
            for index, case in enumerate(cases):
                prefix = f"case_{index:06d}"
                names = {
                    field: f"{prefix}_{field}"
                    for field in ("q", "k", "v", "dense", "valid_keys", "required_keys")
                }
                expected.update(names.values())
                epsilon = case["epsilon"]
                if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
                    raise CacheLoadError("exact case epsilon must be numeric")
                loaded.append(
                    ExactCase(
                        route=_string(case["route"], "exact route"),
                        stratum=_string(case["stratum"], "exact stratum"),
                        q=_array(archive, names["q"]),
                        k=_array(archive, names["k"]),
                        v=_array(archive, names["v"]),
                        dense=_array(archive, names["dense"]),
                        valid_keys=_array(archive, names["valid_keys"]),
                        required_keys=_array(archive, names["required_keys"]),
                        epsilon=float(epsilon),
                    )
                )
            _exact_keys(archive, expected)
    except CacheLoadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CacheLoadError("exact cache arrays cannot be read") from exc
    return tuple(loaded)


def _validate_cache_index(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    expected_format: str,
    case_fields: set[str],
    array_fields: tuple[str, ...],
    routes: tuple[str, ...],
    strata: Mapping[str, float],
    sample_count: int | None,
    min_blocks: int,
    max_blocks: int,
    head_dim: int,
) -> None:
    if (
        type(min_blocks) is not int
        or type(max_blocks) is not int
        or min_blocks <= 0
        or max_blocks < min_blocks
    ):
        raise ValueError("block budget range is invalid")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("attention head dimension must be a positive integer")
    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    try:
        with np.load(cache_path, allow_pickle=False) as archive, zipfile.ZipFile(
            cache_path
        ) as zipped:
            metadata = _metadata(archive, expected_format=expected_format)
            cases = [
                _object(raw_case, case_fields, f"{expected_format} case")
                for raw_case in metadata["cases"]
            ]
            expected = {"metadata"}
            case_routes: list[str] = []
            case_strata: list[str] = []
            route_set = set(routes)
            for index, case in enumerate(cases):
                route = _string(case["route"], "cache route")
                stratum = _string(case["stratum"], "cache stratum")
                if route not in route_set:
                    raise CacheLoadError("cache references an unknown selector route")
                if stratum not in strata:
                    raise CacheLoadError("cache references an unknown scoring stratum")
                case_routes.append(route)
                case_strata.append(stratum)
                if "epsilon" in case:
                    epsilon = case["epsilon"]
                    if (
                        isinstance(epsilon, bool)
                        or not isinstance(epsilon, (int, float))
                        or not isfinite(float(epsilon))
                        or epsilon <= 0
                    ):
                        raise CacheLoadError("exact case epsilon must be finite and positive")
                prefix = f"case_{index:06d}"
                expected.update(f"{prefix}_{field}" for field in array_fields)
                _validate_case_headers(
                    zipped,
                    prefix=prefix,
                    expected_format=expected_format,
                    head_dim=head_dim,
                )
                valid_keys = _array(archive, f"{prefix}_valid_keys")
                required_keys = _array(archive, f"{prefix}_required_keys")
                if (
                    valid_keys.ndim != 1
                    or valid_keys.dtype != np.bool_
                    or np.count_nonzero(valid_keys) < max_blocks
                ):
                    raise CacheLoadError(
                        "cache case cannot satisfy the maximum block budget"
                    )
                if (
                    required_keys.dtype != np.bool_
                    or np.any(required_keys & ~valid_keys[None, :])
                    or np.any(required_keys.sum(axis=1) > min_blocks)
                ):
                    raise CacheLoadError("cache required-key mask is invalid")
            if sample_count is not None and len(cases) != sample_count:
                raise CacheLoadError("cache case count does not match the track")
            sampled_routes = set(case_routes)
            if sample_count is not None and len(sampled_routes) != sample_count // len(strata):
                raise CacheLoadError("cache route sample count does not match the track")
            expected_pairs = {
                (route, stratum) for route in sampled_routes for stratum in strata
            }
            actual_pairs = set(zip(case_routes, case_strata, strict=True))
            if len(actual_pairs) != len(cases) or actual_pairs != expected_pairs:
                raise CacheLoadError(
                    "cache must cover every sampled route across every stratum exactly once"
                )
            _exact_keys(archive, expected)
    except CacheLoadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CacheLoadError("cache index cannot be read") from exc


def _verify_archive(path: str | Path, expected_sha256: str, max_bytes: int) -> Path:
    cache_path = Path(path)
    if max_bytes <= 0:
        raise ValueError("cache byte limit must be positive")
    if len(expected_sha256) != 64:
        raise ValueError("cache digest must contain 32 bytes")
    try:
        expected = bytes.fromhex(expected_sha256)
    except ValueError as exc:
        raise ValueError("cache digest must be hexadecimal") from exc

    digest = hashlib.sha256()
    try:
        with cache_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.digest() != expected:
            raise CacheLoadError("cache sha256 does not match the track specification")
        with zipfile.ZipFile(cache_path) as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                raise CacheLoadError("cache archive contains duplicate members")
            decoded_bytes = sum(item.file_size for item in members)
            if decoded_bytes > max_bytes:
                raise CacheLoadError("cache decoded size exceeds the track limit")
    except CacheLoadError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise CacheLoadError("cache is not a readable NPZ archive") from exc
    return cache_path


def _metadata(archive: Any, *, expected_format: str) -> dict[str, Any]:
    raw = _array(archive, "metadata")
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise CacheLoadError("cache metadata must be a byte vector")
    try:
        metadata = json.loads(raw.tobytes().decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheLoadError("cache metadata is not valid JSON") from exc
    root = _object(
        metadata,
        {"format", "generated_block", "source_digest", "cases"},
        "cache metadata",
    )
    if root["format"] != expected_format:
        raise CacheLoadError("cache format identifier is invalid")
    if not isinstance(root["cases"], list) or not root["cases"]:
        raise CacheLoadError("cache must contain at least one case")
    return root


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CacheLoadError("cache is not readable") from exc
    return digest.hexdigest()


def _array(archive: Any, name: str) -> np.ndarray:
    try:
        value = np.array(archive[name], copy=True)
    except KeyError as exc:
        raise CacheLoadError(f"cache array is missing: {name}") from exc
    if value.dtype.hasobject:
        raise CacheLoadError("cache arrays cannot contain objects")
    value.setflags(write=False)
    return value


def _validate_case_headers(
    archive: zipfile.ZipFile,
    *,
    prefix: str,
    expected_format: str,
    head_dim: int,
) -> None:
    if expected_format == "veda-exact-cache":
        q_shape, q_dtype = _array_header(archive, f"{prefix}_q")
        k_shape, k_dtype = _array_header(archive, f"{prefix}_k")
        v_shape, v_dtype = _array_header(archive, f"{prefix}_v")
        dense_shape, dense_dtype = _array_header(archive, f"{prefix}_dense")
        valid_shape, valid_dtype = _array_header(archive, f"{prefix}_valid_keys")
        required_shape, required_dtype = _array_header(archive, f"{prefix}_required_keys")
        valid = (
            len(q_shape) == 3
            and len(k_shape) == 3
            and all(dimension > 0 for dimension in (*q_shape, *k_shape))
            and q_shape[-1] == head_dim
            and k_shape[-1] == head_dim
            and v_shape == k_shape
            and dense_shape == q_shape
            and valid_shape == (k_shape[0],)
            and required_shape == (q_shape[0], k_shape[0])
            and _floating(q_dtype)
            and _floating(k_dtype)
            and _floating(v_dtype)
            and _floating(dense_dtype)
            and valid_dtype == np.dtype(np.bool_)
            and required_dtype == np.dtype(np.bool_)
        )
    else:
        raise AssertionError("unsupported cache format")
    if not valid:
        raise CacheLoadError("cache case array headers are incompatible with the track")


def _array_header(archive: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], np.dtype]:
    member_name = f"{name}.npy"
    try:
        member = archive.getinfo(member_name)
        with archive.open(member) as file:
            version = read_magic(file)
            if version == (1, 0):
                shape, _fortran_order, dtype = read_array_header_1_0(file)
            elif version == (2, 0):
                shape, _fortran_order, dtype = read_array_header_2_0(file)
            else:
                raise CacheLoadError("cache array uses an unsupported NPY format")
            data_offset = file.tell()
    except CacheLoadError:
        raise
    except (KeyError, OSError, ValueError, EOFError) as exc:
        raise CacheLoadError("cache array header cannot be read") from exc
    if dtype.hasobject:
        raise CacheLoadError("cache arrays cannot contain objects")
    if member.file_size - data_offset != prod(shape) * dtype.itemsize:
        raise CacheLoadError("cache array payload does not match its header")
    return shape, dtype


def _floating(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def _exact_keys(archive: Any, expected: set[str]) -> None:
    if set(archive.files) != expected:
        raise CacheLoadError("cache contains missing or unexpected arrays")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheLoadError(f"duplicate cache metadata field: {key}")
        result[key] = value
    return result


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CacheLoadError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheLoadError(f"{name} must be a non-empty string")
    return value
