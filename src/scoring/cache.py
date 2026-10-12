"""Verified loading of frozen lightweight and exact scoring caches."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from scoring.evaluate import route_stratified_sample_indices
from scoring.local import ExactCase, ScreenCase


class CacheLoadError(ValueError):
    """Raised when a frozen scoring cache is missing, changed, or malformed."""


def validate_screen_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
) -> None:
    """Validate the lightweight cache index without materializing case arrays."""

    _validate_cache_index(
        path,
        expected_sha256=expected_sha256,
        max_bytes=max_bytes,
        expected_format="veda-screen-cache",
        case_fields={"route", "stratum"},
        array_fields=("q_descriptors", "k_descriptors", "teacher_distribution", "valid_keys"),
        routes=routes,
        strata=strata,
        sample_count=None,
    )


def validate_exact_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
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
        array_fields=("q", "k", "v", "dense", "valid_keys"),
        routes=routes,
        strata=strata,
        sample_count=sample_count,
    )


def load_screen_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[ScreenCase, ...]:
    """Load a hash-pinned lightweight cache without pickle support."""
    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = _metadata(archive, expected_format="veda-screen-cache")
            cases = [
                _object(raw_case, {"route", "stratum"}, "screen case")
                for raw_case in metadata["cases"]
            ]
            expected = {"metadata"}
            loaded: list[ScreenCase] = []
            for index, case in enumerate(cases):
                prefix = f"case_{index:06d}"
                names = {
                    "q": f"{prefix}_q_descriptors",
                    "k": f"{prefix}_k_descriptors",
                    "teacher": f"{prefix}_teacher_distribution",
                    "valid": f"{prefix}_valid_keys",
                }
                expected.update(names.values())
                loaded.append(
                    ScreenCase(
                        route=_string(case["route"], "screen route"),
                        stratum=_string(case["stratum"], "screen stratum"),
                        q_descriptors=_array(archive, names["q"]),
                        k_descriptors=_array(archive, names["k"]),
                        teacher_distribution=_array(archive, names["teacher"]),
                        valid_keys=_array(archive, names["valid"]),
                    )
                )
            _exact_keys(archive, expected)
    except CacheLoadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CacheLoadError("screen cache arrays cannot be read") from exc
    return tuple(loaded)


def load_exact_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[ExactCase, ...]:
    """Load a hash-pinned exact QKV cache without pickle support."""

    return _load_exact_cache(path, expected_sha256, max_bytes)


def sample_exact_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    count: int,
    seed: int,
    routes: tuple[str, ...],
    strata: Mapping[str, float],
) -> tuple[ExactCase, ...]:
    """Load only the deterministic exact sample needed by one challenge."""

    return _load_exact_cache(
        path,
        expected_sha256,
        max_bytes,
        count=count,
        seed=seed,
        routes=routes,
        strata=strata,
    )


def _load_exact_cache(
    path: str | Path,
    expected_sha256: str,
    max_bytes: int,
    *,
    count: int | None = None,
    seed: int = 0,
    routes: tuple[str, ...] | None = None,
    strata: Mapping[str, float] | None = None,
) -> tuple[ExactCase, ...]:
    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = _metadata(archive, expected_format="veda-exact-cache")
            cases = [
                _object(raw_case, {"route", "stratum", "epsilon"}, "exact case")
                for raw_case in metadata["cases"]
            ]
            indices = _selected_indices(
                cases,
                count=count,
                seed=seed,
                routes=routes,
                strata=strata,
            )
            selected = set(indices)
            expected = {"metadata"}
            loaded: dict[int, ExactCase] = {}
            for index, case in enumerate(cases):
                prefix = f"case_{index:06d}"
                names = {
                    field: f"{prefix}_{field}" for field in ("q", "k", "v", "dense", "valid_keys")
                }
                expected.update(names.values())
                epsilon = case["epsilon"]
                if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
                    raise CacheLoadError("exact case epsilon must be numeric")
                if index in selected:
                    loaded[index] = ExactCase(
                        route=_string(case["route"], "exact route"),
                        stratum=_string(case["stratum"], "exact stratum"),
                        q=_array(archive, names["q"]),
                        k=_array(archive, names["k"]),
                        v=_array(archive, names["v"]),
                        dense=_array(archive, names["dense"]),
                        valid_keys=_array(archive, names["valid_keys"]),
                        epsilon=float(epsilon),
                    )
            _exact_keys(archive, expected)
    except CacheLoadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CacheLoadError("exact cache arrays cannot be read") from exc
    return tuple(loaded[index] for index in indices)


def _selected_indices(
    cases: list[dict[str, Any]],
    *,
    count: int | None,
    seed: int,
    routes: tuple[str, ...] | None,
    strata: Mapping[str, float] | None,
) -> tuple[int, ...]:
    if count is None:
        return tuple(range(len(cases)))
    if routes is None or strata is None:
        raise ValueError("routes and strata are required for sampled cache loading")
    case_routes = tuple(_string(case["route"], "case route") for case in cases)
    case_strata = tuple(_string(case["stratum"], "case stratum") for case in cases)
    return route_stratified_sample_indices(
        case_routes,
        case_strata,
        count,
        seed,
        routes,
        strata,
    )


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
) -> None:
    cache_path = _verify_archive(path, expected_sha256, max_bytes)
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
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
            coverage_count = len(routes) * len(strata) if sample_count is None else sample_count
            try:
                route_stratified_sample_indices(
                    case_routes,
                    case_strata,
                    coverage_count,
                    0,
                    routes,
                    strata,
                )
            except ValueError as exc:
                raise CacheLoadError(
                    "cache does not cover every route and frozen stratum"
                ) from exc
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
            decoded_bytes = sum(item.file_size for item in archive.infolist())
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
    root = _object(metadata, {"format", "cases"}, "cache metadata")
    if root["format"] != expected_format:
        raise CacheLoadError("cache format identifier is invalid")
    if not isinstance(root["cases"], list) or not root["cases"]:
        raise CacheLoadError("cache must contain at least one case")
    return root


def _array(archive: Any, name: str) -> np.ndarray:
    try:
        value = np.array(archive[name], copy=True)
    except KeyError as exc:
        raise CacheLoadError(f"cache array is missing: {name}") from exc
    if value.dtype.hasobject:
        raise CacheLoadError("cache arrays cannot contain objects")
    value.setflags(write=False)
    return value


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
