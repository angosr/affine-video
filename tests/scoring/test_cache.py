import hashlib
import json

import numpy as np
import pytest

from scoring import cache as cache_module
from scoring.cache import (
    CacheLoadError,
    load_exact_cache,
    load_screen_cache,
    sample_screen_cache,
    validate_exact_cache,
    validate_screen_cache,
)


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(value: dict) -> np.ndarray:
    return np.frombuffer(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
    )


def test_loads_hash_pinned_screen_cache_without_objects(tmp_path) -> None:
    path = tmp_path / "screen.npz"
    np.savez(
        path,
        metadata=metadata(
            {
                "format": "veda-screen-cache",
                "cases": [{"route": "route0", "stratum": "all"}],
            }
        ),
        case_000000_q_descriptors=np.ones((2, 6), dtype=np.float32),
        case_000000_k_descriptors=np.ones((3, 6), dtype=np.float32),
        case_000000_teacher_mass=np.full((2, 3), 1 / 3, dtype=np.float32),
        case_000000_valid_keys=np.ones(3, dtype=np.bool_),
    )

    cases = load_screen_cache(path, expected_sha256=digest(path), max_bytes=100_000)
    validate_screen_cache(
        path,
        expected_sha256=digest(path),
        max_bytes=100_000,
        routes=("route0",),
        strata={"all": 1.0},
        sample_count=1,
    )

    assert len(cases) == 1
    assert cases[0].route == "route0"
    assert cases[0].stratum == "all"
    assert cases[0].q_descriptors.flags.writeable is False


def test_sampled_screen_cache_only_materializes_selected_cases(tmp_path, monkeypatch) -> None:
    path = tmp_path / "screen.npz"
    arrays = {
        "metadata": metadata(
            {
                "format": "veda-screen-cache",
                "cases": [
                    {"route": "route0", "stratum": "a"},
                    {"route": "route0", "stratum": "a"},
                    {"route": "route0", "stratum": "b"},
                ],
            }
        )
    }
    for index in range(3):
        prefix = f"case_{index:06d}"
        arrays[f"{prefix}_q_descriptors"] = np.full((1, 2), index, dtype=np.float32)
        arrays[f"{prefix}_k_descriptors"] = np.full((2, 2), index, dtype=np.float32)
        arrays[f"{prefix}_teacher_mass"] = np.full((1, 2), 0.5, dtype=np.float32)
        arrays[f"{prefix}_valid_keys"] = np.ones(2, dtype=np.bool_)
    np.savez(path, **arrays)

    accessed: list[str] = []
    original = cache_module._array

    def recording_array(archive, name):
        accessed.append(name)
        return original(archive, name)

    monkeypatch.setattr(cache_module, "_array", recording_array)
    first = sample_screen_cache(
        path,
        expected_sha256=digest(path),
        max_bytes=100_000,
        count=2,
        seed=9,
        strata={"a": 0.5, "b": 0.5},
    )
    second = sample_screen_cache(
        path,
        expected_sha256=digest(path),
        max_bytes=100_000,
        count=2,
        seed=9,
        strata={"a": 0.5, "b": 0.5},
    )

    assert [case.stratum for case in first] == ["a", "b"]
    assert [case.q_descriptors.tolist() for case in first] == [
        case.q_descriptors.tolist() for case in second
    ]
    assert len(accessed) == 2 * (1 + 2 * 4)


def test_loads_exact_cache_and_rejects_digest_or_decoded_size(tmp_path) -> None:
    path = tmp_path / "exact.npz"
    np.savez(
        path,
        metadata=metadata(
            {
                "format": "veda-exact-cache",
                "cases": [{"route": "route0", "stratum": "all", "epsilon": 1e-6}],
            }
        ),
        case_000000_q=np.ones((2, 2, 2), dtype=np.float32),
        case_000000_k=np.ones((3, 2, 2), dtype=np.float32),
        case_000000_v=np.ones((3, 2, 2), dtype=np.float32),
        case_000000_dense=np.ones((2, 2, 2), dtype=np.float32),
        case_000000_valid_keys=np.ones(3, dtype=np.bool_),
    )

    cases = load_exact_cache(path, expected_sha256=digest(path), max_bytes=100_000)
    validate_exact_cache(
        path,
        expected_sha256=digest(path),
        max_bytes=100_000,
        routes=("route0",),
        strata={"all": 1.0},
        sample_count=1,
    )

    assert len(cases) == 1
    assert cases[0].epsilon == 1e-6
    with pytest.raises(CacheLoadError, match="sha256"):
        load_exact_cache(path, expected_sha256="00" * 32, max_bytes=100_000)
    with pytest.raises(CacheLoadError, match="decoded size"):
        load_exact_cache(path, expected_sha256=digest(path), max_bytes=1)


def test_cache_index_rejects_unknown_route_and_insufficient_cases(tmp_path) -> None:
    path = tmp_path / "screen.npz"
    np.savez(
        path,
        metadata=metadata(
            {
                "format": "veda-screen-cache",
                "cases": [{"route": "route0", "stratum": "all"}],
            }
        ),
        case_000000_q_descriptors=np.ones((1, 2), dtype=np.float32),
        case_000000_k_descriptors=np.ones((2, 2), dtype=np.float32),
        case_000000_teacher_mass=np.full((1, 2), 0.5, dtype=np.float32),
        case_000000_valid_keys=np.ones(2, dtype=np.bool_),
    )
    arguments = {
        "expected_sha256": digest(path),
        "max_bytes": 100_000,
        "strata": {"all": 1.0},
    }

    with pytest.raises(CacheLoadError, match="unknown selector route"):
        validate_screen_cache(path, routes=("other",), sample_count=1, **arguments)
    with pytest.raises(CacheLoadError, match="index cannot be read"):
        validate_screen_cache(path, routes=("route0",), sample_count=2, **arguments)
