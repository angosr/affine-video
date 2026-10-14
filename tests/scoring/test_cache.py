import hashlib
import json

import numpy as np
import pytest

from scoring.cache import (
    CacheLoadError,
    load_exact_cache,
    validate_exact_cache,
)


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(value: dict) -> np.ndarray:
    return np.frombuffer(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
    )


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
        case_000000_required_keys=np.array(
            [[True, False, False], [False, True, False]], dtype=np.bool_
        ),
    )

    cases = load_exact_cache(path, expected_sha256=digest(path), max_bytes=100_000)
    validate_exact_cache(
        path,
        expected_sha256=digest(path),
        max_bytes=100_000,
        min_blocks=1,
        max_blocks=3,
        head_dim=2,
        routes=("route0",),
        strata={"all": 1.0},
        sample_count=1,
    )

    assert len(cases) == 1
    assert cases[0].epsilon == 1e-6
    with pytest.raises(CacheLoadError, match="maximum block budget"):
        validate_exact_cache(
            path,
            expected_sha256=digest(path),
            max_bytes=100_000,
            min_blocks=1,
            max_blocks=4,
            head_dim=2,
            routes=("route0",),
            strata={"all": 1.0},
            sample_count=1,
        )
    with pytest.raises(CacheLoadError, match="array headers"):
        validate_exact_cache(
            path,
            expected_sha256=digest(path),
            max_bytes=100_000,
            min_blocks=1,
            max_blocks=3,
            head_dim=3,
            routes=("route0",),
            strata={"all": 1.0},
            sample_count=1,
        )
    with pytest.raises(CacheLoadError, match="sha256"):
        load_exact_cache(path, expected_sha256="00" * 32, max_bytes=100_000)
    with pytest.raises(CacheLoadError, match="decoded size"):
        load_exact_cache(path, expected_sha256=digest(path), max_bytes=1)


def test_exact_cache_rejects_unknown_route_and_insufficient_cases(tmp_path) -> None:
    path = tmp_path / "exact.npz"
    np.savez(
        path,
        metadata=metadata(
            {
                "format": "veda-exact-cache",
                "cases": [{"route": "route0", "stratum": "all", "epsilon": 1e-6}],
            }
        ),
        case_000000_q=np.ones((1, 2, 2), dtype=np.float32),
        case_000000_k=np.ones((2, 2, 2), dtype=np.float32),
        case_000000_v=np.ones((2, 2, 2), dtype=np.float32),
        case_000000_dense=np.ones((1, 2, 2), dtype=np.float32),
        case_000000_valid_keys=np.ones(2, dtype=np.bool_),
        case_000000_required_keys=np.array([[True, False]], dtype=np.bool_),
    )
    arguments = {
        "expected_sha256": digest(path),
        "max_bytes": 100_000,
        "min_blocks": 1,
        "max_blocks": 1,
        "head_dim": 2,
        "strata": {"all": 1.0},
    }

    with pytest.raises(CacheLoadError, match="unknown selector route"):
        validate_exact_cache(path, routes=("other",), sample_count=1, **arguments)
    with pytest.raises(CacheLoadError, match="every route"):
        validate_exact_cache(path, routes=("route0", "missing"), sample_count=2, **arguments)
    with pytest.raises(CacheLoadError, match="cover every"):
        validate_exact_cache(
            path,
            routes=("route0",),
            expected_sha256=digest(path),
            max_bytes=100_000,
            min_blocks=1,
            max_blocks=1,
            head_dim=2,
            strata={"all": 0.5, "missing": 0.5},
            sample_count=2,
        )
