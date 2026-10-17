import hashlib
import json

import numpy as np
import pytest

from scoring.cache import (
    CacheLoadError,
    inspect_exact_cache_batch,
    load_exact_cache,
    select_exact_cache_batch,
    validate_exact_cache,
    write_exact_cache_batch,
)
from scoring.exact import ExactCase


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
                "generated_block": 120,
                "source_digest": "12" * 32,
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
    batch = inspect_exact_cache_batch(path, max_bytes=100_000)
    assert batch.generated_block == 120
    assert batch.source_digest == "12" * 32
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
                "generated_block": 120,
                "source_digest": "12" * 32,
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


def test_cache_batch_must_be_generated_after_submission(tmp_path) -> None:
    temporary = tmp_path / "temporary"
    np.savez(
        temporary,
        metadata=metadata(
            {
                "format": "veda-exact-cache",
                "generated_block": 120,
                "source_digest": "34" * 32,
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
    temporary = temporary.with_suffix(".npz")
    path = tmp_path / f"120-{digest(temporary)}.npz"
    temporary.rename(path)
    arguments = {
        "directory": tmp_path,
        "minimum_delay_blocks": 1,
        "max_bytes": 100_000,
        "min_blocks": 1,
        "max_blocks": 1,
        "head_dim": 2,
        "routes": ("route0",),
        "strata": {"all": 1.0},
        "sample_count": 1,
    }

    selected = select_exact_cache_batch(submission_block=119, **arguments)
    assert selected.path == path
    with pytest.raises(CacheLoadError, match="after the submission"):
        select_exact_cache_batch(submission_block=120, **arguments)


def test_cache_batch_writer_is_atomic_and_self_identifying(tmp_path) -> None:
    case = ExactCase(
        "route0",
        "all",
        np.ones((1, 2, 2), dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones((1, 2, 2), dtype=np.float32),
        np.ones(2, dtype=np.bool_),
        np.array([[True, False]], dtype=np.bool_),
        1e-6,
    )

    batch = write_exact_cache_batch(
        tmp_path,
        (case,),
        generated_block=200,
        source_digest="56" * 32,
    )

    assert batch.path.name == f"200-{batch.sha256}.npz"
    assert sorted(path.name for path in tmp_path.iterdir()) == [batch.path.name]
    assert inspect_exact_cache_batch(batch.path, max_bytes=100_000) == batch
