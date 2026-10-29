import hashlib
from dataclasses import replace

import numpy as np
import pytest

from scoring.cache import (
    CacheLoadError,
    ExactCacheReader,
    assemble_exact_cache_batch,
    inspect_exact_cache_batch,
    validate_exact_cache,
    write_exact_cache_batch,
)
from scoring.evaluate import reference_attention
from scoring.exact import ExactCase


def exact_case(route: str, rollout: str, step: int) -> ExactCase:
    generator = np.random.default_rng(step + int(rollout[-1]))
    q = generator.normal(size=(2, 2, 2)).astype(np.float16)
    k = generator.normal(size=(3, 2, 2)).astype(np.float16)
    v = generator.normal(size=(3, 2, 2)).astype(np.float16)
    dense = reference_attention(q, k, v, np.ones((2, 3), dtype=np.bool_))
    return ExactCase(
        route=route,
        stratum="all",
        rollout_id=rollout,
        source_id=rollout.replace("rollout", "source"),
        step_index=step,
        step_window="early" if step == 0 else "late",
        scheduler_timestep=float(step),
        sigma=1.0 - step * 0.5,
        tile_plan_hash="12" * 32,
        query_tile_ids=(0, 1),
        q=q,
        k=k,
        v=v,
        dense=dense,
        query_valid_counts=np.full(2, 2, dtype=np.int16),
        key_valid_counts=np.full(3, 2, dtype=np.int16),
        key_segments=np.zeros(3, dtype=np.int8),
        required_keys=np.array([[True, False, False], [False, True, False]], dtype=np.bool_),
        epsilon=1e-6,
    )


def write_batch(tmp_path):
    cases = tuple(
        exact_case(route, rollout, step)
        for route in ("route-0", "route-1")
        for rollout in ("rollout-0", "rollout-1")
        for step in (0, 1)
    )
    return write_exact_cache_batch(
        tmp_path,
        cases,
        generated_block=120,
        seed_block_hash="0x" + "ab" * 32,
        plan_digest="23" * 32,
        track_spec_digest="34" * 32,
        artifact_manifest_sha256="45" * 32,
        source_pool_merkle_root="56" * 32,
        source_proofs=({"source_id": "source-0"},),
        dense_references=({"video_sha256": "67" * 32},),
        sample_cache={"index_sha256": "78" * 32},
    )


def validation_arguments(batch):
    return {
        "expected_sha256": batch.sha256,
        "expected_track_spec_digest": "34" * 32,
        "max_bytes": 1_000_000,
        "head_dim": 2,
        "tile_size": 2,
        "routes": ("route-0", "route-1", "route-2"),
        "strata": {"all": 1.0},
        "sampled_route_count": 2,
        "rollouts_per_stratum": 2,
        "sampled_step_count": 2,
        "query_tiles_per_case": 2,
    }


def test_cache_v2_is_self_identifying_and_streams_cases(tmp_path) -> None:
    batch = write_batch(tmp_path)

    validate_exact_cache(batch.path, **validation_arguments(batch))
    inspected = inspect_exact_cache_batch(batch.path, max_bytes=1_000_000)
    assert inspected == batch
    assert batch.path.name == f"120-{batch.sha256}.npz"
    with ExactCacheReader(
        batch.path,
        expected_sha256=batch.sha256,
        max_bytes=1_000_000,
    ) as reader:
        cases = list(reader)
    assert len(cases) == 8
    assert all(case.q.dtype == np.float16 for case in cases)
    assert cases[0].route == "route-0"


def test_cache_rejects_wrong_digest_track_or_decoded_limit(tmp_path) -> None:
    batch = write_batch(tmp_path)
    arguments = validation_arguments(batch)

    with pytest.raises(CacheLoadError, match="different track"):
        validate_exact_cache(
            batch.path,
            **{**arguments, "expected_track_spec_digest": "00" * 32},
        )
    with pytest.raises(CacheLoadError, match="SHA-256"):
        ExactCacheReader(batch.path, expected_sha256="00" * 32, max_bytes=1_000_000)
    with pytest.raises(CacheLoadError, match="decoded size"):
        ExactCacheReader(batch.path, expected_sha256=batch.sha256, max_bytes=1)


def test_cache_rejects_rehashed_dense_values_that_do_not_match_qkv(tmp_path) -> None:
    cases = tuple(
        exact_case(route, rollout, step)
        for route in ("route-0", "route-1")
        for rollout in ("rollout-0", "rollout-1")
        for step in (0, 1)
    )
    cases = tuple(replace(case, dense=case.dense + 100) for case in cases)
    batch = write_exact_cache_batch(
        tmp_path,
        cases,
        generated_block=120,
        seed_block_hash="0x" + "ab" * 32,
        plan_digest="23" * 32,
        track_spec_digest="34" * 32,
        artifact_manifest_sha256="45" * 32,
        source_pool_merkle_root="56" * 32,
        source_proofs=({"source_id": "source-0"},),
        dense_references=({"video_sha256": "67" * 32},),
        sample_cache={"index_sha256": "78" * 32},
    )

    with pytest.raises(CacheLoadError, match="dense reference"):
        validate_exact_cache(batch.path, **validation_arguments(batch))


def test_hardware_reference_rounding_keeps_original_archive_and_scoring_dense(
    tmp_path, monkeypatch,
):
    import scoring.exact as module

    batch = write_batch(tmp_path)
    before = batch.path.read_bytes()
    canonical = module.canonical_dense_reference
    with ExactCacheReader(batch.path, expected_sha256=batch.sha256, max_bytes=1_000_000) as reader:
        original_dense = [case.dense.tobytes() for case in reader]

    def alternate(*args, **kwargs):
        assert kwargs.pop("device") == "cuda:0"  # A CPU fixture, not an actual CUDA call.
        dense = canonical(*args, **kwargs)
        return np.nextafter(dense, np.full_like(dense, np.inf))

    monkeypatch.setattr(module, "canonical_dense_reference", alternate)
    validate_exact_cache(batch.path, **validation_arguments(batch), dense_device="cuda:0")
    assert batch.path.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == batch.sha256
    with ExactCacheReader(batch.path, expected_sha256=batch.sha256, max_bytes=1_000_000) as reader:
        assert [case.dense.tobytes() for case in reader] == original_dense


def test_one_bin_dense_change_still_requires_the_exact_original_archive_sha(tmp_path):
    batch = write_batch(tmp_path)
    with np.load(batch.path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    dense = arrays["case_000000_dense"]
    dense.flat[0] = np.nextafter(dense.flat[0], np.float16(np.inf))
    altered = tmp_path / "altered.npz"
    np.savez(altered, **arrays)
    with pytest.raises(CacheLoadError, match="SHA-256"):
        validate_exact_cache(altered, **validation_arguments(batch))




def test_written_file_digest_matches_batch_identity(tmp_path) -> None:
    batch = write_batch(tmp_path)
    digest = hashlib.sha256(batch.path.read_bytes()).hexdigest()
    assert digest == batch.sha256


def test_rollout_shards_assemble_without_loading_the_complete_batch(tmp_path) -> None:
    cases = tuple(
        exact_case(route, rollout, step)
        for route in ("route-0", "route-1")
        for rollout in ("rollout-0", "rollout-1")
        for step in (0, 1)
    )
    common = {
        "seed_block_hash": "0x" + "ab" * 32,
        "plan_digest": "23" * 32,
        "track_spec_digest": "34" * 32,
        "artifact_manifest_sha256": "45" * 32,
        "source_pool_merkle_root": "56" * 32,
        "source_proofs": (),
        "dense_references": (),
        "sample_cache": {"index_sha256": "78" * 32},
    }
    first = write_exact_cache_batch(tmp_path / "first", cases[:4], generated_block=120, **common)
    second = write_exact_cache_batch(tmp_path / "second", cases[4:], generated_block=120, **common)

    assembled = assemble_exact_cache_batch(
        tmp_path / "assembled",
        (first.path, second.path),
        generated_block=120,
        **common,
    )

    validate_exact_cache(assembled.path, **validation_arguments(assembled))
    with ExactCacheReader(
        assembled.path, expected_sha256=assembled.sha256, max_bytes=1_000_000
    ) as reader:
        assert len(list(reader)) == 8
