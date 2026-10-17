import pytest

from scoring.generator import generate_sources, plan_batch, scheduled_seed_block, source_digest


def test_scheduled_seed_block_uses_only_fixed_boundaries() -> None:
    assert (
        scheduled_seed_block(
            100,
            track_start_block=100,
            minimum_delay_blocks=1,
            generation_interval_blocks=20,
        )
        == 120
    )
    assert (
        scheduled_seed_block(
            119,
            track_start_block=100,
            minimum_delay_blocks=1,
            generation_interval_blocks=20,
        )
        == 120
    )
    assert (
        scheduled_seed_block(
            120,
            track_start_block=100,
            minimum_delay_blocks=1,
            generation_interval_blocks=20,
        )
        == 140
    )


def test_batch_plan_is_reproducible_complete_and_seeded() -> None:
    arguments = {
        "track_spec_digest": "12" * 32,
        "routes": ("r0", "r1", "r2"),
        "samples_by_stratum": {
            "landscape": ("l0", "l1", "l2"),
            "portrait": ("p0", "p1", "p2"),
        },
        "sample_count": 4,
    }
    first = plan_batch(seed_block_hash="0x" + "ab" * 32, **arguments)
    second = plan_batch(seed_block_hash="0x" + "ab" * 32, **arguments)
    changed = plan_batch(seed_block_hash="0x" + "cd" * 32, **arguments)

    assert first == second
    assert first != changed
    assert len(first) == 4
    routes = {case.route for case in first}
    assert len(routes) == 2
    assert {(case.route, case.stratum) for case in first} == {
        (route, stratum) for route in routes for stratum in ("landscape", "portrait")
    }


def test_batch_plan_rejects_insufficient_source_pool() -> None:
    with pytest.raises(ValueError, match="not enough samples"):
        plan_batch(
            track_spec_digest="12" * 32,
            seed_block_hash="34" * 32,
            routes=("r0", "r1"),
            samples_by_stratum={"all": ("only",)},
            sample_count=2,
        )


def test_source_generation_is_fresh_reproducible_and_auditable() -> None:
    arguments = {
        "track_spec_digest": "12" * 32,
        "strata": ("landscape", "portrait"),
        "samples_per_stratum": 2,
    }
    first = generate_sources(seed_block_hash="ab" * 32, **arguments)
    repeated = generate_sources(seed_block_hash="ab" * 32, **arguments)
    changed = generate_sources(seed_block_hash="cd" * 32, **arguments)

    assert first == repeated
    assert first != changed
    assert len({sample.sample_id for sample in first}) == 4
    assert source_digest(first) == source_digest(repeated)
    assert source_digest(first) != source_digest(changed)
