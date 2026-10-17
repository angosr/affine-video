import pytest

from scoring.video import VideoCell, paired_video_gain_bound


def test_video_gain_bootstrap_keeps_budgets_inside_rollout_cluster() -> None:
    challenger = tuple(
        VideoCell(stratum, rollout, budget, score)
        for stratum, rollout, score in (
            ("wide", "w0", 0.8),
            ("wide", "w1", 0.7),
            ("wide", "w2", 0.8),
            ("wide", "w3", 0.7),
            ("square", "s0", 0.6),
            ("square", "s1", 0.5),
            ("square", "s2", 0.6),
            ("square", "s3", 0.5),
        )
        for budget in (16, 32)
    )
    incumbent = tuple(
        VideoCell(row.stratum, row.rollout_id, row.budget, row.score - 0.1) for row in challenger
    )

    result = paired_video_gain_bound(
        challenger,
        incumbent,
        stratum_weights={"wide": 0.5, "square": 0.5},
        budget_costs={16: 1.0, 32: 2.0},
        confidence=0.95,
        resamples=100,
        seed=7,
    )

    assert result.challenger_score == pytest.approx(0.65)
    assert result.incumbent_score == pytest.approx(0.55)
    assert result.gain_lower_bound == pytest.approx(0.1)


def test_video_gain_rejects_incomplete_budget_crossing() -> None:
    challenger = (VideoCell("wide", "w0", 16, 0.8),)
    incumbent = (VideoCell("wide", "w0", 32, 0.7),)

    with pytest.raises(ValueError, match="align"):
        paired_video_gain_bound(
            challenger,
            incumbent,
            stratum_weights={"wide": 1.0},
            budget_costs={16: 1.0, 32: 2.0},
            confidence=0.95,
            resamples=10,
            seed=0,
        )


def test_video_gain_rejects_too_few_rollouts_per_stratum() -> None:
    challenger = tuple(VideoCell("wide", f"w{index}", 16, 0.8) for index in range(3))
    incumbent = tuple(VideoCell(row.stratum, row.rollout_id, row.budget, 0.7) for row in challenger)

    with pytest.raises(ValueError, match="four rollouts"):
        paired_video_gain_bound(
            challenger,
            incumbent,
            stratum_weights={"wide": 1.0},
            budget_costs={16: 1.0},
            confidence=0.95,
            resamples=10,
            seed=0,
        )


def test_video_score_integrates_quality_on_the_calibrated_cost_axis() -> None:
    challenger = tuple(
        VideoCell("wide", f"w{index}", budget, quality)
        for index in range(4)
        for budget, quality in ((16, 0.2), (32, 0.8), (64, 0.2))
    )
    incumbent = tuple(VideoCell(row.stratum, row.rollout_id, row.budget, 0.1) for row in challenger)

    result = paired_video_gain_bound(
        challenger,
        incumbent,
        stratum_weights={"wide": 1.0},
        budget_costs={16: 1.0, 32: 2.0, 64: 4.0},
        confidence=0.95,
        resamples=100,
        seed=0,
    )

    assert result.challenger_score == pytest.approx(0.5)
    assert result.incumbent_score == pytest.approx(0.1)
    assert result.gain_lower_bound == pytest.approx(0.4)
