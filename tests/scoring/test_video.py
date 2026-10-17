import numpy as np
import pytest

from scoring.video import (
    VideoCell,
    VideoFidelityMetric,
    paired_video_gain_bound,
    trajectory_latent_quality,
)


def test_trajectory_score_distinguishes_equal_final_states() -> None:
    dense = {
        (0, "early"): np.ones((2, 3), dtype=np.float32),
        (1, "late"): np.ones((2, 3), dtype=np.float32),
    }
    matching = {key: value.copy() for key, value in dense.items()}
    drifting = {key: value.copy() for key, value in dense.items()}
    drifting[(0, "early")] = np.zeros((2, 3), dtype=np.float32)

    identity = trajectory_latent_quality(
        dense,
        matching,
        step_window_weights={"early": 0.5, "late": 0.5},
    )
    changed = trajectory_latent_quality(
        dense,
        drifting,
        step_window_weights={"early": 0.5, "late": 0.5},
    )

    assert identity == pytest.approx(1.0)
    assert changed == pytest.approx(0.75)


def test_non_finite_sparse_latents_receive_zero_fidelity() -> None:
    metric = object.__new__(VideoFidelityMetric)
    dense_rows = np.ones((2, 3), dtype=np.float32)
    sparse_rows = dense_rows.copy()
    sparse_rows[0, 0] = np.nan
    frames = np.ones((2, 2, 2, 3), dtype=np.uint8)

    result = metric.score(
        dense_video_rows=dense_rows,
        sparse_video_rows=sparse_rows,
        dense_frames=frames,
        sparse_frames=frames,
        dense_trajectory={},
        sparse_trajectory={},
        step_window_weights={},
        metric_version="veda-video-fidelity-v1",
        latent_weight=0.4,
        frame_weight=0.3,
        temporal_weight=0.3,
        black_frame_threshold=0.01,
        flicker_threshold=2.0,
    )

    assert result.score == 0.0
    assert result.eligible is False
    assert result.latent_quality == 0.0


def test_non_finite_dense_latents_remain_an_infrastructure_error() -> None:
    metric = object.__new__(VideoFidelityMetric)
    dense_rows = np.full((2, 3), np.nan, dtype=np.float32)
    frames = np.ones((2, 2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="dense video latents"):
        metric.score(
            dense_video_rows=dense_rows,
            sparse_video_rows=np.ones((2, 3), dtype=np.float32),
            dense_frames=frames,
            sparse_frames=frames,
            dense_trajectory={},
            sparse_trajectory={},
            step_window_weights={},
            metric_version="veda-video-fidelity-v1",
            latent_weight=0.4,
            frame_weight=0.3,
            temporal_weight=0.3,
            black_frame_threshold=0.01,
            flicker_threshold=2.0,
        )


def test_non_finite_sparse_trajectory_receives_zero_fidelity() -> None:
    metric = object.__new__(VideoFidelityMetric)
    rows = np.ones((2, 3), dtype=np.float32)
    sparse_trajectory = {(0, "early"): rows.copy()}
    sparse_trajectory[(0, "early")][0, 0] = np.inf
    frames = np.ones((2, 2, 2, 3), dtype=np.uint8)

    result = metric.score(
        dense_video_rows=rows,
        sparse_video_rows=rows,
        dense_frames=frames,
        sparse_frames=frames,
        dense_trajectory={(0, "early"): rows},
        sparse_trajectory=sparse_trajectory,
        step_window_weights={"early": 1.0},
        metric_version="veda-video-fidelity-v2",
        latent_weight=0.4,
        frame_weight=0.3,
        temporal_weight=0.3,
        black_frame_threshold=0.01,
        flicker_threshold=2.0,
    )

    assert result.score == 0.0
    assert result.eligible is False


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
