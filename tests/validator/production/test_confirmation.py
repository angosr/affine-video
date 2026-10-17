from types import SimpleNamespace

import numpy as np
import pytest

from validator.production.confirmation import _score_rollout_budget
from validator.production.runtime import SparseOutputInvalid


def track() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(selector=object()),
        scoring=SimpleNamespace(step_window_weight_map={"early": 1.0}),
        full_video=SimpleNamespace(
            metric_version="veda-video-fidelity-v2",
            latent_weight=0.4,
            frame_weight=0.3,
            temporal_weight=0.3,
            black_frame_threshold=0.01,
            flicker_threshold=2.0,
        ),
    )


def arguments(runtime) -> dict:
    return {
        "runtime": runtime,
        "metric": SimpleNamespace(score=lambda **_kwargs: pytest.fail("metric must not run")),
        "rollout": SimpleNamespace(source_id="source", stratum="wide", noise_seed=7),
        "selector_path": "selector.safetensors",
        "track": track(),
        "budget": SimpleNamespace(target_blocks=16),
        "selected_steps": (SimpleNamespace(index=0, window="early"),),
        "dense": {"video_rows": np.ones((2, 3), dtype=np.float32)},
        "dense_frames": np.ones((2, 2, 2, 3), dtype=np.uint8),
        "dense_trajectory": {(0, "early"): np.ones((2, 3), dtype=np.float32)},
    }


def test_non_finite_sparse_rollout_becomes_a_zero_score_cell() -> None:
    class Runtime:
        def generate_sparse(self, **_kwargs):
            raise SparseOutputInvalid("non-finite")

    assert _score_rollout_budget(**arguments(Runtime())) == 0.0


def test_infrastructure_runtime_failure_is_not_converted_to_zero() -> None:
    class Runtime:
        def generate_sparse(self, **_kwargs):
            raise RuntimeError("gpu unavailable")

    with pytest.raises(RuntimeError, match="gpu unavailable"):
        _score_rollout_budget(**arguments(Runtime()))
