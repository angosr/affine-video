from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scoring.video import VideoCell
from validator.production import confirmation
from validator.production.confirmation import ConfirmationWorkerPool, _score_rollout_budget
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

    score, output_sha256 = _score_rollout_budget(**arguments(Runtime()))

    assert score == 0.0
    assert len(output_sha256) == 64


def test_infrastructure_runtime_failure_is_not_converted_to_zero() -> None:
    class Runtime:
        def generate_sparse(self, **_kwargs):
            raise RuntimeError("gpu unavailable")

    with pytest.raises(RuntimeError, match="gpu unavailable"):
        _score_rollout_budget(**arguments(Runtime()))


def test_full_video_cell_hash_binds_raw_sparse_outputs() -> None:
    class Runtime:
        def __init__(self, value):
            self.value = value

        def generate_sparse(self, **_kwargs):
            return SimpleNamespace(
                audio_rows=np.array([[1.0]], dtype=np.float16),
                video_rows=np.array([[self.value, 2.0, 3.0]], dtype=np.float16),
                trajectory_video_rows=(
                    (0, "early", np.array([[4.0, 5.0, 6.0]], dtype=np.float16)),
                ),
            )

        def decode_frames(self, _rows, *, geometry):
            del geometry
            return np.ones((2, 2, 2, 3), dtype=np.uint8)

    values = arguments(Runtime(1.0))
    values["metric"] = SimpleNamespace(score=lambda **_kwargs: SimpleNamespace(score=0.75))
    first_score, first_hash = _score_rollout_budget(**values)
    values["runtime"] = Runtime(1.5)
    second_score, second_hash = _score_rollout_budget(**values)

    assert first_score == second_score == 0.75
    assert first_hash != second_hash


def test_confirmation_setup_uses_only_the_ready_selected_cache(tmp_path, monkeypatch) -> None:
    """Exercise the worker setup after the evaluator's hidden-pool reads are denied."""

    selected_cache = tmp_path / "ready-inputs"
    selected_cache.mkdir()
    evaluator_root = tmp_path / "evaluator-runtime"
    evaluator_root.mkdir()
    initialized = []

    class Runtime:
        def __init__(self, **kwargs):
            initialized.append(kwargs)

        def verify_trajectory_schedule(self, sigmas):
            assert sigmas == (1.0, 0.0)

    class Metric:
        def __init__(self, path, *, expected_sha256, device):
            assert path == evaluator_root / "weights/feature.safetensors"
            assert expected_sha256 == "12" * 32
            assert device == "cuda:0"

    class Tasks:
        def __init__(self):
            self.values = [
                (
                    "challenger",
                    "selector.safetensors",
                    SimpleNamespace(
                        sample_cache_path=selected_cache,
                        latent_paths=(),
                        plan=SimpleNamespace(rollouts=(), selected_steps=()),
                    ),
                ),
                None,
            ]

        def get(self):
            return self.values.pop(0)

    class Results:
        def __init__(self):
            self.values = []

        def put(self, value):
            self.values.append(value)

    runtime = SimpleNamespace(
        model_root="weights/model",
        adapter="weights/adapter.safetensors",
        predictor="weights/predictor.safetensors",
        sample_cache="samples/production",
        feature_model="weights/feature.safetensors",
        offload_blocks=0,
        prefetch=1,
        mlp_chunk_rows=128,
    )

    def hidden_artifact_path(_logical_name):
        pytest.fail("confirmation must not resolve the producer's hidden artifact root")

    def evaluator_artifact_path(logical_name):
        assert not logical_name.startswith("samples/")
        assert not logical_name.startswith("audit/")
        return evaluator_root / logical_name

    config = SimpleNamespace(
        runtime=runtime,
        artifact_path=hidden_artifact_path,
        evaluator_artifact_path=evaluator_artifact_path,
    )
    candidate_track = SimpleNamespace(
        scoring=SimpleNamespace(
            trajectory_schedule=SimpleNamespace(sigmas=(1.0, 0.0))
        ),
        full_video=SimpleNamespace(feature_model_sha256="12" * 32),
    )
    results = Results()
    monkeypatch.setattr("validator.production.runtime.DenseRolloutRuntime", Runtime)
    monkeypatch.setattr(confirmation, "VideoFidelityMetric", Metric)

    confirmation._confirmation_worker(0, config, candidate_track, Tasks(), results)

    assert len(initialized) == 1
    assert initialized[0]["sample_cache"] == str(selected_cache)
    assert results.values == [("ok", "challenger", ())]


def test_partial_worker_start_is_cleaned_up_and_pool_can_be_reused(monkeypatch) -> None:
    cells = (VideoCell("wide", "rollout", 16, 0.5),)

    class FakeQueue:
        def __init__(self, values=(), *, fail_shutdown=False):
            self.values = list(values)
            self.closed = False
            self.fail_shutdown = fail_shutdown

        def put(self, value):
            if self.closed:
                raise ValueError("queue is closed")
            if value is None and self.fail_shutdown:
                raise OSError("injected shutdown write failure")
            self.values.append(value)

        def get(self, *, timeout):
            del timeout
            if not self.values:
                pytest.fail("result queue was unexpectedly empty")
            return self.values.pop(0)

        def close(self):
            self.closed = True

        def join_thread(self):
            return None

    class FakeProcess:
        def __init__(self, *, fail_start=False):
            self.fail_start = fail_start
            self.started = False
            self.terminated = False
            self.joined = False
            self.exitcode = None

        def start(self):
            if self.fail_start:
                raise RuntimeError("injected second start failure")
            self.started = True

        def join(self, *, timeout):
            del timeout
            if not self.started:
                raise AssertionError("cannot join before start")
            self.joined = True

        def is_alive(self):
            if not self.started:
                raise AssertionError("cannot inspect before start")
            return not self.terminated

        def terminate(self):
            if not self.started:
                raise AttributeError("cannot terminate before start")
            self.terminated = True

    class FakeContext:
        def __init__(self):
            self.attempt = 0
            self.queue_index = 0
            self.process_index = 0
            self.processes = []

        def Queue(self):
            if self.queue_index == 0 and self.attempt == 1:
                value = FakeQueue((("ok", "challenger", cells), ("ok", "incumbent", cells)))
            else:
                value = FakeQueue(fail_shutdown=self.attempt == 0 and self.queue_index == 1)
            self.queue_index += 1
            return value

        def Process(self, **_kwargs):
            fail = self.attempt == 0 and self.process_index == 1
            value = FakeProcess(fail_start=fail)
            self.processes.append(value)
            self.process_index += 1
            if self.process_index == 2:
                self.attempt += 1
                self.queue_index = 0
                self.process_index = 0
            return value

    context = FakeContext()
    monkeypatch.setattr(confirmation.multiprocessing, "get_context", lambda _mode: context)
    monkeypatch.setattr(
        confirmation,
        "paired_video_gain_bound",
        Mock(
            return_value=SimpleNamespace(
                challenger_score=0.5,
                incumbent_score=0.5,
                gain_lower_bound=0.0,
            )
        ),
    )
    config = SimpleNamespace(evaluation_gpu_ids=(0, 1))
    candidate_track = SimpleNamespace(
        scoring=SimpleNamespace(
            stratum_weight_map={"wide": 1.0},
            budgets=(SimpleNamespace(target_blocks=16, calibrated_cost=1.0),),
            bootstrap_resamples=10,
            bootstrap_seed=1,
        ),
        match_confidence=0.95,
    )
    pool = ConfirmationWorkerPool(config, candidate_track, timeout_seconds=1)

    with pytest.raises(RuntimeError, match="injected second start failure"):
        pool.run(challenger_path="challenger", incumbent_path="incumbent", batch=object())

    first_process = context.processes[0]
    assert first_process.joined
    assert first_process.terminated
    assert pool._processes == []
    assert pool._queues == []

    result = pool.run(challenger_path="challenger", incumbent_path="incumbent", batch=object())

    assert result.gain_lower_bound == 0.0
    assert pool._processes == []
    assert pool._queues == []


def test_score_models_dispatches_every_model_and_returns_one_shared_result(monkeypatch) -> None:
    cells = {
        name: (VideoCell("wide", f"rollout-{name}", 16, score),)
        for name, score in (("baseline", 0.8), ("near", 0.7), ("other", 0.6))
    }

    class FakeQueue:
        def __init__(self, values=()):
            self.values = list(values)
            self.puts = []

        def put(self, value):
            self.puts.append(value)

        def get(self, *, timeout):
            del timeout
            return self.values.pop(0)

        def close(self):
            return None

        def join_thread(self):
            return None

    class FakeProcess:
        def join(self, *, timeout):
            del timeout

        def is_alive(self):
            return False

    result = FakeQueue(("ok", name, value) for name, value in cells.items())
    tasks = [FakeQueue(), FakeQueue()]
    config = SimpleNamespace(evaluation_gpu_ids=(0, 1))
    pool = ConfirmationWorkerPool(config, track(), timeout_seconds=1)

    def start() -> None:
        pool._result = result
        pool._queues = tasks
        pool._processes = [FakeProcess(), FakeProcess()]

    monkeypatch.setattr(pool, "_start", start)
    result_cells = pool.score_models(
        model_paths={
            "baseline": "baseline.safetensors",
            "near": "near.safetensors",
            "other": "other.safetensors",
        },
        batch=object(),
    )

    assert result_cells == cells
    assert [value[0] for value in tasks[0].puts[:-1]] == ["baseline", "other"]
    assert [value[0] for value in tasks[1].puts[:-1]] == ["near"]
    assert tasks[0].puts[-1] is None
    assert tasks[1].puts[-1] is None
    assert pool._processes == []


def test_score_models_rejects_empty_model_set_without_starting(monkeypatch) -> None:
    pool = ConfirmationWorkerPool(
        SimpleNamespace(evaluation_gpu_ids=(0, 1)), track(), timeout_seconds=1
    )
    start = Mock()
    monkeypatch.setattr(pool, "_start", start)

    with pytest.raises(ValueError, match="labels"):
        pool.score_models(model_paths={}, batch=object())

    start.assert_not_called()
