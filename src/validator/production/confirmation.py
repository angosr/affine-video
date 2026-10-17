"""Two-GPU execution of paired full-video confirmation."""

from __future__ import annotations

import multiprocessing
import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoring.spec import TrackSpec
from scoring.video import VideoCell, VideoFidelityMetric, paired_video_gain_bound
from validator.production.batches import ConfirmationBatch
from validator.production.config import ProducerConfig


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    challenger_score: float
    incumbent_score: float
    gain_lower_bound: float
    challenger_cells: tuple[VideoCell, ...]
    incumbent_cells: tuple[VideoCell, ...]


class ConfirmationWorkerPool:
    """Run both models concurrently without occupying GPUs between confirmations."""

    def __init__(
        self,
        config: ProducerConfig,
        track: TrackSpec,
        *,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("confirmation timeout must be positive")
        self.config = config
        self.track = track
        self.timeout_seconds = timeout_seconds
        self._queues: list[Any] = []
        self._processes: list[multiprocessing.Process] = []

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._result = context.Queue()
        self._queues = []
        self._processes = []
        for gpu_id in self.config.evaluation_gpu_ids:
            tasks = context.Queue()
            process = context.Process(
                target=_confirmation_worker,
                args=(gpu_id, self.config, self.track, tasks, self._result),
                name=f"veda-confirmation-gpu-{gpu_id}",
            )
            process.start()
            self._queues.append(tasks)
            self._processes.append(process)

    def run(
        self,
        *,
        challenger_path: str | Path,
        incumbent_path: str | Path,
        batch: ConfirmationBatch,
    ) -> ConfirmationResult:
        if self._processes:
            raise RuntimeError("a full-video confirmation is already running")
        self._start()
        jobs = (
            ("challenger", str(challenger_path)),
            ("incumbent", str(incumbent_path)),
        )
        for tasks, job in zip(self._queues, jobs, strict=True):
            tasks.put((*job, batch))
        deadline = time.monotonic() + self.timeout_seconds
        cells: dict[str, tuple[VideoCell, ...]] = {}
        try:
            while len(cells) < 2:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("full-video confirmation exceeded its wall-clock limit")
                try:
                    kind, side, payload = self._result.get(timeout=min(30.0, remaining))
                except queue.Empty:
                    failed = [process for process in self._processes if not process.is_alive()]
                    if failed:
                        raise RuntimeError(
                            f"confirmation worker exited unexpectedly: {failed[0].exitcode}"
                        ) from None
                    continue
                if kind == "error":
                    raise RuntimeError(f"{side} confirmation failed: {payload}")
                if side in cells or side not in {"challenger", "incumbent"}:
                    raise RuntimeError("confirmation worker returned an invalid result")
                value = tuple(payload)
                if not value or any(not isinstance(item, VideoCell) for item in value):
                    raise RuntimeError("confirmation worker returned invalid video cells")
                cells[side] = value
        except Exception:
            self.close()
            raise
        try:
            paired = paired_video_gain_bound(
                cells["challenger"],
                cells["incumbent"],
                stratum_weights=self.track.scoring.stratum_weight_map,
                budget_costs={
                    budget.target_blocks: budget.calibrated_cost
                    for budget in self.track.scoring.budgets
                },
                confidence=self.track.match_confidence,
                resamples=self.track.scoring.bootstrap_resamples,
                seed=self.track.scoring.bootstrap_seed,
            )
            return ConfirmationResult(
                paired.challenger_score,
                paired.incumbent_score,
                paired.gain_lower_bound,
                cells["challenger"],
                cells["incumbent"],
            )
        finally:
            self.close()

    def close(self) -> None:
        for tasks in self._queues:
            tasks.put(None)
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self._queues = []
        self._processes = []


def _confirmation_worker(
    gpu_id: int,
    config: ProducerConfig,
    track: TrackSpec,
    tasks: Any,
    results: Any,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        from safetensors.numpy import load_file

        from validator.production.runtime import DenseRolloutRuntime

        runtime = DenseRolloutRuntime(
            model_root=str(config.artifact_path(config.runtime.model_root)),
            adapter_path=str(config.artifact_path(config.runtime.adapter)),
            predictor_bundle=str(config.artifact_path(config.runtime.predictor)),
            sample_cache=str(config.artifact_path(config.runtime.sample_cache)),
            offload_blocks=config.runtime.offload_blocks,
            prefetch=config.runtime.prefetch,
            mlp_chunk_rows=config.runtime.mlp_chunk_rows,
        )
        metric = VideoFidelityMetric(
            config.artifact_path(config.runtime.feature_model),
            expected_sha256=track.full_video.feature_model_sha256,
            device="cuda:0",
        )
        while True:
            task = tasks.get()
            if task is None:
                return
            side, selector_path, batch = task
            try:
                cells = []
                paths = {reference.rollout_id: path for reference, path in batch.latent_paths}
                for rollout in batch.plan.rollouts:
                    dense = load_file(paths[rollout.rollout_id])
                    if set(dense) != {"audio_rows", "video_rows"}:
                        raise ValueError("dense latent object contains unexpected tensors")
                    dense_frames = runtime.decode_frames(
                        dense["video_rows"], geometry=rollout.stratum
                    )
                    for budget in track.scoring.budgets:
                        sparse = runtime.generate_sparse(
                            sample_id=rollout.source_id,
                            geometry=rollout.stratum,
                            seed=rollout.noise_seed,
                            selector_path=selector_path,
                            architecture=track.model.selector,
                            budget=budget,
                        )
                        sparse_frames = runtime.decode_frames(
                            sparse.video_rows, geometry=rollout.stratum
                        )
                        fidelity = metric.score(
                            dense_video_rows=dense["video_rows"],
                            sparse_video_rows=sparse.video_rows,
                            dense_frames=dense_frames,
                            sparse_frames=sparse_frames,
                            latent_weight=track.full_video.latent_weight,
                            frame_weight=track.full_video.frame_weight,
                            temporal_weight=track.full_video.temporal_weight,
                            black_frame_threshold=track.full_video.black_frame_threshold,
                            flicker_threshold=track.full_video.flicker_threshold,
                        )
                        cells.append(
                            VideoCell(
                                rollout.stratum,
                                rollout.rollout_id,
                                budget.target_blocks,
                                fidelity.score,
                            )
                        )
                results.put(("ok", side, tuple(cells)))
            except Exception as exc:  # process boundary returns a bounded diagnostic
                results.put(("error", side, f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        results.put(("error", f"gpu-{gpu_id}", f"startup failed: {type(exc).__name__}: {exc}"))
