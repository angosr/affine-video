"""Frozen full-video fidelity metric and paired rollout inference."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from pathlib import Path

import numpy as np

from scoring.evaluate import percentile
from scoring.metrics import log_cost_auc


@dataclass(frozen=True, slots=True)
class VideoFidelity:
    score: float
    latent_quality: float
    trajectory_quality: float
    frame_quality: float
    temporal_quality: float
    black_frame_fraction: float
    flicker_ratio: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class VideoCell:
    stratum: str
    rollout_id: str
    budget: int
    score: float


@dataclass(frozen=True, slots=True)
class PairedVideoScore:
    challenger_score: float
    incumbent_score: float
    gain_lower_bound: float


class VideoFidelityMetric:
    """R3D-18 feature metric bound to one immutable safetensors file."""

    def __init__(self, weights: str | Path, *, expected_sha256: str, device: str) -> None:
        path = Path(weights)
        if _file_sha256(path) != expected_sha256:
            raise ValueError("video feature model does not match its track commitment")
        try:
            import torch
            from safetensors.torch import load_file
            from torchvision.models.video import r3d_18
        except ImportError as exc:
            raise RuntimeError("full-video scoring requires the pinned GPU environment") from exc
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("full-video feature scoring requires CUDA")
        model = r3d_18(weights=None)
        model.load_state_dict(load_file(path, device="cpu"), strict=True)
        self._features = (
            torch.nn.Sequential(
                model.stem,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            .to(resolved)
            .eval()
        )
        self._torch = torch
        self._device = resolved

    def score(
        self,
        *,
        dense_video_rows: np.ndarray,
        sparse_video_rows: np.ndarray,
        dense_frames: np.ndarray,
        sparse_frames: np.ndarray,
        dense_trajectory: Mapping[tuple[int, str], np.ndarray],
        sparse_trajectory: Mapping[tuple[int, str], np.ndarray],
        step_window_weights: Mapping[str, float],
        metric_version: str,
        latent_weight: float,
        frame_weight: float,
        temporal_weight: float,
        black_frame_threshold: float,
        flicker_threshold: float,
    ) -> VideoFidelity:
        """Score one sparse rollout against its same-seed dense reference."""

        weights = (latent_weight, frame_weight, temporal_weight)
        if (
            any(not isfinite(value) or value < 0 for value in weights)
            or abs(fsum(weights) - 1.0) > 1e-12
        ):
            raise ValueError("video metric weights must be non-negative and sum to one")
        if metric_version not in {"veda-video-fidelity-v1", "veda-video-fidelity-v2"}:
            raise ValueError("full-video metric version is unsupported")
        dense_latent = _finite_array(dense_video_rows, "dense video latents")
        sparse_latent = np.asarray(sparse_video_rows)
        if sparse_latent.size == 0:
            raise ValueError("sparse video latents must be non-empty")
        if dense_latent.shape != sparse_latent.shape:
            raise ValueError("dense and sparse video latents must have the same shape")
        dense = _frames(dense_frames, "dense frames")
        sparse = _frames(sparse_frames, "sparse frames")
        if dense.shape != sparse.shape:
            raise ValueError("dense and sparse decoded frames must have the same shape")
        if dense.shape[0] < 2:
            raise ValueError("full-video metric requires at least two frames")

        sparse_trajectory_invalid = False
        trajectory_pairs = None
        if metric_version == "veda-video-fidelity-v2":
            trajectory_pairs, sparse_trajectory_invalid = _trajectory_pairs(
                dense_trajectory,
                sparse_trajectory,
                step_window_weights=step_window_weights,
                allow_nonfinite_sparse=True,
            )
        if not np.isfinite(sparse_latent).all() or sparse_trajectory_invalid:
            return _catastrophic_fidelity(dense, sparse)
        sparse_latent = _finite_array(sparse_latent, "sparse video latents")

        final_latent_quality = _latent_quality(sparse_latent, dense_latent)
        if metric_version == "veda-video-fidelity-v1":
            latent_quality = final_latent_quality
            trajectory_quality = final_latent_quality
        elif metric_version == "veda-video-fidelity-v2":
            assert trajectory_pairs is not None
            trajectory_quality = _trajectory_quality(
                trajectory_pairs,
                step_window_weights,
            )
            latent_quality = (final_latent_quality + trajectory_quality) / 2.0
        dense_features = self._embed(dense)
        sparse_features = self._embed(sparse)
        frame_quality = _mean_cosine(dense_features, sparse_features)
        temporal_quality = _mean_cosine(
            np.diff(dense_features, axis=0), np.diff(sparse_features, axis=0)
        )
        black_fraction = float(np.mean(sparse.mean(axis=(1, 2, 3)) <= 1.0))
        dense_motion = float(np.mean(np.abs(np.diff(dense.astype(np.float32), axis=0))))
        sparse_motion = float(np.mean(np.abs(np.diff(sparse.astype(np.float32), axis=0))))
        flicker_ratio = sparse_motion / max(dense_motion, 1e-6)
        eligible = black_fraction <= black_frame_threshold and flicker_ratio <= flicker_threshold
        score = (
            latent_weight * latent_quality
            + frame_weight * frame_quality
            + temporal_weight * temporal_quality
        )
        if not eligible:
            score = 0.0
        return VideoFidelity(
            float(score),
            float(latent_quality),
            float(trajectory_quality),
            float(frame_quality),
            float(temporal_quality),
            black_fraction,
            flicker_ratio,
            eligible,
        )

    def _embed(self, frames: np.ndarray) -> np.ndarray:
        torch = self._torch
        value = torch.from_numpy(frames).to(self._device, torch.float32)
        value = value.permute(3, 0, 1, 2)[None] / 255.0
        height, width = frames.shape[1:3]
        scale = 128.0 / min(height, width)
        resized = torch.nn.functional.interpolate(
            value,
            size=(frames.shape[0], round(height * scale), round(width * scale)),
            mode="trilinear",
            align_corners=False,
        )
        top = (resized.shape[-2] - 112) // 2
        left = (resized.shape[-1] - 112) // 2
        resized = resized[..., top : top + 112, left : left + 112]
        mean = torch.tensor((0.43216, 0.394666, 0.37645), device=self._device)[
            None, :, None, None, None
        ]
        std = torch.tensor((0.22803, 0.22145, 0.216989), device=self._device)[
            None, :, None, None, None
        ]
        with torch.inference_mode():
            features = self._features((resized - mean) / std).mean(dim=(-1, -2))
        return features[0].transpose(0, 1).float().cpu().numpy()


def trajectory_latent_quality(
    dense: Mapping[tuple[int, str], np.ndarray],
    sparse: Mapping[tuple[int, str], np.ndarray],
    *,
    step_window_weights: Mapping[str, float],
) -> float:
    """Aggregate intermediate latent fidelity without treating steps as samples."""

    by_window, _ = _trajectory_pairs(
        dense,
        sparse,
        step_window_weights=step_window_weights,
        allow_nonfinite_sparse=False,
    )
    return _trajectory_quality(by_window, step_window_weights)


def _trajectory_pairs(
    dense: Mapping[tuple[int, str], np.ndarray],
    sparse: Mapping[tuple[int, str], np.ndarray],
    *,
    step_window_weights: Mapping[str, float],
    allow_nonfinite_sparse: bool,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], bool]:
    if not dense or set(dense) != set(sparse):
        raise ValueError("dense and sparse trajectories must be non-empty and aligned")
    if (
        not step_window_weights
        or any(not isfinite(weight) or weight <= 0 for weight in step_window_weights.values())
        or abs(fsum(step_window_weights.values()) - 1.0) > 1e-12
    ):
        raise ValueError("trajectory window weights must be positive and sum to one")
    by_window: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        window: [] for window in step_window_weights
    }
    nonfinite_sparse = False
    for (step, window), reference in dense.items():
        if type(step) is not int or step < 0 or window not in by_window:
            raise ValueError("trajectory step identity is invalid")
        target = _finite_array(reference, "dense trajectory latents")
        candidate = np.asarray(sparse[(step, window)])
        if candidate.size == 0:
            raise ValueError("sparse trajectory latents must be non-empty")
        if candidate.shape != target.shape:
            raise ValueError("dense and sparse trajectory latents must have the same shape")
        finite = np.isfinite(candidate).all()
        if not finite and not allow_nonfinite_sparse:
            raise ValueError("sparse trajectory latents must be non-empty and finite")
        nonfinite_sparse |= not finite
        by_window[window].append((candidate, target))
    if any(not values for values in by_window.values()):
        raise ValueError("trajectory must cover every weighted step window")
    return by_window, nonfinite_sparse


def _trajectory_quality(
    by_window: Mapping[str, list[tuple[np.ndarray, np.ndarray]]],
    step_window_weights: Mapping[str, float],
) -> float:
    return fsum(
        step_window_weights[window]
        * fsum(_latent_quality(candidate, target) for candidate, target in values)
        / len(values)
        for window, values in by_window.items()
    )


def paired_video_gain_bound(
    challenger: Sequence[VideoCell],
    incumbent: Sequence[VideoCell],
    *,
    stratum_weights: Mapping[str, float],
    budget_costs: Mapping[int, float],
    confidence: float,
    resamples: int,
    seed: int,
) -> PairedVideoScore:
    """Paired rollout-cluster bound; budgets stay correlated within a rollout."""

    if len(challenger) != len(incumbent) or not challenger:
        raise ValueError("paired video cells must be non-empty and aligned")
    if not 0 < confidence < 1 or resamples <= 0 or seed < 0:
        raise ValueError("video bootstrap configuration is invalid")
    left = {(row.stratum, row.rollout_id, row.budget): row.score for row in challenger}
    right = {(row.stratum, row.rollout_id, row.budget): row.score for row in incumbent}
    if set(left) != set(right):
        raise ValueError("challenger and incumbent video cells do not align")
    if (
        set(row.stratum for row in challenger) != set(stratum_weights)
        or abs(fsum(stratum_weights.values()) - 1.0) > 1e-12
    ):
        raise ValueError("video cells must cover every weighted stratum")
    rollouts = {
        stratum: tuple(sorted({key[1] for key in left if key[0] == stratum}))
        for stratum in stratum_weights
    }
    if any(len(items) < 4 for items in rollouts.values()):
        raise ValueError("video scoring requires at least four rollouts per stratum")
    budgets = tuple(sorted({key[2] for key in left}))
    if set(budgets) != set(budget_costs):
        raise ValueError("video budget costs do not match the scored budgets")
    costs = tuple(budget_costs[budget] for budget in budgets)
    expected = {
        (stratum, rollout, budget)
        for stratum, values in rollouts.items()
        for rollout in values
        for budget in budgets
    }
    if set(left) != expected:
        raise ValueError("video cells do not form a complete stratum/rollout/budget design")

    def rollout_scores(
        values: Mapping[tuple[str, str, int], float],
    ) -> dict[tuple[str, str], float]:
        return {
            (stratum, rollout): log_cost_auc(
                costs,
                tuple(values[(stratum, rollout, budget)] for budget in budgets),
            )
            for stratum, items in rollouts.items()
            for rollout in items
        }

    def aggregate(values: Mapping[tuple[str, str], float]) -> float:
        return fsum(
            stratum_weights[stratum]
            * fsum(values[(stratum, rollout)] for rollout in rollouts[stratum])
            / len(rollouts[stratum])
            for stratum in stratum_weights
        )

    challenger_rollouts = rollout_scores(left)
    incumbent_rollouts = rollout_scores(right)
    challenger_score = aggregate(challenger_rollouts)
    incumbent_score = aggregate(incumbent_rollouts)
    differences = {
        key: challenger_rollouts[key] - incumbent_rollouts[key] for key in challenger_rollouts
    }
    generator = random.Random(seed)
    statistics = []
    for _ in range(resamples):
        value = 0.0
        for stratum, weight in stratum_weights.items():
            sampled = [
                rollouts[stratum][generator.randrange(len(rollouts[stratum]))]
                for _ in rollouts[stratum]
            ]
            value += (
                weight * fsum(differences[(stratum, rollout)] for rollout in sampled) / len(sampled)
            )
        statistics.append(value)
    return PairedVideoScore(
        challenger_score,
        incumbent_score,
        percentile(statistics, 1 - confidence),
    )


def _mean_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = np.sum(reference * candidate, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)))


def _relative_mse(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.mean(np.square(candidate.astype(np.float32) - reference.astype(np.float32)))
        / max(float(np.mean(np.square(reference.astype(np.float32)))), 1e-8)
    )


def _latent_quality(candidate: np.ndarray, reference: np.ndarray) -> float:
    return 1.0 / (1.0 + _relative_mse(candidate, reference))


def _catastrophic_fidelity(dense_frames: np.ndarray, sparse_frames: np.ndarray) -> VideoFidelity:
    black_fraction = float(np.mean(sparse_frames.mean(axis=(1, 2, 3)) <= 1.0))
    dense_motion = float(
        np.mean(np.abs(np.diff(dense_frames.astype(np.float32), axis=0)))
    )
    sparse_motion = float(
        np.mean(np.abs(np.diff(sparse_frames.astype(np.float32), axis=0)))
    )
    return VideoFidelity(
        score=0.0,
        latent_quality=0.0,
        trajectory_quality=0.0,
        frame_quality=0.0,
        temporal_quality=0.0,
        black_frame_fraction=black_fraction,
        flicker_ratio=sparse_motion / max(dense_motion, 1e-6),
        eligible=False,
    )


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _frames(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.uint8 or result.ndim != 4 or result.shape[-1] != 3:
        raise ValueError(f"{name} must be uint8 [time,height,width,3]")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
