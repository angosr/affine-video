"""Real MiniMax-H3 rollout hook used by the private producer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scoring.evaluate import reference_attention, segmented_top_k, trip_pool
from scoring.exact import (
    GLOBAL_SEGMENT,
    REFERENCE_SEGMENT,
    TARGET_SEGMENT,
    ExactCase,
    canonical_dense_reference,
)
from scoring.selector import SelectorArchitecture, load_selector, route_indices, route_name
from scoring.spec import BudgetSpec
from validator.production.plan import select_query_tiles


@dataclass(frozen=True, slots=True)
class RolloutCapture:
    cases: tuple[ExactCase, ...]
    elapsed_seconds: float
    used_tokens: int
    key_tiles_by_route: tuple[tuple[str, int], ...]
    video_rows: np.ndarray
    audio_rows: np.ndarray
    trajectory_video_rows: tuple[tuple[int, str, np.ndarray], ...]


@dataclass(frozen=True, slots=True)
class DenseOutputFiles:
    video_path: Path
    latent_path: Path
    decode_seconds: float
    black_frame_fraction: float


@dataclass(frozen=True, slots=True)
class SparseRollout:
    video_rows: np.ndarray
    audio_rows: np.ndarray
    elapsed_seconds: float
    sparse_calls: int
    dense_calls: int
    trajectory_video_rows: tuple[tuple[int, str, np.ndarray], ...]


@dataclass(frozen=True, slots=True)
class RuntimeTraceVerification:
    descriptor_max_abs: float
    logits_max_abs: float
    mask_mismatches: tuple[int, ...]
    output_max_abs: tuple[float, ...]
    output_mean_abs: tuple[float, ...]


class SparseOutputInvalid(RuntimeError):
    """A schema-valid selector produced unusable sparse rollout values."""


class DenseRolloutRuntime:
    """One long-lived teacher per GPU; rollout inputs and outputs remain disposable."""

    def __init__(
        self,
        *,
        model_root: str,
        adapter_path: str,
        predictor_bundle: str,
        sample_cache: str,
        offload_blocks: int = 0,
        prefetch: int = 1,
        mlp_chunk_rows: int = 8192,
    ) -> None:
        try:
            import torch
            from miowtion.h3 import attention as h3_attention
            from miowtion.train import data, parallel
            from miowtion.train import teacher as teacher_lib
            from miowtion.train import trajectory as trajectory_lib
            from miowtion.veda import attention as veda_attention
            from miowtion.veda import bundle as veda_bundle
            from miowtion.veda import mask as veda_mask
            from miowtion.veda import plan as veda_plan
        except ImportError as exc:
            raise RuntimeError("GPU production requires the pinned Miowtion environment") from exc
        self.torch = torch
        self.h3_attention = h3_attention
        self.data = data
        self.trajectory_lib = trajectory_lib
        self.veda_attention = veda_attention
        self.model_root = model_root
        self.environment = parallel.init_distributed()
        self.cache = data.SampleCache(sample_cache)
        self.samples = {sample.id: sample for sample in self.cache.samples}
        bundle_metadata = veda_bundle.read_metadata(predictor_bundle)
        self.plans = veda_plan.PlanTable(
            [
                veda_plan.TilePlan.from_json(document)
                for document in json.loads(bundle_metadata["plans"]).values()
            ]
        )
        visual_conditions = any(sample.task != "t2va" for sample in self.samples.values())
        self.teacher = teacher_lib.build_teacher(
            model_root,
            "FL2VA",
            "turbo",
            8,
            adapter_path,
            self.environment,
            visual_conditions=visual_conditions,
            audio_references=False,
            offload_blocks=offload_blocks,
            prefetch=prefetch,
            mlp_chunk_rows=mlp_chunk_rows,
        )
        self.config = veda_attention.VedaConfig(target_budget=veda_mask.Budget(ratio=0.1))
        self.decoder: Any | None = None

    def verify_trajectory_schedule(self, expected_sigmas: tuple[float, ...]) -> None:
        """Fail before generation when the committed sigma grid is rounded or stale."""

        actual = tuple(float(value) for value in self.teacher.schedule.video)
        if len(actual) != len(expected_sigmas) or any(
            not isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
            for value, expected in zip(actual, expected_sigmas, strict=True)
        ):
            raise RuntimeError("runtime trajectory schedule differs from the track commitment")

    def generate_sparse(
        self,
        *,
        sample_id: str,
        geometry: str,
        seed: int,
        selector_path: str | Path,
        architecture: SelectorArchitecture,
        budget: BudgetSpec,
        steps: tuple[tuple[int, str], ...],
    ) -> SparseRollout:
        """Run one real VEDA trajectory with submitted projection weights."""

        from miowtion.veda import mask as veda_mask
        from miowtion.veda import predictor as veda_predictor
        from safetensors import safe_open

        if sample_id not in self.samples:
            raise KeyError(f"{sample_id} is absent from the encoded sample cache")
        torch = self.torch
        predictor = veda_predictor.TileScorePredictor(
            architecture.num_layers,
            architecture.num_heads,
            architecture.head_dim,
        )
        state = {}
        with safe_open(selector_path, framework="pt", device="cpu") as tensors:
            expected = {
                f"layers.{layer}.{side}"
                for layer in range(architecture.num_layers)
                for side in ("proj_q", "proj_k")
            }
            if set(tensors.keys()) != expected:
                raise ValueError("selector tensor keys do not match the runtime predictor")
            for name in sorted(expected):
                state[name] = tensors.get_tensor(name)
        predictor.load_state_dict(state, strict=True)
        # Miowtion pools BF16 activations into FP32 descriptors and its predictor
        # explicitly evaluates projections in FP32. Preserve submitted FP16 values
        # exactly when materializing that FP32 side branch.
        predictor = predictor.to(self.environment.device, torch.float32).eval()
        parsed_geometry = self.data.parse_geometry(geometry)
        tile_plan = self.plans.select(parsed_geometry)
        ref_budget = (
            None
            if budget.reference_blocks == 0
            else veda_mask.Budget(tiles=budget.reference_blocks)
        )
        config = self.veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(tiles=budget.target_blocks),
            ref_budget=ref_budget,
        )
        step_windows = dict(steps)
        if len(step_windows) != len(steps) or any(index < 0 for index in step_windows):
            raise ValueError("sparse trajectory steps must be non-negative and unique")
        rollout = self.trajectory_lib.Trajectory(
            self.teacher.model,
            self.cache,
            self.samples[sample_id],
            parsed_geometry,
            self.teacher.schedule,
            seed,
            self.environment.device,
        )
        clip = self.veda_attention.ClipTiling(rollout.layout, config, self.environment.device)
        trajectory = []
        sparse_calls = 0
        dense_calls = 0
        torch.cuda.synchronize(self.environment.device)
        started = time.monotonic()
        with torch.no_grad():
            while not rollout.done:
                inputs = rollout.inputs()
                attention = self.veda_attention.SparseStudent(clip, tile_plan, predictor)
                video_v, audio_v = self.teacher.model(
                    rollout.clip,
                    inputs.video_rows,
                    inputs.audio_rows,
                    inputs.timestep,
                    attention,
                    self.teacher.tables.get(inputs.timestep.timesteps),
                )
                sparse_calls += attention.calls["sparse"]
                dense_calls += attention.calls["dense"]
                rollout.advance(video_v, audio_v)
                if inputs.step in step_windows:
                    trajectory.append(
                        (
                            inputs.step,
                            step_windows[inputs.step],
                            rollout.video_rows[rollout.n_cond_video :].float().cpu().numpy(),
                        )
                    )
        torch.cuda.synchronize(self.environment.device)
        elapsed = time.monotonic() - started
        if len(trajectory) != len(steps):
            raise RuntimeError("sparse rollout omitted frozen trajectory steps")
        video = rollout.video_rows[rollout.n_cond_video :].float().cpu().numpy()
        audio = rollout.audio_rows[rollout.n_cond_audio :].float().cpu().numpy()
        invalid = (
            not np.isfinite(video).all()
            or not np.isfinite(audio).all()
            or any(not np.isfinite(rows).all() for _, _, rows in trajectory)
        )
        del predictor, state
        torch.cuda.empty_cache()
        if invalid:
            raise SparseOutputInvalid("sparse rollout produced non-finite latents")
        return SparseRollout(
            video,
            audio,
            elapsed,
            sparse_calls,
            dense_calls,
            tuple(trajectory),
        )

    def decode_frames(self, video_rows: np.ndarray, *, geometry: str) -> np.ndarray:
        """Decode final video latents through the same pinned VAE as production."""

        from miowtion.infer import decode

        if not np.isfinite(video_rows).all():
            raise ValueError("video latents must be finite")
        if self.decoder is None:
            self.decoder = decode.Decoder(
                os.path.join(self.model_root, "FL2VA"),
                self.environment.device,
                self.torch.bfloat16,
            )
        frames = self.decoder.video(
            self.torch.from_numpy(video_rows), self.data.parse_geometry(geometry)
        )
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise RuntimeError("video decoder returned invalid frames")
        return frames

    def capture(
        self,
        *,
        sample_id: str,
        source_id: str,
        stratum: str,
        rollout_id: str,
        geometry: str,
        seed: int,
        routes: tuple[str, ...],
        steps: tuple[tuple[int, str], ...],
        query_selection_seed: str,
        query_tiles_per_case: int,
    ) -> RolloutCapture:
        """Run one dense trajectory without rebuilding the resident teacher."""

        requested = set(routes)
        if not requested or len(requested) != len(routes):
            raise ValueError("capture routes must be non-empty and unique")
        step_windows = dict(steps)
        if len(step_windows) != len(steps) or any(index < 0 for index in step_windows):
            raise ValueError("capture steps must be non-negative and unique")
        if sample_id not in self.samples:
            raise KeyError(f"{sample_id} is absent from the encoded sample cache")
        parsed_geometry = self.data.parse_geometry(geometry)
        tile_plan = self.plans.select(parsed_geometry)
        rollout = self.trajectory_lib.Trajectory(
            self.teacher.model,
            self.cache,
            self.samples[sample_id],
            parsed_geometry,
            self.teacher.schedule,
            seed,
            self.environment.device,
        )
        clip = self.veda_attention.ClipTiling(rollout.layout, self.config, self.environment.device)
        torch = self.torch
        runtime = self

        class CaptureProbe:
            def __init__(self) -> None:
                self.step = 0
                self.cases: list[ExactCase] = []
                self.key_tiles: dict[str, int] = {}

            @torch.no_grad()
            def __call__(self, q, k, v, layer_index):
                output, _ = runtime.h3_attention.dense_attention(q, k, v, clip.layout.used)
                if self.step not in step_windows:
                    return output
                selected_by_head = {
                    head
                    for route in requested
                    for layer, head in (route_indices(route),)
                    if layer == layer_index
                }
                if not selected_by_head:
                    return output
                for group in tile_plan.head_groups(layer_index, clip.device):
                    heads = [
                        int(head) for head in group.heads.tolist() if int(head) in selected_by_head
                    ]
                    if heads:
                        self._capture_group(q, k, v, layer_index, group, heads)
                return output

            def _capture_group(self, q, k, v, layer_index, group, heads):
                head_tensor = torch.tensor(heads, dtype=torch.long, device=clip.device)
                layout = clip.get(group.shape)
                gather = runtime.veda_attention._gather  # pylint: disable=protected-access
                rows = [gather(value, layout, head_tensor) for value in (q, k, v)]
                shaped = [
                    value.view(layout.n_tiles, 128, len(heads), -1).permute(2, 0, 1, 3)
                    for value in rows
                ]
                valid_counts = layout.valid_count.to(torch.int16).cpu().numpy()
                segments = np.full(layout.n_tiles, GLOBAL_SEGMENT, dtype=np.int8)
                segments[: layout.n_ref_tiles] = REFERENCE_SEGMENT
                segments[layout.n_ref_tiles : layout.n_video_tiles] = TARGET_SEGMENT
                plan_hash = _tile_plan_hash(group.shape, layout, heads)
                for local_head, head in enumerate(heads):
                    route = route_name(layer_index, head)
                    case_id = f"{rollout_id}:{self.step}:{route}"
                    query_ids = select_query_tiles(
                        query_selection_seed=query_selection_seed,
                        case_id=case_id,
                        target_tiles=tuple(range(layout.n_ref_tiles, layout.n_video_tiles)),
                        reference_tiles=tuple(range(layout.n_ref_tiles)),
                        count=query_tiles_per_case,
                    )
                    required = np.zeros((len(query_ids), layout.n_tiles), dtype=np.bool_)
                    for row, query_id in enumerate(query_ids):
                        required[row, query_id] = True
                    q_value, k_value, v_value = (
                        _to_float16(value[local_head]) for value in shaped
                    )
                    q_value = q_value[list(query_ids)]
                    query_valid_counts = valid_counts[list(query_ids)].copy()
                    dense_value = canonical_dense_reference(
                        q_value,
                        k_value,
                        v_value,
                        query_valid_counts,
                        valid_counts,
                        device=str(clip.device),
                    )
                    self.cases.append(
                        ExactCase(
                            route=route,
                            stratum=stratum,
                            rollout_id=rollout_id,
                            source_id=source_id,
                            step_index=self.step,
                            step_window=step_windows[self.step],
                            scheduler_timestep=float(runtime.teacher.schedule.video[self.step]),
                            sigma=float(runtime.teacher.schedule.video[self.step]),
                            tile_plan_hash=plan_hash,
                            query_tile_ids=query_ids,
                            q=q_value,
                            k=k_value,
                            v=v_value,
                            dense=dense_value,
                            query_valid_counts=query_valid_counts,
                            key_valid_counts=valid_counts.copy(),
                            key_segments=segments.copy(),
                            required_keys=required,
                            epsilon=1e-6,
                        )
                    )
                    self.key_tiles[route] = layout.n_tiles
                del rows, shaped

        probe = CaptureProbe()
        trajectory = []
        started = time.monotonic()
        with torch.no_grad():
            while not rollout.done:
                inputs = rollout.inputs()
                probe.step = inputs.step
                video_v, audio_v = self.teacher.model(
                    rollout.clip,
                    inputs.video_rows,
                    inputs.audio_rows,
                    inputs.timestep,
                    probe,
                    self.teacher.tables.get(inputs.timestep.timesteps),
                )
                rollout.advance(video_v, audio_v)
                if inputs.step in step_windows:
                    trajectory.append(
                        (
                            inputs.step,
                            step_windows[inputs.step],
                            rollout.video_rows[rollout.n_cond_video :].float().cpu().numpy(),
                        )
                    )
        expected = len(routes) * len(steps)
        if len(probe.cases) != expected:
            raise RuntimeError(f"runtime captured {len(probe.cases)} cells, expected {expected}")
        if len(trajectory) != len(steps):
            raise RuntimeError("dense rollout omitted frozen trajectory steps")
        return RolloutCapture(
            tuple(probe.cases),
            time.monotonic() - started,
            rollout.layout.used,
            tuple(sorted(probe.key_tiles.items())),
            rollout.video_rows[rollout.n_cond_video :].float().cpu().numpy(),
            rollout.audio_rows[rollout.n_cond_audio :].float().cpu().numpy(),
            tuple(trajectory),
        )

    def write_dense_outputs(
        self,
        capture: RolloutCapture,
        *,
        geometry: str,
        rollout_id: str,
        directory: str | Path,
    ) -> DenseOutputFiles:
        """Persist lossless final latents and a decoded dense MP4 atomically."""

        from miowtion.infer import decode
        from safetensors.numpy import save_file

        if not rollout_id:
            raise ValueError("rollout ID cannot be empty")
        if (
            not np.isfinite(capture.video_rows).all()
            or not np.isfinite(capture.audio_rows).all()
            or any(not np.isfinite(rows).all() for _, _, rows in capture.trajectory_video_rows)
        ):
            raise RuntimeError("dense rollout produced non-finite final latents")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        latent_path = root / f"{rollout_id}.safetensors"
        video_path = root / f"{rollout_id}.mp4"
        latent_temporary = root / f".{rollout_id}.latent.tmp"
        video_temporary = root / f".{rollout_id}.video.tmp.mp4"
        try:
            tensors = {
                "audio_rows": np.ascontiguousarray(capture.audio_rows),
                "video_rows": np.ascontiguousarray(capture.video_rows),
                **{
                    trajectory_tensor_key(step): np.ascontiguousarray(rows)
                    for step, _window, rows in capture.trajectory_video_rows
                },
            }
            save_file(
                tensors,
                latent_temporary,
                metadata={"geometry": geometry, "rollout_id": rollout_id},
            )
            _fsync(latent_temporary)
            os.replace(latent_temporary, latent_path)
            parsed_geometry = self.data.parse_geometry(geometry)
            if self.decoder is None:
                self.decoder = decode.Decoder(
                    os.path.join(self.model_root, "FL2VA"),
                    self.environment.device,
                    self.torch.bfloat16,
                )
            started = time.monotonic()
            frames = self.decoder.video(self.torch.from_numpy(capture.video_rows), parsed_geometry)
            audio = self.decoder.audio(self.torch.from_numpy(capture.audio_rows))
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                raise RuntimeError("dense video decoder returned invalid frames")
            black_fraction = float(np.mean(frames.mean(axis=(1, 2, 3)) <= 1.0))
            decode.write_mp4(
                str(video_temporary),
                frames,
                [(audio, "Dense reference")],
                self.decoder.sample_rate,
            )
            _fsync(video_temporary)
            os.replace(video_temporary, video_path)
            return DenseOutputFiles(
                video_path,
                latent_path,
                time.monotonic() - started,
                black_fraction,
            )
        finally:
            for path in (latent_temporary, video_temporary):
                if path.exists():
                    path.unlink()


def verify_runtime_trace(
    case: ExactCase,
    *,
    selector_path: str,
    architecture: SelectorArchitecture,
    budgets: tuple[BudgetSpec, ...],
    device: str,
    descriptor_atol: float = 1e-6,
    logits_atol: float = 1e-4,
    output_atol: float = 1.25e-1,
    output_mean_atol: float = 1e-4,
) -> RuntimeTraceVerification:
    """Compare one cached case with Miowtion's native predictor and kernel semantics."""

    try:
        import torch
        from miowtion.kernels import reference
        from miowtion.veda import mask as veda_mask
        from miowtion.veda import predictor as veda_predictor
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "runtime trace verification requires the pinned GPU environment"
        ) from exc
    if min(descriptor_atol, logits_atol, output_atol, output_mean_atol) <= 0:
        raise ValueError("runtime trace tolerances must be positive")
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("runtime trace verification requires CUDA")

    selector = load_selector(selector_path, architecture)
    layer_index, head_index = route_indices(case.route)
    with safe_open(selector_path, framework="pt", device="cpu") as tensors, torch.no_grad():
        predictor = veda_predictor.LayerPredictor(architecture.num_heads, architecture.head_dim).to(
            dtype=torch.float32, device=resolved
        )
        predictor.proj_q.copy_(tensors.get_tensor(f"layers.{layer_index}.proj_q").to(resolved))
        predictor.proj_k.copy_(tensors.get_tensor(f"layers.{layer_index}.proj_k").to(resolved))

    q = torch.as_tensor(case.q, device=resolved)
    k = torch.as_tensor(case.k, device=resolved)
    v = torch.as_tensor(case.v, device=resolved)
    query_layout = _pool_layout(case.query_valid_counts, resolved, torch)
    key_layout = _pool_layout(case.key_valid_counts, resolved, torch)
    q_features = veda_predictor.pool_tiles(q.reshape(-1, 1, architecture.head_dim), query_layout)
    k_features = veda_predictor.pool_tiles(k.reshape(-1, 1, architecture.head_dim), key_layout)
    expected_q = trip_pool(case.q, case.query_valid_counts)
    expected_k = trip_pool(case.k, case.key_valid_counts)
    descriptor_delta = max(
        float(np.max(np.abs(q_features[0].cpu().numpy() - expected_q))),
        float(np.max(np.abs(k_features[0].cpu().numpy() - expected_k))),
    )
    heads = torch.tensor([head_index], dtype=torch.long, device=resolved)
    runtime_logits = predictor(q_features, k_features, heads)[0]
    expected_logits = selector.logits(case.route, expected_q, expected_k)
    logits_delta = float(np.max(np.abs(runtime_logits.detach().cpu().numpy() - expected_logits)))
    if descriptor_delta > descriptor_atol or logits_delta > logits_atol:
        raise RuntimeError("cached predictor arithmetic diverges from the Miowtion runtime")

    reference_tiles = int(np.count_nonzero(case.key_segments == REFERENCE_SEGMENT))
    target_tiles = int(np.count_nonzero(case.key_segments == TARGET_SEGMENT))
    video_tiles = reference_tiles + target_tiles
    if not np.array_equal(
        case.key_segments,
        np.concatenate(
            (
                np.full(reference_tiles, REFERENCE_SEGMENT, dtype=np.int8),
                np.full(target_tiles, TARGET_SEGMENT, dtype=np.int8),
                np.full(len(case.key_segments) - video_tiles, GLOBAL_SEGMENT, dtype=np.int8),
            )
        ),
    ):
        raise ValueError("cached key segments are not in runtime order")
    layout = SimpleNamespace(
        n_tiles=len(case.key_segments),
        n_ref_tiles=reference_tiles,
        n_video_tiles=video_tiles,
        valid_count=key_layout.valid_count,
        kv_ok=key_layout.kv_ok,
        target_tokens=int(case.key_valid_counts[case.key_segments == TARGET_SEGMENT].sum()),
        ref_tokens=int(case.key_valid_counts[case.key_segments == REFERENCE_SEGMENT].sum()),
    )
    rows = torch.tensor(case.query_tile_ids, dtype=torch.long, device=resolved)
    mismatches = []
    maxima = []
    means = []
    for budget in budgets:
        if reference_tiles and budget.reference_blocks <= 0:
            raise ValueError("a runtime case with references requires a reference budget")
        blocks = veda_mask.column_blocks(
            layout,
            veda_mask.Budget(tiles=budget.target_blocks),
            (veda_mask.Budget(tiles=budget.reference_blocks) if reference_tiles else None),
        )
        selection = veda_mask.select_video_blocks(runtime_logits[None], layout, blocks, rows=rows)
        runtime_mask = veda_mask.dense_block_mask(selection, layout, global_rows=False)[0]
        expected_mask = segmented_top_k(
            expected_logits,
            case.key_segments,
            case.key_valid_counts,
            target_blocks=budget.target_blocks,
            reference_blocks=budget.reference_blocks,
            required_keys=case.required_keys,
        )
        mismatch = int(np.count_nonzero(runtime_mask.cpu().numpy() != expected_mask))
        mismatches.append(mismatch)
        runtime_output = reference.block_sparse_attention(
            q.reshape(-1, 1, architecture.head_dim),
            k.reshape(-1, 1, architecture.head_dim),
            v.reshape(-1, 1, architecture.head_dim),
            runtime_mask[None],
            key_layout.valid_count,
        ).reshape(len(case.query_tile_ids), 128, 1, architecture.head_dim)[:, :, 0]
        expected_output = reference_attention(
            case.q,
            case.k,
            case.v,
            expected_mask,
            query_valid_counts=case.query_valid_counts,
            key_valid_counts=case.key_valid_counts,
        ).astype(np.float16)
        valid = np.stack([np.arange(128) < int(count) for count in case.query_valid_counts])
        delta = np.abs(runtime_output.cpu().numpy()[valid] - expected_output[valid])
        maxima.append(float(delta.max()))
        means.append(float(delta.mean()))
    result = RuntimeTraceVerification(
        descriptor_delta,
        logits_delta,
        tuple(mismatches),
        tuple(maxima),
        tuple(means),
    )
    if (
        any(result.mask_mismatches)
        or any(value > output_atol for value in result.output_max_abs)
        or any(value > output_mean_atol for value in result.output_mean_abs)
    ):
        raise RuntimeError(
            "cached mask or sparse output diverges from the Miowtion runtime: "
            f"mask_mismatches={result.mask_mismatches}, "
            f"output_max_abs={result.output_max_abs}, "
            f"output_mean_abs={result.output_mean_abs}"
        )
    return result


def _to_float16(value: Any) -> np.ndarray:
    return value.to(dtype=value.new_empty(()).half().dtype).cpu().numpy()


def _pool_layout(counts: np.ndarray, device: Any, torch: Any) -> Any:
    valid_count = torch.as_tensor(counts, dtype=torch.long, device=device)
    return SimpleNamespace(
        n_tiles=len(counts),
        valid_count=valid_count,
        partial_tiles=torch.nonzero(valid_count < 128, as_tuple=False).flatten(),
        kv_ok=valid_count > 0,
    )


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def trajectory_tensor_key(step_index: int) -> str:
    if type(step_index) is not int or step_index < 0:
        raise ValueError("trajectory step index must be non-negative")
    return f"video_step_{step_index:03d}_rows"


def _tile_plan_hash(shape: Any, layout: Any, heads: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"veda-tile-plan-trace-v1\0")
    digest.update(str(shape).encode("ascii"))
    digest.update(np.asarray(heads, dtype=np.int16).tobytes())
    digest.update(
        layout.perm.to(dtype=layout.perm.new_empty(()).long().dtype).cpu().numpy().tobytes()
    )
    digest.update(layout.valid_count.cpu().numpy().tobytes())
    digest.update(layout.n_ref_tiles.to_bytes(4, "big"))
    digest.update(layout.n_video_tiles.to_bytes(4, "big"))
    return digest.hexdigest()
