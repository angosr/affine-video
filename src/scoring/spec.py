"""Strict loading and canonical hashing of one frozen evaluation track."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import ceil, fsum, isfinite, prod
from pathlib import Path
from typing import Any

from scoring.model import TensorSpec
from scoring.selector import SelectorArchitecture

_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_TRACK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

VEDA_REPO_ID = "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview"
VEDA_REVISION = "76f202874608115408d73280be9531c4ab888242"
VEDA_FILENAME = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
VEDA_FILE_SHA256 = "2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a"
VEDA_EVALUATION_FILE_SHA256 = "64addb5da430059e37e405d28ca0f2887132905f077ae92fba86982c8b0e9d3f"
VEDA_MODEL_DIGEST = "3fef3eca0eebee238b1ed9c692b217dd25cb1082ef94779ec2ef7c93ae0560c9"

EXACT_CACHE_FORMAT = "veda-exact-cache-v2"
PRIVATE_GENERATOR = "veda-private-rollout-v1"
STEP_SELECTION_ALGORITHM = "sha256-window-without-replacement-v1"
QUERY_SELECTION_ALGORITHM = "sha256-segment-quota-v1"


class TrackSpecError(ValueError):
    """Raised when a track specification is ambiguous or incomplete."""


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    target_blocks: int
    reference_blocks: int
    calibrated_cost: float


@dataclass(frozen=True, slots=True)
class ModelSpec:
    filename: str
    max_bytes: int
    selector: SelectorArchitecture

    @property
    def tensor_schema(self) -> dict[str, TensorSpec]:
        return self.selector.tensor_schema


@dataclass(frozen=True, slots=True)
class BaseModelSpec:
    repo_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class InitialChampionSpec:
    uid: int
    hotkey: str
    repo_id: str
    revision: str
    filename: str
    file_sha256: str
    evaluation_file_sha256: str
    model_digest: str
    score: float


@dataclass(frozen=True, slots=True)
class CacheSpec:
    exact_max_bytes: int
    minimum_generation_delay_blocks: int
    generation_interval_blocks: int
    generator: str
    format: str
    storage_dtype: str
    compute_dtype: str
    accumulation_dtype: str


@dataclass(frozen=True, slots=True)
class StepWindowSpec:
    name: str
    indices: tuple[int, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class TrajectoryScheduleSpec:
    name: str
    sigmas: tuple[float, ...]
    step_windows: tuple[StepWindowSpec, ...]


@dataclass(frozen=True, slots=True)
class FullVideoSpec:
    minimum_gain: float
    metric_version: str
    feature_model_sha256: str
    latent_weight: float
    frame_weight: float
    temporal_weight: float
    black_frame_threshold: float
    flicker_threshold: float


@dataclass(frozen=True, slots=True)
class ScoringSpec:
    budgets: tuple[BudgetSpec, ...]
    sampled_route_count: int
    rollouts_per_stratum: int
    sampled_step_count: int
    query_tiles_per_case: int
    step_selection_algorithm: str
    query_tile_selection_algorithm: str
    trajectory_schedule: TrajectoryScheduleSpec
    tail_mean_weight: float
    stratum_weights: tuple[tuple[str, float], ...]
    minimum_frontier_gain: float
    season_false_promotion_alpha: float
    bootstrap_resamples: int
    bootstrap_seed: int

    @property
    def stratum_weight_map(self) -> dict[str, float]:
        return dict(self.stratum_weights)

    @property
    def case_count(self) -> int:
        return (
            self.sampled_route_count
            * self.rollouts_per_stratum
            * self.sampled_step_count
            * len(self.stratum_weights)
        )

    @property
    def step_window_weight_map(self) -> dict[str, float]:
        return {window.name: window.weight for window in self.trajectory_schedule.step_windows}


@dataclass(frozen=True, slots=True)
class TrackSpec:
    track_id: str
    netuid: int
    mechid: int
    track_start_block: int
    publication_grace_blocks: int
    maximum_generation_batches: int
    maximum_effective_challenges: int
    track_close_grace_blocks: int
    source_pool_merkle_root: str
    artifact_manifest_sha256: str
    initial_champion: InitialChampionSpec
    model: ModelSpec
    base_model: BaseModelSpec
    cache: CacheSpec
    scoring: ScoringSpec
    full_video: FullVideoSpec
    digest: str

    @property
    def alpha_per_match(self) -> float:
        return self.scoring.season_false_promotion_alpha / self.maximum_effective_challenges

    @property
    def match_confidence(self) -> float:
        return 1.0 - self.alpha_per_match

    @property
    def last_generation_block(self) -> int:
        return (
            self.track_start_block
            + self.maximum_generation_batches * self.cache.generation_interval_blocks
        )

    @property
    def last_submission_block(self) -> int:
        return (
            self.last_generation_block
            - self.cache.generation_interval_blocks
            - self.cache.minimum_generation_delay_blocks
        )

    @property
    def track_close_block(self) -> int:
        return self.last_generation_block + self.track_close_grace_blocks


def track_spec_dict(spec: TrackSpec) -> dict[str, Any]:
    """Return the canonical public document whose SHA-256 is ``spec.digest``."""

    document = asdict(spec)
    document.pop("digest")
    document["model"]["selector"] = {
        "num_layers": spec.model.selector.num_layers,
        "num_heads": spec.model.selector.num_heads,
        "head_dim": spec.model.selector.head_dim,
        "storage_dtype": spec.model.selector.storage_dtype,
    }
    document["scoring"]["stratum_weights"] = dict(spec.scoring.stratum_weights)
    document["scoring"]["budgets"] = [asdict(budget) for budget in spec.scoring.budgets]
    schedule = document["scoring"]["trajectory_schedule"]
    schedule["sigmas"] = list(spec.scoring.trajectory_schedule.sigmas)
    schedule["step_windows"] = [
        {"indices": list(window.indices), "name": window.name, "weight": window.weight}
        for window in spec.scoring.trajectory_schedule.step_windows
    ]
    return document


def load_track_spec(path: str | Path) -> TrackSpec:
    """Load one JSON spec, rejecting duplicate keys before validation."""

    try:
        text = Path(path).read_text(encoding="utf-8")
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except OSError as exc:
        raise TrackSpecError("track specification cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise TrackSpecError("track specification is not valid JSON") from exc
    return parse_track_spec(raw)


def parse_track_spec(raw: Any) -> TrackSpec:
    """Validate every score-affecting field and compute its canonical hash."""

    root = _object(
        raw,
        {
            "artifact_manifest_sha256",
            "base_model",
            "cache",
            "full_video",
            "initial_champion",
            "maximum_effective_challenges",
            "maximum_generation_batches",
            "mechid",
            "model",
            "netuid",
            "publication_grace_blocks",
            "scoring",
            "source_pool_merkle_root",
            "track_close_grace_blocks",
            "track_id",
            "track_start_block",
        },
        "track specification",
    )
    track_id = _nonempty_string(root["track_id"], "track_id")
    if not _TRACK_ID.fullmatch(track_id):
        raise TrackSpecError("track_id must contain only lowercase letters, digits, and hyphens")
    netuid = _integer(root["netuid"], "netuid", minimum=0)
    if netuid > 0xFFFF:
        raise TrackSpecError("netuid must fit in an unsigned 16-bit integer")
    maximum_effective = _integer(
        root["maximum_effective_challenges"], "maximum_effective_challenges", minimum=1
    )
    values = {
        "track_id": track_id,
        "netuid": netuid,
        "mechid": _integer(root["mechid"], "mechid", minimum=0),
        "track_start_block": _integer(root["track_start_block"], "track_start_block", minimum=0),
        "publication_grace_blocks": _integer(
            root["publication_grace_blocks"], "publication_grace_blocks", minimum=1
        ),
        "maximum_generation_batches": _integer(
            root["maximum_generation_batches"], "maximum_generation_batches", minimum=2
        ),
        "maximum_effective_challenges": maximum_effective,
        "track_close_grace_blocks": _integer(
            root["track_close_grace_blocks"], "track_close_grace_blocks", minimum=1
        ),
        "source_pool_merkle_root": _digest(
            root["source_pool_merkle_root"], "source_pool_merkle_root"
        ),
        "artifact_manifest_sha256": _digest(
            root["artifact_manifest_sha256"], "artifact_manifest_sha256"
        ),
        "initial_champion": _parse_initial(root["initial_champion"]),
        "model": _parse_model(root["model"]),
        "base_model": _parse_base_model(root["base_model"]),
        "cache": _parse_cache(root["cache"]),
        "full_video": _parse_full_video(root["full_video"]),
    }
    if values["cache"].minimum_generation_delay_blocks <= values["publication_grace_blocks"]:
        raise TrackSpecError(
            "minimum generation delay must exceed the model publication grace period"
        )
    values["scoring"] = _parse_scoring(root["scoring"], values["model"].selector, maximum_effective)
    if values["full_video"].minimum_gain < values["scoring"].minimum_frontier_gain:
        raise TrackSpecError("full-video minimum gain cannot be lower than the exact-stage gate")
    canonical = _spec_document(values)
    return TrackSpec(**values, digest=hashlib.sha256(_canonical_json(canonical)).hexdigest())


def _parse_initial(value: Any) -> InitialChampionSpec:
    raw = _object(
        value,
        {
            "evaluation_file_sha256",
            "file_sha256",
            "filename",
            "hotkey",
            "model_digest",
            "repo_id",
            "revision",
            "score",
            "uid",
        },
        "initial_champion",
    )
    result = InitialChampionSpec(
        _integer(raw["uid"], "initial champion uid", minimum=0),
        _nonempty_string(raw["hotkey"], "initial champion hotkey"),
        _nonempty_string(raw["repo_id"], "initial champion repo_id"),
        _nonempty_string(raw["revision"], "initial champion revision").lower(),
        _nonempty_string(raw["filename"], "initial champion filename"),
        _digest(raw["file_sha256"], "initial champion file_sha256"),
        _digest(raw["evaluation_file_sha256"], "initial champion evaluation_file_sha256"),
        _digest(raw["model_digest"], "initial champion model_digest"),
        _unit(raw["score"], "initial champion score"),
    )
    if result.uid != 0:
        raise TrackSpecError("initial champion uid must be 0")
    actual = (
        result.repo_id,
        result.revision,
        result.filename,
        result.file_sha256,
        result.evaluation_file_sha256,
        result.model_digest,
    )
    expected = (
        VEDA_REPO_ID,
        VEDA_REVISION,
        VEDA_FILENAME,
        VEDA_FILE_SHA256,
        VEDA_EVALUATION_FILE_SHA256,
        VEDA_MODEL_DIGEST,
    )
    if actual != expected:
        raise TrackSpecError("initial champion must be the pinned official VEDA model")
    return result


def _parse_model(value: Any) -> ModelSpec:
    raw = _object(value, {"filename", "max_bytes", "selector"}, "model")
    if raw["filename"] != "selector.safetensors":
        raise TrackSpecError("model filename must be selector.safetensors")
    item = _object(
        raw["selector"], {"head_dim", "num_heads", "num_layers", "storage_dtype"}, "selector"
    )
    try:
        selector = SelectorArchitecture(
            _integer(item["num_layers"], "selector num_layers", minimum=1),
            _integer(item["num_heads"], "selector num_heads", minimum=1),
            _integer(item["head_dim"], "selector head_dim", minimum=1),
            _nonempty_string(item["storage_dtype"], "selector storage_dtype"),
        )
    except ValueError as exc:
        raise TrackSpecError(str(exc)) from exc
    max_bytes = _integer(raw["max_bytes"], "model max_bytes", minimum=1)
    if _minimum_safetensors_bytes(selector) > max_bytes:
        raise TrackSpecError("model max_bytes cannot hold the fixed selector tensors")
    return ModelSpec("selector.safetensors", max_bytes, selector)


def _parse_base_model(value: Any) -> BaseModelSpec:
    raw = _object(value, {"repo_id", "revision"}, "base_model")
    repo = _nonempty_string(raw["repo_id"], "base model repo_id")
    if repo.count("/") != 1 or any(not part for part in repo.split("/")):
        raise TrackSpecError("base model repo_id must use owner/repository form")
    revision = _nonempty_string(raw["revision"], "base model revision")
    if not _REVISION.fullmatch(revision):
        raise TrackSpecError("base model revision must be a complete commit SHA")
    return BaseModelSpec(repo, revision.lower())


def _parse_cache(value: Any) -> CacheSpec:
    raw = _object(
        value,
        {
            "accumulation_dtype",
            "compute_dtype",
            "exact_max_bytes",
            "format",
            "generation_interval_blocks",
            "generator",
            "minimum_generation_delay_blocks",
            "storage_dtype",
        },
        "cache",
    )
    generator = _nonempty_string(raw["generator"], "cache generator")
    if generator != PRIVATE_GENERATOR:
        raise TrackSpecError(f"cache generator must be {PRIVATE_GENERATOR}")
    cache_format = _nonempty_string(raw["format"], "cache format")
    if cache_format != EXACT_CACHE_FORMAT:
        raise TrackSpecError(f"cache format must be {EXACT_CACHE_FORMAT}")
    return CacheSpec(
        _integer(raw["exact_max_bytes"], "exact cache max bytes", minimum=1),
        _integer(
            raw["minimum_generation_delay_blocks"],
            "minimum generation delay blocks",
            minimum=1,
        ),
        _integer(raw["generation_interval_blocks"], "generation interval blocks", minimum=1),
        generator,
        cache_format,
        _choice(raw["storage_dtype"], "storage_dtype", {"float16"}),
        _choice(raw["compute_dtype"], "compute_dtype", {"float32"}),
        _choice(raw["accumulation_dtype"], "accumulation_dtype", {"float32"}),
    )


def _parse_scoring(
    value: Any,
    selector: SelectorArchitecture,
    maximum_effective: int,
) -> ScoringSpec:
    raw = _object(
        value,
        {
            "bootstrap_resamples",
            "bootstrap_seed",
            "budgets",
            "minimum_frontier_gain",
            "query_tile_selection_algorithm",
            "query_tiles_per_case",
            "rollouts_per_stratum",
            "sampled_route_count",
            "sampled_step_count",
            "season_false_promotion_alpha",
            "step_selection_algorithm",
            "stratum_weights",
            "tail_mean_weight",
            "trajectory_schedule",
        },
        "scoring",
    )
    raw_budgets = raw["budgets"]
    if not isinstance(raw_budgets, list) or len(raw_budgets) < 3:
        raise TrackSpecError("scoring requires at least three budgets")
    budgets = tuple(
        BudgetSpec(
            _integer(item["target_blocks"], "target blocks", minimum=1),
            _integer(item["reference_blocks"], "reference blocks", minimum=0),
            _number(item["calibrated_cost"], "budget calibrated_cost", minimum_exclusive=0),
        )
        for item in (
            _object(row, {"calibrated_cost", "reference_blocks", "target_blocks"}, "budget")
            for row in raw_budgets
        )
    )
    if any(
        right.target_blocks <= left.target_blocks or right.reference_blocks < left.reference_blocks
        for left, right in pairwise(budgets)
    ):
        raise TrackSpecError("budget block counts must increase monotonically")
    if any(right.calibrated_cost <= left.calibrated_cost for left, right in pairwise(budgets)):
        raise TrackSpecError("budget costs must be strictly increasing")
    routes = _integer(raw["sampled_route_count"], "sampled_route_count", minimum=1)
    if routes > len(selector.routes):
        raise TrackSpecError("sampled_route_count cannot exceed available selector routes")
    schedule = _parse_schedule(raw["trajectory_schedule"])
    sampled_steps = _integer(raw["sampled_step_count"], "sampled_step_count", minimum=1)
    if sampled_steps < len(schedule.step_windows) or sampled_steps > len(schedule.sigmas) - 1:
        raise TrackSpecError("sampled_step_count must cover every window within the schedule")
    step_algorithm = _nonempty_string(raw["step_selection_algorithm"], "step algorithm")
    query_algorithm = _nonempty_string(raw["query_tile_selection_algorithm"], "query algorithm")
    if step_algorithm != STEP_SELECTION_ALGORITHM:
        raise TrackSpecError(f"step selection algorithm must be {STEP_SELECTION_ALGORITHM}")
    if query_algorithm != QUERY_SELECTION_ALGORITHM:
        raise TrackSpecError(f"query tile algorithm must be {QUERY_SELECTION_ALGORITHM}")
    season_alpha = _unit(
        raw["season_false_promotion_alpha"],
        "season_false_promotion_alpha",
        include_zero=False,
        include_one=False,
    )
    resamples = _integer(raw["bootstrap_resamples"], "bootstrap_resamples", minimum=1)
    if resamples < ceil(10 / (season_alpha / maximum_effective)):
        raise TrackSpecError("bootstrap_resamples is too small for the season error tail")
    return ScoringSpec(
        budgets,
        routes,
        _integer(raw["rollouts_per_stratum"], "rollouts_per_stratum", minimum=4),
        sampled_steps,
        _integer(raw["query_tiles_per_case"], "query_tiles_per_case", minimum=1),
        step_algorithm,
        query_algorithm,
        schedule,
        _unit(raw["tail_mean_weight"], "tail_mean_weight"),
        _weights(raw["stratum_weights"], "stratum"),
        _unit(raw["minimum_frontier_gain"], "minimum_frontier_gain", include_zero=False),
        season_alpha,
        resamples,
        _integer(raw["bootstrap_seed"], "bootstrap_seed", minimum=0),
    )


def _parse_schedule(value: Any) -> TrajectoryScheduleSpec:
    raw = _object(value, {"name", "sigmas", "step_windows"}, "trajectory_schedule")
    raw_sigmas = raw["sigmas"]
    if not isinstance(raw_sigmas, list) or len(raw_sigmas) < 2:
        raise TrackSpecError("trajectory sigmas must contain at least two states")
    sigmas = tuple(_unit(item, "trajectory sigma") for item in raw_sigmas)
    if any(right >= left for left, right in pairwise(sigmas)) or sigmas[-1] != 0:
        raise TrackSpecError("trajectory sigmas must strictly decrease to zero")
    raw_windows = raw["step_windows"]
    if not isinstance(raw_windows, list) or not raw_windows:
        raise TrackSpecError("trajectory step_windows must be a non-empty list")
    windows = []
    for row in raw_windows:
        item = _object(row, {"indices", "name", "weight"}, "step window")
        raw_indices = item["indices"]
        if not isinstance(raw_indices, list) or not raw_indices:
            raise TrackSpecError("step window indices must be a non-empty list")
        indices = tuple(_integer(index, "step index", minimum=0) for index in raw_indices)
        if len(set(indices)) != len(indices) or tuple(sorted(indices)) != indices:
            raise TrackSpecError("step window indices must be sorted and unique")
        windows.append(
            StepWindowSpec(
                _nonempty_string(item["name"], "step window name"),
                indices,
                _number(item["weight"], "step window weight", minimum_exclusive=0),
            )
        )
    if len({window.name for window in windows}) != len(windows):
        raise TrackSpecError("step window names must be unique")
    if abs(fsum(window.weight for window in windows) - 1.0) > 1e-12:
        raise TrackSpecError("step window weights must sum to one")
    covered = [index for window in windows for index in window.indices]
    if sorted(covered) != list(range(len(sigmas) - 1)):
        raise TrackSpecError("step windows must partition every denoising step")
    return TrajectoryScheduleSpec(
        _nonempty_string(raw["name"], "trajectory schedule name"),
        sigmas,
        tuple(windows),
    )


def _parse_full_video(value: Any) -> FullVideoSpec:
    raw = _object(
        value,
        {
            "black_frame_threshold",
            "feature_model_sha256",
            "flicker_threshold",
            "frame_weight",
            "latent_weight",
            "metric_version",
            "minimum_gain",
            "temporal_weight",
        },
        "full_video",
    )
    weights = tuple(
        _unit(raw[name], name) for name in ("latent_weight", "frame_weight", "temporal_weight")
    )
    if abs(fsum(weights) - 1.0) > 1e-12:
        raise TrackSpecError("full-video metric weights must sum to one")
    metric_version = _nonempty_string(raw["metric_version"], "full-video metric version")
    if metric_version not in {"veda-video-fidelity-v1", "veda-video-fidelity-v2"}:
        raise TrackSpecError("full-video metric version is unsupported")
    return FullVideoSpec(
        _unit(raw["minimum_gain"], "full-video minimum gain", include_zero=False),
        metric_version,
        _digest(raw["feature_model_sha256"], "full-video feature model SHA-256"),
        weights[0],
        weights[1],
        weights[2],
        _unit(raw["black_frame_threshold"], "black frame threshold"),
        _number(raw["flicker_threshold"], "flicker threshold", minimum_exclusive=0),
    )


def _spec_document(values: dict[str, Any]) -> dict[str, Any]:
    return track_spec_dict(TrackSpec(**values, digest=""))


def _weights(value: Any, name: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, dict) or not value:
        raise TrackSpecError(f"{name} weights must be a non-empty object")
    result = tuple(
        (
            _nonempty_string(key, f"{name} name"),
            _number(weight, f"{name} {key} weight", minimum_exclusive=0),
        )
        for key, weight in sorted(value.items())
    )
    if abs(fsum(weight for _, weight in result) - 1.0) > 1e-12:
        raise TrackSpecError(f"{name} weights must sum to one")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrackSpecError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise TrackSpecError(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise TrackSpecError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _number(value: Any, name: str, *, minimum_exclusive: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackSpecError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or result <= minimum_exclusive:
        raise TrackSpecError(f"{name} must be greater than {minimum_exclusive}")
    return result


def _unit(value: Any, name: str, *, include_zero: bool = True, include_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackSpecError(f"{name} must be numeric")
    result = float(value)
    lower = result >= 0 if include_zero else result > 0
    upper = result <= 1 if include_one else result < 1
    if not isfinite(result) or not lower or not upper:
        raise TrackSpecError(f"{name} must be within the configured unit interval")
    return result


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _nonempty_string(value, name)
    if result not in choices:
        raise TrackSpecError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrackSpecError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any, name: str) -> str:
    text = _nonempty_string(value, name).lower()
    if len(text) != 64:
        raise TrackSpecError(f"{name} must contain 32 bytes")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise TrackSpecError(f"{name} must be hexadecimal") from exc
    return text


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _minimum_safetensors_bytes(selector: SelectorArchitecture) -> int:
    item_size = 2 if selector.storage_dtype == "float16" else 4
    dtype = "F16" if selector.storage_dtype == "float16" else "F32"
    offset = 0
    header: dict[str, Any] = {}
    for name, tensor in sorted(selector.tensor_schema.items()):
        size = prod(tensor.shape) * item_size
        header[name] = {
            "data_offsets": [offset, offset + size],
            "dtype": dtype,
            "shape": list(tensor.shape),
        }
        offset += size
    encoded_header = _canonical_json(header)
    return 8 + (len(encoded_header) + 7) // 8 * 8 + offset
