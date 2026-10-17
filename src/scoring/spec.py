"""Strict loading and canonical hashing of a frozen track specification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import pairwise
from math import fsum, isfinite, prod
from pathlib import Path
from typing import Any

from scoring.model import TensorSpec
from scoring.selector import SelectorArchitecture

_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")

VEDA_REPO_ID = "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview"
VEDA_REVISION = "76f202874608115408d73280be9531c4ab888242"
VEDA_FILENAME = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
VEDA_FILE_SHA256 = "2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a"
VEDA_EVALUATION_FILE_SHA256 = (
    "64addb5da430059e37e405d28ca0f2887132905f077ae92fba86982c8b0e9d3f"
)
VEDA_MODEL_DIGEST = "3fef3eca0eebee238b1ed9c692b217dd25cb1082ef94779ec2ef7c93ae0560c9"


class TrackSpecError(ValueError):
    """Raised when a track specification is ambiguous or incomplete."""


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    blocks: int
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


@dataclass(frozen=True, slots=True)
class ScoringSpec:
    budgets: tuple[BudgetSpec, ...]
    sample_count: int
    tail_mean_weight: float
    stratum_weights: tuple[tuple[str, float], ...]
    minimum_frontier_gain: float
    bootstrap_confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int

    @property
    def stratum_weight_map(self) -> dict[str, float]:
        return dict(self.stratum_weights)


@dataclass(frozen=True, slots=True)
class TrackSpec:
    track_id: str
    netuid: int
    track_start_block: int
    publication_grace_blocks: int
    initial_champion: InitialChampionSpec
    model: ModelSpec
    base_model: BaseModelSpec
    cache: CacheSpec
    scoring: ScoringSpec
    digest: str


def track_spec_dict(spec: TrackSpec) -> dict[str, Any]:
    """Return the canonical public document whose SHA-256 is ``spec.digest``."""

    return {
        "track_id": spec.track_id,
        "netuid": spec.netuid,
        "track_start_block": spec.track_start_block,
        "publication_grace_blocks": spec.publication_grace_blocks,
        "initial_champion": {
            "uid": spec.initial_champion.uid,
            "hotkey": spec.initial_champion.hotkey,
            "repo_id": spec.initial_champion.repo_id,
            "revision": spec.initial_champion.revision,
            "filename": spec.initial_champion.filename,
            "file_sha256": spec.initial_champion.file_sha256,
            "evaluation_file_sha256": spec.initial_champion.evaluation_file_sha256,
            "model_digest": spec.initial_champion.model_digest,
            "score": spec.initial_champion.score,
        },
        "model": {
            "filename": spec.model.filename,
            "max_bytes": spec.model.max_bytes,
            "selector": {
                "num_layers": spec.model.selector.num_layers,
                "num_heads": spec.model.selector.num_heads,
                "head_dim": spec.model.selector.head_dim,
                "storage_dtype": spec.model.selector.storage_dtype,
            },
        },
        "base_model": {
            "repo_id": spec.base_model.repo_id,
            "revision": spec.base_model.revision,
        },
        "cache": {
            "exact_max_bytes": spec.cache.exact_max_bytes,
            "minimum_generation_delay_blocks": spec.cache.minimum_generation_delay_blocks,
        },
        "scoring": {
            "budgets": [
                {"blocks": budget.blocks, "calibrated_cost": budget.calibrated_cost}
                for budget in spec.scoring.budgets
            ],
            "sample_count": spec.scoring.sample_count,
            "tail_mean_weight": spec.scoring.tail_mean_weight,
            "stratum_weights": dict(spec.scoring.stratum_weights),
            "minimum_frontier_gain": spec.scoring.minimum_frontier_gain,
            "bootstrap_confidence": spec.scoring.bootstrap_confidence,
            "bootstrap_resamples": spec.scoring.bootstrap_resamples,
            "bootstrap_seed": spec.scoring.bootstrap_seed,
        },
    }


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
    """Validate all score-affecting fields and compute their canonical hash."""

    root = _object(
        raw,
        {
            "track_id",
            "netuid",
            "track_start_block",
            "publication_grace_blocks",
            "initial_champion",
            "model",
            "base_model",
            "cache",
            "scoring",
        },
        "track specification",
    )
    track_id = _nonempty_string(root["track_id"], "track_id")
    netuid = _integer(root["netuid"], "netuid", minimum=0)
    if netuid > 0xFFFF:
        raise TrackSpecError("netuid must fit in an unsigned 16-bit integer")

    initial_raw = _object(
        root["initial_champion"],
        {
            "uid",
            "hotkey",
            "repo_id",
            "revision",
            "filename",
            "file_sha256",
            "evaluation_file_sha256",
            "model_digest",
            "score",
        },
        "initial_champion",
    )
    initial_uid = _integer(initial_raw["uid"], "initial champion uid", minimum=0)
    if initial_uid != 0:
        raise TrackSpecError("initial champion uid must be 0")
    initial_hotkey = _nonempty_string(initial_raw["hotkey"], "initial champion hotkey")
    initial_repo = _nonempty_string(initial_raw["repo_id"], "initial champion repo_id")
    initial_revision = _nonempty_string(
        initial_raw["revision"], "initial champion revision"
    ).lower()
    initial_filename = _nonempty_string(
        initial_raw["filename"], "initial champion filename"
    )
    initial_sha256 = _digest(initial_raw["file_sha256"], "initial champion file_sha256")
    initial_evaluation_sha256 = _digest(
        initial_raw["evaluation_file_sha256"],
        "initial champion evaluation_file_sha256",
    )
    initial_model_digest = _digest(
        initial_raw["model_digest"], "initial champion model_digest"
    )
    if (
        initial_repo != VEDA_REPO_ID
        or initial_revision != VEDA_REVISION
        or initial_filename != VEDA_FILENAME
        or initial_sha256 != VEDA_FILE_SHA256
        or initial_evaluation_sha256 != VEDA_EVALUATION_FILE_SHA256
        or initial_model_digest != VEDA_MODEL_DIGEST
    ):
        raise TrackSpecError("initial champion must be the pinned official VEDA model")
    initial_score = _unit(initial_raw["score"], "initial champion score")

    model_raw = _object(root["model"], {"filename", "max_bytes", "selector"}, "model")
    if model_raw["filename"] != "selector.safetensors":
        raise TrackSpecError("model filename must be selector.safetensors")
    selector_raw = _object(
        model_raw["selector"],
        {
            "num_layers",
            "num_heads",
            "head_dim",
            "storage_dtype",
        },
        "selector",
    )
    try:
        selector = SelectorArchitecture(
            num_layers=_integer(selector_raw["num_layers"], "selector num_layers", minimum=1),
            num_heads=_integer(selector_raw["num_heads"], "selector num_heads", minimum=1),
            head_dim=_integer(selector_raw["head_dim"], "selector head_dim", minimum=1),
            storage_dtype=_nonempty_string(selector_raw["storage_dtype"], "selector storage_dtype"),
        )
    except ValueError as exc:
        raise TrackSpecError(str(exc)) from exc
    max_model_bytes = _integer(model_raw["max_bytes"], "model max_bytes", minimum=1)
    if _minimum_safetensors_bytes(selector) > max_model_bytes:
        raise TrackSpecError("model max_bytes cannot hold the fixed selector tensors")

    base_raw = _object(root["base_model"], {"repo_id", "revision"}, "base_model")
    repo_id = _nonempty_string(base_raw["repo_id"], "base model repo_id")
    if repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
        raise TrackSpecError("base model repo_id must use owner/repository form")
    revision = _nonempty_string(base_raw["revision"], "base model revision")
    if not _REVISION.fullmatch(revision):
        raise TrackSpecError("base model revision must be a complete commit SHA")

    cache_raw = _object(
        root["cache"],
        {"exact_max_bytes", "minimum_generation_delay_blocks"},
        "cache",
    )

    scoring_raw = _object(
        root["scoring"],
        {
            "budgets",
            "sample_count",
            "tail_mean_weight",
            "stratum_weights",
            "minimum_frontier_gain",
            "bootstrap_confidence",
            "bootstrap_resamples",
            "bootstrap_seed",
        },
        "scoring",
    )
    budgets_raw = scoring_raw["budgets"]
    if not isinstance(budgets_raw, list) or len(budgets_raw) < 3:
        raise TrackSpecError("scoring requires at least three budgets")
    budget_items: list[BudgetSpec] = []
    for item in budgets_raw:
        budget = _object(item, {"blocks", "calibrated_cost"}, "budget")
        budget_items.append(
            BudgetSpec(
                blocks=_integer(budget["blocks"], "budget blocks", minimum=1),
                calibrated_cost=_number(
                    budget["calibrated_cost"],
                    "budget calibrated_cost",
                    minimum_exclusive=0,
                ),
            )
        )
    budgets = tuple(budget_items)
    if any(right.blocks <= left.blocks for left, right in pairwise(budgets)):
        raise TrackSpecError("budget block counts must be strictly increasing")
    if any(right.calibrated_cost <= left.calibrated_cost for left, right in pairwise(budgets)):
        raise TrackSpecError("budget costs must be strictly increasing")

    strata_raw = scoring_raw["stratum_weights"]
    if not isinstance(strata_raw, dict) or not strata_raw:
        raise TrackSpecError("stratum_weights must be a non-empty object")
    strata = tuple(
        (
            _nonempty_string(name, "stratum name"),
            _number(weight, f"stratum {name} weight", minimum_exclusive=0),
        )
        for name, weight in sorted(strata_raw.items())
    )
    if abs(fsum(weight for _, weight in strata) - 1.0) > 1e-12:
        raise TrackSpecError("stratum weights must sum to one")
    sample_count = _integer(scoring_raw["sample_count"], "sample_count", minimum=1)
    if sample_count < len(selector.routes) * len(strata):
        raise TrackSpecError("sample count must cover every route and scoring stratum")

    spec_without_digest = {
        "track_id": track_id,
        "netuid": netuid,
        "track_start_block": _integer(root["track_start_block"], "track_start_block", minimum=0),
        "publication_grace_blocks": _integer(
            root["publication_grace_blocks"],
            "publication_grace_blocks",
            minimum=1,
        ),
        "initial_champion": {
            "uid": initial_uid,
            "hotkey": initial_hotkey,
            "repo_id": initial_repo,
            "revision": initial_revision,
            "filename": initial_filename,
            "file_sha256": initial_sha256,
            "evaluation_file_sha256": initial_evaluation_sha256,
            "model_digest": initial_model_digest,
            "score": initial_score,
        },
        "model": {
            "filename": "selector.safetensors",
            "max_bytes": max_model_bytes,
            "selector": {
                "num_layers": selector.num_layers,
                "num_heads": selector.num_heads,
                "head_dim": selector.head_dim,
                "storage_dtype": selector.storage_dtype,
            },
        },
        "base_model": {"repo_id": repo_id, "revision": revision.lower()},
        "cache": {
            "exact_max_bytes": _integer(
                cache_raw["exact_max_bytes"], "exact cache max bytes", minimum=1
            ),
            "minimum_generation_delay_blocks": _integer(
                cache_raw["minimum_generation_delay_blocks"],
                "minimum generation delay blocks",
                minimum=1,
            ),
        },
        "scoring": {
            "budgets": [
                {"blocks": item.blocks, "calibrated_cost": item.calibrated_cost} for item in budgets
            ],
            "sample_count": sample_count,
            "tail_mean_weight": _unit(scoring_raw["tail_mean_weight"], "tail_mean_weight"),
            "stratum_weights": dict(strata),
            "minimum_frontier_gain": _unit(
                scoring_raw["minimum_frontier_gain"],
                "minimum_frontier_gain",
                include_zero=False,
            ),
            "bootstrap_confidence": _unit(
                scoring_raw["bootstrap_confidence"],
                "bootstrap_confidence",
                include_zero=False,
                include_one=False,
            ),
            "bootstrap_resamples": _integer(
                scoring_raw["bootstrap_resamples"],
                "bootstrap_resamples",
                minimum=1,
            ),
            "bootstrap_seed": _integer(scoring_raw["bootstrap_seed"], "bootstrap_seed", minimum=0),
        },
    }
    digest = hashlib.sha256(_canonical_json(spec_without_digest)).hexdigest()
    scoring = spec_without_digest["scoring"]
    cache = spec_without_digest["cache"]
    return TrackSpec(
        track_id=track_id,
        netuid=netuid,
        track_start_block=spec_without_digest["track_start_block"],
        publication_grace_blocks=spec_without_digest["publication_grace_blocks"],
        initial_champion=InitialChampionSpec(**spec_without_digest["initial_champion"]),
        model=ModelSpec(
            filename="selector.safetensors",
            max_bytes=spec_without_digest["model"]["max_bytes"],
            selector=selector,
        ),
        base_model=BaseModelSpec(repo_id, revision.lower()),
        cache=CacheSpec(
            exact_max_bytes=cache["exact_max_bytes"],
            minimum_generation_delay_blocks=cache["minimum_generation_delay_blocks"],
        ),
        scoring=ScoringSpec(
            budgets=budgets,
            sample_count=scoring["sample_count"],
            tail_mean_weight=scoring["tail_mean_weight"],
            stratum_weights=strata,
            minimum_frontier_gain=scoring["minimum_frontier_gain"],
            bootstrap_confidence=scoring["bootstrap_confidence"],
            bootstrap_resamples=scoring["bootstrap_resamples"],
            bootstrap_seed=scoring["bootstrap_seed"],
        ),
        digest=digest,
    )


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


def _unit(
    value: Any,
    name: str,
    *,
    include_zero: bool = True,
    include_one: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackSpecError(f"{name} must be numeric")
    result = float(value)
    lower = result >= 0 if include_zero else result > 0
    upper = result <= 1 if include_one else result < 1
    if not isfinite(result) or not lower or not upper:
        raise TrackSpecError(f"{name} must be within the configured unit interval")
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
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _minimum_safetensors_bytes(selector: SelectorArchitecture) -> int:
    item_size = 2 if selector.storage_dtype == "float16" else 4
    dtype = "F16" if selector.storage_dtype == "float16" else "F32"
    offset = 0
    header: dict[str, Any] = {}
    for name, tensor in sorted(selector.tensor_schema.items()):
        size = prod(tensor.shape) * item_size
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded_header = _canonical_json(header)
    padded_header_bytes = (len(encoded_header) + 7) // 8 * 8
    return 8 + padded_header_bytes + offset
