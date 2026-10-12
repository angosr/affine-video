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
class CacheSpec:
    screen_sha256: str
    screen_max_bytes: int
    exact_sha256: str
    exact_max_bytes: int


@dataclass(frozen=True, slots=True)
class RolloutSpec:
    runner_sha256: str
    panel_sha256: str
    panel_max_bytes: int


@dataclass(frozen=True, slots=True)
class ScoringSpec:
    budgets: tuple[BudgetSpec, ...]
    local_sample_count: int
    tail_mean_weight: float
    stratum_weights: tuple[tuple[str, float], ...]
    screening_threshold: float
    rollout_threshold: float
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
    submission_cooldown_blocks: int
    model: ModelSpec
    base_model: BaseModelSpec
    cache: CacheSpec
    rollout: RolloutSpec
    scoring: ScoringSpec
    digest: str


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
            "submission_cooldown_blocks",
            "model",
            "base_model",
            "cache",
            "rollout",
            "scoring",
        },
        "track specification",
    )
    track_id = _nonempty_string(root["track_id"], "track_id")
    netuid = _integer(root["netuid"], "netuid", minimum=0)
    if netuid > 0xFFFF:
        raise TrackSpecError("netuid must fit in an unsigned 16-bit integer")

    model_raw = _object(root["model"], {"filename", "max_bytes", "selector"}, "model")
    if model_raw["filename"] != "selector.safetensors":
        raise TrackSpecError("model filename must be selector.safetensors")
    selector_raw = _object(
        model_raw["selector"],
        {
            "routes",
            "num_layers",
            "num_heads",
            "head_dim",
            "storage_dtype",
        },
        "selector",
    )
    routes_raw = selector_raw["routes"]
    if not isinstance(routes_raw, list) or not routes_raw:
        raise TrackSpecError("selector routes must be a non-empty list")
    routes = tuple(_nonempty_string(route, "selector route") for route in routes_raw)
    try:
        selector = SelectorArchitecture(
            num_layers=_integer(selector_raw["num_layers"], "selector num_layers", minimum=1),
            num_heads=_integer(selector_raw["num_heads"], "selector num_heads", minimum=1),
            head_dim=_integer(selector_raw["head_dim"], "selector head_dim", minimum=1),
            routes=routes,
            storage_dtype=_nonempty_string(selector_raw["storage_dtype"], "selector storage_dtype"),
        )
    except ValueError as exc:
        raise TrackSpecError(str(exc)) from exc
    max_model_bytes = _integer(model_raw["max_bytes"], "model max_bytes", minimum=1)
    item_size = 2 if selector.storage_dtype == "float16" else 4
    tensor_bytes = sum(prod(tensor.shape) * item_size for tensor in selector.tensor_schema.values())
    if tensor_bytes + 8 >= max_model_bytes:
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
        {"screen_sha256", "screen_max_bytes", "exact_sha256", "exact_max_bytes"},
        "cache",
    )
    rollout_raw = _object(
        root["rollout"],
        {"runner_sha256", "panel_sha256", "panel_max_bytes"},
        "rollout",
    )

    scoring_raw = _object(
        root["scoring"],
        {
            "budgets",
            "local_sample_count",
            "tail_mean_weight",
            "stratum_weights",
            "screening_threshold",
            "rollout_threshold",
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
    local_sample_count = _integer(
        scoring_raw["local_sample_count"], "local_sample_count", minimum=1
    )
    if local_sample_count < len(routes) * len(strata):
        raise TrackSpecError("local sample count must cover every route and scoring stratum")

    spec_without_digest = {
        "track_id": track_id,
        "netuid": netuid,
        "track_start_block": _integer(root["track_start_block"], "track_start_block", minimum=0),
        "submission_cooldown_blocks": _integer(
            root["submission_cooldown_blocks"],
            "submission_cooldown_blocks",
            minimum=0,
        ),
        "model": {
            "filename": "selector.safetensors",
            "max_bytes": max_model_bytes,
            "selector": {
                "routes": list(selector.routes),
                "num_layers": selector.num_layers,
                "num_heads": selector.num_heads,
                "head_dim": selector.head_dim,
                "storage_dtype": selector.storage_dtype,
            },
        },
        "base_model": {"repo_id": repo_id, "revision": revision.lower()},
        "cache": {
            "screen_sha256": _digest(cache_raw["screen_sha256"], "screen cache sha256"),
            "screen_max_bytes": _integer(
                cache_raw["screen_max_bytes"], "screen cache max bytes", minimum=1
            ),
            "exact_sha256": _digest(cache_raw["exact_sha256"], "exact cache sha256"),
            "exact_max_bytes": _integer(
                cache_raw["exact_max_bytes"], "exact cache max bytes", minimum=1
            ),
        },
        "rollout": {
            "runner_sha256": _digest(rollout_raw["runner_sha256"], "rollout runner sha256"),
            "panel_sha256": _digest(rollout_raw["panel_sha256"], "rollout panel sha256"),
            "panel_max_bytes": _integer(
                rollout_raw["panel_max_bytes"], "rollout panel max bytes", minimum=1
            ),
        },
        "scoring": {
            "budgets": [
                {"blocks": item.blocks, "calibrated_cost": item.calibrated_cost} for item in budgets
            ],
            "local_sample_count": local_sample_count,
            "tail_mean_weight": _unit(scoring_raw["tail_mean_weight"], "tail_mean_weight"),
            "stratum_weights": dict(strata),
            "screening_threshold": _unit(scoring_raw["screening_threshold"], "screening_threshold"),
            "rollout_threshold": _unit(scoring_raw["rollout_threshold"], "rollout_threshold"),
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
    rollout = spec_without_digest["rollout"]
    return TrackSpec(
        track_id=track_id,
        netuid=netuid,
        track_start_block=spec_without_digest["track_start_block"],
        submission_cooldown_blocks=spec_without_digest["submission_cooldown_blocks"],
        model=ModelSpec(
            filename="selector.safetensors",
            max_bytes=spec_without_digest["model"]["max_bytes"],
            selector=selector,
        ),
        base_model=BaseModelSpec(repo_id, revision.lower()),
        cache=CacheSpec(
            screen_sha256=cache["screen_sha256"],
            screen_max_bytes=cache["screen_max_bytes"],
            exact_sha256=cache["exact_sha256"],
            exact_max_bytes=cache["exact_max_bytes"],
        ),
        rollout=RolloutSpec(
            runner_sha256=rollout["runner_sha256"],
            panel_sha256=rollout["panel_sha256"],
            panel_max_bytes=rollout["panel_max_bytes"],
        ),
        scoring=ScoringSpec(
            budgets=budgets,
            local_sample_count=scoring["local_sample_count"],
            tail_mean_weight=scoring["tail_mean_weight"],
            stratum_weights=strata,
            screening_threshold=scoring["screening_threshold"],
            rollout_threshold=scoring["rollout_threshold"],
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
