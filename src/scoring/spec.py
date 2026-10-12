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
class ScoringSpec:
    budgets: tuple[BudgetSpec, ...]
    screen_sample_count: int
    local_sample_count: int
    tail_mean_weight: float
    stratum_weights: tuple[tuple[str, float], ...]
    screening_threshold: float
    rollout_threshold: float
    q_floor: float
    gamma: float
    bootstrap_confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int

    @property
    def stratum_weight_map(self) -> dict[str, float]:
        return dict(self.stratum_weights)


@dataclass(frozen=True, slots=True)
class ChainWeightSpec:
    freshness_half_life_blocks: int
    max_age_blocks: int
    provisional_max_weight: float


@dataclass(frozen=True, slots=True)
class TrackSpec:
    schema_version: int
    track_id: str
    netuid: int
    track_start_block: int
    submission_cooldown_blocks: int
    seed_delay_blocks: int
    model: ModelSpec
    base_model: BaseModelSpec
    cache: CacheSpec
    scoring: ScoringSpec
    weights: ChainWeightSpec
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
            "schema_version",
            "track_id",
            "netuid",
            "track_start_block",
            "submission_cooldown_blocks",
            "seed_delay_blocks",
            "model",
            "base_model",
            "cache",
            "scoring",
            "weights",
        },
        "track specification",
    )
    schema_version = _integer(root["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        raise TrackSpecError("unsupported track specification schema version")
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
            "descriptor_dim",
            "hidden_dim",
            "projection_dim",
            "storage_dtype",
            "activation",
        },
        "selector",
    )
    routes_raw = selector_raw["routes"]
    if not isinstance(routes_raw, list) or not routes_raw:
        raise TrackSpecError("selector routes must be a non-empty list")
    routes = tuple(_nonempty_string(route, "selector route") for route in routes_raw)
    try:
        selector = SelectorArchitecture(
            routes=routes,
            descriptor_dim=_integer(
                selector_raw["descriptor_dim"], "selector descriptor_dim", minimum=1
            ),
            hidden_dim=_integer(selector_raw["hidden_dim"], "selector hidden_dim", minimum=1),
            projection_dim=_integer(
                selector_raw["projection_dim"], "selector projection_dim", minimum=1
            ),
            storage_dtype=_nonempty_string(selector_raw["storage_dtype"], "selector storage_dtype"),
            activation=_nonempty_string(selector_raw["activation"], "selector activation"),
        )
    except ValueError as exc:
        raise TrackSpecError(str(exc)) from exc
    max_model_bytes = _integer(model_raw["max_bytes"], "model max_bytes", minimum=1)
    item_size = 2 if selector.storage_dtype == "float16" else 4
    tensor_bytes = sum(prod(tensor.shape) * item_size for tensor in selector.tensor_schema.values())
    if tensor_bytes >= max_model_bytes:
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

    scoring_raw = _object(
        root["scoring"],
        {
            "budgets",
            "screen_sample_count",
            "local_sample_count",
            "tail_mean_weight",
            "stratum_weights",
            "screening_threshold",
            "rollout_threshold",
            "q_floor",
            "gamma",
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
    screen_sample_count = _integer(
        scoring_raw["screen_sample_count"], "screen_sample_count", minimum=1
    )
    local_sample_count = _integer(
        scoring_raw["local_sample_count"], "local_sample_count", minimum=1
    )
    if min(screen_sample_count, local_sample_count) < len(strata):
        raise TrackSpecError("sample counts must cover every scoring stratum")

    weights_raw = _object(
        root["weights"],
        {
            "freshness_half_life_blocks",
            "max_age_blocks",
            "provisional_max_weight",
        },
        "weights",
    )
    spec_without_digest = {
        "schema_version": schema_version,
        "track_id": track_id,
        "netuid": netuid,
        "track_start_block": _integer(root["track_start_block"], "track_start_block", minimum=0),
        "submission_cooldown_blocks": _integer(
            root["submission_cooldown_blocks"],
            "submission_cooldown_blocks",
            minimum=0,
        ),
        "seed_delay_blocks": _integer(root["seed_delay_blocks"], "seed_delay_blocks", minimum=0),
        "model": {
            "filename": "selector.safetensors",
            "max_bytes": max_model_bytes,
            "selector": {
                "routes": list(selector.routes),
                "descriptor_dim": selector.descriptor_dim,
                "hidden_dim": selector.hidden_dim,
                "projection_dim": selector.projection_dim,
                "storage_dtype": selector.storage_dtype,
                "activation": selector.activation,
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
        "scoring": {
            "budgets": [
                {"blocks": item.blocks, "calibrated_cost": item.calibrated_cost} for item in budgets
            ],
            "screen_sample_count": screen_sample_count,
            "local_sample_count": local_sample_count,
            "tail_mean_weight": _unit(scoring_raw["tail_mean_weight"], "tail_mean_weight"),
            "stratum_weights": dict(strata),
            "screening_threshold": _unit(scoring_raw["screening_threshold"], "screening_threshold"),
            "rollout_threshold": _unit(scoring_raw["rollout_threshold"], "rollout_threshold"),
            "q_floor": _unit(scoring_raw["q_floor"], "q_floor", include_one=False),
            "gamma": _number(scoring_raw["gamma"], "gamma", minimum_exclusive=0),
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
        "weights": {
            "freshness_half_life_blocks": _integer(
                weights_raw["freshness_half_life_blocks"],
                "freshness_half_life_blocks",
                minimum=1,
            ),
            "max_age_blocks": _integer(weights_raw["max_age_blocks"], "max_age_blocks", minimum=1),
            "provisional_max_weight": _unit(
                weights_raw["provisional_max_weight"],
                "provisional_max_weight",
                include_zero=False,
            ),
        },
    }
    digest = hashlib.sha256(_canonical_json(spec_without_digest)).hexdigest()
    scoring = spec_without_digest["scoring"]
    cache = spec_without_digest["cache"]
    chain_weights = spec_without_digest["weights"]
    return TrackSpec(
        schema_version=schema_version,
        track_id=track_id,
        netuid=netuid,
        track_start_block=spec_without_digest["track_start_block"],
        submission_cooldown_blocks=spec_without_digest["submission_cooldown_blocks"],
        seed_delay_blocks=spec_without_digest["seed_delay_blocks"],
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
        scoring=ScoringSpec(
            budgets=budgets,
            screen_sample_count=scoring["screen_sample_count"],
            local_sample_count=scoring["local_sample_count"],
            tail_mean_weight=scoring["tail_mean_weight"],
            stratum_weights=strata,
            screening_threshold=scoring["screening_threshold"],
            rollout_threshold=scoring["rollout_threshold"],
            q_floor=scoring["q_floor"],
            gamma=scoring["gamma"],
            bootstrap_confidence=scoring["bootstrap_confidence"],
            bootstrap_resamples=scoring["bootstrap_resamples"],
            bootstrap_seed=scoring["bootstrap_seed"],
        ),
        weights=ChainWeightSpec(
            freshness_half_life_blocks=chain_weights["freshness_half_life_blocks"],
            max_age_blocks=chain_weights["max_age_blocks"],
            provisional_max_weight=chain_weights["provisional_max_weight"],
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
