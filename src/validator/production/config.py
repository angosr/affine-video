"""Strict local configuration for the single private producer service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    model_root: str
    adapter: str
    predictor: str
    sample_cache: str
    feature_model: str
    offload_blocks: int
    prefetch: int
    mlp_chunk_rows: int


@dataclass(frozen=True, slots=True)
class ProducerConfig:
    track_spec: Path
    validator_database: Path
    source_pool: Path
    artifact_manifest: Path
    artifact_root: Path
    work_dir: Path
    storage_bucket: str
    storage_prefix: str
    gpu_ids: tuple[int, ...]
    evaluation_gpu_ids: tuple[int, ...]
    poll_seconds: float
    runtime: RuntimeConfig

    def artifact_path(self, logical_name: str) -> Path:
        """Resolve a manifest-owned path without permitting root escape."""

        candidate = (self.artifact_root / logical_name).resolve()
        root = self.artifact_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("runtime artifact path escapes artifact_root")
        return candidate

    def verify_manifest(self, manifest: Any) -> None:
        manifest.verify_local(self.artifact_root)
        manifest.verify_runtime_coverage(
            self.artifact_root,
            files=(
                self.runtime.adapter,
                self.runtime.predictor,
                self.runtime.feature_model,
            ),
            directories=(self.runtime.model_root, self.runtime.sample_cache),
        )


def load_producer_config(path: str | Path) -> ProducerConfig:
    """Load the one producer config format and reject silent extra settings."""

    location = Path(path)
    try:
        value = json.loads(location.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except OSError as exc:
        raise ValueError("producer configuration cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("producer configuration is not valid JSON") from exc
    root = _object(
        value,
        {
            "artifact_manifest",
            "artifact_root",
            "format",
            "gpu_ids",
            "evaluation_gpu_ids",
            "poll_seconds",
            "runtime",
            "source_pool",
            "storage_bucket",
            "storage_prefix",
            "track_spec",
            "validator_database",
            "work_dir",
        },
        "producer configuration",
    )
    if root["format"] != "veda-producer-config-v1":
        raise ValueError("producer configuration format is unsupported")
    runtime = _object(
        root["runtime"],
        {
            "adapter",
            "mlp_chunk_rows",
            "model_root",
            "offload_blocks",
            "predictor",
            "prefetch",
            "sample_cache",
            "feature_model",
        },
        "producer runtime",
    )
    gpu_ids = root["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or any(type(value) is not int for value in gpu_ids)
        or tuple(gpu_ids) != (2, 3, 4, 5)
    ):
        raise ValueError("formal producer gpu_ids must be exactly [2,3,4,5]")
    evaluation_gpu_ids = root["evaluation_gpu_ids"]
    if (
        not isinstance(evaluation_gpu_ids, list)
        or any(type(value) is not int for value in evaluation_gpu_ids)
        or tuple(evaluation_gpu_ids) != (0, 1)
    ):
        raise ValueError("formal evaluator evaluation_gpu_ids must be exactly [0,1]")
    poll_seconds = root["poll_seconds"]
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not isfinite(float(poll_seconds))
        or float(poll_seconds) <= 0
    ):
        raise ValueError("producer poll_seconds must be finite and positive")
    config = ProducerConfig(
        track_spec=_path(root["track_spec"], "track_spec"),
        validator_database=_path(root["validator_database"], "validator_database"),
        source_pool=_path(root["source_pool"], "source_pool"),
        artifact_manifest=_path(root["artifact_manifest"], "artifact_manifest"),
        artifact_root=_path(root["artifact_root"], "artifact_root"),
        work_dir=_path(root["work_dir"], "work_dir"),
        storage_bucket=_string(root["storage_bucket"], "storage_bucket"),
        storage_prefix=_string(root["storage_prefix"], "storage_prefix").strip("/"),
        gpu_ids=tuple(gpu_ids),
        evaluation_gpu_ids=tuple(evaluation_gpu_ids),
        poll_seconds=float(poll_seconds),
        runtime=RuntimeConfig(
            model_root=_relative(runtime["model_root"], "model_root"),
            adapter=_relative(runtime["adapter"], "adapter"),
            predictor=_relative(runtime["predictor"], "predictor"),
            sample_cache=_relative(runtime["sample_cache"], "sample_cache"),
            feature_model=_relative(runtime["feature_model"], "feature_model"),
            offload_blocks=_integer(runtime["offload_blocks"], "offload_blocks", 0),
            prefetch=_integer(runtime["prefetch"], "prefetch", 1),
            mlp_chunk_rows=_integer(runtime["mlp_chunk_rows"], "mlp_chunk_rows", 1),
        ),
    )
    if not config.storage_prefix:
        raise ValueError("storage_prefix cannot be empty")
    for name in (
        config.runtime.model_root,
        config.runtime.adapter,
        config.runtime.predictor,
        config.runtime.sample_cache,
        config.runtime.feature_model,
    ):
        config.artifact_path(name)
    return config


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate producer configuration field: {key}")
        result[key] = value
    return result


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path(value: Any, name: str) -> Path:
    return Path(_string(value, name)).expanduser()


def _relative(value: Any, name: str) -> str:
    result = _string(value, name)
    path = Path(result)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"runtime {name} must be a normalized relative path")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"runtime {name} must be an integer >= {minimum}")
    return value
