"""Strict local configuration for the single private producer service."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from validator.production.storage import ARTIFACTS_NAMESPACE


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    model_root: str
    adapter: str
    predictor: str
    sample_cache: str
    feature_model: str
    source_binding: str
    source_bundle: str
    dependency_lock: str
    oci_image: str
    offload_blocks: int
    prefetch: int
    mlp_chunk_rows: int


@dataclass(frozen=True, slots=True)
class ProducerConfig:
    track_spec: Path
    validator_database: Path
    source_pool: Path
    artifact_manifest: Path
    competition_manifest: Path
    artifact_root: Path
    evaluator_artifact_root: Path
    work_dir: Path
    storage_bucket: str
    storage_prefix: str
    gpu_ids: tuple[int, ...]
    evaluation_gpu_ids: tuple[int, ...]
    poll_seconds: float
    runtime: RuntimeConfig
    data_track_spec: Path | None = None
    data_competition_manifest: Path | None = None

    def data_binding(self, track: Any, runtime_manifest: bytes) -> tuple[Any, bytes]:
        """Verify the one immutable data origin and its unchanged asset contract."""
        from validator.release import verify_data_binding

        separated = track.data_spec_sha256 is not None
        if separated != (self.data_track_spec is not None
                         and self.data_competition_manifest is not None):
            raise ValueError("separate data identity requires its original spec and manifest")
        if not separated and (self.data_track_spec or self.data_competition_manifest):
            raise ValueError("uncommitted data origin paths are forbidden")
        payload = (self.data_competition_manifest if separated
                   else self.competition_manifest).read_bytes()
        data = verify_data_binding(track, runtime_manifest,
                                   self.data_track_spec.read_bytes() if separated else None,
                                   payload)
        return data, payload

    def artifact_path(self, logical_name: str) -> Path:
        """Resolve a manifest-owned path without permitting root escape."""

        candidate = (self.artifact_root / logical_name).resolve()
        root = self.artifact_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("runtime artifact path escapes artifact_root")
        return candidate

    def evaluator_artifact_path(self, logical_name: str) -> Path:
        """Resolve an evaluator-only runtime path outside the producer's hidden tree."""

        candidate = (self.evaluator_artifact_root / logical_name).resolve()
        root = self.evaluator_artifact_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("runtime artifact path escapes evaluator_artifact_root")
        return candidate

    def verify_manifest(self, manifest: Any) -> dict[str, str]:
        files = (
            self.runtime.adapter, self.runtime.predictor, self.runtime.feature_model,
            self.runtime.source_binding, self.runtime.source_bundle,
            self.runtime.dependency_lock, self.runtime.oci_image,
        )
        manifest.verify_runtime_coverage(
            self.artifact_root,
            files=files,
            directories=(self.runtime.model_root, self.runtime.sample_cache),
        )
        prefixes = (self.runtime.model_root.rstrip("/") + "/",
                    self.runtime.sample_cache.rstrip("/") + "/")
        names = tuple(item.logical_name for item in manifest.artifacts
                      if item.logical_name in files or item.logical_name.startswith(prefixes))
        manifest.verify_local(
            self.artifact_root, logical_names=names,
        )
        from validator.production.environment import verify_runtime_environment

        verify_runtime_environment(
            self.artifact_path(self.runtime.source_bundle),
            self.artifact_path(self.runtime.dependency_lock),
        )
        return self.verify_image_descriptor(self.artifact_root)

    def evaluator_artifact_names(self, manifest: Any) -> tuple[str, ...]:
        """Select runtime bytes needed for scoring, excluding every hidden source input."""

        files = {
            self.runtime.adapter,
            self.runtime.predictor,
            self.runtime.feature_model,
            self.runtime.source_bundle,
            self.runtime.dependency_lock,
            self.runtime.oci_image,
        }
        model_prefix = self.runtime.model_root.rstrip("/") + "/"
        files.update(
            artifact.logical_name
            for artifact in manifest.artifacts
            if artifact.logical_name.startswith(model_prefix)
        )
        known = {artifact.logical_name for artifact in manifest.artifacts}
        if not files <= known or not any(name.startswith(model_prefix) for name in files):
            raise ValueError("artifact manifest omits evaluator runtime bytes")
        forbidden = {
            self.runtime.source_binding,
            *(
                artifact.logical_name
                for artifact in manifest.artifacts
                if artifact.logical_name.startswith(self.runtime.sample_cache.rstrip("/") + "/")
            ),
        }
        if files & forbidden:
            raise ValueError("evaluator runtime selection includes hidden source inputs")
        by_name = {artifact.logical_name: artifact for artifact in manifest.artifacts}
        if any(
            not by_name[name].object_key.startswith(f"{ARTIFACTS_NAMESPACE}/")
            for name in files
        ):
            raise ValueError("evaluator runtime artifact is outside its IAM namespace")
        return tuple(sorted(files))

    def verify_evaluator_manifest(self, manifest: Any) -> dict[str, str]:
        """Verify the committed runtime without reading the hidden pool or full SampleCache."""

        names = self.evaluator_artifact_names(manifest)
        manifest.verify_local(
            self.evaluator_artifact_root, logical_names=names,
        )
        manifest.verify_runtime_coverage(
            self.evaluator_artifact_root,
            files=(
                self.runtime.adapter,
                self.runtime.predictor,
                self.runtime.feature_model,
                self.runtime.source_bundle,
                self.runtime.dependency_lock,
                self.runtime.oci_image,
            ),
            directories=(self.runtime.model_root,),
        )
        from validator.production.environment import verify_runtime_environment

        verify_runtime_environment(
            self.evaluator_artifact_path(self.runtime.source_bundle),
            self.evaluator_artifact_path(self.runtime.dependency_lock),
        )
        return self.verify_image_descriptor(self.evaluator_artifact_root)

    def competition_payload(self, track: Any) -> bytes:
        payload = self.competition_manifest.read_bytes()
        if hashlib.sha256(payload).hexdigest() != track.artifact_manifest_sha256:
            raise ValueError("competition manifest does not match the track commitment")
        return payload

    def verify_image_descriptor(self, root: Path) -> dict[str, str]:
        """Bind OCI identity without requiring a docker-save archive or socket.

        The existing owner must compare this image ID with Engine's actual
        container Image before start; the process verifies source/lock itself.
        """
        from validator.production.commitment import decode_canonical_json
        from validator.production.environment import (
            evaluation_environment_digest,
            evaluation_source_digest,
        )
        from validator.release import _image_id

        payload = decode_canonical_json((root / self.runtime.oci_image).read_bytes())
        if (not isinstance(payload, dict) or set(payload) != {
            "format", "image_id", "source_sha256", "environment_sha256",
        } or payload["format"] != "veda-runtime-image-v1"
                or not _image_id(payload["image_id"])
                or payload["source_sha256"] != evaluation_source_digest()
                or payload["environment_sha256"] != evaluation_environment_digest()):
            raise ValueError("OCI descriptor differs from the installed runtime")
        return payload

    def validate_evaluation_device(self, device: str | None) -> str:
        """Resolve a CUDA ordinal to an allowed physical evaluation GPU."""

        if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
            raise ValueError("formal scoring_device must be an explicit CUDA ordinal")
        ordinal = int(device.removeprefix("cuda:"))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            physical = ordinal
        else:
            entries = [item.strip() for item in visible.split(",")]
            if (
                not entries
                or any(not item.isdecimal() for item in entries)
                or ordinal >= len(entries)
            ):
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES does not expose a verifiable physical scoring GPU"
                )
            physical = int(entries[ordinal])
        if physical not in self.evaluation_gpu_ids:
            raise ValueError("formal scoring_device must resolve to physical GPU 0 or 1")
        return device


def load_producer_config(path: str | Path) -> ProducerConfig:
    """Load the one producer config format and reject silent extra settings."""

    location = Path(path)
    try:
        value = json.loads(location.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except OSError as exc:
        raise ValueError("producer configuration cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("producer configuration is not valid JSON") from exc
    origin = {}
    if isinstance(value, dict):
        origin = {key: value[key] for key in ("data_track_spec", "data_competition_manifest")
                  if key in value}
        if origin and len(origin) != 2:
            raise ValueError("both original data paths are required")
        value = {key: item for key, item in value.items() if key not in origin}
    root = _object(
        value,
        {
            "artifact_manifest",
            "competition_manifest",
            "artifact_root",
            "evaluator_artifact_root",
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
            "source_binding",
            "source_bundle",
            "dependency_lock",
            "oci_image",
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
        data_track_spec=None if not origin else _path(origin["data_track_spec"], "data_track_spec"),
        data_competition_manifest=(None if not origin else _path(
            origin["data_competition_manifest"], "data_competition_manifest",
        )),
        track_spec=_path(root["track_spec"], "track_spec"),
        validator_database=_path(root["validator_database"], "validator_database"),
        source_pool=_path(root["source_pool"], "source_pool"),
        artifact_manifest=_path(root["artifact_manifest"], "artifact_manifest"),
        competition_manifest=_path(root["competition_manifest"], "competition_manifest"),
        artifact_root=_path(root["artifact_root"], "artifact_root"),
        evaluator_artifact_root=_path(
            root["evaluator_artifact_root"], "evaluator_artifact_root"
        ),
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
            source_binding=_relative(runtime["source_binding"], "source_binding"),
            source_bundle=_relative(runtime["source_bundle"], "source_bundle"),
            dependency_lock=_relative(runtime["dependency_lock"], "dependency_lock"),
            oci_image=_relative(runtime["oci_image"], "oci_image"),
            offload_blocks=_integer(runtime["offload_blocks"], "offload_blocks", 0),
            prefetch=_integer(runtime["prefetch"], "prefetch", 1),
            mlp_chunk_rows=_integer(runtime["mlp_chunk_rows"], "mlp_chunk_rows", 1),
        ),
    )
    if not config.storage_prefix:
        raise ValueError("storage_prefix cannot be empty")
    producer_root = config.artifact_root.resolve()
    evaluator_root = config.evaluator_artifact_root.resolve()
    if (
        evaluator_root == producer_root
        or evaluator_root in producer_root.parents
        or producer_root in evaluator_root.parents
    ):
        raise ValueError("evaluator_artifact_root must be isolated from artifact_root")
    for name in (
        config.runtime.model_root,
        config.runtime.adapter,
        config.runtime.predictor,
        config.runtime.sample_cache,
        config.runtime.feature_model,
        config.runtime.source_binding,
        config.runtime.source_bundle,
        config.runtime.dependency_lock,
        config.runtime.oci_image,
    ):
        config.artifact_path(name)
        config.evaluator_artifact_path(name)
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
