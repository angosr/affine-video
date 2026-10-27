import hashlib
import json
from dataclasses import replace

import pytest

from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
from validator.production.config import load_producer_config


def document(tmp_path):
    return {
        "artifact_manifest": str(tmp_path / "artifacts.json"),
        "competition_manifest": str(tmp_path / "competition.json"),
        "artifact_root": str(tmp_path / "artifacts"),
        "evaluator_artifact_root": str(tmp_path / "evaluator-artifacts"),
        "format": "veda-producer-config-v1",
        "gpu_ids": [2, 3, 4, 5],
        "evaluation_gpu_ids": [0, 1],
        "poll_seconds": 12,
        "runtime": {
            "adapter": "weights/turbo.safetensors",
            "mlp_chunk_rows": 8192,
            "model_root": "weights/model",
            "offload_blocks": 0,
            "predictor": "weights/predictor.safetensors",
            "prefetch": 1,
            "sample_cache": "samples/production",
            "feature_model": "weights/r3d18.safetensors",
            "source_binding": "samples/source-binding.json",
            "source_bundle": "runtime/source.tar.zst",
            "dependency_lock": "runtime/dependencies.lock",
            "oci_image": "runtime/producer.oci.tar.zst",
        },
        "source_pool": str(tmp_path / "sources.json"),
        "storage_bucket": "private",
        "storage_prefix": "evaluation",
        "track_spec": str(tmp_path / "track.json"),
        "validator_database": str(tmp_path / "validator.sqlite"),
        "work_dir": str(tmp_path / "work"),
    }


def test_config_fixes_formal_producer_gpu_assignment(tmp_path) -> None:
    value = document(tmp_path)
    path = tmp_path / "producer.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    config = load_producer_config(path)

    assert config.gpu_ids == (2, 3, 4, 5)
    assert config.evaluation_gpu_ids == (0, 1)
    assert config.artifact_path(config.runtime.adapter).is_relative_to(
        config.artifact_root.resolve()
    )
    assert config.evaluator_artifact_path(config.runtime.adapter).is_relative_to(
        config.evaluator_artifact_root.resolve()
    )


def test_config_rejects_gpu_or_path_escape(tmp_path) -> None:
    value = document(tmp_path)
    value["gpu_ids"] = [0, 1, 2, 3]
    path = tmp_path / "producer.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_producer_config(path)

    value = document(tmp_path)
    value["runtime"]["adapter"] = "../escape"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized relative"):
        load_producer_config(path)

    value = document(tmp_path)
    value["evaluator_artifact_root"] = value["artifact_root"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="must be isolated"):
        load_producer_config(path)


def test_config_resolves_only_formal_physical_evaluation_gpus(tmp_path, monkeypatch) -> None:
    path = tmp_path / "producer.json"
    path.write_text(json.dumps(document(tmp_path)), encoding="utf-8")
    config = load_producer_config(path)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert config.validate_evaluation_device("cuda:0") == "cuda:0"
    assert config.validate_evaluation_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError, match="physical GPU 0 or 1"):
        config.validate_evaluation_device("cuda:6")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,0,1")
    with pytest.raises(ValueError, match="physical GPU 0 or 1"):
        config.validate_evaluation_device("cuda:0")
    assert config.validate_evaluation_device("cuda:1") == "cuda:1"
    assert config.validate_evaluation_device("cuda:2") == "cuda:2"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-opaque")
    with pytest.raises(ValueError, match="verifiable physical"):
        config.validate_evaluation_device("cuda:0")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="verifiable physical"):
        config.validate_evaluation_device("cuda:0")


def test_evaluator_artifact_selection_excludes_hidden_source_inputs(tmp_path) -> None:
    path = tmp_path / "producer.json"
    path.write_text(json.dumps(document(tmp_path)), encoding="utf-8")
    config = load_producer_config(path)
    names = (
        "runtime/dependencies.lock",
        "runtime/producer.oci.tar.zst",
        "runtime/source.tar.zst",
        "samples/production/index.json",
        "samples/production/text.safetensors",
        "samples/source-binding.json",
        "weights/model/shard.safetensors",
        "weights/predictor.safetensors",
        "weights/r3d18.safetensors",
        "weights/turbo.safetensors",
    )
    manifest = ArtifactManifest(
        tuple(
            Artifact(name, f"{index + 1:064x}", 1, f"artifacts/{index}")
            for index, name in enumerate(names)
        )
    )

    selected = config.evaluator_artifact_names(manifest)

    assert "weights/model/shard.safetensors" in selected
    assert "samples/source-binding.json" not in selected
    assert not any(name.startswith("samples/production/") for name in selected)

    substituted = ArtifactManifest(
        tuple(
            replace(artifact, object_key="source-pool/substituted")
            if artifact.logical_name == "weights/predictor.safetensors"
            else artifact
            for artifact in manifest.artifacts
        )
    )
    with pytest.raises(ValueError, match="outside its IAM namespace"):
        config.evaluator_artifact_names(substituted)


def test_runtime_verification_uses_a_bound_image_descriptor_not_an_archive(tmp_path, monkeypatch):
    from validator.production import commitment, environment

    raw = document(tmp_path)
    raw["runtime"]["oci_image"] = "runtime/image.json"
    path = tmp_path / "config.json"
    path.write_bytes(canonical_json(raw))
    config = load_producer_config(path)
    image = {"format": "veda-runtime-image-v1", "image_id": "sha256:" + "11" * 32,
             "source_sha256": "22" * 32, "environment_sha256": "33" * 32}
    monkeypatch.setattr(environment, "evaluation_source_digest", lambda: "22" * 32)
    monkeypatch.setattr(environment, "evaluation_environment_digest", lambda: "33" * 32)
    checked = []
    monkeypatch.setattr(environment, "verify_runtime_environment",
                        lambda *paths: checked.append(paths))
    files = {
        config.runtime.adapter: b"adapter", config.runtime.predictor: b"predictor",
        config.runtime.feature_model: b"feature", config.runtime.source_binding: b"binding",
        config.runtime.source_bundle: b"source", config.runtime.dependency_lock: b"lock",
        config.runtime.oci_image: canonical_json(image),
        config.runtime.model_root + "/weights.bin": b"teacher",
        config.runtime.sample_cache + "/index.json": b"sample",
    }
    artifacts = []
    for name, payload in sorted(files.items()):
        for root in (config.artifact_root, config.evaluator_artifact_root):
            item = root / name
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        artifacts.append(Artifact(name, digest, len(payload), f"artifacts/{digest}"))
    manifest = ArtifactManifest(tuple(artifacts))
    original_sha = commitment._file_sha256
    hashed = []

    def count_hash(path):
        hashed.append(path)
        return original_sha(path)

    monkeypatch.setattr(commitment, "_file_sha256", count_hash)
    assert config.verify_manifest(manifest) == image
    assert config.verify_evaluator_manifest(manifest) == image
    assert config.verify_manifest(manifest) == image
    assert len(checked) == 3  # Actual source/lock validation is never skipped.
    assert all(hashed.count(config.artifact_root / name) == 2 for name in files)
    selected = config.evaluator_artifact_names(manifest)
    assert all(hashed.count(config.evaluator_artifact_root / name) == 1 for name in selected)
    assert len(hashed) == 2 * len(files) + len(selected)
    assert not (config.work_dir / "artifact-verification").exists()
    assert not list(tmp_path.rglob("*.tar"))
    image["source_sha256"] = "00" * 32
    (config.artifact_root / config.runtime.oci_image).write_bytes(canonical_json(image))
    with pytest.raises(ValueError, match="OCI descriptor"):
        config.verify_image_descriptor(config.artifact_root)
