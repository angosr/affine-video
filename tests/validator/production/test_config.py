import json

import pytest

from validator.production.config import load_producer_config


def document(tmp_path):
    return {
        "artifact_manifest": str(tmp_path / "artifacts.json"),
        "artifact_root": str(tmp_path / "artifacts"),
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
