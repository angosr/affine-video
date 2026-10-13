import json

import pytest

from scoring.spec import TrackSpecError, load_track_spec, parse_track_spec


def valid_spec() -> dict:
    return {
        "track_id": "veda-test-v1",
        "netuid": 7,
        "track_start_block": 100,
        "publication_grace_blocks": 20,
        "submission_cooldown_blocks": 50,
        "model": {
            "filename": "selector.safetensors",
            "max_bytes": 1024,
            "selector": {
                "routes": ["layer_0_head_0"],
                "num_layers": 1,
                "num_heads": 1,
                "head_dim": 2,
                "storage_dtype": "float32",
            },
        },
        "base_model": {
            "repo_id": "owner/base-model",
            "revision": "12" * 20,
        },
        "cache": {
            "screen_sha256": "34" * 32,
            "screen_max_bytes": 100_000,
            "exact_sha256": "56" * 32,
            "exact_max_bytes": 1_000_000,
        },
        "rollout": {
            "runner_sha256": "78" * 32,
            "panel_sha256": "9a" * 32,
            "panel_max_bytes": 2_000_000,
        },
        "scoring": {
            "budgets": [
                {"blocks": 2, "calibrated_cost": 1.0},
                {"blocks": 4, "calibrated_cost": 2.0},
                {"blocks": 8, "calibrated_cost": 4.0},
            ],
            "local_sample_count": 20,
            "tail_mean_weight": 0.8,
            "stratum_weights": {"layer-0": 0.5, "layer-1": 0.5},
            "screening_threshold": 0.5,
            "rollout_threshold": 0.7,
            "minimum_frontier_gain": 0.001,
            "bootstrap_confidence": 0.95,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 7,
        },
    }


def test_parse_track_spec_produces_typed_values_and_stable_digest() -> None:
    raw = valid_spec()
    first = parse_track_spec(raw)
    reordered = {key: raw[key] for key in reversed(raw)}
    second = parse_track_spec(reordered)

    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.model.tensor_schema["layers.0.proj_q"].shape == (1, 6, 2)
    assert first.scoring.stratum_weight_map == {"layer-0": 0.5, "layer-1": 0.5}
    assert first.base_model.revision == "12" * 20
    assert first.publication_grace_blocks == 20
    assert first.rollout.panel_sha256 == "9a" * 32
    assert first.scoring.minimum_frontier_gain == 0.001


def test_score_affecting_change_changes_digest() -> None:
    first = valid_spec()
    second = valid_spec()
    second["scoring"]["rollout_threshold"] = 0.8

    assert parse_track_spec(first).digest != parse_track_spec(second).digest


def test_publication_grace_is_positive_and_changes_digest() -> None:
    invalid = valid_spec()
    invalid["publication_grace_blocks"] = 0
    with pytest.raises(TrackSpecError, match="greater than or equal to 1"):
        parse_track_spec(invalid)

    changed = valid_spec()
    changed["publication_grace_blocks"] += 1
    assert parse_track_spec(changed).digest != parse_track_spec(valid_spec()).digest


def test_minimum_frontier_gain_must_be_positive_and_changes_digest() -> None:
    invalid = valid_spec()
    invalid["scoring"]["minimum_frontier_gain"] = 0
    with pytest.raises(TrackSpecError, match="unit interval"):
        parse_track_spec(invalid)

    changed = valid_spec()
    changed["scoring"]["minimum_frontier_gain"] = 0.002
    assert parse_track_spec(changed).digest != parse_track_spec(valid_spec()).digest


def test_selector_semantics_are_explicit_and_frozen() -> None:
    missing = valid_spec()
    del missing["model"]["selector"]["head_dim"]
    with pytest.raises(TrackSpecError, match="exactly"):
        parse_track_spec(missing)

    unsupported = valid_spec()
    unsupported["model"]["selector"]["routes"] = ["layer_1_head_0"]
    with pytest.raises(TrackSpecError, match="exceeds"):
        parse_track_spec(unsupported)


def test_unknown_fields_and_invalid_budget_order_are_rejected() -> None:
    unknown = valid_spec()
    unknown["ignored"] = True
    with pytest.raises(TrackSpecError, match="exactly"):
        parse_track_spec(unknown)

    unordered = valid_spec()
    unordered["scoring"]["budgets"][1]["blocks"] = 1
    with pytest.raises(TrackSpecError, match="strictly increasing"):
        parse_track_spec(unordered)

    duplicate_route = valid_spec()
    duplicate_route["model"]["selector"]["routes"] *= 2
    with pytest.raises(TrackSpecError, match="unique"):
        parse_track_spec(duplicate_route)

    impossible_size = valid_spec()
    impossible_size["model"]["max_bytes"] = 100
    with pytest.raises(TrackSpecError, match="cannot hold"):
        parse_track_spec(impossible_size)

    header_too_large = valid_spec()
    header_too_large["model"]["max_bytes"] = 255
    with pytest.raises(TrackSpecError, match="cannot hold"):
        parse_track_spec(header_too_large)

    exact_minimum_size = valid_spec()
    exact_minimum_size["model"]["max_bytes"] = 256
    assert parse_track_spec(exact_minimum_size).model.max_bytes == 256

    undersampled = valid_spec()
    undersampled["scoring"]["stratum_weights"] = {f"layer-{index}": 0.2 for index in range(5)}
    undersampled["scoring"]["local_sample_count"] = 3
    with pytest.raises(TrackSpecError, match="cover every"):
        parse_track_spec(undersampled)

    uncovered_route = valid_spec()
    uncovered_route["model"]["selector"]["num_heads"] = 2
    uncovered_route["model"]["selector"]["routes"] = ["layer_0_head_0", "layer_0_head_1"]
    uncovered_route["scoring"]["local_sample_count"] = 3
    with pytest.raises(TrackSpecError, match="every route"):
        parse_track_spec(uncovered_route)


def test_loader_rejects_duplicate_json_fields(tmp_path) -> None:
    path = tmp_path / "track_spec.json"
    path.write_text('{"track_id":"a","track_id":"b"}', encoding="utf-8")

    with pytest.raises(TrackSpecError, match="duplicate"):
        load_track_spec(path)


def test_loader_accepts_canonical_json_file(tmp_path) -> None:
    path = tmp_path / "track_spec.json"
    path.write_text(json.dumps(valid_spec()), encoding="utf-8")

    assert load_track_spec(path).track_id == "veda-test-v1"
