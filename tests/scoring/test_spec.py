import json

import pytest

from scoring.spec import (
    VEDA_EVALUATION_FILE_SHA256,
    VEDA_FILE_SHA256,
    VEDA_FILENAME,
    VEDA_MODEL_DIGEST,
    VEDA_REPO_ID,
    VEDA_REVISION,
    TrackSpecError,
    load_track_spec,
    parse_track_spec,
    track_spec_dict,
)


def valid_spec() -> dict:
    return {
        "track_id": "veda-test-v1",
        "netuid": 7,
        "mechid": 0,
        "track_start_block": 100,
        "publication_grace_blocks": 2,
        "maximum_generation_batches": 10,
        "maximum_effective_challenges": 2,
        "track_close_grace_blocks": 50,
        "source_pool_merkle_root": "ab" * 32,
        "artifact_manifest_sha256": "cd" * 32,
        "initial_champion": {
            "uid": 0,
            "hotkey": "genesis-hotkey",
            "repo_id": VEDA_REPO_ID,
            "revision": VEDA_REVISION,
            "filename": VEDA_FILENAME,
            "file_sha256": VEDA_FILE_SHA256,
            "evaluation_file_sha256": VEDA_EVALUATION_FILE_SHA256,
            "model_digest": VEDA_MODEL_DIGEST,
            "score": 0.7,
        },
        "model": {
            "filename": "selector.safetensors",
            "max_bytes": 1024,
            "selector": {
                "num_layers": 1,
                "num_heads": 2,
                "head_dim": 2,
                "storage_dtype": "float32",
            },
        },
        "base_model": {
            "repo_id": "owner/base-model",
            "revision": "12" * 20,
        },
        "cache": {
            "exact_max_bytes": 1_000_000,
            "minimum_generation_delay_blocks": 4,
            "generation_interval_blocks": 100,
            "generator": "veda-private-rollout-v1",
            "format": "veda-exact-cache-v2",
            "storage_dtype": "float16",
            "compute_dtype": "float32",
            "accumulation_dtype": "float32",
        },
        "scoring": {
            "budgets": [
                {"target_blocks": 2, "reference_blocks": 0, "calibrated_cost": 1.0},
                {"target_blocks": 4, "reference_blocks": 0, "calibrated_cost": 2.0},
                {"target_blocks": 8, "reference_blocks": 0, "calibrated_cost": 4.0},
            ],
            "sampled_route_count": 2,
            "rollouts_per_stratum": 4,
            "sampled_step_count": 3,
            "query_tiles_per_case": 2,
            "step_selection_algorithm": "sha256-window-without-replacement-v1",
            "query_tile_selection_algorithm": "sha256-segment-quota-v1",
            "trajectory_schedule": {
                "name": "test-3-step",
                "sigmas": [1.0, 0.8, 0.4, 0.0],
                "step_windows": [
                    {"name": "early", "indices": [0], "weight": 0.3},
                    {"name": "middle", "indices": [1], "weight": 0.4},
                    {"name": "late", "indices": [2], "weight": 0.3},
                ],
            },
            "tail_mean_weight": 0.8,
            "stratum_weights": {"all": 1.0},
            "minimum_frontier_gain": 0.001,
            "season_false_promotion_alpha": 0.1,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 7,
        },
        "full_video": {
            "minimum_gain": 0.001,
            "metric_version": "veda-video-fidelity-v1",
            "feature_model_sha256": "ef" * 32,
            "latent_weight": 0.4,
            "frame_weight": 0.3,
            "temporal_weight": 0.3,
            "black_frame_threshold": 0.01,
            "flicker_threshold": 2.0,
        },
    }


def test_parse_track_spec_produces_typed_values_and_stable_digest() -> None:
    raw = valid_spec()
    first = parse_track_spec(raw)
    reordered = {key: raw[key] for key in reversed(raw)}
    second = parse_track_spec(reordered)

    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.model.tensor_schema["layers.0.proj_q"].shape == (2, 6, 2)
    assert first.scoring.stratum_weight_map == {"all": 1.0}
    assert first.base_model.revision == "12" * 20
    assert first.mechid == 0
    assert first.publication_grace_blocks == 2
    assert first.initial_champion.uid == 0
    assert first.initial_champion.repo_id == VEDA_REPO_ID
    assert first.cache.minimum_generation_delay_blocks == 4
    assert first.last_generation_block == 1100
    assert first.last_submission_block == 996
    assert first.track_close_block == 1150
    assert first.scoring.minimum_frontier_gain == 0.001
    assert first.scoring.case_count == 24
    assert first.match_confidence == 0.95
    assert parse_track_spec(track_spec_dict(first)) == first


def test_hidden_batch_seed_is_not_known_during_publication_grace() -> None:
    raw = valid_spec()
    raw["cache"]["minimum_generation_delay_blocks"] = raw["publication_grace_blocks"]

    with pytest.raises(TrackSpecError, match="must exceed"):
        parse_track_spec(raw)


def test_confirmation_gate_cannot_be_weaker_than_exact_gate() -> None:
    raw = valid_spec()
    raw["full_video"]["minimum_gain"] = 0.0005

    with pytest.raises(TrackSpecError, match="cannot be lower"):
        parse_track_spec(raw)


def test_full_video_metric_version_is_frozen() -> None:
    version_two = valid_spec()
    version_two["full_video"]["metric_version"] = "veda-video-fidelity-v2"
    assert parse_track_spec(version_two).full_video.metric_version.endswith("-v2")

    unknown = valid_spec()
    unknown["full_video"]["metric_version"] = "operator-choice"
    with pytest.raises(TrackSpecError, match="metric version"):
        parse_track_spec(unknown)


def test_score_affecting_change_changes_digest() -> None:
    first = valid_spec()
    second = valid_spec()
    second["scoring"]["budgets"][1]["calibrated_cost"] = 2.5

    assert parse_track_spec(first).digest != parse_track_spec(second).digest


def test_production_shape_uses_bounded_hidden_route_sample() -> None:
    raw = valid_spec()
    raw["model"]["selector"] = {
        "num_layers": 50,
        "num_heads": 56,
        "head_dim": 128,
        "storage_dtype": "float16",
    }
    raw["model"]["max_bytes"] = 600_000_000
    raw["scoring"]["sampled_route_count"] = 128

    track = parse_track_spec(raw)

    assert len(track.model.selector.routes) == 2_800
    assert track.scoring.sampled_route_count == 128
    assert track.scoring.case_count == 1536


def test_publication_grace_is_positive_and_changes_digest() -> None:
    invalid = valid_spec()
    invalid["publication_grace_blocks"] = 0
    with pytest.raises(TrackSpecError, match="greater than or equal to 1"):
        parse_track_spec(invalid)

    changed = valid_spec()
    changed["publication_grace_blocks"] += 1
    assert parse_track_spec(changed).digest != parse_track_spec(valid_spec()).digest


def test_initial_champion_is_uid_zero_and_exactly_the_official_veda_model() -> None:
    wrong_uid = valid_spec()
    wrong_uid["initial_champion"]["uid"] = 1
    with pytest.raises(TrackSpecError, match="uid must be 0"):
        parse_track_spec(wrong_uid)

    wrong_model = valid_spec()
    wrong_model["initial_champion"]["file_sha256"] = "00" * 32
    with pytest.raises(TrackSpecError, match="official VEDA"):
        parse_track_spec(wrong_model)


def test_minimum_frontier_gain_must_be_positive_and_changes_digest() -> None:
    invalid = valid_spec()
    invalid["scoring"]["minimum_frontier_gain"] = 0
    with pytest.raises(TrackSpecError, match="unit interval"):
        parse_track_spec(invalid)

    changed = valid_spec()
    changed["scoring"]["minimum_frontier_gain"] = 0.002
    changed["full_video"]["minimum_gain"] = 0.002
    assert parse_track_spec(changed).digest != parse_track_spec(valid_spec()).digest


def test_selector_semantics_are_explicit_and_frozen() -> None:
    missing = valid_spec()
    del missing["model"]["selector"]["head_dim"]
    with pytest.raises(TrackSpecError, match="exactly"):
        parse_track_spec(missing)


def test_unknown_fields_and_invalid_budget_order_are_rejected() -> None:
    unknown = valid_spec()
    unknown["ignored"] = True
    with pytest.raises(TrackSpecError, match="exactly"):
        parse_track_spec(unknown)

    unordered = valid_spec()
    unordered["scoring"]["budgets"][1]["target_blocks"] = 1
    with pytest.raises(TrackSpecError, match="monotonically"):
        parse_track_spec(unordered)

    redundant_routes = valid_spec()
    redundant_routes["model"]["selector"]["routes"] = ["layer_0_head_0"]
    with pytest.raises(TrackSpecError, match="exactly"):
        parse_track_spec(redundant_routes)

    impossible_size = valid_spec()
    impossible_size["model"]["max_bytes"] = 100
    with pytest.raises(TrackSpecError, match="cannot hold"):
        parse_track_spec(impossible_size)

    header_too_large = valid_spec()
    header_too_large["model"]["max_bytes"] = 351
    with pytest.raises(TrackSpecError, match="cannot hold"):
        parse_track_spec(header_too_large)

    exact_minimum_size = valid_spec()
    exact_minimum_size["model"]["max_bytes"] = 352
    assert parse_track_spec(exact_minimum_size).model.max_bytes == 352

    uncovered_route = valid_spec()
    uncovered_route["model"]["selector"]["num_heads"] = 2
    uncovered_route["scoring"]["sampled_route_count"] = 6
    with pytest.raises(TrackSpecError, match="available selector routes"):
        parse_track_spec(uncovered_route)

    missing_window = valid_spec()
    missing_window["scoring"]["trajectory_schedule"]["step_windows"][2]["indices"] = [1]
    with pytest.raises(TrackSpecError, match="partition every"):
        parse_track_spec(missing_window)


def test_loader_rejects_duplicate_json_fields(tmp_path) -> None:
    path = tmp_path / "track_spec.json"
    path.write_text('{"track_id":"a","track_id":"b"}', encoding="utf-8")

    with pytest.raises(TrackSpecError, match="duplicate"):
        load_track_spec(path)


def test_loader_accepts_canonical_json_file(tmp_path) -> None:
    path = tmp_path / "track_spec.json"
    path.write_text(json.dumps(valid_spec()), encoding="utf-8")

    assert load_track_spec(path).track_id == "veda-test-v1"
