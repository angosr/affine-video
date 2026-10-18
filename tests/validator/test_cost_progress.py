"""The observer reports deadline risk but never turns an estimate into a failed job."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cost_progress",
    Path(__file__).resolve().parents[2] / "evidence/system-panel/launch/watch_cost_attempt.py",
)
assert SPEC and SPEC.loader
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def fixture():
    started = "2026-10-18T00:00:00Z"
    start = datetime.fromisoformat(started).timestamp()
    registration = {
        "timeout_seconds_per_attempt": 21600,
        "runtime_image_id": "image",
        "text_token_counts": [77, 256],
        "models": ["pattern_spread", "baseline_self"],
        "layers": [0, 1],
        "budgets": [16, 32, 64],
    }
    job = {
        "container_id": "container",
        "started_at_utc": started,
        "gpu": 7,
        "geometries": ["4:3@102", "9:16@37"],
    }
    observation = {
        "id": "container",
        "image": "image",
        "entrypoint": ["timeout"],
        "command": ["21600s", "python", "cost_calibration.py"],
        "now": start + 120,
        "state": {"Status": "running", "ExitCode": 0, "StartedAt": started},
        "hashes": {"protocol.json": "protocol"},
        "mtimes": {
            f"4x3@102-77-pattern_spread-layer{layer}-{budget}.json": start + 10 * (i + 1)
            for i, (layer, budget) in enumerate(
                [(layer, budget) for layer in (0, 1) for budget in (16, 32, 64)]
            )
        },
    }
    return registration, job, observation


def test_completed_group_rate_uses_original_remote_completion_boundaries():
    result = watcher.progress(*fixture())
    assert result["rows"] == 6
    assert result["expected_rows"] == 48
    assert result["completed_groups"][0]["rows_per_minute"] == 6
    assert result["remaining_seconds_at_completed_group_mean_rate"] == 420
    assert len(result["incomplete_groups"]) == 7
    assert result["release_accepted"] is False


def test_slow_projection_does_not_fail_or_stop_a_healthy_attempt():
    registration, job, observation = fixture()
    start = datetime.fromisoformat(job["started_at_utc"]).timestamp()
    observation["mtimes"] = {
        name: start + 1000 * (i + 1) for i, name in enumerate(observation["mtimes"])
    }
    observation["now"] = start + 6000
    result = watcher.progress(registration, job, observation)
    assert result["projection_exceeds_remaining_time"]
    assert not result["measurement_failed"]
    assert result["status"] == "running"
    assert not result["measurement_complete_awaiting_registered_analysis"]


def test_registered_timeout_exit_is_a_failed_attempt():
    registration, job, observation = fixture()
    start = datetime.fromisoformat(job["started_at_utc"]).timestamp()
    observation["state"].update(
        Status="exited",
        ExitCode=124,
        FinishedAt=datetime.fromtimestamp(start + 21600, UTC).isoformat(),
    )
    observation["now"] = start + 22000
    result = watcher.progress(registration, job, observation)
    assert result["measurement_failed"]
    assert result["deadline_elapsed"]
    assert result["elapsed_seconds"] == 21600


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "another-attempt"),
        ("image", "another-image"),
        ("command", ["36000s"]),
        ("entrypoint", ["python"]),
    ],
)
def test_changed_handle_or_extended_timeout_is_rejected(field, value):
    registration, job, observation = fixture()
    observation[field] = value
    with pytest.raises(ValueError, match="live container"):
        watcher.progress(registration, job, observation)


def test_missing_rows_do_not_complete_a_boundary_and_future_group_cannot_hide_a_hole():
    registration, job, observation = fixture()
    del observation["mtimes"][next(iter(observation["mtimes"]))]
    assert watcher.progress(registration, job, observation)["completed_groups"] == []
    for layer in (0, 1):
        for budget in (16, 32, 64):
            observation["mtimes"][f"4x3@102-77-baseline_self-layer{layer}-{budget}.json"] = (
                observation["now"] + 1
            )
    with pytest.raises(ValueError, match="incomplete registered group"):
        watcher.progress(registration, job, observation)


def test_success_requires_every_row_and_summary_and_retains_original_finish_time():
    registration, job, observation = fixture()
    start = datetime.fromisoformat(job["started_at_utc"]).timestamp()
    observation["mtimes"] = {}
    for geometry in job["geometries"]:
        for text in registration["text_token_counts"]:
            for model in registration["models"]:
                for layer in registration["layers"]:
                    for budget in registration["budgets"]:
                        name = (
                            f"{geometry.replace(':', 'x')}-{text}-{model}"
                            f"-layer{layer}-{budget}.json"
                        )
                        observation["mtimes"][name] = start + len(observation["mtimes"]) + 1
    observation["state"].update(
        Status="exited", ExitCode=0, FinishedAt=datetime.fromtimestamp(start + 50, UTC).isoformat()
    )
    observation["now"] = start + 30000
    assert watcher.progress(registration, job, observation)["measurement_failed"]
    observation["hashes"]["summary.json"] = "summary"
    result = watcher.progress(registration, job, observation)
    assert result["measurement_complete_awaiting_registered_analysis"]
    assert not result["measurement_failed"]
    assert not result["deadline_elapsed"]
    assert result["elapsed_seconds"] == 50
    assert not result["release_accepted"]
