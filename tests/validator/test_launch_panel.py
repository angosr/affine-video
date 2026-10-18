"""Launch-only gates are tested without importing them into production scoring."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PANEL = Path(__file__).resolve().parents[2] / "evidence/system-panel/launch/runner.py"
spec = importlib.util.spec_from_file_location("launch_panel", PANEL)
assert spec is not None and spec.loader is not None
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def controls():
    return {
        name: {"gain_lower_bound": 0.0 if name == "baseline_self" else -0.01, "promoted": False}
        for name in panel.EXPECTED_IDENTITIES
    }


def test_panel_positive_and_negative_directions_are_required():
    result = panel._directional_gates(controls(), 0.02, 0.005, "production-00")
    assert result["reverse_veda_vs_shuffled"]["promoted"] is True


@pytest.mark.parametrize("name", sorted(panel.HARD_STABILITY_IDENTITIES))
def test_consistently_wrong_control_direction_fails(name):
    values = controls()
    values[name] = {"gain_lower_bound": 0.01, "promoted": True}
    with pytest.raises(RuntimeError):
        panel._directional_gates(values, 0.02, 0.005, "production-00")


@pytest.mark.parametrize("reverse", [0.005, 0.0, -0.02, float("nan"), float("inf")])
def test_missing_positive_power_fails(reverse):
    with pytest.raises(RuntimeError):
        panel._directional_gates(controls(), reverse, 0.005, "production-00")


def test_full_video_covers_dynamic_incumbent_even_when_baseline_comparison_rejects(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    document = {
        "candidate_track_spec_digest": "track",
        "runtime_contract_sha256": panel.RUNTIME_CONTRACT,
        "model_identities": panel.EXPECTED_IDENTITIES,
        "batches": [
            {"label": f"production-{index:02d}", "partition": "production", "controls": controls()}
            for index in range(panel.PRODUCTION_EXACT)
        ],
    }
    (tmp_path / "exact-summary.json").write_text(json.dumps(document))
    decisions = panel._full_video_eligibility(
        SimpleNamespace(digest="track", scoring=SimpleNamespace(minimum_frontier_gain=0.005))
    )["decisions"]
    assert all(row["eligible"] for row in decisions.values())
    calls = []
    replay = panel._two_stage_fifo_replay(
        order=("trained_near_neighbor", "mean_pool"),
        exact_labels=("production-00", "production-01"),
        confirmation_labels=("production-10", "production-11"),
        exact_bound=lambda challenger, incumbent, _label: (
            0.02 if challenger == "trained_near_neighbor" or incumbent != "baseline_self" else -0.02
        ),
        video_bound=lambda challenger, incumbent, label: (
            calls.append((challenger, incumbent, label)) or 0.02
        ),
        exact_threshold=0.005, video_threshold=0.005,
    )
    assert replay["terminal_champion"] == "mean_pool"
    assert calls[-1][:2] == ("mean_pool", "trained_near_neighbor")
    assert all(decisions[name]["eligible"] for call in calls for name in call[:2])


def test_intermediate_runtime_binding_cannot_launch(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"format": "veda-panel-launch-binding-v1", "frozen": False}))
    with pytest.raises(RuntimeError, match="frozen final runtime"):
        panel.configure(binding)


def test_panel_interval_percentile_and_two_stage_self_checks():
    panel.self_test()
