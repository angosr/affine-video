import pytest

from monitor import render_status


@pytest.mark.parametrize("observed_at_ns", [None, 1_000_000_000])
def test_monitor_flags_missing_or_stale_inventory(monkeypatch, observed_at_ns):
    monkeypatch.setattr("monitor.time.time_ns", lambda: 200_000_000_000)
    status = {"participants": [], "inventory": {
        "observed_at_ns": observed_at_ns, "ready_batches": 4,
        "producing_batches": 0, "target_batches": 2,
    }}
    rendered = render_status(status, color=False)
    assert "4 ready / 0 producing / 2 target" in rendered
    assert "STALE" in rendered
    assert ("unknown" if observed_at_ns is None else "199s ago") in rendered


def test_monitor_inventory_and_disclosure_error_recovery(monkeypatch):
    monkeypatch.setattr("monitor.time.time_ns", lambda: 200_000_000_000)
    inventory = {
        "observed_at_ns": 199_000_000_000, "disclosure_observed_at_ns": 198_000_000_000,
        "ready_batches": 4, "producing_batches": 0, "target_batches": 2,
        "refresh_error": "INVENTORY_REFRESH_FAILED", "disclosure_error": "DATA_DISCLOSURE_FAILURE",
        "private_upload_error": "PRIVATE_UPLOAD_ISSUANCE_FAILED",
        "private_upload_cleanup_error": "PRIVATE_UPLOAD_CLEANUP_FAILED",
    }
    status = {"participants": [], "inventory": inventory}
    failed = render_status(status, color=False)
    assert "INVENTORY_REFRESH_FAILED" in failed and "DATA_DISCLOSURE_FAILURE" in failed
    assert "PRIVATE_UPLOAD_ISSUANCE_FAILED" in failed and "PRIVATE_UPLOAD_CLEANUP_FAILED" in failed
    assert "1970-01-01T00:03:19+00:00" in failed
    assert "1970-01-01T00:03:18+00:00" in failed
    inventory.update(refresh_error=None, disclosure_error=None, private_upload_error=None,
                     private_upload_cleanup_error=None)
    recovered = render_status(status, color=False)
    assert "INVENTORY_REFRESH_FAILED" not in recovered
    assert "DATA_DISCLOSURE_FAILURE" not in recovered and "STALE" not in recovered
    assert "PRIVATE_UPLOAD_ISSUANCE_FAILED" not in recovered
    assert "PRIVATE_UPLOAD_CLEANUP_FAILED" not in recovered
    assert "4 ready / 0 producing / 2 target" in recovered


def test_monitor_renders_all_active_competitors_in_uid_order() -> None:
    status = {
        "netuid": 120,
        "source_block": 12345,
        "system_matches": [
            {
                "sequence": 1,
                "reference": "VEDA baseline",
                "challenger": "MeanPool control",
                "reference_score": 0.549221386869737,
                "challenger_score": 0.505509267860990,
                "outcome": "REFERENCE_WON",
                "detail": "system_matches/000000000001.json",
            }
        ],
        "participants": [
            {
                "uid": 0,
                "hotkey": "abcdef012345",
                "state": "SUCCEEDED",
                "queue_position": None,
                "match_sequence": 1,
                "score": 0.81234567,
                "outcome": "INITIAL_CHAMPION",
                "champion": True,
            },
            {
                "uid": 9,
                "hotkey": "123456abcdef",
                "state": "QUEUED",
                "queue_position": 2,
                "match_sequence": None,
                "score": None,
                "outcome": None,
                "champion": False,
            },
        ],
    }

    rendered = render_status(status, color=False)

    assert "VEDA SN120  block 12345  UNKNOWN" in rendered
    assert "abcdef" in rendered and "123456" in rendered
    assert "CHAMPION" in rendered
    assert "#1" in rendered and "#2" in rendered
    assert "0.812346" in rendered
    assert "SYSTEM BENCHMARKS" in rendered
    assert "MeanPool control" in rendered
    assert "0.549221" in rendered and "0.505509" in rendered
    assert "do not affect tournament rank" in rendered


@pytest.mark.parametrize("phase", ["SCHEDULED", "ACCEPTING"])
def test_monitor_displays_only_the_reported_phase(phase):
    rendered = render_status(
        {"netuid": 120, "source_block": 12345, "track_phase": phase, "participants": []},
        color=False,
    )
    assert f"VEDA SN120  block 12345  {phase}" in rendered
