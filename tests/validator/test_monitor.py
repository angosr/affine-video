from validator.monitor import render_status


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

    assert "VEDA SN120  block 12345  ACTIVE" in rendered
    assert "abcdef" in rendered and "123456" in rendered
    assert "CHAMPION" in rendered
    assert "#1" in rendered and "#2" in rendered
    assert "0.812346" in rendered
    assert "SYSTEM BENCHMARKS" in rendered
    assert "MeanPool control" in rendered
    assert "0.549221" in rendered and "0.505509" in rendered
    assert "do not affect tournament rank" in rendered
