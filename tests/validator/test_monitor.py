from validator.monitor import render_status


def test_monitor_renders_all_active_competitors_in_uid_order() -> None:
    status = {
        "netuid": 120,
        "source_block": 12345,
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

    assert "VEDA SN120  block 12345" in rendered
    assert "abcdef" in rendered and "123456" in rendered
    assert "CHAMPION" in rendered
    assert "#1" in rendered and "#2" in rendered
    assert "0.812346" in rendered
