import hashlib
import pickle
from types import SimpleNamespace

import pytest

from scoring.rollout import (
    ProcessRollout,
    RolloutParticipantError,
    RolloutTemporaryError,
)
from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec


def write_runner(path, body: str) -> str:
    path.write_text(f"#!/usr/bin/env python3\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(tmp_path, body: str, *, timeout: float = 2.0):
    runner = tmp_path / "runner"
    runner_digest = write_runner(runner, body)
    panel = tmp_path / "panel.bin"
    panel.write_bytes(b"hidden-panel-with-random-nonce")
    panel_digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    raw = valid_spec()
    raw["rollout"] = {
        "runner_sha256": runner_digest,
        "panel_sha256": panel_digest,
        "panel_max_bytes": 1024,
    }
    track = parse_track_spec(raw)
    rollout = ProcessRollout(
        str(runner),
        str(panel),
        runner_digest,
        panel_digest,
        1024,
        timeout,
    )
    task = SimpleNamespace(
        challenge_id="challenge-1",
        file_sha256="ab" * 32,
        model_path=str(tmp_path / "selector.safetensors"),
    )
    return rollout, task, track, panel


def test_process_rollout_binds_success_to_model_challenge_and_track(tmp_path) -> None:
    rollout, task, track, _panel = build(
        tmp_path,
        """import json, sys
request = json.load(sys.stdin)
print(json.dumps({
    "status": "ok",
    "challenge_id": request["challenge_id"],
    "file_sha256": request["file_sha256"],
    "track_digest": request["track"]["digest"],
    "final_score": 0.73,
    "attention_cost": 0.18,
    "metrics": {"samples": 4},
}, sort_keys=True))""",
    )

    restored = pickle.loads(pickle.dumps(rollout))
    result = restored(task, None, track)

    assert result.final_score == 0.73
    assert result.attention_cost == 0.18
    assert result.metrics == {"samples": 4}


def test_process_rollout_preserves_participant_failure_class(tmp_path) -> None:
    rollout, task, track, _panel = build(
        tmp_path,
        """import json, sys
request = json.load(sys.stdin)
print(json.dumps({
    "status": "participant_error",
    "challenge_id": request["challenge_id"],
    "file_sha256": request["file_sha256"],
    "track_digest": request["track"]["digest"],
    "code": "ROLLOUT_MODEL_INVALID",
    "message": "model output is invalid",
}))""",
    )

    with pytest.raises(RolloutParticipantError) as caught:
        rollout(task, None, track)

    assert caught.value.code == "ROLLOUT_MODEL_INVALID"


def test_process_rollout_rejects_stale_result_binding(tmp_path) -> None:
    rollout, task, track, _panel = build(
        tmp_path,
        """import json, sys
request = json.load(sys.stdin)
print(json.dumps({
    "status": "ok",
    "challenge_id": request["challenge_id"],
    "file_sha256": "00" * 32,
    "track_digest": request["track"]["digest"],
    "final_score": True,
    "attention_cost": 0.1,
    "metrics": {},
}))""",
    )

    with pytest.raises(RolloutTemporaryError) as caught:
        rollout(task, None, track)

    assert caught.value.code == "ROLLOUT_RESPONSE_MISMATCH"


def test_process_rollout_rejects_boolean_score(tmp_path) -> None:
    rollout, task, track, _panel = build(
        tmp_path,
        """import json, sys
request = json.load(sys.stdin)
print(json.dumps({
    "status": "ok",
    "challenge_id": request["challenge_id"],
    "file_sha256": request["file_sha256"],
    "track_digest": request["track"]["digest"],
    "final_score": True,
    "attention_cost": 0.1,
    "metrics": {},
}))""",
    )

    with pytest.raises(RolloutTemporaryError) as caught:
        rollout(task, None, track)

    assert caught.value.code == "ROLLOUT_RESPONSE_INVALID"


def test_process_rollout_rechecks_panel_and_kills_timeout(tmp_path) -> None:
    rollout, task, track, panel = build(
        tmp_path,
        """import json, sys, time
json.load(sys.stdin)
time.sleep(10)""",
        timeout=0.05,
    )

    panel.write_bytes(b"changed-panel-with-random-nonce")
    with pytest.raises(RolloutTemporaryError) as changed:
        rollout(task, None, track)
    assert changed.value.code == "ROLLOUT_PANEL_CHANGED"

    panel.write_bytes(b"hidden-panel-with-random-nonce")
    with pytest.raises(RolloutTemporaryError) as timeout:
        rollout(task, None, track)
    assert timeout.value.code == "ROLLOUT_TIMEOUT"
