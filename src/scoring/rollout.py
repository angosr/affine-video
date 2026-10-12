"""Hash-pinned local runner for authoritative video rollout scoring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from scoring.selector import FixedSelector
from scoring.spec import TrackSpec

_MAX_RESPONSE_BYTES = 1024 * 1024
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class RolloutParticipantError(RuntimeError):
    """The trusted runner classified a model-specific rollout failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RolloutTemporaryError(RuntimeError):
    """The runner or its pinned inputs were unavailable or invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class RolloutResult:
    final_score: float
    attention_cost: float
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.final_score, bool)
            or not isinstance(self.final_score, (int, float))
            or not isfinite(self.final_score)
            or not 0 <= self.final_score <= 1
        ):
            raise ValueError("rollout final score must be within [0, 1]")
        if (
            isinstance(self.attention_cost, bool)
            or not isinstance(self.attention_cost, (int, float))
            or not isfinite(self.attention_cost)
            or self.attention_cost < 0
        ):
            raise ValueError("rollout attention cost must be finite and non-negative")
        if not isinstance(self.metrics, dict):
            raise ValueError("rollout metrics must be a JSON object")
        try:
            json.dumps(
                self.metrics,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("rollout metrics must contain canonical JSON values") from exc


@dataclass(frozen=True, slots=True)
class ProcessRollout:
    """Execute one trusted local rollout program without a shell.

    The runner reads one JSON request from stdin and writes one bounded JSON
    response to stdout. Its executable bytes and the hidden panel are both
    pinned by the track specification and checked again before every run.
    """

    runner: str
    panel: str
    runner_sha256: str
    panel_sha256: str
    panel_max_bytes: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("rollout timeout must be finite and positive")
        if self.panel_max_bytes <= 0:
            raise ValueError("rollout panel byte limit must be positive")
        _expected_digest(self.runner_sha256, "rollout runner")
        _expected_digest(self.panel_sha256, "rollout panel")
        self.validate_assets()

    def validate_assets(self) -> None:
        _verify_file(
            self.runner,
            expected_sha256=self.runner_sha256,
            name="rollout runner",
            executable=True,
        )
        _verify_file(
            self.panel,
            expected_sha256=self.panel_sha256,
            name="rollout panel",
            max_bytes=self.panel_max_bytes,
        )

    def __call__(self, task: Any, _selector: FixedSelector, track: TrackSpec) -> RolloutResult:
        runner = _verify_file(
            self.runner,
            expected_sha256=self.runner_sha256,
            name="rollout runner",
            executable=True,
        )
        panel = _verify_file(
            self.panel,
            expected_sha256=self.panel_sha256,
            name="rollout panel",
            max_bytes=self.panel_max_bytes,
        )
        request = {
            "challenge_id": task.challenge_id,
            "file_sha256": task.file_sha256,
            "model_path": task.model_path,
            "panel_path": str(panel),
            "track": {
                "base_model": {
                    "repo_id": track.base_model.repo_id,
                    "revision": track.base_model.revision,
                },
                "budgets": [
                    {"blocks": item.blocks, "calibrated_cost": item.calibrated_cost}
                    for item in track.scoring.budgets
                ],
                "digest": track.digest,
                "selector": {
                    "head_dim": track.model.selector.head_dim,
                    "num_heads": track.model.selector.num_heads,
                    "num_layers": track.model.selector.num_layers,
                },
                "track_id": track.track_id,
            },
        }
        encoded = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        try:
            process = subprocess.Popen(
                [str(runner)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise RolloutTemporaryError(
                "ROLLOUT_RUNNER_UNAVAILABLE", "rollout runner could not be started"
            ) from exc
        try:
            stdout, _stderr = process.communicate(encoded, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(process)
            process.communicate()
            raise RolloutTemporaryError(
                "ROLLOUT_TIMEOUT", "rollout runner exceeded the wall-clock limit"
            ) from exc
        if process.returncode != 0:
            raise RolloutTemporaryError(
                "ROLLOUT_RUNNER_FAILURE",
                f"rollout runner exited with status {process.returncode}",
            )
        if len(stdout) > _MAX_RESPONSE_BYTES:
            raise RolloutTemporaryError(
                "ROLLOUT_RESPONSE_INVALID", "rollout response exceeds the byte limit"
            )
        response = _response(stdout)
        _verify_binding(response, task, track)
        status = response.get("status")
        common = {"status", "challenge_id", "file_sha256", "track_digest"}
        if status == "ok":
            if set(response) != common | {"final_score", "attention_cost", "metrics"}:
                raise RolloutTemporaryError(
                    "ROLLOUT_RESPONSE_INVALID", "rollout success response has unknown fields"
                )
            try:
                return RolloutResult(
                    response["final_score"],
                    response["attention_cost"],
                    response["metrics"],
                )
            except ValueError as exc:
                raise RolloutTemporaryError("ROLLOUT_RESPONSE_INVALID", str(exc)) from exc
        if status in {"participant_error", "temporary_error"}:
            if set(response) != common | {"code", "message"}:
                raise RolloutTemporaryError(
                    "ROLLOUT_RESPONSE_INVALID", "rollout error response has unknown fields"
                )
            code = response["code"]
            message = response["message"]
            if (
                not isinstance(code, str)
                or _ERROR_CODE.fullmatch(code) is None
                or not isinstance(message, str)
                or not message
                or len(message) > 500
            ):
                raise RolloutTemporaryError(
                    "ROLLOUT_RESPONSE_INVALID", "rollout error response is malformed"
                )
            if status == "participant_error":
                raise RolloutParticipantError(code, message)
            raise RolloutTemporaryError(code, message)
        raise RolloutTemporaryError(
            "ROLLOUT_RESPONSE_INVALID", "rollout response has an unknown status"
        )


def _response(stdout: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RolloutTemporaryError(
            "ROLLOUT_RESPONSE_INVALID", "rollout response is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RolloutTemporaryError(
            "ROLLOUT_RESPONSE_INVALID", "rollout response must be a JSON object"
        )
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _verify_binding(response: dict[str, Any], task: Any, track: TrackSpec) -> None:
    if response.get("challenge_id") != task.challenge_id:
        raise RolloutTemporaryError(
            "ROLLOUT_RESPONSE_MISMATCH", "rollout response challenge does not match the request"
        )
    if response.get("file_sha256") != task.file_sha256:
        raise RolloutTemporaryError(
            "ROLLOUT_RESPONSE_MISMATCH", "rollout response model does not match the request"
        )
    if response.get("track_digest") != track.digest:
        raise RolloutTemporaryError(
            "ROLLOUT_RESPONSE_MISMATCH", "rollout response track does not match the request"
        )


def _verify_file(
    path: str,
    *,
    expected_sha256: str,
    name: str,
    max_bytes: int | None = None,
    executable: bool = False,
) -> Path:
    expected = _expected_digest(expected_sha256, name)
    candidate = Path(path).resolve()
    try:
        stat = candidate.stat()
    except OSError as exc:
        raise RolloutTemporaryError(
            f"{name.upper().replace(' ', '_')}_UNAVAILABLE",
            f"{name} cannot be read",
        ) from exc
    if not candidate.is_file():
        raise RolloutTemporaryError(
            f"{name.upper().replace(' ', '_')}_UNAVAILABLE", f"{name} is not a regular file"
        )
    if max_bytes is not None and stat.st_size > max_bytes:
        raise RolloutTemporaryError("ROLLOUT_PANEL_INVALID", "rollout panel exceeds its byte limit")
    if executable and not os.access(candidate, os.X_OK):
        raise RolloutTemporaryError(
            "ROLLOUT_RUNNER_UNAVAILABLE", "rollout runner is not executable"
        )
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RolloutTemporaryError(
            f"{name.upper().replace(' ', '_')}_UNAVAILABLE", f"{name} cannot be read"
        ) from exc
    if digest.digest() != expected:
        raise RolloutTemporaryError(
            f"{name.upper().replace(' ', '_')}_CHANGED", f"{name} checksum does not match the track"
        )
    return candidate


def _expected_digest(value: str, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} digest must contain 32 bytes")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} digest must be hexadecimal") from exc


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
