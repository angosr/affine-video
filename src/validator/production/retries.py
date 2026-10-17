"""CAS-backed retry checkpoints that survive loss of the evaluator host."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from validator.production.commitment import canonical_json, decode_canonical_json
from validator.production.storage import PreconditionFailed, PrivateObjectStore
from validator.retries import RetryCheckpoint

_STAGES = frozenset({"ADMISSION", "SCORING", "CONFIRMING"})
_STATES = frozenset({"ACTIVE", "DEFERRED", "BLOCKED"})
_FIELDS = {
    "attempts",
    "challenge_id",
    "error_code",
    "error_summary",
    "format",
    "generation",
    "hotkey",
    "last_attempt",
    "repo_id",
    "retry_after_ms",
    "revision",
    "stage",
    "state",
    "submission_id",
    "track_spec_digest",
}


@dataclass(frozen=True, slots=True)
class _StoredRetry:
    document: dict[str, Any]
    attempts: int
    generation: int
    state: str
    retry_after_ms: int | None
    error_code: str | None
    error_summary: str | None
    last_attempt: str | None

    def checkpoint(self) -> RetryCheckpoint | None:
        if self.state == "ACTIVE":
            return None
        assert self.error_code is not None and self.error_summary is not None
        return RetryCheckpoint(
            self.attempts,
            self.state == "BLOCKED",
            self.retry_after_ms,
            self.error_code,
            self.error_summary,
        )


class PrivateRetryStore:
    """One authoritative retry projection per challenge stage."""

    def __init__(self, objects: PrivateObjectStore, *, track_spec_digest: str) -> None:
        if not _digest(track_spec_digest):
            raise ValueError("track specification digest is invalid")
        self.objects = objects
        self.track_spec_digest = track_spec_digest

    def restore(self, task: object, stage: str) -> RetryCheckpoint | None:
        stored = self.objects.get_optional(self._name(task.challenge_id, stage))
        if stored is None:
            return None
        return self._decode(stored.body, task=task, stage=stage).checkpoint()

    def record_failure(
        self,
        task: object,
        stage: str,
        *,
        code: str,
        summary: str,
        retry_after_ms: int,
        max_attempts: int,
    ) -> RetryCheckpoint:
        if not code or not summary:
            raise ValueError("retry error identity cannot be empty")
        if retry_after_ms < 0 or max_attempts <= 0:
            raise ValueError("retry limits are invalid")
        name = self._name(task.challenge_id, stage)
        attempt_token = hashlib.sha256(task.lease_token.encode()).hexdigest()
        for _ in range(8):
            stored = self.objects.get_optional(name)
            if stored is None:
                current = None
                attempts = 1
                generation = 0
            else:
                current = self._decode(stored.body, task=task, stage=stage)
                checkpoint = current.checkpoint()
                if current.last_attempt == attempt_token or current.state == "BLOCKED":
                    if checkpoint is None:
                        raise RuntimeError("active retry state retained a failed attempt token")
                    return checkpoint
                attempts = current.attempts + 1
                generation = current.generation + 1
            blocked = attempts >= max_attempts
            document = self._document(
                task,
                stage,
                attempts=attempts,
                generation=generation,
                state="BLOCKED" if blocked else "DEFERRED",
                retry_after_ms=None if blocked else retry_after_ms,
                error_code=code[:100],
                error_summary=summary[:1000],
                last_attempt=attempt_token,
            )
            payload = canonical_json(document)
            try:
                if stored is None:
                    updated = self.objects.create_plan(name, payload)
                else:
                    updated = self.objects.replace_plan(name, payload, etag=stored.etag)
            except PreconditionFailed:
                continue
            checkpoint = self._decode(updated.body, task=task, stage=stage).checkpoint()
            assert checkpoint is not None
            return checkpoint
        raise RuntimeError("durable retry state changed repeatedly")

    def reset(self, challenge_id: str, stage: str) -> None:
        """Persist an explicit operator retry before releasing local state."""

        name = self._name(challenge_id, stage)
        for _ in range(8):
            stored = self.objects.get_optional(name)
            if stored is None:
                raise RuntimeError("durable blocked retry state is missing")
            current = self._decode(stored.body, challenge_id=challenge_id, stage=stage)
            if current.state == "ACTIVE":
                return
            document = dict(current.document)
            document.update(
                {
                    "attempts": 0,
                    "error_code": None,
                    "error_summary": None,
                    "generation": current.generation + 1,
                    "last_attempt": None,
                    "retry_after_ms": None,
                    "state": "ACTIVE",
                }
            )
            try:
                self.objects.replace_plan(name, canonical_json(document), etag=stored.etag)
            except PreconditionFailed:
                continue
            return
        raise RuntimeError("durable retry state changed repeatedly")

    def _decode(
        self,
        payload: bytes,
        *,
        stage: str,
        task: object | None = None,
        challenge_id: str | None = None,
    ) -> _StoredRetry:
        value = decode_canonical_json(payload)
        if not isinstance(value, dict) or set(value) != _FIELDS:
            raise RuntimeError("durable retry state fields are invalid")
        expected_challenge = task.challenge_id if task is not None else challenge_id
        expected = {
            "format": "veda-retry-state-v1",
            "track_spec_digest": self.track_spec_digest,
            "challenge_id": expected_challenge,
            "stage": self._stage(stage),
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise RuntimeError("durable retry state has the wrong identity")
        if (
            not _digest(value["submission_id"])
            or not isinstance(value["hotkey"], str)
            or not value["hotkey"]
            or not isinstance(value["repo_id"], str)
            or not value["repo_id"]
            or not isinstance(value["revision"], str)
            or len(value["revision"]) != 40
            or not _hex(value["revision"])
        ):
            raise RuntimeError("durable retry submission identity is invalid")
        if task is not None and any(
            value.get(key) != getattr(task, key)
            for key in ("submission_id", "hotkey", "repo_id", "revision")
        ):
            raise RuntimeError("durable retry state conflicts with its submission")
        attempts = value["attempts"]
        generation = value["generation"]
        state = value["state"]
        retry_after = value["retry_after_ms"]
        code = value["error_code"]
        summary = value["error_summary"]
        last_attempt = value["last_attempt"]
        if (
            type(attempts) is not int
            or attempts < 0
            or type(generation) is not int
            or generation < 0
            or state not in _STATES
            or (retry_after is not None and (type(retry_after) is not int or retry_after < 0))
            or (code is not None and (not isinstance(code, str) or not code or len(code) > 100))
            or (
                summary is not None
                and (not isinstance(summary, str) or not summary or len(summary) > 1000)
            )
            or (last_attempt is not None and not _digest(last_attempt))
        ):
            raise RuntimeError("durable retry state values are invalid")
        if state == "ACTIVE":
            valid_state = (
                attempts == 0
                and retry_after is None
                and code is None
                and summary is None
                and last_attempt is None
            )
        elif state == "DEFERRED":
            valid_state = (
                attempts > 0
                and retry_after is not None
                and code is not None
                and summary is not None
                and last_attempt is not None
            )
        else:
            valid_state = (
                attempts > 0
                and retry_after is None
                and code is not None
                and summary is not None
                and last_attempt is not None
            )
        if not valid_state:
            raise RuntimeError("durable retry state is inconsistent")
        return _StoredRetry(
            value,
            attempts,
            generation,
            state,
            retry_after,
            code,
            summary,
            last_attempt,
        )

    def _document(
        self,
        task: object,
        stage: str,
        **state: object,
    ) -> dict[str, Any]:
        return {
            "format": "veda-retry-state-v1",
            "track_spec_digest": self.track_spec_digest,
            "challenge_id": task.challenge_id,
            "submission_id": task.submission_id,
            "hotkey": task.hotkey,
            "repo_id": task.repo_id,
            "revision": task.revision,
            "stage": self._stage(stage),
            **state,
        }

    @staticmethod
    def _name(challenge_id: str, stage: str) -> str:
        if not _digest(challenge_id):
            raise ValueError("challenge ID is invalid")
        return f"retry_state/{challenge_id}/{PrivateRetryStore._stage(stage).lower()}.json"

    @staticmethod
    def _stage(stage: str) -> str:
        if stage not in _STAGES:
            raise ValueError("retry stage is invalid")
        return stage


def _digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def _hex(value: str) -> bool:
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()
