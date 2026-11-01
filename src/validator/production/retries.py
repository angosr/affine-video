"""CAS-backed retry checkpoints that survive loss of the evaluator host."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from miner.submit import PrivateModelSubmission, declared_model_sha256, model_source_fields
from validator.production.commitment import canonical_json, decode_canonical_json
from validator.production.storage import (
    RETRY_STATE_NAMESPACE,
    PreconditionFailed,
    PrivateObjectStore,
)
from validator.queue import ChallengeState
from validator.retries import (
    HF_PUBLICATION_GRACE_MS,
    HF_PUBLICATION_WAIT,
    PublicationGrace,
    RetryCheckpoint,
)

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
    http_grace: PublicationGrace | None

    def checkpoint(self) -> RetryCheckpoint | None:
        if self.state == "ACTIVE":
            return (None if self.http_grace is None else RetryCheckpoint(
                0, False, self.http_grace.deadline_ms, HF_PUBLICATION_WAIT,
                "Waiting for anonymous HF publication", self.http_grace,
            ))
        assert self.error_code is not None and self.error_summary is not None
        assert self.retry_after_ms is not None
        return RetryCheckpoint(
            self.attempts,
            self.state == "BLOCKED",
            self.retry_after_ms,
            self.error_code,
            self.error_summary,
            self.http_grace,
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
        if type(task.attempts) is not int or task.attempts <= 0:
            raise ValueError("retry lease ordinal must be positive")
        name = self._name(task.challenge_id, stage)
        attempt_token = hashlib.sha256(task.lease_token.encode()).hexdigest()
        for _ in range(8):
            stored = self.objects.get_optional(name)
            if stored is None:
                current = None
                attempts = task.attempts
                generation = 0
            else:
                current = self._decode(stored.body, task=task, stage=stage)
                checkpoint = current.checkpoint()
                if (current.last_attempt == attempt_token
                        and current.error_code != HF_PUBLICATION_WAIT):
                    if checkpoint is None:
                        raise RuntimeError("active retry state retained a failed attempt token")
                    return checkpoint
                if current.state == "BLOCKED":
                    assert checkpoint is not None
                    return checkpoint
                if current.attempts >= max_attempts:
                    raise RuntimeError("durable retry budget requires an operator reset")
                attempts = current.attempts + 1
                generation = current.generation + 1
            if attempts != task.attempts:
                raise RuntimeError("durable retry ordinal disagrees with the active lease")
            blocked = attempts >= max_attempts
            document = self._document(
                task,
                stage,
                attempts=attempts,
                generation=generation,
                state="BLOCKED" if blocked else "DEFERRED",
                retry_after_ms=retry_after_ms,
                error_code=code[:100],
                error_summary=summary[:1000],
                last_attempt=attempt_token,
                **({"http_grace": current.document["http_grace"]}
                   if current is not None and "http_grace" in current.document else
                   {"http_grace": None} if current is None and stage == "ADMISSION"
                   and getattr(task, "private_submission", None) is None
                   and code != "HF_PUBLICATION_CLOCK_UNAVAILABLE" else {}),
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

    def record_publication_wait(
        self, task: object, *, http_status: int, observed_at_ms: int, max_attempts: int,
    ) -> RetryCheckpoint:
        """Seal an HF publication wait in the original row without spending infra budget."""
        if (task.state not in (ChallengeState.FETCHING, ChallengeState.VALIDATING)
                or getattr(task, "private_submission", None) is not None or not task.repo_id
                or http_status not in (401, 403, 404) or type(http_status) is not int
                or type(observed_at_ms) is not int or observed_at_ms < 0 or max_attempts <= 0):
            raise ValueError("HF publication observation is invalid")
        name = self._name(task.challenge_id, "ADMISSION")
        token = hashlib.sha256(task.lease_token.encode()).hexdigest()
        for _ in range(8):
            stored = self.objects.get_optional(name)
            current = (None if stored is None else
                       self._decode(stored.body, task=task, stage="ADMISSION"))
            if current is not None:
                if current.state == "BLOCKED" or current.attempts >= max_attempts:
                    raise RuntimeError("durable retry budget requires an operator reset")
                if "http_grace" not in current.document:
                    raise RuntimeError("historical HF publication clock is unknown")
                if current.last_attempt == token and current.error_code == HF_PUBLICATION_WAIT:
                    checkpoint = current.checkpoint()
                    assert checkpoint is not None
                    return checkpoint
            attempts = 0 if current is None else current.attempts
            if task.attempts != attempts + 1:
                raise RuntimeError("publication wait ordinal disagrees with the active lease")
            grace = None if current is None else current.http_grace
            if grace is None:
                grace = PublicationGrace(observed_at_ms,
                                         observed_at_ms + HF_PUBLICATION_GRACE_MS, http_status)
            if observed_at_ms < grace.first_observed_at_ms:
                raise RuntimeError("HF publication observation precedes its fixed clock")
            checkpoint = RetryCheckpoint(attempts, False, grace.deadline_ms,
                                         HF_PUBLICATION_WAIT, "Waiting for HF publication",
                                         grace)
            if current is not None and observed_at_ms >= grace.deadline_ms:
                return checkpoint
            payload = canonical_json(self._document(
                task, "ADMISSION", attempts=attempts,
                generation=0 if current is None else current.generation + 1,
                state="DEFERRED", retry_after_ms=grace.deadline_ms,
                error_code=HF_PUBLICATION_WAIT, error_summary=checkpoint.error_summary,
                last_attempt=token, http_grace=asdict(grace),
            ))
            try:
                if stored is None:
                    self.objects.create_plan(name, payload)
                else:
                    self.objects.replace_plan(name, payload, etag=stored.etag)
            except PreconditionFailed:
                continue
            except Exception:
                # A lost PUT response is not an infrastructure failure when the same row
                # proves this lease's exact publication wait was already committed.
                committed = self.objects.get_optional(name)
                if committed is None or committed.body != payload:
                    raise
                restored = self._decode(committed.body, task=task, stage="ADMISSION").checkpoint()
                assert restored is not None
                return restored
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

    def disclosure_attempt(
        self, plan_digest: str, *, now_ms: int, lease_ms: int,
        completed_attempt: int | None = None, error: str | None = None,
    ) -> int | None:
        """One durable, bounded bulk attempt; never change a miner's retry state.

        The next-attempt fence is written before I/O and survives a killed host
        or a lost PUT response. Finishing updates this same fact, not a second
        queue. A successful public receipt remains the completion authority.
        """
        if not _digest(plan_digest) or now_ms < 0 or lease_ms <= 0:
            raise ValueError("disclosure retry identity or limits are invalid")
        name = f"{RETRY_STATE_NAMESPACE}/disclosures/{plan_digest}.json"
        for _ in range(8):
            stored = self.objects.get_optional(name)
            row = None if stored is None else self._decode_disclosure(stored.body, plan_digest)
            if row is not None and "operator_waiver" in row:
                return None
            if completed_attempt is None:
                if row is not None and row["retry_after_ms"] > now_ms:
                    return None
                attempt = 1 if row is None else row["attempt"] + 1
                retry_after = now_ms + lease_ms
            else:
                if row is None or row["attempt"] != completed_attempt:
                    raise RuntimeError("disclosure completion lost its attempt fence")
                attempt = completed_attempt
                retry_after = (0 if error is None else
                               now_ms + min(3_600_000, 60_000 * 2 ** min(attempt - 1, 6)))
            payload = canonical_json({
                "format": "veda-disclosure-retry-v1", "track_spec_digest": self.track_spec_digest,
                "plan_digest": plan_digest, "attempt": attempt, "retry_after_ms": retry_after,
                "error": (row["error"] if completed_attempt is None and row is not None
                          else None if error is None else error[:1000]),
            })
            try:
                if stored is None:
                    self.objects.create_plan(name, payload)
                else:
                    self.objects.replace_plan(name, payload, etag=stored.etag)
            except PreconditionFailed:
                continue
            return attempt
        raise RuntimeError("disclosure retry authority changed repeatedly")

    def disclosure_failed(self, plan_digest: str) -> bool:
        """Read unresolved failure from the original checkpoint, including in-flight retries."""
        return self.disclosure_status(plan_digest, now_ms=0)["state"] == "FAILED"

    def disclosure_status(self, plan_digest: str, *, now_ms: int) -> dict[str, Any]:
        """Project the original retry authority; a waiver is never a public receipt."""
        if not _digest(plan_digest):
            raise ValueError("disclosure retry identity is invalid")
        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("disclosure observation time is invalid")
        stored = self.objects.get_optional(
            f"{RETRY_STATE_NAMESPACE}/disclosures/{plan_digest}.json",
        )
        row = None if stored is None else self._decode_disclosure(stored.body, plan_digest)
        state = "OVERDUE"
        if row is not None:
            if "operator_waiver" in row:
                state = "WAIVED"
            elif row["error"] is not None:
                state = "FAILED"  # Reserving another lease does not clear the prior failure.
            elif row["retry_after_ms"] > now_ms:
                state = "UPLOADING"
        return {
            "plan_digest": plan_digest, "state": state,
            "attempt": 0 if row is None else row["attempt"],
            "retry_after_ms": (None if row is None or state == "WAIVED"
                               else row["retry_after_ms"]),
            "waived_at_ms": (row["operator_waiver"]["decided_at_ms"]
                             if state == "WAIVED" else None),
        }

    def waive_disclosure(self, plan_digest: str, *, now_ms: int, reason: str) -> None:
        """Explicit operator decision in the same CAS row, preserving attempt/error history.

        An unexpired fence must settle first: this operation cannot retract bytes
        already being uploaded by another process. It never creates a receipt.
        """
        if (not _digest(plan_digest) or type(now_ms) is not int or now_ms < 0
                or not isinstance(reason, str) or not reason.strip() or len(reason) > 256
                or any(ord(char) < 32 or ord(char) == 127 for char in reason)):
            raise ValueError("disclosure waiver decision is invalid")
        name = f"{RETRY_STATE_NAMESPACE}/disclosures/{plan_digest}.json"
        for _ in range(8):
            stored = self.objects.get_optional(name)
            row = None if stored is None else self._decode_disclosure(stored.body, plan_digest)
            if row is not None and "operator_waiver" in row:
                if row["operator_waiver"]["reason"] != reason:
                    raise RuntimeError("disclosure already has a different operator decision")
                return  # A lost response/restart must not replace the original decision time.
            if row is not None and row["retry_after_ms"] > now_ms:
                raise RuntimeError("disclosure has an unexpired attempt fence")
            document = dict(row) if row is not None else {
                "format": "veda-disclosure-retry-v1", "track_spec_digest": self.track_spec_digest,
                "plan_digest": plan_digest, "attempt": 0, "retry_after_ms": 0, "error": None,
            }
            document["operator_waiver"] = {"decided_at_ms": now_ms, "reason": reason}
            try:
                if stored is None:
                    self.objects.create_plan(name, canonical_json(document))
                else:
                    self.objects.replace_plan(name, canonical_json(document), etag=stored.etag)
            except PreconditionFailed:
                continue
            return
        raise RuntimeError("disclosure retry authority changed repeatedly")

    def _decode_disclosure(self, payload: bytes, plan_digest: str) -> dict[str, Any]:
        row = decode_canonical_json(payload)
        fields = {"format", "track_spec_digest", "plan_digest", "attempt",
                  "retry_after_ms", "error"}
        if (
            not isinstance(row, dict) or set(row) not in (fields, fields | {"operator_waiver"})
            or row["format"] != "veda-disclosure-retry-v1"
            or row["track_spec_digest"] != self.track_spec_digest
            or row["plan_digest"] != plan_digest
            or type(row["attempt"]) is not int or row["attempt"] < 0
            or type(row["retry_after_ms"]) is not int or row["retry_after_ms"] < 0
            or (row["error"] is not None and not isinstance(row["error"], str))
        ):
            raise RuntimeError("disclosure retry authority is invalid")
        waiver = row.get("operator_waiver")
        if "operator_waiver" in row and (
            not isinstance(waiver, dict) or set(waiver) != {"decided_at_ms", "reason"}
            or type(waiver["decided_at_ms"]) is not int or waiver["decided_at_ms"] < 0
            or not isinstance(waiver["reason"], str) or not waiver["reason"].strip()
            or len(waiver["reason"]) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in waiver["reason"])
        ):
            raise RuntimeError("disclosure retry authority is invalid")
        if row["attempt"] == 0 and (
            waiver is None or row["retry_after_ms"] != 0 or row["error"] is not None
        ):
            raise RuntimeError("disclosure retry authority is invalid")
        return row

    def _decode(
        self,
        payload: bytes,
        *,
        stage: str,
        task: object | None = None,
        challenge_id: str | None = None,
    ) -> _StoredRetry:
        value = decode_canonical_json(payload)
        if (not isinstance(value, dict) or not set(value) >= _FIELDS
                or set(value) - _FIELDS - {"private_upload", "http_grace", "declared_file_sha256"}):
            raise RuntimeError("durable retry state fields are invalid")
        private = None
        if "private_upload" in value:
            try:
                private = PrivateModelSubmission(**value["private_upload"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("durable retry private source is invalid") from exc
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
            or (private is not None and (value["repo_id"] is not None
                                         or value["revision"] is not None))
            or (private is None and (
                not isinstance(value["repo_id"], str) or not value["repo_id"]
                or not isinstance(value["revision"], str) or len(value["revision"]) != 40
                or not _hex(value["revision"])
            ))
        ):
            raise RuntimeError("durable retry submission identity is invalid")
        if task is not None and any(
            value.get(key) != getattr(task, key)
            for key in ("submission_id", "hotkey", "repo_id", "revision")
        ):
            raise RuntimeError("durable retry state conflicts with its submission")
        if task is not None and private != getattr(task, "private_submission", None):
            raise RuntimeError("durable retry state conflicts with its private source")
        declaration = declared_model_sha256(value)
        if task is not None and declaration != getattr(task, "declared_file_sha256", None):
            raise RuntimeError("durable retry state conflicts with its HF declaration")
        grace = None
        if "http_grace" in value:
            if stage != "ADMISSION" or private is not None:
                raise RuntimeError("HF publication clock requires an HF admission")
            if value["http_grace"] is not None:
                try:
                    grace = PublicationGrace(**value["http_grace"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("HF publication clock is invalid") from exc
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
        elif code == HF_PUBLICATION_WAIT:
            valid_state = (state == "DEFERRED" and grace is not None
                           and retry_after == grace.deadline_ms and attempts >= 0
                           and summary is not None and last_attempt is not None)
        else:
            valid_state = (
                attempts > 0
                and retry_after is not None
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
            grace,
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
            **model_source_fields(task.repo_id, task.revision,
                                  getattr(task, "private_submission", None),
                                  declared_file_sha256=getattr(task, "declared_file_sha256", None)),
            "stage": self._stage(stage),
            **state,
        }

    @staticmethod
    def _name(challenge_id: str, stage: str) -> str:
        if not _digest(challenge_id):
            raise ValueError("challenge ID is invalid")
        return (
            f"{RETRY_STATE_NAMESPACE}/{challenge_id}/"
            f"{PrivateRetryStore._stage(stage).lower()}.json"
        )

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
