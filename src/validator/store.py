"""Durable challenge history and current-state projection."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from validator.queue import (
    ChainPosition,
    ChallengeState,
    challenge_id,
    passes_cooldown,
    require_transition,
)
from validator.queue import (
    submission_id as derive_submission_id,
)

_SQLITE_BUSY_TIMEOUT_MS = 30_000


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """One finalized on-chain submission in its canonical order."""

    submission_id: str
    hotkey: str
    netuid: int
    block_hash: str
    position: ChainPosition
    payload: bytes
    repo_id: str | None
    revision: str | None

    def __post_init__(self) -> None:
        if not self.hotkey:
            raise ValueError("hotkey cannot be empty")
        if self.netuid < 0 or self.netuid > 0xFFFF:
            raise ValueError("netuid must fit in an unsigned 16-bit integer")
        if not self.payload:
            raise ValueError("payload cannot be empty")
        normalized_hash = _block_hash(self.block_hash)
        object.__setattr__(self, "block_hash", normalized_hash)
        if self.submission_id != derive_submission_id(self.netuid, normalized_hash, self.position):
            raise ValueError("submission identifier does not match its chain position")
        if (self.repo_id is None) != (self.revision is None):
            raise ValueError("repository and revision must either both exist or both be absent")


@dataclass(frozen=True, slots=True)
class ChallengeRecord:
    challenge_id: str
    submission_id: str
    state: ChallengeState
    attempts: int
    error_code: str | None
    error_summary: str | None
    final_score: float | None
    attention_cost: float | None
    score_block: int | None
    provisional: bool
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ChallengeEvent:
    sequence: int
    previous_state: ChallengeState | None
    state: ChallengeState
    occurred_at_ms: int
    error_code: str | None
    error_summary: str | None
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RecordedStageScore:
    stage: ChallengeState
    next_state: ChallengeState
    score: float
    attention_cost: float
    score_block: int
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    hotkey: str
    challenge_id: str
    final_score: float
    attention_cost: float | None
    provisional: bool
    score_block: int
    position: ChainPosition


@dataclass(frozen=True, slots=True)
class LeasedChallenge:
    challenge_id: str
    submission_id: str
    state: ChallengeState
    hotkey: str
    repo_id: str
    revision: str
    position: ChainPosition
    lease_token: str
    lease_expires_at_ms: int
    attempts: int
    model_path: str | None
    file_sha256: str | None


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    accepted: bool
    duplicate_of: str | None


class ChallengeStore:
    """Small SQLite boundary used only by the validator process."""

    def __init__(
        self,
        path: str | Path,
        *,
        track_spec_digest: str | None = None,
    ) -> None:
        self._connection = sqlite3.connect(Path(path), timeout=_SQLITE_BUSY_TIMEOUT_MS / 1_000)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._initialize()
            _enable_wal(self._connection)
            if track_spec_digest is not None:
                self.bind_track_spec(track_spec_digest)
        except Exception:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ChallengeStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        initialized = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chain_cursor'"
        ).fetchone()
        if initialized is not None:
            return

        self._connection.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE IF NOT EXISTS chain_cursor (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                block_height INTEGER NOT NULL CHECK (block_height >= 0),
                block_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS submissions (
                submission_id TEXT PRIMARY KEY,
                hotkey TEXT NOT NULL,
                netuid INTEGER NOT NULL,
                block_hash TEXT NOT NULL,
                block_height INTEGER NOT NULL,
                extrinsic_index INTEGER NOT NULL,
                event_index INTEGER NOT NULL,
                payload BLOB NOT NULL,
                repo_id TEXT,
                revision TEXT,
                admitted INTEGER NOT NULL CHECK (admitted IN (0, 1)),
                created_at_ms INTEGER NOT NULL,
                UNIQUE (block_height, extrinsic_index, event_index)
            );

            CREATE TABLE IF NOT EXISTS challenges (
                challenge_id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE REFERENCES submissions(submission_id),
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                total_attempts INTEGER NOT NULL DEFAULT 0 CHECK (total_attempts >= 0),
                error_code TEXT,
                error_summary TEXT,
                final_score REAL,
                attention_cost REAL,
                score_block INTEGER,
                provisional INTEGER NOT NULL DEFAULT 0
                    CHECK (provisional IN (0, 1)),
                lease_token TEXT,
                lease_expires_at_ms INTEGER,
                resume_state TEXT,
                retry_after_ms INTEGER,
                model_path TEXT,
                file_sha256 TEXT,
                model_digest TEXT,
                model_size_bytes INTEGER,
                track_spec_digest TEXT,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS challenge_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id TEXT NOT NULL REFERENCES challenges(challenge_id),
                sequence INTEGER NOT NULL CHECK (sequence >= 0),
                previous_state TEXT,
                state TEXT NOT NULL,
                occurred_at_ms INTEGER NOT NULL,
                error_code TEXT,
                error_summary TEXT,
                details_json TEXT NOT NULL,
                UNIQUE (challenge_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS stage_scores (
                challenge_id TEXT NOT NULL REFERENCES challenges(challenge_id),
                stage TEXT NOT NULL,
                next_state TEXT NOT NULL,
                score REAL NOT NULL,
                attention_cost REAL NOT NULL,
                score_block INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                PRIMARY KEY (challenge_id, stage)
            );

            CREATE TABLE IF NOT EXISTS leaderboard (
                hotkey TEXT PRIMARY KEY,
                challenge_id TEXT NOT NULL UNIQUE REFERENCES challenges(challenge_id),
                submission_id TEXT NOT NULL REFERENCES submissions(submission_id),
                block_height INTEGER NOT NULL,
                extrinsic_index INTEGER NOT NULL,
                event_index INTEGER NOT NULL,
                final_score REAL NOT NULL,
                attention_cost REAL,
                score_block INTEGER NOT NULL,
                provisional INTEGER NOT NULL
                    CHECK (provisional IN (0, 1)),
                updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_owners (
                model_digest TEXT PRIMARY KEY,
                challenge_id TEXT NOT NULL UNIQUE REFERENCES challenges(challenge_id)
            );

            CREATE INDEX IF NOT EXISTS submissions_chain_order
                ON submissions (block_height, extrinsic_index, event_index);
            CREATE INDEX IF NOT EXISTS challenges_state_order
                ON challenges (state, created_at_ms);
            COMMIT;
            """
        )

    def bind_track_spec(self, digest: str) -> None:
        """Bind all future challenges in this database to one frozen spec."""

        if len(digest) != 64:
            raise ValueError("track specification digest must contain 32 bytes")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError("track specification digest must be hexadecimal") from exc
        normalized = digest.lower()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'track_spec_digest'"
            ).fetchone()
            if row is not None and row["value"] != normalized:
                raise RuntimeError("database is already bound to a different track specification")
            if row is None:
                self._connection.execute(
                    "INSERT INTO store_metadata (key, value) VALUES ('track_spec_digest', ?)",
                    (normalized,),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def track_spec_digest(self) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM store_metadata WHERE key = 'track_spec_digest'"
        ).fetchone()
        return None if row is None else row["value"]

    def chain_cursor(self) -> tuple[int, str] | None:
        row = self._connection.execute(
            "SELECT block_height, block_hash FROM chain_cursor WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return None
        return row["block_height"], row["block_hash"]

    def advance_chain_cursor(self, block_height: int, block_hash: str) -> bool:
        """Persist the next scanned finalized block, enforcing continuity."""

        if block_height < 0:
            raise ValueError("block height must be non-negative")
        block_hash = _block_hash(block_hash)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self.chain_cursor()
            if current is not None:
                current_height, current_hash = current
                if block_height == current_height and block_hash == current_hash:
                    self._connection.commit()
                    return False
                if block_height != current_height + 1:
                    raise ValueError("chain cursor must advance by exactly one block")
            self._connection.execute(
                """
                INSERT INTO chain_cursor (singleton, block_height, block_hash)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    block_height = excluded.block_height,
                    block_hash = excluded.block_hash
                """,
                (block_height, block_hash),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return True

    def add_submission(
        self,
        submission: SubmissionRecord,
        *,
        cooldown_blocks: int = 0,
        occurred_at_ms: int | None = None,
    ) -> tuple[str, bool]:
        """Create a queued challenge, returning ``(challenge_id, inserted)``.

        Replaying the identical chain record is idempotent. Reusing an identifier
        for different content is treated as database corruption.
        """

        if cooldown_blocks < 0:
            raise ValueError("cooldown blocks must be non-negative")
        timestamp = _timestamp(occurred_at_ms)
        identifier = challenge_id(submission.submission_id)
        values = (
            submission.submission_id,
            submission.hotkey,
            submission.netuid,
            submission.block_hash,
            submission.position.block,
            submission.position.extrinsic_index,
            submission.position.event_index,
            submission.payload,
            submission.repo_id,
            submission.revision,
        )

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._connection.execute(
                """
                SELECT submission_id, hotkey, netuid, block_hash, block_height,
                       extrinsic_index, event_index, payload, repo_id, revision
                FROM submissions WHERE submission_id = ?
                """,
                (submission.submission_id,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != values:
                    raise RuntimeError("submission_id collision with different chain content")
                self._connection.commit()
                return identifier, False

            latest = self._connection.execute(
                """
                SELECT block_height, extrinsic_index, event_index
                FROM submissions
                ORDER BY block_height DESC, extrinsic_index DESC, event_index DESC
                LIMIT 1
                """
            ).fetchone()
            if latest is not None and (
                latest["block_height"],
                latest["extrinsic_index"],
                latest["event_index"],
            ) > (
                submission.position.block,
                submission.position.extrinsic_index,
                submission.position.event_index,
            ):
                raise RuntimeError("submissions must be inserted in finalized chain order")

            malformed = submission.repo_id is None or submission.revision is None
            last_accepted = self._connection.execute(
                """
                SELECT block_height FROM submissions
                WHERE hotkey = ? AND admitted = 1
                ORDER BY block_height DESC, extrinsic_index DESC, event_index DESC
                LIMIT 1
                """,
                (submission.hotkey,),
            ).fetchone()
            admitted = not malformed and passes_cooldown(
                current_block=submission.position.block,
                last_accepted_block=(
                    None if last_accepted is None else last_accepted["block_height"]
                ),
                cooldown_blocks=cooldown_blocks,
            )

            self._connection.execute(
                """
                INSERT INTO submissions (
                    submission_id, hotkey, netuid, block_hash, block_height,
                    extrinsic_index, event_index, payload, repo_id, revision, admitted,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, int(admitted), timestamp),
            )
            self._connection.execute(
                """
                INSERT INTO challenges (
                    challenge_id, submission_id, state, track_spec_digest,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    submission.submission_id,
                    ChallengeState.QUEUED,
                    self.track_spec_digest(),
                    timestamp,
                    timestamp,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO challenge_events (
                    challenge_id, sequence, previous_state, state, occurred_at_ms,
                    error_code, error_summary, details_json
                ) VALUES (?, 0, NULL, ?, ?, NULL, NULL, '{}')
                """,
                (identifier, ChallengeState.QUEUED, timestamp),
            )
            terminal = (
                ChallengeState.MALFORMED_SUBMISSION
                if malformed
                else ChallengeState.REJECTED_RATE_LIMIT
                if not admitted
                else None
            )
            if terminal is not None:
                self._connection.execute(
                    """
                    INSERT INTO challenge_events (
                        challenge_id, sequence, previous_state, state, occurred_at_ms,
                        error_code, error_summary, details_json
                    ) VALUES (?, 1, ?, ?, ?, ?, NULL, '{}')
                    """,
                    (identifier, ChallengeState.QUEUED, terminal, timestamp, terminal),
                )
                self._connection.execute(
                    """
                    UPDATE challenges SET state = ?, error_code = ?, updated_at_ms = ?
                    WHERE challenge_id = ?
                    """,
                    (terminal, terminal, timestamp, identifier),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return identifier, True

    def claim_next_admission(
        self,
        worker_id: str,
        *,
        lease_ms: int,
        ready_through_block: int,
        now_ms: int | None = None,
    ) -> LeasedChallenge | None:
        """Claim the oldest mature download/validation task without bypassing FIFO."""

        return self._claim_next(
            worker_id,
            stages=(ChallengeState.FETCHING, ChallengeState.VALIDATING),
            include_queued=True,
            strict_fifo=True,
            lease_ms=lease_ms,
            now_ms=now_ms,
            ready_through_block=ready_through_block,
        )

    def claim_next_scoring(
        self,
        worker_id: str,
        *,
        lease_ms: int,
        now_ms: int | None = None,
    ) -> LeasedChallenge | None:
        """Claim one ready scoring task while allowing other tasks to run in parallel."""

        return self._claim_next(
            worker_id,
            stages=(
                ChallengeState.SCREENING,
                ChallengeState.SCORING,
                ChallengeState.ROLLOUT,
            ),
            include_queued=False,
            strict_fifo=False,
            lease_ms=lease_ms,
            now_ms=now_ms,
            ready_through_block=None,
        )

    def earlier_reference_owner(self, identifier: str) -> str | None:
        """Return the earliest still-valid challenge for the same immutable HF reference."""

        row = self._connection.execute(
            """
            SELECT earlier_challenge.challenge_id
            FROM challenges current_challenge
            JOIN submissions current_submission
              ON current_submission.submission_id = current_challenge.submission_id
            JOIN submissions earlier_submission
              ON earlier_submission.repo_id = current_submission.repo_id
             AND earlier_submission.revision = current_submission.revision
             AND earlier_submission.admitted = 1
             AND (
                    earlier_submission.block_height < current_submission.block_height
                 OR (
                        earlier_submission.block_height = current_submission.block_height
                    AND earlier_submission.extrinsic_index
                        < current_submission.extrinsic_index
                 )
                 OR (
                        earlier_submission.block_height = current_submission.block_height
                    AND earlier_submission.extrinsic_index
                        = current_submission.extrinsic_index
                    AND earlier_submission.event_index < current_submission.event_index
                 )
             )
            JOIN challenges earlier_challenge
              ON earlier_challenge.submission_id = earlier_submission.submission_id
            WHERE current_challenge.challenge_id = ?
              AND earlier_challenge.state NOT IN (?, ?)
            ORDER BY earlier_submission.block_height,
                     earlier_submission.extrinsic_index,
                     earlier_submission.event_index
            LIMIT 1
            """,
            (
                identifier,
                ChallengeState.PARTICIPANT_FAILED,
                ChallengeState.SYSTEM_FAILED,
            ),
        ).fetchone()
        return None if row is None else row["challenge_id"]

    def advance_leased(
        self,
        identifier: str,
        lease_token: str,
        target: ChallengeState,
        *,
        release: bool,
        occurred_at_ms: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Advance a task only if the caller still owns its lease."""

        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_lease_locked(identifier, lease_token)
            self._transition_locked(
                identifier,
                target,
                timestamp=timestamp,
                error_code=None,
                error_summary=None,
                details=details,
            )
            if release:
                self._clear_lease_locked(identifier)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def record_downloaded_model(
        self,
        identifier: str,
        lease_token: str,
        *,
        model_path: str | Path,
        file_sha256: str,
        size_bytes: int,
        occurred_at_ms: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a complete cached download and enter CPU validation atomically."""

        if len(file_sha256) != 64 or size_bytes < 0:
            raise ValueError("download metadata is invalid")
        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current is not ChallengeState.FETCHING:
                raise ValueError("challenge is not fetching")
            self._connection.execute(
                """
                UPDATE challenges
                SET model_path = ?, file_sha256 = ?, model_size_bytes = ?
                WHERE challenge_id = ?
                """,
                (str(model_path), file_sha256, size_bytes, identifier),
            )
            self._transition_locked(
                identifier,
                ChallengeState.VALIDATING,
                timestamp=timestamp,
                error_code=None,
                error_summary=None,
                details=details,
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def register_validated_model(
        self,
        identifier: str,
        lease_token: str,
        *,
        model_digest: str,
        occurred_at_ms: int | None = None,
    ) -> ModelRegistration:
        """Assign content ownership in FIFO order and release accepted work to scoring."""

        if len(model_digest) != 64:
            raise ValueError("model digest must contain 32 bytes")
        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current is not ChallengeState.VALIDATING:
                raise ValueError("challenge is not validating")
            owner = self._connection.execute(
                """
                SELECT o.challenge_id, c.state
                FROM model_owners o
                JOIN challenges c ON c.challenge_id = o.challenge_id
                WHERE o.model_digest = ?
                """,
                (model_digest,),
            ).fetchone()
            self._connection.execute(
                "UPDATE challenges SET model_digest = ? WHERE challenge_id = ?",
                (model_digest, identifier),
            )
            if (
                owner is not None
                and ChallengeState(owner["state"]) is not ChallengeState.SYSTEM_FAILED
            ):
                self._connection.commit()
                return ModelRegistration(False, owner["challenge_id"])

            if owner is None:
                self._connection.execute(
                    "INSERT INTO model_owners (model_digest, challenge_id) VALUES (?, ?)",
                    (model_digest, identifier),
                )
            else:
                self._connection.execute(
                    """
                    UPDATE model_owners SET challenge_id = ?
                    WHERE model_digest = ? AND challenge_id = ?
                    """,
                    (identifier, model_digest, owner["challenge_id"]),
                )
            self._transition_locked(
                identifier,
                ChallengeState.SCREENING,
                timestamp=timestamp,
                error_code=None,
                error_summary=None,
                details={"model_digest": model_digest},
            )
            self._clear_lease_locked(identifier)
            self._connection.commit()
            return ModelRegistration(True, None)
        except Exception:
            self._connection.rollback()
            raise

    def defer_leased(
        self,
        identifier: str,
        lease_token: str,
        *,
        retry_after_ms: int,
        error_code: str,
        error_summary: str,
        resume_state: ChallengeState | None = None,
        occurred_at_ms: int | None = None,
    ) -> None:
        """Release a temporary failure for retry at the stage where it occurred."""

        timestamp = _timestamp(occurred_at_ms)
        if retry_after_ms <= timestamp:
            raise ValueError("retry time must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            retryable = {
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
                ChallengeState.SCREENING,
                ChallengeState.SCORING,
                ChallengeState.ROLLOUT,
            }
            if current not in retryable:
                raise ValueError("challenge is not in a retryable stage")
            resume = current if resume_state is None else resume_state
            if resume not in retryable:
                raise ValueError("resume state is not retryable")
            self._transition_locked(
                identifier,
                ChallengeState.SYSTEM_DEFERRED,
                timestamp=timestamp,
                error_code=error_code,
                error_summary=error_summary,
                details={"resume_state": resume, "retry_after_ms": retry_after_ms},
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET lease_token = NULL, lease_expires_at_ms = NULL,
                    resume_state = ?, retry_after_ms = ?
                WHERE challenge_id = ?
                """,
                (resume, retry_after_ms, identifier),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def fail_leased_system(
        self,
        identifier: str,
        lease_token: str,
        *,
        error_code: str,
        error_summary: str,
        occurred_at_ms: int | None = None,
    ) -> None:
        """End repeatedly failing infrastructure work without changing ranking."""

        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current not in {
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
                ChallengeState.SCREENING,
                ChallengeState.SCORING,
                ChallengeState.ROLLOUT,
            }:
                raise ValueError("challenge is not in an active stage")
            self._transition_locked(
                identifier,
                ChallengeState.SYSTEM_FAILED,
                timestamp=timestamp,
                error_code=error_code,
                error_summary=error_summary,
                details={},
            )
            self._clear_lease_locked(identifier)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def transition(
        self,
        identifier: str,
        target: ChallengeState,
        *,
        occurred_at_ms: int | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one event and update its current-state projection atomically."""

        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._transition_locked(
                identifier,
                target,
                timestamp=timestamp,
                error_code=error_code,
                error_summary=error_summary,
                details=details,
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def get_challenge(self, identifier: str) -> ChallengeRecord:
        row = self._connection.execute(
            "SELECT * FROM challenges WHERE challenge_id = ?", (identifier,)
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        return ChallengeRecord(
            challenge_id=row["challenge_id"],
            submission_id=row["submission_id"],
            state=ChallengeState(row["state"]),
            attempts=row["total_attempts"],
            error_code=row["error_code"],
            error_summary=row["error_summary"],
            final_score=row["final_score"],
            attention_cost=row["attention_cost"],
            score_block=row["score_block"],
            provisional=bool(row["provisional"]),
            created_at_ms=row["created_at_ms"],
            updated_at_ms=row["updated_at_ms"],
        )

    def challenge_events(self, identifier: str) -> list[ChallengeEvent]:
        rows = self._connection.execute(
            """
            SELECT sequence, previous_state, state, occurred_at_ms,
                   error_code, error_summary, details_json
            FROM challenge_events WHERE challenge_id = ? ORDER BY sequence
            """,
            (identifier,),
        ).fetchall()
        return [
            ChallengeEvent(
                sequence=row["sequence"],
                previous_state=(
                    ChallengeState(row["previous_state"])
                    if row["previous_state"] is not None
                    else None
                ),
                state=ChallengeState(row["state"]),
                occurred_at_ms=row["occurred_at_ms"],
                error_code=row["error_code"],
                error_summary=row["error_summary"],
                details=json.loads(row["details_json"]),
            )
            for row in rows
        ]

    def record_stage_score(
        self,
        identifier: str,
        lease_token: str,
        stage: ChallengeState,
        *,
        next_state: ChallengeState,
        score: float,
        attention_cost: float,
        score_block: int,
        metrics: Mapping[str, Any],
        occurred_at_ms: int | None = None,
    ) -> bool:
        """Persist one aggregate stage result; exact replay is idempotent."""

        if stage not in {
            ChallengeState.SCREENING,
            ChallengeState.SCORING,
            ChallengeState.ROLLOUT,
        }:
            raise ValueError("stage does not produce a score")
        require_transition(stage, next_state)
        if (
            not isfinite(score)
            or not isfinite(attention_cost)
            or not 0 <= score <= 1
            or attention_cost < 0
        ):
            raise ValueError("stage score or attention cost is out of range")
        if type(score_block) is not int or score_block < 0:
            raise ValueError("score block must be a non-negative integer")
        timestamp = _timestamp(occurred_at_ms)
        metrics_json = _json(metrics)
        values = (
            identifier,
            stage,
            next_state,
            score,
            attention_cost,
            score_block,
            metrics_json,
        )

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current is not stage:
                raise ValueError("challenge is not in the scored stage")
            submission = self._connection.execute(
                """
                SELECT s.block_height
                FROM challenges c
                JOIN submissions s ON s.submission_id = c.submission_id
                WHERE c.challenge_id = ?
                """,
                (identifier,),
            ).fetchone()
            if submission is None:
                raise KeyError(identifier)
            if score_block < submission["block_height"]:
                raise ValueError("score block cannot precede the submission block")
            existing = self._connection.execute(
                """
                SELECT challenge_id, stage, next_state, score, attention_cost,
                       score_block, metrics_json
                FROM stage_scores WHERE challenge_id = ? AND stage = ?
                """,
                (identifier, stage),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != values:
                    raise RuntimeError("stage score replay contains different data")
                self._connection.commit()
                return False
            self._connection.execute(
                """
                INSERT INTO stage_scores (
                    challenge_id, stage, next_state, score, attention_cost,
                    score_block, metrics_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*values, timestamp),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return True

    def stage_score(self, identifier: str, stage: ChallengeState) -> RecordedStageScore | None:
        """Return a durable stage result left before its state transition."""

        row = self._connection.execute(
            """
            SELECT stage, next_state, score, attention_cost, score_block, metrics_json
            FROM stage_scores WHERE challenge_id = ? AND stage = ?
            """,
            (identifier, stage),
        ).fetchone()
        if row is None:
            return None
        recorded_stage = ChallengeState(row["stage"])
        next_state = ChallengeState(row["next_state"])
        require_transition(recorded_stage, next_state)
        return RecordedStageScore(
            stage=recorded_stage,
            next_state=next_state,
            score=row["score"],
            attention_cost=row["attention_cost"],
            score_block=row["score_block"],
            metrics=json.loads(row["metrics_json"]),
        )

    def complete_challenge(
        self,
        identifier: str,
        target: ChallengeState,
        *,
        final_score: float | None = None,
        attention_cost: float | None = None,
        occurred_at_ms: int | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
        lease_token: str | None = None,
        score_block: int | None = None,
    ) -> None:
        """Commit a scored terminal state and its leaderboard projection atomically."""

        if target not in {
            ChallengeState.SUCCEEDED,
            ChallengeState.SCREENED_OUT,
            ChallengeState.PARTICIPANT_FAILED,
        }:
            raise ValueError("target is not a scored terminal state")
        if target is ChallengeState.SUCCEEDED:
            if final_score is None or attention_cost is None:
                raise ValueError("successful challenge requires score and attention cost")
            if (
                not isfinite(final_score)
                or not isfinite(attention_cost)
                or not 0 <= final_score <= 1
                or attention_cost < 0
            ):
                raise ValueError("final score or attention cost is out of range")
        else:
            final_score = 0.0
            attention_cost = None

        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT c.state, s.submission_id, s.hotkey, s.block_height,
                       s.extrinsic_index, s.event_index
                FROM challenges c
                JOIN submissions s ON s.submission_id = c.submission_id
                WHERE c.challenge_id = ?
                """,
                (identifier,),
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            if score_block is not None and type(score_block) is not int:
                raise ValueError("score block must be an integer")
            effective_score_block = row["block_height"] if score_block is None else score_block
            if effective_score_block < row["block_height"]:
                raise ValueError("score block cannot precede the submission block")
            if lease_token is not None:
                self._require_lease_locked(identifier, lease_token)
            self._transition_locked(
                identifier,
                target,
                timestamp=timestamp,
                error_code=error_code,
                error_summary=error_summary,
                details={"final_score": final_score, "attention_cost": attention_cost},
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET final_score = ?, attention_cost = ?, score_block = ?, provisional = ?
                WHERE challenge_id = ?
                """,
                (final_score, attention_cost, effective_score_block, 0, identifier),
            )

            current = self._connection.execute(
                """
                SELECT challenge_id, block_height, extrinsic_index, event_index
                FROM leaderboard WHERE hotkey = ?
                """,
                (row["hotkey"],),
            ).fetchone()
            position = (row["block_height"], row["extrinsic_index"], row["event_index"])
            current_position = (
                (current["block_height"], current["extrinsic_index"], current["event_index"])
                if current is not None
                else None
            )
            if (
                current_position is None
                or current["challenge_id"] == identifier
                or position > current_position
            ):
                self._update_leaderboard_locked(
                    identifier,
                    final_score=final_score,
                    attention_cost=attention_cost,
                    score_block=effective_score_block,
                    provisional=False,
                    timestamp=timestamp,
                )
            self._clear_lease_locked(identifier)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def publish_provisional(
        self,
        identifier: str,
        lease_token: str,
        *,
        final_score: float,
        attention_cost: float,
        score_block: int,
        schedule_rollout: bool = False,
        occurred_at_ms: int | None = None,
    ) -> None:
        """Publish a local score that remains capped until a rollout completes."""

        if (
            not isfinite(final_score)
            or not isfinite(attention_cost)
            or not 0 <= final_score <= 1
            or attention_cost < 0
        ):
            raise ValueError("provisional score or attention cost is out of range")
        if type(score_block) is not int or score_block < 0:
            raise ValueError("score block must be a non-negative integer")
        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current is not ChallengeState.SCORING:
                raise ValueError("only a local scoring task can publish a provisional result")
            row = self._connection.execute(
                """
                SELECT s.hotkey, s.block_height, s.extrinsic_index, s.event_index
                FROM challenges c
                JOIN submissions s ON s.submission_id = c.submission_id
                WHERE c.challenge_id = ?
                """,
                (identifier,),
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            if score_block < row["block_height"]:
                raise ValueError("score block cannot precede the submission block")
            target = ChallengeState.ROLLOUT if schedule_rollout else ChallengeState.PROVISIONAL
            self._transition_locked(
                identifier,
                target,
                timestamp=timestamp,
                error_code=None,
                error_summary=None,
                details={"final_score": final_score, "attention_cost": attention_cost},
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET final_score = ?, attention_cost = ?, score_block = ?, provisional = 1
                WHERE challenge_id = ?
                """,
                (final_score, attention_cost, score_block, identifier),
            )
            current_entry = self._connection.execute(
                """
                SELECT challenge_id, block_height, extrinsic_index, event_index,
                       provisional
                FROM leaderboard WHERE hotkey = ?
                """,
                (row["hotkey"],),
            ).fetchone()
            position = (row["block_height"], row["extrinsic_index"], row["event_index"])
            current_position = (
                (
                    current_entry["block_height"],
                    current_entry["extrinsic_index"],
                    current_entry["event_index"],
                )
                if current_entry is not None
                else None
            )
            if (
                current_position is None
                or current_entry["challenge_id"] == identifier
                or (position > current_position and bool(current_entry["provisional"]))
            ):
                self._update_leaderboard_locked(
                    identifier,
                    final_score=final_score,
                    attention_cost=attention_cost,
                    score_block=score_block,
                    provisional=True,
                    timestamp=timestamp,
                )
            self._clear_lease_locked(identifier)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def schedule_qualified_rollouts(
        self,
        minimum_score: float,
        *,
        occurred_at_ms: int | None = None,
    ) -> int:
        """Queue every qualifying provisional result for an available rollout."""

        if not isfinite(minimum_score) or not 0 <= minimum_score <= 1:
            raise ValueError("rollout threshold must be finite and within [0, 1]")
        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                """
                SELECT c.challenge_id
                FROM challenges c
                JOIN stage_scores score
                  ON score.challenge_id = c.challenge_id
                 AND score.stage = ?
                JOIN submissions submission
                  ON submission.submission_id = c.submission_id
                WHERE c.state = ? AND score.score >= ?
                ORDER BY submission.block_height,
                         submission.extrinsic_index,
                         submission.event_index
                """,
                (
                    ChallengeState.SCORING,
                    ChallengeState.PROVISIONAL,
                    minimum_score,
                ),
            ).fetchall()
            for row in rows:
                self._transition_locked(
                    row["challenge_id"],
                    ChallengeState.ROLLOUT,
                    timestamp=timestamp,
                    error_code=None,
                    error_summary=None,
                    details={"scheduled": True},
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return len(rows)

    def leaderboard(self) -> list[LeaderboardEntry]:
        rows = self._connection.execute(
            """
            SELECT hotkey, challenge_id, final_score, attention_cost,
                   score_block, provisional, block_height, extrinsic_index, event_index
            FROM leaderboard
            ORDER BY final_score DESC,
                     CASE WHEN attention_cost IS NULL THEN 1 ELSE 0 END,
                     attention_cost ASC,
                     block_height ASC, extrinsic_index ASC, event_index ASC
            """
        ).fetchall()
        return [
            LeaderboardEntry(
                hotkey=row["hotkey"],
                challenge_id=row["challenge_id"],
                final_score=row["final_score"],
                attention_cost=row["attention_cost"],
                score_block=row["score_block"],
                provisional=bool(row["provisional"]),
                position=ChainPosition(
                    row["block_height"], row["extrinsic_index"], row["event_index"]
                ),
            )
            for row in rows
        ]

    def reward_entries(self) -> list[LeaderboardEntry]:
        """Return chronological successful improvements for currently valid hotkeys."""

        rows = self._connection.execute(
            """
            SELECT submission.hotkey, challenge.challenge_id,
                   challenge.final_score, challenge.attention_cost,
                   challenge.score_block,
                   submission.block_height, submission.extrinsic_index,
                   submission.event_index
            FROM challenges challenge
            JOIN submissions submission
              ON submission.submission_id = challenge.submission_id
            JOIN leaderboard current_entry
              ON current_entry.hotkey = submission.hotkey
            JOIN challenges current_challenge
              ON current_challenge.challenge_id = current_entry.challenge_id
            WHERE challenge.state = ?
              AND current_challenge.state = ?
              AND current_entry.provisional = 0
              AND (
                    submission.block_height < current_entry.block_height
                 OR (
                        submission.block_height = current_entry.block_height
                    AND submission.extrinsic_index < current_entry.extrinsic_index
                 )
                 OR (
                        submission.block_height = current_entry.block_height
                    AND submission.extrinsic_index = current_entry.extrinsic_index
                    AND submission.event_index <= current_entry.event_index
                 )
              )
            ORDER BY submission.block_height,
                     submission.extrinsic_index,
                     submission.event_index
            """,
            (ChallengeState.SUCCEEDED, ChallengeState.SUCCEEDED),
        ).fetchall()
        return [
            LeaderboardEntry(
                hotkey=row["hotkey"],
                challenge_id=row["challenge_id"],
                final_score=row["final_score"],
                attention_cost=row["attention_cost"],
                score_block=row["score_block"],
                provisional=False,
                position=ChainPosition(
                    row["block_height"], row["extrinsic_index"], row["event_index"]
                ),
            )
            for row in rows
        ]

    def public_snapshot(self) -> dict[str, Any]:
        """Return the fixed public allowlist from one SQLite read transaction."""

        snapshot, _ = self.public_snapshot_inputs()
        return snapshot

    def public_snapshot_inputs(self) -> tuple[dict[str, Any], list[LeaderboardEntry]]:
        """Return one public snapshot and its exact weight-planning inputs."""

        self._connection.execute("BEGIN")
        try:
            leaderboard_entries = self.leaderboard()
            reward_entries = self.reward_entries()
            leaderboard = [
                {"rank": rank, **_public_entry(entry)}
                for rank, entry in enumerate(leaderboard_entries, start=1)
            ]
            rewards = [_public_entry(entry) for entry in reward_entries]
            challenge_rows = self._connection.execute(
                """
                SELECT c.challenge_id, c.submission_id, c.state,
                       c.total_attempts AS attempts,
                       c.error_code, c.error_summary, c.final_score, c.attention_cost,
                       c.score_block,
                       c.file_sha256, c.model_digest, c.model_size_bytes,
                       c.track_spec_digest, c.provisional,
                       c.created_at_ms, c.updated_at_ms,
                       s.hotkey, s.repo_id, s.revision, s.block_height,
                       s.extrinsic_index, s.event_index
                FROM challenges c
                JOIN submissions s ON s.submission_id = c.submission_id
                ORDER BY s.block_height, s.extrinsic_index, s.event_index
                """
            ).fetchall()
            challenges = [self._public_challenge_row(row) for row in challenge_rows]
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return {
            "leaderboard": leaderboard,
            "rewards": rewards,
            "challenges": challenges,
        }, reward_entries

    def public_challenge(self, identifier: str) -> dict[str, Any]:
        """Return one public challenge record and its state history."""

        self._connection.execute("BEGIN")
        try:
            row = self._connection.execute(
                """
                SELECT c.challenge_id, c.submission_id, c.state,
                       c.total_attempts AS attempts,
                       c.error_code, c.error_summary, c.final_score, c.attention_cost,
                       c.score_block,
                       c.file_sha256, c.model_digest, c.model_size_bytes,
                       c.track_spec_digest, c.provisional,
                       c.created_at_ms, c.updated_at_ms,
                       s.hotkey, s.repo_id, s.revision, s.block_height,
                       s.extrinsic_index, s.event_index
                FROM challenges c
                JOIN submissions s ON s.submission_id = c.submission_id
                WHERE c.challenge_id = ?
                """,
                (identifier,),
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            result = self._public_challenge_row(row)
            self._connection.commit()
            return result
        except Exception:
            self._connection.rollback()
            raise

    def _public_challenge_row(self, row: sqlite3.Row) -> dict[str, Any]:
        events = self._connection.execute(
            """
            SELECT sequence, previous_state, state, occurred_at_ms,
                   error_code, error_summary
            FROM challenge_events WHERE challenge_id = ? ORDER BY sequence
            """,
            (row["challenge_id"],),
        ).fetchall()
        stage_rows = self._connection.execute(
            """
            SELECT stage, next_state, score, attention_cost, score_block, created_at_ms
            FROM stage_scores WHERE challenge_id = ?
            ORDER BY CASE stage
                         WHEN 'SCREENING' THEN 0
                         WHEN 'SCORING' THEN 1
                         WHEN 'ROLLOUT' THEN 2
                         ELSE 3
                     END
            """,
            (row["challenge_id"],),
        ).fetchall()
        return {
            "challenge_id": row["challenge_id"],
            "submission_id": row["submission_id"],
            "hotkey": row["hotkey"],
            "repo_id": row["repo_id"],
            "revision": row["revision"],
            "block_height": row["block_height"],
            "extrinsic_index": row["extrinsic_index"],
            "event_index": row["event_index"],
            "state": row["state"],
            "attempts": row["attempts"],
            "error_code": row["error_code"],
            "error_summary": row["error_summary"],
            "file_sha256": row["file_sha256"],
            "model_digest": row["model_digest"],
            "model_size_bytes": row["model_size_bytes"],
            "track_spec_digest": row["track_spec_digest"],
            "provisional": bool(row["provisional"]),
            "final_score": row["final_score"],
            "attention_cost": row["attention_cost"],
            "score_block": row["score_block"],
            "created_at_ms": row["created_at_ms"],
            "updated_at_ms": row["updated_at_ms"],
            "stages": [
                {
                    "stage": stage["stage"],
                    "next_state": stage["next_state"],
                    "score": stage["score"],
                    "attention_cost": stage["attention_cost"],
                    "score_block": stage["score_block"],
                    "created_at_ms": stage["created_at_ms"],
                }
                for stage in stage_rows
            ],
            "events": [
                {
                    "sequence": event["sequence"],
                    "previous_state": event["previous_state"],
                    "state": event["state"],
                    "occurred_at_ms": event["occurred_at_ms"],
                    "error_code": event["error_code"],
                    "error_summary": event["error_summary"],
                }
                for event in events
            ],
        }

    def _transition_locked(
        self,
        identifier: str,
        target: ChallengeState,
        *,
        timestamp: int,
        error_code: str | None,
        error_summary: str | None,
        details: Mapping[str, Any] | None,
    ) -> None:
        row = self._connection.execute(
            "SELECT state FROM challenges WHERE challenge_id = ?", (identifier,)
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        current = ChallengeState(row["state"])
        require_transition(current, target)
        sequence = self._connection.execute(
            """
            SELECT COALESCE(MAX(sequence), -1) + 1
            FROM challenge_events WHERE challenge_id = ?
            """,
            (identifier,),
        ).fetchone()[0]
        self._connection.execute(
            """
            INSERT INTO challenge_events (
                challenge_id, sequence, previous_state, state, occurred_at_ms,
                error_code, error_summary, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                sequence,
                current,
                target,
                timestamp,
                error_code,
                error_summary,
                _json(details or {}),
            ),
        )
        self._connection.execute(
            """
            UPDATE challenges
            SET state = ?, error_code = ?, error_summary = ?, updated_at_ms = ?
            WHERE challenge_id = ?
            """,
            (target, error_code, error_summary, timestamp, identifier),
        )

    def _claim_next(
        self,
        worker_id: str,
        *,
        stages: tuple[ChallengeState, ...],
        include_queued: bool,
        strict_fifo: bool,
        lease_ms: int,
        now_ms: int | None,
        ready_through_block: int | None,
    ) -> LeasedChallenge | None:
        if not worker_id:
            raise ValueError("worker id cannot be empty")
        if lease_ms <= 0:
            raise ValueError("lease duration must be positive")
        now = _timestamp(now_ms)
        direct_states = (ChallengeState.QUEUED, *stages) if include_queued else stages
        direct_placeholders = ",".join("?" for _ in direct_states)
        resume_placeholders = ",".join("?" for _ in stages)
        lease_filter = (
            "" if strict_fifo else "AND (c.lease_token IS NULL OR c.lease_expires_at_ms <= ?)"
        )
        parameters: list[Any] = [
            *direct_states,
            ChallengeState.SYSTEM_DEFERRED,
            *stages,
            now,
        ]
        block_filter = ""
        if ready_through_block is not None:
            block_filter = "AND s.block_height <= ?"
            parameters.append(ready_through_block)
        if not strict_fifo:
            parameters.append(now)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                f"""
                SELECT c.challenge_id, c.submission_id, c.state,
                       c.lease_token, c.lease_expires_at_ms,
                       c.resume_state, c.attempts, s.hotkey, s.repo_id, s.revision,
                       s.block_height, s.extrinsic_index, s.event_index,
                       c.model_path, c.file_sha256
                FROM challenges c
                JOIN submissions s ON s.submission_id = c.submission_id
                WHERE (
                    c.state IN ({direct_placeholders})
                    OR (
                        c.state = ? AND c.resume_state IN ({resume_placeholders})
                        AND c.retry_after_ms <= ?
                    )
                )
                {block_filter}
                {lease_filter}
                ORDER BY s.block_height, s.extrinsic_index, s.event_index
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                self._connection.commit()
                return None
            if strict_fifo and row["lease_token"] is not None and row["lease_expires_at_ms"] > now:
                self._connection.commit()
                return None

            current = ChallengeState(row["state"])
            if current is ChallengeState.QUEUED:
                target = ChallengeState.FETCHING
                self._transition_locked(
                    row["challenge_id"],
                    target,
                    timestamp=now,
                    error_code=None,
                    error_summary=None,
                    details=None,
                )
            elif current is ChallengeState.SYSTEM_DEFERRED:
                target = ChallengeState(row["resume_state"])
                self._transition_locked(
                    row["challenge_id"],
                    target,
                    timestamp=now,
                    error_code=None,
                    error_summary=None,
                    details={"retry": True},
                )
            else:
                target = current

            token = f"{worker_id}:{secrets.token_hex(16)}"
            expires = now + lease_ms
            self._connection.execute(
                """
                UPDATE challenges
                SET lease_token = ?, lease_expires_at_ms = ?,
                    attempts = attempts + 1, total_attempts = total_attempts + 1,
                    resume_state = NULL, retry_after_ms = NULL, updated_at_ms = ?
                WHERE challenge_id = ?
                """,
                (token, expires, now, row["challenge_id"]),
            )
            self._connection.commit()
            return LeasedChallenge(
                challenge_id=row["challenge_id"],
                submission_id=row["submission_id"],
                state=target,
                hotkey=row["hotkey"],
                repo_id=row["repo_id"],
                revision=row["revision"],
                position=ChainPosition(
                    row["block_height"], row["extrinsic_index"], row["event_index"]
                ),
                lease_token=token,
                lease_expires_at_ms=expires,
                attempts=row["attempts"] + 1,
                model_path=row["model_path"],
                file_sha256=row["file_sha256"],
            )
        except Exception:
            self._connection.rollback()
            raise

    def _require_lease_locked(self, identifier: str, lease_token: str) -> ChallengeState:
        row = self._connection.execute(
            "SELECT state, lease_token FROM challenges WHERE challenge_id = ?", (identifier,)
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        if not lease_token or row["lease_token"] != lease_token:
            raise RuntimeError("challenge lease is no longer owned by this worker")
        return ChallengeState(row["state"])

    def _clear_lease_locked(self, identifier: str) -> None:
        self._connection.execute(
            """
            UPDATE challenges
            SET lease_token = NULL, lease_expires_at_ms = NULL,
                resume_state = NULL, retry_after_ms = NULL, attempts = 0
            WHERE challenge_id = ?
            """,
            (identifier,),
        )

    def _update_leaderboard_locked(
        self,
        identifier: str,
        *,
        final_score: float,
        attention_cost: float | None,
        score_block: int,
        provisional: bool,
        timestamp: int,
    ) -> None:
        row = self._connection.execute(
            """
            SELECT s.submission_id, s.hotkey, s.block_height,
                   s.extrinsic_index, s.event_index
            FROM challenges c
            JOIN submissions s ON s.submission_id = c.submission_id
            WHERE c.challenge_id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        self._connection.execute(
            """
            INSERT INTO leaderboard (
                hotkey, challenge_id, submission_id, block_height,
                extrinsic_index, event_index, final_score, attention_cost,
                score_block, provisional, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hotkey) DO UPDATE SET
                challenge_id = excluded.challenge_id,
                submission_id = excluded.submission_id,
                block_height = excluded.block_height,
                extrinsic_index = excluded.extrinsic_index,
                event_index = excluded.event_index,
                final_score = excluded.final_score,
                attention_cost = excluded.attention_cost,
                score_block = excluded.score_block,
                provisional = excluded.provisional,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                row["hotkey"],
                identifier,
                row["submission_id"],
                row["block_height"],
                row["extrinsic_index"],
                row["event_index"],
                final_score,
                attention_cost,
                score_block,
                int(provisional),
                timestamp,
            ),
        )


def _timestamp(value: int | None) -> int:
    timestamp = time.time_ns() // 1_000_000 if value is None else value
    if timestamp < 0:
        raise ValueError("timestamp must be non-negative")
    return timestamp


def _enable_wal(connection: sqlite3.Connection) -> None:
    """Enable WAL despite SQLite's non-waiting journal-mode lock race."""

    deadline = time.monotonic() + _SQLITE_BUSY_TIMEOUT_MS / 1_000
    while True:
        try:
            row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        except sqlite3.OperationalError as exc:
            error_code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
            if error_code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
            continue
        if row is None or str(row[0]).lower() != "wal":
            raise RuntimeError("validator database could not enable WAL mode")
        return


def _block_hash(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("block hash must be a string")
    normalized = value.removeprefix("0x")
    try:
        raw = bytes.fromhex(normalized)
    except ValueError as exc:
        raise ValueError("block hash must be hexadecimal") from exc
    if len(raw) != 32:
        raise ValueError("block hash must contain 32 bytes")
    return "0x" + raw.hex()


def _public_entry(entry: LeaderboardEntry) -> dict[str, Any]:
    return {
        "hotkey": entry.hotkey,
        "challenge_id": entry.challenge_id,
        "final_score": entry.final_score,
        "attention_cost": entry.attention_cost,
        "score_block": entry.score_block,
        "provisional": entry.provisional,
        "block_height": entry.position.block,
        "extrinsic_index": entry.position.extrinsic_index,
        "event_index": entry.position.event_index,
    }


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
