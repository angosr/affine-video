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
from typing import TYPE_CHECKING, Any

from validator.queue import (
    TERMINAL_STATES,
    ChainPosition,
    ChallengeState,
    challenge_id,
    require_transition,
)
from validator.queue import (
    submission_id as derive_submission_id,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from validator.tournament import MatchRecord

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
    score_block: int | None
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
    score: float
    score_block: int
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    hotkey: str
    challenge_id: str
    final_score: float
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
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE IF NOT EXISTS chain_cursor (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                block_height INTEGER NOT NULL CHECK (block_height >= 0),
                block_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS finalized_blocks (
                block_height INTEGER PRIMARY KEY CHECK (block_height >= 0),
                block_hash TEXT NOT NULL UNIQUE
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
                score_block INTEGER,
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
                score REAL NOT NULL,
                score_block INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                PRIMARY KEY (challenge_id, stage)
            );

            CREATE TABLE IF NOT EXISTS model_owners (
                model_digest TEXT PRIMARY KEY,
                challenge_id TEXT NOT NULL UNIQUE REFERENCES challenges(challenge_id)
            );

            CREATE TABLE IF NOT EXISTS season_slots (
                sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
                challenge_id TEXT NOT NULL UNIQUE REFERENCES challenges(challenge_id)
            );

            CREATE TABLE IF NOT EXISTS commitment_choices (
                hotkey TEXT PRIMARY KEY,
                netuid INTEGER NOT NULL,
                block_hash TEXT NOT NULL,
                block_height INTEGER NOT NULL,
                extrinsic_index INTEGER NOT NULL,
                event_index INTEGER NOT NULL,
                payload BLOB NOT NULL,
                submission_id TEXT REFERENCES submissions(submission_id)
            );

            CREATE TABLE IF NOT EXISTS competition_exclusions (
                hotkey TEXT PRIMARY KEY,
                first_block INTEGER NOT NULL CHECK (first_block >= 0),
                reason TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS submissions_chain_order
                ON submissions (block_height, extrinsic_index, event_index);
            CREATE INDEX IF NOT EXISTS submissions_hotkey
                ON submissions (hotkey);
            CREATE INDEX IF NOT EXISTS challenges_state_order
                ON challenges (state, created_at_ms);

            DELETE FROM season_slots
            WHERE challenge_id IN (
                SELECT challenge_id FROM challenges WHERE state = 'PARTICIPANT_FAILED'
            ) AND challenge_id NOT IN (
                SELECT challenge_id FROM model_owners
            );
            COMMIT;
            """
        )

    def observe_commitment(
        self,
        *,
        hotkey: str,
        netuid: int,
        block_hash: str,
        position: ChainPosition,
        payload: bytes,
        submission_id: str | None,
        legacy_excluded: bool,
    ) -> None:
        """Record a hotkey's first recognized contest choice permanently."""

        if not hotkey or not payload:
            raise ValueError("commitment hotkey and payload cannot be empty")
        if not 0 <= netuid <= 0xFFFF:
            raise ValueError("netuid must fit in an unsigned 16-bit integer")
        if submission_id is None and not legacy_excluded:
            return
        normalized_hash = _block_hash(block_hash)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            selected_submission_id = None
            if submission_id is not None:
                submitted = self._connection.execute(
                    """
                    SELECT hotkey FROM submissions WHERE submission_id = ?
                    """,
                    (submission_id,),
                ).fetchone()
                if submitted is None or submitted["hotkey"] != hotkey:
                    raise ValueError("commitment submission does not belong to its hotkey")
                first = self._connection.execute(
                    """
                    SELECT submission_id
                    FROM submissions
                    WHERE hotkey = ?
                    ORDER BY block_height, extrinsic_index, event_index
                    LIMIT 1
                    """,
                    (hotkey,),
                ).fetchone()
                if first is None:
                    raise RuntimeError("VDA commitment has no recorded submission")
                selected_submission_id = first["submission_id"]
            current = self._connection.execute(
                """
                SELECT block_height, extrinsic_index, event_index, block_hash, payload,
                       submission_id
                FROM commitment_choices WHERE hotkey = ?
                """,
                (hotkey,),
            ).fetchone()
            incoming = (position.block, position.extrinsic_index, position.event_index)
            if current is not None:
                existing = (
                    current["block_height"],
                    current["extrinsic_index"],
                    current["event_index"],
                )
                if incoming < existing:
                    raise ValueError("commitment choice cannot move backward")
                if incoming == existing and (
                    current["block_hash"] != normalized_hash
                    or bytes(current["payload"]) != payload
                    or current["submission_id"] != selected_submission_id
                ):
                    raise RuntimeError("replayed commitment choice contains different data")
            else:
                self._connection.execute(
                    """
                    INSERT INTO commitment_choices (
                        hotkey, netuid, block_hash, block_height, extrinsic_index,
                        event_index, payload, submission_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hotkey,
                        netuid,
                        normalized_hash,
                        position.block,
                        position.extrinsic_index,
                        position.event_index,
                        payload,
                        selected_submission_id,
                    ),
                )
            if current is None and legacy_excluded:
                self._connection.execute(
                    """
                    INSERT INTO competition_exclusions (hotkey, first_block, reason)
                    VALUES (?, ?, 'LEGACY_CONTEST_ACTIVATED')
                    ON CONFLICT(hotkey) DO NOTHING
                    """,
                    (hotkey, position.block),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def seed_legacy_exclusions(
        self,
        exclusions: Mapping[str, int],
    ) -> None:
        """Persist old-contest choices observed before this track opened."""

        if any(
            not hotkey or type(block) is not int or block < 0
            for hotkey, block in exclusions.items()
        ):
            raise ValueError("legacy exclusion snapshot is invalid")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for hotkey, block in sorted(exclusions.items()):
                self._connection.execute(
                    """
                    INSERT INTO competition_exclusions (hotkey, first_block, reason)
                    VALUES (?, ?, 'LEGACY_CONTEST_ACTIVATED')
                    ON CONFLICT(hotkey) DO UPDATE SET
                        first_block = MIN(first_block, excluded.first_block)
                    """,
                    (hotkey, block),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def reserve_uid_zero(self, hotkey: str, *, track_start_block: int) -> None:
        """Reserve the synthetic baseline champion's chain identity."""

        if not hotkey or type(track_start_block) is not int or track_start_block < 0:
            raise ValueError("UID0 reservation is invalid")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                INSERT INTO competition_exclusions (hotkey, first_block, reason)
                VALUES (?, ?, 'RESERVED_UID0')
                ON CONFLICT(hotkey) DO UPDATE SET
                    first_block = MIN(first_block, excluded.first_block),
                    reason = 'RESERVED_UID0'
                """,
                (hotkey, track_start_block),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def active_competitors(self, uid_by_hotkey: Mapping[str, int]) -> list[dict[str, Any]]:
        """Project current VDA choices onto one finalized metagraph snapshot."""

        rows = self._connection.execute(
            """
            SELECT choice.hotkey, submission.block_height,
                   submission.extrinsic_index, submission.event_index,
                   choice.submission_id, submission.repo_id, submission.revision,
                   exclusion.reason AS exclusion_reason
            FROM commitment_choices choice
            LEFT JOIN submissions submission
              ON submission.submission_id = choice.submission_id
            LEFT JOIN competition_exclusions exclusion
              ON exclusion.hotkey = choice.hotkey
            ORDER BY choice.block_height, choice.extrinsic_index, choice.event_index
            """
        ).fetchall()
        return [
            {
                "uid": uid_by_hotkey[row["hotkey"]],
                "hotkey": row["hotkey"],
                "submission_id": row["submission_id"],
                "repo_id": row["repo_id"],
                "revision": row["revision"],
                "block_height": row["block_height"],
                "extrinsic_index": row["extrinsic_index"],
                "event_index": row["event_index"],
            }
            for row in rows
            if row["hotkey"] in uid_by_hotkey
            and row["submission_id"] is not None
            and row["repo_id"] is not None
            and row["exclusion_reason"] is None
        ]

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

    def effective_challenge_count(self) -> int:
        """Return effective challenges plus pending admission reservations."""

        return int(self._connection.execute("SELECT COUNT(*) FROM season_slots").fetchone()[0])

    def has_season_slot(self, identifier: str) -> bool:
        """Return whether a challenge reserves or permanently consumed a season slot."""

        return (
            self._connection.execute(
                "SELECT 1 FROM season_slots WHERE challenge_id = ?", (identifier,)
            ).fetchone()
            is not None
        )

    def model_owner(self, model_digest: str) -> str | None:
        """Return the challenge that already owns canonical selector weights."""

        if len(model_digest) != 64:
            raise ValueError("model digest must contain 32 bytes")
        try:
            bytes.fromhex(model_digest)
        except ValueError as exc:
            raise ValueError("model digest must be hexadecimal") from exc
        row = self._connection.execute(
            "SELECT challenge_id FROM model_owners WHERE model_digest = ?",
            (model_digest.lower(),),
        ).fetchone()
        return None if row is None else row["challenge_id"]

    def reserve_season_slot(
        self,
        identifier: str,
        lease_token: str,
        *,
        maximum_effective_challenges: int | None,
    ) -> bool:
        """Reserve FIFO capacity before I/O; invalid admission releases it atomically."""

        if maximum_effective_challenges is not None and maximum_effective_challenges <= 0:
            raise ValueError("maximum effective challenges must be positive")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current not in {ChallengeState.FETCHING, ChallengeState.VALIDATING}:
                raise ValueError("challenge is not in admission")
            existing = self._connection.execute(
                "SELECT sequence FROM season_slots WHERE challenge_id = ?",
                (identifier,),
            ).fetchone()
            if existing is not None:
                self._connection.commit()
                return True
            slot_count = int(
                self._connection.execute("SELECT COUNT(*) FROM season_slots").fetchone()[0]
            )
            if (
                maximum_effective_challenges is not None
                and slot_count >= maximum_effective_challenges
            ):
                self._connection.commit()
                return False
            next_sequence = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM season_slots"
                ).fetchone()[0]
            )
            self._connection.execute(
                "INSERT INTO season_slots (sequence, challenge_id) VALUES (?, ?)",
                (next_sequence, identifier),
            )
            self._connection.commit()
            return True
        except Exception:
            self._connection.rollback()
            raise

    def chain_cursor(self) -> tuple[int, str] | None:
        row = self._connection.execute(
            "SELECT block_height, block_hash FROM chain_cursor WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return None
        return row["block_height"], row["block_hash"]

    def admitted_challenge_ids(self, first_block: int, last_block: int) -> tuple[str, ...]:
        """Return the immutable admitted group for one finalized block interval."""

        if not 0 <= first_block <= last_block:
            raise ValueError("challenge block interval is invalid")
        rows = self._connection.execute(
            """
            SELECT c.challenge_id
            FROM season_slots slot
            JOIN challenges c ON c.challenge_id = slot.challenge_id
            JOIN model_owners owner ON owner.challenge_id = c.challenge_id
            JOIN submissions s ON s.submission_id = c.submission_id
            WHERE s.block_height BETWEEN ? AND ?
            ORDER BY s.block_height, s.extrinsic_index, s.event_index
            """,
            (first_block, last_block),
        ).fetchall()
        return tuple(row["challenge_id"] for row in rows)

    def admission_interval_complete(self, first_block: int, last_block: int) -> bool:
        """Return whether every eligible submission in an interval has left admission."""

        if not 0 <= first_block <= last_block:
            raise ValueError("challenge block interval is invalid")
        row = self._connection.execute(
            """
            SELECT 1
            FROM challenges c
            JOIN submissions s ON s.submission_id = c.submission_id
            WHERE s.admitted = 1
              AND s.block_height BETWEEN ? AND ?
              AND (
                    c.state IN (?, ?, ?)
                 OR (
                        c.state IN (?, ?)
                    AND c.resume_state IN (?, ?)
                    )
                  )
            LIMIT 1
            """,
            (
                first_block,
                last_block,
                ChallengeState.QUEUED,
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
                ChallengeState.SYSTEM_DEFERRED,
                ChallengeState.SYSTEM_BLOCKED,
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
            ),
        ).fetchone()
        return row is None

    def confirmation_challenge_ids(
        self, first_decision_block: int, last_decision_block: int
    ) -> tuple[str, ...]:
        """Return exact winners whose decision maps to one future batch interval."""

        if not 0 <= first_decision_block <= last_decision_block:
            raise ValueError("confirmation decision interval is invalid")
        rows = self._connection.execute(
            """
            SELECT c.challenge_id
            FROM challenges c
            JOIN submissions s ON s.submission_id = c.submission_id
            JOIN stage_scores score
              ON score.challenge_id = c.challenge_id AND score.stage = ?
            WHERE (
                    c.state = ?
                 OR (c.state IN (?, ?) AND c.resume_state = ?)
                  )
              AND score.score_block BETWEEN ? AND ?
            ORDER BY s.block_height, s.extrinsic_index, s.event_index
            """,
            (
                ChallengeState.SCORING,
                ChallengeState.CONFIRMING,
                ChallengeState.SYSTEM_DEFERRED,
                ChallengeState.SYSTEM_BLOCKED,
                ChallengeState.CONFIRMING,
                first_decision_block,
                last_decision_block,
            ),
        ).fetchall()
        return tuple(row["challenge_id"] for row in rows)

    def challenge_ids_in_state(self, state: ChallengeState) -> tuple[str, ...]:
        """Return challenges in canonical chain order for private batch planning."""

        rows = self._connection.execute(
            """
            SELECT c.challenge_id
            FROM challenges c
            JOIN submissions s ON s.submission_id = c.submission_id
            WHERE c.state = ?
            ORDER BY s.block_height, s.extrinsic_index, s.event_index
            """,
            (state,),
        ).fetchall()
        return tuple(row["challenge_id"] for row in rows)

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
            known = self._connection.execute(
                "SELECT block_hash FROM finalized_blocks WHERE block_height = ?",
                (block_height,),
            ).fetchone()
            if known is not None and known["block_hash"] != block_hash:
                raise RuntimeError("finalized block hash changed")
            self._connection.execute(
                """
                INSERT INTO finalized_blocks (block_height, block_hash)
                VALUES (?, ?)
                ON CONFLICT(block_height) DO NOTHING
                """,
                (block_height, block_hash),
            )
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

    def finalized_block_hash(self, block_height: int) -> str | None:
        """Return the hash observed while scanning one finalized block."""

        if type(block_height) is not int or block_height < 0:
            raise ValueError("block height must be non-negative")
        row = self._connection.execute(
            "SELECT block_hash FROM finalized_blocks WHERE block_height = ?",
            (block_height,),
        ).fetchone()
        return None if row is None else row["block_hash"]

    def add_submission(
        self,
        submission: SubmissionRecord,
        *,
        occurred_at_ms: int | None = None,
    ) -> tuple[str, bool]:
        """Record a submission, admitting only the hotkey's first one on this track.

        Replaying the identical chain record is idempotent. Reusing an identifier
        for different content is treated as database corruption.
        """

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
            exclusion = self._connection.execute(
                "SELECT reason FROM competition_exclusions WHERE hotkey = ?",
                (submission.hotkey,),
            ).fetchone()
            excluded = exclusion is not None
            previous_hotkey_submission = self._connection.execute(
                """
                SELECT submission_id FROM submissions
                WHERE hotkey = ?
                LIMIT 1
                """,
                (submission.hotkey,),
            ).fetchone()
            first_for_hotkey = previous_hotkey_submission is None
            admitted = not excluded and not malformed and first_for_hotkey

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
                ChallengeState.REJECTED_DUPLICATE_HOTKEY
                if not first_for_hotkey
                or (exclusion is not None and exclusion["reason"] == "RESERVED_UID0")
                else ChallengeState.REJECTED_OTHER_COMPETITION
                if excluded
                else ChallengeState.MALFORMED_SUBMISSION
                if malformed
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

    def reject_submissions_after(
        self,
        last_submission_block: int,
        *,
        decision_block: int,
        occurred_at_ms: int | None = None,
    ) -> int:
        """Terminalize scanned submissions that cannot belong to this track."""

        if min(last_submission_block, decision_block) < 0:
            raise ValueError("submission cutoff and decision block must be non-negative")
        if decision_block < last_submission_block:
            raise ValueError("decision block cannot precede the submission cutoff")
        timestamp = _timestamp(occurred_at_ms)
        terminal = tuple(state.value for state in TERMINAL_STATES)
        placeholders = ",".join("?" for _ in terminal)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                f"""
                SELECT challenge.challenge_id, challenge.state, submission.block_height
                FROM challenges challenge
                JOIN submissions submission
                  ON submission.submission_id = challenge.submission_id
                WHERE submission.block_height > ?
                  AND challenge.state NOT IN ({placeholders})
                ORDER BY submission.block_height, submission.extrinsic_index,
                         submission.event_index
                """,
                (last_submission_block, *terminal),
            ).fetchall()
            for row in rows:
                current = ChallengeState(row["state"])
                if current not in {ChallengeState.QUEUED, ChallengeState.FETCHING}:
                    raise RuntimeError(
                        "an ineligible late submission advanced beyond admission"
                    )
                if row["block_height"] > decision_block:
                    raise RuntimeError("late submission is newer than its decision block")
                self._transition_locked(
                    row["challenge_id"],
                    ChallengeState.REJECTED_TRACK_CLOSED,
                    timestamp=timestamp,
                    error_code="TRACK_CLOSED",
                    error_summary=(
                        "submission is later than the final frozen evaluation interval"
                    ),
                    details={"final_score": None},
                )
                self._connection.execute(
                    """
                    UPDATE challenges
                    SET final_score = NULL, score_block = ?
                    WHERE challenge_id = ?
                    """,
                    (decision_block, row["challenge_id"]),
                )
                self._clear_lease_locked(row["challenge_id"])
            self._connection.commit()
            return len(rows)
        except Exception:
            self._connection.rollback()
            raise

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
        ready_through_block: int | None = None,
        now_ms: int | None = None,
    ) -> LeasedChallenge | None:
        """Claim the oldest authoritative scoring task without bypassing FIFO."""

        if ready_through_block is not None and ready_through_block < 0:
            return None

        return self._claim_next(
            worker_id,
            stages=(ChallengeState.SCORING, ChallengeState.CONFIRMING),
            include_queued=False,
            strict_fifo=True,
            lease_ms=lease_ms,
            now_ms=now_ms,
            ready_through_block=None,
            scoring_ready_through_block=ready_through_block,
        )

    def wait_leased(
        self,
        identifier: str,
        lease_token: str,
        *,
        retry_after_ms: int,
        reason: str,
        occurred_at_ms: int | None = None,
    ) -> None:
        """Release a normal wait without consuming a failure attempt."""

        timestamp = _timestamp(occurred_at_ms)
        if retry_after_ms <= timestamp:
            raise ValueError("retry time must be in the future")
        if not reason:
            raise ValueError("wait reason cannot be empty")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current not in {
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
                ChallengeState.SCORING,
                ChallengeState.CONFIRMING,
            }:
                raise ValueError("challenge is not in a waitable stage")
            self._transition_locked(
                identifier,
                ChallengeState.SYSTEM_DEFERRED,
                timestamp=timestamp,
                error_code=None,
                error_summary=None,
                details={
                    "resume_state": current,
                    "retry_after_ms": retry_after_ms,
                    "wait_reason": reason,
                },
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET lease_token = NULL, lease_expires_at_ms = NULL,
                    resume_state = ?, retry_after_ms = ?,
                    attempts = max(attempts - 1, 0),
                    total_attempts = max(total_attempts - 1, 0)
                WHERE challenge_id = ?
                """,
                (current, retry_after_ms, identifier),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def synchronize_leased_attempts(
        self,
        identifier: str,
        lease_token: str,
        *,
        attempts: int,
    ) -> None:
        """Align a disposable local lease with its durable retry ordinal."""

        if attempts <= 0:
            raise ValueError("leased attempts must be positive")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current not in {
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
                ChallengeState.SCORING,
                ChallengeState.CONFIRMING,
            }:
                raise ValueError("challenge is not in a retryable stage")
            self._connection.execute(
                """
                UPDATE challenges
                SET attempts = ?, total_attempts = max(total_attempts, ?)
                WHERE challenge_id = ?
                """,
                (attempts, attempts, identifier),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

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
              AND earlier_challenge.state != ?
            ORDER BY earlier_submission.block_height,
                     earlier_submission.extrinsic_index,
                     earlier_submission.event_index
            LIMIT 1
            """,
            (
                identifier,
                ChallengeState.PARTICIPANT_FAILED,
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
        """Assign unique model ownership and enter scoring for a reserved challenge."""

        if len(model_digest) != 64:
            raise ValueError("model digest must contain 32 bytes")
        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current is not ChallengeState.VALIDATING:
                raise ValueError("challenge is not validating")
            slot = self._connection.execute(
                "SELECT sequence FROM season_slots WHERE challenge_id = ?",
                (identifier,),
            ).fetchone()
            if slot is None:
                raise RuntimeError("validated model has no reserved season slot")
            owner = self._connection.execute(
                """
                SELECT o.challenge_id, c.state
                FROM model_owners o
                JOIN challenges c ON c.challenge_id = o.challenge_id
                WHERE o.model_digest = ?
                """,
                (model_digest,),
            ).fetchone()
            if owner is not None:
                self._connection.execute(
                    "UPDATE challenges SET model_digest = ? WHERE challenge_id = ?",
                    (model_digest, identifier),
                )
                self._connection.commit()
                return ModelRegistration(False, owner["challenge_id"])

            self._connection.execute(
                "UPDATE challenges SET model_digest = ? WHERE challenge_id = ?",
                (model_digest, identifier),
            )

            self._connection.execute(
                "INSERT INTO model_owners (model_digest, challenge_id) VALUES (?, ?)",
                (model_digest, identifier),
            )
            self._transition_locked(
                identifier,
                ChallengeState.SCORING,
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
                ChallengeState.SCORING,
                ChallengeState.CONFIRMING,
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

    def block_leased(
        self,
        identifier: str,
        lease_token: str,
        *,
        error_code: str,
        error_summary: str,
        occurred_at_ms: int | None = None,
    ) -> None:
        """Stop automatic retries without turning infrastructure failure into a loss."""

        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._require_lease_locked(identifier, lease_token)
            if current not in {
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
                ChallengeState.SCORING,
                ChallengeState.CONFIRMING,
            }:
                raise ValueError("challenge is not in a blockable stage")
            self._transition_locked(
                identifier,
                ChallengeState.SYSTEM_BLOCKED,
                timestamp=timestamp,
                error_code=error_code,
                error_summary=error_summary,
                details={"retry_exhausted": True, "blocked_stage": current},
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET lease_token = NULL, lease_expires_at_ms = NULL,
                    resume_state = ?, retry_after_ms = NULL
                WHERE challenge_id = ?
                """,
                (current, identifier),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def resume_blocked(
        self,
        identifier: str,
        *,
        occurred_at_ms: int | None = None,
    ) -> ChallengeState:
        """Explicitly return one visible infrastructure block to its failed stage."""

        timestamp = _timestamp(occurred_at_ms)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT state, resume_state FROM challenges WHERE challenge_id = ?",
                (identifier,),
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            if ChallengeState(row["state"]) is not ChallengeState.SYSTEM_BLOCKED:
                raise ValueError("challenge is not blocked by infrastructure")
            if row["resume_state"] is None:
                raise ValueError("blocked challenge has no resumable stage")
            target = ChallengeState(row["resume_state"])
            require_transition(ChallengeState.SYSTEM_BLOCKED, target)
            self._transition_locked(
                identifier,
                target,
                timestamp=timestamp,
                error_code=None,
                error_summary=None,
                details={"operator_retry": True},
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET attempts = 0, lease_token = NULL, lease_expires_at_ms = NULL,
                    resume_state = NULL, retry_after_ms = NULL,
                    error_code = NULL, error_summary = NULL
                WHERE challenge_id = ?
                """,
                (identifier,),
            )
            self._connection.commit()
            return target
        except Exception:
            self._connection.rollback()
            raise

    def blocked_resume_state(self, identifier: str) -> ChallengeState:
        """Return the stage fenced by one visible infrastructure block."""

        row = self._connection.execute(
            "SELECT state, resume_state FROM challenges WHERE challenge_id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        if ChallengeState(row["state"]) is not ChallengeState.SYSTEM_BLOCKED:
            raise ValueError("challenge is not blocked by infrastructure")
        if row["resume_state"] is None:
            raise ValueError("blocked challenge has no resumable stage")
        stage = ChallengeState(row["resume_state"])
        if stage not in {
            ChallengeState.FETCHING,
            ChallengeState.VALIDATING,
            ChallengeState.SCORING,
            ChallengeState.CONFIRMING,
        }:
            raise ValueError("blocked challenge has an invalid resumable stage")
        return stage

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
            score_block=row["score_block"],
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
        score: float,
        score_block: int,
        metrics: Mapping[str, Any],
        occurred_at_ms: int | None = None,
    ) -> bool:
        """Persist one aggregate stage result; exact replay is idempotent."""

        if stage not in {ChallengeState.SCORING, ChallengeState.CONFIRMING}:
            raise ValueError("stage does not produce a score")
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("stage score is out of range")
        if type(score_block) is not int or score_block < 0:
            raise ValueError("score block must be a non-negative integer")
        timestamp = _timestamp(occurred_at_ms)
        metrics_json = _json(metrics)
        values = (
            identifier,
            stage,
            score,
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
                SELECT challenge_id, stage, score, score_block, metrics_json
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
                    challenge_id, stage, score,
                    score_block, metrics_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
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
            SELECT stage, score, score_block, metrics_json
            FROM stage_scores WHERE challenge_id = ? AND stage = ?
            """,
            (identifier, stage),
        ).fetchone()
        if row is None:
            return None
        recorded_stage = ChallengeState(row["stage"])
        if recorded_stage not in {ChallengeState.SCORING, ChallengeState.CONFIRMING}:
            raise RuntimeError("stored score has an invalid stage")
        return RecordedStageScore(
            stage=recorded_stage,
            score=row["score"],
            score_block=row["score_block"],
            metrics=json.loads(row["metrics_json"]),
        )

    def stage_transition_applied(self, identifier: str, stage: ChallengeState) -> bool:
        """Return whether one archived stage fact has advanced the state machine."""

        if stage not in {ChallengeState.SCORING, ChallengeState.CONFIRMING}:
            raise ValueError("stage does not produce a score")
        row = self._connection.execute(
            "SELECT state, resume_state FROM challenges WHERE challenge_id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        state = ChallengeState(row["state"])
        if state in TERMINAL_STATES:
            return True
        if stage is ChallengeState.CONFIRMING:
            return False
        if state is ChallengeState.CONFIRMING:
            return True
        return (
            state in {ChallengeState.SYSTEM_DEFERRED, ChallengeState.SYSTEM_BLOCKED}
            and row["resume_state"] == ChallengeState.CONFIRMING
        )

    def model_path(self, identifier: str) -> str | None:
        """Return a disposable local model path for a known challenge."""

        row = self._connection.execute(
            "SELECT model_path FROM challenges WHERE challenge_id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        return row["model_path"]

    def complete_challenge(
        self,
        identifier: str,
        target: ChallengeState,
        *,
        final_score: float | None = None,
        occurred_at_ms: int | None = None,
        error_code: str | None = None,
        error_summary: str | None = None,
        lease_token: str | None = None,
        score_block: int | None = None,
    ) -> None:
        """Commit a terminal state and its decision block atomically."""

        if target not in TERMINAL_STATES:
            raise ValueError("target is not terminal")
        if target is ChallengeState.SUCCEEDED:
            if final_score is None:
                raise ValueError("successful challenge requires a score")
            if error_code is not None or error_summary is not None:
                raise ValueError("successful challenge cannot contain an error")
            if not isfinite(final_score) or not 0 <= final_score <= 1:
                raise ValueError("final score is out of range")
        else:
            if not error_code:
                raise ValueError("failed challenge requires an error code")
            final_score = None

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
                details={"final_score": final_score},
            )
            self._connection.execute(
                """
                UPDATE challenges
                SET final_score = ?, score_block = ?
                WHERE challenge_id = ?
                """,
                (final_score, effective_score_block, identifier),
            )

            if target is ChallengeState.PARTICIPANT_FAILED and row["state"] in {
                ChallengeState.FETCHING,
                ChallengeState.VALIDATING,
            }:
                self._connection.execute(
                    "DELETE FROM season_slots WHERE challenge_id = ?", (identifier,)
                )
            self._clear_lease_locked(identifier)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def leaderboard(self) -> list[LeaderboardEntry]:
        rows = self._connection.execute(
            """
            SELECT submission.hotkey, challenge.challenge_id,
                   challenge.final_score, challenge.score_block,
                   submission.block_height, submission.extrinsic_index,
                   submission.event_index
            FROM challenges challenge
            JOIN submissions submission
              ON submission.submission_id = challenge.submission_id
            WHERE challenge.state = ?
            ORDER BY challenge.final_score DESC,
                     submission.block_height ASC,
                     submission.extrinsic_index ASC,
                     submission.event_index ASC
            """,
            (ChallengeState.SUCCEEDED,),
        ).fetchall()
        return [
            LeaderboardEntry(
                hotkey=row["hotkey"],
                challenge_id=row["challenge_id"],
                final_score=row["final_score"],
                score_block=row["score_block"],
                position=ChainPosition(
                    row["block_height"], row["extrinsic_index"], row["event_index"]
                ),
            )
            for row in rows
        ]

    def restore_public_matches(
        self,
        matches: Sequence[MatchRecord],
        *,
        maximum_effective_challenges: int | None = None,
    ) -> int:
        """Restore the locally available prefix of the verified public ledger."""

        if maximum_effective_challenges is not None and maximum_effective_challenges <= 0:
            raise ValueError("maximum effective challenges must be positive")
        restored = 0
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for match in matches:
                document = match.document
                submission = document["submission"]
                evaluation = document["evaluation"]
                row = self._connection.execute(
                    """
                    SELECT c.state, c.error_code, c.final_score, c.score_block,
                           c.file_sha256, c.model_digest,
                           c.model_size_bytes, c.created_at_ms,
                           s.submission_id, s.hotkey, s.repo_id, s.revision,
                           s.block_height, s.extrinsic_index, s.event_index
                    FROM challenges c
                    JOIN submissions s ON s.submission_id = c.submission_id
                    WHERE c.challenge_id = ?
                    """,
                    (match.challenge_id,),
                ).fetchone()
                if row is None:
                    break
                expected_submission = (
                    document["submission_id"],
                    submission["hotkey"],
                    submission["repo_id"],
                    submission["revision"],
                    submission["block_height"],
                    submission["extrinsic_index"],
                    submission["event_index"],
                )
                actual_submission = (
                    row["submission_id"],
                    row["hotkey"],
                    row["repo_id"],
                    row["revision"],
                    row["block_height"],
                    row["extrinsic_index"],
                    row["event_index"],
                )
                if actual_submission != expected_submission:
                    raise RuntimeError("public match conflicts with its on-chain submission")

                target = ChallengeState(evaluation["state"])
                current = ChallengeState(row["state"])
                if current in TERMINAL_STATES:
                    if current is not target or row["error_code"] != evaluation["error_code"]:
                        raise RuntimeError("public match conflicts with a local terminal challenge")
                    if row["score_block"] != evaluation["score_block"]:
                        raise RuntimeError("public decision block conflicts with local state")
                    if (
                        row["file_sha256"] != evaluation["file_sha256"]
                        or row["model_digest"] != evaluation["model_digest"]
                        or row["model_size_bytes"] != evaluation["model_size_bytes"]
                    ):
                        raise RuntimeError(
                            "public model facts conflict with a local terminal challenge"
                        )
                    if (
                        target is ChallengeState.SUCCEEDED
                        and row["final_score"] != evaluation["final_score"]
                    ):
                        raise RuntimeError("public score conflicts with a local terminal challenge")
                else:
                    self._connection.execute(
                        """
                        UPDATE challenges
                        SET state = ?, error_code = ?, error_summary = NULL,
                            final_score = ?, score_block = ?,
                            lease_token = NULL, lease_expires_at_ms = NULL, resume_state = NULL,
                            retry_after_ms = NULL, attempts = 0,
                            file_sha256 = ?, model_digest = ?, model_size_bytes = ?,
                            updated_at_ms = ?
                        WHERE challenge_id = ?
                        """,
                        (
                            target,
                            evaluation["error_code"],
                            evaluation["final_score"],
                            evaluation["score_block"],
                            evaluation["file_sha256"],
                            evaluation["model_digest"],
                            evaluation["model_size_bytes"],
                            row["created_at_ms"],
                            match.challenge_id,
                        ),
                    )
                    self._connection.execute(
                        "DELETE FROM challenge_events WHERE challenge_id = ?",
                        (match.challenge_id,),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO challenge_events (
                            challenge_id, sequence, previous_state, state,
                            occurred_at_ms, error_code, error_summary, details_json
                        ) VALUES (?, 0, NULL, ?, ?, ?, NULL, '{}')
                        """,
                        (
                            match.challenge_id,
                            target,
                            row["created_at_ms"],
                            evaluation["error_code"],
                        ),
                    )
                    self._connection.execute(
                        "DELETE FROM stage_scores WHERE challenge_id = ?",
                        (match.challenge_id,),
                    )
                    stages = evaluation.get("stages", [])
                    if target is ChallengeState.SUCCEEDED and not stages:
                        stages = [
                            {
                                "stage": ChallengeState.SCORING,
                                "score": evaluation["final_score"],
                                "score_block": evaluation["score_block"],
                                "incumbent_score": evaluation["incumbent_score"],
                                "paired_gain_lower_bound": evaluation["paired_gain_lower_bound"],
                                "evaluation_batch": evaluation["evaluation_batch"],
                            }
                        ]
                    for index, stage in enumerate(stages):
                        batch = stage.get("evaluation_batch")
                        metrics = {}
                        if batch is not None:
                            metrics = {
                                "incumbent_score": stage["incumbent_score"],
                                "paired_gain_lower_bound": stage["paired_gain_lower_bound"],
                                "cache_sha256": batch["sha256"],
                                "cache_generated_block": batch["generated_block"],
                                "cache_plan_digest": batch["plan_digest"],
                                "cache_seed_block_hash": batch["seed_block_hash"],
                            }
                        self._connection.execute(
                            """
                            INSERT INTO stage_scores (
                                challenge_id, stage, score, score_block,
                                metrics_json, created_at_ms
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                match.challenge_id,
                                stage["stage"],
                                stage["score"],
                                stage["score_block"],
                                _json(metrics),
                                row["created_at_ms"] + index,
                            ),
                        )
                if target is ChallengeState.SUCCEEDED or (
                    target is ChallengeState.PARTICIPANT_FAILED
                    and evaluation["model_digest"] is not None
                    and evaluation["error_code"] not in {"EXACT_DUPLICATE", "REFERENCE_DUPLICATE"}
                ):
                    slot = self._connection.execute(
                        "SELECT sequence FROM season_slots WHERE challenge_id = ?",
                        (match.challenge_id,),
                    ).fetchone()
                    if slot is None:
                        slot_count = int(
                            self._connection.execute(
                                "SELECT COUNT(*) FROM season_slots"
                            ).fetchone()[0]
                        )
                        if (
                            maximum_effective_challenges is not None
                            and slot_count >= maximum_effective_challenges
                        ):
                            raise RuntimeError(
                                "public matches exceed the frozen effective challenge capacity"
                            )
                        next_sequence = int(
                            self._connection.execute(
                                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM season_slots"
                            ).fetchone()[0]
                        )
                        self._connection.execute(
                            "INSERT INTO season_slots (sequence, challenge_id) VALUES (?, ?)",
                            (next_sequence, match.challenge_id),
                        )
                elif target is ChallengeState.PARTICIPANT_FAILED:
                    self._connection.execute(
                        "DELETE FROM season_slots WHERE challenge_id = ?", (match.challenge_id,)
                    )

                model_digest = evaluation["model_digest"]
                if model_digest is not None and evaluation["error_code"] not in {
                    "EXACT_DUPLICATE",
                    "TRACK_CAPACITY_REACHED",
                }:
                    owner = self._connection.execute(
                        "SELECT challenge_id FROM model_owners WHERE model_digest = ?",
                        (model_digest,),
                    ).fetchone()
                    if owner is None:
                        self._connection.execute(
                            "INSERT INTO model_owners (model_digest, challenge_id) VALUES (?, ?)",
                            (model_digest, match.challenge_id),
                        )
                    elif owner["challenge_id"] != match.challenge_id:
                        raise RuntimeError("public model ownership conflicts with local state")
                restored += 1
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return restored

    def public_snapshot(self) -> dict[str, Any]:
        """Return the fixed public allowlist from one SQLite read transaction."""

        self._connection.execute("BEGIN")
        try:
            leaderboard_entries = self.leaderboard()
            leaderboard = [
                {"rank": rank, **_public_entry(entry)}
                for rank, entry in enumerate(leaderboard_entries, start=1)
            ]
            challenge_rows = self._connection.execute(
                """
                SELECT c.challenge_id, c.submission_id, c.state,
                       c.total_attempts AS attempts,
                       c.error_code, c.error_summary, c.final_score,
                       c.score_block,
                       c.file_sha256, c.model_digest, c.model_size_bytes,
                       c.track_spec_digest,
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
            "challenges": challenges,
        }

    def public_challenge(self, identifier: str) -> dict[str, Any]:
        """Return one public challenge record and its state history."""

        self._connection.execute("BEGIN")
        try:
            row = self._connection.execute(
                """
                SELECT c.challenge_id, c.submission_id, c.state,
                       c.total_attempts AS attempts,
                       c.error_code, c.error_summary, c.final_score,
                       c.score_block,
                       c.file_sha256, c.model_digest, c.model_size_bytes,
                       c.track_spec_digest,
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
            SELECT stage, score, score_block, metrics_json, created_at_ms
            FROM stage_scores WHERE challenge_id = ?
            ORDER BY created_at_ms
            """,
            (row["challenge_id"],),
        ).fetchall()
        stage_documents = []
        for stage in stage_rows:
            stage_metrics = json.loads(stage["metrics_json"])
            document = {
                "stage": stage["stage"],
                "score": stage["score"],
                "score_block": stage["score_block"],
                "created_at_ms": stage["created_at_ms"],
            }
            if "cache_sha256" in stage_metrics:
                document["evaluation_batch"] = {
                    "sha256": stage_metrics["cache_sha256"],
                    "generated_block": stage_metrics.get("cache_generated_block"),
                    "plan_digest": stage_metrics.get("cache_plan_digest"),
                    "seed_block_hash": stage_metrics.get("cache_seed_block_hash"),
                }
                document["incumbent_score"] = stage_metrics.get("incumbent_score")
                document["paired_gain_lower_bound"] = stage_metrics.get("paired_gain_lower_bound")
            stage_documents.append(document)
        paired = {}
        if stage_rows:
            metrics = json.loads(stage_rows[-1]["metrics_json"])
            paired = {
                "incumbent_score": metrics.get("incumbent_score"),
                "paired_gain_lower_bound": metrics.get("paired_gain_lower_bound"),
                "evaluation_batch": (
                    {
                        "sha256": metrics.get("cache_sha256"),
                        "generated_block": metrics.get("cache_generated_block"),
                        "plan_digest": metrics.get("cache_plan_digest"),
                        "seed_block_hash": metrics.get("cache_seed_block_hash"),
                    }
                    if "cache_sha256" in metrics
                    else None
                ),
            }
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
            "final_score": row["final_score"],
            "score_block": row["score_block"],
            "incumbent_score": paired.get("incumbent_score"),
            "paired_gain_lower_bound": paired.get("paired_gain_lower_bound"),
            "evaluation_batch": paired.get("evaluation_batch"),
            "created_at_ms": row["created_at_ms"],
            "updated_at_ms": row["updated_at_ms"],
            "stages": stage_documents,
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
        scoring_ready_through_block: int | None = None,
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
            current = ChallengeState(row["state"])
            effective_stage = (
                ChallengeState(row["resume_state"])
                if current is ChallengeState.SYSTEM_DEFERRED
                else current
            )
            if scoring_ready_through_block is not None:
                if effective_stage is ChallengeState.SCORING:
                    decision_block = row["block_height"]
                elif effective_stage is ChallengeState.CONFIRMING:
                    exact = self._connection.execute(
                        """
                        SELECT score_block FROM stage_scores
                        WHERE challenge_id = ? AND stage = ?
                        """,
                        (row["challenge_id"], ChallengeState.SCORING),
                    ).fetchone()
                    if exact is None:
                        raise RuntimeError("confirmation challenge has no exact-stage decision")
                    decision_block = exact["score_block"]
                else:
                    raise RuntimeError("scoring claim selected a non-scoring stage")
                if decision_block > scoring_ready_through_block:
                    self._connection.commit()
                    return None
            if strict_fifo:
                terminal_placeholders = ",".join("?" for _ in TERMINAL_STATES)
                ownership_filter = (
                    "AND earlier_challenge.model_digest IS NULL" if include_queued else ""
                )
                earlier = self._connection.execute(
                    f"""
                    SELECT 1
                    FROM challenges earlier_challenge
                    JOIN submissions earlier_submission
                      ON earlier_submission.submission_id = earlier_challenge.submission_id
                    WHERE earlier_challenge.state NOT IN ({terminal_placeholders})
                      {ownership_filter}
                      AND (
                        earlier_submission.block_height,
                        earlier_submission.extrinsic_index,
                        earlier_submission.event_index
                      ) < (?, ?, ?)
                    LIMIT 1
                    """,
                    (
                        *TERMINAL_STATES,
                        row["block_height"],
                        row["extrinsic_index"],
                        row["event_index"],
                    ),
                ).fetchone()
                if earlier is not None:
                    self._connection.commit()
                    return None
            if strict_fifo and row["lease_token"] is not None and row["lease_expires_at_ms"] > now:
                self._connection.commit()
                return None

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
        "score_block": entry.score_block,
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
