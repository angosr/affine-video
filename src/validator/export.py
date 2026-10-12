"""Atomic public snapshot export."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from validator.store import ChallengeStore


def export_snapshot(store: ChallengeStore, output_dir: str | Path) -> Path:
    """Write the public leaderboard and challenge history as one atomic file."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".snapshot-", dir=directory)
    temporary = Path(temporary_name)
    target = directory / "snapshot.json"
    try:
        payload = json.dumps(
            store.public_snapshot(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
