"""Bounded disposal of worker processes and their disposable queues."""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any


def close_workers(queues: list[Any], processes: list[Any], result: Any | None) -> None:
    """Survive partial startup, broken pipes and unread queue feeder buffers."""

    failed: set[int] = set()
    for index, tasks in enumerate(queues):
        try:
            tasks.put(None)
        except Exception:
            failed.add(index)
    deadline = time.monotonic() + 30
    for index, process in enumerate(processes):
        if index not in failed:
            with suppress(Exception):  # an unstarted Process cannot be joined
                process.join(timeout=max(0, deadline - time.monotonic()))
        with suppress(Exception):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    for value in [*queues, *([result] if result is not None else [])]:
        # A killed reader cannot drain the feeder pipe. Joining that feeder can
        # hang forever, even after every child was successfully terminated.
        with suppress(Exception):
            value.cancel_join_thread()
        with suppress(Exception):
            value.close()
