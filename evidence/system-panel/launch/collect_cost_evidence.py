"""Persist hash-verified measurement events on the non-GPU control host.

Pipe the detached container's complete logs through stdin. Reconnecting safely
replays identical events; conflicting content fails without replacing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def persist(root, event):
    if event.get("format") != "veda-cost-raw-event-v1":
        return False
    name = event["filename"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json"):
        raise ValueError("unsafe raw evidence name")
    payload = json.dumps(
        event["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(payload).hexdigest() != event["sha256"]:
        raise ValueError("raw evidence event hash mismatch")
    target = root / name
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("immutable control-host evidence conflict")
        return False
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    with (args.output / "collector.log").open("a") as log:
        for line in sys.stdin:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.write(line)
                log.flush()
                continue
            if not isinstance(event, dict) or event.get("format") != "veda-cost-raw-event-v1":
                log.write(line)
                log.flush()
                continue
            count += persist(args.output, event)
            if count and count % 100 == 0:
                print(f"Persisted {count} immutable raw files on control host", flush=True)
    if not (args.output / "summary.json").exists():
        raise RuntimeError("stream ended before completion; reuse verified rows on retry")
    print(f"Evidence stream complete: {count} new immutable files", flush=True)
