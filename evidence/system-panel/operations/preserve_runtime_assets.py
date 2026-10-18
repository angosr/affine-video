"""Preserve an explicit runtime manifest in one locked, resumable control-host mirror."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path

from validator.production.commitment import ArtifactManifest

WORKERS = 4
ATTEMPTS = 5
GROUP_RATE_KIB = 6144
RESERVE_BYTES = 64 * 1024**3
_output_lock = threading.Lock()


def emit(event, **fields):
    with _output_lock:
        print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def safe_name(name):
    if (
        not isinstance(name, str)
        or not name
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or "\\" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError("invalid artifact path")
    return name


def checked_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("local paths must be absolute without traversal")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("local paths must not contain symlinks")
    return path


def destination_file(destination, name):
    return checked_path(checked_path(destination) / safe_name(name))


def read_artifacts(manifest_path, expected_sha256):
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("manifest SHA-256 is invalid")
    payload = checked_path(manifest_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("runtime manifest digest changed")
    artifacts = [artifact.document() for artifact in ArtifactManifest.decode(payload).artifacts]
    for artifact in artifacts:
        safe_name(artifact["logical_name"])
    return artifacts


def local_paths(artifacts, destination, reuse):
    """Keep aliases inside the mirror writable; reuse external files read-only."""
    aliases = {}
    for value in reuse:
        prefix, separator, location = value.partition("=")
        if not separator:
            raise ValueError("reuse must be LOGICAL_PREFIX=/absolute/local/path")
        safe_name(prefix)
        if any(
            prefix == old or prefix.startswith(old + "/") or old.startswith(prefix + "/")
            for old in aliases
        ):
            raise ValueError("overlapping reuse prefixes")
        aliases[prefix] = checked_path(location)
    paths = {}
    used = set()
    targets = set()
    for artifact in artifacts:
        name = artifact["logical_name"]
        target = destination_file(destination, name)
        for prefix, base in aliases.items():
            if name == prefix or name.startswith(prefix + "/"):
                target = checked_path(base / name[len(prefix) + 1 :]) if name != prefix else base
                used.add(prefix)
                break
        if target in targets:
            raise ValueError("different artifacts cannot share one local path")
        targets.add(target)
        paths[name] = target
    if used != set(aliases):
        raise ValueError("reuse prefix is absent from the runtime manifest")
    return paths


def rsync_command(artifact, target):
    return [
        "/usr/bin/rsync",
        "--times",
        "--copy-links",
        "--protect-args",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--chmod=Fu=rw,Fgo=",
        f"--bwlimit={GROUP_RATE_KIB}",
        "--timeout=120",
        "--stats",
        "--itemize-changes",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 "
        "-o ServerAliveCountMax=3",
        "--",
        "gpu:/workspace/" + safe_name(artifact["logical_name"]),
        str(target),
    ]


def transfer_artifact(artifact, target, stopped):
    target = checked_path(target)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    checked_path(target.parent / ".rsync-partial" / target.name)
    for attempt in range(1, ATTEMPTS + 1):
        if stopped.is_set():
            raise RuntimeError("preservation stopped; no new transfer attempt")
        with subprocess.Popen(rsync_command(artifact, target), start_new_session=True) as process:
            emit(
                "TRANSFER_STARTED",
                logical_name=artifact["logical_name"],
                attempt=attempt,
                pid=process.pid,
                target=str(target),
                size_bytes=artifact["size_bytes"],
            )
            try:
                while True:
                    if stopped.is_set():
                        raise RuntimeError("preservation stopped during transfer")
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if code == 0:
            verify_artifact(target, artifact, stopped=stopped)
            emit("TRANSFER_FINISHED", logical_name=artifact["logical_name"], attempt=attempt)
            return
        emit(
            "TRANSFER_RETRY", logical_name=artifact["logical_name"], attempt=attempt, exit_code=code
        )
        if attempt < ATTEMPTS and stopped.wait(min(60, 5 * 2 ** (attempt - 1))):
            raise RuntimeError("preservation stopped during transfer backoff")
    raise RuntimeError("bounded transfer attempts exhausted: " + artifact["logical_name"])


def verify_artifact(target, artifact, *, readonly=False, stopped=None):
    target = checked_path(target)
    before = target.stat()
    if not target.is_file() or before.st_size != artifact["size_bytes"]:
        raise ValueError("missing or wrong-size preserved artifact: " + artifact["logical_name"])
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if stopped is not None and stopped.is_set():
                raise RuntimeError("preservation stopped during verification")
            digest.update(chunk)
    after = target.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if (
        any(getattr(before, field) != getattr(after, field) for field in fields)
        or digest.hexdigest() != artifact["sha256"]
    ):
        raise ValueError(
            "preserved artifact hash mismatch or changed file: " + artifact["logical_name"]
        )
    if not readonly:
        target.chmod(0o400)
    return digest.hexdigest()


def save_current(current, document):
    fd, temporary = tempfile.mkstemp(prefix=".preservation-", dir=current.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, current)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(args):
    artifacts = read_artifacts(args.manifest, args.manifest_sha256)
    destination = checked_path(args.destination)
    current = checked_path(args.current)
    if not destination.is_dir() or not current.parent.is_dir():
        raise ValueError("resume requires existing mirror and task directories")
    paths = local_paths(artifacts, destination, args.reuse)
    stopped = threading.Event()
    interrupted = threading.Event()

    def request_stop(_signum, _frame):
        interrupted.set()
        stopped.set()

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        lock_path = destination_file(destination, ".preservation.lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pending = []
            external = set()
            for artifact in artifacts:
                target = paths[artifact["logical_name"]]
                if destination not in target.parents:
                    external.add(artifact["logical_name"])
                    if not target.is_file() or target.stat().st_size != artifact["size_bytes"]:
                        raise ValueError("external reuse requires an existing complete local file")
                elif not target.is_file() or target.stat().st_size != artifact["size_bytes"]:
                    pending.append(artifact)
            required_free = sum(row["size_bytes"] for row in pending) + RESERVE_BYTES
            if shutil.disk_usage(destination).free < required_free:
                raise ValueError("insufficient free disk for a recoverable resumed transfer")
            document = {
                "format": "veda-off-gpu-runtime-preservation-task-v1",
                "state": "RUNNING_NOT_VERIFIED",
                "main_pid": os.getpid(),
                "artifact_manifest_sha256": args.manifest_sha256,
                "manifest_path": str(args.manifest),
                "destination": str(destination),
                "operations_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selected_artifact_count": len(artifacts),
                "selected_bytes": sum(row["size_bytes"] for row in artifacts),
                "pending_transfer_files": len(pending),
                "pending_transfer_bytes": sum(row["size_bytes"] for row in pending),
                "maximum_parallel_transfers": WORKERS,
                "maximum_transfer_attempts_per_file": ATTEMPTS,
                "aggregate_rate_limit_kib_per_second": WORKERS * GROUP_RATE_KIB,
                "started_unix_seconds": time.time(),
                "verified_files": 0,
                "verified_bytes": 0,
                "local_paths": {name: str(path) for name, path in paths.items()},
                "read_only_reuse_names": sorted(external),
            }

            def verified(artifact):
                document["verified_files"] += 1
                document["verified_bytes"] += artifact["size_bytes"]
                save_current(current, document)
                emit(
                    "ASSET_VERIFIED",
                    logical_name=artifact["logical_name"],
                    sha256=artifact["sha256"],
                    size_bytes=artifact["size_bytes"],
                    verified_files=document["verified_files"],
                )

            try:
                save_current(current, document)
                emit("STARTED", **document)
                with ThreadPoolExecutor(max_workers=WORKERS) as workers:
                    try:
                        futures = {}
                        for row in sorted(
                            pending, key=lambda row: (-row["size_bytes"], row["logical_name"])
                        ):
                            future = workers.submit(
                                transfer_artifact, row, paths[row["logical_name"]], stopped
                            )
                            futures[future] = row
                        pending_names = {row["logical_name"] for row in pending}
                        for artifact in artifacts:
                            name = artifact["logical_name"]
                            if name not in pending_names:
                                verify_artifact(
                                    paths[name],
                                    artifact,
                                    readonly=name in external,
                                    stopped=stopped,
                                )
                                verified(artifact)
                        for future in as_completed(futures):
                            future.result()
                            verified(futures[future])
                    finally:
                        stopped.set()
                document["state"] = "COMPLETED_VERIFIED"
                document["all_selected_file_hashes_verified"] = True
                document["completed_unix_seconds"] = time.time()
                save_current(current, document)
                emit("COMPLETED", **document)
            except BaseException as exc:
                document["state"] = (
                    "STOPPED_NOT_VERIFIED" if interrupted.is_set() else "FAILED_NOT_VERIFIED"
                )
                document["error"] = str(exc) or type(exc).__name__
                save_current(current, document)
                emit("FAILED", **document)
                raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--reuse",
        action="append",
        default=[],
        help="LOGICAL_PREFIX=/absolute/local/path; outside mirror is read-only",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
