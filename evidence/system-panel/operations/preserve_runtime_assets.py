"""Resume the frozen 067 mirror, online assets first; never alter the GPU source."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVIDENCE = Path("/home/aiuser/work/veda-evidence/evidence/system-panel")
BOOTSTRAP = EVIDENCE / "pipeline-7cf0e10/private-bootstrap.tar.gz"
DESTINATION = Path("/home/aiuser/work/veda-evidence/runtime-assets")
CURRENT = EVIDENCE / "runtime-assets-preservation-task.json"
MANIFEST_SHA256 = "4c158932ef095cd7dafb68a4fe4249bf4fead6294649b7f2757c69e12b6e22b2"
SOURCE_COMMIT = "067aa3d11aa00e2459250cb14443f93c7b4272c7"
OLD_SERVICE = "veda-preserve-runtime-assets-067aa3d.service"
SERVICE = "veda-preserve-runtime-assets-067aa3d-priority.service"
WORKERS = 4
ATTEMPTS = 5
GROUP_RATE_KIB = 6144
_output_lock = threading.Lock()


def emit(event, **fields):
    with _output_lock:
        print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def read_artifacts(bootstrap=BOOTSTRAP):
    with tarfile.open(bootstrap, "r:gz") as archive:
        payload = archive.extractfile("artifact_manifest.json").read()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen source manifest digest changed")
    artifacts = [
        row for row in json.loads(payload)["artifacts"]
        if row["logical_name"].startswith(
            ("veda-assets/", "veda-private/source-v6-final-candidate/")
        )
    ]
    for row in artifacts:
        name = row["logical_name"]
        if (
            any(part in {"", ".", ".."} for part in name.split("/"))
            or "\0" in name or "\\" in name
            or type(row["size_bytes"]) is not int or row["size_bytes"] <= 0
            or len(row["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["sha256"])
        ):
            raise ValueError("invalid frozen artifact")
    if (
        len(artifacts) != 88
        or len({row["logical_name"] for row in artifacts}) != 88
        or sum(row["size_bytes"] for row in artifacts) != 146703809034
    ):
        raise ValueError("the preservation scope must remain exactly the same 88 artifacts")
    return sorted(artifacts, key=lambda row: row["logical_name"])


def file_list(artifacts):
    return b"".join(row["logical_name"].encode() + b"\0" for row in artifacts)


def transfer_groups(artifacts):
    """Deterministic, byte-balanced disjoint groups within each strict phase."""
    phases = []
    for encoder in (False, True):
        groups = [[] for _ in range(WORKERS)]
        sizes = [0] * WORKERS
        selected = [
            row for row in artifacts if ("/text_encoder/" in row["logical_name"]) == encoder
        ]
        for row in sorted(selected, key=lambda row: (-row["size_bytes"], row["logical_name"])):
            index = min(range(WORKERS), key=lambda index: (sizes[index], index))
            groups[index].append(row)
            sizes[index] += row["size_bytes"]
        phases.append([
            sorted(group, key=lambda row: row["logical_name"])
            for group in groups if group
        ])
    return phases


def destination_file(destination, name):
    target = destination / name
    if destination.resolve() not in target.resolve().parents:
        raise ValueError("mirror path escapes the fixed destination")
    if any(parent.is_symlink() for parent in (target, *target.parents) if parent != destination):
        raise ValueError("mirror must not contain symlinks")
    return target


def rsync_command(destination):
    return [
        "/usr/bin/rsync", "--recursive", "--times", "--copy-links", "--relative",
        "--from0", "--files-from=-", "--partial", "--partial-dir=.rsync-partial",
        "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", f"--bwlimit={GROUP_RATE_KIB}",
        "--timeout=120", "--stats", "--itemize-changes", "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 "
        "-o ServerAliveCountMax=3",
        "gpu:/workspace/", str(destination) + "/",
    ]


def transfer_group(artifacts, destination, phase, group, stopped):
    for attempt in range(1, ATTEMPTS + 1):
        if stopped.is_set():
            raise RuntimeError("preservation stopped; no new transfer attempt")
        with subprocess.Popen(rsync_command(destination), stdin=subprocess.PIPE) as process:
            emit("TRANSFER_STARTED", phase=phase, group=group, attempt=attempt, pid=process.pid,
                 files=len(artifacts), bytes=sum(row["size_bytes"] for row in artifacts))
            process.communicate(file_list(artifacts))
            code = process.returncode
        if code == 0:
            emit("TRANSFER_FINISHED", phase=phase, group=group, attempt=attempt)
            return
        emit("TRANSFER_RETRY", phase=phase, group=group, attempt=attempt, exit_code=code)
        if stopped.wait(min(60, 5 * 2 ** (attempt - 1))):
            raise RuntimeError("preservation stopped during transfer backoff")
    raise RuntimeError(f"bounded transfer attempts exhausted: phase={phase} group={group}")


def verify_artifact(destination, artifact):
    target = destination_file(destination, artifact["logical_name"])
    if not target.is_file() or target.stat().st_size != artifact["size_bytes"]:
        raise ValueError(f"missing or wrong-size preserved artifact: {artifact['logical_name']}")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError(f"preserved artifact hash mismatch: {artifact['logical_name']}")
    target.chmod(0o400)
    return digest


def save_current(document):
    fd, temporary = tempfile.mkstemp(prefix=".preservation-", dir=CURRENT.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CURRENT)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run():
    old_state = subprocess.check_output(
        ["systemctl", "--user", "show", OLD_SERVICE, "-p", "ActiveState", "--value"], text=True,
    ).strip()
    if old_state not in {"inactive", "failed"}:
        raise RuntimeError("old writer is not retired; refusing concurrent destination writes")
    artifacts = read_artifacts()
    if not DESTINATION.is_dir() or DESTINATION.is_symlink():
        raise ValueError("resume requires the original real mirror directory")
    phases = transfer_groups(artifacts)
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: stopped.set())
    with (DESTINATION / ".preservation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending_bytes = 0
        for artifact in artifacts:
            target = destination_file(DESTINATION, artifact["logical_name"])
            if not target.is_file() or target.stat().st_size != artifact["size_bytes"]:
                pending_bytes += artifact["size_bytes"]
        # Completed files already occupy disk. The reserve also covers four
        # simultaneous replacement temporaries; size does not waive SHA checks.
        required_free = pending_bytes + 64 * 1024**3
        if shutil.disk_usage(DESTINATION).free < required_free:
            raise ValueError("insufficient free disk for a recoverable resumed transfer")
        document = {
            "format": "veda-off-gpu-runtime-preservation-task-v1",
            "state": "RUNNING_NOT_VERIFIED", "service": SERVICE, "main_pid": os.getpid(),
            "source_runtime_commit": SOURCE_COMMIT, "artifact_manifest_sha256": MANIFEST_SHA256,
            "selected_file_list_sha256": hashlib.sha256(file_list(artifacts)).hexdigest(),
            "selected_artifact_count": len(artifacts),
            "selected_bytes": sum(row["size_bytes"] for row in artifacts),
            "destination": str(DESTINATION), "strategy": "non-encoder-first-four-way",
            "operations_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cpu_quota_percent": 100, "memory_high_bytes": 384 * 1024**2,
            "memory_max_bytes": 768 * 1024**2, "service_runtime_limit_hours": 48,
            "maximum_transfer_attempts_per_group": ATTEMPTS,
            "aggregate_rate_limit_kib_per_second": WORKERS * GROUP_RATE_KIB,
            "retired_service": OLD_SERVICE, "started_unix_seconds": time.time(),
            "verified_files": 0, "verified_bytes": 0, "private_credentials_copied": False,
            "gpu_requested": False, "target_r2_proven": False, "release_accepted": False,
            "groups": [[{
                "files": [row["logical_name"] for row in group],
                "bytes": sum(row["size_bytes"] for row in group),
                "file_list_sha256": hashlib.sha256(file_list(group)).hexdigest(),
            } for group in phase] for phase in phases],
        }
        try:
            save_current(document)
            emit("STARTED", **document)
            for phase, groups in enumerate(phases):
                document["phase"] = "ENCODER" if phase else "NON_ENCODER"
                save_current(document)
                with ThreadPoolExecutor(max_workers=WORKERS) as workers:
                    futures = [
                        workers.submit(transfer_group, group, DESTINATION, phase, index, stopped)
                        for index, group in enumerate(groups)
                    ]
                    for future in as_completed(futures):
                        future.result()
                for group in groups:
                    for artifact in group:
                        if stopped.is_set():
                            raise RuntimeError("preservation stopped before verification completed")
                        digest = verify_artifact(DESTINATION, artifact)
                        document["verified_files"] += 1
                        document["verified_bytes"] += artifact["size_bytes"]
                        save_current(document)
                        emit(
                            "ASSET_VERIFIED", logical_name=artifact["logical_name"], sha256=digest,
                            size_bytes=artifact["size_bytes"],
                            verified_files=document["verified_files"],
                        )
                emit("PHASE_VERIFIED", phase=phase, verified_files=document["verified_files"],
                     verified_bytes=document["verified_bytes"])
            document["state"] = "COMPLETED_VERIFIED"
            document["all_selected_file_hashes_verified"] = True
            document["completed_unix_seconds"] = time.time()
            save_current(document)
            emit("COMPLETED", **document)
        except BaseException as exc:
            document["state"] = (
                "STOPPED_NOT_VERIFIED" if stopped.is_set() else "FAILED_NOT_VERIFIED"
            )
            document["error"] = str(exc) or type(exc).__name__
            save_current(document)
            emit("FAILED", **document)
            raise


if __name__ == "__main__":
    run()
