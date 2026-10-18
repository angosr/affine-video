"""Observe immutable cost attempts without restarting jobs or extending deadlines.

Completion times come from the GPU host's files, never the replaying collectors.
Rate projections are diagnostics, not an early-stop or acceptance criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from itertools import product
from pathlib import Path

REMOTE = r"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
execution = json.loads(sys.argv[1])
result = []
for job in execution["jobs"]:
    container = json.loads(subprocess.check_output(["docker", "inspect", job["name"]]))[0]
    root = Path(execution["raw_root_gpu"]) / job["output"]
    paths = list(root.glob("*.json"))
    hashes = {}
    for name in ("protocol.json", "summary.json"):
        path = root / name
        if path.exists():
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    result.append({
        "id": container["Id"], "image": container["Image"],
        "state": container["State"], "command": container["Config"]["Cmd"],
        "entrypoint": container["Config"]["Entrypoint"], "now": time.time(),
        "mtimes": {path.name: path.stat().st_mtime for path in paths}, "hashes": hashes,
    })
print(json.dumps(result))
"""


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def progress(registration, job, observation):
    """Validate the live handle and summarize completed model/text boundaries."""
    state = observation["state"]
    timeout = registration["timeout_seconds_per_attempt"]
    command = observation["command"]
    if (
        observation["id"] != job["container_id"]
        or observation["image"] != registration["runtime_image_id"]
        or observation["entrypoint"] != ["timeout"]
        or command[0] != f"{timeout}s"
        or state["StartedAt"] != job["started_at_utc"]
    ):
        raise ValueError("live container differs from registered attempt or timeout")
    start = datetime.fromisoformat(state["StartedAt"].replace("Z", "+00:00")).timestamp()
    deadline = start + timeout
    mtimes = observation["mtimes"]
    expected, completed, incomplete = set(), [], []
    previous_boundary = start
    for geometry, text, model in product(
        job["geometries"], registration["text_token_counts"], registration["models"]
    ):
        names = {
            f"{geometry.replace(':', 'x')}-{text}-{model}-layer{layer}-{budget}.json"
            for layer, budget in product(registration["layers"], registration["budgets"])
        }
        expected.update(names)
        present = names & mtimes.keys()
        group = f"{geometry}/{text}/{model}"
        if present == names:
            if incomplete:
                raise ValueError("completed group follows an incomplete registered group")
            end = max(mtimes[name] for name in names)
            seconds = end - previous_boundary
            if seconds <= 0:
                raise ValueError("completion times do not follow the registered traversal")
            completed.append(
                {
                    "group": group,
                    "rows": len(names),
                    "seconds": seconds,
                    "rows_per_minute": 60 * len(names) / seconds,
                    "finished_at_unix": end,
                }
            )
            previous_boundary = end
        else:
            incomplete.append({"group": group, "rows": len(present), "expected": len(names)})
    actual = {name for name in mtimes if "-layer" in name}
    if actual - expected:
        raise ValueError("attempt contains unregistered row dimensions")
    count, remaining = len(actual), len(expected) - len(actual)
    completed_rows = sum(group["rows"] for group in completed)
    status = state["Status"]
    terminal = status in {"exited", "dead"}
    end = (
        datetime.fromisoformat(state["FinishedAt"].replace("Z", "+00:00")).timestamp()
        if terminal
        else observation["now"]
    )
    elapsed = max(0, end - start)
    rate_seconds = (
        sum(group["seconds"] for group in completed) / completed_rows if completed_rows else None
    )
    projection = None if rate_seconds is None else remaining * rate_seconds
    complete = count == len(expected) and "summary.json" in observation["hashes"]
    failed = terminal and (state["ExitCode"] != 0 or not complete)
    return {
        "container_id": job["container_id"],
        "gpu": job["gpu"],
        "status": status,
        "exit_code": state["ExitCode"] if terminal else None,
        "rows": count,
        "expected_rows": len(expected),
        "elapsed_seconds": elapsed,
        "deadline_unix": deadline,
        "time_basis": "original_gpu_host_row_file_mtime; first_group_includes_startup",
        "seconds_until_deadline": max(0, deadline - observation["now"]),
        "deadline_elapsed": end >= deadline,
        "completed_groups": completed,
        "incomplete_groups": incomplete,
        "remaining_seconds_at_completed_group_mean_rate": projection,
        "projection_exceeds_remaining_time": projection is not None
        and projection > deadline - observation["now"],
        "projection_scope": "scenario_only_unseen_geometries_and_models_not_measured",
        "measurement_failed": failed,
        "measurement_complete_awaiting_registered_analysis": terminal and not failed and complete,
        "protocol_sha256": observation["hashes"].get("protocol.json"),
        "summary_sha256": observation["hashes"].get("summary.json"),
        "release_accepted": False,
    }


def snapshot(execution, registration, host):
    command = "python - " + shlex.quote(json.dumps(execution))
    remote = json.loads(
        subprocess.check_output(
            ["ssh", "-o", "ConnectTimeout=15", host, command], input=REMOTE, text=True, timeout=60
        )
    )
    reports = []
    for job, observation in zip(execution["jobs"], remote, strict=True):
        root = Path(execution["raw_root_control"]) / job["output"]
        if sha256(root / "protocol.json") != observation["hashes"].get("protocol.json"):
            raise ValueError("collector protocol does not match the live attempt")
        protocol = json.loads((root / "protocol.json").read_text())
        if (
            protocol["registration_sha256"] != execution["registration_sha256"]
            or protocol["phase"] != execution["phase"]
            or protocol["seed"] != execution["seed"]
            or protocol["geometries"] != job["geometries"]
        ):
            raise ValueError("attempt protocol binding mismatch")
        report = progress(registration, job, observation)
        service = f"veda-cost-collector-gpu{job['gpu']}.service"
        report["collector_service"] = service
        report["collector_active_state"] = subprocess.check_output(
            ["systemctl", "--user", "show", service, "-p", "ActiveState", "--value"], text=True
        ).strip()
        report["mirrored_summary_verified"] = (root / "summary.json").exists() and sha256(
            root / "summary.json"
        ) == report["summary_sha256"]
        reports.append(report)
    return {
        "format": "veda-cost-attempt-progress-v1",
        "observed_at_unix": time.time(),
        "registration_sha256": execution["registration_sha256"],
        "attempt": execution["attempt"],
        "jobs": reports,
        "release_accepted": False,
    }


def write_event(root, document, name):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    if path.exists():
        return False
    with path.open("x") as stream:
        json.dump(document, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="gpu")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text())
    if sha256(args.registration) != execution["registration_sha256"]:
        raise ValueError("registration differs from the immutable execution record")
    registration = json.loads(args.registration.read_text())
    geometries = [geometry for job in execution["jobs"] for geometry in job["geometries"]]
    if (
        execution["runtime_image_id"] != registration["runtime_image_id"]
        or len(geometries) != len(set(geometries))
        or set(geometries) != set(registration["geometries"])
        or execution["seed"] not in registration["seeds"][execution["phase"]]
    ):
        raise ValueError("execution shards differ from the registered phase")
    failures = 0
    while True:
        try:
            report = snapshot(execution, registration, args.host)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failures += 1
            write_event(
                args.output,
                {"observation_error": str(error), "jobs_restarted": False},
                f"observation-error-{time.time_ns()}",
            )
            if failures >= 3 or not args.watch:
                raise
        else:
            failures = 0
            boundary = [
                (
                    job["container_id"],
                    job["status"],
                    len(job["completed_groups"]),
                    job["deadline_elapsed"],
                    job["collector_active_state"],
                    job["mirrored_summary_verified"],
                )
                for job in report["jobs"]
            ]
            name = hashlib.sha256(json.dumps(boundary).encode()).hexdigest()
            report["watcher_sha256"] = sha256(Path(__file__))
            report["execution_record_sha256"] = sha256(args.execution)
            if write_event(args.output, report, name):
                print(json.dumps(report, sort_keys=True), flush=True)
            if not args.watch or all(
                job["measurement_failed"]
                or (
                    job["measurement_complete_awaiting_registered_analysis"]
                    and job["mirrored_summary_verified"]
                )
                for job in report["jobs"]
            ):
                if any(job["measurement_failed"] for job in report["jobs"]):
                    raise RuntimeError(
                        "registered attempt failed; preserve rows without mixing attempts"
                    )
                return
            if all(
                report["observed_at_unix"] > job["deadline_unix"] + 300 for job in report["jobs"]
            ):
                raise RuntimeError(
                    "registered deadline elapsed; inspect existing handles, do not extend"
                )
        time.sleep(60)


if __name__ == "__main__":
    main()
