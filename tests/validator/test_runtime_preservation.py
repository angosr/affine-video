import hashlib
import importlib.util
import json
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from validator.production.commitment import canonical_json

path = (
    Path(__file__).resolve().parents[2]
    / "evidence/system-panel/operations/preserve_runtime_assets.py"
)
spec = importlib.util.spec_from_file_location("preserve_runtime_assets", path)
preservation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preservation)


def artifact(name, payload=b"frozen"):
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "logical_name": name,
        "size_bytes": len(payload),
        "sha256": digest,
        "object_key": "artifacts/" + digest,
    }


def arguments(tmp_path, rows, reuse=()):
    manifest = tmp_path / "manifest.json"
    raw = canonical_json({"format": "veda-artifact-manifest-v1", "artifacts": rows})
    manifest.write_bytes(raw)
    destination = tmp_path / "mirror"
    destination.mkdir(exist_ok=True)
    return SimpleNamespace(
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        destination=destination,
        current=tmp_path / "task.json",
        reuse=list(reuse),
    )


def transfer(group, target, stopped):
    assert not stopped.is_set()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"frozen")
    preservation.verify_artifact(target, group, stopped=stopped)


def test_scope_is_exactly_the_authenticated_manifest_not_an_old_prefix_or_file_count(tmp_path):
    rows = [artifact("new/model"), artifact("other/cache")]
    args = arguments(tmp_path, rows)
    assert preservation.read_artifacts(args.manifest, args.manifest_sha256) == rows
    with pytest.raises(ValueError, match="digest changed"):
        preservation.read_artifacts(args.manifest, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        preservation.read_artifacts(args.manifest, "Z" * 64)
    args = arguments(tmp_path, [*rows, rows[0]])
    with pytest.raises(ValueError, match="unique"):
        preservation.read_artifacts(args.manifest, args.manifest_sha256)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/absolute",
        "../escape",
        "x/../y",
        "x//y",
        "x/./y",
        "x/",
        "x\\y",
        "x\x00y",
        "x\ny",
        "x\ty",
    ],
)
def test_manifest_and_destination_reject_unsafe_paths(tmp_path, name):
    args = arguments(tmp_path, [artifact(name)])
    with pytest.raises(ValueError):
        preservation.read_artifacts(args.manifest, args.manifest_sha256)
    with pytest.raises(ValueError):
        preservation.destination_file(args.destination, name)


@pytest.mark.parametrize(
    "changed", [{"size_bytes": 0}, {"size_bytes": True}, {"sha256": "A" * 64}, {"sha256": "0" * 63}]
)
def test_manifest_rejects_invalid_identity_fields(tmp_path, changed):
    args = arguments(tmp_path, [{**artifact("x"), **changed}])
    with pytest.raises(ValueError):
        preservation.read_artifacts(args.manifest, args.manifest_sha256)


def test_aliases_preserve_old_partial_paths_without_adding_old_artifacts(tmp_path, monkeypatch):
    args = arguments(tmp_path, [artifact("current/model"), artifact("runtime/image")])
    old = args.destination / "old/model"
    partial = old.parent / ".rsync-partial/model"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"fro")
    extra = args.destination / "old/text_encoder"
    extra.write_bytes(b"historical-extra")
    image = tmp_path / "image"
    image.write_bytes(b"frozen")
    image.chmod(0o600)
    args.reuse = [f"current={old.parent}", f"runtime/image={image}"]
    calls = []

    def record(row, target, stopped):
        calls.append((row["logical_name"], target))
        assert partial.read_bytes() == b"fro"
        transfer(row, target, stopped)

    monkeypatch.setattr(preservation, "transfer_artifact", record)
    preservation.run(args)
    result = json.loads(args.current.read_text())
    assert calls == [("current/model", old)]
    assert result["selected_artifact_count"] == result["verified_files"] == 2
    assert result["pending_transfer_files"] == 1
    assert result["read_only_reuse_names"] == ["runtime/image"]
    assert result["local_paths"] == {"current/model": str(old), "runtime/image": str(image)}
    assert result["state"] == "COMPLETED_VERIFIED"
    assert result["all_selected_file_hashes_verified"] is True
    assert not (
        {"target_r2_proven", "source_runtime_commit", "phase", "retired_service"} & result.keys()
    )
    assert image.stat().st_mode & 0o777 == 0o600  # Reuse never chmods shared external files.
    assert partial.read_bytes() == b"fro" and extra.read_bytes() == b"historical-extra"
    assert not (args.destination / "current").exists()


@pytest.mark.parametrize("reuse", ["x", "x=relative", "../x=/tmp/x", "absent=/tmp/x"])
def test_bad_or_unused_alias_is_rejected(tmp_path, reuse):
    args = arguments(tmp_path, [artifact("x/model")])
    with pytest.raises(ValueError):
        preservation.local_paths([artifact("x/model")], args.destination, [reuse])


def test_aliases_cannot_overlap_collide_or_follow_symlinks(tmp_path):
    args = arguments(tmp_path, [artifact("x/a"), artifact("y/b")])
    for reuse in (
        [f"x={tmp_path}/a", f"x/a={tmp_path}/b"],
        [f"x/a={tmp_path}/same", f"y/b={tmp_path}/same"],
    ):
        with pytest.raises(ValueError):
            preservation.local_paths([artifact("x/a"), artifact("y/b")], args.destination, reuse)
    link = tmp_path / "link"
    link.symlink_to(args.destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preservation.local_paths([artifact("x/a")], args.destination, [f"x={link}"])
    (args.destination / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preservation.destination_file(args.destination, "alias/model")


def test_external_reuse_missing_or_corrupt_never_downloads_or_changes_it(tmp_path, monkeypatch):
    external = tmp_path / "external"
    args = arguments(tmp_path, [artifact("x")], [f"x={external}"])
    calls = []
    monkeypatch.setattr(preservation, "transfer_artifact", lambda *_args: calls.append(True))
    with pytest.raises(ValueError, match="existing complete"):
        preservation.run(args)
    assert not calls and not args.current.exists()
    external.write_bytes(b"broken")
    external.chmod(0o600)
    with pytest.raises(ValueError, match="hash mismatch"):
        preservation.run(args)
    assert not calls and external.read_bytes() == b"broken"
    assert external.stat().st_mode & 0o777 == 0o600
    assert json.loads(args.current.read_text())["state"] == "FAILED_NOT_VERIFIED"


def test_verification_is_streaming_and_preserves_partial_and_corrupt_bytes(tmp_path, monkeypatch):
    row = artifact("model")
    target = tmp_path / "model"
    target.write_bytes(b"frozen")
    partial = tmp_path / ".rsync-partial/model"
    partial.parent.mkdir()
    partial.write_bytes(b"partial")
    monkeypatch.setattr(Path, "read_bytes", lambda _p: pytest.fail("must stream artifact bytes"))
    assert preservation.verify_artifact(target, row) == row["sha256"]
    assert target.stat().st_mode & 0o777 == 0o400
    for invalid in (dict(row, size_bytes=999), dict(row, sha256="0" * 64)):
        with pytest.raises(ValueError):
            preservation.verify_artifact(target, invalid)
    assert partial.stat().st_size == 7 and target.stat().st_size == 6


def test_transfer_has_bounded_retries_and_keeps_exact_partial_destination(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            assert kwargs == {"start_new_session": True}
            calls.append(command)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def wait(self, **_kwargs):
            return 12

        def poll(self):
            return 12

    monkeypatch.setattr(preservation.subprocess, "Popen", Process)
    stopped = SimpleNamespace(is_set=lambda: False, wait=lambda _seconds: False)
    target = tmp_path / "old/model"
    with pytest.raises(RuntimeError, match="bounded"):
        preservation.transfer_artifact(artifact("new/model"), target, stopped)
    assert len(calls) == 5
    assert calls[0][-2:] == ["gpu:/workspace/new/model", str(target)]
    assert "--partial-dir=.rsync-partial" in calls[0] and "--protect-args" in calls[0]
    assert "--bwlimit=6144" in calls[0] and "--timeout=120" in calls[0]
    assert not any("delete" in option or "inplace" in option for option in calls[0])
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="stopped"):
        preservation.transfer_artifact(artifact("new/model"), target, event)
    assert len(calls) == 5


def test_partial_symlinks_are_rejected_before_spawning_rsync(tmp_path, monkeypatch):
    (tmp_path / ".rsync-partial").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    monkeypatch.setattr(preservation.subprocess, "Popen", lambda *_a, **_k: pytest.fail("spawned"))
    with pytest.raises(ValueError, match="symlink"):
        preservation.transfer_artifact(artifact("model"), tmp_path / "model", threading.Event())


def test_same_destination_lock_rejects_a_second_real_process_writer(tmp_path, monkeypatch):
    args = arguments(tmp_path, [artifact("model")])
    lock = args.destination / ".preservation.lock"
    code = (
        "import fcntl,sys; f=open(sys.argv[1],'a'); "
        "fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); sys.stdin.read()"
    )
    monkeypatch.setattr(preservation, "transfer_artifact", lambda *_a: pytest.fail("second writer"))
    with subprocess.Popen(
        [sys.executable, "-c", code, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as writer:
        assert writer.stdout.readline() == "locked\n"
        try:
            with pytest.raises(BlockingIOError):
                preservation.run(args)
            assert not args.current.exists()
        finally:
            writer.communicate("")


def test_resume_disk_budget_keeps_partial_and_reserve_but_excludes_complete(tmp_path, monkeypatch):
    rows = [artifact(name) for name in ("complete", "missing", "partial")]
    args = arguments(tmp_path, rows)
    (args.destination / "complete").write_bytes(b"frozen")
    partial = args.destination / ".rsync-partial/partial"
    partial.parent.mkdir()
    partial.write_bytes(b"fro")
    free = SimpleNamespace(free=preservation.RESERVE_BYTES + 11)
    monkeypatch.setattr(preservation.shutil, "disk_usage", lambda _p: free)
    monkeypatch.setattr(preservation, "transfer_artifact", transfer)
    with pytest.raises(ValueError, match="insufficient free disk"):
        preservation.run(args)
    assert not args.current.exists()
    free.free += 1
    preservation.run(args)
    result = json.loads(args.current.read_text())
    assert result["verified_files"] == 3 and result["pending_transfer_bytes"] == 12
    assert result["state"] == "COMPLETED_VERIFIED"
    assert partial.read_bytes() == b"fro" and not list(tmp_path.glob(".preservation-*"))


def test_corrupt_transfer_cannot_be_counted_as_completed(tmp_path, monkeypatch):
    args = arguments(tmp_path, [artifact("model")])

    def corrupt(row, target, stopped):
        target.write_bytes(b"broken")
        preservation.verify_artifact(target, row, stopped=stopped)

    monkeypatch.setattr(preservation, "transfer_artifact", corrupt)
    with pytest.raises(ValueError, match="hash mismatch"):
        preservation.run(args)
    result = json.loads(args.current.read_text())
    assert result["state"] == "FAILED_NOT_VERIFIED" and result["verified_files"] == 0
    assert not result.get("all_selected_file_hashes_verified", False)


def test_stop_terminates_only_its_private_transfer_process_group(tmp_path, monkeypatch):
    stopped = threading.Event()
    killed = []

    class Process:
        pid = 31415
        done = False

        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

        def wait(self, **_kwargs):
            if self.done:
                return -15
            stopped.set()
            raise subprocess.TimeoutExpired("test-rsync", 1)

        def poll(self):
            return -15 if self.done else None

    def kill(pid, signum):
        killed.append((pid, signum))
        Process.done = True

    monkeypatch.setattr(preservation.subprocess, "Popen", Process)
    monkeypatch.setattr(preservation.os, "killpg", kill)
    with pytest.raises(RuntimeError, match="stopped during transfer"):
        preservation.transfer_artifact(artifact("model"), tmp_path / "model", stopped)
    assert killed == [(31415, signal.SIGTERM)]


@pytest.mark.skipif(
    not Path("/usr/bin/rsync").is_file(),
    reason="the control-host mirror integration test requires /usr/bin/rsync",
)
def test_real_rsync_resumes_the_existing_partial_in_the_same_mirror(tmp_path, monkeypatch, capfd):
    payload = b"frozen-weights-0123456789" * 131072
    source = tmp_path / "source"
    source.write_bytes(payload)
    target = tmp_path / "old/model"
    partial = target.parent / ".rsync-partial/model"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[: 1024 * 1024])
    row = artifact("current/model", payload)
    original = preservation.rsync_command

    def local_command(artifact, destination):
        command = original(artifact, destination)
        command[-2] = str(source)
        command.insert(1, "--no-whole-file")  # Remote transfers already use delta by default.
        return command

    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setattr(preservation, "rsync_command", local_command)
    preservation.transfer_artifact(row, target, threading.Event())
    assert target.read_bytes() == payload
    matched = re.search(r"Matched data: ([\d,]+)", capfd.readouterr().out)
    assert matched is not None and int(matched.group(1).replace(",", "")) > 0


def test_lock_symlink_is_rejected_without_touching_external_file(tmp_path):
    args = arguments(tmp_path, [artifact("model")])
    external = tmp_path / "external"
    external.write_bytes(b"unchanged")
    (args.destination / ".preservation.lock").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        preservation.run(args)
    assert external.read_bytes() == b"unchanged" and not args.current.exists()


def test_second_submit_failure_stops_and_joins_the_first_transfer(tmp_path, monkeypatch):
    args = arguments(tmp_path, [artifact("a"), artifact("b")])
    real_executor = preservation.ThreadPoolExecutor
    started = threading.Event()
    joined = threading.Event()

    class Executor:
        submissions = 0

        def __init__(self, **kwargs):
            self.executor = real_executor(**kwargs)

        def __enter__(self):
            self.executor.__enter__()
            return self

        def __exit__(self, *args):
            return self.executor.__exit__(*args)

        def submit(self, *args):
            self.submissions += 1
            if self.submissions == 2:
                assert started.wait(2)
                raise RuntimeError("injected second submit failure")
            return self.executor.submit(*args)

    def waiting(_row, _target, stopped):
        started.set()
        assert stopped.wait(2)
        joined.set()

    monkeypatch.setattr(preservation, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(preservation, "transfer_artifact", waiting)
    with pytest.raises(RuntimeError, match="injected second submit"):
        preservation.run(args)
    assert joined.is_set()
    assert json.loads(args.current.read_text())["state"] == "FAILED_NOT_VERIFIED"


def test_signal_stop_is_recorded_as_stopped_not_success(tmp_path, monkeypatch):
    args = arguments(tmp_path, [artifact("model")])
    target = args.destination / "model"
    target.write_bytes(b"frozen")

    def interrupted(*_args, **_kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        raise RuntimeError("controlled stop")

    monkeypatch.setattr(preservation, "verify_artifact", interrupted)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match="controlled stop"):
        preservation.run(args)
    assert json.loads(args.current.read_text())["state"] == "STOPPED_NOT_VERIFIED"
    assert signal.getsignal(signal.SIGTERM) is before
