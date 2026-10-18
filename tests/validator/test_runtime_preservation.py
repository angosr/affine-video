import hashlib
import importlib.util
import io
import json
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

path = (
    Path(__file__).resolve().parents[2]
    / "evidence/system-panel/operations/preserve_runtime_assets.py"
)
spec = importlib.util.spec_from_file_location("preserve_runtime_assets", path)
preservation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preservation)


def artifact(name, payload=b"frozen"):
    return {"logical_name": name, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def test_groups_cover_every_file_once_and_never_mix_encoder_phases():
    rows = [artifact(f"veda-assets/transformer/{i}", b"x" * (i + 1)) for i in range(9)]
    rows += [artifact(f"veda-assets/text_encoder/{i}", b"x" * (i + 1)) for i in range(7)]
    phases = preservation.transfer_groups(rows)
    assert phases == preservation.transfer_groups(list(reversed(rows)))
    assert [len(phase) for phase in phases] == [4, 4]
    names = [row["logical_name"] for phase in phases for group in phase for row in group]
    assert len(names) == len(set(names)) == len(rows)
    assert set(names) == {row["logical_name"] for row in rows}
    for phase, groups in enumerate(phases):
        assert all(("/text_encoder/" in row["logical_name"]) == bool(phase)
                   for group in groups for row in group)


def test_scope_is_pinned_to_the_same_manifest_count_size_and_safe_paths(tmp_path, monkeypatch):
    rows = [artifact(f"veda-assets/file-{i}", b"x") for i in range(88)]
    rows[-1]["size_bytes"] = 146703809034 - 87

    def load(values):
        raw = json.dumps({"artifacts": values}).encode()
        bootstrap = tmp_path / "bootstrap.tar.gz"
        with tarfile.open(bootstrap, "w:gz") as archive:
            info = tarfile.TarInfo("artifact_manifest.json")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        monkeypatch.setattr(preservation, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
        return preservation.read_artifacts(bootstrap)

    assert len(load(rows)) == 88
    for changed in (
        rows[:-1],
        [*rows[:-1], dict(rows[-1], logical_name=rows[0]["logical_name"])],
        [*rows[:-1], dict(rows[-1], size_bytes=1)],
        [*rows[:-1], dict(rows[-1], logical_name="veda-assets/../escape")],
    ):
        with pytest.raises(ValueError):
            load(changed)


def test_verification_is_streaming_and_fail_closed_preserving_partial_data(tmp_path):
    row = artifact("veda-assets/model")
    target = tmp_path / row["logical_name"]
    target.parent.mkdir()
    target.write_bytes(b"frozen")
    partial = target.parent / ".rsync-partial/model"
    partial.parent.mkdir()
    partial.write_bytes(b"partial")
    assert preservation.verify_artifact(tmp_path, row) == row["sha256"]
    assert target.stat().st_mode & 0o777 == 0o400
    for invalid in (dict(row, size_bytes=999), dict(row, sha256="0" * 64)):
        with pytest.raises(ValueError):
            preservation.verify_artifact(tmp_path, invalid)
    assert target.read_bytes() == b"frozen"
    assert partial.read_bytes() == b"partial"
    (tmp_path / "alias").symlink_to(target.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        preservation.destination_file(tmp_path, "alias/model")


def test_transfer_has_bounded_retries_and_preserves_rsync_resume_flags(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 123
        returncode = 12

        def __init__(self, command, **kwargs):
            calls.append(command)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def communicate(self, data):
            assert data == b"veda-assets/model\0"

    monkeypatch.setattr(preservation.subprocess, "Popen", Process)
    stopped = SimpleNamespace(is_set=lambda: False, wait=lambda _seconds: False)
    with pytest.raises(RuntimeError, match="bounded"):
        preservation.transfer_group([artifact("veda-assets/model")], tmp_path, 0, 0, stopped)
    assert len(calls) == 5
    assert "--partial" in calls[0] and "--partial-dir=.rsync-partial" in calls[0]
    assert "--bwlimit=6144" in calls[0] and "--timeout=120" in calls[0]
    assert not any("delete" in option or "inplace" in option for option in calls[0])
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="stopped"):
        preservation.transfer_group([artifact("veda-assets/model")], tmp_path, 0, 0, event)
    assert len(calls) == 5


def test_run_refuses_an_active_previous_writer(monkeypatch):
    monkeypatch.setattr(preservation.subprocess, "check_output", lambda *_a, **_k: "active\n")
    with pytest.raises(RuntimeError, match="not retired"):
        preservation.run()


def test_resume_disk_budget_excludes_completed_files_but_keeps_partial_and_reserve(
    tmp_path, monkeypatch,
):
    rows = [artifact(f"veda-assets/{name}") for name in ("complete", "partial", "missing")]
    destination = tmp_path / "mirror"
    destination.mkdir()
    completed = destination / rows[0]["logical_name"]
    completed.parent.mkdir()
    completed.write_bytes(b"frozen")
    partial = completed.parent / ".rsync-partial/partial"
    partial.parent.mkdir()
    partial.write_bytes(b"fro")
    current = tmp_path / "task.json"
    monkeypatch.setattr(preservation, "CURRENT", current)
    monkeypatch.setattr(preservation, "DESTINATION", destination)
    monkeypatch.setattr(preservation, "read_artifacts", lambda: rows)
    monkeypatch.setattr(preservation.subprocess, "check_output", lambda *_a, **_k: "inactive\n")
    monkeypatch.setattr(preservation.signal, "signal", lambda *_a: None)
    required_free = 64 * 1024**3 + 2 * len(b"frozen")
    free = SimpleNamespace(free=required_free - 1)
    monkeypatch.setattr(preservation.shutil, "disk_usage", lambda _p: free)
    calls = []

    def transfer(group, root, *_args):
        calls.append(group)
        for row in group:
            (root / row["logical_name"]).write_bytes(b"frozen")

    monkeypatch.setattr(preservation, "transfer_group", transfer)
    with pytest.raises(ValueError, match="insufficient free disk"):
        preservation.run()
    assert not calls and not current.exists()
    assert partial.read_bytes() == b"fro"
    free.free = required_free
    preservation.run()
    result = json.loads(current.read_text())
    assert result["state"] == "COMPLETED_VERIFIED"
    assert result["verified_files"] == 3  # Existing files still require their SHA check.
    assert partial.read_bytes() == b"fro"


@pytest.mark.parametrize("corrupt", [False, True])
def test_resumed_run_verifies_online_phase_before_encoder_and_never_accepts_corruption(
    tmp_path, monkeypatch, corrupt,
):
    rows = [artifact("veda-assets/transformer/model"), artifact("veda-assets/text_encoder/model")]
    current = tmp_path / "task.json"
    destination = tmp_path / "mirror"
    destination.mkdir()
    events = []
    monkeypatch.setattr(preservation, "CURRENT", current)
    monkeypatch.setattr(preservation, "DESTINATION", destination)
    monkeypatch.setattr(preservation, "read_artifacts", lambda: rows)
    monkeypatch.setattr(preservation.subprocess, "check_output", lambda *_a, **_k: "inactive\n")
    monkeypatch.setattr(preservation.shutil, "disk_usage", lambda _p: SimpleNamespace(free=10**15))
    monkeypatch.setattr(preservation.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(preservation, "emit", lambda event, **kw: events.append((event, kw)))

    def transfer(group, root, phase, index, stopped):
        if phase:
            assert json.loads(current.read_text())["verified_files"] == 1
        for row in group:
            target = root / row["logical_name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"broken" if corrupt else b"frozen")

    monkeypatch.setattr(preservation, "transfer_group", transfer)
    if corrupt:
        with pytest.raises(ValueError, match="hash mismatch"):
            preservation.run()
    else:
        preservation.run()
    result = json.loads(current.read_text())
    assert result["state"] == ("FAILED_NOT_VERIFIED" if corrupt else "COMPLETED_VERIFIED")
    assert result.get("all_selected_file_hashes_verified", False) is (not corrupt)
    assert result["verified_files"] == (0 if corrupt else 2)
    assert not list(tmp_path.glob(".preservation-*"))
