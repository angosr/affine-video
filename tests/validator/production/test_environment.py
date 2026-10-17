import json
import zipfile
from types import SimpleNamespace

import pytest

from validator.production import environment
from validator.production.commitment import canonical_json


def test_source_bundle_detects_active_source_changes(tmp_path) -> None:
    source = tmp_path / "active" / "example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(source, "example/example.py")

    environment._verify_source_bundle(bundle, {"example/example.py": source})

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="active runtime source differs"):
        environment._verify_source_bundle(bundle, {"example/example.py": source})


def test_source_bundle_requires_the_exact_active_file_set(tmp_path) -> None:
    source = tmp_path / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("example/other.py", b"VALUE = 1\n")

    with pytest.raises(ValueError, match="active source file set"):
        environment._verify_source_bundle(bundle, {"example/example.py": source})


def test_source_bundle_builder_is_deterministic(tmp_path, monkeypatch) -> None:
    source = tmp_path / "active" / "example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        environment, "_active_source_files", lambda: {"example/example.py": source}
    )

    first = environment.build_source_bundle(tmp_path / "first.zip")
    second = environment.build_source_bundle(tmp_path / "second.zip")

    assert first.read_bytes() == second.read_bytes()
    environment._verify_source_bundle(first, {"example/example.py": source})
    with pytest.raises(FileExistsError):
        environment.build_source_bundle(first)


def test_runtime_environment_rejects_a_changed_dependency_lock(tmp_path, monkeypatch) -> None:
    source = tmp_path / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(source, "example/example.py")
    lock = tmp_path / "dependencies.lock"
    current = {
        "distributions": [],
        "format": "veda-python-environment-lock-v1",
        "python_executable_sha256": "a" * 64,
        "python_implementation": "CPython",
        "python_version": "3.12.3",
    }
    lock.write_bytes(canonical_json(current))
    monkeypatch.setattr(
        environment, "_active_source_files", lambda: {"example/example.py": source}
    )
    monkeypatch.setattr(
        environment, "_environment_document", lambda *, verify_files: current
    )

    environment.verify_runtime_environment(bundle, lock)

    changed = {**current, "python_version": "3.12.4"}
    monkeypatch.setattr(
        environment, "_environment_document", lambda *, verify_files: changed
    )
    with pytest.raises(ValueError, match="dependency lock"):
        environment.verify_runtime_environment(bundle, lock)


def test_dependency_lock_writer_refuses_overwrite(tmp_path, monkeypatch) -> None:
    document = {
        "distributions": [],
        "format": "veda-python-environment-lock-v1",
        "python_executable_sha256": "a" * 64,
        "python_implementation": "CPython",
        "python_version": "3.12.3",
    }
    monkeypatch.setattr(
        environment, "_environment_document", lambda *, verify_files: document
    )
    target = environment.write_dependency_lock(tmp_path / "runtime" / "dependencies.lock")

    assert target.read_bytes() == canonical_json(document)
    with pytest.raises(FileExistsError):
        environment.write_dependency_lock(target)


def test_dependency_selection_uses_active_python_path_precedence(
    tmp_path, monkeypatch
) -> None:
    record = tmp_path / "RECORD"
    record.write_text("committed", encoding="utf-8")
    legacy = SimpleNamespace(version="1.0")
    installed = SimpleNamespace(version="1.0")
    monkeypatch.setattr(
        environment,
        "_distribution_record",
        lambda distribution: record if distribution is installed else None,
    )
    monkeypatch.setattr(
        environment,
        "_distribution_priority",
        lambda distribution: 2 if distribution is installed else 8,
    )

    assert environment._select_distribution("example", [legacy, installed]) == (
        installed,
        record,
    )

    monkeypatch.setattr(environment, "_distribution_priority", lambda _distribution: 2)
    with pytest.raises(ValueError, match="ambiguous"):
        environment._select_distribution("example", [legacy, installed])


def test_dependency_tree_hashes_metadata_when_no_file_list_exists(tmp_path) -> None:
    metadata = tmp_path / "example.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("Version: 1.0\n", encoding="utf-8")
    distribution = SimpleNamespace(
        files=None,
        metadata={"Name": "example"},
        _path=metadata,
    )

    first = environment._distribution_content_sha256(distribution)
    (metadata / "PKG-INFO").write_text("Version: 2.0\n", encoding="utf-8")

    assert environment._distribution_content_sha256(distribution) != first


def test_dependency_lock_must_be_canonical(tmp_path, monkeypatch) -> None:
    source = tmp_path / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(source, "example/example.py")
    lock = tmp_path / "dependencies.lock"
    lock.write_text(json.dumps({"format": "wrong"}, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        environment, "_active_source_files", lambda: {"example/example.py": source}
    )

    with pytest.raises(ValueError, match="canonical"):
        environment.verify_runtime_environment(bundle, lock)
