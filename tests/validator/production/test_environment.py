import json
import zipfile
from types import SimpleNamespace

import pytest

from validator.production import environment
from validator.production.commitment import canonical_json


def test_public_monitor_is_included_in_the_frozen_source(monkeypatch):
    assert "monitor" in environment._SOURCE_MODULES
    monkeypatch.setattr(environment, "_SOURCE_MODULES", ("monitor",))

    sources = environment._active_source_files()

    assert set(sources) == {"monitor.py"}
    assert sources["monitor.py"].is_file()


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


@pytest.mark.parametrize("name", [
    "cli.py", "monitor.py", "miner/submit.py", "scoring/exact.py", "validator/system_matches.py",
    "validator/release.py", "miowtion/runtime.py",
])
def test_evaluation_source_digest_and_bundle_bind_every_runtime_source(
    tmp_path, monkeypatch, name,
):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(environment, "_active_source_files", lambda: {name: source})
    environment.evaluation_source_digest.cache_clear()
    before = environment.evaluation_source_digest()
    bundle = environment.build_source_bundle(tmp_path / "frozen.zip")
    source.write_text("VALUE = 2\n")
    environment.evaluation_source_digest.cache_clear()
    assert environment.evaluation_source_digest() != before
    with pytest.raises(ValueError, match="active runtime source differs"):
        environment._verify_source_bundle(bundle, {name: source})
    environment.evaluation_source_digest.cache_clear()


def test_evaluation_environment_digest_excludes_project_distributions(monkeypatch) -> None:
    document = {
        "distributions": [
            {"content_sha256": "11" * 32, "name": "torch", "version": "1"},
            {"content_sha256": "22" * 32, "name": "veda-subnet", "version": "1"},
        ],
        "python_executable_sha256": "33" * 32,
        "python_implementation": "CPython",
        "python_version": "3.12.3",
    }
    monkeypatch.setattr(
        environment,
        "_environment_document",
        lambda *, verify_files: document,
    )
    environment.evaluation_environment_digest.cache_clear()
    first = environment.evaluation_environment_digest()
    document["distributions"][1]["content_sha256"] = "44" * 32
    environment.evaluation_environment_digest.cache_clear()
    assert environment.evaluation_environment_digest() == first
    document["distributions"][0]["content_sha256"] = "55" * 32
    environment.evaluation_environment_digest.cache_clear()
    assert environment.evaluation_environment_digest() != first
    environment.evaluation_environment_digest.cache_clear()


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

    dist_info = SimpleNamespace(version="1.0", _path=tmp_path / "example.dist-info")
    egg_info = SimpleNamespace(version="1.0", _path=tmp_path / "example.egg-info")
    monkeypatch.setattr(environment, "_distribution_record", lambda _distribution: None)
    assert environment._select_distribution("example", [egg_info, dist_info])[0] is dist_info


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


def test_import_root_hash_covers_code_missing_from_distribution_file_list(tmp_path) -> None:
    metadata = tmp_path / "example.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("Version: 1.0\n", encoding="utf-8")
    package = tmp_path / "example"
    package.mkdir()
    module = package / "runtime.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")

    first = environment._import_root_content_sha256(tmp_path)
    module.write_text("VALUE = 2\n", encoding="utf-8")

    assert environment._import_root_content_sha256(tmp_path) != first


def test_import_root_hash_ignores_generated_python_cache(tmp_path) -> None:
    package = tmp_path / "example"
    package.mkdir()
    (package / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = environment._import_root_content_sha256(tmp_path)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"generated")

    assert environment._import_root_content_sha256(tmp_path) == first


def test_dist_info_without_record_uses_metadata_tree(tmp_path) -> None:
    metadata = tmp_path / "example-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Version: 1.0\n", encoding="utf-8")

    assert environment._distribution_record(SimpleNamespace(_path=metadata)) is None


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
