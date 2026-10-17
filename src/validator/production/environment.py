"""Bind the active Python runtime to the committed release artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from validator.production.commitment import canonical_json, decode_canonical_json

_SOURCE_MODULES = ("cli", "miner", "scoring", "validator", "miowtion")
_DISTRIBUTION_NAME = re.compile(r"[-_.]+")


def verify_runtime_environment(
    source_bundle: str | Path,
    dependency_lock: str | Path,
) -> None:
    """Reject ambient source or dependency bytes outside the frozen artifacts."""

    _verify_source_bundle(Path(source_bundle), _active_source_files())
    payload = Path(dependency_lock).read_bytes()
    expected = decode_canonical_json(payload)
    if expected != _environment_document(verify_files=True):
        raise ValueError("active Python environment does not match the dependency lock")


def build_dependency_lock() -> bytes:
    """Return the canonical lock for the currently executing Python environment."""

    return canonical_json(_environment_document(verify_files=True))


def write_dependency_lock(destination: str | Path) -> Path:
    """Write the canonical lock without replacing an existing commitment."""

    payload = build_dependency_lock()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def build_source_bundle(destination: str | Path) -> Path:
    """Write the deterministic ZIP containing the active scoring runtime source."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, path in sorted(_active_source_files().items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return target


def _active_source_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for module_name in _SOURCE_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ValueError(f"committed runtime module is unavailable: {module_name}")
        locations = tuple(spec.submodule_search_locations or ())
        if locations:
            if len(locations) != 1:
                raise ValueError(f"runtime package has ambiguous roots: {module_name}")
            root = Path(locations[0]).resolve()
            for path in sorted(root.rglob("*")):
                if not _runtime_source_file(path, root):
                    continue
                relative = path.relative_to(root).as_posix()
                files[f"{module_name.replace('.', '/')}/{relative}"] = path
        else:
            if spec.origin is None:
                raise ValueError(f"runtime module has no source origin: {module_name}")
            path = Path(spec.origin).resolve()
            if not path.is_file():
                raise ValueError(f"runtime module source is unavailable: {module_name}")
            files[f"{module_name.replace('.', '/')}{path.suffix}"] = path
    return files


def _runtime_source_file(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(root)
    return "__pycache__" not in relative.parts and path.suffix not in {".pyc", ".pyo"}


def _verify_source_bundle(bundle: Path, active: dict[str, Path]) -> None:
    if not active:
        raise ValueError("active runtime source set cannot be empty")
    try:
        archive = zipfile.ZipFile(bundle)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("runtime source bundle is not a valid ZIP archive") from exc
    with archive:
        entries: dict[str, zipfile.ZipInfo] = {}
        for item in archive.infolist():
            if item.is_dir():
                continue
            name = item.filename
            path = Path(name)
            if (
                path.is_absolute()
                or name != path.as_posix()
                or any(part in {"", ".", ".."} for part in path.parts)
                or name in entries
            ):
                raise ValueError("runtime source bundle contains an unsafe or duplicate path")
            entries[name] = item
        active_prefixes = {name.split("/", 1)[0] for name in active}
        committed = {
            name
            for name in entries
            if name.split("/", 1)[0] in active_prefixes
            and "__pycache__" not in Path(name).parts
            and Path(name).suffix not in {".pyc", ".pyo"}
        }
        if committed != set(active):
            raise ValueError("runtime source bundle does not match the active source file set")
        for name, path in active.items():
            expected = _file_sha256(path)
            with archive.open(entries[name]) as stream:
                actual = _stream_sha256(stream)
            if actual != expected:
                raise ValueError(f"active runtime source differs from the bundle: {name}")


def _environment_document(*, verify_files: bool) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    distributions = []
    grouped: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("installed distribution has no canonical name")
        name = _normalize_distribution_name(raw_name)
        grouped.setdefault(name, []).append(distribution)
    for name, candidates in grouped.items():
        distribution, record = _select_distribution(name, candidates)
        version = distribution.version
        if not isinstance(version, str) or not version:
            raise ValueError(f"installed distribution has no version: {name}")
        content_sha256 = (
            _distribution_content_sha256(distribution) if verify_files else "0" * 64
        )
        distributions.append(
            {
                "content_sha256": content_sha256,
                "name": name,
                "record_sha256": None if record is None else _file_sha256(record),
                "version": version,
            }
        )
    distributions.sort(key=lambda item: item["name"])
    return {
        "distributions": distributions,
        "format": "veda-python-environment-lock-v1",
        "python_executable_sha256": _file_sha256(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def _select_distribution(
    name: str,
    candidates: list[importlib.metadata.Distribution],
) -> tuple[importlib.metadata.Distribution, Path | None]:
    ranked = sorted(
        ((_distribution_priority(distribution), distribution) for distribution in candidates),
        key=lambda item: item[0],
    )
    if not ranked or ranked[0][0] >= len(sys.path):
        raise ValueError(f"installed distribution is outside the active Python path: {name}")
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise ValueError(f"installed distribution identity is ambiguous: {name}")
    selected = ranked[0][1]
    return selected, _distribution_record(selected)


def _distribution_priority(distribution: importlib.metadata.Distribution) -> int:
    metadata_path = getattr(distribution, "_path", None)
    if metadata_path is None:
        return len(sys.path)
    root = Path(metadata_path).resolve().parent
    for index, entry in enumerate(sys.path):
        try:
            active = Path(entry or ".").resolve()
        except OSError:
            continue
        if active == root:
            return index
    return len(sys.path)


def _distribution_record(distribution: importlib.metadata.Distribution) -> Path | None:
    metadata_path = getattr(distribution, "_path", None)
    if metadata_path is None:
        return None
    root = Path(metadata_path).resolve()
    if not root.name.endswith(".dist-info"):
        return None
    path = root / "RECORD"
    if not path.is_file():
        raise ValueError("installed distribution RECORD is unavailable")
    return path


def _distribution_content_sha256(distribution: importlib.metadata.Distribution) -> str:
    """Hash the installed bytes, not only the wheel's self-reported RECORD."""

    files = sorted(distribution.files or (), key=str)
    if not files:
        name = distribution.metadata.get("Name", "unknown")
        raise ValueError(f"installed distribution has no recorded files: {name}")
    digest = hashlib.sha256(b"veda-installed-distribution-v1\0")
    checked = 0
    for item in files:
        path = Path(distribution.locate_file(item)).resolve()
        if not path.is_file():
            raise ValueError(f"installed dependency file is unavailable: {item}")
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
        checked += 1
    if checked == 0:
        name = distribution.metadata.get("Name", "unknown")
        raise ValueError(f"installed distribution has no files: {name}")
    return digest.hexdigest()


def _normalize_distribution_name(value: str) -> str:
    return _DISTRIBUTION_NAME.sub("-", value).lower()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _stream_sha256(stream)


def _stream_sha256(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
