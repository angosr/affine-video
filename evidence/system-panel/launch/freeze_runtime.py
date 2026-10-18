"""Assemble restorable runtime provenance inside its exact OCI environment.

This writes an intermediate candidate, never a formal launch authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)


def freeze(root, release, parent_manifest, parent_track):
    from scoring.spec import parse_track_spec
    from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
    from validator.production.environment import verify_runtime_environment
    from validator.system_matches import _evidence_contract_digest, _evidence_runtime_digest

    source = release / "runtime/source.zip"
    lock = release / "runtime/dependencies.lock"
    verify_runtime_environment(source, lock)
    parent = ArtifactManifest.decode(parent_manifest.read_bytes())
    artifacts = []
    replaced = set()
    for artifact in parent.artifacts:
        name = next(
            (
                name
                for name in ("source.zip", "dependencies.lock", "image.tar")
                if artifact.logical_name.endswith("/runtime/" + name)
            ),
            None,
        )
        if name is None:
            artifacts.append(artifact)
            continue
        if name in replaced:
            raise ValueError("parent manifest contains ambiguous runtime packaging")
        replaced.add(name)
        path = release / "runtime" / name
        digest = sha256(path)
        artifacts.append(
            Artifact(
                path.relative_to(root).as_posix(),
                digest,
                path.stat().st_size,
                f"artifacts/{digest}",
            )
        )
    if replaced != {"source.zip", "dependencies.lock", "image.tar"}:
        raise ValueError("parent manifest omits required runtime artifacts")
    manifest = ArtifactManifest(tuple(sorted(artifacts, key=lambda item: item.logical_name)))
    document = json.loads(parent_track.read_text())
    document["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(document)
    report = {
        "format": "veda-runtime-freeze-provenance-v1",
        "release_accepted": False,
        "status": "intermediate_until_acceptance_and_coordination",
        "parent_manifest_sha256": parent.digest,
        "candidate_track_spec_digest": track.digest,
        "artifact_manifest_sha256": manifest.digest,
        "evidence_contract_sha256": _evidence_contract_digest(track),
        "runtime_contract_sha256": _evidence_runtime_digest(manifest.encode()),
        "script_sha256": sha256(Path(__file__)),
        "runtime_artifacts": [
            item.document()
            for item in manifest.artifacts
            if item.logical_name.startswith(release.relative_to(root).as_posix())
        ],
        "inherited_assets": "unchanged hashes from parent; verify bytes at production startup",
    }
    write_once(release / "artifact_manifest.json", manifest.encode())
    write_once(release / "track.candidate.json", canonical_json(document))
    write_once(release / "freeze-provenance.json", canonical_json(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ("root", "release", "parent-manifest", "parent-track"):
        parser.add_argument("--" + argument, type=Path, required=True)
    arguments = parser.parse_args()
    freeze(arguments.root, arguments.release, arguments.parent_manifest, arguments.parent_track)
