"""Require runtime-bound acceptance before a formal release can publish."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from scoring.spec import TrackSpec
from validator.production.commitment import ArtifactManifest, decode_canonical_json
from validator.system_matches import _evidence_contract_digest, _evidence_runtime_digest

PRODUCTION_TRACK_ID = "veda-sn120-v2"
PRODUCTION_NETUID = 120
ACCEPTANCE_LOGICAL_NAME = "evidence/system-panel/acceptance/accepted.json"
# Numerical fidelity, real two-stage GPU execution, and deterministic/error-safe
# behavior are release requirements. Broader research stays in the inventory.
REQUIRED_EXPERIMENTS = frozenset({"E2", "E3", "E16"})
_MAX_ACCEPTANCE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReleaseApproval:
    coordinated_track_digest: str
    acceptance: bytes

    @classmethod
    def from_environment(cls) -> ReleaseApproval | None:
        digest = os.environ.get("VEDA_COORDINATED_RELEASE_DIGEST")
        path = os.environ.get("VEDA_RELEASE_ACCEPTANCE")
        if digest is None and path is None:
            return None
        if not digest or not path:
            raise ValueError("coordinated release digest and acceptance manifest are both required")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("coordinated release digest must be lowercase SHA-256")
        with Path(path).open("rb") as stream:
            acceptance = stream.read(_MAX_ACCEPTANCE_BYTES + 1)
        if not acceptance or len(acceptance) > _MAX_ACCEPTANCE_BYTES:
            raise ValueError("release acceptance has an invalid size")
        return cls(digest, acceptance)

    def verify(self, track: TrackSpec, runtime_manifest: bytes | None) -> None:
        if self.coordinated_track_digest != track.digest:
            raise ValueError("candidate digest has not been coordinated for formal publication")
        if runtime_manifest is None:
            raise ValueError("formal release requires its runtime manifest")
        manifest = ArtifactManifest.decode(runtime_manifest)
        if manifest.digest != track.artifact_manifest_sha256:
            raise ValueError("formal release runtime manifest differs from candidate")
        if not isinstance(self.acceptance, bytes) or not 0 < len(
            self.acceptance
        ) <= _MAX_ACCEPTANCE_BYTES:
            raise ValueError("release acceptance has an invalid size")
        artifacts = {artifact.logical_name: artifact for artifact in manifest.artifacts}
        accepted = artifacts.get(ACCEPTANCE_LOGICAL_NAME)
        if (
            accepted is None
            or accepted.sha256 != hashlib.sha256(self.acceptance).hexdigest()
            or accepted.size_bytes != len(self.acceptance)
        ):
            raise ValueError("runtime manifest does not bind the release acceptance")
        result = decode_canonical_json(self.acceptance)
        if (
            not isinstance(result, dict)
            or result.get("format") != "veda-release-acceptance-v1"
            or result.get("release_accepted") is not True
            or result.get("evidence_contract_sha256") != _evidence_contract_digest(track)
            or result.get("runtime_contract_sha256") != _evidence_runtime_digest(runtime_manifest)
        ):
            raise ValueError("release acceptance differs from active scoring or runtime contracts")
        experiments = result.get("experiments")
        if not isinstance(experiments, dict) or set(experiments) != REQUIRED_EXPERIMENTS:
            raise ValueError("release acceptance is missing mandatory experiments")
        for identifier, experiment in experiments.items():
            if (
                not isinstance(experiment, dict)
                or experiment.get("pass") is not True
                or not isinstance(experiment.get("method"), str)
                or not experiment["method"].strip()
                or not isinstance(experiment.get("actual_values"), dict)
                or not experiment["actual_values"]
                or not isinstance(experiment.get("raw_evidence"), list)
                or not experiment["raw_evidence"]
            ):
                raise ValueError(f"release experiment is incomplete or failed: {identifier}")
            for reference in [experiment.get("acceptance_manifest"), *experiment["raw_evidence"]]:
                if not isinstance(reference, dict):
                    raise ValueError(f"release experiment has no bound evidence: {identifier}")
                name = reference.get("logical_name")
                artifact = artifacts.get(name) if isinstance(name, str) else None
                if (
                    artifact is None
                    or not name.startswith("evidence/system-panel/")
                    or name == ACCEPTANCE_LOGICAL_NAME
                    or artifact.sha256 == "0" * 64
                    or reference.get("sha256") != artifact.sha256
                ):
                    raise ValueError(f"release evidence is not bound by the manifest: {identifier}")


def require_release_approval(
    track: TrackSpec, runtime_manifest: bytes | None, approval: ReleaseApproval | None
) -> None:
    if approval is None:
        raise ValueError(
            "formal publication requires verified acceptance and coordinated track digest"
        )
    approval.verify(track, runtime_manifest)
