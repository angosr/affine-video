"""Fail-closed authorization for changing the formal public track."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from scoring.spec import TrackSpec
from validator.production.commitment import ArtifactManifest, decode_canonical_json
from validator.system_match_data import PRODUCTION_ACCEPTANCE_SHA256
from validator.system_matches import (
    _evidence_contract_digest,
    _evidence_runtime_digest,
    system_match_artifacts,
)

ACCEPTANCE_LOGICAL_NAME = "evidence/system-panel/acceptance/accepted.json"
# Launch requires correct execution, bounded cost, isolated data and recovery.
# Broader quality/power studies remain visible in the acceptance inventory but
# are not prerequisites for the initial release or claims of proven coverage.
_REQUIRED_EXPERIMENTS = frozenset(
    f"E{index}" for index in (1, 2, 3, 5, 6, 10, 11, 14, 16, 18)
)
LEGACY_PUBLIC_IDENTITY = (
    "veda-sn120-v1",
    "0fcae991aef29f87b97fbb07cf79c49e5eb9cc2a5d9aecdd2a566443e2b81dd5",
)


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
        return cls(digest, Path(path).read_bytes())

    def verify(self, track: TrackSpec, runtime_manifest: bytes | None) -> None:
        if self.coordinated_track_digest != track.digest:
            raise ValueError("candidate digest has not been coordinated for formal publication")
        if runtime_manifest is None:
            raise ValueError("formal release requires its runtime manifest")
        manifest = ArtifactManifest.decode(runtime_manifest)
        if manifest.digest != track.artifact_manifest_sha256:
            raise ValueError("formal release runtime manifest differs from candidate")
        digest = hashlib.sha256(self.acceptance).hexdigest()
        if digest != PRODUCTION_ACCEPTANCE_SHA256 or digest == "0" * 64:
            raise ValueError("release acceptance is not the verified frozen acceptance result")
        artifacts = {artifact.logical_name: artifact for artifact in manifest.artifacts}
        accepted = artifacts.get(ACCEPTANCE_LOGICAL_NAME)
        if (
            accepted is None
            or accepted.sha256 != digest
            or accepted.size_bytes != len(self.acceptance)
        ):
            raise ValueError("release manifest does not bind its acceptance result")
        result = decode_canonical_json(self.acceptance)
        if (
            not isinstance(result, dict)
            or result.get("format") != "veda-release-acceptance-v1"
            or result.get("release_accepted") is not True
            or result.get("evidence_contract_sha256") != _evidence_contract_digest(track)
            or result.get("runtime_contract_sha256") != _evidence_runtime_digest(runtime_manifest)
        ):
            raise ValueError(
                "release acceptance does not match active scoring and runtime contracts"
            )
        if not system_match_artifacts(track, runtime_manifest):
            raise ValueError("formal release has no bound system match artifacts")
        experiments = result.get("experiments")
        if not isinstance(experiments, dict) or set(experiments) != _REQUIRED_EXPERIMENTS:
            raise ValueError("release acceptance is missing mandatory design experiments")
        for identifier, experiment in experiments.items():
            if (
                not isinstance(experiment, dict)
                or experiment.get("pass") is not True
                or not experiment.get("method")
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
                    or reference.get("sha256") != artifact.sha256
                ):
                    raise ValueError(f"release evidence is not bound by the manifest: {identifier}")


def require_release_approval(
    track: TrackSpec, runtime_manifest: bytes | None, approval: ReleaseApproval | None
) -> None:
    if (track.track_id, track.digest) == LEGACY_PUBLIC_IDENTITY:
        return
    if approval is None:
        raise ValueError("formal publication requires verified acceptance and coordinated digest")
    approval.verify(track, runtime_manifest)
