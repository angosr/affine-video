"""Coordinate the public track identity without coupling startup to research panels."""

from __future__ import annotations

import os
from dataclasses import dataclass

from scoring.spec import TrackSpec
from validator.production.commitment import ArtifactManifest

PRODUCTION_TRACK_ID = "veda-sn120-v2"
PRODUCTION_NETUID = 120


@dataclass(frozen=True, slots=True)
class ReleaseApproval:
    coordinated_track_digest: str

    @classmethod
    def from_environment(cls) -> ReleaseApproval | None:
        digest = os.environ.get("VEDA_COORDINATED_RELEASE_DIGEST")
        if digest is None:
            return None
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("coordinated release digest must be lowercase SHA-256")
        return cls(digest)

    def verify(self, track: TrackSpec, runtime_manifest: bytes | None) -> None:
        if self.coordinated_track_digest != track.digest:
            raise ValueError("candidate digest has not been coordinated for formal publication")
        if runtime_manifest is None:
            raise ValueError("formal release requires its runtime manifest")
        manifest = ArtifactManifest.decode(runtime_manifest)
        if manifest.digest != track.artifact_manifest_sha256:
            raise ValueError("formal release runtime manifest differs from candidate")


def require_release_approval(
    track: TrackSpec, runtime_manifest: bytes | None, approval: ReleaseApproval | None
) -> None:
    if approval is None:
        raise ValueError("formal publication requires a coordinated track digest")
    approval.verify(track, runtime_manifest)
