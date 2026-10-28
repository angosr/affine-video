"""Require runtime-bound acceptance before a formal release can publish."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from scoring.spec import (
    TrackSpec,
    data_contract_digest,
    data_spec_logical_name,
    parse_track_spec,
    validate_data_track,
)
from validator import system_matches
from validator.production.commitment import ArtifactManifest, decode_canonical_json
from validator.system_matches import (
    _evidence_contract_digest,
    _evidence_runtime_digest,
    _is_release_packaging,
)

PRODUCTION_TRACK_ID = "veda-sn120-v2"
PRODUCTION_NETUID = 120
ACCEPTANCE_LOGICAL_NAME = "evidence/system-panel/acceptance/accepted.json"
# Numerical fidelity, real two-stage GPU execution, and deterministic/error-safe
# behavior are release requirements. Broader research is not a release gate.
REQUIRED_EXPERIMENTS = frozenset({"E2", "E3", "E16"})
_MAX_ACCEPTANCE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReleaseApproval:
    coordinated_track_digest: str
    acceptance: bytes
    competition_manifest: bytes | None = None

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

    def verify(
        self, track: TrackSpec, runtime_manifest: bytes | None, *, image_id: str | None = None,
    ) -> None:
        if self.coordinated_track_digest != track.digest:
            raise ValueError("candidate digest has not been coordinated for formal publication")
        if runtime_manifest is None:
            raise ValueError("formal release requires its runtime manifest")
        manifest = ArtifactManifest.decode(runtime_manifest)
        competition = verify_competition_manifest(
            track, runtime_manifest, self.competition_manifest,
        )
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
        if track.data_spec_sha256 is not None:
            origin = artifacts.get(data_spec_logical_name(track))
            if (origin is None or origin.sha256 != track.data_spec_sha256
                    or origin.object_key != f"artifacts/{track.data_spec_sha256}"
                    or result.get("data_spec_sha256") != track.data_spec_sha256
                    or result.get("data_contract_sha256") != data_contract_digest(track)):
                raise ValueError("release acceptance does not bind the original data contract")
        if manifest.digest != competition.digest and (
            result.get("track_spec_digest") != track.digest
            or result.get("competition_manifest_sha256") != competition.digest
            or result.get("source_sha256") != system_matches.evaluation_source_digest()
            or result.get("environment_sha256") != system_matches.evaluation_environment_digest()
            or not _image_id(result.get("image_id"))
        ):
            raise ValueError("runtime release is not bound to this competition and installation")
        if image_id is not None and result.get("image_id") != image_id:
            raise ValueError("release acceptance differs from the OCI descriptor")
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

    def public_identity(self, track: TrackSpec, runtime_manifest: bytes) -> dict[str, object]:
        """Describe the verified running release, not a new competition identity."""
        self.verify(track, runtime_manifest)
        result = decode_canonical_json(self.acceptance)
        identity = {
            "track_spec_digest": track.digest,
            "competition_manifest_sha256": track.artifact_manifest_sha256,
            "artifact_manifest_sha256": hashlib.sha256(runtime_manifest).hexdigest(),
            "acceptance_sha256": hashlib.sha256(self.acceptance).hexdigest(),
            "image_id": result.get("image_id"),
            "source_sha256": result.get("source_sha256"),
            "environment_sha256": result.get("environment_sha256"),
        }
        if track.data_spec_sha256 is not None:
            identity.update(data_spec_sha256=track.data_spec_sha256,
                            data_contract_sha256=result["data_contract_sha256"])
        return identity


def verify_data_binding(
    track: TrackSpec, runtime_manifest: bytes, data_spec: bytes | None,
    data_manifest: bytes,
) -> TrackSpec:
    """Authenticate one original data specification without rewriting any plan."""
    from scoring.spec import track_spec_dict
    from validator.production.commitment import canonical_json

    data = track if data_spec is None else parse_track_spec(decode_canonical_json(data_spec))
    validate_data_track(track, data)
    original = ArtifactManifest.decode(data_manifest)
    runtime = ArtifactManifest.decode(runtime_manifest)
    if original.digest != data.artifact_manifest_sha256:
        raise ValueError("original data manifest differs from its specification")
    if tuple(a for a in original.artifacts if not _is_release_packaging(a.logical_name)) != tuple(
        a for a in runtime.artifacts if not _is_release_packaging(a.logical_name)
    ):
        raise ValueError("runtime changes original data assets")
    if track.data_spec_sha256 is not None:
        if data_spec is None or data_spec != canonical_json(track_spec_dict(data)):
            raise ValueError("original data specification is not canonical")
        origin = next((a for a in runtime.artifacts
                       if a.logical_name == data_spec_logical_name(track)), None)
        if (origin is None or origin.sha256 != data.digest
                or origin.object_key != f"artifacts/{data.digest}"
                or origin.size_bytes != len(data_spec)):
            raise ValueError("manifest does not bind the original data specification")
    elif data_spec is not None:
        raise ValueError("data origin is not committed by the active rules")
    return data


def verify_competition_manifest(
    track: TrackSpec, runtime_manifest: bytes, competition_manifest: bytes | None = None,
) -> ArtifactManifest:
    """Keep the original track commitment; only runtime packaging may be replaced.

    The pinned original manifest also authenticates historical references and
    READY data. It is never replaced by a new executable package in bootstrap.
    """
    runtime = ArtifactManifest.decode(runtime_manifest)
    competition = ArtifactManifest.decode(
        runtime_manifest if competition_manifest is None else competition_manifest,
    )
    if competition.digest != track.artifact_manifest_sha256:
        raise ValueError("formal release runtime manifest differs from candidate competition")
    def assets(manifest: ArtifactManifest):
        return tuple(artifact for artifact in manifest.artifacts
                     if not _is_release_packaging(artifact.logical_name))
    if assets(runtime) != assets(competition):
        raise ValueError("compatible runtime release changes committed competition assets")
    return competition


def _image_id(value: object) -> bool:
    return (isinstance(value, str) and value.startswith("sha256:") and len(value) == 71
            and all(char in "0123456789abcdef" for char in value[7:]))


def require_release_approval(
    track: TrackSpec, runtime_manifest: bytes | None, approval: ReleaseApproval | None,
    *, image_id: str | None = None,
) -> None:
    if approval is None:
        raise ValueError(
            "formal publication requires verified acceptance and coordinated track digest"
        )
    approval.verify(track, runtime_manifest, image_id=image_id)
