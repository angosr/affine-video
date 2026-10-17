"""Immutable summaries of the final launch-time H200 stability run."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from math import fsum
from typing import Any

from scoring.spec import TrackSpec, track_spec_dict
from validator.production.commitment import ArtifactManifest, canonical_json
from validator.production.environment import (
    evaluation_environment_digest,
    evaluation_source_digest,
)
from validator.system_match_data import (
    BASELINE_SCORES,
    CONTROLS,
    EVIDENCE_SHA256,
    MINIMUM_GAIN,
    PRODUCTION_EVIDENCE_CONTRACT_DIGEST,
    PRODUCTION_EVIDENCE_RUNTIME_DIGEST,
    PRODUCTION_NETUID,
    PRODUCTION_TRACK_ID,
    REFERENCE,
    RUNTIME_TRACE_EVIDENCE_SHA256,
)


def system_match_artifacts(
    track: TrackSpec,
    runtime_manifest: bytes | None,
) -> tuple[tuple[str, bytes], ...]:
    """Return evidence only for its frozen, non-circular scoring contract."""

    if (
        runtime_manifest is None
        or track.track_id != PRODUCTION_TRACK_ID
        or track.netuid != PRODUCTION_NETUID
        or _evidence_contract_digest(track) != PRODUCTION_EVIDENCE_CONTRACT_DIGEST
        or _evidence_runtime_digest(runtime_manifest) != PRODUCTION_EVIDENCE_RUNTIME_DIGEST
    ):
        return ()
    return tuple(
        (
            f"system_matches/{track.track_id}/{sequence:012d}.json",
            _json_bytes(_match_document(sequence, control, track.digest)),
        )
        for sequence, control in enumerate(CONTROLS, start=1)
    )


def system_match_summaries(
    track: TrackSpec,
    runtime_manifest: bytes | None,
) -> list[dict[str, Any]]:
    """Return compact launch-check rows for the live status document."""

    summaries = []
    for name, payload in system_match_artifacts(track, runtime_manifest):
        document = json.loads(payload)
        summaries.append(
            {
                "sequence": document["sequence"],
                "reference": document["reference"]["label"],
                "challenger": document["challenger"]["label"],
                "reference_score": document["scores"]["reference"],
                "challenger_score": document["scores"]["challenger"],
                "outcome": document["outcome"],
                "detail": name,
            }
        )
    return summaries


def _match_document(
    sequence: int,
    control: dict[str, Any],
    track_spec_digest: str,
) -> dict[str, Any]:
    challenger_scores = control["scores"]
    batches = [
        {
            "batch": index,
            "reference_score": reference,
            "challenger_score": challenger,
            "challenger_gain_lower_bound": gain,
        }
        for index, (reference, challenger, gain) in enumerate(
            zip(
                BASELINE_SCORES,
                challenger_scores,
                control["gain_lower_bounds"],
                strict=True,
            )
        )
    ]
    reference_gains = control.get("reference_gain_lower_bounds")
    if reference_gains is not None:
        for batch, gain in zip(batches, reference_gains, strict=True):
            batch["reference_gain_lower_bound"] = gain
    return {
        "record_type": "SYSTEM_BENCHMARK",
        "sequence": sequence,
        "track_id": PRODUCTION_TRACK_ID,
        "track_spec_digest": track_spec_digest,
        "netuid": PRODUCTION_NETUID,
        "affects_tournament": False,
        "reward_eligible": False,
        "reference": REFERENCE,
        "challenger": {key: control[key] for key in ("model_id", "label", "sha256")},
        "evaluation": {
            "hardware": "NVIDIA H200",
            "torch": "2.12.0+cu130",
            "cuda": "13.0",
            "track_contract_sha256": PRODUCTION_EVIDENCE_CONTRACT_DIGEST,
            "runtime_contract_sha256": PRODUCTION_EVIDENCE_RUNTIME_DIGEST,
            "batch_count": 10,
            "cases_per_batch": 384,
            "independent_videos_per_stratum": 4,
            "confidence": 0.9975,
            "minimum_gain": MINIMUM_GAIN,
            "cluster_unit": "source_video",
            "evidence_sha256": EVIDENCE_SHA256,
            "runtime_trace": {
                "evidence_sha256": RUNTIME_TRACE_EVIDENCE_SHA256,
                "cases": 16,
                "sampled_routes": 8,
                "descriptor_max_abs": 2.384185791015625e-07,
                "logits_max_abs": 6.198883056640625e-06,
                "mask_mismatches": 0,
                "output_max_abs": 0.0625,
                "output_mean_abs": 3.039836883544922e-06,
            },
            "batches": batches,
        },
        "scores": {
            "reference": fsum(BASELINE_SCORES) / len(BASELINE_SCORES),
            "challenger": fsum(challenger_scores) / len(challenger_scores),
        },
        "outcome": control["outcome"],
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _evidence_contract_digest(track: TrackSpec) -> str:
    """Bind score-affecting track data without depending on its runtime manifest."""

    document = track_spec_dict(track)
    del document["artifact_manifest_sha256"]
    return hashlib.sha256(_json_bytes(document)).hexdigest()


@lru_cache(maxsize=1)
def _evidence_runtime_digest(runtime_manifest: bytes) -> str:
    """Bind benchmark evidence to scoring semantics without binding release packaging."""

    manifest = ArtifactManifest.decode(runtime_manifest)
    artifacts = [
        {
            "logical_name": artifact.logical_name,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in manifest.artifacts
        if not _is_release_packaging(artifact.logical_name)
    ]
    if not artifacts:
        raise ValueError("evidence runtime asset contract cannot be empty")
    document = {
        "artifacts": artifacts,
        "environment_sha256": evaluation_environment_digest(),
        "format": "veda-evidence-runtime-contract-v1",
        "source_sha256": evaluation_source_digest(),
    }
    return hashlib.sha256(canonical_json(document)).hexdigest()


def _is_release_packaging(logical_name: str) -> bool:
    packaging = any(
        logical_name == name or logical_name.endswith("/" + name)
        for name in (
            "runtime/dependencies.lock",
            "runtime/image.tar",
            "runtime/source.zip",
        )
    )
    launch_evidence = logical_name.startswith("evidence/system-panel/")
    return packaging or launch_evidence
