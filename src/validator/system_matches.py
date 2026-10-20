"""Immutable, phase-specific summaries attached only to their measured runtime."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from math import fsum, isfinite
from typing import Any

from scoring.spec import TrackSpec, track_spec_dict
from validator.production.commitment import ArtifactManifest, canonical_json
from validator.production.environment import (
    evaluation_environment_digest,
    evaluation_source_digest,
)

_LOG = logging.getLogger(__name__)
MAX_SYSTEM_MATCHES = 64


def system_match_artifacts(
    track: TrackSpec,
    runtime_manifest: bytes | None,
    records: tuple[dict[str, Any], ...] = (),
) -> tuple[tuple[str, bytes], ...]:
    """Never relabel old measurements or renumber records on a packaging rebuild."""

    if runtime_manifest is None:
        return ()
    namespace = [record for record in records if record["track_id"] == track.track_id]
    sequences = [record["sequence"] for record in namespace]
    if any(
        type(sequence) is not int or not 1 <= sequence <= MAX_SYSTEM_MATCHES
        for sequence in sequences
    ) or sequences != sorted(set(sequences)):
        raise ValueError("system benchmark sequences must be unique and increasing")
    contract = _evidence_contract_digest(track)
    candidates = [
        record
        for record in namespace
        if record["netuid"] == track.netuid and record["track_contract_sha256"] == contract
    ]
    if not candidates:
        return ()
    # Reject historical source bytes before any expensive environment scan.
    source = evaluation_source_digest()
    candidates = [record for record in candidates if record["source_sha256"] == source]
    if not candidates:
        return ()
    environment = evaluation_environment_digest()
    candidates = [record for record in candidates if record["environment_sha256"] == environment]
    if not candidates:
        return ()
    for record in candidates:
        minimum_gain = (
            track.full_video.minimum_gain
            if record["phase"] == "full_video"
            else track.scoring.minimum_frontier_gain
        )
        if (
            record["confidence"] != track.scoring.match_confidence
            or record["minimum_gain"] != minimum_gain
            or tuple(record["budgets"])
            != tuple(budget.target_blocks for budget in track.scoring.budgets)
        ):
            raise ValueError("system benchmark context differs from its track contract")
    runtime = _evidence_runtime_digest(runtime_manifest)
    return tuple(
        (
            f"system_matches/{record['track_id']}/{record['sequence']:012d}.json",
            _json_bytes(_match_document(record)),
        )
        for record in candidates
        if record["runtime_contract_sha256"] == runtime
    )


def validate_system_match_artifact(
    track: TrackSpec,
    runtime_manifest: bytes | None,
    name: str,
    payload: bytes,
) -> tuple[str, bytes]:
    """Verify the original JSON body through the same constructor and runtime gate."""

    if not isinstance(payload, bytes) or not payload or len(payload) > 1024 * 1024:
        raise ValueError("system benchmark payload size is invalid")
    document = json.loads(payload)
    evaluation = document["evaluation"]
    record = {
        **evaluation,
        **{
            key: document[key]
            for key in (
                "sequence",
                "track_id",
                "track_spec_digest",
                "netuid",
                "reference",
                "challenger",
            )
        },
        "batches": tuple(
            {key: value for key, value in batch.items() if key != "batch"}
            for batch in evaluation["batches"]
        ),
    }
    if _json_bytes(_match_document(record)) != payload:
        raise ValueError("system benchmark is not the canonical measured record")
    artifacts = system_match_artifacts(track, runtime_manifest, (record,))
    if artifacts != ((name, payload),):
        raise ValueError("system benchmark identity or measured runtime differs")
    return name, payload


def validated_system_matches(
    track: TrackSpec,
    runtime_manifest: bytes | None,
    artifacts: tuple[tuple[str, bytes], ...],
) -> tuple[tuple[str, bytes], ...]:
    """Informational evidence must never fail a real competition publication."""

    verified = {}
    for name, payload in artifacts:
        try:
            validate_system_match_artifact(track, runtime_manifest, name, payload)
            if name in verified and verified[name] != payload:
                raise ValueError("system benchmark sequence has conflicting bodies")
            verified[name] = payload
        except Exception as exc:
            _LOG.warning("system benchmark skipped: %s (%s)", name, exc)
    return tuple(sorted(verified.items()))


def system_match_summaries(
    artifacts: tuple[tuple[str, bytes], ...],
) -> list[dict[str, Any]]:
    """Return compact status rows from already verified immutable artifacts."""

    summaries = []
    for name, payload in artifacts:
        document = json.loads(payload)
        phase = document["evaluation"]["phase"]
        summaries.append(
            {
                "sequence": document["sequence"],
                "reference": document["reference"]["label"],
                "challenger": f"{document['challenger']['label']} ({phase})",
                "reference_score": document["scores"]["reference"],
                "challenger_score": document["scores"]["challenger"],
                "outcome": document["outcome"],
                "detail": name,
            }
        )
    return summaries


def _match_document(record: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(record["track_id"], str)
        or not record["track_id"]
        or type(record["netuid"]) is not int
        or record["netuid"] < 0
    ):
        raise ValueError("system benchmark track identity is invalid")
    phase = record["phase"]
    batches = record["batches"]
    if phase not in {"exact", "full_video"} or not batches:
        raise ValueError("system benchmark requires a measured phase and non-empty cohorts")
    digest_fields = (
        "track_spec_digest",
        "track_contract_sha256",
        "runtime_contract_sha256",
        "source_sha256",
        "environment_sha256",
        "image_sha256",
    )
    for field in digest_fields:
        _require_digest(record[field])
    for role in ("reference", "challenger"):
        model = record[role]
        _require_digest(model["sha256"])
        _require_digest(model["model_digest"])
        if not all(isinstance(model[key], str) and model[key] for key in ("model_id", "label")):
            raise ValueError("system benchmark model identity is incomplete")
    minimum_gain = record["minimum_gain"]
    if (
        not isfinite(minimum_gain)
        or minimum_gain <= 0
        or not 0.5 < record["confidence"] < 1
        or record["cluster_unit"] != "source_video"
        or not record["budgets"]
    ):
        raise ValueError("system benchmark statistical context is invalid")
    cohorts = set()
    batch_fields = (
        "cohort",
        "partition",
        "generation_block",
        "plan_digest",
        "cache_sha256",
        "ready_sha256",
        "evidence_sha256",
        "cases",
        "independent_videos",
        "independent_videos_per_stratum",
        "reference_score",
        "challenger_score",
        "reference_gain_lower_bound",
        "challenger_gain_lower_bound",
    )
    for batch in batches:
        for field in ("plan_digest", "cache_sha256", "ready_sha256", "evidence_sha256"):
            _require_digest(batch[field])
        if batch["cohort"] in cohorts or batch["partition"] not in {"production", "audit"}:
            raise ValueError("system benchmark cohorts must be distinct and partitioned")
        cohorts.add(batch["cohort"])
        for field in (
            "generation_block",
            "cases",
            "independent_videos",
            "independent_videos_per_stratum",
        ):
            if type(batch[field]) is not int or batch[field] <= 0:
                raise ValueError("system benchmark cohort dimensions must be positive integers")
        for field in (
            "reference_score",
            "challenger_score",
            "reference_gain_lower_bound",
            "challenger_gain_lower_bound",
        ):
            if not isfinite(batch[field]):
                raise ValueError("system benchmark scores and bounds must be finite")
        if record["reference"] == record["challenger"] and (
            batch["reference_score"] != batch["challenger_score"]
            or batch["reference_gain_lower_bound"] != 0
            or batch["challenger_gain_lower_bound"] != 0
        ):
            raise ValueError(
                "system benchmark self comparison must have identical scores and zero bounds"
            )
    if (
        len({batch["cases"] for batch in batches}) != 1
        or len({batch["independent_videos_per_stratum"] for batch in batches}) != 1
    ):
        raise ValueError("different cohort dimensions require separate benchmark records")
    outcome = "REFERENCE_RETAINED"
    if all(batch["reference_gain_lower_bound"] > minimum_gain for batch in batches):
        outcome = "REFERENCE_WON"
    elif all(batch["challenger_gain_lower_bound"] > minimum_gain for batch in batches):
        outcome = "CHALLENGER_WON"
    evaluation = {
        key: record[key]
        for key in (
            "phase",
            "hardware",
            "scope",
            "track_contract_sha256",
            "runtime_contract_sha256",
            "source_sha256",
            "environment_sha256",
            "image_sha256",
            "confidence",
            "minimum_gain",
            "budgets",
            "cluster_unit",
        )
    }
    evaluation.update(
        batch_count=len(batches),
        cases_per_batch=batches[0]["cases"],
        independent_videos_per_stratum=batches[0]["independent_videos_per_stratum"],
        batches=[
            {"batch": index, **{key: batch[key] for key in batch_fields}}
            for index, batch in enumerate(batches)
        ],
    )
    if len(batches) == 1:
        evaluation["evidence_sha256"] = batches[0]["evidence_sha256"]
    return {
        "record_type": "SYSTEM_BENCHMARK",
        "sequence": record["sequence"],
        "track_id": record["track_id"],
        # The report's original full digest, NOT the latest release wrapper digest.
        "track_spec_digest": record["track_spec_digest"],
        "netuid": record["netuid"],
        "affects_tournament": False,
        "reward_eligible": False,
        "reference": {
            key: record["reference"][key] for key in ("model_id", "label", "sha256", "model_digest")
        },
        "challenger": {
            key: record["challenger"][key]
            for key in ("model_id", "label", "sha256", "model_digest")
        },
        "evaluation": evaluation,
        "scores": {
            role: fsum(batch[f"{role}_score"] for batch in batches) / len(batches)
            for role in ("reference", "challenger")
        },
        "outcome": outcome,
    }


def _require_digest(value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
        or value == "0" * 64
    ):
        raise ValueError("system benchmark requires non-placeholder SHA-256 identities")


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
