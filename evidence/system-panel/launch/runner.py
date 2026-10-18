from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict
from math import ceil, fsum, isfinite
from pathlib import Path

# A deployment binding is required for every data-producing command. Intermediate
# releases remain provenance and cannot be selected by default.
ROOT = Path("/unconfigured")
RELEASE = ROOT
TRACK = ROOT
MANIFEST = ROOT
CONFIG = ROOT
SOURCE_POOL = ROOT
SOURCE_BINDING = ROOT
EVIDENCE_CONTRACT = ""
RUNTIME_CONTRACT = ""
COST_BINDING = {}
PRODUCTION_EXACT = 10
PRODUCTION_CONFIRMATION = 5
AUDIT_EXACT = 8
AUDIT_CONFIRMATION = 2
AUDIT_EXACT_MAX_HARD_PROMOTION_RATE_SHIFT = 1 / AUDIT_EXACT
AUDIT_FULL_VIDEO_MAX_HARD_PROMOTION_RATE_SHIFT = 1 / AUDIT_CONFIRMATION
EXPECTED_IDENTITIES = {
    "baseline_self": {
        "file_sha256": "64addb5da430059e37e405d28ca0f2887132905f077ae92fba86982c8b0e9d3f",
        "model_digest": "3fef3eca0eebee238b1ed9c692b217dd25cb1082ef94779ec2ef7c93ae0560c9",
    },
    "mean_pool": {
        "file_sha256": "53bd891d84783b382a9b2f015d62c62fe594886272bf063386fc6170f82fb025",
        "model_digest": "80d7806f70fbc488718d0c7cd0008aa7a46b1224ff951d80ee3f9da8fd042326",
    },
    "shuffled_heads": {
        "file_sha256": "8323c0dc6c584a88de976189f1cfb4e55ab0eaa4301b40d8a49b388467b54360",
        "model_digest": "7c7798ca66eed2a9a8a48bbc5091d361c2ba9b56a908aa790a33c1434b92a42b",
    },
    "perturbed_baseline": {
        "file_sha256": "74b8a085d764cee004c847584305e52e4feca950368f88b35e77099c1c2b0eaf",
        "model_digest": "11f48247cbcfc4555f31a9258745bebf2030c6b76a0304695e3415f612e64499",
    },
    "trained_near_neighbor": {
        "file_sha256": "2382c490bae7a64baacb4c3051a3161938324963478d254270a4a073d9b20796",
        "model_digest": "0731415109dcca5ad19ec7d6949d1aad328699907ed8a526eca1a7cee34a152f",
    },
    "independent_seed_selector": {
        "file_sha256": "bcb59a4c51e4f499a530f114b1243f93dc48d493a893d782de23e3349c388dec",
        "model_digest": "037f74f380bd3ae56255a1cdac788a1e0fc1e1cd8e418c21a0d7ccbd0f282711",
    },
}
TRAINING_TRACK_SPEC_DIGEST = "e00b328f9c45365fdaa74bfdb33e44d12dcee7cd3f1908e13923225f4044eaed"
CONTROL_PROVENANCE = {
    "baseline_self": {"kind": "initial_champion", "training_data": None},
    "mean_pool": {"kind": "deterministic_degenerate_control", "training_data": None},
    "shuffled_heads": {"kind": "deterministic_degenerate_control", "training_data": None},
    "perturbed_baseline": {"kind": "deterministic_near_copy", "training_data": None},
    "trained_near_neighbor": {
        "kind": "external_trained_control",
        "training_track_spec_digest": TRAINING_TRACK_SPEC_DIGEST,
        "training_source_family_count": 224,
        "training_manifest": "provenance/near-neighbor-manifest.json",
        "training_manifest_sha256": (
            "a830475ae5589c9a4795c7a8f37a9481e47ccc0e9fca79df9de8724de7850ac8"
        ),
    },
    "independent_seed_selector": {
        "kind": "external_trained_control",
        "training_track_spec_digest": TRAINING_TRACK_SPEC_DIGEST,
        "training_source_family_count": 224,
        "training_manifest": "provenance/independent-seed-manifest.json",
        "training_manifest_sha256": (
            "16d98685bc507a3e90dd5c500f995327d35a0c2cc3124b90d1b174dadd90f1be"
        ),
    },
}
TRAINING_SOURCE_MANIFEST = {
    "path": "provenance/calibration-training-manifest.json",
    "sha256": "7b0f87fdd6e1b810815ae519ddee871df807f4806653f2fc81cc51e64ad55b3b",
    "source_family_count": 224,
    "disjoint_from_final_production_and_audit": True,
}
SOURCE_QUARANTINE_EVIDENCE = (
    {
        "path": "provenance/source-v6-quarantine-report.json",
        "sha256": "a0c432bb17ebd8d18415625e92a5fd241765dc0644f12cf4acd0e575ce2518cb",
    },
    {
        "path": "provenance/source-v6-verification.json",
        "sha256": "4f79e6a0b99023ec3608c01f2aa7f38b71d313e11fb93503425e18f76f1d26b1",
    },
)
PRIOR_MECHANISM_EVIDENCE = (
    {
        "path": "provenance/mechanism-full-video-gpu67.json",
        "sha256": "6fa8e4c2c563041f426f9560d832dd173ff4825875c4c42e667d221368d26d8d",
        "scope": "mechanism_only_not_release_acceptance",
    },
    {
        "path": "provenance/evaluator-permission-b641208.json",
        "sha256": "d2d68b03cbbcaf79bcf18a9e3ab4ab9d6000de4a4b1d69cd49dedf751bf97533",
        "scope": "development_minio_not_target_r2",
    },
)
REQUIRED_FULL_VIDEO_IDENTITIES = frozenset(EXPECTED_IDENTITIES)
HARD_STABILITY_IDENTITIES = {
    "baseline_self",
    "mean_pool",
    "shuffled_heads",
    "perturbed_baseline",
}
CONTROLS = {name: ROOT / "controls" / f"{name}.safetensors" for name in EXPECTED_IDENTITIES}



def configure(binding_path: Path) -> None:
    global ROOT, RELEASE, TRACK, MANIFEST, CONFIG, SOURCE_POOL, SOURCE_BINDING
    global EVIDENCE_CONTRACT, RUNTIME_CONTRACT, CONTROLS, COST_BINDING

    binding = json.loads(binding_path.read_text())
    if binding.get("format") != "veda-panel-launch-binding-v1" or binding.get("frozen") is not True:
        raise RuntimeError("panel requires an explicitly frozen final runtime binding")
    if set(binding.get("cost_confirmation", {})) != {"registration", "calibration", "confirmation"}:
        raise RuntimeError("panel requires registered cost calibration and confirmation")
    COST_BINDING = binding["cost_confirmation"]
    ROOT = Path(binding["evidence_root"])
    RELEASE = Path(binding["release_root"])
    TRACK = RELEASE / "track.candidate.json"
    MANIFEST = RELEASE / "artifact_manifest.json"
    CONFIG = ROOT / "production.panel.json"
    SOURCE_POOL = Path(binding["source_pool"])
    SOURCE_BINDING = Path(binding["source_binding"])
    EVIDENCE_CONTRACT = binding["evidence_contract_sha256"]
    RUNTIME_CONTRACT = binding["runtime_contract_sha256"]
    CONTROLS = {name: ROOT / "controls" / f"{name}.safetensors" for name in EXPECTED_IDENTITIES}
    if sha256(Path(__file__)) != binding["runner_sha256"]:
        raise RuntimeError("panel runner differs from its frozen binding")


def _directional_gates(controls, reverse_bound, threshold, label):
    """Require known negative controls to lose and the reverse positive to win."""

    if set(controls) != set(EXPECTED_IDENTITIES):
        raise RuntimeError(f"directional gate omitted an identity: {label}")
    if not isfinite(reverse_bound) or not isfinite(threshold) or threshold <= 0:
        raise RuntimeError(f"non-finite or invalid directional gate: {label}")
    if controls["baseline_self"]["gain_lower_bound"] != 0.0:
        raise RuntimeError(f"self bound is not zero: {label}")
    for name in HARD_STABILITY_IDENTITIES:
        gain = controls[name]["gain_lower_bound"]
        if not isfinite(gain) or gain > threshold or controls[name]["promoted"]:
            raise RuntimeError(f"negative control promoted: {label}/{name}")
    if reverse_bound <= threshold:
        raise RuntimeError(f"reverse VEDA-versus-shuffled positive did not promote: {label}")
    return {
        "negative_controls_retained": sorted(HARD_STABILITY_IDENTITIES),
        "reverse_veda_vs_shuffled": {"gain_lower_bound": reverse_bound, "promoted": True},
        "threshold": threshold,
    }

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_cost_confirmation(track, *, source_sha256, environment_sha256):
    """Check hash-bound registered analysis results before consuming panel sources.

    The registered analyzer remains the sole raw-measurement replay/fit path.
    This gate checks its bounded reports, all dimensions and their numerical
    gates, and the exact active track/runtime. Full run digests remain provenance.
    """
    from scoring.spec import track_spec_dict
    from validator.system_matches import _evidence_contract_digest

    if set(COST_BINDING) != {"registration", "calibration", "confirmation"}:
        raise RuntimeError("panel requires registered cost calibration and confirmation")
    documents = {}
    for name, reference in COST_BINDING.items():
        path = Path(reference["path"])
        if sha256(path) != reference["sha256"]:
            raise RuntimeError(f"cost evidence hash changed: {name}")
        documents[name] = json.loads(path.read_text())
    registration, fit, confirmation = (
        documents[name] for name in ("registration", "calibration", "confirmation")
    )
    analysis_path = Path(__file__).with_name("cost_analysis.py")
    if (
        registration["format"] != "veda-h200-cost-registration-v2"
        or registration["runtime_contract_sha256"] != RUNTIME_CONTRACT
        or sha256(analysis_path) != registration["analysis_script_sha256"]
        or sha256(analysis_path.with_name("cost_patterns.py"))
        != registration["pattern_script_sha256"]
        or set(registration["seeds"]["calibration"]) & set(registration["seeds"]["confirmation"])
    ):
        raise RuntimeError("cost registration differs from the frozen runtime/protocol")
    spec = importlib.util.spec_from_file_location("registered_cost_analysis", analysis_path)
    assert spec and spec.loader
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.verify_phase_track(registration, track_spec_dict(track), "confirmation", fit)
    contexts = {
        f"{geometry}/{text}"
        for geometry in registration["geometries"]
        for text in registration["text_token_counts"]
    }
    expected_groups = {context: len(registration["models"]) for context in contexts}
    expected_groups.update(
        {f"{context}/{model}": 1 for context in contexts for model in registration["models"]}
    )
    if set(fit["scales"]) != contexts or any(
        not isfinite(value) or value <= 0 for value in fit["scales"].values()
    ):
        raise RuntimeError("cost fit omits a scale or contains an invalid scale")
    for phase, report in (("calibration", fit), ("confirmation", confirmation)):
        if (
            report["format"] != "veda-h200-cost-fit-v2"
            or report["phase"] != phase
            or report["registration_sha256"] != COST_BINDING["registration"]["sha256"]
            or report["fit_sha256"]
            != (None if phase == "calibration" else COST_BINDING["calibration"]["sha256"])
            or report["source_sha256"] != source_sha256
            or report["environment_sha256"] != environment_sha256
            or report["model_identities"] != registration["model_identities"]
            or report["predictor_bundle_sha256"] != registration["predictor_bundle_sha256"]
            or report["cost_independent_track_sha256"]
            != registration["cost_independent_track_sha256"]
            or report["confirmed_evidence_contract_sha256"] != _evidence_contract_digest(track)
            or report["costs"] != fit["costs"]
            or report["scales"] != fit["scales"]
            or set(report["gates"]) != set(expected_groups)
            or report["e10_confirmation_pass"] is not (phase == "confirmation")
            or report["e11_confirmation_pass"] is not (phase == "confirmation")
        ):
            raise RuntimeError(f"cost {phase} is incomplete or bound to another contract")
        references = report.get("raw_evidence", [])
        if (
            len(references) < len(registration["seeds"][phase])
            or any(set(row) != {"protocol_sha256", "summary_sha256"} for row in references)
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for row in references
                for digest in row.values()
            )
            or any(
                len({row[key] for row in references}) != len(references)
                for key in ("protocol_sha256", "summary_sha256")
            )
        ):
            raise RuntimeError(f"cost {phase} omits unique raw protocol/summary bindings")
        for group, models in expected_groups.items():
            gate = report["gates"][group]
            curves = models * len(registration["layers"]) * len(registration["seeds"][phase])
            if (
                gate["row_count"] != curves * len(registration["budgets"])
                or gate["matched_curve_count"] != curves
                or set(gate["budget_error_gates"]) != {str(b) for b in registration["budgets"]}
                or gate["e10_pass"] is not True
                or gate["e11_pass"] is not True
            ):
                raise RuntimeError(f"cost {phase} group is incomplete or failed: {group}")
            metrics = [
                (gate["minimum_matched_curve_spearman"], 0.95, 1.0),
                (gate["minimum_overhead_fraction"], 0.0, 0.1),
                (gate["maximum_overhead_fraction"], gate["minimum_overhead_fraction"], 0.1),
            ]
            for errors in (gate, *gate["budget_error_gates"].values()):
                metrics.extend(
                    (
                        (errors["median_relative_error"], 0.0, 0.05),
                        (errors["p95_relative_error"], 0.0, 0.10),
                    )
                )
            if any(
                not isfinite(value) or not lower <= value <= upper
                for value, lower, upper in metrics
            ):
                raise RuntimeError(f"cost {phase} numerical gate failed: {group}")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def write_once(path: Path, value: object) -> None:
    payload = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        with path.open("rb") as stream:
            if stream.read() != payload:
                raise RuntimeError(f"immutable evidence conflicts: {path}") from None


def record_event(kind: str, **values: object) -> None:
    payload = canonical({"event": kind, "time_ns": time.time_ns(), **values}) + b"\n"
    path = ROOT / "events" / f"{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def labels(*, exact_only: bool = False) -> tuple[str, ...]:
    exact = tuple(f"production-{index:02d}" for index in range(PRODUCTION_EXACT)) + tuple(
        f"audit-{index:02d}" for index in range(AUDIT_EXACT)
    )
    if exact_only:
        return exact
    return (
        exact
        + tuple(
            f"production-{index:02d}"
            for index in range(PRODUCTION_EXACT, PRODUCTION_EXACT + PRODUCTION_CONFIRMATION)
        )
        + tuple(
            f"audit-{index:02d}" for index in range(AUDIT_EXACT, AUDIT_EXACT + AUDIT_CONFIRMATION)
        )
    )


def confirmation_labels() -> tuple[str, ...]:
    exact = frozenset(labels(exact_only=True))
    return tuple(label for label in labels() if label not in exact)


def _plan_block(track, sequence: int) -> int:
    return track.track_start_block + sequence * track.cache.generation_interval_blocks


def build_plans(track, pool) -> dict[str, object]:
    from validator.production.plan import (
        build_audit_evidence_plan,
        build_plan,
        generation_seed,
        select_source_proofs,
    )

    result = {}
    sequence = 1
    families: set[str] = set()
    source_ids: set[str] = set()
    for partition, count in (
        ("production", PRODUCTION_EXACT + PRODUCTION_CONFIRMATION),
        ("audit", AUDIT_EXACT + AUDIT_CONFIRMATION),
    ):
        used: set[str] = set()
        for index in range(count):
            label = f"{partition}-{index:02d}"
            block_hash = (
                "0x"
                + hashlib.sha256(
                    b"veda-final-system-panel-v2\0"
                    + bytes.fromhex(EVIDENCE_CONTRACT)
                    + bytes.fromhex(RUNTIME_CONTRACT)
                    + partition.encode("ascii")
                    + index.to_bytes(4, "big")
                ).hexdigest()
            )
            batch_seed = generation_seed(track.digest, block_hash)
            proofs = select_source_proofs(
                pool,
                commitments=track.source_commitments,
                partition=partition,
                strata=tuple(track.scoring.stratum_weight_map),
                sources_per_stratum=track.scoring.rollouts_per_stratum,
                batch_seed=batch_seed,
                used_source_digests=frozenset(used),
            )
            used.update(proof.leaf.digest for proof in proofs)
            current_families = {proof.leaf.source_family_id for proof in proofs}
            current_sources = {proof.leaf.source_id for proof in proofs}
            if (
                len(current_families) != len(proofs)
                or families & current_families
                or source_ids & current_sources
            ):
                raise RuntimeError("panel plans reuse a source family or video")
            families.update(current_families)
            source_ids.update(current_sources)
            common = dict(
                track_spec_digest=track.digest,
                generation_block=_plan_block(track, sequence),
                block_hash=block_hash,
                track_start_block=track.track_start_block,
                minimum_generation_delay_blocks=track.cache.minimum_generation_delay_blocks,
                generation_interval_blocks=track.cache.generation_interval_blocks,
                source_proofs=proofs,
                scoring=track.scoring,
                available_routes=track.model.selector.routes,
            )
            if partition == "production":
                is_confirmation = index >= PRODUCTION_EXACT
                plan = build_plan(
                    **common,
                    exact_challenge_ids=() if is_confirmation else (f"panel-{label}",),
                    confirmation_challenge_ids=(f"panel-{label}",) if is_confirmation else (),
                )
            else:
                plan = build_audit_evidence_plan(
                    **common,
                    evidence_id=f"panel-{label}",
                )
            if {proof.leaf.partition for proof in plan.source_proofs} != {partition}:
                raise RuntimeError("panel plan crosses source partitions")
            result[label] = plan
            sequence += 1
    sources_per_batch = track.scoring.rollouts_per_stratum * len(track.scoring.stratum_weight_map)
    if len(families) != sources_per_batch * len(result):
        raise RuntimeError("panel source families are not globally independent")
    return result


def load_runtime_context(*, evaluator: bool):
    from scoring.spec import load_track_spec
    from validator.production.commitment import ArtifactManifest
    from validator.production.config import load_producer_config
    from validator.production.environment import (
        evaluation_environment_digest,
        evaluation_source_digest,
    )

    config = load_producer_config(CONFIG)
    track = load_track_spec(TRACK)
    manifest = ArtifactManifest.decode(MANIFEST.read_bytes())
    if manifest.digest != track.artifact_manifest_sha256:
        raise RuntimeError("runtime manifest differs from the candidate track")
    from validator.system_matches import _evidence_contract_digest, _evidence_runtime_digest

    if (
        _evidence_contract_digest(track) != EVIDENCE_CONTRACT
        or _evidence_runtime_digest(MANIFEST.read_bytes()) != RUNTIME_CONTRACT
    ):
        raise RuntimeError("active scoring or runtime contract differs from the frozen binding")
    if evaluator:
        config.verify_evaluator_manifest(manifest)
    else:
        config.verify_manifest(manifest)
    _verify_cost_confirmation(
        track,
        source_sha256=evaluation_source_digest(),
        environment_sha256=evaluation_environment_digest(),
    )
    return config, track, manifest


def load_producer_context():
    from validator.production.commitment import load_bound_source_pool

    config, track, manifest = load_runtime_context(evaluator=False)
    pool = load_bound_source_pool(
        SOURCE_POOL.read_bytes(),
        SOURCE_BINDING.read_bytes(),
        expected_root=track.source_pool_merkle_root,
        expected_commitments=track.source_commitment_documents,
        sample_cache=config.artifact_path(config.runtime.sample_cache),
    )
    return config, track, manifest, pool, build_plans(track, pool)


def prepare() -> None:
    from scoring.model import validate_model
    from validator.production.commitment import verify_source_proof
    from validator.system_matches import _evidence_contract_digest, _evidence_runtime_digest

    config, track, manifest, pool, plans = load_producer_context()
    config.verify_evaluator_manifest(manifest)
    if _evidence_contract_digest(track) != EVIDENCE_CONTRACT:
        raise RuntimeError("candidate scoring evidence contract changed")
    if _evidence_runtime_digest(MANIFEST.read_bytes()) != RUNTIME_CONTRACT:
        raise RuntimeError("frozen scoring runtime contract changed")
    identities = {}
    for name, path in CONTROLS.items():
        identity = asdict(
            validate_model(
                path,
                schema=track.model.tensor_schema,
                max_bytes=track.model.max_bytes,
                expected_hotkey=None,
            )
        )
        if {key: identity[key] for key in ("file_sha256", "model_digest")} != (
            EXPECTED_IDENTITIES[name]
        ):
            raise RuntimeError(f"control identity changed: {name}")
        identities[name] = identity
    for provenance in (
        TRAINING_SOURCE_MANIFEST,
        *SOURCE_QUARANTINE_EVIDENCE,
        *PRIOR_MECHANISM_EVIDENCE,
        *(row for row in CONTROL_PROVENANCE.values() if "training_manifest" in row),
    ):
        relative = provenance.get("path", provenance.get("training_manifest"))
        expected = provenance.get("sha256", provenance.get("training_manifest_sha256"))
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"control provenance changed: {relative}")
    training = json.loads((ROOT / TRAINING_SOURCE_MANIFEST["path"]).read_text())
    training_families = set(training["source_families"])
    source_families = {leaf.source_family_id for leaf in pool.leaves}
    quarantine = json.loads((ROOT / SOURCE_QUARANTINE_EVIDENCE[0]["path"]).read_text())
    if (
        len(training_families) != TRAINING_SOURCE_MANIFEST["source_family_count"]
        or training_families & source_families
        or quarantine["candidate_root"] != pool.root
        or quarantine["candidate_calibration_family_intersection"]
        or quarantine["candidate_new_shadow_family_intersection"]
        or quarantine["production_audit_family_intersection"]
    ):
        raise RuntimeError("panel sources overlap calibration or shadow source families")
    commitment_digests = tuple(item.sha256 for item in track.source_commitments)
    plan_documents = {}
    for label, plan in plans.items():
        if any(
            not verify_source_proof(
                proof,
                track.source_pool_merkle_root,
                commitment_digests=commitment_digests,
            )
            for proof in plan.source_proofs
        ):
            raise RuntimeError(f"invalid source proof in {label}")
        path = ROOT / "plans" / f"{label}.json"
        write_once(path, json.loads(plan.encode()))
        plan_documents[label] = {
            "plan_digest": plan.plan_digest,
            "plan_sha256": sha256(path),
            "partition": label.split("-", 1)[0],
            "purpose": "exact" if label in labels(exact_only=True) else "full_video_reserved",
            "selected_commitment_digests": [proof.leaf.digest for proof in plan.source_proofs],
            "source_proofs": [proof.document() for proof in plan.source_proofs],
        }
    script = ROOT / "system_panel.py"
    verification = {
        "format": "veda-system-panel-runtime-v2",
        "artifact_manifest_sha256": manifest.digest,
        "candidate_track_spec_digest": track.digest,
        "evidence_contract_sha256": EVIDENCE_CONTRACT,
        "runtime_contract_sha256": RUNTIME_CONTRACT,
        "block_provenance": {
            "kind": "synthetic_deterministic_hash",
            "purpose": "launch_calibration_only",
            "live_chain_lifecycle_evidence": False,
        },
        "oci_image_sha256": sha256(config.artifact_path(config.runtime.oci_image)),
        "script_sha256": sha256(script),
        "source_pool_merkle_root": pool.root,
        "controls": identities,
        "control_provenance": CONTROL_PROVENANCE,
        "training_source_manifest": TRAINING_SOURCE_MANIFEST,
        "source_quarantine_evidence": SOURCE_QUARANTINE_EVIDENCE,
        "prior_mechanism_evidence": PRIOR_MECHANISM_EVIDENCE,
        "plans": plan_documents,
        "exact_matrix": {"production_batches": PRODUCTION_EXACT, "audit_batches": AUDIT_EXACT},
        "reserved_full_video": {
            "production_batches": PRODUCTION_CONFIRMATION,
            "audit_batches": AUDIT_CONFIRMATION,
        },
    }
    write_once(ROOT / "runtime-verification.json", verification)
    record_event("prepare_complete", plan_count=len(plans))
    print(json.dumps(verification, sort_keys=True))


def _ready_references(results):
    from validator.production.plan import DenseReference, ReadyObject

    references = []
    local = []
    for result in results:
        video = Path(result.video_path)
        latent = Path(result.latent_path)
        video_digest = sha256(video)
        latent_digest = sha256(latent)
        reference = DenseReference(
            result.rollout_id,
            ReadyObject(f"videos/{video_digest}.mp4", video_digest, video.stat().st_size),
            ReadyObject(
                f"latents/{latent_digest}.safetensors", latent_digest, latent.stat().st_size
            ),
        )
        references.append(reference)
        local.append(
            {
                "rollout_id": result.rollout_id,
                "video_path": str(video),
                "video_sha256": video_digest,
                "latent_path": str(latent),
                "latent_sha256": latent_digest,
            }
        )
    return tuple(references), local


def generate() -> None:
    from scoring.cache import assemble_exact_cache_batch, inspect_exact_cache_batch
    from validator.production.service import WorkerPool, _build_selected_sample_cache

    config, track, _manifest, _pool, plans = load_producer_context()
    if not (ROOT / "runtime-verification.json").is_file():
        raise RuntimeError("panel prepare must succeed before generation")
    pending = [label for label in labels() if not (ROOT / "batches" / f"{label}.json").is_file()]
    workers = WorkerPool(config, track) if pending else None
    try:
        for label in pending:
            plan = plans[label]
            started_ns = time.time_ns()
            record_event("generation_started", label=label, plan_digest=plan.plan_digest)
            results = workers.run(plan)
            references, local_references = _ready_references(results)
            batch_root = ROOT / "batch-data" / label
            selected_cache, _files = _build_selected_sample_cache(
                config, plan, batch_root / "sample-cache"
            )
            exact = assemble_exact_cache_batch(
                batch_root,
                tuple(result.shard_path for result in results),
                generated_block=plan.generation_block,
                seed_block_hash=plan.block_hash,
                plan_digest=plan.plan_digest,
                track_spec_digest=track.digest,
                artifact_manifest_sha256=track.artifact_manifest_sha256,
                source_pool_merkle_root=track.source_pool_merkle_root,
                source_proofs=tuple(proof.document() for proof in plan.source_proofs),
                dense_references=tuple(reference.document() for reference in references),
                selected_inputs=selected_cache.document(),
            )
            inspected = inspect_exact_cache_batch(exact.path, max_bytes=track.cache.exact_max_bytes)
            if inspected.sha256 != exact.sha256 or len(inspected.cases) != track.scoring.case_count:
                raise RuntimeError(f"assembled exact batch failed inspection: {label}")
            finished_ns = time.time_ns()
            write_once(
                ROOT / "batches" / f"{label}.json",
                {
                    "format": "veda-system-panel-batch-v2",
                    "label": label,
                    "partition": label.split("-", 1)[0],
                    "candidate_track_spec_digest": track.digest,
                    "artifact_manifest_sha256": track.artifact_manifest_sha256,
                    "evidence_contract_sha256": EVIDENCE_CONTRACT,
                    "runtime_contract_sha256": RUNTIME_CONTRACT,
                    "block_provenance": "synthetic_launch_calibration",
                    "plan_digest": plan.plan_digest,
                    "plan_path": str(ROOT / "plans" / f"{label}.json"),
                    "batch_seed": plan.batch_seed,
                    "block_hash": plan.block_hash,
                    "generation_block": plan.generation_block,
                    "exact_path": str(exact.path),
                    "exact_sha256": exact.sha256,
                    "bytes": exact.path.stat().st_size,
                    "case_count": len(inspected.cases),
                    "started_ns": started_ns,
                    "finished_ns": finished_ns,
                    "seconds": (finished_ns - started_ns) / 1e9,
                    "source_ids": [proof.leaf.source_id for proof in plan.source_proofs],
                    "source_families": [
                        proof.leaf.source_family_id for proof in plan.source_proofs
                    ],
                    "selected_commitment_digests": [
                        proof.leaf.digest for proof in plan.source_proofs
                    ],
                    "source_proofs": [proof.document() for proof in plan.source_proofs],
                    "selected_inputs": selected_cache.document(),
                    "selected_cache_path": str(batch_root / "sample-cache"),
                    "dense_references": local_references,
                },
            )
            record_event("generation_finished", label=label, exact_sha256=exact.sha256)
            print(
                json.dumps({"label": label, "seconds": (finished_ns - started_ns) / 1e9}),
                flush=True,
            )
    except BaseException as exc:
        record_event("generation_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if workers is not None:
            workers.close()
    write_once(
        ROOT / "generation-complete.json",
        {"format": "veda-system-panel-generation-v2", "labels": labels()},
    )


def case_sha256(case) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical(
            {
                "query_tile_ids": case.query_tile_ids,
                "rollout_id": case.rollout_id,
                "route": case.route,
                "source_id": case.source_id,
                "step_index": case.step_index,
                "step_window": case.step_window,
                "stratum": case.stratum,
                "tile_plan_hash": case.tile_plan_hash,
            }
        )
    )
    for value in (
        case.q,
        case.k,
        case.v,
        case.dense,
        case.query_valid_counts,
        case.key_valid_counts,
        case.key_segments,
        case.required_keys,
    ):
        digest.update(value.tobytes())
    return digest.hexdigest()


def score(worker: int) -> None:
    import torch

    from scoring.cache import ExactCacheReader
    from scoring.metrics import log_cost_auc
    from scoring.model import validate_model
    from scoring.selector import load_selector
    from scoring.torch_backend import TorchCaseScorer
    from validator.production.environment import verify_runtime_environment

    if worker not in (0, 1):
        raise ValueError("formal scorer worker must be zero or one")
    config, track, _manifest = load_runtime_context(evaluator=True)
    verify_runtime_environment(
        config.evaluator_artifact_path(config.runtime.source_bundle),
        config.evaluator_artifact_path(config.runtime.dependency_lock),
    )
    selectors = {name: load_selector(path, track.model.selector) for name, path in CONTROLS.items()}
    identities = {
        name: asdict(
            validate_model(path, schema=track.model.tensor_schema, max_bytes=track.model.max_bytes)
        )
        for name, path in CONTROLS.items()
    }
    for name, identity in identities.items():
        if {key: identity[key] for key in ("file_sha256", "model_digest")} != EXPECTED_IDENTITIES[
            name
        ]:
            raise RuntimeError(f"score worker control identity changed: {name}")
    costs = [budget.calibrated_cost for budget in track.scoring.budgets]
    assigned = labels(exact_only=True)[worker::2]
    deadline = time.monotonic() + 48 * 60 * 60
    for label in assigned:
        output = ROOT / "scores" / f"{label}.json"
        if output.is_file():
            continue
        record = ROOT / "batches" / f"{label}.json"
        while not record.is_file():
            if (ROOT / "generation-complete.json").is_file():
                raise RuntimeError("completed generation omitted a scorer input")
            if time.monotonic() >= deadline:
                raise RuntimeError("scorer exceeded the bounded wait for generated batches")
            time.sleep(10)
        batch = json.loads(record.read_text())
        started_ns = time.time_ns()
        record_event("scoring_started", label=label, worker=worker)
        cells = []
        with ExactCacheReader(
            batch["exact_path"],
            expected_sha256=batch["exact_sha256"],
            max_bytes=track.cache.exact_max_bytes,
        ) as reader:
            for case in reader:
                scorer = TorchCaseScorer(case, device="cuda:0")
                scores = {}
                for name, selector in selectors.items():
                    qualities, _errors = scorer.score(
                        selector, list(track.scoring.budgets), track.scoring.tail_mean_weight
                    )
                    scores[name] = log_cost_auc(costs, qualities)
                cells.append(
                    {
                        "case_sha256": case_sha256(case),
                        "query_tile_ids": list(case.query_tile_ids),
                        "rollout_id": case.rollout_id,
                        "route": case.route,
                        "source_id": case.source_id,
                        "step_index": case.step_index,
                        "step_window": case.step_window,
                        "stratum": case.stratum,
                        "scores": scores,
                    }
                )
                del scorer, case
        finished_ns = time.time_ns()
        write_once(
            output,
            {
                "format": "veda-system-panel-scores-v2",
                "label": label,
                "partition": batch["partition"],
                "batch_sha256": batch["exact_sha256"],
                "candidate_track_spec_digest": track.digest,
                "evidence_contract_sha256": EVIDENCE_CONTRACT,
                "runtime_contract_sha256": RUNTIME_CONTRACT,
                "model_identities": identities,
                "cells": cells,
                "worker": worker,
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "seconds": (finished_ns - started_ns) / 1e9,
            },
        )
        record_event("scoring_finished", label=label, worker=worker)
        print(json.dumps({"label": label, "seconds": (finished_ns - started_ns) / 1e9}), flush=True)


def _case_design(cells):
    from scoring.exact import CaseDesign

    return [
        CaseDesign(
            cell["route"],
            cell["stratum"],
            cell["rollout_id"],
            cell["source_id"],
            cell["step_index"],
            cell["step_window"],
        )
        for cell in cells
    ]


def _bound(track, challenger, incumbent, design) -> float:
    from scoring.exact import paired_score_gain_bound

    return paired_score_gain_bound(
        challenger,
        incumbent,
        design,
        stratum_weights=track.scoring.stratum_weight_map,
        step_window_weights=track.scoring.step_window_weight_map,
        confidence=track.match_confidence,
        resamples=track.scoring.bootstrap_resamples,
        seed=track.scoring.bootstrap_seed,
    )


def _weighted(track, values, design) -> float:
    total = 0.0
    for stratum, stratum_weight in track.scoring.stratum_weight_map.items():
        for window, step_weight in track.scoring.step_window_weight_map.items():
            selected = [
                values[index]
                for index, case in enumerate(design)
                if case.stratum == stratum and case.step_window == window
            ]
            if not selected:
                raise RuntimeError("panel is missing a weighted design cell")
            total += stratum_weight * step_weight * fsum(selected) / len(selected)
    return total


def _percentile(values, percentile: float) -> float:
    """Return the documented nearest-rank percentile."""

    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


def _overlap_pairs(generation_intervals, scoring_intervals) -> list[tuple[str, str]]:
    """Return distinct-batch score/generation pairs with intersecting open intervals."""

    return sorted(
        (score_label, generation_label)
        for score_label, score_start, score_finish in scoring_intervals
        for generation_label, generation_start, generation_finish in generation_intervals
        if score_label != generation_label
        and max(score_start, generation_start) < min(score_finish, generation_finish)
    )


def _audit_shift_acceptance(*, audit_batches: int, maximum_shift: float) -> dict[str, object]:
    quantum = 1 / audit_batches
    if maximum_shift != quantum:
        raise RuntimeError("audit shift gate must equal one independent audit-batch quantum")
    return {
        "sample_unit": "independent_batch",
        "audit_batches": audit_batches,
        "smallest_nonzero_audit_promotion_rate_step": quantum,
        "maximum_absolute_hard_identity_promotion_rate_shift": maximum_shift,
        "rationale": (
            "allow at most one audit-rate quantum while requiring zero production outcome "
            "flips; this coarse drift alarm does not replace threshold-near power estimates"
        ),
    }


def self_test() -> None:
    overlapping = _overlap_pairs(
        (("generation-next", 20, 40),),
        (("score-ready", 10, 30),),
    )
    serial = _overlap_pairs(
        (("generation-next", 30, 40),),
        (("score-ready", 10, 20),),
    )
    same_batch = _overlap_pairs(
        (("same", 20, 40),),
        (("same", 10, 30),),
    )
    if overlapping != [("score-ready", "generation-next")]:
        raise RuntimeError("cross-batch interval overlap detection is invalid")
    if serial or same_batch:
        raise RuntimeError("serial or same-batch intervals were incorrectly accepted")
    if _percentile((1.0, 2.0, 3.0, 4.0), 0.95) != 4.0:
        raise RuntimeError("nearest-rank percentile implementation is invalid")
    exact_calls = []
    video_calls = []

    def exact_bound(challenger, incumbent, label):
        exact_calls.append((challenger, incumbent, label))
        return {"exact-a": 0.02, "exact-b": 0.02}[label]

    def video_bound(challenger, incumbent, label):
        video_calls.append((challenger, incumbent, label))
        return {"confirm-0": -0.01, "confirm-1": 0.02}[label]

    replay = _two_stage_fifo_replay(
        order=("a", "b"),
        exact_labels=("exact-a", "exact-b"),
        confirmation_labels=("confirm-0", "confirm-1"),
        exact_bound=exact_bound,
        video_bound=video_bound,
        exact_threshold=0.005,
        video_threshold=0.005,
    )
    if (
        exact_calls != [("a", "baseline_self", "exact-a"), ("b", "baseline_self", "exact-b")]
        or video_calls != [("a", "baseline_self", "confirm-0"), ("b", "baseline_self", "confirm-1")]
        or replay["terminal_champion"] != "b"
    ):
        raise RuntimeError("two-stage FIFO did not retain the incumbent after failed confirmation")
    capacity = _two_stage_fifo_replay(
        order=("a", "b", "c", "d", "e"),
        exact_labels=("e0", "e1", "e2", "e3", "e4"),
        confirmation_labels=("c0", "c1", "c2", "c3", "c4"),
        exact_bound=lambda _challenger, _incumbent, _label: 0.02,
        video_bound=lambda _challenger, _incumbent, _label: -0.01,
        exact_threshold=0.005,
        video_threshold=0.005,
    )
    if capacity["confirmation_batches_consumed"] != 5:
        raise RuntimeError("two-stage FIFO does not cover every possible exact pass")
    reverse_video_calls = []
    reverse = _two_stage_fifo_replay(
        order=("b", "a"),
        exact_labels=("exact-b", "exact-a"),
        confirmation_labels=("confirm-0", "confirm-1"),
        exact_bound=lambda _challenger, _incumbent, _label: 0.02,
        video_bound=lambda challenger, incumbent, label: (
            reverse_video_calls.append((challenger, incumbent, label)) or -0.01
        ),
        exact_threshold=0.005,
        video_threshold=0.005,
    )
    if [call[2] for call in reverse_video_calls] != ["confirm-0", "confirm-1"] or reverse[
        "confirmation_batches_consumed"
    ] != 2:
        raise RuntimeError("counterfactual replay did not reuse the frozen confirmation sequence")
    print(
        json.dumps(
            {
                "cross_batch_overlap": True,
                "serial_rejected": True,
                "two_stage_fifo": True,
            },
            sort_keys=True,
        )
    )


def aggregate() -> None:
    config, track, _manifest = load_runtime_context(evaluator=True)
    del config
    expected_labels = labels(exact_only=True)
    geometries = set(track.scoring.stratum_weight_map)
    routes = track.scoring.sampled_route_count
    threshold = track.scoring.minimum_frontier_gain
    batches = []
    identities = None
    environment = None
    seen_sources: set[str] = set()
    seen_families: set[str] = set()
    raw_scores = {}
    generation_seconds = []
    scoring_seconds = []
    generation_intervals = []
    scoring_intervals = []
    for label in expected_labels:
        batch_path = ROOT / "batches" / f"{label}.json"
        score_path = ROOT / "scores" / f"{label}.json"
        batch = json.loads(batch_path.read_text())
        score = json.loads(score_path.read_text())
        if (
            batch["label"] != label
            or score["label"] != label
            or batch["partition"] != label.split("-", 1)[0]
            or score["batch_sha256"] != batch["exact_sha256"]
            or score["candidate_track_spec_digest"] != track.digest
            or score["runtime_contract_sha256"] != RUNTIME_CONTRACT
            or len(score["cells"]) != track.scoring.case_count
            or sha256(Path(batch["exact_path"])) != batch["exact_sha256"]
        ):
            raise RuntimeError(f"panel batch/score binding failed: {label}")
        if identities is None:
            identities = score["model_identities"]
            environment = {key: score[key] for key in ("cuda", "gpu", "torch")}
        elif identities != score["model_identities"] or environment != {
            key: score[key] for key in ("cuda", "gpu", "torch")
        }:
            raise RuntimeError("panel workers used different models or environments")
        if set(identities) != set(EXPECTED_IDENTITIES):
            raise RuntimeError("panel omitted or added a selector identity")
        for name, expected in EXPECTED_IDENTITIES.items():
            if {key: identities[name][key] for key in expected} != expected:
                raise RuntimeError(f"panel identity mismatch: {name}")
        cells = score["cells"]
        design = _case_design(cells)
        if (
            {cell["stratum"] for cell in cells} != geometries
            or len({cell["route"] for cell in cells}) != routes
            or any(
                len(cell["query_tile_ids"]) != track.scoring.query_tiles_per_case for cell in cells
            )
            or {cell["source_id"] for cell in cells} != set(batch["source_ids"])
            or len(batch["source_ids"]) != track.scoring.rollouts_per_stratum * len(geometries)
        ):
            raise RuntimeError(f"panel design dimensions are incomplete: {label}")
        if len(set(batch["source_families"])) != len(batch["source_families"]):
            raise RuntimeError(f"batch does not have independent source families: {label}")
        if seen_sources & set(batch["source_ids"]) or seen_families & set(batch["source_families"]):
            raise RuntimeError("panel batches reuse source videos or families")
        seen_sources.update(batch["source_ids"])
        seen_families.update(batch["source_families"])
        values = {name: [cell["scores"][name] for cell in cells] for name in EXPECTED_IDENTITIES}
        if any(set(cell["scores"]) != set(EXPECTED_IDENTITIES) for cell in cells):
            raise RuntimeError(f"score cells omit an identity: {label}")
        self_bound = _bound(track, values["baseline_self"], values["baseline_self"], design)
        if self_bound != 0.0:
            raise RuntimeError(f"self-vs-self bound is not exactly zero: {label}")
        controls = {}
        for name in EXPECTED_IDENTITIES:
            gain = _bound(track, values[name], values["baseline_self"], design)
            controls[name] = {
                "score": _weighted(track, values[name], design),
                "gain_lower_bound": gain,
                "promoted": gain > threshold,
            }
        reverse_bound = _bound(track, values["baseline_self"], values["shuffled_heads"], design)
        directional = (
            _directional_gates(controls, reverse_bound, threshold, label)
            if batch["partition"] == "production" else None
        )
        row = {
            "label": label,
            "directional_gates": directional,
            "reverse_veda_vs_shuffled_bound": reverse_bound,
            "partition": batch["partition"],
            "batch_sha256": batch["exact_sha256"],
            "self_zero_bound": self_bound,
            "controls": controls,
        }
        batches.append(row)
        raw_scores[label] = (values, design)
        generation_seconds.append(batch["seconds"])
        scoring_seconds.append(score["seconds"])
        generation_intervals.append((label, batch["started_ns"], batch["finished_ns"]))
        scoring_intervals.append((label, score["started_ns"], score["finished_ns"]))
    stability = {}
    for name in EXPECTED_IDENTITIES:
        production_outcomes = [
            row["controls"][name]["promoted"] for row in batches if row["partition"] == "production"
        ]
        audit_outcomes = [
            row["controls"][name]["promoted"] for row in batches if row["partition"] == "audit"
        ]
        production_rate = sum(production_outcomes) / len(production_outcomes)
        audit_rate = sum(audit_outcomes) / len(audit_outcomes)
        audit_shift = audit_rate - production_rate
        stability[name] = {
            "production_outcome": production_outcomes[0],
            "production_outcome_flips": sum(
                left != right
                for left, right in zip(production_outcomes, production_outcomes[1:], strict=False)
            ),
            "production_stable": len(set(production_outcomes)) == 1,
            "production_promotion_rate": production_rate,
            "audit_outcome_flips": sum(
                left != right
                for left, right in zip(audit_outcomes, audit_outcomes[1:], strict=False)
            ),
            "audit_promotion_rate": audit_rate,
            "audit_promotion_rate_shift": audit_shift,
            "audit_shift_pass": (abs(audit_shift) <= AUDIT_EXACT_MAX_HARD_PROMOTION_RATE_SHIFT),
            "hard_gate": name in HARD_STABILITY_IDENTITIES,
        }
    if any(
        not value["production_stable"]
        for name, value in stability.items()
        if name in HARD_STABILITY_IDENTITIES
    ):
        raise RuntimeError(f"production panel promotion outcome flipped: {stability}")
    if any(
        not value["audit_shift_pass"]
        for name, value in stability.items()
        if name in HARD_STABILITY_IDENTITIES
    ):
        raise RuntimeError(f"audit exact shift exceeded its frozen threshold: {stability}")
    partition_report = {}
    for partition in ("production", "audit"):
        selected = [row for row in batches if row["partition"] == partition]
        partition_report[partition] = {
            name: {
                "mean_score": fsum(row["controls"][name]["score"] for row in selected)
                / len(selected),
                "mean_gain_lower_bound": fsum(
                    row["controls"][name]["gain_lower_bound"] for row in selected
                )
                / len(selected),
                "promotion_rate": sum(row["controls"][name]["promoted"] for row in selected)
                / len(selected),
            }
            for name in EXPECTED_IDENTITIES
        }
    shift = {
        name: {
            key: partition_report["audit"][name][key] - partition_report["production"][name][key]
            for key in ("mean_score", "mean_gain_lower_bound", "promotion_rate")
        }
        for name in EXPECTED_IDENTITIES
    }
    challengers = tuple(name for name in EXPECTED_IDENTITIES if name != "baseline_self")
    replay = {}
    for replay_name, order, batch_labels in (
        ("forward", challengers, tuple(f"production-{index:02d}" for index in range(5))),
        (
            "reverse",
            tuple(reversed(challengers)),
            tuple(f"production-{index:02d}" for index in range(5)),
        ),
    ):
        champion = "baseline_self"
        steps = []
        for challenger, label in zip(order, batch_labels, strict=True):
            values, design = raw_scores[label]
            gain = _bound(track, values[challenger], values[champion], design)
            promoted = gain > threshold
            steps.append(
                {
                    "label": label,
                    "incumbent": champion,
                    "challenger": challenger,
                    "gain_lower_bound": gain,
                    "promoted": promoted,
                }
            )
            if promoted:
                champion = challenger
        replay[replay_name] = {"steps": steps, "terminal_champion": champion}
    near = {}
    for target in (0.0, 0.004, 0.005, 0.006, 0.01):
        promotions = []
        bounds = []
        for label in expected_labels:
            values, design = raw_scores[label]
            residual = [
                challenger - baseline
                for challenger, baseline in zip(
                    values["perturbed_baseline"], values["baseline_self"], strict=True
                )
            ]
            centered = [value - _weighted(track, residual, design) + target for value in residual]
            synthetic = [
                baseline + delta
                for baseline, delta in zip(values["baseline_self"], centered, strict=True)
            ]
            bound = _bound(track, synthetic, values["baseline_self"], design)
            bounds.append(bound)
            promotions.append(bound > threshold)
        near[str(target)] = {
            "synthetic_shift_on_real_perturbed_residuals": True,
            "promotion_rate": sum(promotions) / len(promotions),
            "false_promotion_rate": sum(promotions) / len(promotions)
            if target <= threshold
            else None,
            "false_rejection_rate": (1 - sum(promotions) / len(promotions))
            if target > threshold
            else None,
            "power": sum(promotions) / len(promotions) if target > threshold else None,
            "bounds": bounds,
        }
    overlap_pairs = _overlap_pairs(generation_intervals, scoring_intervals)
    summary = {
        "format": "veda-system-panel-summary-v2",
        "candidate_track_spec_digest": track.digest,
        "artifact_manifest_sha256": track.artifact_manifest_sha256,
        "evidence_contract_sha256": EVIDENCE_CONTRACT,
        "runtime_contract_sha256": RUNTIME_CONTRACT,
        "block_provenance": {
            "kind": "synthetic_deterministic_hash",
            "live_chain_lifecycle_evidence": False,
            "purpose": "launch_calibration_only",
        },
        "exact_matrix": {
            "production_batches": PRODUCTION_EXACT,
            "audit_batches": AUDIT_EXACT,
            "cases_per_batch": track.scoring.case_count,
        },
        "reserved_full_video": {
            "production_batches": PRODUCTION_CONFIRMATION,
            "audit_batches": AUDIT_CONFIRMATION,
        },
        "confidence": track.match_confidence,
        "minimum_gain": threshold,
        "cluster_unit": "route/source_video/step_window",
        "model_identities": identities,
        "control_provenance": CONTROL_PROVENANCE,
        "training_source_manifest": TRAINING_SOURCE_MANIFEST,
        "source_quarantine_evidence": SOURCE_QUARANTINE_EVIDENCE,
        "environment": environment,
        "batches": batches,
        "outcome_stability": stability,
        "stability_gate": {
            "production_hard_identities": sorted(HARD_STABILITY_IDENTITIES),
            "production_max_outcome_flips": 0,
            "audit_exact_max_hard_promotion_rate_shift": (
                AUDIT_EXACT_MAX_HARD_PROMOTION_RATE_SHIFT
            ),
            "audit_shift_acceptance": _audit_shift_acceptance(
                audit_batches=AUDIT_EXACT,
                maximum_shift=AUDIT_EXACT_MAX_HARD_PROMOTION_RATE_SHIFT,
            ),
        },
        "partition_report": partition_report,
        "production_to_audit_shift": shift,
        "exact_stage_fifo_replay": replay,
        "threshold_near_calibration": near,
        "concurrency": {
            "generation_p95_seconds": _percentile(generation_seconds, 0.95),
            "scoring_p95_seconds": _percentile(scoring_seconds, 0.95),
            "percentile_definition": "nearest-rank",
            "interval_definition": "open wall-clock intervals from distinct batch labels",
            "overlap_pairs": overlap_pairs,
        },
        "raw_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for directory in (ROOT / "batches", ROOT / "scores", ROOT / "plans")
            for path in sorted(directory.glob("*.json"))
        },
        "runtime_verification_sha256": sha256(ROOT / "runtime-verification.json"),
        "script_sha256": sha256(ROOT / "system_panel.py"),
    }
    if not overlap_pairs:
        raise RuntimeError("generation and scoring did not overlap")
    write_once(ROOT / "exact-summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))


def _load_confirmation_batch(label, track):
    from scoring.cache import inspect_exact_cache_batch, validate_exact_cache
    from validator.production.batches import (
        ConfirmationBatch,
        _validate_plan_case_binding,
        validate_selected_sample_cache,
    )
    from validator.production.plan import (
        BatchPlan,
        DenseReference,
        ReadyObject,
        ReadySampleCache,
    )

    if label not in confirmation_labels():
        raise ValueError("full-video evidence requires a reserved confirmation label")
    record = json.loads((ROOT / "batches" / f"{label}.json").read_text())
    plan = BatchPlan.decode((ROOT / "plans" / f"{label}.json").read_bytes())
    if record["plan_digest"] != plan.plan_digest or plan.track_spec_digest != track.digest:
        raise RuntimeError(f"full-video plan binding failed: {label}")
    exact_path = Path(record["exact_path"])
    exact = inspect_exact_cache_batch(exact_path, max_bytes=track.cache.exact_max_bytes)
    if exact.sha256 != record["exact_sha256"]:
        raise RuntimeError(f"full-video exact binding failed: {label}")
    references = []
    for value in record["dense_references"]:
        video_path = Path(value["video_path"])
        latent_path = Path(value["latent_path"])
        if (
            ROOT.resolve() not in video_path.resolve().parents
            or ROOT.resolve() not in latent_path.resolve().parents
            or sha256(video_path) != value["video_sha256"]
            or sha256(latent_path) != value["latent_sha256"]
        ):
            raise RuntimeError(f"full-video dense reference changed: {label}")
        reference = DenseReference(
            value["rollout_id"],
            ReadyObject(
                f"videos/{value['video_sha256']}.mp4",
                value["video_sha256"],
                video_path.stat().st_size,
            ),
            ReadyObject(
                f"latents/{value['latent_sha256']}.safetensors",
                value["latent_sha256"],
                latent_path.stat().st_size,
            ),
        )
        references.append((reference, latent_path))
    sample_cache = ROOT / "batch-data" / label / "sample-cache"
    if Path(record["selected_cache_path"]).resolve() != sample_cache.resolve():
        raise RuntimeError(f"full-video selected input path is invalid: {label}")
    local_inputs = {
        "index": sample_cache / "index.json",
        "text": sample_cache / "text.safetensors",
        "condition": sample_cache / "cond.safetensors",
    }
    for name, document in record["selected_inputs"].items():
        path = local_inputs[name]
        if document is None:
            if path.exists():
                raise RuntimeError(f"full-video selected inputs contain an extra file: {label}")
            continue
        if (
            not path.is_file()
            or path.stat().st_size != document["size_bytes"]
            or sha256(path) != document["sha256"]
        ):
            raise RuntimeError(f"full-video selected input changed: {label}")
    dense_references = tuple(reference for reference, _path in references)
    if {reference.rollout_id for reference in dense_references} != {
        rollout.rollout_id for rollout in plan.rollouts
    }:
        raise RuntimeError(f"full-video references do not cover the plan: {label}")
    selected_documents = record["selected_inputs"]

    def ready_object(value):
        return ReadyObject(value["object_key"], value["sha256"], value["size_bytes"])

    selected_inputs = ReadySampleCache(
        ready_object(selected_documents["index"]),
        ready_object(selected_documents["text"]),
        None
        if selected_documents["condition"] is None
        else ready_object(selected_documents["condition"]),
    )
    exact_object = ReadyObject(
        f"exact/{plan.generation_block}-{exact.sha256}.npz",
        exact.sha256,
        exact_path.stat().st_size,
    )
    ready_plan = plan.ready(
        exact=exact_object,
        dense_references=dense_references,
        sample_cache=selected_inputs,
    )
    validate_selected_sample_cache(sample_cache, ready_plan)
    validate_exact_cache(
        exact_path,
        expected_sha256=exact.sha256,
        expected_track_spec_digest=track.digest,
        max_bytes=track.cache.exact_max_bytes,
        head_dim=track.model.selector.head_dim,
        tile_size=128,
        routes=ready_plan.selected_routes,
        strata=track.scoring.stratum_weight_map,
        sampled_route_count=track.scoring.sampled_route_count,
        rollouts_per_stratum=track.scoring.rollouts_per_stratum,
        sampled_step_count=track.scoring.sampled_step_count,
        query_tiles_per_case=track.scoring.query_tiles_per_case,
        dense_device="cuda:0",
    )
    _validate_plan_case_binding(exact_path, ready_plan, track)
    if (
        exact.generated_block != ready_plan.generation_block
        or exact.plan_digest != ready_plan.plan_digest
        or exact.track_spec_digest != track.digest
        or exact.seed_block_hash.lower() != ready_plan.block_hash.lower()
        or exact.artifact_manifest_sha256 != track.artifact_manifest_sha256
        or exact.source_pool_merkle_root != track.source_pool_merkle_root
        or exact.source_proofs != tuple(proof.document() for proof in ready_plan.source_proofs)
        or exact.dense_references
        != tuple(reference.document() for reference in ready_plan.dense_references)
        or exact.selected_inputs != ready_plan.sample_cache.document()
    ):
        raise RuntimeError(f"full-video exact data conflicts with its READY plan: {label}")
    return ConfirmationBatch(ready_plan, exact, tuple(references), sample_cache), record


def _full_video_eligibility(track) -> dict[str, object]:
    exact = json.loads((ROOT / "exact-summary.json").read_text())
    if (
        exact["candidate_track_spec_digest"] != track.digest
        or exact["runtime_contract_sha256"] != RUNTIME_CONTRACT
        or set(exact["model_identities"]) != set(EXPECTED_IDENTITIES)
        or any(
            {key: exact["model_identities"][name][key] for key in expected} != expected
            for name, expected in EXPECTED_IDENTITIES.items()
        )
    ):
        raise RuntimeError("exact summary cannot authorize full-video evidence")
    production = [row for row in exact["batches"] if row["partition"] == "production"]
    if (
        len(production) != PRODUCTION_EXACT
        or {row["label"] for row in production}
        != {f"production-{index:02d}" for index in range(PRODUCTION_EXACT)}
    ):
        raise RuntimeError("exact summary has an incomplete production matrix")
    decisions = {}
    for name in EXPECTED_IDENTITIES:
        bounds = [row["controls"][name]["gain_lower_bound"] for row in production]
        maximum = max(bounds)
        eligible = name in REQUIRED_FULL_VIDEO_IDENTITIES
        decisions[name] = {
            "eligible": eligible,
            "maximum_production_exact_gain_lower_bound": maximum,
            "reason": (
                "shared_baseline"
                if name == "baseline_self"
                else "complete_zoo_for_dynamic_incumbent_closure"
                if name in REQUIRED_FULL_VIDEO_IDENTITIES
                else "reached_or_within_one_minimum_gain_of_exact_gate"
                if eligible
                else "exact_rejected_below_approach_band"
            ),
        }
    return {
        "format": "veda-system-panel-full-video-eligibility-v1",
        "candidate_track_spec_digest": track.digest,
        "evidence_contract_sha256": EVIDENCE_CONTRACT,
        "runtime_contract_sha256": RUNTIME_CONTRACT,
        "rule": {
            "partition": "production",
            "approach_floor": 0.0,
            "exact_gate": track.scoring.minimum_frontier_gain,
            "definition": (
                "all six frozen identities; covers every reachable incumbent and challenger"
            ),
        },
        "decisions": decisions,
        "exact_summary_sha256": sha256(ROOT / "exact-summary.json"),
    }


def _two_stage_fifo_replay(
    *,
    order,
    exact_labels,
    confirmation_labels,
    exact_bound,
    video_bound,
    exact_threshold,
    video_threshold,
):
    champion = "baseline_self"
    confirmation_index = 0
    steps = []
    for challenger, exact_label in zip(order, exact_labels, strict=True):
        incumbent = champion
        exact_gain = exact_bound(challenger, incumbent, exact_label)
        exact_pass = exact_gain > exact_threshold
        confirmation_label = None
        video_gain = None
        video_pass = False
        if exact_pass:
            if confirmation_index >= len(confirmation_labels):
                raise RuntimeError("two-stage replay exhausted independent confirmation batches")
            confirmation_label = confirmation_labels[confirmation_index]
            confirmation_index += 1
            video_gain = video_bound(challenger, incumbent, confirmation_label)
            video_pass = video_gain > video_threshold
            if video_pass:
                champion = challenger
        steps.append(
            {
                "exact_label": exact_label,
                "confirmation_label": confirmation_label,
                "incumbent": incumbent,
                "challenger": challenger,
                "exact_gain_lower_bound": exact_gain,
                "exact_pass": exact_pass,
                "full_video_gain_lower_bound": video_gain,
                "full_video_pass": video_pass,
                "promoted": exact_pass and video_pass,
                "champion_after": champion,
            }
        )
    return {
        "steps": steps,
        "terminal_champion": champion,
        "confirmation_batches_consumed": confirmation_index,
    }


def full_video() -> None:
    from scoring.video import paired_video_gain_bound
    from validator.production.confirmation import ConfirmationWorkerPool

    config, track, _manifest = load_runtime_context(evaluator=True)
    from scoring.model import validate_model

    identities = {
        name: asdict(validate_model(
            path, schema=track.model.tensor_schema, max_bytes=track.model.max_bytes
        ))
        for name, path in CONTROLS.items()
    }
    for name, expected in EXPECTED_IDENTITIES.items():
        if {key: identities[name][key] for key in expected} != expected:
            raise RuntimeError(f"full-video control identity changed: {name}")
    eligibility = _full_video_eligibility(track)
    write_once(ROOT / "full-video-eligibility.json", eligibility)
    selected = {
        name: CONTROLS[name]
        for name, decision in eligibility["decisions"].items()
        if decision["eligible"]
    }
    if "baseline_self" not in selected:
        raise RuntimeError("full-video panel omitted its shared baseline")
    pool = ConfirmationWorkerPool(config, track, timeout_seconds=48 * 60 * 60)
    budget_costs = {
        budget.target_blocks: budget.calibrated_cost for budget in track.scoring.budgets
    }
    for label in confirmation_labels():
        output = ROOT / "full-video" / f"{label}.json"
        if output.is_file():
            continue
        batch, batch_record = _load_confirmation_batch(label, track)
        started_ns = time.time_ns()
        record_event("full_video_started", label=label, identities=tuple(selected))
        model_cells = pool.score_models(model_paths=selected, batch=batch)
        finished_ns = time.time_ns()
        baseline = model_cells["baseline_self"]
        expected_cells = len(batch.plan.rollouts) * len(track.scoring.budgets)
        if any(len(cells) != expected_cells for cells in model_cells.values()):
            raise RuntimeError(f"full-video result has incomplete cells: {label}")
        controls = {}
        for name, decision in eligibility["decisions"].items():
            if not decision["eligible"]:
                controls[name] = {"eligible": False, "reason": decision["reason"]}
                continue
            cells = model_cells[name]
            paired = paired_video_gain_bound(
                cells,
                baseline,
                stratum_weights=track.scoring.stratum_weight_map,
                budget_costs=budget_costs,
                confidence=track.match_confidence,
                resamples=track.scoring.bootstrap_resamples,
                seed=track.scoring.bootstrap_seed,
            )
            raw_cells = [asdict(cell) for cell in cells]
            controls[name] = {
                "eligible": True,
                "score": paired.challenger_score,
                "baseline_score": paired.incumbent_score,
                "gain_lower_bound": paired.gain_lower_bound,
                "promoted": paired.gain_lower_bound > track.full_video.minimum_gain,
                "cells": raw_cells,
                "cells_sha256": hashlib.sha256(canonical(raw_cells)).hexdigest(),
                "raw_output_sha256": [
                    {
                        "budget": cell.budget,
                        "output_sha256": cell.output_sha256,
                        "rollout_id": cell.rollout_id,
                        "stratum": cell.stratum,
                    }
                    for cell in cells
                ],
            }
        document = {
            "format": "veda-system-panel-full-video-v1",
            "label": label,
            "partition": batch_record["partition"],
            "candidate_track_spec_digest": track.digest,
            "artifact_manifest_sha256": track.artifact_manifest_sha256,
            "evidence_contract_sha256": EVIDENCE_CONTRACT,
            "runtime_contract_sha256": RUNTIME_CONTRACT,
            "plan_digest": batch.plan.plan_digest,
            "exact_sha256": batch.exact.sha256,
            "selected_inputs": batch_record["selected_inputs"],
            "dense_references": batch_record["dense_references"],
            "model_identities": identities,
            "control_provenance": CONTROL_PROVENANCE,
            "controls": controls,
            "started_ns": started_ns,
            "finished_ns": finished_ns,
            "seconds": (finished_ns - started_ns) / 1e9,
        }
        write_once(output, document)
        record_event("full_video_finished", label=label, output_sha256=sha256(output))
        print(json.dumps({"label": label, "seconds": document["seconds"]}), flush=True)
    write_once(
        ROOT / "full-video-complete.json",
        {
            "format": "veda-system-panel-full-video-complete-v1",
            "labels": confirmation_labels(),
            "eligibility_sha256": sha256(ROOT / "full-video-eligibility.json"),
        },
    )


def aggregate_full_video() -> None:
    from scoring.video import VideoCell, paired_video_gain_bound

    config, track, _manifest = load_runtime_context(evaluator=True)
    del config
    eligibility = json.loads((ROOT / "full-video-eligibility.json").read_text())
    rows = []
    for label in confirmation_labels():
        path = ROOT / "full-video" / f"{label}.json"
        value = json.loads(path.read_text())
        if (
            value["label"] != label
            or value["candidate_track_spec_digest"] != track.digest
            or value["runtime_contract_sha256"] != RUNTIME_CONTRACT
            or set(value["controls"]) != set(EXPECTED_IDENTITIES)
            or set(value["model_identities"]) != set(EXPECTED_IDENTITIES)
            or any(
                {key: value["model_identities"][name][key] for key in expected} != expected
                for name, expected in EXPECTED_IDENTITIES.items()
            )
            or value["partition"] != label.split("-", 1)[0]
        ):
            raise RuntimeError(f"full-video evidence binding failed: {label}")
        for name, decision in eligibility["decisions"].items():
            control = value["controls"][name]
            if control["eligible"] != decision["eligible"]:
                raise RuntimeError(f"full-video eligibility changed: {label}/{name}")
            if control["eligible"]:
                cells = control["cells"]
                if (
                    hashlib.sha256(canonical(cells)).hexdigest() != control["cells_sha256"]
                    or any(
                        not isinstance(cell["output_sha256"], str)
                        or len(cell["output_sha256"]) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in cell["output_sha256"]
                        )
                        for cell in cells
                    )
                    or control["raw_output_sha256"]
                    != [
                        {
                            "budget": cell["budget"],
                            "output_sha256": cell["output_sha256"],
                            "rollout_id": cell["rollout_id"],
                            "stratum": cell["stratum"],
                        }
                        for cell in cells
                    ]
                    or {cell["stratum"] for cell in cells} != set(track.scoring.stratum_weight_map)
                    or len(cells) != (
                        track.scoring.rollouts_per_stratum
                        * len(track.scoring.stratum_weight_map) * len(track.scoring.budgets)
                    )
                    or len({(cell["rollout_id"], cell["budget"]) for cell in cells}) != len(cells)
                    or {cell["budget"] for cell in cells}
                    != {budget.target_blocks for budget in track.scoring.budgets}
                    or any(
                        len({cell["rollout_id"] for cell in cells if cell["stratum"] == stratum})
                        != track.scoring.rollouts_per_stratum
                        for stratum in track.scoring.stratum_weight_map
                    )
                    or len({cell["rollout_id"] for cell in cells})
                    != track.scoring.rollouts_per_stratum * len(track.scoring.stratum_weight_map)
                ):
                    raise RuntimeError(f"full-video cells are incomplete: {label}/{name}")
        if value["controls"]["baseline_self"]["gain_lower_bound"] != 0.0:
            raise RuntimeError(f"full-video self bound is not zero: {label}")
        costs = {budget.target_blocks: budget.calibrated_cost for budget in track.scoring.budgets}
        baseline = tuple(VideoCell(**cell) for cell in value["controls"]["baseline_self"]["cells"])
        for name, control in value["controls"].items():
            if not control["eligible"]:
                continue
            paired = paired_video_gain_bound(
                tuple(VideoCell(**cell) for cell in control["cells"]), baseline,
                stratum_weights=track.scoring.stratum_weight_map, budget_costs=costs,
                confidence=track.match_confidence, resamples=track.scoring.bootstrap_resamples,
                seed=track.scoring.bootstrap_seed,
            )
            if (
                control["gain_lower_bound"] != paired.gain_lower_bound
                or control["score"] != paired.challenger_score
                or control["promoted"] != (paired.gain_lower_bound > track.full_video.minimum_gain)
            ):
                raise RuntimeError(f"full-video aggregate differs from raw cells: {label}/{name}")
        reverse = paired_video_gain_bound(
            baseline,
            tuple(VideoCell(**cell) for cell in value["controls"]["shuffled_heads"]["cells"]),
            stratum_weights=track.scoring.stratum_weight_map, budget_costs=costs,
            confidence=track.match_confidence, resamples=track.scoring.bootstrap_resamples,
            seed=track.scoring.bootstrap_seed,
        ).gain_lower_bound
        value["reverse_veda_vs_shuffled_bound"] = reverse
        value["directional_gates"] = (
            _directional_gates(value["controls"], reverse, track.full_video.minimum_gain, label)
            if value["partition"] == "production" else None
        )
        rows.append(value)
    stability = {}
    for name, decision in eligibility["decisions"].items():
        if not decision["eligible"]:
            stability[name] = {"eligible": False, "hard_gate": False}
            continue
        production_outcomes = [
            row["controls"][name]["promoted"] for row in rows if row["partition"] == "production"
        ]
        audit_outcomes = [
            row["controls"][name]["promoted"] for row in rows if row["partition"] == "audit"
        ]
        production_rate = sum(production_outcomes) / len(production_outcomes)
        audit_rate = sum(audit_outcomes) / len(audit_outcomes)
        audit_shift = audit_rate - production_rate
        stability[name] = {
            "eligible": True,
            "production_outcome": production_outcomes[0],
            "production_outcome_flips": sum(
                left != right
                for left, right in zip(production_outcomes, production_outcomes[1:], strict=False)
            ),
            "production_stable": len(set(production_outcomes)) == 1,
            "production_promotion_rate": production_rate,
            "audit_outcome_flips": sum(
                left != right
                for left, right in zip(audit_outcomes, audit_outcomes[1:], strict=False)
            ),
            "audit_promotion_rate": audit_rate,
            "audit_promotion_rate_shift": audit_shift,
            "audit_shift_pass": (
                abs(audit_shift) <= AUDIT_FULL_VIDEO_MAX_HARD_PROMOTION_RATE_SHIFT
            ),
            "hard_gate": name in HARD_STABILITY_IDENTITIES,
        }
    if any(
        value.get("production_stable") is False
        for name, value in stability.items()
        if name in HARD_STABILITY_IDENTITIES
    ):
        raise RuntimeError(f"production full-video outcome flipped: {stability}")
    if any(
        value.get("audit_shift_pass") is False
        for name, value in stability.items()
        if name in HARD_STABILITY_IDENTITIES
    ):
        raise RuntimeError(f"audit full-video shift exceeded its frozen threshold: {stability}")
    partitions = {}
    for partition in ("production", "audit"):
        selected = [row for row in rows if row["partition"] == partition]
        expected = PRODUCTION_CONFIRMATION if partition == "production" else AUDIT_CONFIRMATION
        if len(selected) != expected:
            raise RuntimeError(f"full-video partition is incomplete: {partition}")
        partitions[partition] = {
            name: {
                "mean_score": fsum(row["controls"][name]["score"] for row in selected)
                / len(selected),
                "mean_gain_lower_bound": fsum(
                    row["controls"][name]["gain_lower_bound"] for row in selected
                )
                / len(selected),
                "promotion_rate": sum(row["controls"][name]["promoted"] for row in selected)
                / len(selected),
            }
            for name, decision in eligibility["decisions"].items()
            if decision["eligible"]
        }
    shift = {
        name: {
            key: partitions["audit"][name][key] - partitions["production"][name][key]
            for key in ("mean_score", "mean_gain_lower_bound", "promotion_rate")
        }
        for name in partitions["production"]
    }
    by_label = {row["label"]: row for row in rows}
    confirmation_batches = tuple(
        f"production-{index:02d}"
        for index in range(PRODUCTION_EXACT, PRODUCTION_EXACT + PRODUCTION_CONFIRMATION)
    )
    budget_costs = {
        budget.target_blocks: budget.calibrated_cost for budget in track.scoring.budgets
    }
    challengers = tuple(name for name in EXPECTED_IDENTITIES if name != "baseline_self")

    def exact_replay_bound(challenger, incumbent, exact_label):
        exact_score = json.loads((ROOT / "scores" / f"{exact_label}.json").read_text())
        design = _case_design(exact_score["cells"])
        values = {
            name: [cell["scores"][name] for cell in exact_score["cells"]]
            for name in (challenger, incumbent)
        }
        return _bound(track, values[challenger], values[incumbent], design)

    def video_replay_bound(challenger, incumbent, confirmation_label):
        evidence = by_label[confirmation_label]["controls"]
        if not evidence[challenger]["eligible"] or not evidence[incumbent]["eligible"]:
            raise RuntimeError("exact-qualified selector lacks full-video evidence")
        challenger_cells = tuple(VideoCell(**cell) for cell in evidence[challenger]["cells"])
        incumbent_cells = tuple(VideoCell(**cell) for cell in evidence[incumbent]["cells"])
        return paired_video_gain_bound(
            challenger_cells,
            incumbent_cells,
            stratum_weights=track.scoring.stratum_weight_map,
            budget_costs=budget_costs,
            confidence=track.match_confidence,
            resamples=track.scoring.bootstrap_resamples,
            seed=track.scoring.bootstrap_seed,
        ).gain_lower_bound

    two_stage = {}
    confirmation_dataset = [
        {
            "label": label,
            "sha256": sha256(ROOT / "full-video" / f"{label}.json"),
        }
        for label in confirmation_batches
    ]
    confirmation_dataset_sha256 = hashlib.sha256(canonical(confirmation_dataset)).hexdigest()
    exact_dataset = [
        {"label": f"production-{index:02d}",
         "sha256": sha256(ROOT / "scores" / f"production-{index:02d}.json")}
        for index in range(5)
    ]
    exact_dataset_sha256 = hashlib.sha256(canonical(exact_dataset)).hexdigest()
    for replay_name, order, batch_labels, counterfactual in (
        (
            "forward",
            challengers,
            tuple(f"production-{index:02d}" for index in range(5)),
            False,
        ),
        (
            "reverse",
            tuple(reversed(challengers)),
            tuple(f"production-{index:02d}" for index in range(5)),
            True,
        ),
    ):
        try:
            replay = _two_stage_fifo_replay(
                order=order,
                exact_labels=batch_labels,
                confirmation_labels=confirmation_batches,
                exact_bound=exact_replay_bound,
                video_bound=video_replay_bound,
                exact_threshold=track.scoring.minimum_frontier_gain,
                video_threshold=track.full_video.minimum_gain,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{replay_name} replay failed: {exc}") from exc
        replay["counterfactual"] = counterfactual
        replay["confirmation_dataset_sha256"] = confirmation_dataset_sha256
        replay["exact_dataset_sha256"] = exact_dataset_sha256
        two_stage[replay_name] = replay
    reference_scores = partitions["production"]
    best_score = max(row["mean_score"] for row in reference_scores.values())
    best_identity = next(
        name for name, row in reference_scores.items() if row["mean_score"] == best_score
    )
    for replay in two_stage.values():
        terminal = replay["terminal_champion"]
        terminal_score = reference_scores[terminal]["mean_score"]
        replay["terminal_champion_reference_mean_full_video_score"] = terminal_score
        replay["champion_regret"] = best_score - terminal_score
    order_sensitivity = {
        "reference_partition": "production",
        "reference_metric": "mean full-video score over the frozen production confirmations",
        "best_reference_identity": best_identity,
        "best_reference_score": best_score,
        "exact_dataset": exact_dataset,
        "exact_dataset_sha256": exact_dataset_sha256,
        "confirmation_dataset": confirmation_dataset,
        "confirmation_dataset_sha256": confirmation_dataset_sha256,
        "reverse_reuses_forward_confirmation_dataset": True,
        "reverse_reuses_forward_exact_dataset": True,
        "terminal_champions_equal": (
            two_stage["forward"]["terminal_champion"] == two_stage["reverse"]["terminal_champion"]
        ),
        "reverse_minus_forward_champion_regret": (
            two_stage["reverse"]["champion_regret"] - two_stage["forward"]["champion_regret"]
        ),
    }
    summary = {
        "format": "veda-system-panel-full-video-summary-v2",
        "candidate_track_spec_digest": track.digest,
        "artifact_manifest_sha256": track.artifact_manifest_sha256,
        "evidence_contract_sha256": EVIDENCE_CONTRACT,
        "runtime_contract_sha256": RUNTIME_CONTRACT,
        "minimum_gain": track.full_video.minimum_gain,
        "cluster_unit": "source_video",
        "shared_baseline_per_batch": True,
        "full_video_p95_seconds": _percentile(
            [row["seconds"] for row in rows],
            0.95,
        ),
        "percentile_definition": "nearest-rank",
        "eligibility": eligibility,
        "control_provenance": CONTROL_PROVENANCE,
        "training_source_manifest": TRAINING_SOURCE_MANIFEST,
        "source_quarantine_evidence": SOURCE_QUARANTINE_EVIDENCE,
        "batches": [
            {
                "label": row["label"],
                "partition": row["partition"],
                "seconds": row["seconds"],
                "sha256": sha256(ROOT / "full-video" / f"{row['label']}.json"),
            }
            for row in rows
        ],
        "outcome_stability": stability,
        "stability_gate": {
            "production_hard_identities": sorted(HARD_STABILITY_IDENTITIES),
            "production_max_outcome_flips": 0,
            "audit_full_video_max_hard_promotion_rate_shift": (
                AUDIT_FULL_VIDEO_MAX_HARD_PROMOTION_RATE_SHIFT
            ),
            "audit_shift_acceptance": _audit_shift_acceptance(
                audit_batches=AUDIT_CONFIRMATION,
                maximum_shift=AUDIT_FULL_VIDEO_MAX_HARD_PROMOTION_RATE_SHIFT,
            ),
        },
        "partition_report": partitions,
        "production_to_audit_shift": shift,
        "launch_calibration_two_stage_fifo_replay": two_stage,
        "two_stage_order_sensitivity": order_sensitivity,
        "block_provenance": {
            "kind": "synthetic_deterministic_hash",
            "live_chain_lifecycle_evidence": False,
            "separate_live_chain_evidence_required": True,
        },
        "exact_summary_sha256": sha256(ROOT / "exact-summary.json"),
        "raw_output_sha256": {
            row["label"]: {
                name: control["raw_output_sha256"]
                for name, control in row["controls"].items()
                if control["eligible"]
            }
            for row in rows
        },
    }
    write_once(ROOT / "full-video-summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("generate")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--worker", required=True, type=int)
    commands.add_parser("aggregate")
    commands.add_parser("full-video")
    commands.add_parser("aggregate-full-video")
    commands.add_parser("self-test")
    arguments = parser.parse_args()
    if arguments.command != "self-test":
        if arguments.binding is None:
            parser.error("--binding is required; intermediate images cannot launch a formal panel")
        configure(arguments.binding)
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "generate":
        generate()
    elif arguments.command == "score":
        score(arguments.worker)
    elif arguments.command == "aggregate":
        aggregate()
    elif arguments.command == "full-video":
        full_video()
    elif arguments.command == "aggregate-full-video":
        aggregate_full_video()
    else:
        self_test()


if __name__ == "__main__":
    main()
