"""Restore authoritative READY evaluation batches from private object storage."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose
from pathlib import Path

import numpy as np

from scoring.cache import (
    CacheLoadError,
    ExactCacheBatch,
    ExactCacheReader,
    inspect_exact_cache_batch,
    validate_exact_cache,
)
from scoring.exact import GLOBAL_SEGMENT, REFERENCE_SEGMENT, TARGET_SEGMENT
from scoring.spec import TrackSpec
from validator.production.commitment import verify_source_proof
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    build_plan,
    select_query_tiles,
)
from validator.production.storage import PrivateObjectStore


@dataclass(frozen=True, slots=True)
class ConfirmationBatch:
    plan: BatchPlan
    exact: ExactCacheBatch
    latent_paths: tuple[tuple[DenseReference, Path], ...]


@dataclass(frozen=True, slots=True)
class ArchivedStage:
    score: float
    score_block: int
    metrics: Mapping[str, object]


class ReadyBatchSource:
    """Resolve challenge-bound READY plans and restore their immutable content."""

    def __init__(
        self,
        *,
        track: TrackSpec,
        objects: PrivateObjectStore,
        cache_dir: str | Path,
    ) -> None:
        self.track = track
        self.objects = objects
        self.cache_dir = Path(cache_dir)

    def archive_model(self, path: str | Path, sha256: str, size_bytes: int) -> None:
        """Persist a validated selector before it can affect tournament state."""

        if Path(path).stat().st_size != size_bytes:
            raise RuntimeError("validated selector size changed before archival")
        self.objects.put_content(
            path,
            sha256=sha256,
            object_name=f"models/{sha256}.safetensors",
        )

    def restore_model(self, sha256: str, size_bytes: int) -> Path:
        return self.objects.download_content(
            f"models/{sha256}.safetensors",
            self.cache_dir / "models" / f"{sha256}.safetensors",
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def archive_stage(self, task: object, result: object, score_block: int) -> None:
        """Persist a scored nonterminal stage before changing local tournament state."""

        document = {
            "format": "veda-stage-result-v1",
            "track_spec_digest": self.track.digest,
            "challenge_id": task.challenge_id,
            "submission_id": task.submission_id,
            "hotkey": task.hotkey,
            "repo_id": task.repo_id,
            "revision": task.revision,
            "file_sha256": task.file_sha256,
            "stage": task.state.value,
            "score": result.score,
            "score_block": score_block,
            "metrics": dict(result.metrics),
        }
        self.objects.create_plan(
            self._stage_name(task.challenge_id, task.state.value), _json(document)
        )

    def restore_stage(self, task: object) -> ArchivedStage | None:
        """Restore an exact immutable stage result after loss of local SQLite state."""

        stored = self.objects.get_optional(self._stage_name(task.challenge_id, task.state.value))
        if stored is None:
            return None
        try:
            document = json.loads(stored.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("archived stage result is invalid") from exc
        expected_identity = {
            "format": "veda-stage-result-v1",
            "track_spec_digest": self.track.digest,
            "challenge_id": task.challenge_id,
            "submission_id": task.submission_id,
            "hotkey": task.hotkey,
            "repo_id": task.repo_id,
            "revision": task.revision,
            "file_sha256": task.file_sha256,
            "stage": task.state.value,
        }
        if (
            not isinstance(document, dict)
            or _json(document) != stored.body
            or any(document.get(key) != value for key, value in expected_identity.items())
            or set(document) != set(expected_identity) | {"score", "score_block", "metrics"}
            or isinstance(document.get("score"), bool)
            or not isinstance(document.get("score"), (int, float))
            or not 0 <= document["score"] <= 1
            or type(document.get("score_block")) is not int
            or document["score_block"] < task.position.block
            or not isinstance(document.get("metrics"), dict)
        ):
            raise RuntimeError("archived stage result conflicts with its challenge")
        return ArchivedStage(
            float(document["score"]),
            document["score_block"],
            document["metrics"],
        )

    @staticmethod
    def _stage_name(challenge_id: str, stage: str) -> str:
        return f"stages/{challenge_id}/{stage}.json"

    def exact_batch(self, challenge_id: str) -> ExactCacheBatch:
        plan = self._plan_for(challenge_id, confirmation=False)
        return self._restore_exact(plan)

    def _restore_exact(self, plan: BatchPlan) -> ExactCacheBatch:
        if plan.exact is None:
            raise RuntimeError("READY plan omitted its exact object")
        destination = self.cache_dir / f"{plan.generation_block}-{plan.exact.sha256}.npz"
        path = self.objects.download_content(
            plan.exact.object_key,
            destination,
            sha256=plan.exact.sha256,
            size_bytes=plan.exact.size_bytes,
        )
        try:
            batch = inspect_exact_cache_batch(path, max_bytes=self.track.cache.exact_max_bytes)
            validate_exact_cache(
                path,
                expected_sha256=batch.sha256,
                expected_track_spec_digest=self.track.digest,
                max_bytes=self.track.cache.exact_max_bytes,
                head_dim=self.track.model.selector.head_dim,
                tile_size=128,
                routes=plan.selected_routes,
                strata=self.track.scoring.stratum_weight_map,
                sampled_route_count=self.track.scoring.sampled_route_count,
                rollouts_per_stratum=self.track.scoring.rollouts_per_stratum,
                sampled_step_count=self.track.scoring.sampled_step_count,
                query_tiles_per_case=self.track.scoring.query_tiles_per_case,
            )
            _validate_plan_case_binding(path, plan, self.track)
        except (OSError, CacheLoadError, ValueError) as exc:
            raise RuntimeError("restored exact batch is invalid") from exc
        if (
            batch.sha256 != plan.exact.sha256
            or batch.generated_block != plan.generation_block
            or batch.plan_digest != plan.plan_digest
            or batch.track_spec_digest != self.track.digest
            or batch.seed_block_hash.lower() != plan.block_hash.lower()
            or batch.artifact_manifest_sha256 != self.track.artifact_manifest_sha256
            or batch.source_pool_merkle_root != self.track.source_pool_merkle_root
            or batch.source_proofs != tuple(proof.document() for proof in plan.source_proofs)
            or batch.dense_references
            != tuple(reference.document() for reference in plan.dense_references)
        ):
            raise RuntimeError("restored exact batch conflicts with its READY plan")
        return batch

    def confirmation_batch(self, challenge_id: str) -> ConfirmationBatch:
        plan = self._plan_for(challenge_id, confirmation=True)
        destination = self.cache_dir / "dense" / plan.plan_digest
        references = []
        for reference in plan.dense_references:
            latent = reference.latent
            path = self.objects.download_content(
                latent.object_key,
                destination / f"{latent.sha256}.safetensors",
                sha256=latent.sha256,
                size_bytes=latent.size_bytes,
            )
            references.append((reference, path))
        return ConfirmationBatch(plan, self._restore_exact(plan), tuple(references))

    def batch_challenge_ids(self, plan_digest: str) -> tuple[str, ...]:
        """Return every challenge sharing a plan, for delayed public disclosure."""

        if len(plan_digest) != 64:
            raise ValueError("plan digest must contain 32 bytes")
        try:
            bytes.fromhex(plan_digest)
        except ValueError as exc:
            raise ValueError("plan digest must be hexadecimal") from exc
        matches = []
        for name in self.objects.list_names("plans/"):
            if not name.startswith("plans/") or not name.endswith(".json"):
                raise RuntimeError("private plan index contains an unexpected object")
            plan = BatchPlan.decode(self.objects.get(name).body)
            self._validate_plan(name, plan)
            if plan.plan_digest == plan_digest:
                matches.append(plan)
        if len(matches) != 1:
            raise RuntimeError("evaluation batch does not identify one private plan")
        plan = matches[0]
        if plan.state != "READY":
            raise RuntimeError("evaluation batch plan is not READY")
        return tuple(dict.fromkeys(plan.exact_challenge_ids + plan.confirmation_challenge_ids))

    def prune_disclosed(self, terminal_challenge_ids: set[str]) -> int:
        """Remove only local copies of READY batches whose whole group is terminal."""

        removed = 0
        for name in self.objects.list_names("plans/"):
            if not name.startswith("plans/") or not name.endswith(".json"):
                raise RuntimeError("private plan index contains an unexpected object")
            plan = BatchPlan.decode(self.objects.get(name).body)
            self._validate_plan(name, plan)
            references = set(plan.exact_challenge_ids + plan.confirmation_challenge_ids)
            if plan.state != "READY" or not references or not references <= terminal_challenge_ids:
                continue
            assert plan.exact is not None
            exact = self.cache_dir / f"{plan.generation_block}-{plan.exact.sha256}.npz"
            if exact.is_file():
                exact.unlink()
                removed += 1
            dense = self.cache_dir / "dense" / plan.plan_digest
            if dense.is_dir():
                shutil.rmtree(dense)
                removed += 1
        return removed

    def _plan_for(self, challenge_id: str, *, confirmation: bool) -> BatchPlan:
        if not challenge_id:
            raise ValueError("challenge ID cannot be empty")
        matches = []
        for name in self.objects.list_names("plans/"):
            if not name.startswith("plans/") or not name.endswith(".json"):
                raise RuntimeError("private plan index contains an unexpected object")
            plan = BatchPlan.decode(self.objects.get(name).body)
            self._validate_plan(name, plan)
            identifiers = (
                plan.confirmation_challenge_ids if confirmation else plan.exact_challenge_ids
            )
            if challenge_id in identifiers:
                matches.append(plan)
        if len(matches) > 1:
            raise RuntimeError("challenge is assigned to multiple private plans")
        if matches and matches[0].state == "READY":
            return matches[0]
        if matches:
            raise RuntimeError("challenge batch is not READY")
        raise RuntimeError("challenge has no scheduled private batch")

    def _validate_plan(self, name: str, plan: BatchPlan) -> None:
        if name != f"plans/{plan.generation_block}.json":
            raise RuntimeError("private plan key disagrees with its generation block")
        if plan.track_spec_digest != self.track.digest:
            raise RuntimeError("private plan belongs to another track")
        if any(
            not verify_source_proof(proof, self.track.source_pool_merkle_root)
            for proof in plan.source_proofs
        ):
            raise RuntimeError("private plan source proof is invalid")
        rebuilt = build_plan(
            track_spec_digest=self.track.digest,
            generation_block=plan.generation_block,
            block_hash=plan.block_hash,
            track_start_block=self.track.track_start_block,
            minimum_generation_delay_blocks=self.track.cache.minimum_generation_delay_blocks,
            generation_interval_blocks=self.track.cache.generation_interval_blocks,
            exact_challenge_ids=plan.exact_challenge_ids,
            confirmation_challenge_ids=plan.confirmation_challenge_ids,
            source_proofs=plan.source_proofs,
            scoring=self.track.scoring,
            available_routes=self.track.model.selector.routes,
        )
        if rebuilt.planned_document() != plan.planned_document():
            raise RuntimeError("private plan conflicts with the deterministic track schedule")


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _validate_plan_case_binding(path: Path, plan: BatchPlan, track: TrackSpec) -> None:
    """Bind every exact case and deterministic query sample to its frozen plan."""

    reader = ExactCacheReader(
        path,
        expected_sha256=plan.exact.sha256 if plan.exact is not None else "",
        max_bytes=track.cache.exact_max_bytes,
    )
    cases = reader.metadata["cases"]
    rollouts = {rollout.rollout_id: rollout for rollout in plan.rollouts}
    steps = {step.index: step for step in plan.selected_steps}
    expected = {
        (route, rollout.rollout_id, step.index)
        for route in plan.selected_routes
        for rollout in plan.rollouts
        for step in plan.selected_steps
    }
    actual = {(case["route"], case["rollout_id"], case["step_index"]) for case in cases}
    if actual != expected or len(actual) != len(cases):
        raise CacheLoadError("cache cases do not match the frozen batch plan")

    with np.load(path, allow_pickle=False) as archive:
        for index, case in enumerate(cases):
            rollout = rollouts[case["rollout_id"]]
            step = steps[case["step_index"]]
            if (
                case["route"] not in plan.selected_routes
                or case["source_id"] != rollout.source_id
                or case["stratum"] != rollout.stratum
                or case["step_window"] != step.window
                or isinstance(case["scheduler_timestep"], bool)
                or not isinstance(case["scheduler_timestep"], (int, float))
                or isinstance(case["sigma"], bool)
                or not isinstance(case["sigma"], (int, float))
                or not isclose(float(case["scheduler_timestep"]), step.sigma, abs_tol=1e-7)
                or not isclose(float(case["sigma"]), step.sigma, abs_tol=1e-7)
            ):
                raise CacheLoadError("cache case metadata conflicts with the frozen batch plan")
            try:
                bytes.fromhex(case["tile_plan_hash"])
            except ValueError as exc:
                raise CacheLoadError("cache tile plan hash is invalid") from exc
            if len(case["tile_plan_hash"]) != 64:
                raise CacheLoadError("cache tile plan hash is invalid")

            prefix = f"case_{index:06d}"
            segments = np.asarray(archive[f"{prefix}_key_segments"])
            required = np.asarray(archive[f"{prefix}_required_keys"])
            query_valid = np.asarray(archive[f"{prefix}_query_valid_counts"])
            key_valid = np.asarray(archive[f"{prefix}_key_valid_counts"])
            target = tuple(int(item) for item in np.flatnonzero(segments == TARGET_SEGMENT))
            reference = tuple(int(item) for item in np.flatnonzero(segments == REFERENCE_SEGMENT))
            if not target or not set(np.unique(segments)).issubset(
                {TARGET_SEGMENT, REFERENCE_SEGMENT, GLOBAL_SEGMENT}
            ):
                raise CacheLoadError("cache key segments are invalid")
            case_id = f"{rollout.rollout_id}:{step.index}:{case['route']}"
            selected = select_query_tiles(
                query_selection_seed=plan.query_selection_seed,
                case_id=case_id,
                target_tiles=target,
                reference_tiles=reference,
                count=track.scoring.query_tiles_per_case,
            )
            if tuple(case["query_tile_ids"]) != selected:
                raise CacheLoadError("cache query sample conflicts with the frozen batch plan")
            expected_required = np.zeros(required.shape, dtype=np.bool_)
            for row, query_id in enumerate(selected):
                expected_required[row, query_id] = True
            if not np.array_equal(required, expected_required) or not np.array_equal(
                query_valid, key_valid[list(selected)]
            ):
                raise CacheLoadError("cache query masks conflict with selected query tiles")
