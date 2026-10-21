"""Resolve GPU-local READY bytes and archive small authoritative decisions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isclose
from pathlib import Path

import bittensor as bt
import numpy as np

from scoring.cache import (
    CacheLoadError,
    ExactCacheBatch,
    ExactCacheReader,
    inspect_exact_cache_batch,
    validate_exact_cache,
)
from scoring.exact import GLOBAL_SEGMENT, REFERENCE_SEGMENT, TARGET_SEGMENT
from scoring.model import ValidatedModel
from scoring.spec import TrackSpec
from validator.admission import RejectedAdmission
from validator.chunks import check_deadline
from validator.download import DownloadedModel, download_chunked_file
from validator.production.commitment import canonical_json, verify_source_proof
from validator.production.inventory import ALLOCATION_OBJECT, Inventory
from validator.production.plan import (
    BatchPlan,
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    build_plan,
    select_query_tiles,
    validate_source_selection_sequence,
)
from validator.production.storage import (
    ADMISSIONS_NAMESPACE,
    EXACT_NAMESPACE,
    INPUTS_NAMESPACE,
    LATENTS_NAMESPACE,
    MODELS_NAMESPACE,
    PLANS_NAMESPACE,
    STAGES_NAMESPACE,
    VIDEOS_NAMESPACE,
    LocalBatchStore,
    PrivateObjectStore,
    StorageError,
)
from validator.queue import TERMINAL_STATES


def _submission_timestamp_ms(block: int) -> int:
    """Use the original finalized block's Timestamp, never admission wall-clock."""
    async def read() -> int:
        async with bt.Client("finney") as client:
            info = await client.block_info(block=block)
            if info is None or info.number != block:
                raise RuntimeError("original submission block is unavailable")
            stamp = info.timestamp
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise RuntimeError("submission Timestamp must be timezone-aware")
            return (stamp - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)

    async def bounded() -> int:
        return await asyncio.wait_for(read(), timeout=30)

    return asyncio.run(bounded())


@dataclass(frozen=True, slots=True)
class ConfirmationBatch:
    plan: BatchPlan
    exact: ExactCacheBatch
    latent_paths: tuple[tuple[DenseReference, Path], ...]
    sample_cache_path: Path


@dataclass(frozen=True, slots=True)
class ArchivedStage:
    score: float
    score_block: int
    metrics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DisclosureFile:
    object_name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class BatchDisclosure:
    record_name: str
    record_payload: bytes
    files: tuple[DisclosureFile, ...]


def public_selected_inputs_document(
    selected: ReadySampleCache,
) -> dict[str, dict[str, object] | None]:
    """Derive the one public mapping for the private READY input references."""

    result: dict[str, dict[str, object] | None] = {}
    for kind, item in (
        ("index", selected.index),
        ("text", selected.text),
        ("condition", selected.condition),
    ):
        if item is None:
            result[kind] = None
            continue
        result[kind] = {
            "object": f"batch_inputs/{item.sha256}/manifest.json",
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
    return result


class ReadyBatchSource:
    """Resolve challenge-bound READY plans and restore their immutable content."""

    def __init__(
        self,
        *,
        track: TrackSpec,
        objects: PrivateObjectStore,
        cache_dir: str | Path,
        batch_dir: str | Path | None = None,
        validation_device: str | None = None,
        public_base_url: str | None = None,
    ) -> None:
        self.track = track
        self.objects = objects
        self.cache_dir = Path(cache_dir)
        self.content = LocalBatchStore(
            self.cache_dir / "content" if batch_dir is None else batch_dir
        )
        self.validation_device = validation_device
        self.public_base_url = public_base_url
        self._validated_exact: set[str] = set()
        self.inventory = Inventory(objects, track.digest)

    def _materialize(self, plan: BatchPlan, item: ReadyObject, destination: Path) -> Path:
        try:
            return self.content.materialize(
                destination, sha256=item.sha256, size_bytes=item.size_bytes,
            )
        except StorageError:
            if (self.public_base_url is None or time.time_ns() // 1_000_000
                    < plan.completed_at_ms + self.track.cache.disclosure_delay_seconds * 1000):
                raise
        namespace = (
            "batches" if item == plan.exact else
            "batch_inputs" if plan.sample_cache is not None and item in (
                plan.sample_cache.index, plan.sample_cache.text, plan.sample_cache.condition,
            ) else "batch_assets"
        )
        path = download_chunked_file(
            f"{self.public_base_url}/{namespace}/{item.sha256}/manifest.json",
            cache_dir=self.cache_dir / "public-restored", expected_sha256=item.sha256,
            max_bytes=item.size_bytes, suffix=destination.suffix or ".bin",
        )
        self.content.put(path, sha256=item.sha256, size_bytes=item.size_bytes)
        return self.content.materialize(
            destination, sha256=item.sha256, size_bytes=item.size_bytes,
        )

    def _admission_identity(self, task: object) -> dict[str, object]:
        return {
            "track_spec_digest": self.track.digest,
            "challenge_id": task.challenge_id,
            "submission_id": task.submission_id,
            "hotkey": task.hotkey,
            "repo_id": task.repo_id,
            "revision": task.revision,
            "submission_block": task.position.block,
        }

    def finalized_seed_hash(self, block: int) -> str:
        """Verify pre-opening production blocks not present in the miner scan DB."""
        async def read() -> str:
            async with bt.Client("finney") as client:
                if block > await client.finalized_block():
                    raise RuntimeError("production seed is not finalized")
                return await client._substrate.block_hash(block)

        async def bounded() -> str:
            return await asyncio.wait_for(read(), timeout=30)

        return asyncio.run(bounded())

    def archive_model(
        self,
        task: object,
        path: str | Path,
        validated: ValidatedModel,
    ) -> None:
        """Persist a validated selector and its challenge binding before scoring."""

        if Path(path).stat().st_size != validated.size_bytes:
            raise RuntimeError("validated selector size changed before archival")
        self.objects.put_content(
            path,
            sha256=validated.file_sha256,
            object_name=f"{MODELS_NAMESPACE}/{validated.file_sha256}.safetensors",
        )
        document = {
            "format": "veda-admission-model-v1",
            **self._admission_identity(task),
            "file_sha256": validated.file_sha256,
            "model_digest": validated.model_digest,
            "size_bytes": validated.size_bytes,
            "submission_timestamp_ms": _submission_timestamp_ms(task.position.block),
        }
        self.objects.create_plan(
            self._admission_name(task.challenge_id),
            _json(document),
        )

    def archive_rejection(self, task: object, rejection: RejectedAdmission) -> None:
        """Preserve an observed repository/model failure before changing local state."""

        document = {
            "format": "veda-admission-rejection-v1",
            **self._admission_identity(task),
            "error_code": rejection.code,
            "error_summary": rejection.summary,
            "score_block": rejection.score_block,
            "file_sha256": rejection.file_sha256,
            "model_size_bytes": rejection.model_size_bytes,
            "duplicate_of": rejection.duplicate_of,
        }
        self.objects.create_plan(self._admission_name(task.challenge_id), _json(document))

    def restore_admission(self, task: object) -> DownloadedModel | RejectedAdmission | None:
        """Restore the immutable admission outcome without contacting its repository."""

        stored = self.objects.get_optional(self._admission_name(task.challenge_id))
        if stored is None:
            return None
        try:
            document = json.loads(stored.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("archived admission model is invalid") from exc
        identity = {
            "format": "veda-admission-model-v1",
            **self._admission_identity(task),
        }
        if isinstance(document, dict) and document.get("format") == "veda-admission-rejection-v1":
            identity["format"] = "veda-admission-rejection-v1"
            if (
                _json(document) != stored.body
                or set(document) != set(identity) | {
                    "error_code", "error_summary", "score_block", "file_sha256", "model_size_bytes",
                    "duplicate_of",
                }
                or any(document.get(key) != value for key, value in identity.items())
                or not isinstance(document.get("error_code"), str)
                or not document["error_code"]
                or not isinstance(document.get("error_summary"), str)
                or type(document.get("score_block")) is not int
                or document["score_block"] < task.position.block
                or (
                    document.get("duplicate_of") is not None
                    and not _digest(document["duplicate_of"])
                )
                or (
                    (document["error_code"] in {
                        "REFERENCE_DUPLICATE", "EXACT_DUPLICATE", "DUPLICATE_HOTKEY",
                    })
                    != (document.get("duplicate_of") is not None)
                )
                or (
                    (document.get("file_sha256") is None)
                    != (document.get("model_size_bytes") is None)
                )
                or (
                    document.get("file_sha256") is not None
                    and (
                        not _digest(document["file_sha256"])
                        or type(document["model_size_bytes"]) is not int
                        or document["model_size_bytes"] < 0
                    )
                )
            ):
                raise RuntimeError("archived admission rejection conflicts with its challenge")
            return RejectedAdmission(
                document["error_code"], document["error_summary"], document["score_block"],
                document["file_sha256"], document["model_size_bytes"],
                document["duplicate_of"],
            )
        file_sha256 = document.get("file_sha256") if isinstance(document, dict) else None
        model_digest = document.get("model_digest") if isinstance(document, dict) else None
        size_bytes = document.get("size_bytes") if isinstance(document, dict) else None
        if (
            not isinstance(document, dict)
            or _json(document) != stored.body
            or any(document.get(key) != value for key, value in identity.items())
            or set(document) != set(identity) | {
                "file_sha256", "model_digest", "size_bytes", "submission_timestamp_ms",
            }
            or type(document.get("submission_timestamp_ms")) is not int
            or document["submission_timestamp_ms"] <= 0
            or not _digest(file_sha256)
            or not _digest(model_digest)
            or type(size_bytes) is not int
            or size_bytes <= 0
        ):
            raise RuntimeError("archived admission model conflicts with its challenge")
        path = self.restore_model(file_sha256, size_bytes)
        return DownloadedModel(
            path=path,
            file_sha256=file_sha256,
            size_bytes=size_bytes,
            etag=None,
            download_host="private-archive",
        )

    def restore_model(self, sha256: str, size_bytes: int) -> Path:
        return self.objects.download_content(
            f"{MODELS_NAMESPACE}/{sha256}.safetensors",
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
        generated_block = document["metrics"].get("cache_generated_block")
        if generated_block is not None and (
            type(generated_block) is not int
            or generated_block < 0
            or generated_block > document["score_block"]
        ):
            raise RuntimeError("archived stage batch conflicts with its decision block")
        if generated_block is not None:
            plan = self._allocated_plan(
                task.challenge_id, confirmation=task.state.value == "CONFIRMING",
            )
            expected_batch = {
                **self.stage_binding(task.challenge_id,
                                     confirmation=task.state.value == "CONFIRMING"),
                "cache_generated_block": plan.generation_block,
                "cache_plan_digest": plan.plan_digest,
                "cache_seed_block_hash": plan.block_hash,
                "cache_sha256": plan.exact.sha256,
            }
            if any(document["metrics"].get(key) != value
                   for key, value in expected_batch.items()):
                raise RuntimeError("archived stage conflicts with its allocated READY batch")
            _, authority = self.inventory.read()
            claim = next(row for row in authority["claims"]
                         if row["challenge_id"] == task.challenge_id
                         and row["stage"] == task.state.value)
            admission = self.objects.get(self._admission_name(task.challenge_id))
            identity = json.loads(admission.body)
            if (claim["file_sha256"] != task.file_sha256
                    or claim["submission_block"] != task.position.block
                    or identity.get("model_digest") != claim["model_digest"]
                    or identity.get("file_sha256") != task.file_sha256
                    or identity.get("track_spec_digest") != self.track.digest
                    or identity.get("challenge_id") != task.challenge_id
                    or _json(identity) != admission.body):
                raise RuntimeError("archived stage conflicts with its allocated model")
        return ArchivedStage(
            float(document["score"]),
            document["score_block"],
            document["metrics"],
        )

    @staticmethod
    def _stage_name(challenge_id: str, stage: str) -> str:
        return f"{STAGES_NAMESPACE}/{challenge_id}/{stage}.json"

    @staticmethod
    def _admission_name(challenge_id: str) -> str:
        return f"{ADMISSIONS_NAMESPACE}/{challenge_id}.json"

    def exact_batch(self, challenge_id: str) -> ExactCacheBatch:
        plan = self._allocated_plan(challenge_id, confirmation=False)
        return self._restore_exact(plan)

    def _restore_exact(self, plan: BatchPlan, *, validate_dense: bool = True) -> ExactCacheBatch:
        if plan.exact is None:
            raise RuntimeError("READY plan omitted its exact object")
        destination = self.cache_dir / f"{plan.generation_block}-{plan.exact.sha256}.npz"
        path = self._materialize(plan, plan.exact, destination)
        try:
            batch = inspect_exact_cache_batch(path, max_bytes=self.track.cache.exact_max_bytes)
            if validate_dense and batch.sha256 not in self._validated_exact:
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
                    dense_device=self.validation_device,
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
            or plan.sample_cache is None
            or batch.sample_cache != plan.sample_cache.document()
        ):
            raise RuntimeError("restored exact batch conflicts with its READY plan")
        if validate_dense:
            self._validated_exact.add(batch.sha256)
        return batch

    def confirmation_batch(self, challenge_id: str) -> ConfirmationBatch:
        plan = self._allocated_plan(challenge_id, confirmation=True)
        destination = self.cache_dir / "dense" / plan.plan_digest
        references = []
        for reference in plan.dense_references:
            latent = reference.latent
            path = self._materialize(plan, latent, destination / f"{latent.sha256}.safetensors")
            references.append((reference, path))
        return ConfirmationBatch(
            plan,
            self._restore_exact(plan),
            tuple(references),
            self._restore_sample_cache(plan),
        )

    def _restore_sample_cache(self, plan: BatchPlan) -> Path:
        selected = plan.sample_cache
        if selected is None:
            raise RuntimeError("READY plan omitted its selected source inputs")
        destination = self.cache_dir / INPUTS_NAMESPACE / plan.plan_digest
        destination.mkdir(parents=True, exist_ok=True)
        objects = (("index.json", selected.index), ("text.safetensors", selected.text))
        for name, item in objects:
            self._materialize(plan, item, destination / name)
        condition_path = destination / "cond.safetensors"
        if selected.condition is None:
            if condition_path.exists():
                condition_path.unlink()
        else:
            self._materialize(plan, selected.condition, condition_path)
        validate_selected_sample_cache(destination, plan)
        return destination

    def claim_ready(
        self, challenge_id: str, *, confirmation: bool, file_sha256: str, model_digest: str,
        submission_block: int,
    ) -> bool:
        """CAS-claim available inventory for the current leased, admitted FIFO stage."""

        stage = "CONFIRMING" if confirmation else "SCORING"
        admission = self.objects.get_optional(self._admission_name(challenge_id))
        if admission is None:
            raise RuntimeError("inventory claim has no durable admission")
        document = json.loads(admission.body)
        if (document.get("format") != "veda-admission-model-v1"
                or document.get("track_spec_digest") != self.track.digest
                or document.get("challenge_id") != challenge_id
                or document.get("file_sha256") != file_sha256
                or document.get("submission_block") != submission_block
                or document.get("model_digest") != model_digest
                or _json(document) != admission.body):
            raise RuntimeError("inventory claim admission identity is invalid")
        plans = tuple(plan for _, plan in self._plans())
        # Selection is a metadata decision at S, not a scan of whatever bytes
        # happen to survive at admission time. Missing bytes recover the bound
        # READY; they must not silently select a different historical batch.
        claim = self.inventory.claim(
            challenge_id=challenge_id, stage=stage,
            file_sha256=file_sha256, model_digest=model_digest,
            plans=plans,
            submission_block=submission_block,
            submission_timestamp_ms=document["submission_timestamp_ms"],
            disclosure_delay_seconds=self.track.cache.disclosure_delay_seconds,
        )
        return claim is not None

    def eligible_submission(self, task: object) -> bool:
        """Admission needs both independent stages before the original submission.

        This reads only authoritative metadata, never GPU availability or payload
        bytes. A temporarily lost original cache is not a new non-admission.
        """
        stamp = _submission_timestamp_ms(task.position.block)
        eligible = [plan for _, plan in self._plans()
                    if Inventory.eligible(plan, task.position.block, stamp,
                                           self.track.cache.disclosure_delay_seconds)]
        return any(
            first.inventory_slot != second.inventory_slot and not {
                proof.leaf.source_family_id for proof in first.source_proofs
            }.intersection(proof.leaf.source_family_id for proof in second.source_proofs)
            for first in eligible for second in eligible
        )

    def stage_binding(self, challenge_id: str, *, confirmation: bool) -> dict[str, object]:
        plan = self._allocated_plan(challenge_id, confirmation=confirmation)
        _, authority = self.inventory.read()
        stage = "CONFIRMING" if confirmation else "SCORING"
        claim = next(row for row in authority["claims"]
                     if row["challenge_id"] == challenge_id and row["stage"] == stage)
        return {
            "cache_ready_sha256": hashlib.sha256(plan.encode()).hexdigest(),
            "cache_completed_block": plan.completed_block,
            "cache_completed_at_ms": plan.completed_at_ms,
            "cache_allocation": claim,
        }

    def inventory_status(self, plans: tuple[BatchPlan, ...]) -> dict[str, object]:
        """Expose commitments/counts only; seeds, proofs and inputs stay private."""

        _, authority = self.inventory.read()
        now = time.time_ns() // 1_000_000
        available = [plan for plan in plans
                     if (plan.state == "PLANNED" or plan.completed_at_ms
                          + self.track.cache.disclosure_delay_seconds * 1000 > now)]
        return {
            "observed_at_ns": now * 1_000_000,
            "ready_batches": sum(plan.state == "READY" for plan in available),
            "producing_batches": sum(plan.state == "PLANNED" for plan in available),
            "target_batches": self.track.cache.inventory_target_batches,
            "max_bytes": self.track.cache.inventory_max_bytes,
            "disclosure_pending_batches": sum(
                plan.state == "READY" and plan.completed_at_ms
                + self.track.cache.disclosure_delay_seconds * 1000 <= now
                and self.objects.get_optional(f"disclosed/{plan.plan_digest}.json") is None
                for plan in plans
            ),
            "commitments": [{
                "plan_digest": plan.plan_digest, "secret_commitment": plan.secret_commitment,
                "generation_block": plan.generation_block, "state": plan.state,
                "completed_at_ms": plan.completed_at_ms,
                "disclose_at_ms": (None if plan.completed_at_ms is None else
                                   plan.completed_at_ms
                                   + self.track.cache.disclosure_delay_seconds * 1000),
                "ready_sha256": (hashlib.sha256(plan.encode()).hexdigest()
                                 if plan.state == "READY" else None),
            } for plan in available],
            "allocated_stages": [(row["challenge_id"], row["stage"])
                                 for row in authority["claims"]],
        }

    def public_disclosure(
        self,
        plan_digest: str,
        *, now_ms: int, deadline: float | None = None,
    ) -> BatchDisclosure | None:
        """Reveal at production age, including unused and still-in-flight data.

        The dataset object never includes a mutable list of its users. Each
        match carries its own immutable allocation; a late scan can bind the
        original pre-expiry submission without rewriting this public record.
        """

        plan = self._plan_by_digest(plan_digest)
        check_deadline(deadline)
        assert plan.completed_block is not None
        if now_ms < plan.completed_at_ms + self.track.cache.disclosure_delay_seconds * 1000:
            return None
        assert plan.exact is not None
        exact = self._restore_exact(plan, validate_dense=False)
        files = [
            DisclosureFile(
                f"batches/{plan.exact.sha256}/manifest.json",
                exact.path,
                plan.exact.sha256,
            )
        ]
        references = []
        directory = self.cache_dir / "disclosures" / plan.plan_digest
        for reference in plan.dense_references:
            check_deadline(deadline)
            public = {"rollout_id": reference.rollout_id}
            for kind, item, suffix in (
                ("video", reference.video, ".mp4"),
                ("latent", reference.latent, ".safetensors"),
            ):
                path = self._materialize(plan, item, directory / f"{item.sha256}{suffix}")
                object_name = f"batch_assets/{item.sha256}/manifest.json"
                files.append(DisclosureFile(object_name, path, item.sha256))
                public[kind] = {
                    "object": object_name,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
            references.append(public)
        selected_inputs = self._restore_sample_cache(plan)
        assert plan.sample_cache is not None
        public_inputs = public_selected_inputs_document(plan.sample_cache)
        for kind, item, name in (
            ("index", plan.sample_cache.index, "index.json"),
            ("text", plan.sample_cache.text, "text.safetensors"),
            ("condition", plan.sample_cache.condition, "cond.safetensors"),
        ):
            if item is None:
                continue
            path = selected_inputs / name
            public = public_inputs[kind]
            assert public is not None
            object_name = public["object"]
            assert isinstance(object_name, str)
            files.append(DisclosureFile(object_name, path, item.sha256))
        record = canonical_json(
            {
                "format": "veda-batch-disclosure-v3",
                "plan": plan.document(),
                "public_exact_object": f"batches/{plan.exact.sha256}/manifest.json",
                "public_dense_references": references,
                "public_selected_inputs": public_inputs,
            }
        )
        check_deadline(deadline)
        return BatchDisclosure(
            f"batch_groups/{plan.plan_digest}.json",
            record,
            tuple(files),
        )

    def due_plan_digests(self, now_ms: int) -> tuple[str, ...]:
        return tuple(
            plan.plan_digest for _, plan in self._plans()
            if plan.state == "READY" and plan.completed_block is not None
            and plan.completed_at_ms + self.track.cache.disclosure_delay_seconds * 1000 <= now_ms
            and self.objects.get_optional(f"disclosed/{plan.plan_digest}.json") is None
        )

    def record_disclosed(self, plan_digest: str) -> None:
        """Persist only after the public immutable record and all bytes verified."""
        plan = self._plan_by_digest(plan_digest)
        self.objects.create_plan(
            f"disclosed/{plan_digest}.json", canonical_json({
                "plan_digest": plan_digest, "exact_sha256": plan.exact.sha256,
            }),
        )

    def protected_plan_digests(
        self, challenges: list[dict[str, object]], scanned_block: int,
    ) -> frozenset[str]:
        pending = [row for row in challenges
                   if row["state"] not in TERMINAL_STATES
                   and row.get("track_spec_digest", self.track.digest) == self.track.digest]
        scanned_time = _submission_timestamp_ms(scanned_block)
        # An RPC failure aborts pruning conservatively; it is not evidence that
        # the model lacked eligible data. Reuse immutable chain times per S.
        submission_times = {
            block: _submission_timestamp_ms(block)
            for block in {row["block_height"] for row in pending}
        }
        return frozenset(
            plan.plan_digest for _, plan in self._plans()
            if plan.state == "PLANNED"
            or plan.completed_at_ms + self.track.cache.disclosure_delay_seconds * 1000
            > scanned_time
            or any(Inventory.eligible(
                plan, row["block_height"], submission_times[row["block_height"]],
                self.track.cache.disclosure_delay_seconds,
            ) for row in pending)
        )

    def _plan_by_digest(self, plan_digest: str) -> BatchPlan:
        if len(plan_digest) != 64:
            raise ValueError("plan digest must contain 32 bytes")
        try:
            bytes.fromhex(plan_digest)
        except ValueError as exc:
            raise ValueError("plan digest must be hexadecimal") from exc
        matches = []
        for _name, plan in self._plans():
            if plan.plan_digest == plan_digest:
                matches.append(plan)
        if len(matches) != 1:
            raise RuntimeError("evaluation batch does not identify one private plan")
        plan = matches[0]
        if plan.state != "READY":
            raise RuntimeError("evaluation batch plan is not READY")
        return plan

    def prune_disclosed(self, protected_plan_digests: frozenset[str]) -> int:
        """Reclaim only verified public bytes not needed by replay or in-flight work."""

        removed = 0
        plans = self._plans()
        published = {
            plan.plan_digest for _name, plan in plans
            if self.objects.get_optional(f"disclosed/{plan.plan_digest}.json") is not None
        }
        retained = {
            item.sha256
            for _name, plan in plans
            if plan.plan_digest not in published or plan.plan_digest in protected_plan_digests
            for item in ready_objects(plan)
        }
        for _name, plan in plans:
            if (plan.state != "READY" or plan.plan_digest not in published
                    or plan.plan_digest in protected_plan_digests):
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
            inputs = self.cache_dir / INPUTS_NAMESPACE / plan.plan_digest
            if inputs.is_dir():
                shutil.rmtree(inputs)
                removed += 1
            disclosure = self.cache_dir / "disclosures" / plan.plan_digest
            if disclosure.is_dir():
                shutil.rmtree(disclosure)
                removed += 1
            for item in ready_objects(plan):
                if item.sha256 not in retained:
                    self.content.path(item.sha256).unlink(missing_ok=True)
                    for suffix in (".npz", ".mp4", ".safetensors", ".json", ".bin"):
                        (self.cache_dir / "public-restored" / f"{item.sha256}{suffix}").unlink(
                            missing_ok=True,
                        )
        return removed

    def _allocated_plan(self, challenge_id: str, *, confirmation: bool) -> BatchPlan:
        """Read immutable allocation metadata, without demanding its bulk bytes."""

        if not challenge_id:
            raise ValueError("challenge ID cannot be empty")
        _, authority = self.inventory.read()
        stage = "CONFIRMING" if confirmation else "SCORING"
        claim = next((row for row in authority["claims"]
                      if row["challenge_id"] == challenge_id and row["stage"] == stage), None)
        if claim is not None:
            plan = self._plan_by_digest(claim["plan_digest"])
            self.inventory.validate_claim(
                claim, plan, disclosure_delay_seconds=self.track.cache.disclosure_delay_seconds,
            )
            return plan
        raise RuntimeError("challenge has no allocated hidden inventory")

    def _plans(self) -> tuple[tuple[str, BatchPlan], ...]:
        rows = []
        prefix = f"{PLANS_NAMESPACE}/"
        for name in self.objects.list_names(prefix):
            if name == ALLOCATION_OBJECT:
                continue
            if not name.startswith(prefix) or not name.endswith(".json"):
                raise RuntimeError("private plan index contains an unexpected object")
            plan = BatchPlan.decode(self.objects.get(name).body)
            self._validate_plan(name, plan)
            rows.append((name, plan))
        rows.sort(key=lambda row: row[1].generation_block)
        try:
            validate_source_selection_sequence(
                tuple(plan for _name, plan in rows),
                commitments=self.track.source_commitments,
                scoring=self.track.scoring,
            )
        except ValueError as exc:
            raise RuntimeError("private plan source selection is invalid") from exc
        return tuple(rows)

    def _validate_plan(self, name: str, plan: BatchPlan) -> None:
        if name != f"{PLANS_NAMESPACE}/{plan.generation_block}.json":
            raise RuntimeError("private plan key disagrees with its generation block")
        if plan.track_spec_digest != self.track.digest:
            raise RuntimeError("private plan belongs to another track")
        commitment_digests = tuple(item.sha256 for item in self.track.source_commitments)
        if any(
            not verify_source_proof(
                proof,
                self.track.source_pool_merkle_root,
                commitment_digests=commitment_digests,
            )
            for proof in plan.source_proofs
        ):
            raise RuntimeError("private plan source proof is invalid")
        rebuilt = build_plan(
            track_spec_digest=self.track.digest,
            generation_block=plan.generation_block,
            block_hash=plan.block_hash,
            secret=plan.secret,
            inventory_slot=plan.inventory_slot,
            source_proofs=plan.source_proofs,
            scoring=self.track.scoring,
            available_routes=self.track.model.selector.routes,
        )
        if rebuilt.planned_document() != plan.planned_document():
            raise RuntimeError("private plan conflicts with the deterministic track schedule")
        if plan.state == "READY":
            assert plan.exact is not None and plan.sample_cache is not None
            objects = [
                (plan.exact, EXACT_NAMESPACE),
                (plan.sample_cache.index, INPUTS_NAMESPACE),
                (plan.sample_cache.text, INPUTS_NAMESPACE),
                *((reference.video, VIDEOS_NAMESPACE) for reference in plan.dense_references),
                *((reference.latent, LATENTS_NAMESPACE) for reference in plan.dense_references),
            ]
            if plan.sample_cache.condition is not None:
                objects.append((plan.sample_cache.condition, INPUTS_NAMESPACE))
            if any(
                not item.object_key.startswith(f"{namespace}/")
                for item, namespace in objects
            ):
                raise RuntimeError("READY plan object is outside its runtime namespace")


def ready_objects(plan: BatchPlan) -> tuple[ReadyObject, ...]:
    if plan.exact is None or plan.sample_cache is None:
        return ()
    return (
        plan.exact, plan.sample_cache.index, plan.sample_cache.text,
        *((plan.sample_cache.condition,) if plan.sample_cache.condition is not None else ()),
        *(item for ref in plan.dense_references for item in (ref.video, ref.latent)),
    )


def validate_selected_sample_cache(path: Path, plan: BatchPlan) -> None:
    """Require the selected evaluator inputs to cover exactly the proved sources."""

    try:
        document = json.loads((path / "index.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("READY selected source index is invalid") from exc
    rows = document.get("samples") if isinstance(document, dict) else None
    identifiers = (
        [row.get("id") for row in rows if isinstance(row, dict)]
        if isinstance(rows, list)
        else []
    )
    expected = {proof.leaf.source_id for proof in plan.source_proofs}
    if (
        not isinstance(document, dict)
        or set(document) != {"samples"}
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
        or any(not isinstance(identifier, str) for identifier in identifiers)
        or len(identifiers) != len(set(identifiers))
        or set(identifiers) != expected
    ):
        raise RuntimeError("READY selected source index conflicts with its plan")


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _validate_plan_case_binding(path: Path, plan: BatchPlan, track: TrackSpec) -> None:
    """Bind every exact case and deterministic query sample to its frozen plan."""

    reader = ExactCacheReader(
        path,
        expected_sha256=plan.exact.sha256 if plan.exact is not None else "",
        max_bytes=track.cache.exact_max_bytes,
    )
    if plan.sample_cache is None or reader.metadata["sample_cache"] != (
        plan.sample_cache.document()
    ):
        raise CacheLoadError("cache selected inputs conflict with the frozen READY plan")
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
