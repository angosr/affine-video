"""Secret-seeded, challenge-independent inventory plans and their READY seal."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from scoring.spec import ScoringSpec, SourceCommitmentSpec
from validator.production.commitment import (
    SourcePool,
    SourceProof,
    canonical_json,
    decode_canonical_json,
)


@dataclass(frozen=True, slots=True)
class SelectedStep:
    index: int
    window: str
    sigma: float
    next_sigma: float

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("selected step index must be non-negative")
        if not isinstance(self.window, str) or not self.window:
            raise ValueError("selected step window cannot be empty")
        if (
            isinstance(self.sigma, bool)
            or isinstance(self.next_sigma, bool)
            or not isinstance(self.sigma, (int, float))
            or not isinstance(self.next_sigma, (int, float))
            or not isfinite(float(self.sigma))
            or not isfinite(float(self.next_sigma))
            or not 0 <= self.next_sigma < self.sigma <= 1
        ):
            raise ValueError("selected step sigma transition is invalid")

    def document(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "next_sigma": self.next_sigma,
            "sigma": self.sigma,
            "window": self.window,
        }


@dataclass(frozen=True, slots=True)
class RolloutPlan:
    rollout_id: str
    source_id: str
    stratum: str
    noise_seed: int

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.rollout_id,
                self.source_id,
                self.stratum,
            )
        ):
            raise ValueError("rollout identifiers cannot be empty")
        if type(self.noise_seed) is not int or not 0 <= self.noise_seed < 2**64:
            raise ValueError("rollout noise seed must fit in an unsigned 64-bit integer")

    def document(self) -> dict[str, Any]:
        return {
            "noise_seed": self.noise_seed,
            "rollout_id": self.rollout_id,
            "source_id": self.source_id,
            "stratum": self.stratum,
        }


@dataclass(frozen=True, slots=True)
class ReadyObject:
    object_key: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.object_key or self.object_key.startswith("/"):
            raise ValueError("ready object key is invalid")
        _digest_bytes(self.sha256, "ready object SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("ready object size must be positive")

    def document(self) -> dict[str, Any]:
        return {
            "object_key": self.object_key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DenseReference:
    rollout_id: str
    video: ReadyObject
    latent: ReadyObject

    def __post_init__(self) -> None:
        if not self.rollout_id:
            raise ValueError("dense reference rollout ID cannot be empty")

    def document(self) -> dict[str, Any]:
        return {
            "latent": self.latent.document(),
            "rollout_id": self.rollout_id,
            "video": self.video.document(),
        }


@dataclass(frozen=True, slots=True)
class ReadySampleCache:
    """The encoded inputs selected by one READY plan, never the hidden pool."""

    index: ReadyObject
    text: ReadyObject
    condition: ReadyObject | None = None

    def document(self) -> dict[str, Any]:
        return {
            "condition": None if self.condition is None else self.condition.document(),
            "index": self.index.document(),
            "text": self.text.document(),
        }


@dataclass(frozen=True, slots=True)
class BatchPlan:
    track_spec_digest: str
    generation_block: int
    block_hash: str
    batch_seed: str
    secret: str
    secret_commitment: str
    inventory_slot: int | None
    source_proofs: tuple[SourceProof, ...]
    selected_routes: tuple[str, ...]
    selected_steps: tuple[SelectedStep, ...]
    rollouts: tuple[RolloutPlan, ...]
    query_selection_seed: str
    plan_digest: str
    state: str = "PLANNED"
    exact: ReadyObject | None = None
    dense_references: tuple[DenseReference, ...] = ()
    sample_cache: ReadySampleCache | None = None
    completed_block: int | None = None
    completed_at_ms: int | None = None

    def __post_init__(self) -> None:
        _digest_bytes(self.track_spec_digest, "track specification digest")
        _digest_bytes(self.block_hash.removeprefix("0x"), "generation block hash")
        _digest_bytes(self.batch_seed, "batch seed")
        _digest_bytes(self.secret, "inventory secret")
        if self.secret_commitment != inventory_commitment(self.secret):
            raise ValueError("inventory secret differs from its commitment")
        if self.batch_seed != generation_seed(self.track_spec_digest, self.block_hash, self.secret):
            raise ValueError("batch seed differs from its committed secret")
        if self.inventory_slot is not None and (
            type(self.inventory_slot) is not int or self.inventory_slot not in {0, 1}
        ):
            raise ValueError("inventory source slot must be zero or one")
        _digest_bytes(self.query_selection_seed, "query selection seed")
        _digest_bytes(self.plan_digest, "plan digest")
        if type(self.generation_block) is not int or self.generation_block < 0:
            raise ValueError("generation block must be non-negative")
        if (
            not self.source_proofs
            or not self.selected_routes
            or not self.selected_steps
            or not self.rollouts
        ):
            raise ValueError("batch plan selection cannot be empty")
        for values, name in (
            (self.selected_routes, "selected route"),
        ):
            if len(set(values)) != len(values) or any(not value for value in values):
                raise ValueError(f"{name} identifiers must be non-empty and unique")
        if self.state == "PLANNED":
            if (self.exact is not None or self.dense_references or self.sample_cache is not None
                    or self.completed_block is not None or self.completed_at_ms is not None):
                raise ValueError("a PLANNED batch cannot reference completed objects")
        elif self.state == "READY":
            if self.exact is None or not self.dense_references or self.sample_cache is None:
                raise ValueError(
                    "a READY batch requires exact data, dense references, and selected inputs"
                )
            if (
                type(self.completed_block) is not int
                or self.completed_block <= self.generation_block
            ):
                raise ValueError("READY completion must follow the generation plan")
            if type(self.completed_at_ms) is not int or self.completed_at_ms <= 0:
                raise ValueError("READY completion time must be a positive UTC millisecond")
        else:
            raise ValueError("batch plan state must be PLANNED or READY")
        if self.plan_digest != hashlib.sha256(canonical_json(self.planned_document())).hexdigest():
            raise ValueError("batch plan digest does not match its planned fields")
        if len({step.index for step in self.selected_steps}) != len(self.selected_steps):
            raise ValueError("selected step indices must be unique")
        source_ids = {proof.leaf.source_id for proof in self.source_proofs}
        if len(source_ids) != len(self.source_proofs):
            raise ValueError("plan source proofs must be unique")
        if {rollout.source_id for rollout in self.rollouts} != source_ids:
            raise ValueError("plan rollouts must cover exactly the proved sources")
        if self.state == "READY" and {
            reference.rollout_id for reference in self.dense_references
        } != {rollout.rollout_id for rollout in self.rollouts}:
            raise ValueError("READY references must cover every planned rollout")

    def planned_document(self) -> dict[str, Any]:
        return _planned_document(
            {
                "batch_seed": self.batch_seed,
                "block_hash": self.block_hash,
                "secret": self.secret,
                "secret_commitment": self.secret_commitment,
                "inventory_slot": self.inventory_slot,
                "generation_block": self.generation_block,
                "query_selection_seed": self.query_selection_seed,
                "rollouts": self.rollouts,
                "selected_routes": self.selected_routes,
                "selected_steps": self.selected_steps,
                "source_proofs": self.source_proofs,
                "track_spec_digest": self.track_spec_digest,
            }
        )

    def document(self) -> dict[str, Any]:
        result = {
            **self.planned_document(),
            "plan_digest": self.plan_digest,
            "state": self.state,
        }
        if self.state == "READY":
            result["completed_block"] = self.completed_block
            result["completed_at_ms"] = self.completed_at_ms
            result["exact"] = self.exact.document() if self.exact is not None else None
            result["dense_references"] = [
                reference.document() for reference in self.dense_references
            ]
            result["sample_cache"] = (
                None if self.sample_cache is None else self.sample_cache.document()
            )
        return result

    def encode(self) -> bytes:
        return canonical_json(self.document())

    def ready(
        self,
        *,
        exact: ReadyObject,
        dense_references: tuple[DenseReference, ...],
        sample_cache: ReadySampleCache,
        completed_block: int,
        completed_at_ms: int,
    ) -> BatchPlan:
        if self.state != "PLANNED":
            raise ValueError("only a PLANNED batch can become READY")
        return replace(
            self,
            state="READY",
            exact=exact,
            dense_references=dense_references,
            sample_cache=sample_cache,
            completed_block=completed_block,
            completed_at_ms=completed_at_ms,
        )

    @classmethod
    def decode(cls, payload: bytes) -> BatchPlan:
        """Recover a strictly canonical PLANNED or READY document."""

        value = decode_canonical_json(payload)
        if not isinstance(value, dict):
            raise ValueError("batch plan must be a JSON object")
        common = {
            "batch_seed",
            "block_hash",
            "secret",
            "secret_commitment",
            "inventory_slot",
            "format",
            "generation_block",
            "plan_digest",
            "query_selection_seed",
            "rollouts",
            "selected_routes",
            "selected_steps",
            "source_proofs",
            "state",
            "track_spec_digest",
        }
        state = value.get("state")
        expected = common | (
            {"dense_references", "exact", "sample_cache", "completed_block", "completed_at_ms"}
            if state == "READY" else set()
        )
        if set(value) != expected or value.get("format") != "veda-inventory-plan-v1":
            raise ValueError("batch plan fields or format are invalid")

        def strings(name: str) -> tuple[str, ...]:
            rows = value[name]
            if not isinstance(rows, list) or any(not isinstance(item, str) for item in rows):
                raise ValueError(f"batch plan {name} must be a string list")
            return tuple(rows)

        steps_value = value["selected_steps"]
        if not isinstance(steps_value, list):
            raise ValueError("batch plan selected_steps must be a list")
        steps = tuple(_parse_step(row) for row in steps_value)
        rollouts_value = value["rollouts"]
        if not isinstance(rollouts_value, list):
            raise ValueError("batch plan rollouts must be a list")
        rollouts = tuple(_parse_rollout(row) for row in rollouts_value)
        proofs_value = value["source_proofs"]
        if not isinstance(proofs_value, list):
            raise ValueError("batch plan source_proofs must be a list")
        exact = _parse_ready(value["exact"]) if state == "READY" else None
        sample_cache = _parse_sample_cache(value["sample_cache"]) if state == "READY" else None
        references_value = value.get("dense_references", [])
        if not isinstance(references_value, list):
            raise ValueError("batch plan dense_references must be a list")
        return cls(
            track_spec_digest=value["track_spec_digest"],
            generation_block=value["generation_block"],
            block_hash=value["block_hash"],
            batch_seed=value["batch_seed"],
            secret=value["secret"],
            secret_commitment=value["secret_commitment"],
            inventory_slot=value["inventory_slot"],
            source_proofs=tuple(SourceProof.from_document(row) for row in proofs_value),
            selected_routes=strings("selected_routes"),
            selected_steps=steps,
            rollouts=rollouts,
            query_selection_seed=value["query_selection_seed"],
            plan_digest=value["plan_digest"],
            state=state,
            exact=exact,
            dense_references=tuple(_parse_reference(row) for row in references_value),
            sample_cache=sample_cache,
            completed_block=value.get("completed_block"),
            completed_at_ms=value.get("completed_at_ms"),
        )


def build_plan(
    *,
    track_spec_digest: str,
    generation_block: int,
    block_hash: str,
    secret: str,
    inventory_slot: int,
    source_proofs: tuple[SourceProof, ...],
    scoring: ScoringSpec,
    available_routes: tuple[str, ...],
) -> BatchPlan:
    """Build an inventory item without knowing any future challenger identity."""

    return _build_plan(
        track_spec_digest=track_spec_digest,
        generation_block=generation_block,
        block_hash=block_hash,
        secret=secret,
        inventory_slot=inventory_slot,
        source_proofs=source_proofs,
        scoring=scoring,
        available_routes=available_routes,
        source_partition="production",
    )


def build_audit_evidence_plan(
    *,
    track_spec_digest: str,
    generation_block: int,
    block_hash: str,
    secret: str,
    source_proofs: tuple[SourceProof, ...],
    scoring: ScoringSpec,
    available_routes: tuple[str, ...],
) -> BatchPlan:
    """Build an audit-only launch-evidence plan outside the live coordinator."""

    return _build_plan(
        track_spec_digest=track_spec_digest,
        generation_block=generation_block,
        block_hash=block_hash,
        secret=secret,
        inventory_slot=None,
        source_proofs=source_proofs,
        scoring=scoring,
        available_routes=available_routes,
        source_partition="audit",
    )


def _build_plan(
    *,
    track_spec_digest: str,
    generation_block: int,
    block_hash: str,
    secret: str,
    inventory_slot: int | None,
    source_proofs: tuple[SourceProof, ...],
    scoring: ScoringSpec,
    available_routes: tuple[str, ...],
    source_partition: str,
) -> BatchPlan:
    if source_partition == "production" and (
        type(inventory_slot) is not int or inventory_slot not in {0, 1}
    ):
        raise ValueError("production inventory requires a source slot")
    batch_seed = generation_seed(track_spec_digest, block_hash, secret)
    routes = _ranked(available_routes, batch_seed, "route")[: scoring.sampled_route_count]
    if len(routes) != scoring.sampled_route_count:
        raise ValueError("not enough selector routes for the batch plan")
    steps = select_steps(batch_seed, scoring)
    rollouts = _rollouts(
        batch_seed,
        source_proofs,
        scoring,
        source_partition=source_partition,
    )
    query_seed = hashlib.sha256(
        b"veda-query-selection-v1\0" + bytes.fromhex(batch_seed)
    ).hexdigest()
    arguments = {
        "track_spec_digest": track_spec_digest,
        "generation_block": generation_block,
        "block_hash": "0x" + block_hash.removeprefix("0x").lower(),
        "batch_seed": batch_seed,
        "secret": secret,
        "secret_commitment": inventory_commitment(secret),
        "inventory_slot": inventory_slot,
        "source_proofs": tuple(source_proofs),
        "selected_routes": routes,
        "selected_steps": steps,
        "rollouts": rollouts,
        "query_selection_seed": query_seed,
    }
    digest = hashlib.sha256(canonical_json(_planned_document(arguments))).hexdigest()
    return BatchPlan(**arguments, plan_digest=digest)


def inventory_commitment(secret: str) -> str:
    return hashlib.sha256(
        b"veda-inventory-secret-v1\0" + _digest_bytes(secret, "inventory secret")
    ).hexdigest()


def generation_seed(track_spec_digest: str, block_hash: str, secret: str) -> str:
    track = _digest_bytes(track_spec_digest, "track specification digest")
    block = _digest_bytes(block_hash.removeprefix("0x"), "generation block hash")
    return hashlib.sha256(
        b"veda-inventory-batch-v1\0" + track + block
        + _digest_bytes(secret, "inventory secret")
    ).hexdigest()


def select_steps(batch_seed: str, scoring: ScoringSpec) -> tuple[SelectedStep, ...]:
    allocations = {window.name: 1 for window in scoring.trajectory_schedule.step_windows}
    for _ in range(scoring.sampled_step_count - len(allocations)):
        chosen = max(
            scoring.trajectory_schedule.step_windows,
            key=lambda window: (
                window.weight * scoring.sampled_step_count - allocations[window.name],
                window.name,
            ),
        )
        if allocations[chosen.name] >= len(chosen.indices):
            candidates = [
                window
                for window in scoring.trajectory_schedule.step_windows
                if allocations[window.name] < len(window.indices)
            ]
            if not candidates:
                raise ValueError("step windows cannot supply the requested sample count")
            chosen = max(
                candidates,
                key=lambda window: (
                    window.weight * scoring.sampled_step_count - allocations[window.name],
                    window.name,
                ),
            )
        allocations[chosen.name] += 1
    selected = []
    sigmas = scoring.trajectory_schedule.sigmas
    for window in scoring.trajectory_schedule.step_windows:
        ranked = _ranked(
            tuple(str(index) for index in window.indices),
            batch_seed,
            f"step:{window.name}",
        )
        selected.extend(
            SelectedStep(index, window.name, sigmas[index], sigmas[index + 1])
            for index in sorted(int(value) for value in ranked[: allocations[window.name]])
        )
    return tuple(sorted(selected, key=lambda step: step.index))


def select_query_tiles(
    *,
    query_selection_seed: str,
    case_id: str,
    target_tiles: tuple[int, ...],
    reference_tiles: tuple[int, ...],
    count: int,
) -> tuple[int, ...]:
    """Select target/reference queries; global rows are deliberately excluded."""

    if count <= 0 or not case_id:
        raise ValueError("query selection inputs are invalid")
    target = tuple(sorted(set(target_tiles)))
    reference = tuple(sorted(set(reference_tiles)))
    if set(target) & set(reference) or not target:
        raise ValueError("query tile segments are invalid")
    if reference:
        reference_count = min(len(reference), max(1, count // 4))
        target_count = min(len(target), count - reference_count)
        remaining = count - reference_count - target_count
        target_count += min(remaining, len(target) - target_count)
        remaining = count - reference_count - target_count
        reference_count += min(remaining, len(reference) - reference_count)
    else:
        target_count = min(count, len(target))
        reference_count = 0
    if target_count + reference_count != count:
        raise ValueError("query tile universe is smaller than query_tiles_per_case")
    selected = _ranked(target, query_selection_seed, f"query:{case_id}:target")[:target_count]
    selected += _ranked(reference, query_selection_seed, f"query:{case_id}:reference")[
        :reference_count
    ]
    return tuple(sorted(selected))


def select_source_proofs(
    pool: SourcePool,
    *,
    commitments: tuple[SourceCommitmentSpec, ...],
    partition: str,
    strata: tuple[str, ...],
    sources_per_stratum: int,
    batch_seed: str,
    excluded_source_digests: frozenset[str],
    inventory_slot: int | None = None,
) -> tuple[SourceProof, ...]:
    """Select distinct source families for a fresh block-seeded rollout batch."""

    selected_digests = select_source_digests(
        commitments,
        partition=partition,
        strata=strata,
        sources_per_stratum=sources_per_stratum,
        batch_seed=batch_seed,
        excluded_source_digests=excluded_source_digests,
        inventory_slot=inventory_slot,
    )
    leaves = {leaf.digest: leaf for leaf in pool.leaves}
    if set(leaves) != {item.sha256 for item in commitments}:
        raise ValueError("private source pool differs from the public selection index")
    try:
        return tuple(pool.proof(leaves[digest].source_id) for digest in selected_digests)
    except KeyError as exc:
        raise ValueError("public source selection is absent from the private pool") from exc


def select_source_digests(
    commitments: tuple[SourceCommitmentSpec, ...],
    *,
    partition: str,
    strata: tuple[str, ...],
    sources_per_stratum: int,
    batch_seed: str,
    excluded_source_digests: frozenset[str],
    inventory_slot: int | None = None,
) -> tuple[str, ...]:
    """Select sources from one salted partition published before the track."""

    if partition not in {"production", "audit"}:
        raise ValueError("source partition must be production or audit")
    if inventory_slot is not None and (
        partition != "production" or type(inventory_slot) is not int
        or inventory_slot not in {0, 1}
    ):
        raise ValueError("inventory source slot requires production and zero or one")
    if not strata or len(set(strata)) != len(strata):
        raise ValueError("source strata must be non-empty and unique")
    if type(sources_per_stratum) is not int or sources_per_stratum <= 0:
        raise ValueError("sources per stratum must be positive")
    known = {item.sha256 for item in commitments}
    if len(known) != len(commitments) or not excluded_source_digests <= known:
        raise ValueError("source commitment index or exclusion set is invalid")
    selected: list[str] = []
    reserved = set(excluded_source_digests)
    for stratum in sorted(strata):
        universe = tuple(sorted(
            item.sha256
            for item in commitments
            if item.partition == partition
            and item.generation_geometry == stratum
        ))
        # Two public, fixed family-disjoint halves remove any need to choose new
        # sources after seeing an exact result. The secret still hides the sample.
        candidates = universe if inventory_slot is None else universe[inventory_slot::2]
        candidates = tuple(digest for digest in candidates if digest not in reserved)
        ranked = _ranked(candidates, batch_seed, f"source:{stratum}")
        chosen_for_stratum = ranked[:sources_per_stratum]
        if len(chosen_for_stratum) != sources_per_stratum:
            raise ValueError(f"source pool is exhausted for stratum {stratum}")
        reserved.update(chosen_for_stratum)
        selected.extend(chosen_for_stratum)
    return tuple(selected)


def validate_source_selection_sequence(
    plans: tuple[BatchPlan, ...],
    *,
    commitments: tuple[SourceCommitmentSpec, ...],
    scoring: ScoringSpec,
) -> None:
    """Verify every production item's public half and secret-seeded ranking."""

    ordered = tuple(sorted(plans, key=lambda plan: plan.generation_block))
    if len({plan.generation_block for plan in ordered}) != len(ordered):
        raise ValueError("source selection plans repeat a generation block")
    for plan in ordered:
        if plan.inventory_slot not in {0, 1}:
            raise ValueError("production inventory requires a source slot")
        expected = select_source_digests(
            commitments,
            partition="production",
            strata=tuple(scoring.stratum_weight_map),
            sources_per_stratum=scoring.rollouts_per_stratum,
            batch_seed=plan.batch_seed,
            excluded_source_digests=frozenset(),
            inventory_slot=plan.inventory_slot,
        )
        actual = tuple(proof.leaf.digest for proof in plan.source_proofs)
        if actual != expected:
            raise ValueError("plan source selection differs from the public commitment index")


def _rollouts(
    batch_seed: str,
    proofs: tuple[SourceProof, ...],
    scoring: ScoringSpec,
    *,
    source_partition: str,
) -> tuple[RolloutPlan, ...]:
    if source_partition not in {"production", "audit"}:
        raise ValueError("rollout source partition must be production or audit")
    if any(proof.leaf.partition != source_partition for proof in proofs):
        raise ValueError(f"batch plan can only select {source_partition} sources")
    expected_strata = set(scoring.stratum_weight_map)
    by_stratum = {
        stratum: tuple(proof for proof in proofs if proof.leaf.generation_geometry == stratum)
        for stratum in expected_strata
    }
    if (
        any(proof.leaf.generation_geometry not in expected_strata for proof in proofs)
        or any(len(items) != scoring.rollouts_per_stratum for items in by_stratum.values())
        or sum(map(len, by_stratum.values())) != len(proofs)
    ):
        raise ValueError(
            "source proofs must contain one independent source per rollout and stratum"
        )
    families = [proof.leaf.source_family_id for proof in proofs]
    if len(set(families)) != len(families):
        raise ValueError("rollout source families must be independent")
    result = []
    for stratum in sorted(expected_strata):
        for proof in by_stratum[stratum]:
            material = hashlib.sha256(
                bytes.fromhex(batch_seed) + b"\0rollout\0" + proof.leaf.source_id.encode()
            ).digest()
            rollout_id = hashlib.sha256(b"veda-rollout-v1\0" + material).hexdigest()
            result.append(
                RolloutPlan(
                    rollout_id,
                    proof.leaf.source_id,
                    proof.leaf.generation_geometry,
                    int.from_bytes(material[8:16], "big"),
                )
            )
    return tuple(result)


def _ranked(values: tuple[Any, ...], seed: str, domain: str) -> tuple[Any, ...]:
    if len(set(values)) != len(values):
        raise ValueError("ranked values must be unique")
    material = _digest_bytes(seed, "ranking seed")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(
                    material + b"\0" + domain.encode() + b"\0" + str(value).encode()
                ).digest(),
                str(value),
            ),
        )
    )


def _planned_document(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_seed": arguments["batch_seed"],
        "block_hash": arguments["block_hash"],
        "secret": arguments["secret"],
        "secret_commitment": arguments["secret_commitment"],
        "inventory_slot": arguments["inventory_slot"],
        "format": "veda-inventory-plan-v1",
        "generation_block": arguments["generation_block"],
        "query_selection_seed": arguments["query_selection_seed"],
        "rollouts": [rollout.document() for rollout in arguments["rollouts"]],
        "selected_routes": list(arguments["selected_routes"]),
        "selected_steps": [step.document() for step in arguments["selected_steps"]],
        "source_proofs": [proof.document() for proof in arguments["source_proofs"]],
        "track_spec_digest": arguments["track_spec_digest"],
    }


def _parse_step(value: Any) -> SelectedStep:
    row = _row(value, {"index", "next_sigma", "sigma", "window"}, "selected step")
    return SelectedStep(row["index"], row["window"], row["sigma"], row["next_sigma"])


def _parse_rollout(value: Any) -> RolloutPlan:
    row = _row(
        value,
        {"noise_seed", "rollout_id", "source_id", "stratum"},
        "rollout",
    )
    return RolloutPlan(
        row["rollout_id"],
        row["source_id"],
        row["stratum"],
        row["noise_seed"],
    )


def _parse_ready(value: Any) -> ReadyObject:
    row = _row(value, {"object_key", "sha256", "size_bytes"}, "ready object")
    return ReadyObject(row["object_key"], row["sha256"], row["size_bytes"])


def _parse_reference(value: Any) -> DenseReference:
    row = _row(value, {"latent", "rollout_id", "video"}, "dense reference")
    return DenseReference(
        row["rollout_id"], _parse_ready(row["video"]), _parse_ready(row["latent"])
    )


def _parse_sample_cache(value: Any) -> ReadySampleCache:
    row = _row(value, {"condition", "index", "text"}, "READY sample cache")
    condition = row["condition"]
    if condition is not None:
        condition = _parse_ready(condition)
    return ReadySampleCache(_parse_ready(row["index"]), _parse_ready(row["text"]), condition)


def _row(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _digest_bytes(value: str, name: str) -> bytes:
    try:
        result = bytes.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    if len(result) != 32:
        raise ValueError(f"{name} must contain 32 bytes")
    return result
