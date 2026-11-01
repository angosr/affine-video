"""One CAS authority for immutable model/data bindings in the reusable pool.

Generation plans never name challengers. This small, private authority is the
only place that can assign a READY item; GPU-local payloads are not authoritative.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from validator.production.commitment import canonical_json, decode_canonical_json
from validator.production.plan import BatchPlan
from validator.production.storage import PreconditionFailed, PrivateObjectStore, StoredObject

ALLOCATION_OBJECT = "plans/allocations.json"
_STAGES = {"SCORING", "CONFIRMING"}
_CLAIM_FIELDS = {
    "ordinal", "challenge_id", "stage", "plan_digest", "ready_sha256",
    "file_sha256", "model_digest", "submission_block", "submission_timestamp_ms",
}


def _digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


class Inventory:
    """CAS linearizes model bindings; byte availability never resamples a model."""

    def __init__(self, objects: PrivateObjectStore, track_spec_digest: str, *,
                 validate_plans: Callable[[tuple[BatchPlan, ...]], None] | None = None) -> None:
        if not _digest(track_spec_digest):
            raise ValueError("inventory track digest is invalid")
        self.objects = objects
        self.track_spec_digest = track_spec_digest
        self.validate_plans = validate_plans

    def read(self) -> tuple[StoredObject | None, dict[str, Any]]:
        stored = self.objects.get_optional(ALLOCATION_OBJECT)
        document = {
            "format": "veda-inventory-allocations-v1",
            "track_spec_digest": self.track_spec_digest,
            "claims": [],
        } if stored is None else decode_canonical_json(stored.body)
        if (
            not isinstance(document, dict)
            or set(document) != {"format", "track_spec_digest", "claims"}
            or document["format"] != "veda-inventory-allocations-v1"
            or document["track_spec_digest"] != self.track_spec_digest
            or not isinstance(document["claims"], list)
        ):
            raise RuntimeError("inventory allocation authority is invalid")
        plans: dict[str, str] = {}
        stages: set[tuple[str, str]] = set()
        identities: dict[str, tuple[str, str, int, int]] = {}
        for ordinal, row in enumerate(document["claims"], 1):
            if (
                not isinstance(row, dict)
                or set(row) not in (_CLAIM_FIELDS, _CLAIM_FIELDS | {"selection"})
                or type(row["ordinal"]) is not int or row["ordinal"] != ordinal
                or any(not _digest(row[key]) for key in (
                    "challenge_id", "plan_digest", "ready_sha256", "file_sha256", "model_digest",
                ))
                or row["stage"] not in _STAGES
                or type(row["submission_block"]) is not int or row["submission_block"] < 0
                or type(row["submission_timestamp_ms"]) is not int
                or row["submission_timestamp_ms"] <= 0
                or (row["challenge_id"], row["stage"]) in stages
                or (row["stage"] == "CONFIRMING"
                    and (row["challenge_id"], "SCORING") not in stages)
            ):
                raise RuntimeError("inventory claim sequence is invalid")
            _validate_selection(row)
            if "selection" in row:
                observed = self.observed_plans(row)
                exact = next((prior for prior in document["claims"][:ordinal - 1]
                              if prior["challenge_id"] == row["challenge_id"]
                              and prior["stage"] == "SCORING"), None)
                previous = (None if row["stage"] == "SCORING" or exact is None else
                            next((plan for plan in observed
                                  if plan.plan_digest == exact["plan_digest"]), None))
                if row["stage"] == "CONFIRMING" and previous is None:
                    raise RuntimeError("selection inputs omit the original exact plan")
                if row["selection"]["candidates"] != _candidate_rows(
                    observed, row, row["selection"]["disclosure_delay_seconds"], previous,
                ):
                    raise RuntimeError("selection evidence differs from its original plan facts")
            if "selection" in row and row["selection"]["previous_authority_sha256"] != (
                hashlib.sha256(canonical_json(
                    {**document, "claims": document["claims"][:ordinal - 1]})).hexdigest()
            ):
                raise RuntimeError("inventory selection changes its prior CAS authority")
            identity = (row["file_sha256"], row["model_digest"], row["submission_block"],
                        row["submission_timestamp_ms"])
            if identities.setdefault(row["challenge_id"], identity) != identity:
                raise RuntimeError("inventory claims change the admitted model")
            if plans.setdefault(row["plan_digest"], row["ready_sha256"]) != row["ready_sha256"]:
                raise RuntimeError("inventory claims change a shared READY")
            stages.add((row["challenge_id"], row["stage"]))
        return stored, document

    def _change(self, update: Callable[[dict[str, Any]], Any]) -> Any:
        # Bound CAS contention; the caller's ordinary infrastructure retry is durable.
        for _ in range(16):
            stored, document = self.read()
            before = canonical_json(document)
            result = update(document)
            after = canonical_json(document)
            if after == before:
                return result
            try:
                if stored is None:
                    self.objects.create_plan(ALLOCATION_OBJECT, after)
                else:
                    self.objects.replace_plan(ALLOCATION_OBJECT, after, etag=stored.etag)
            except PreconditionFailed:
                continue
            return result
        raise RuntimeError("inventory CAS contention exceeded the bounded retry limit")

    def claim(
        self, *, challenge_id: str, stage: str, file_sha256: str, model_digest: str,
        plans: tuple[BatchPlan, ...], submission_block: int, disclosure_delay_seconds: int,
        submission_timestamp_ms: int,
        pool_read_interval_ms: tuple[int, int] | None = None,
        enumeration: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Bind the earliest-expiring READY that was secret at original submission.

        A batch is reusable, not consumed by a claim. Late scanning, retries and
        publication never move the submission boundary or substitute newer data.
        Disclosure after S cannot invalidate even a delayed first claim. Existing
        durable claims are reused before considering pool ordering.
        """
        if stage not in _STAGES or any(not _digest(value) for value in (
            challenge_id, file_sha256, model_digest,
        )) or type(submission_block) is not int or submission_block < 0 or (
            type(disclosure_delay_seconds) is not int or disclosure_delay_seconds <= 0
        ) or (
            type(submission_timestamp_ms) is not int or submission_timestamp_ms <= 0
        ):
            raise ValueError("inventory claimant identity is invalid")
        ordered = tuple(sorted(plans, key=lambda plan: (
            plan.completed_at_ms or -1, plan.completed_block or -1,
            plan.generation_block, plan.plan_digest,
        )))
        if len({plan.plan_digest for plan in ordered}) != len(ordered):
            raise ValueError("inventory repeats a plan")
        by_digest = {plan.plan_digest: plan for plan in ordered}
        if any(plan.track_spec_digest != self.track_spec_digest for plan in ordered):
            raise RuntimeError("inventory contains another track")
        attempt = 0

        def update(document: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal attempt
            claims = document["claims"]
            own = [row for row in claims if row["challenge_id"] == challenge_id]
            if any(row["file_sha256"] != file_sha256 or row["model_digest"] != model_digest
                   or row["submission_block"] != submission_block
                   or row["submission_timestamp_ms"] != submission_timestamp_ms
                   for row in own):
                raise RuntimeError("inventory claimant differs from its original admission")
            existing = next((row for row in own if row["stage"] == stage), None)
            if existing is not None:
                self.validate_claim(existing, by_digest[existing["plan_digest"]],
                                    disclosure_delay_seconds=disclosure_delay_seconds)
                return existing
            attempt += 1
            exact = next((row for row in own if row["stage"] == "SCORING"), None)
            if stage == "CONFIRMING" and exact is None:
                raise RuntimeError("confirmation inventory requires an exact allocation")
            candidates = _candidate_rows(ordered, {
                "stage": stage, "submission_block": submission_block,
                "submission_timestamp_ms": submission_timestamp_ms,
            }, disclosure_delay_seconds, None if exact is None else by_digest[exact["plan_digest"]])
            for plan, candidate in zip(ordered, candidates, strict=True):
                if not all(candidate[key] for key in
                           ("s_eligible", "slot_eligible", "families_eligible")):
                    continue
                from validator.production.environment import evaluation_source_digest

                if self.validate_plans is not None:
                    self.validate_plans(ordered)
                inputs = canonical_json({
                    "format": "veda-selection-inputs-v1",
                    "track_spec_digest": self.track_spec_digest,
                    "enumeration": enumeration,
                    "plans": [{"name": f"plans/{p.generation_block}.json", "plan": p.document()}
                              for p in ordered],
                })
                digest = hashlib.sha256(inputs).hexdigest()
                name = f"stages/{challenge_id}/{stage}.inputs/{digest}.json"
                _validate_enumeration(enumeration, candidates)
                if enumeration is not None and any(
                    row["bucket"] != self.objects._bucket
                    or row["authority_prefix"] != self.objects._prefix
                    for row in enumeration["observations"]
                ):
                    raise RuntimeError("plan enumeration originates from another object authority")
                self.objects.create_plan(name, inputs)
                row = {
                    "ordinal": len(claims) + 1, "challenge_id": challenge_id, "stage": stage,
                    "file_sha256": file_sha256, "model_digest": model_digest,
                    "submission_block": submission_block,
                    "submission_timestamp_ms": submission_timestamp_ms,
                    "plan_digest": plan.plan_digest,
                    "ready_sha256": hashlib.sha256(plan.encode()).hexdigest(),
                    "selection": {
                        "algorithm": "earliest-expiry-stable-v1",
                        "source_sha256": evaluation_source_digest(),
                        "cas_attempt": attempt,
                        "captured_at_ms": time.time_ns() // 1_000_000,
                        "clock": "local-before-successful-cas",
                        "pool_read_interval_ms": (
                            None if pool_read_interval_ms is None else list(pool_read_interval_ms)),
                        "disclosure_delay_seconds": disclosure_delay_seconds,
                        # This authenticates only the allocation predecessor, not a
                        # global atomic view of the independently enumerated plan keys.
                        "previous_authority_sha256": hashlib.sha256(
                            canonical_json(document)).hexdigest(),
                        "pool_sha256": _pool_digest(candidates),
                        "inputs": {"object": name, "sha256": digest, "size_bytes": len(inputs)},
                        "enumeration": enumeration,
                        "candidates": candidates,
                    },
                }
                claims.append(row)
                return row
            return None

        return self._change(update)

    def observed_plans(self, claim: dict[str, Any]) -> tuple[BatchPlan, ...]:
        """Verify immutable enumeration inputs, not today's mutable plan directory."""
        reference = claim["selection"]["inputs"]
        stored = self.objects.get(reference["object"])
        if (len(stored.body) != reference["size_bytes"]
                or hashlib.sha256(stored.body).hexdigest() != reference["sha256"]):
            raise RuntimeError("selection input reference differs from its original bytes")
        document = decode_canonical_json(stored.body)
        if (not isinstance(document, dict) or set(document) != {
            "format", "track_spec_digest", "enumeration", "plans",
        } or document["format"] != "veda-selection-inputs-v1"
                or document["track_spec_digest"] != self.track_spec_digest
                or document["enumeration"] != claim["selection"]["enumeration"]
                or not isinstance(document["plans"], list)):
            raise RuntimeError("selection input document is invalid")
        plans = []
        for row in document["plans"]:
            if not isinstance(row, dict) or set(row) != {"name", "plan"}:
                raise RuntimeError("selection input plan reference is invalid")
            plan = BatchPlan.decode(canonical_json(row["plan"]))
            if (row["name"] != f"plans/{plan.generation_block}.json"
                    or plan.track_spec_digest != self.track_spec_digest):
                raise RuntimeError("selection input belongs to another plan authority")
            plans.append(plan)
        if self.validate_plans is not None:
            self.validate_plans(tuple(plans))
        return tuple(plans)

    @staticmethod
    def eligible(
        plan: BatchPlan, submission_block: int, submission_timestamp_ms: int,
        disclosure_delay_seconds: int,
    ) -> bool:
        return (
            plan.state == "READY" and plan.completed_block is not None
            and plan.completed_block < submission_block
            and plan.completed_at_ms < submission_timestamp_ms
            < plan.completed_at_ms + disclosure_delay_seconds * 1000
        )

    @staticmethod
    def validate_claim(
        claim: dict[str, Any], plan: BatchPlan, *, disclosure_delay_seconds: int,
    ) -> None:
        if (
            not isinstance(claim, dict)
            or set(claim) not in (_CLAIM_FIELDS, _CLAIM_FIELDS | {"selection"})
            or type(claim["ordinal"]) is not int or claim["ordinal"] <= 0
            or claim["stage"] not in _STAGES
            or type(claim["submission_block"]) is not int
            or type(claim["submission_timestamp_ms"]) is not int
            or any(not _digest(claim[key]) for key in (
                "challenge_id", "plan_digest", "ready_sha256", "file_sha256", "model_digest",
            ))
            or not Inventory.eligible(plan, claim["submission_block"],
                                      claim["submission_timestamp_ms"], disclosure_delay_seconds)
            or claim["plan_digest"] != plan.plan_digest
            or claim["ready_sha256"] != hashlib.sha256(plan.encode()).hexdigest()
        ):
            raise RuntimeError("inventory claim differs from its immutable READY")
        _validate_selection(claim)
        if ("selection" in claim and claim["selection"]["disclosure_delay_seconds"]
                != disclosure_delay_seconds):
            raise RuntimeError("inventory selection changes its original disclosure delay")


def _validate_selection(claim: dict[str, Any]) -> None:
    """Legacy rows have unknown first-selection facts; never backfill them."""

    if "selection" not in claim:
        return
    selection = claim["selection"]
    if (not isinstance(selection, dict) or set(selection) != {
        "algorithm", "captured_at_ms", "clock", "disclosure_delay_seconds",
        "previous_authority_sha256", "candidates", "pool_read_interval_ms",
        "source_sha256",
        "pool_sha256", "cas_attempt", "inputs", "enumeration",
    } or selection["algorithm"] != "earliest-expiry-stable-v1"
            or selection["clock"] != "local-before-successful-cas"
            or type(selection["captured_at_ms"]) is not int or selection["captured_at_ms"] <= 0
            or type(selection["disclosure_delay_seconds"]) is not int
            or selection["disclosure_delay_seconds"] <= 0
            or not _digest(selection["previous_authority_sha256"])
            or not _digest(selection["source_sha256"])
            or type(selection["cas_attempt"]) is not int
            or not 1 <= selection["cas_attempt"] <= 16
            or not isinstance(selection["candidates"], list) or not selection["candidates"]):
        raise RuntimeError("inventory first-selection evidence is invalid")
    interval = selection["pool_read_interval_ms"]
    if interval is not None and (
        not isinstance(interval, list) or len(interval) != 2
        or any(type(value) is not int for value in interval)
        or not 0 < interval[0] <= interval[1] <= selection["captured_at_ms"]
    ):
        raise RuntimeError("inventory pool read interval is invalid")
    candidates = selection["candidates"]
    reference = selection["inputs"]
    if (not isinstance(reference, dict) or set(reference) != {"object", "sha256", "size_bytes"}
            or not _digest(reference["sha256"])
            or reference["object"] != f"stages/{claim['challenge_id']}/{claim['stage']}.inputs/"
            + reference["sha256"] + ".json"
            or type(reference["size_bytes"]) is not int or reference["size_bytes"] <= 0):
        raise RuntimeError("selection input reference is invalid")
    seen = set()
    for candidate in candidates:
        if (not isinstance(candidate, dict) or set(candidate) != {
            "plan_digest", "plan_sha256", "generation_block", "state", "completed_block",
            "completed_at_ms", "inventory_slot", "source_selection_sha256",
            "s_eligible", "slot_eligible", "families_eligible",
        } or not _digest(candidate["plan_digest"]) or not _digest(candidate["plan_sha256"])
                or candidate["plan_digest"] in seen
                or any(type(candidate[key]) is not bool for key in
                       ("s_eligible", "slot_eligible", "families_eligible"))
                or candidate["state"] not in {"PLANNED", "READY"}
                or type(candidate["generation_block"]) is not int
                or candidate["generation_block"] < 0
                or type(candidate["inventory_slot"]) is not int
                or candidate["inventory_slot"] not in {0, 1}
                or not _digest(candidate["source_selection_sha256"])):
            raise RuntimeError("inventory candidate evidence is invalid")
        seen.add(candidate["plan_digest"])
        complete = candidate["state"] == "READY"
        if any((type(candidate[key]) is not int or candidate[key] <= 0) if complete
               else candidate[key] is not None for key in ("completed_block", "completed_at_ms")):
            raise RuntimeError("inventory candidate completion is invalid")
        eligible = (complete and candidate["completed_block"] < claim["submission_block"]
                    and candidate["completed_at_ms"] < claim["submission_timestamp_ms"]
                    < candidate["completed_at_ms"] + selection["disclosure_delay_seconds"] * 1000)
        if candidate["s_eligible"] != eligible:
            raise RuntimeError("inventory candidate original-S eligibility differs")
        if claim["stage"] == "SCORING" and not (
            candidate["slot_eligible"] and candidate["families_eligible"]
        ):
            raise RuntimeError("exact selection cannot exclude a source family or slot")
    if candidates != sorted(candidates, key=lambda row: (
        row["completed_at_ms"] or -1, row["completed_block"] or -1,
        row["generation_block"], row["plan_digest"],
    )):
        raise RuntimeError("inventory candidate order differs")
    if selection["pool_sha256"] != _pool_digest(candidates):
        raise RuntimeError("inventory candidate pool version differs")
    _validate_enumeration(selection["enumeration"], candidates)
    selected = next((row for row in candidates if all(row[key] for key in
                    ("s_eligible", "slot_eligible", "families_eligible"))), None)
    if (selected is None or selected["plan_digest"] != claim["plan_digest"]
            or selected["plan_sha256"] != claim["ready_sha256"]):
        raise RuntimeError("inventory selected READY differs from first-selection evidence")


def _pool_digest(candidates: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json([
        {key: row[key] for key in ("plan_digest", "plan_sha256")} for row in candidates
    ])).hexdigest()


def _candidate_rows(plans: tuple[BatchPlan, ...], claim: dict[str, Any], delay: int,
                    previous: BatchPlan | None) -> list[dict[str, Any]]:
    ordered = sorted(plans, key=lambda p: (p.completed_at_ms or -1, p.completed_block or -1,
                                           p.generation_block, p.plan_digest))
    excluded = set() if previous is None else {
        proof.leaf.source_family_id for proof in previous.source_proofs}
    return [{
        "plan_digest": p.plan_digest, "plan_sha256": hashlib.sha256(p.encode()).hexdigest(),
        "generation_block": p.generation_block, "state": p.state,
        "completed_block": p.completed_block, "completed_at_ms": p.completed_at_ms,
        "inventory_slot": p.inventory_slot, "source_selection_sha256": hashlib.sha256(
            canonical_json([proof.document() for proof in p.source_proofs])).hexdigest(),
        "s_eligible": Inventory.eligible(p, claim["submission_block"],
                                         claim["submission_timestamp_ms"], delay),
        "slot_eligible": previous is None or p.inventory_slot != previous.inventory_slot,
        "families_eligible": not bool(excluded.intersection(
            proof.leaf.source_family_id for proof in p.source_proofs)),
    } for p in ordered]


def _validate_enumeration(value: Any, candidates: list[dict[str, Any]]) -> None:
    if value is None:
        return  # Explicit caller-supplied input, never a claim of complete S3 enumeration.
    if (not isinstance(value, dict) or set(value) != {"algorithm", "observations"}
            or value["algorithm"] != "ReadyBatchSource._plans-v1"
            or not isinstance(value["observations"], list) or len(value["observations"]) != 2):
        raise RuntimeError("plan enumeration provenance is invalid")
    source = None
    expected = {f"plans/{row['generation_block']}.json" for row in candidates}
    for observation in value["observations"]:
        if (not isinstance(observation, dict) or set(observation) != {
            "api", "bucket", "endpoint", "authority_prefix", "directory", "limit", "after",
            "pages", "complete",
        } or observation["api"] != "ListObjectsV2" or observation["directory"] != "plans/"
                or observation["limit"] is not None or observation["after"] is not None
                or observation["complete"] is not True
                or not isinstance(observation["pages"], list) or not observation["pages"]):
            raise RuntimeError("plan enumeration did not prove a complete directory traversal")
        identity = observation["bucket"], observation["authority_prefix"], observation["endpoint"]
        if (any(not isinstance(s, str) or not s for s in identity[:2])
                or (identity[2] is not None and not isinstance(identity[2], str))) or (
            source is not None and source != identity
        ):
            raise RuntimeError("plan enumeration changes its object authority")
        source = identity
        names, token = [], None
        for index, page in enumerate(observation["pages"]):
            last = index == len(observation["pages"]) - 1
            if (not isinstance(page, dict) or set(page) != {
                "request_token_sha256", "next_token_sha256", "names", "is_truncated",
            } or page["request_token_sha256"] != token
                    or page["is_truncated"] is not (not last)
                    or not isinstance(page["names"], list)
                    or any(not isinstance(name, str) for name in page["names"])
                    or (page["next_token_sha256"] is not None if last else
                        not _digest(page["next_token_sha256"]))):
                raise RuntimeError("plan enumeration page chain is incomplete")
            names.extend(page["names"])
            token = page["next_token_sha256"]
        if len(set(names)) != len(names) or set(names) - {ALLOCATION_OBJECT} != expected:
            raise RuntimeError("plan enumeration omitted or replaced a referenced candidate")
