"""One CAS authority for irreversible FIFO inventory claims and invalidations.

Generation plans never name challengers. This small, private authority is the
only place that can assign a READY item; GPU-local payloads are not authoritative.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from validator.production.commitment import canonical_json, decode_canonical_json
from validator.production.plan import BatchPlan
from validator.production.storage import PreconditionFailed, PrivateObjectStore, StoredObject

ALLOCATION_OBJECT = "plans/allocations.json"
_STAGES = {"SCORING", "CONFIRMING"}


def _digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


class Inventory:
    """CAS linearizes claim versus claim and claim versus lost-data invalidation."""

    def __init__(self, objects: PrivateObjectStore, track_spec_digest: str) -> None:
        if not _digest(track_spec_digest):
            raise ValueError("inventory track digest is invalid")
        self.objects = objects
        self.track_spec_digest = track_spec_digest

    def read(self) -> tuple[StoredObject | None, dict[str, Any]]:
        stored = self.objects.get_optional(ALLOCATION_OBJECT)
        document = {
            "format": "veda-inventory-allocations-v1",
            "track_spec_digest": self.track_spec_digest,
            "claims": [],
            "invalidations": [],
        } if stored is None else decode_canonical_json(stored.body)
        if (
            not isinstance(document, dict)
            or set(document) != {"format", "track_spec_digest", "claims", "invalidations"}
            or document["format"] != "veda-inventory-allocations-v1"
            or document["track_spec_digest"] != self.track_spec_digest
            or not isinstance(document["claims"], list)
            or not isinstance(document["invalidations"], list)
        ):
            raise RuntimeError("inventory allocation authority is invalid")
        plans: set[str] = set()
        stages: set[tuple[str, str]] = set()
        identities: dict[str, tuple[str, str]] = {}
        for ordinal, row in enumerate(document["claims"], 1):
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "ordinal", "challenge_id", "stage", "plan_digest", "ready_sha256",
                    "file_sha256", "model_digest",
                }
                or type(row["ordinal"]) is not int or row["ordinal"] != ordinal
                or any(not _digest(row[key]) for key in (
                    "challenge_id", "plan_digest", "ready_sha256", "file_sha256", "model_digest",
                ))
                or row["stage"] not in _STAGES
                or row["plan_digest"] in plans
                or (row["challenge_id"], row["stage"]) in stages
                or (row["stage"] == "CONFIRMING"
                    and (row["challenge_id"], "SCORING") not in stages)
            ):
                raise RuntimeError("inventory claim sequence is invalid")
            identity = (row["file_sha256"], row["model_digest"])
            if identities.setdefault(row["challenge_id"], identity) != identity:
                raise RuntimeError("inventory claims change the admitted model")
            plans.add(row["plan_digest"])
            stages.add((row["challenge_id"], row["stage"]))
        for row in document["invalidations"]:
            if (
                not isinstance(row, dict) or set(row) != {"plan_digest", "reason"}
                or not _digest(row["plan_digest"])
                or row["plan_digest"] in plans
                or row["reason"] != "UNASSIGNED_DATA_LOST"
            ):
                raise RuntimeError("inventory invalidation sequence is invalid")
            plans.add(row["plan_digest"])
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
        plans: tuple[BatchPlan, ...],
    ) -> dict[str, Any] | None:
        """Claim the oldest compatible READY, or reuse this stage's original claim.

        Only the existing leased FIFO scorer calls this method. Model identity is
        read from its immutable admission; no candidate quality enters selection.
        """
        if stage not in _STAGES or any(not _digest(value) for value in (
            challenge_id, file_sha256, model_digest,
        )):
            raise ValueError("inventory claimant identity is invalid")
        ordered = tuple(sorted(plans, key=lambda plan: plan.generation_block))
        if len({plan.plan_digest for plan in ordered}) != len(ordered):
            raise ValueError("inventory repeats a plan")
        by_digest = {plan.plan_digest: plan for plan in ordered}
        if any(plan.track_spec_digest != self.track_spec_digest for plan in ordered):
            raise RuntimeError("inventory contains another track")

        def update(document: dict[str, Any]) -> dict[str, Any] | None:
            claims = document["claims"]
            own = [row for row in claims if row["challenge_id"] == challenge_id]
            if any(row["file_sha256"] != file_sha256 or row["model_digest"] != model_digest
                   for row in own):
                raise RuntimeError("inventory claimant differs from its original admission")
            existing = next((row for row in own if row["stage"] == stage), None)
            if existing is not None:
                self.validate_claim(existing, by_digest[existing["plan_digest"]])
                return existing
            exact = next((row for row in own if row["stage"] == "SCORING"), None)
            if stage == "CONFIRMING" and exact is None:
                raise RuntimeError("confirmation inventory requires an exact allocation")
            excluded_families = set() if stage == "SCORING" else {
                proof.leaf.source_family_id
                for proof in by_digest[exact["plan_digest"]].source_proofs
            }
            unavailable = {
                row["plan_digest"] for row in claims + document["invalidations"]
            }
            for plan in ordered:
                if (
                    plan.state != "READY" or plan.plan_digest in unavailable
                    or (stage == "CONFIRMING" and plan.inventory_slot
                        == by_digest[exact["plan_digest"]].inventory_slot)
                    or excluded_families.intersection(
                        proof.leaf.source_family_id for proof in plan.source_proofs
                    )
                ):
                    continue
                row = {
                    "ordinal": len(claims) + 1, "challenge_id": challenge_id, "stage": stage,
                    "file_sha256": file_sha256, "model_digest": model_digest,
                    "plan_digest": plan.plan_digest,
                    "ready_sha256": hashlib.sha256(plan.encode()).hexdigest(),
                }
                claims.append(row)
                return row
            return None

        return self._change(update)

    @staticmethod
    def validate_claim(claim: dict[str, Any], plan: BatchPlan) -> None:
        if (
            not isinstance(claim, dict) or set(claim) != {
                "ordinal", "challenge_id", "stage", "plan_digest", "ready_sha256",
                "file_sha256", "model_digest",
            }
            or type(claim["ordinal"]) is not int or claim["ordinal"] <= 0
            or claim["stage"] not in _STAGES
            or any(not _digest(claim[key]) for key in (
                "challenge_id", "plan_digest", "ready_sha256", "file_sha256", "model_digest",
            ))
            or plan.state != "READY" or claim["plan_digest"] != plan.plan_digest
                or claim["ready_sha256"] != hashlib.sha256(plan.encode()).hexdigest()):
            raise RuntimeError("inventory claim differs from its immutable READY")

    def invalidate_unassigned(self, plan_digest: str) -> bool:
        if not _digest(plan_digest):
            raise ValueError("inventory plan digest is invalid")

        def update(document: dict[str, Any]) -> bool:
            if any(row["plan_digest"] == plan_digest for row in document["claims"]):
                return False
            if not any(row["plan_digest"] == plan_digest for row in document["invalidations"]):
                document["invalidations"].append({
                    "plan_digest": plan_digest, "reason": "UNASSIGNED_DATA_LOST",
                })
            return True

        return self._change(update)
