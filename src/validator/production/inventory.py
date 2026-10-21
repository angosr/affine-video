"""One CAS authority for immutable model/data bindings in the reusable pool.

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
    """CAS linearizes model bindings; byte availability never resamples a model."""

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
                or set(row) != {
                    "ordinal", "challenge_id", "stage", "plan_digest", "ready_sha256",
                    "file_sha256", "model_digest", "submission_block", "submission_timestamp_ms",
                }
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
    ) -> dict[str, Any] | None:
        """Bind the newest eligible READY at the original on-chain submission.

        A batch is reusable, not consumed by a claim. Late scanning, retries and
        publication never move the submission boundary or substitute newer data.
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
        ), reverse=True))
        if len({plan.plan_digest for plan in ordered}) != len(ordered):
            raise ValueError("inventory repeats a plan")
        by_digest = {plan.plan_digest: plan for plan in ordered}
        if any(plan.track_spec_digest != self.track_spec_digest for plan in ordered):
            raise RuntimeError("inventory contains another track")

        def update(document: dict[str, Any]) -> dict[str, Any] | None:
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
            exact = next((row for row in own if row["stage"] == "SCORING"), None)
            if stage == "CONFIRMING" and exact is None:
                raise RuntimeError("confirmation inventory requires an exact allocation")
            excluded_families = set() if stage == "SCORING" else {
                proof.leaf.source_family_id
                for proof in by_digest[exact["plan_digest"]].source_proofs
            }
            for plan in ordered:
                if (
                    plan.state != "READY"
                    or not self.eligible(plan, submission_block, submission_timestamp_ms,
                                         disclosure_delay_seconds)
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
                    "submission_block": submission_block,
                    "submission_timestamp_ms": submission_timestamp_ms,
                    "plan_digest": plan.plan_digest,
                    "ready_sha256": hashlib.sha256(plan.encode()).hexdigest(),
                }
                claims.append(row)
                return row
            return None

        return self._change(update)

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
            not isinstance(claim, dict) or set(claim) != {
                "ordinal", "challenge_id", "stage", "plan_digest", "ready_sha256",
                "file_sha256", "model_digest", "submission_block", "submission_timestamp_ms",
            }
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
