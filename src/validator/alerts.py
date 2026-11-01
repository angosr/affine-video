"""Failure logs with optional best-effort HTTPS notifications."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping

import httpx

LOGGER = logging.getLogger(__name__)

_PUBLIC_REASONS = {
    "BASELINE_DUPLICATE": "Model tensors match the official baseline.",
    "EXACT_DUPLICATE": "Model tensors match an earlier admitted model.",
    "REFERENCE_DUPLICATE": "The pinned repository revision was already submitted.",
    "DUPLICATE_HOTKEY": "An earlier submission used this hotkey's opportunity.",
    "NO_PRE_SUBMISSION_DATA": "No eligible data existed at the original submission time.",
    "MODEL_OWNER_MISMATCH": "The model is not bound to the submitting hotkey.",
    "MODEL_METADATA_INVALID": "Model metadata does not satisfy the submission contract.",
    "MODEL_SCHEMA_INVALID": "Model tensors do not satisfy the frozen schema.",
    "MODEL_NON_FINITE": "Model tensors contain non-finite values.",
    "MODEL_TOO_LARGE": "The submitted model exceeds the size limit.",
    "SCORING_CACHE_UNAVAILABLE": "The original bound scoring cache is unavailable; retry retained.",
    "SCORING_CACHE_INVALID": "The bound scoring cache could not be verified; retry retained.",
    "SCORING_TIMEOUT": "Scoring exceeded its execution limit; the original task is retained.",
}


def public_error_summary(code: str | None) -> str | None:
    """Never copy arbitrary exception text, paths, URLs, frames or credentials to public views."""
    if code is None:
        return None
    return _PUBLIC_REASONS.get(
        code, "Evaluation could not proceed; inspect the published error code.")


def public_duplicate_of(value: object) -> str | None:
    if value == "genesis":
        return "genesis"  # Reference to the official baseline, never a synthetic challenge row.
    if (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)):
        return value
    return None


def webhook_alerter(
    url: str | None,
    *,
    service: str,
) -> Callable[[str, Mapping[str, object]], None]:
    if not service:
        raise ValueError("alert service cannot be empty")
    if url is None:
        LOGGER.warning("%s push alerts are not configured; failure events use logs", service)

        def log(event: str, details: Mapping[str, object]) -> None:
            LOGGER.error(
                "%s: %s %s", service, event, dict(details),
                extra={"alert_event": event, "alert_service": service},
            )

        return log
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, TypeError):
        raise ValueError("alert webhook URL must be an absolute HTTPS URL") from None
    if parsed.scheme != "https" or not parsed.host or parsed.userinfo or parsed.fragment:
        raise ValueError("alert webhook URL must use HTTPS without credentials or fragments")
    def send(event: str, details: Mapping[str, object]) -> None:
        payload = {"event": event, "details": dict(details), "service": service}
        try:
            response = httpx.post(
                url,
                content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                headers={"content-type": "application/json"},
                timeout=5.0,
            )
            response.raise_for_status()
        except Exception as exc:
            LOGGER.error(
                "push alert delivery failed (%s)",
                type(exc).__name__,
                extra={"alert_event": event, "alert_service": service},
            )

    return send
