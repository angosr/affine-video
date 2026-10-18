"""Failure logs with optional best-effort HTTPS notifications."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping

import httpx

LOGGER = logging.getLogger(__name__)


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
