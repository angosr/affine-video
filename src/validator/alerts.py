"""Best-effort push alerts for durable validator failure states."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping

import httpx

LOGGER = logging.getLogger(__name__)


def webhook_alerter(
    url: str,
    *,
    service: str,
) -> Callable[[str, Mapping[str, object]], None]:
    if not url.startswith("https://"):
        raise ValueError("alert webhook URL must use HTTPS")
    if not service:
        raise ValueError("alert service cannot be empty")

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
        except Exception:
            LOGGER.exception("push alert delivery failed", extra={"alert_event": event})

    return send
