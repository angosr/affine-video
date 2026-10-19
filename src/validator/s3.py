"""Shared S3 transport settings and errors for public and private storage."""

from __future__ import annotations

import os
from typing import Any


def is_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = response.get("Error", {}).get("Code")
    return status == 404 or str(code) in {"404", "NoSuchKey", "NotFound"}


def is_precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    code = response.get("Error", {}).get("Code")
    return status == 412 or str(code) in {"412", "PreconditionFailed"}


def client_config() -> Any:
    from botocore.config import Config

    return Config(
        region_name=os.environ.get("AWS_DEFAULT_REGION") or "auto",
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
    )
