"""Shared S3 transport settings for public and private object storage."""

from __future__ import annotations

import os
from typing import Any


def client_config() -> Any:
    from botocore.config import Config

    return Config(
        region_name=os.environ.get("AWS_DEFAULT_REGION") or "auto",
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
    )
