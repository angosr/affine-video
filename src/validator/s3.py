"""Shared Supabase S3 compatibility settings for large immutable files."""

from __future__ import annotations

from typing import Any


def client_config() -> Any:
    from botocore.config import Config

    return Config(
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
    )
