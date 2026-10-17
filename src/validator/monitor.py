"""Read-only terminal monitor for the public competition status."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from validator.publication import PublicArtifactError, validate_public_weights_url

_MAX_STATUS_BYTES = 1024 * 1024


async def fetch_status(url: str, *, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch one bounded public status snapshot."""

    validate_public_weights_url(url)
    received = bytearray()
    try:
        async with client.stream(
            "GET", url, headers={"Accept": "application/json", "Cache-Control": "no-cache"}
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                received.extend(chunk)
                if len(received) > _MAX_STATUS_BYTES:
                    raise PublicArtifactError("public status exceeds the byte limit")
    except httpx.HTTPError as exc:
        raise PublicArtifactError("public status could not be fetched") from exc
    try:
        value = json.loads(received)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicArtifactError("public status is not valid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("participants"), list):
        raise PublicArtifactError("public status fields are invalid")
    return value


def render_status(status: Mapping[str, Any], *, color: bool | None = None) -> str:
    """Render a compact leaderboard, active match history, and FIFO queue."""

    participants = status.get("participants")
    if not isinstance(participants, Sequence) or isinstance(participants, (str, bytes)):
        raise PublicArtifactError("public status participants are invalid")
    use_color = sys.stdout.isatty() if color is None else color
    cyan = "\033[36m" if use_color else ""
    green = "\033[32m" if use_color else ""
    yellow = "\033[33m" if use_color else ""
    bold = "\033[1m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    title = (
        f"{bold}{cyan}VEDA SN{status.get('netuid', '?')}  "
        f"block {status.get('source_block', '?')}{reset}"
    )
    headers = ("UID", "HOTKEY", "STATUS", "QUEUE", "MATCH", "SCORE", "GAIN", "RESULT")
    rows: list[tuple[str, ...]] = []
    for raw in participants:
        if not isinstance(raw, Mapping):
            raise PublicArtifactError("public status participant is invalid")
        hotkey = raw.get("hotkey")
        if not isinstance(hotkey, str) or not hotkey:
            raise PublicArtifactError("public status hotkey is invalid")
        champion = raw.get("champion") is True
        state = str(raw.get("state", "?"))
        status_text = f"{green}CHAMPION{reset}" if champion else state
        queue = raw.get("queue_position")
        match = raw.get("match_sequence")
        score = raw.get("score")
        gain = raw.get("score_gain")
        outcome = raw.get("outcome")
        rows.append(
            (
                str(raw.get("uid", "?")),
                hotkey[:6],
                status_text,
                f"{yellow}#{queue}{reset}" if queue is not None else "-",
                f"#{match}" if match is not None else "-",
                f"{float(score):.6f}" if isinstance(score, (int, float)) else "-",
                f"{float(gain):+.6f}" if isinstance(gain, (int, float)) else "-",
                str(outcome) if outcome is not None else "-",
            )
        )
    widths = [len(header) for header in headers]
    plain_rows = [[_without_ansi(cell) for cell in row] for row in rows]
    for row in plain_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    separator = "  ".join("─" * width for width in widths)
    lines = [title, "", _format_row(headers, widths), separator]
    if rows:
        lines.extend(
            _format_colored_row(row, plain, widths)
            for row, plain in zip(rows, plain_rows, strict=True)
        )
    else:
        lines.append("No participants yet")
    system_matches = status.get("system_matches", [])
    if not isinstance(system_matches, Sequence) or isinstance(system_matches, (str, bytes)):
        raise PublicArtifactError("public status system matches are invalid")
    if system_matches:
        lines.extend(["", f"{bold}SYSTEM BENCHMARKS{reset}"])
        system_headers = ("MATCH", "REFERENCE", "CHALLENGER", "REF SCORE", "CHAL SCORE", "RESULT")
        system_rows: list[tuple[str, ...]] = []
        for raw in system_matches:
            if not isinstance(raw, Mapping):
                raise PublicArtifactError("public status system match is invalid")
            reference_score = raw.get("reference_score")
            challenger_score = raw.get("challenger_score")
            if not isinstance(reference_score, (int, float)) or not isinstance(
                challenger_score, (int, float)
            ):
                raise PublicArtifactError("public status system match score is invalid")
            system_rows.append(
                (
                    f"#{raw.get('sequence', '?')}",
                    str(raw.get("reference", "?")),
                    str(raw.get("challenger", "?")),
                    f"{float(reference_score):.6f}",
                    f"{float(challenger_score):.6f}",
                    str(raw.get("outcome", "?")),
                )
            )
        system_widths = [len(header) for header in system_headers]
        for row in system_rows:
            for index, cell in enumerate(row):
                system_widths[index] = max(system_widths[index], len(cell))
        lines.extend(
            [
                _format_row(system_headers, system_widths),
                "  ".join("─" * width for width in system_widths),
                *(_format_row(row, system_widths) for row in system_rows),
            ]
        )
        lines.append("System benchmarks do not affect tournament rank or chain weights.")
    return "\n".join(lines)


def _format_row(row: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))


def _format_colored_row(row: Sequence[str], plain: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(
        value + " " * (width - len(plain_value))
        for value, plain_value, width in zip(row, plain, widths, strict=True)
    )


def _without_ansi(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index : index + 2] == "\033[":
            end = value.find("m", index + 2)
            if end >= 0:
                index = end + 1
                continue
        result.append(value[index])
        index += 1
    return "".join(result)
