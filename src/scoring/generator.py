"""Public deterministic planning for post-submission evaluation batches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannedCase:
    route: str
    stratum: str
    sample_id: str


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    sample_id: str
    stratum: str
    prompt: str
    noise_seed: int


_SUBJECTS = (
    "glass sphere",
    "paper kite",
    "wooden automaton",
    "sailing boat",
    "white deer",
    "clockwork bird",
)
_ACTIONS = (
    "crosses a dark plain",
    "turns above still water",
    "moves through falling rain",
    "circles a stone tower",
    "enters a field of flowers",
    "passes beneath an arch",
)
_CAMERAS = (
    "slow tracking shot",
    "wide aerial shot",
    "low dolly shot",
    "overhead shot",
    "steady close shot",
    "long crane shot",
)
_LIGHTS = (
    "at blue hour",
    "at sunrise",
    "under warm studio light",
    "during a lightning flash",
    "in soft fog",
    "under moonlight",
)


def generate_sources(
    *,
    track_spec_digest: str,
    seed_block_hash: str,
    strata: Sequence[str],
    samples_per_stratum: int,
) -> tuple[GeneratedSample, ...]:
    """Generate a fresh, reproducible prompt/noise manifest from a finalized hash."""

    if samples_per_stratum <= 0:
        raise ValueError("samples per stratum must be positive")
    track = _digest(track_spec_digest, "track specification digest")
    block = _digest(seed_block_hash.removeprefix("0x"), "seed block hash")
    seed = hashlib.sha256(b"veda-source-v1\0" + track + block).digest()
    result: list[GeneratedSample] = []
    for stratum in _unique(strata, "stratum"):
        for index in range(samples_per_stratum):
            material = hashlib.sha256(
                seed + b"\0" + stratum.encode() + index.to_bytes(4, "big")
            ).digest()
            parts = (
                _SUBJECTS[material[0] % len(_SUBJECTS)],
                _ACTIONS[material[1] % len(_ACTIONS)],
                _CAMERAS[material[2] % len(_CAMERAS)],
                _LIGHTS[material[3] % len(_LIGHTS)],
            )
            prompt = f"{parts[2]}: a {parts[0]} {parts[1]} {parts[3]}."
            sample_id = hashlib.sha256(b"veda-sample-v1\0" + material + prompt.encode()).hexdigest()
            result.append(
                GeneratedSample(
                    sample_id,
                    stratum,
                    prompt,
                    int.from_bytes(material[8:16], "big"),
                )
            )
    return tuple(result)


def source_digest(samples: Sequence[GeneratedSample]) -> str:
    """Hash the canonical source manifest recorded by an exact batch."""

    payload = [
        {
            "noise_seed": sample.noise_seed,
            "prompt": sample.prompt,
            "sample_id": sample.sample_id,
            "stratum": sample.stratum,
        }
        for sample in samples
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scheduled_seed_block(
    submission_block: int,
    *,
    track_start_block: int,
    minimum_delay_blocks: int,
    generation_interval_blocks: int,
) -> int:
    """Return the only batch boundary eligible for a submission."""

    if min(submission_block, track_start_block) < 0:
        raise ValueError("block heights must be non-negative")
    if min(minimum_delay_blocks, generation_interval_blocks) <= 0:
        raise ValueError("generation timing must be positive")
    earliest = submission_block + minimum_delay_blocks
    offset = max(0, earliest - track_start_block)
    return (
        track_start_block
        + ((offset + generation_interval_blocks - 1) // generation_interval_blocks)
        * generation_interval_blocks
    )


def plan_batch(
    *,
    track_spec_digest: str,
    seed_block_hash: str,
    routes: Sequence[str],
    samples_by_stratum: Mapping[str, Sequence[str]],
    sample_count: int,
) -> tuple[PlannedCase, ...]:
    """Choose routes and source samples solely from a finalized block hash."""

    track_bytes = _digest(track_spec_digest, "track specification digest")
    block_bytes = _digest(seed_block_hash.removeprefix("0x"), "seed block hash")
    strata = tuple(sorted(samples_by_stratum))
    if not strata or sample_count <= 0 or sample_count % len(strata):
        raise ValueError("sample count must divide evenly across non-empty strata")
    route_count = sample_count // len(strata)
    unique_routes = _unique(routes, "route")
    if route_count > len(unique_routes):
        raise ValueError("not enough routes for the requested batch")
    seed = hashlib.sha256(b"veda-batch-plan-v1\0" + track_bytes + block_bytes).digest()
    selected_routes = _ranked(unique_routes, seed, b"route")[:route_count]
    selected_samples: dict[str, tuple[str, ...]] = {}
    for stratum in strata:
        items = _unique(samples_by_stratum[stratum], "sample identifier")
        if route_count > len(items):
            raise ValueError(f"not enough samples in stratum {stratum}")
        selected_samples[stratum] = _ranked(
            items,
            seed,
            b"sample\0" + stratum.encode("utf-8"),
        )[:route_count]
    return tuple(
        PlannedCase(route, stratum, selected_samples[stratum][index])
        for index, route in enumerate(selected_routes)
        for stratum in strata
    )


def _ranked(values: tuple[str, ...], seed: bytes, domain: bytes) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(seed + b"\0" + domain + b"\0" + value.encode()).digest(),
                value,
            ),
        )
    )


def _unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name}s must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name}s must be unique")
    return result


def _digest(value: str, name: str) -> bytes:
    try:
        result = bytes.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    if len(result) != 32:
        raise ValueError(f"{name} must contain 32 bytes")
    return result
