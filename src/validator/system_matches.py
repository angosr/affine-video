"""Fixed public records for the launch-time GPU system checks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

PRODUCTION_TRACK_ID = "veda-sn120-v1"
PRODUCTION_NETUID = 120

_REFERENCE = {
    "model_id": "veda_baseline_bf16",
    "label": "VEDA baseline",
    "sha256": "472b386ec3df03c204f9685a1cc74ac6bb530d03e7f33c3b668c3afdca6118e6",
}
_CHALLENGERS = (
    {
        "model_id": "mean_pool_control",
        "label": "MeanPool control",
        "sha256": "33a7ed49f87ec1402d0d4e2597391c177dfde94a3cee453a885b6e7f48d6ec3c",
        "result_key": "mean_pool",
    },
    {
        "model_id": "shuffled_heads_control",
        "label": "Shuffled-heads control",
        "sha256": "4177dca16ebe61d1de544aa93b4fa2aff4c240993f1ad2633f51b86c21db3b32",
        "result_key": "shuffled",
    },
)
_PANELS = (
    {
        "panel_id": "panel4_00",
        "geometry": "16x9_t37",
        "tokens": 37855,
        "summary_sha256": ("818fb2105663ef859b4a883b2b83b09de4d3c3a383d07efe0f6b6d9f4ee0a9f9"),
        "results": {
            "release": {"heat_kept": 0.5144105251778716, "recall": 0.5799246942557926},
            "mean_pool": {"heat_kept": 0.4694173505390063, "recall": 0.5114823160127604},
            "shuffled": {"heat_kept": 0.2078434120777196, "recall": 0.15159557184137937},
        },
    },
    {
        "panel_id": "panel4_01",
        "geometry": "9x16_t37",
        "tokens": 37846,
        "summary_sha256": ("720790704eb5900ad290dc15d0a2788c4001e818e19c984c81849f1883aaf5f4"),
        "results": {
            "release": {"heat_kept": 0.5054031360575131, "recall": 0.5814783874859235},
            "mean_pool": {"heat_kept": 0.4547006962295356, "recall": 0.49843065332554815},
            "shuffled": {"heat_kept": 0.2020702588554871, "recall": 0.1487018204839634},
        },
    },
    {
        "panel_id": "panel4_02",
        "geometry": "1x1_t72",
        "tokens": 42460,
        "summary_sha256": ("0b65c67f45aad3e4daa4274deb177714a92cb3ab49960110182781095676bdd8"),
        "results": {
            "release": {"heat_kept": 0.5689236869037684, "recall": 0.6099233027108546},
            "mean_pool": {"heat_kept": 0.5340663304319605, "recall": 0.5624384149456663},
            "shuffled": {"heat_kept": 0.2424563023045526, "recall": 0.1795284706849738},
        },
    },
    {
        "panel_id": "panel4_03",
        "geometry": "4x3_t102",
        "tokens": 79697,
        "summary_sha256": ("c3280cd1c7d7c90fc8c51086ebe27af16973f5ea2d86ccbe02ed7701f1328c7d"),
        "results": {
            "release": {"heat_kept": 0.6081481993397964, "recall": 0.6148099459521472},
            "mean_pool": {"heat_kept": 0.5638526942434587, "recall": 0.5629377723731367},
            "shuffled": {"heat_kept": 0.2466982182348147, "recall": 0.18051299558387005},
        },
    },
)


def system_match_artifacts(
    track_id: str,
    track_spec_digest: str,
    netuid: int,
) -> tuple[tuple[str, bytes], ...]:
    """Return immutable system-match objects for the production track only."""

    if track_id != PRODUCTION_TRACK_ID or netuid != PRODUCTION_NETUID:
        return ()
    return tuple(
        (
            f"system_matches/{track_id}/{sequence:012d}.json",
            _json_bytes(
                _match_document(
                    sequence,
                    challenger,
                    track_id=track_id,
                    track_spec_digest=track_spec_digest,
                    netuid=netuid,
                )
            ),
        )
        for sequence, challenger in enumerate(_CHALLENGERS, start=1)
    )


def system_match_summaries(
    track_id: str,
    track_spec_digest: str,
    netuid: int,
) -> list[dict[str, Any]]:
    """Return the compact system-match rows embedded in the live status."""

    artifacts = system_match_artifacts(track_id, track_spec_digest, netuid)
    summaries: list[dict[str, Any]] = []
    for name, payload in artifacts:
        document = json.loads(payload)
        summaries.append(
            {
                "sequence": document["sequence"],
                "reference": document["reference"]["label"],
                "challenger": document["challenger"]["label"],
                "reference_score": document["scores"]["reference"]["heat_kept"],
                "challenger_score": document["scores"]["challenger"]["heat_kept"],
                "outcome": document["outcome"],
                "detail": name,
            }
        )
    return summaries


def _match_document(
    sequence: int,
    challenger: dict[str, str],
    *,
    track_id: str,
    track_spec_digest: str,
    netuid: int,
) -> dict[str, Any]:
    result_key = challenger["result_key"]
    panels = [
        {
            "panel_id": panel["panel_id"],
            "geometry": panel["geometry"],
            "tokens": panel["tokens"],
            "summary_sha256": panel["summary_sha256"],
            "reference": panel["results"]["release"],
            "challenger": panel["results"][result_key],
        }
        for panel in _PANELS
    ]
    reference_score = _mean(panel["reference"]["heat_kept"] for panel in panels)
    challenger_score = _mean(panel["challenger"]["heat_kept"] for panel in panels)
    reference_recall = _mean(panel["reference"]["recall"] for panel in panels)
    challenger_recall = _mean(panel["challenger"]["recall"] for panel in panels)
    return {
        "record_type": "SYSTEM_BENCHMARK",
        "sequence": sequence,
        "track_id": track_id,
        "track_spec_digest": track_spec_digest,
        "netuid": netuid,
        "affects_tournament": False,
        "reward_eligible": False,
        "reference": _REFERENCE,
        "challenger": {key: challenger[key] for key in ("model_id", "label", "sha256")},
        "evaluation": {
            "hardware": "NVIDIA H200",
            "precision": "bfloat16",
            "steps_per_panel": 8,
            "keep_ratio": 0.1,
            "training_step": 100,
            "primary_metric": "heat_kept",
            "aggregate_method": "arithmetic_mean_across_panels",
            "panels": panels,
        },
        "scores": {
            "reference": {"heat_kept": reference_score, "recall": reference_recall},
            "challenger": {"heat_kept": challenger_score, "recall": challenger_recall},
        },
        "outcome": "REFERENCE_WON",
    }


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return sum(items) / len(items)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
