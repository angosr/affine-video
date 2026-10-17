"""Immutable summaries of the final launch-time H200 stability run."""

from __future__ import annotations

import hashlib
import json
from math import fsum
from typing import Any

from scoring.spec import TrackSpec, track_spec_dict

PRODUCTION_TRACK_ID = "veda-sn120-v2"
PRODUCTION_NETUID = 120
PRODUCTION_EVIDENCE_CONTRACT_DIGEST = (
    "eb091449d2ab176ad6f9136461d9b88c1d69061b1db3175b11192e289c58aa59"
)
_EVIDENCE_SHA256 = "87ba4871c29df4f5113b4051c4f2d1e6905be834ef9a54dfcb780a675fc357db"
_RUNTIME_TRACE_EVIDENCE_SHA256 = "c6e873d8d9acbcb02814eae888fb30fa73e620d73fca03f99a7f8f30f10b3fa5"
_MINIMUM_GAIN = 0.005
_BASELINE_SCORES = (
    0.965935974938656,
    0.942681827805732,
    0.962320576535515,
    0.974050067809449,
    0.947151104436117,
    0.940292660403239,
    0.897184126503376,
    0.923065939590511,
    0.927028997571669,
    0.971341256693279,
)
_REFERENCE = {
    "model_id": "veda_baseline",
    "label": "VEDA baseline",
    "sha256": "64addb5da430059e37e405d28ca0f2887132905f077ae92fba86982c8b0e9d3f",
}
_CONTROLS = (
    {
        "model_id": "mean_pool_control",
        "label": "MeanPool control",
        "sha256": "53bd350490e16702f70f69d33e10e241a134573f27f706caf82647092342b025",
        "scores": (
            0.962799095078332,
            0.934511712220001,
            0.956489263685866,
            0.961562762012376,
            0.943030204044916,
            0.937266331544423,
            0.892189299288133,
            0.919444076867836,
            0.923476631198779,
            0.965579680875613,
        ),
        "gain_lower_bounds": (
            -0.006498988939666,
            -0.031164102845540,
            -0.014396288190415,
            -0.032205051765224,
            -0.009379684047755,
            -0.009432251177008,
            -0.067030688874896,
            -0.011586045684992,
            -0.010026242209518,
            -0.009397013291712,
        ),
        "outcome": "REFERENCE_RETAINED",
    },
    {
        "model_id": "shuffled_heads_control",
        "label": "Shuffled-heads control",
        "sha256": "83235ac1c8c0500c108f8f23c5a45eb13ab4a69f294f506269641615a7394360",
        "scores": (
            0.852781882861242,
            0.812557190537774,
            0.834177581969323,
            0.863479207110985,
            0.838107927972904,
            0.837237025404604,
            0.750811432166654,
            0.821082490911290,
            0.788100556446850,
            0.819857745358149,
        ),
        "gain_lower_bounds": (
            -0.135478781935666,
            -0.160052462233410,
            -0.173646161759022,
            -0.193122903638280,
            -0.165247508554978,
            -0.166235029383888,
            -0.226321600365791,
            -0.173563771818016,
            -0.181302089781703,
            -0.215340142268243,
        ),
        "reference_gain_lower_bounds": (
            0.092622429320872,
            0.103585650444588,
            0.087541190870046,
            0.043737593915000,
            0.060357618632067,
            0.042795471317278,
            0.040568520436958,
            0.047172395266923,
            0.102444352400404,
            0.090486359687219,
        ),
        "outcome": "REFERENCE_WON",
    },
    {
        "model_id": "perturbed_veda_control",
        "label": "Perturbed VEDA control",
        "sha256": "74b8c5ad043450250acee23572f35c2be8eabf2141f01cb4e06abf259d694eaf",
        "scores": (
            0.965955427449611,
            0.942422722618746,
            0.962379876857487,
            0.973877472191853,
            0.947140700239371,
            0.940142028668577,
            0.897135290183366,
            0.923316191936039,
            0.927029572062086,
            0.971308320091918,
        ),
        "gain_lower_bounds": (
            -0.000617914064848,
            -0.000959966919615,
            -0.000353700764687,
            -0.000736079039864,
            -0.000490573831764,
            -0.000878954560204,
            -0.000753818736879,
            -0.000283273693010,
            -0.000703698577301,
            -0.000474816048444,
        ),
        "outcome": "REFERENCE_RETAINED",
    },
)


def system_match_artifacts(track: TrackSpec) -> tuple[tuple[str, bytes], ...]:
    """Return evidence only for its frozen, non-circular scoring contract."""

    if (
        track.track_id != PRODUCTION_TRACK_ID
        or track.netuid != PRODUCTION_NETUID
        or _evidence_contract_digest(track) != PRODUCTION_EVIDENCE_CONTRACT_DIGEST
    ):
        return ()
    return tuple(
        (
            f"system_matches/{track.track_id}/{sequence:012d}.json",
            _json_bytes(_match_document(sequence, control, track.digest)),
        )
        for sequence, control in enumerate(_CONTROLS, start=1)
    )


def system_match_summaries(track: TrackSpec) -> list[dict[str, Any]]:
    """Return compact launch-check rows for the live status document."""

    summaries = []
    for name, payload in system_match_artifacts(track):
        document = json.loads(payload)
        summaries.append(
            {
                "sequence": document["sequence"],
                "reference": document["reference"]["label"],
                "challenger": document["challenger"]["label"],
                "reference_score": document["scores"]["reference"],
                "challenger_score": document["scores"]["challenger"],
                "outcome": document["outcome"],
                "detail": name,
            }
        )
    return summaries


def _match_document(
    sequence: int,
    control: dict[str, Any],
    track_spec_digest: str,
) -> dict[str, Any]:
    challenger_scores = control["scores"]
    batches = [
        {
            "batch": index,
            "reference_score": reference,
            "challenger_score": challenger,
            "challenger_gain_lower_bound": gain,
        }
        for index, (reference, challenger, gain) in enumerate(
            zip(
                _BASELINE_SCORES,
                challenger_scores,
                control["gain_lower_bounds"],
                strict=True,
            )
        )
    ]
    reference_gains = control.get("reference_gain_lower_bounds")
    if reference_gains is not None:
        for batch, gain in zip(batches, reference_gains, strict=True):
            batch["reference_gain_lower_bound"] = gain
    return {
        "record_type": "SYSTEM_BENCHMARK",
        "sequence": sequence,
        "track_id": PRODUCTION_TRACK_ID,
        "track_spec_digest": track_spec_digest,
        "netuid": PRODUCTION_NETUID,
        "affects_tournament": False,
        "reward_eligible": False,
        "reference": _REFERENCE,
        "challenger": {key: control[key] for key in ("model_id", "label", "sha256")},
        "evaluation": {
            "hardware": "NVIDIA H200",
            "torch": "2.12.0+cu130",
            "cuda": "13.0",
            "track_contract_sha256": PRODUCTION_EVIDENCE_CONTRACT_DIGEST,
            "batch_count": 10,
            "cases_per_batch": 384,
            "independent_videos_per_stratum": 4,
            "confidence": 0.9975,
            "minimum_gain": _MINIMUM_GAIN,
            "cluster_unit": "source_video",
            "evidence_sha256": _EVIDENCE_SHA256,
            "runtime_trace": {
                "evidence_sha256": _RUNTIME_TRACE_EVIDENCE_SHA256,
                "cases": 16,
                "sampled_routes": 8,
                "descriptor_max_abs": 2.384185791015625e-07,
                "logits_max_abs": 6.198883056640625e-06,
                "mask_mismatches": 0,
                "output_max_abs": 0.0625,
                "output_mean_abs": 3.039836883544922e-06,
            },
            "batches": batches,
        },
        "scores": {
            "reference": fsum(_BASELINE_SCORES) / len(_BASELINE_SCORES),
            "challenger": fsum(challenger_scores) / len(challenger_scores),
        },
        "outcome": control["outcome"],
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _evidence_contract_digest(track: TrackSpec) -> str:
    """Bind score-affecting track data without depending on its runtime manifest."""

    document = track_spec_dict(track)
    del document["artifact_manifest_sha256"]
    return hashlib.sha256(_json_bytes(document)).hexdigest()
