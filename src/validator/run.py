"""One bounded validator work cycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import bittensor as bt

from scoring.cache import validate_exact_cache, validate_screen_cache
from scoring.model import ValidatedModel, validate_model
from scoring.spec import load_track_spec
from validator.admission import (
    AdmissionOutcome,
    AdmissionSettings,
    process_next_admission,
)
from validator.chain import ScanResult, scan_finalized_blocks
from validator.download import DownloadedModel, download_public_model
from validator.scoring import (
    FileTrackEvaluator,
    IsolatedEvaluator,
    ScoringOutcome,
    ScoringSettings,
    StageOutcome,
    process_next_scoring,
)
from validator.store import ChallengeStore, LeasedChallenge, WeightRun
from validator.weights import update_weights


@dataclass(frozen=True, slots=True)
class CycleResult:
    finalized_block: int
    scan: ScanResult
    admission: AdmissionOutcome | None
    scoring: ScoringOutcome | None
    weight_run: WeightRun | None = None
    weight_error: str | None = None


async def run_validator(
    *,
    track_spec: str | Path,
    database: str | Path,
    model_cache: str | Path,
    screen_cache: str | Path,
    exact_cache: str | Path,
    network: str,
    archive_endpoints: list[str] | None,
    worker_id: str,
    poll_seconds: float = 12.0,
    max_scan_blocks: int = 100,
    scoring_timeout_seconds: float = 30 * 60,
    scoring_max_memory_bytes: int | None = None,
    scoring_max_cpu_seconds: int | None = None,
    submit_chain_weights: bool = False,
    weight_interval_blocks: int = 100,
    wallet_name: str = "default",
    hotkey_name: str = "default",
    wallet_path: str | None = None,
    mechid: int = 0,
    version_key: int = 0,
    once: bool = False,
    report: Callable[[CycleResult], None] | None = None,
) -> None:
    """Run the validator loop using one frozen track and two verified caches."""

    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    track = load_track_spec(track_spec)
    validate_screen_cache(
        screen_cache,
        expected_sha256=track.cache.screen_sha256,
        max_bytes=track.cache.screen_max_bytes,
        routes=track.model.selector.routes,
        strata=track.scoring.stratum_weight_map,
        sample_count=track.scoring.screen_sample_count,
    )
    validate_exact_cache(
        exact_cache,
        expected_sha256=track.cache.exact_sha256,
        max_bytes=track.cache.exact_max_bytes,
        routes=track.model.selector.routes,
        strata=track.scoring.stratum_weight_map,
        sample_count=track.scoring.local_sample_count,
    )
    admission_settings = AdmissionSettings(
        cache_dir=Path(model_cache),
        max_model_bytes=track.model.max_bytes,
        tensor_schema=track.model.tensor_schema,
    )
    evaluator = IsolatedEvaluator(
        FileTrackEvaluator(track, str(screen_cache), str(exact_cache)),
        timeout_seconds=scoring_timeout_seconds,
        max_memory_bytes=scoring_max_memory_bytes,
        max_cpu_seconds=scoring_max_cpu_seconds,
    )
    scoring_settings = ScoringSettings(lease_ms=ceil(evaluator.timeout_seconds * 1000) + 60_000)

    with ChallengeStore(database, track_spec_digest=track.digest) as store:
        async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
            while True:
                result = await process_cycle(
                    client,
                    store,
                    netuid=track.netuid,
                    track_start_block=track.track_start_block,
                    cooldown_blocks=track.submission_cooldown_blocks,
                    seed_delay_blocks=track.seed_delay_blocks,
                    admission_settings=admission_settings,
                    scoring_settings=scoring_settings,
                    evaluator=evaluator,
                    worker_id=worker_id,
                    max_scan_blocks=max_scan_blocks,
                )
                if submit_chain_weights:
                    try:
                        weight_run = await update_weights(
                            client,
                            store,
                            netuid=track.netuid,
                            network=network,
                            wallet_name=wallet_name,
                            hotkey_name=hotkey_name,
                            wallet_path=wallet_path,
                            mechid=mechid,
                            version_key=version_key,
                            half_life_blocks=track.weights.freshness_half_life_blocks,
                            max_age_blocks=track.weights.max_age_blocks,
                            provisional_max_weight=track.weights.provisional_max_weight,
                            interval_blocks=weight_interval_blocks,
                        )
                        result = CycleResult(
                            result.finalized_block,
                            result.scan,
                            result.admission,
                            result.scoring,
                            weight_run=weight_run,
                        )
                    except Exception as exc:
                        result = CycleResult(
                            result.finalized_block,
                            result.scan,
                            result.admission,
                            result.scoring,
                            weight_error=str(exc) or type(exc).__name__,
                        )
                if report is not None:
                    report(result)
                if once:
                    return
                await asyncio.sleep(poll_seconds)


async def process_cycle(
    client: Any,
    store: ChallengeStore,
    *,
    netuid: int,
    track_start_block: int,
    cooldown_blocks: int,
    seed_delay_blocks: int,
    admission_settings: AdmissionSettings,
    scoring_settings: ScoringSettings,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    worker_id: str,
    max_scan_blocks: int = 100,
    downloader: Callable[..., DownloadedModel] = download_public_model,
    model_validator: Callable[..., ValidatedModel] = validate_model,
) -> CycleResult:
    """Synchronize the chain, admit one model, and score one ready stage."""

    if not worker_id:
        raise ValueError("worker id cannot be empty")
    scan = await scan_finalized_blocks(
        client,
        store,
        netuid=netuid,
        track_start_block=track_start_block,
        cooldown_blocks=cooldown_blocks,
        seed_delay_blocks=seed_delay_blocks,
        max_blocks=max_scan_blocks,
    )
    finalized_block = await client.finalized_block()
    admission = process_next_admission(
        store,
        worker_id=f"{worker_id}:admission",
        current_finalized_block=finalized_block,
        settings=admission_settings,
        downloader=downloader,
        model_validator=model_validator,
    )
    scoring = process_next_scoring(
        store,
        worker_id=f"{worker_id}:scoring",
        current_finalized_block=finalized_block,
        settings=scoring_settings,
        evaluator=evaluator,
    )
    return CycleResult(finalized_block, scan, admission, scoring)
