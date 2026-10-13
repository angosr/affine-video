"""Owner evaluation and independent weight-setting loops."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Protocol

import bittensor as bt
import httpx

from scoring.cache import validate_exact_cache, validate_screen_cache
from scoring.model import ValidatedModel, validate_model
from scoring.rollout import ProcessRollout
from scoring.spec import load_track_spec
from validator.admission import (
    AdmissionOutcome,
    AdmissionSettings,
    process_next_admission,
)
from validator.chain import ScanResult, scan_finalized_blocks
from validator.download import DownloadedModel, download_public_model
from validator.publication import (
    PublicArtifacts,
    build_public_artifacts,
    fetch_weight_artifact,
    validate_public_weights_url,
)
from validator.scoring import (
    FileTrackEvaluator,
    IsolatedEvaluator,
    ScoringOutcome,
    ScoringSettings,
    StageOutcome,
    process_next_scoring,
)
from validator.store import ChallengeStore, LeasedChallenge
from validator.weights import (
    WeightReceipt,
    read_weight_chain_state,
    submit_weights,
    validate_weight_artifact,
    weight_update_due,
)


class ArtifactPublisher(Protocol):
    async def publish(self, artifacts: PublicArtifacts) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationCycleResult:
    finalized_block: int
    scan: ScanResult
    admission: AdmissionOutcome | None
    scoring: ScoringOutcome | None
    published_block: int | None = None
    publication_error: str | None = None


@dataclass(frozen=True, slots=True)
class WeightCycleResult:
    finalized_block: int | None
    source_block: int | None
    status: str
    receipt: WeightReceipt | None = None
    error: str | None = None


async def run_evaluator(
    *,
    track_spec: str | Path,
    database: str | Path,
    model_cache: str | Path,
    screen_cache: str | Path,
    exact_cache: str | Path,
    rollout_runner: str | Path | None,
    rollout_panel: str | Path | None,
    network: str,
    archive_endpoints: list[str] | None,
    worker_id: str,
    publisher: ArtifactPublisher,
    mechid: int = 0,
    version_key: int = 0,
    poll_seconds: float = 12.0,
    max_scan_blocks: int = 100,
    scoring_timeout_seconds: float = 30 * 60,
    rollout_timeout_seconds: float = 25 * 60,
    scoring_max_memory_bytes: int | None = None,
    scoring_max_cpu_seconds: int | None = None,
    once: bool = False,
    report: Callable[[EvaluationCycleResult], None] | None = None,
) -> None:
    """Evaluate submissions and publish public artifacts without loading a wallet."""

    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if min(mechid, version_key) < 0:
        raise ValueError("mechanism and version identifiers must be non-negative")
    if (rollout_runner is None) != (rollout_panel is None):
        raise ValueError("rollout runner and panel must be configured together")
    if rollout_runner is not None and rollout_timeout_seconds >= scoring_timeout_seconds:
        raise ValueError("rollout timeout must be shorter than the scoring process timeout")
    track = load_track_spec(track_spec)
    rollout = (
        ProcessRollout(
            runner=str(rollout_runner),
            panel=str(rollout_panel),
            runner_sha256=track.rollout.runner_sha256,
            panel_sha256=track.rollout.panel_sha256,
            panel_max_bytes=track.rollout.panel_max_bytes,
            timeout_seconds=rollout_timeout_seconds,
        )
        if rollout_runner is not None and rollout_panel is not None
        else None
    )
    validate_screen_cache(
        screen_cache,
        expected_sha256=track.cache.screen_sha256,
        max_bytes=track.cache.screen_max_bytes,
        max_blocks=track.scoring.budgets[-1].blocks,
        head_dim=track.model.selector.head_dim,
        routes=track.model.selector.routes,
        strata=track.scoring.stratum_weight_map,
    )
    validate_exact_cache(
        exact_cache,
        expected_sha256=track.cache.exact_sha256,
        max_bytes=track.cache.exact_max_bytes,
        max_blocks=track.scoring.budgets[-1].blocks,
        head_dim=track.model.selector.head_dim,
        routes=track.model.selector.routes,
        strata=track.scoring.stratum_weight_map,
        sample_count=track.scoring.local_sample_count,
    )
    admission_settings = AdmissionSettings(
        cache_dir=Path(model_cache),
        max_model_bytes=track.model.max_bytes,
        tensor_schema=track.model.tensor_schema,
        publication_grace_blocks=track.publication_grace_blocks,
    )
    evaluator = IsolatedEvaluator(
        FileTrackEvaluator(track, str(screen_cache), str(exact_cache), rollout=rollout),
        timeout_seconds=scoring_timeout_seconds,
        max_memory_bytes=scoring_max_memory_bytes,
        max_cpu_seconds=scoring_max_cpu_seconds,
    )
    scoring_settings = ScoringSettings(lease_ms=ceil(evaluator.timeout_seconds * 1000) + 60_000)

    with ChallengeStore(database, track_spec_digest=track.digest) as store:
        if rollout is not None:
            store.schedule_qualified_rollouts(track.scoring.rollout_threshold)
        async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
            while True:
                result = await process_cycle(
                    client,
                    store,
                    netuid=track.netuid,
                    track_start_block=track.track_start_block,
                    cooldown_blocks=track.submission_cooldown_blocks,
                    admission_settings=admission_settings,
                    scoring_settings=scoring_settings,
                    evaluator=evaluator,
                    worker_id=worker_id,
                    max_scan_blocks=max_scan_blocks,
                )
                cursor = store.chain_cursor()
                if cursor is not None and cursor[0] == result.finalized_block:
                    try:
                        chain = await read_weight_chain_state(
                            client,
                            netuid=track.netuid,
                            finalized_block=result.finalized_block,
                        )
                        artifacts = build_public_artifacts(
                            store,
                            track,
                            chain,
                            mechid=mechid,
                            version_key=version_key,
                        )
                        if artifacts is not None:
                            await publisher.publish(artifacts)
                            result = EvaluationCycleResult(
                                result.finalized_block,
                                result.scan,
                                result.admission,
                                result.scoring,
                                published_block=artifacts.source_block,
                            )
                    except Exception as exc:
                        result = EvaluationCycleResult(
                            result.finalized_block,
                            result.scan,
                            result.admission,
                            result.scoring,
                            publication_error=str(exc) or type(exc).__name__,
                        )
                if report is not None:
                    report(result)
                if once:
                    return
                await asyncio.sleep(poll_seconds)


async def run_validator(
    *,
    public_weights_url: str,
    netuid: int,
    track_spec_digest: str,
    network: str,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None = None,
    poll_seconds: float = 12.0,
    max_staleness_blocks: int = 100,
    once: bool = False,
    report: Callable[[WeightCycleResult], None] | None = None,
) -> None:
    """Fetch owner-produced weights, validate them, and submit with one Validator wallet."""

    if not network:
        raise ValueError("network cannot be empty")
    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    normalized_track_digest = track_spec_digest.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_track_digest) is None:
        raise ValueError("track specification digest must be 32-byte hexadecimal")
    validate_public_weights_url(public_weights_url)
    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if max_staleness_blocks <= 0:
        raise ValueError("maximum staleness must be positive")
    wallet_arguments = {"name": wallet_name, "hotkey": hotkey_name}
    if wallet_path is not None:
        wallet_arguments["path"] = wallet_path
    wallet = bt.Wallet(**wallet_arguments)
    validator_hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if not isinstance(validator_hotkey, str) or not validator_hotkey:
        raise ValueError("validator wallet does not expose a hotkey address")

    async with (
        bt.Client(network) as client,
        httpx.AsyncClient(follow_redirects=False, timeout=30.0) as http_client,
    ):
        while True:
            try:
                result = await process_weight_cycle(
                    client,
                    http_client,
                    wallet,
                    validator_hotkey=validator_hotkey,
                    public_weights_url=public_weights_url,
                    netuid=netuid,
                    track_spec_digest=normalized_track_digest,
                    max_staleness_blocks=max_staleness_blocks,
                )
            except Exception as exc:
                result = WeightCycleResult(
                    finalized_block=None,
                    source_block=None,
                    status="ERROR",
                    error=str(exc) or type(exc).__name__,
                )
            if report is not None:
                report(result)
            if once:
                if result.error is not None:
                    raise RuntimeError(result.error)
                return
            await asyncio.sleep(poll_seconds)


async def process_weight_cycle(
    client: Any,
    http_client: httpx.AsyncClient,
    wallet: Any,
    *,
    validator_hotkey: str,
    public_weights_url: str,
    netuid: int,
    track_spec_digest: str,
    max_staleness_blocks: int,
) -> WeightCycleResult:
    artifact = await fetch_weight_artifact(public_weights_url, client=http_client)
    chain = await read_weight_chain_state(client, netuid=netuid)
    plan = validate_weight_artifact(
        artifact,
        chain,
        expected_netuid=netuid,
        expected_track_digest=track_spec_digest,
        max_staleness_blocks=max_staleness_blocks,
    )
    if plan is None:
        return WeightCycleResult(chain.finalized_block, artifact.source_block, "NO_WEIGHTS")
    due, reason = weight_update_due(
        artifact,
        chain,
        validator_hotkey=validator_hotkey,
    )
    if not due:
        return WeightCycleResult(chain.finalized_block, artifact.source_block, reason)
    receipt = await submit_weights(
        plan,
        client=client,
        wallet=wallet,
        netuid=artifact.netuid,
        mechid=artifact.mechid,
        version_key=artifact.version_key,
    )
    return WeightCycleResult(
        chain.finalized_block,
        artifact.source_block,
        "SUBMITTED",
        receipt=receipt,
    )


async def process_cycle(
    client: Any,
    store: ChallengeStore,
    *,
    netuid: int,
    track_start_block: int,
    cooldown_blocks: int,
    admission_settings: AdmissionSettings,
    scoring_settings: ScoringSettings,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    worker_id: str,
    max_scan_blocks: int = 100,
    downloader: Callable[..., DownloadedModel] = download_public_model,
    model_validator: Callable[..., ValidatedModel] = validate_model,
) -> EvaluationCycleResult:
    """Synchronize the chain, admit one model, and score one ready stage."""

    if not worker_id:
        raise ValueError("worker id cannot be empty")
    scan = await scan_finalized_blocks(
        client,
        store,
        netuid=netuid,
        track_start_block=track_start_block,
        cooldown_blocks=cooldown_blocks,
        max_blocks=max_scan_blocks,
    )
    finalized_block = await client.finalized_block()
    admission = process_next_admission(
        store,
        worker_id=f"{worker_id}:admission",
        settings=admission_settings,
        current_finalized_block=finalized_block,
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
    return EvaluationCycleResult(finalized_block, scan, admission, scoring)
