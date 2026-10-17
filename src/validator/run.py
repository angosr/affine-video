"""Owner evaluation and independent weight-setting loops."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Protocol

import bittensor as bt
import httpx

from scoring.cache import validate_exact_cache
from scoring.model import ValidatedModel, validate_model
from scoring.spec import VEDA_REPO_ID, VEDA_REVISION, TrackSpec, load_track_spec
from validator.admission import (
    AdmissionOutcome,
    AdmissionSettings,
    process_next_admission,
)
from validator.chain import ScanResult, scan_finalized_blocks, seed_legacy_exclusions
from validator.download import DownloadedModel, download_chunked_asset, download_public_model
from validator.publication import (
    PublicArtifacts,
    PublicLedger,
    build_public_artifacts,
    encode_weight_artifact,
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
    verify_initial_champion,
)
from validator.store import ChallengeStore, LeasedChallenge
from validator.system_matches import (
    PRODUCTION_NETUID,
    PRODUCTION_TRACK_ID,
    PRODUCTION_TRACK_SPEC_DIGEST,
    system_match_artifacts,
    system_match_summaries,
)
from validator.tournament import MatchRecord
from validator.weights import (
    WeightArtifact,
    WeightReceipt,
    plan_champion_weight,
    read_weight_chain_state,
    submit_weights,
    validate_weight_artifact,
    weight_update_due,
)

DEFAULT_PUBLIC_WEIGHTS_URL = (
    "https://iblosevwkuiootbbgnsy.supabase.co/storage/v1/object/public/"
    "affine_video/tracks/main/weights.json"
)
DEFAULT_BASELINE_MANIFEST_URL = (
    "https://iblosevwkuiootbbgnsy.supabase.co/storage/v1/object/public/"
    "affine_video/tracks/sn120/assets/veda_baseline_f16/manifest.json"
)
DEFAULT_NETUID = PRODUCTION_NETUID
DEFAULT_MECHID = 1
DEFAULT_TRACK_SPEC_DIGEST = PRODUCTION_TRACK_SPEC_DIGEST
INITIAL_TRACK_ID = PRODUCTION_TRACK_ID
INITIAL_CHAMPION_HOTKEY = "5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8"


class ArtifactRepository(Protocol):
    async def publish(self, artifacts: PublicArtifacts) -> None: ...

    async def load(self, track: TrackSpec) -> PublicLedger: ...

    async def publish_initial_state(
        self,
        *,
        status: bytes,
        weights: bytes,
        system_matches: tuple[tuple[str, bytes], ...] = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationCycleResult:
    finalized_block: int
    scan: ScanResult
    admission: AdmissionOutcome | None
    scoring: ScoringOutcome | None
    restored_matches: int = 0
    published_block: int | None = None
    publication_error: str | None = None


@dataclass(frozen=True, slots=True)
class WeightCycleResult:
    finalized_block: int | None
    source_block: int | None
    status: str
    receipt: WeightReceipt | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _InitialChampion:
    hotkey: str = INITIAL_CHAMPION_HOTKEY
    uid: int = 0


async def run_evaluator(
    *,
    track_spec: str | Path,
    database: str | Path,
    model_cache: str | Path,
    exact_cache: str | Path,
    initial_model: str | Path | None,
    initial_model_manifest_url: str = DEFAULT_BASELINE_MANIFEST_URL,
    network: str,
    archive_endpoints: list[str] | None,
    worker_id: str,
    publisher: ArtifactRepository,
    mechid: int = 0,
    version_key: int = 0,
    poll_seconds: float = 12.0,
    max_scan_blocks: int = 100,
    scoring_timeout_seconds: float = 30 * 60,
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
    track = load_track_spec(track_spec)
    ledger = await publisher.load(track)
    validate_exact_cache(
        exact_cache,
        expected_sha256=track.cache.exact_sha256,
        max_bytes=track.cache.exact_max_bytes,
        min_blocks=track.scoring.budgets[0].blocks,
        max_blocks=track.scoring.budgets[-1].blocks,
        head_dim=track.model.selector.head_dim,
        routes=track.model.selector.routes,
        strata=track.scoring.stratum_weight_map,
        sample_count=track.scoring.sample_count,
    )
    initial_model_path = (
        Path(initial_model)
        if initial_model is not None
        else await asyncio.to_thread(
            download_chunked_asset,
            initial_model_manifest_url,
            cache_dir=Path(model_cache) / "baseline",
            expected_sha256=track.initial_champion.evaluation_file_sha256,
            expected_model_digest=track.initial_champion.model_digest,
            max_bytes=track.model.max_bytes,
        )
    )
    verify_initial_champion(
        track,
        exact_cache=exact_cache,
        model_path=initial_model_path,
    )
    admission_settings = AdmissionSettings(
        cache_dir=Path(model_cache),
        max_model_bytes=track.model.max_bytes,
        tensor_schema=track.model.tensor_schema,
        publication_grace_blocks=track.publication_grace_blocks,
    )
    evaluator = IsolatedEvaluator(
        FileTrackEvaluator(track, str(exact_cache)),
        timeout_seconds=scoring_timeout_seconds,
        max_memory_bytes=scoring_max_memory_bytes,
        max_cpu_seconds=scoring_max_cpu_seconds,
    )
    scoring_settings = ScoringSettings(lease_ms=ceil(evaluator.timeout_seconds * 1000) + 60_000)

    with ChallengeStore(database, track_spec_digest=track.digest) as store:
        async with bt.Client(network, archive_endpoints=archive_endpoints) as client:
            await seed_legacy_exclusions(
                client,
                store,
                netuid=track.netuid,
                before_block=track.track_start_block - 1,
            )
            recovery_index = 0
            while True:
                result = await process_cycle(
                    client,
                    store,
                    netuid=track.netuid,
                    track_start_block=track.track_start_block,
                    admission_settings=admission_settings,
                    scoring_settings=scoring_settings,
                    evaluator=evaluator,
                    worker_id=worker_id,
                    max_scan_blocks=max_scan_blocks,
                    recovery_matches=ledger.matches[recovery_index:],
                    minimum_finalized_block=ledger.required_finalized_block,
                )
                recovery_index += result.restored_matches
                cursor = store.chain_cursor()
                if (
                    cursor is not None
                    and cursor[0] == result.finalized_block
                    and result.finalized_block >= ledger.required_finalized_block
                ):
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
                                restored_matches=result.restored_matches,
                                published_block=artifacts.source_block,
                            )
                    except Exception as exc:
                        result = EvaluationCycleResult(
                            result.finalized_block,
                            result.scan,
                            result.admission,
                            result.scoring,
                            restored_matches=result.restored_matches,
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
    """Fetch evaluator-produced weights, validate them, and submit with one Validator wallet."""

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


async def run_initial_publisher(
    *,
    publisher: ArtifactRepository,
    network: str = "finney",
    poll_seconds: float = 12.0,
    once: bool = False,
    report: Callable[[WeightCycleResult], None] | None = None,
) -> None:
    """Keep the public SN120 state current while UID0 is the only champion."""

    if not network:
        raise ValueError("network cannot be empty")
    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    async with bt.Client(network) as client:
        while True:
            try:
                chain = await read_weight_chain_state(client, netuid=DEFAULT_NETUID)
                plan = plan_champion_weight(
                    _InitialChampion(),
                    uid_by_hotkey=chain.uid_by_hotkey,
                    current_block=chain.finalized_block,
                    min_allowed_weights=chain.min_allowed_weights,
                    max_weight=chain.max_weight,
                )
                if plan is None:
                    raise RuntimeError("UID0 cannot receive the complete chain weight")
                artifact = WeightArtifact(
                    INITIAL_TRACK_ID,
                    DEFAULT_TRACK_SPEC_DIGEST,
                    DEFAULT_NETUID,
                    DEFAULT_MECHID,
                    0,
                    chain.finalized_block,
                    plan,
                )
                status = _initial_status(chain.finalized_block)
                await publisher.publish_initial_state(
                    status=status,
                    weights=encode_weight_artifact(artifact),
                    system_matches=system_match_artifacts(
                        INITIAL_TRACK_ID,
                        DEFAULT_TRACK_SPEC_DIGEST,
                        DEFAULT_NETUID,
                    ),
                )
                result = WeightCycleResult(
                    chain.finalized_block,
                    chain.finalized_block,
                    "PUBLISHED",
                )
            except Exception as exc:
                result = WeightCycleResult(
                    None,
                    None,
                    "ERROR",
                    error=str(exc) or type(exc).__name__,
                )
            if report is not None:
                report(result)
            if once:
                if result.error is not None:
                    raise RuntimeError(result.error)
                return
            await asyncio.sleep(poll_seconds)


def _initial_status(source_block: int) -> bytes:
    champion = {
        "challenge_id": "genesis",
        "hotkey": INITIAL_CHAMPION_HOTKEY,
        "repo_id": VEDA_REPO_ID,
        "revision": VEDA_REVISION,
        "score": None,
        "uid": 0,
        "source": "VEDA_BASELINE",
        "block_height": 0,
        "extrinsic_index": 0,
        "event_index": 0,
    }
    document = {
        "track_id": INITIAL_TRACK_ID,
        "track_spec_digest": DEFAULT_TRACK_SPEC_DIGEST,
        "netuid": DEFAULT_NETUID,
        "source_block": source_block,
        "champion": champion,
        "system_matches": system_match_summaries(
            INITIAL_TRACK_ID,
            DEFAULT_TRACK_SPEC_DIGEST,
            DEFAULT_NETUID,
        ),
        "participants": [
            {
                "uid": 0,
                "hotkey": INITIAL_CHAMPION_HOTKEY,
                "challenge_id": "genesis",
                "submission_block": 0,
                "state": "CHAMPION",
                "queue_position": None,
                "match_sequence": 0,
                "score": None,
                "outcome": "GENESIS",
                "champion": True,
            }
        ],
    }
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


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
    admission_settings: AdmissionSettings,
    scoring_settings: ScoringSettings,
    evaluator: Callable[[LeasedChallenge], StageOutcome],
    worker_id: str,
    max_scan_blocks: int = 100,
    downloader: Callable[..., DownloadedModel] = download_public_model,
    model_validator: Callable[..., ValidatedModel] = validate_model,
    recovery_matches: Sequence[MatchRecord] = (),
    minimum_finalized_block: int = 0,
) -> EvaluationCycleResult:
    """Synchronize the chain, admit one model, and score one ready stage."""

    if not worker_id:
        raise ValueError("worker id cannot be empty")
    if minimum_finalized_block < 0:
        raise ValueError("minimum finalized block cannot be negative")
    scan = await scan_finalized_blocks(
        client,
        store,
        netuid=netuid,
        track_start_block=track_start_block,
        max_blocks=max_scan_blocks,
    )
    finalized_block = await client.finalized_block()
    restored_matches = store.restore_public_matches(recovery_matches)
    if finalized_block < minimum_finalized_block:
        return EvaluationCycleResult(
            finalized_block,
            scan,
            None,
            None,
            restored_matches=restored_matches,
        )
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
    return EvaluationCycleResult(
        finalized_block,
        scan,
        admission,
        scoring,
        restored_matches=restored_matches,
    )
