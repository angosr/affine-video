"""Command-line entry point for the three Veda modules."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from miner.chain import ChainSubmissionError, submit_model
from miner.prepare import prepare_model
from miner.submit import ModelSubmission, SubmissionFormatError
from scoring.spec import load_track_spec
from validator.monitor import fetch_status, render_status
from validator.publication import S3Repository
from validator.retries import retry_stage
from validator.run import (
    DEFAULT_BASELINE_MANIFEST_URL,
    DEFAULT_NETUID,
    DEFAULT_PUBLIC_WEIGHTS_URL,
    PUBLIC_STORAGE_BUCKET,
    PUBLIC_STORAGE_PREFIX,
    EvaluationCycleResult,
    WeightCycleResult,
    run_evaluator,
    run_validator,
)
from validator.store import ChallengeStore

DEFAULT_PUBLIC_STATUS_URL = DEFAULT_PUBLIC_WEIGHTS_URL.removesuffix("weights.json") + "status.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veda")
    modules = parser.add_subparsers(dest="module", required=True)

    miner = modules.add_parser("miner", help="commit a pinned model reference")
    miner_commands = miner.add_subparsers(dest="command", required=True)
    commit = miner_commands.add_parser(
        "commit",
        help="sign and finalize the hotkey's one allowed model commitment",
    )
    commit.add_argument("--repo-id", required=True)
    commit.add_argument("--revision", required=True)
    commit.add_argument("--netuid", default=DEFAULT_NETUID, type=int)
    commit.add_argument("--network", default="finney")
    commit.add_argument("--wallet-name", default="default")
    commit.add_argument("--hotkey", default="default")
    commit.add_argument("--wallet-path")
    commit.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the payload without opening a chain connection",
    )
    prepare = miner_commands.add_parser(
        "prepare", help="bind selector.safetensors metadata to a Miner hotkey"
    )
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--hotkey-address", required=True)

    validator = modules.add_parser("validator", help="evaluate models or set chain weights")
    validator_commands = validator.add_subparsers(dest="command", required=True)
    produce = validator_commands.add_parser("produce", help="produce private evaluation batches")
    produce.add_argument("--config", required=True)
    produce.add_argument("--once", action="store_true")
    leaderboard = validator_commands.add_parser("leaderboard")
    leaderboard.add_argument("--database", required=True)
    challenges = validator_commands.add_parser("challenges")
    challenges.add_argument("--database", required=True)
    challenge = validator_commands.add_parser("challenge")
    challenge.add_argument("challenge_id")
    challenge.add_argument("--database", required=True)
    retry = validator_commands.add_parser(
        "retry", help="resume one challenge stopped after infrastructure retries"
    )
    retry.add_argument("challenge_id")
    retry.add_argument("--database", required=True)
    retry.add_argument("--production-config", required=True)
    monitor = validator_commands.add_parser("monitor", help="watch the public ranking and queue")
    monitor.add_argument("--status-url", default=DEFAULT_PUBLIC_STATUS_URL)
    monitor.add_argument(
        "--watch",
        metavar="SECONDS",
        type=float,
        help="refresh continuously at this positive interval",
    )

    evaluate = validator_commands.add_parser(
        "evaluate", help="evaluate submissions and publish public artifacts"
    )
    evaluate.add_argument("--track-spec", required=True)
    evaluate.add_argument("--database", required=True)
    evaluate.add_argument("--model-cache", required=True)
    evaluate.add_argument("--exact-cache", required=True)
    evaluate.add_argument("--production-config", required=True)
    evaluate.add_argument("--initial-model")
    evaluate.add_argument(
        "--initial-model-manifest",
        default=DEFAULT_BASELINE_MANIFEST_URL,
    )
    evaluate.add_argument("--network", default="finney")
    evaluate.add_argument("--archive-endpoint", dest="archive_endpoints", action="append")
    evaluate.add_argument("--worker-id", default="evaluator")
    evaluate.add_argument("--poll-seconds", default=12.0, type=float)
    evaluate.add_argument("--max-scan-blocks", default=100, type=int)
    evaluate.add_argument("--scoring-timeout-seconds", default=14400.0, type=float)
    evaluate.add_argument("--scoring-max-memory-bytes", type=int)
    evaluate.add_argument("--scoring-max-cpu-seconds", type=int)
    evaluate.add_argument(
        "--scoring-device",
        choices=("cuda:0", "cuda:1"),
        default="cuda:0",
        help="authoritative CUDA device, for example cuda:0",
    )
    evaluate.add_argument("--version-key", default=0, type=int)
    evaluate.add_argument(
        "--alert-webhook-url",
        default=os.environ.get("VEDA_ALERT_WEBHOOK_URL"),
        help="HTTPS endpoint for retry-exhaustion and evaluator-failure alerts",
    )
    evaluate.add_argument("--once", action="store_true")

    run = validator_commands.add_parser("run", help="fetch public weights and submit them on-chain")
    run.add_argument("--public-weights-url", default=DEFAULT_PUBLIC_WEIGHTS_URL)
    run.add_argument("--network", default="finney")
    run.add_argument("--wallet-name", default="default")
    run.add_argument("--hotkey", default="default")
    run.add_argument("--wallet-path")
    run.add_argument("--poll-seconds", default=12.0, type=float)
    run.add_argument("--max-staleness-blocks", default=100, type=int)
    run.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.module == "validator":
        _run_validator(parser, arguments)
        return
    _run_miner(parser, arguments)


def _run_miner(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    try:
        if arguments.command == "prepare":
            prepared = prepare_model(
                arguments.model,
                arguments.output,
                hotkey=arguments.hotkey_address,
            )
            print(
                json.dumps(
                    {
                        "path": str(prepared.path),
                        "sha256": prepared.sha256,
                        "size_bytes": prepared.size_bytes,
                    },
                    sort_keys=True,
                )
            )
            return
        if arguments.netuid < 0 or arguments.netuid > 0xFFFF:
            raise ValueError("netuid must fit in an unsigned 16-bit integer")
        if not arguments.network:
            raise ValueError("network cannot be empty")
        submission = ModelSubmission(arguments.repo_id, arguments.revision)
        payload = submission.encode()
        if arguments.dry_run:
            output = {
                "hotkey": arguments.hotkey,
                "network": arguments.network,
                "netuid": arguments.netuid,
                "payload_bytes": len(payload),
                "payload_hex": payload.hex(),
                "repo_id": submission.repo_id,
                "revision": submission.revision,
            }
        else:
            receipt = asyncio.run(
                submit_model(
                    submission,
                    netuid=arguments.netuid,
                    network=arguments.network,
                    wallet_name=arguments.wallet_name,
                    hotkey_name=arguments.hotkey,
                    wallet_path=arguments.wallet_path,
                )
            )
            output = {
                "block_hash": receipt.block_hash,
                "extrinsic_id": receipt.extrinsic_id,
                "hotkey": arguments.hotkey,
                "message": receipt.message,
                "network": arguments.network,
                "netuid": arguments.netuid,
                "repo_id": submission.repo_id,
                "revision": submission.revision,
            }
    except (SubmissionFormatError, ChainSubmissionError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(output, sort_keys=True))


def _run_validator(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.command == "produce":
        try:
            from validator.production.service import run_producer

            run_producer(arguments.config, once=arguments.once)
        except KeyboardInterrupt:
            return
        except (RuntimeError, OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        return

    if arguments.command == "retry":
        try:
            from validator.production.config import load_producer_config
            from validator.production.retries import PrivateRetryStore
            from validator.production.storage import PrivateObjectStore

            config = load_producer_config(arguments.production_config)
            if config.validator_database.resolve() != Path(arguments.database).resolve():
                raise ValueError("production configuration uses a different validator database")
            track = load_track_spec(config.track_spec)
            with ChallengeStore(arguments.database, track_spec_digest=track.digest) as store:
                stage = store.blocked_resume_state(arguments.challenge_id)
                objects = PrivateObjectStore.from_environment(
                    bucket=config.storage_bucket,
                    prefix=f"{config.storage_prefix}/tracks/{track.digest}",
                )
                objects.verify_conditional_writes()
                retries = PrivateRetryStore(objects, track_spec_digest=track.digest)
                retries.reset(arguments.challenge_id, retry_stage(stage))
                state = store.resume_blocked(arguments.challenge_id)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        print(
            json.dumps(
                {"challenge_id": arguments.challenge_id, "state": state},
                sort_keys=True,
            )
        )
        return

    if arguments.command == "monitor":
        try:
            asyncio.run(_run_monitor(arguments.status_url, arguments.watch))
        except KeyboardInterrupt:
            return
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        return

    if arguments.command == "evaluate":
        try:
            publisher = S3Repository.from_environment(
                bucket=PUBLIC_STORAGE_BUCKET,
                prefix=PUBLIC_STORAGE_PREFIX,
            )
            asyncio.run(
                run_evaluator(
                    track_spec=arguments.track_spec,
                    database=arguments.database,
                    model_cache=arguments.model_cache,
                    exact_cache=arguments.exact_cache,
                    production_config=arguments.production_config,
                    initial_model=arguments.initial_model,
                    initial_model_manifest_url=arguments.initial_model_manifest,
                    network=arguments.network,
                    archive_endpoints=arguments.archive_endpoints,
                    worker_id=arguments.worker_id,
                    publisher=publisher,
                    version_key=arguments.version_key,
                    poll_seconds=arguments.poll_seconds,
                    max_scan_blocks=arguments.max_scan_blocks,
                    scoring_timeout_seconds=arguments.scoring_timeout_seconds,
                    scoring_max_memory_bytes=arguments.scoring_max_memory_bytes,
                    scoring_max_cpu_seconds=arguments.scoring_max_cpu_seconds,
                    scoring_device=arguments.scoring_device,
                    alert_webhook_url=arguments.alert_webhook_url,
                    once=arguments.once,
                    report=_print_evaluation_cycle,
                )
            )
        except KeyboardInterrupt:
            return
        except (RuntimeError, OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        return

    if arguments.command == "run":
        try:
            asyncio.run(
                run_validator(
                    public_weights_url=arguments.public_weights_url,
                    network=arguments.network,
                    wallet_name=arguments.wallet_name,
                    hotkey_name=arguments.hotkey,
                    wallet_path=arguments.wallet_path,
                    poll_seconds=arguments.poll_seconds,
                    max_staleness_blocks=arguments.max_staleness_blocks,
                    once=arguments.once,
                    report=_print_weight_cycle,
                )
            )
        except KeyboardInterrupt:
            return
        except (RuntimeError, OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        return

    database = Path(arguments.database)
    if not database.is_file():
        parser.exit(1, f"error: validator database does not exist: {database}\n")
    try:
        with ChallengeStore(database) as store:
            if arguments.command == "leaderboard":
                output = store.public_snapshot()["leaderboard"]
            elif arguments.command == "challenges":
                output = store.public_snapshot()["challenges"]
            else:
                output = store.public_challenge(arguments.challenge_id)
    except (KeyError, RuntimeError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(output, sort_keys=True, ensure_ascii=False))


async def _run_monitor(url: str, interval: float | None) -> None:
    if interval is not None and interval <= 0:
        raise ValueError("watch interval must be positive")
    import httpx

    async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
        while True:
            status = await fetch_status(url, client=client)
            if interval is not None and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render_status(status), flush=True)
            if interval is None:
                return
            await asyncio.sleep(interval)


def _print_evaluation_cycle(result: EvaluationCycleResult) -> None:
    output = {
        "finalized_block": result.finalized_block,
        "scan": (
            {
                "first_block": result.scan.first_block,
                "last_block": result.scan.last_block,
                "blocks_scanned": result.scan.blocks_scanned,
                "submissions_found": result.scan.submissions_found,
            }
            if result.scan is not None
            else None
        ),
        "admission": (
            {
                "challenge_id": result.admission.challenge_id,
                "status": result.admission.status,
                "code": result.admission.code,
                "duplicate_of": result.admission.duplicate_of,
            }
            if result.admission is not None
            else None
        ),
        "scoring": (
            {
                "challenge_id": result.scoring.challenge_id,
                "state": result.scoring.state,
                "code": result.scoring.code,
            }
            if result.scoring is not None
            else None
        ),
        "restored_matches": result.restored_matches,
        "published_block": result.published_block,
        "publication_error": result.publication_error,
        "cycle_error": result.cycle_error,
    }
    print(json.dumps(output, sort_keys=True), flush=True)


def _print_weight_cycle(result: WeightCycleResult) -> None:
    output = {
        "finalized_block": result.finalized_block,
        "source_block": result.source_block,
        "status": result.status,
        "error": result.error,
        "block_hash": result.receipt.block_hash if result.receipt is not None else None,
        "extrinsic_id": result.receipt.extrinsic_id if result.receipt is not None else None,
    }
    print(json.dumps(output, sort_keys=True), flush=True)
