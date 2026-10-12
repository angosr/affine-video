"""Command-line entry point for the three Veda modules."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from miner.chain import ChainSubmissionError, submit_model
from miner.submit import ModelSubmission, SubmissionFormatError
from validator.export import export_snapshot
from validator.run import CycleResult, run_validator
from validator.store import ChallengeStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veda")
    modules = parser.add_subparsers(dest="module", required=True)
    miner = modules.add_parser("miner", help="submit a public model reference")
    miner_commands = miner.add_subparsers(dest="command", required=True)
    submit = miner_commands.add_parser("submit", help="publish a model reference on-chain")
    submit.add_argument("--repo-id", required=True)
    submit.add_argument("--revision", required=True)
    submit.add_argument("--netuid", required=True, type=int)
    submit.add_argument("--network", default="finney")
    submit.add_argument("--wallet-name", default="default")
    submit.add_argument("--hotkey", default="default")
    submit.add_argument("--wallet-path")
    submit.add_argument(
        "--execute",
        action="store_true",
        help="sign and submit; without this flag the command only validates and prints the payload",
    )

    validator = modules.add_parser("validator", help="inspect validator state")
    validator_commands = validator.add_subparsers(dest="command", required=True)
    leaderboard = validator_commands.add_parser("leaderboard")
    leaderboard.add_argument("--database", required=True)
    challenges = validator_commands.add_parser("challenges")
    challenges.add_argument("--database", required=True)
    challenge = validator_commands.add_parser("challenge")
    challenge.add_argument("challenge_id")
    challenge.add_argument("--database", required=True)
    export = validator_commands.add_parser("export")
    export.add_argument("--database", required=True)
    export.add_argument("--output", required=True)
    run = validator_commands.add_parser("run", help="scan and evaluate one frozen track")
    run.add_argument("--track-spec", required=True)
    run.add_argument("--database", required=True)
    run.add_argument("--model-cache", required=True)
    run.add_argument("--screen-cache", required=True)
    run.add_argument("--exact-cache", required=True)
    run.add_argument("--rollout-runner")
    run.add_argument("--rollout-panel")
    run.add_argument("--network", default="finney")
    run.add_argument("--archive-endpoint", dest="archive_endpoints", action="append")
    run.add_argument("--worker-id", default="validator")
    run.add_argument("--poll-seconds", default=12.0, type=float)
    run.add_argument("--max-scan-blocks", default=100, type=int)
    run.add_argument("--scoring-timeout-seconds", default=1800.0, type=float)
    run.add_argument("--rollout-timeout-seconds", default=1500.0, type=float)
    run.add_argument("--scoring-max-memory-bytes", type=int)
    run.add_argument("--scoring-max-cpu-seconds", type=int)
    run.add_argument("--submit-weights", action="store_true")
    run.add_argument("--weight-interval-blocks", default=100, type=int)
    run.add_argument("--wallet-name", default="default")
    run.add_argument("--hotkey", default="default")
    run.add_argument("--wallet-path")
    run.add_argument("--mechid", default=0, type=int)
    run.add_argument("--version-key", default=0, type=int)
    run.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)

    if arguments.module == "validator":
        _run_validator(parser, arguments)
        return

    try:
        submission = ModelSubmission(arguments.repo_id, arguments.revision)
        payload = submission.encode()
        if not arguments.execute:
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
    if arguments.command == "run":
        try:
            asyncio.run(
                run_validator(
                    track_spec=arguments.track_spec,
                    database=arguments.database,
                    model_cache=arguments.model_cache,
                    screen_cache=arguments.screen_cache,
                    exact_cache=arguments.exact_cache,
                    rollout_runner=arguments.rollout_runner,
                    rollout_panel=arguments.rollout_panel,
                    network=arguments.network,
                    archive_endpoints=arguments.archive_endpoints,
                    worker_id=arguments.worker_id,
                    poll_seconds=arguments.poll_seconds,
                    max_scan_blocks=arguments.max_scan_blocks,
                    scoring_timeout_seconds=arguments.scoring_timeout_seconds,
                    rollout_timeout_seconds=arguments.rollout_timeout_seconds,
                    scoring_max_memory_bytes=arguments.scoring_max_memory_bytes,
                    scoring_max_cpu_seconds=arguments.scoring_max_cpu_seconds,
                    submit_chain_weights=arguments.submit_weights,
                    weight_interval_blocks=arguments.weight_interval_blocks,
                    wallet_name=arguments.wallet_name,
                    hotkey_name=arguments.hotkey,
                    wallet_path=arguments.wallet_path,
                    mechid=arguments.mechid,
                    version_key=arguments.version_key,
                    once=arguments.once,
                    report=_print_cycle,
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
            elif arguments.command == "challenge":
                output = store.public_challenge(arguments.challenge_id)
            else:
                path = export_snapshot(store, arguments.output)
                output = {"output": str(path)}
    except (KeyError, RuntimeError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(output, sort_keys=True, ensure_ascii=False))


def _print_cycle(result: CycleResult) -> None:
    output = {
        "finalized_block": result.finalized_block,
        "scan": {
            "first_block": result.scan.first_block,
            "last_block": result.scan.last_block,
            "blocks_scanned": result.scan.blocks_scanned,
            "submissions_found": result.scan.submissions_found,
        },
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
        "weight_run": (
            {
                "run_id": result.weight_run.run_id,
                "source_block": result.weight_run.source_block,
                "state": result.weight_run.state,
                "block_hash": result.weight_run.block_hash,
                "extrinsic_id": result.weight_run.extrinsic_id,
            }
            if result.weight_run is not None
            else None
        ),
        "weight_error": result.weight_error,
    }
    print(json.dumps(output, sort_keys=True), flush=True)
