"""Retired v1 writer: keep its existing UID0 allocation, never admit a battle.

Run only against the separately mounted, frozen e9ab14b legacy source. This is
an operational entrypoint for the existing writer, not a v2 evaluator or a new
public protocol. Remove the service at the explicitly coordinated v2 cutover.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import bittensor as bt

from scoring.spec import load_track_spec
from validator.publication import S3Repository, build_genesis_artifacts, decode_weight_artifact
from validator.run import PUBLIC_STORAGE_BUCKET, PUBLIC_STORAGE_PREFIX
from validator.weights import read_weight_chain_state

LEGACY_DIGEST = "0fcae991aef29f87b97fbb07cf79c49e5eb9cc2a5d9aecdd2a566443e2b81dd5"
LEGACY_TRACK_ID = "veda-sn120-v1"


def verified_track(path):
    track = load_track_spec(path)
    if (track.digest, track.track_id, track.netuid, track.mechid) != (
        LEGACY_DIGEST, LEGACY_TRACK_ID, 120, 1,
    ):
        raise ValueError("UID0-only maintenance requires the unchanged public v1 track")
    return track


async def run_loop(
    track, publisher, *, client_factory, report, poll_seconds=12.0,
    read_only=False, maximum_attempts=None,
):
    attempts = 0
    failures = 0
    while maximum_attempts is None or attempts < maximum_attempts:
        attempts += 1
        try:
            # Recheck the public identity before every write. A later deliberate
            # track switch must not be overwritten by an old maintenance process.
            ledger = await publisher.load(track)
            if ledger.matches:
                raise RuntimeError("UID0-only maintenance refuses a non-empty public v1 ledger")
            async with client_factory() as client:
                finalized = await client.finalized_block()
                chain = await read_weight_chain_state(
                    client, netuid=track.netuid, finalized_block=finalized,
                )
            if chain.uid_by_hotkey.get(track.initial_champion.hotkey) != 0:
                raise RuntimeError("the pinned v1 champion no longer owns UID0")
            artifacts = build_genesis_artifacts(track, chain, version_key=0)
            weights = decode_weight_artifact(artifacts.weights)
            if (
                weights.track_spec_digest != LEGACY_DIGEST or weights.plan is None
                or weights.plan.uids != (0,) or weights.plan.weights != (1.0,)
                or weights.plan.hotkeys != (track.initial_champion.hotkey,)
                or artifacts.matches
            ):
                raise RuntimeError("legacy maintenance produced an unexpected allocation")
            if not read_only:
                await publisher.publish(artifacts)
            failures = 0
            report({
                "event": "V1_UID0_ONLY", "finalized_block": finalized,
                "track_spec_digest": track.digest, "uid": 0, "weight": 1.0,
                "after_original_start": finalized >= track.track_start_block,
                "public_write": not read_only,
            })
        except Exception as exc:
            failures += 1
            report({
                "event": "V1_UID0_RETRY", "attempt": attempts,
                "consecutive_failures": failures, "error_type": type(exc).__name__,
                "error": (str(exc) or type(exc).__name__)[:500], "public_write": False,
            })
            if read_only:
                raise
        if maximum_attempts is None or attempts < maximum_attempts:
            await asyncio.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    import validator.run as legacy_run

    if not Path(legacy_run.__file__).resolve().is_relative_to(Path("/recovery/src")):
        raise RuntimeError("maintenance must use the frozen legacy source mount")
    track = verified_track("/recovery/track.json")
    publisher = S3Repository.from_environment(
        bucket=PUBLIC_STORAGE_BUCKET, prefix=PUBLIC_STORAGE_PREFIX,
    )
    asyncio.run(run_loop(
        track, publisher, client_factory=lambda: bt.Client("finney"),
        report=lambda event: print(json.dumps(event, sort_keys=True), flush=True),
        read_only=args.read_only, maximum_attempts=1 if args.read_only else None,
    ))


if __name__ == "__main__":
    main()
