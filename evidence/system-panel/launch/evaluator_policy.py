"""Generate or verify the evaluator IAM policy from the runtime storage contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validator.production.commitment import canonical_json
from validator.production.storage import evaluator_iam_policy, verify_evaluator_iam_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--write", type=Path)
    commands.add_argument("--verify", type=Path)
    arguments = parser.parse_args()

    expected = evaluator_iam_policy(bucket=arguments.bucket, prefix=arguments.prefix)
    if arguments.verify is not None:
        try:
            actual = json.loads(arguments.verify.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid evaluator IAM policy: {exc}") from exc
        verify_evaluator_iam_policy(
            actual,
            bucket=arguments.bucket,
            prefix=arguments.prefix,
        )
        return

    assert arguments.write is not None
    payload = canonical_json(expected) + b"\n"
    try:
        with arguments.write.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise SystemExit("refusing to replace an existing policy document") from exc


if __name__ == "__main__":
    main()
