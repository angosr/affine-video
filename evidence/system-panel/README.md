# Verification and operations

Use the public [track specification](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/track.json)
for the active identity and runtime manifest. A repository checkout is not automatically the
deployed evaluator. Public URLs and the single `weights.json` path stay fixed across releases.

## Published evidence

The [r2 release reports and raw GPU logs](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/system_matches/veda-sn120-v2-r2/evidence/README.md)
retain their original runtime and input identities. Their [inventory](acceptance/inventory.json)
is a release-specific index, not an authorization for newer source.

[`validator/release.py`](../../src/validator/release.py) verifies runtime-bound acceptance for
E2 (numerical fidelity), E3 (two-stage GPU execution), and E16 (determinism and error handling).
Startup also checks runtime integrity, storage and data isolation. Research is not an additional
seventeen-item launch gate; failed cost or overhead studies are not relabeled as passes.
The E3 accelerated chain fixture does not establish real-miner rewards or chain transactions.

## Maintenance tooling

[`operations/preserve_runtime_assets.py`](operations/preserve_runtime_assets.py) preserves an
explicit runtime manifest in a locked, resumable off-GPU mirror. It is an operator tool, not a
service or publication authority. The mirror host needs SSH and `/usr/bin/rsync`. Its regression
tests remain in the normal suite; the real rsync integration case runs only where rsync is installed.

Historical research runners, one-off scripts and their results belong in private development
archives, not the maintained source tree. Removing a retired runner does not alter published
evidence, recorded failures, runtime-bound acceptance, or production tests.

## Operational limits

Private state requires real conditional writes. Public Supabase uses one managed writer,
monotonic checks and readback; this is not distributed atomic publication. Unrevealed batch
payloads currently stay on the GPU: losing bytes permanently after READY can block their original
challenge, so component restore tests do not establish complete host-loss recovery.

Push alerts require a configured HTTPS receiver; logs are not delivery receipts. Fixed scoring
costs do not promise measured GPU latency. Broader ranking, rare-failure coverage, motion-metric
calibration, capacity and the real-miner reward lifecycle need continuing operational validation.
