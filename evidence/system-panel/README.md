# Verification and operations

Use the public [track specification](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/track.json)
for the active identity and runtime manifest. A repository checkout is not automatically the
deployed evaluator. Public URLs and the single `weights.json` path stay fixed across releases.

## Published evidence

Follow the active track's artifact manifest for its bound acceptance and raw evidence.
Each report retains its original runtime, hardware and input identities; a historical result
does not authorize newer source or establish a rerun on different hardware.
The checked-in [research inventory](acceptance/inventory.json) is a historical index,
not the active release's acceptance or an additional launch checklist.

[`validator/release.py`](../../src/validator/release.py) verifies runtime-bound acceptance and
compatibility. E2 (numerical fidelity), E3 (historical two-stage GPU execution), and E16
(determinism and error handling) retain their original frozen identities; E3 does not enable
full-video confirmation in the current exact-only track. Startup also checks runtime integrity,
storage and data isolation. Research studies are separate from these release requirements.
Component tests and accelerated chain fixtures do not establish real-miner rewards, chain
transactions or complete host-loss recovery.

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
