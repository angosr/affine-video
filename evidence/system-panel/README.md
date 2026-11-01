# Verification and operations

Use the public [track specification](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/track.json)
for the active competition and data identity. Resolve the approved runtime manifest through
`status.json.runtime_release`. A repository checkout is not automatically the deployed
evaluator. Public URLs and the single `weights.json` path stay fixed across releases.

## Published evidence

Follow the competition manifest and the actual runtime manifest for their bound evidence
and runtime acceptance respectively.
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

## Source, live runtime and historical execution

The [changelog](../../CHANGELOG.md) labels source candidates and the last operator-reported
deployed baseline separately. Resolve the current software through `status.json.runtime_release`
and its actual source, installed lock, manifest and acceptance. A checkout, descriptor or
current status cannot establish which environment or GPU executed an earlier match.

Evidence-enabled scoring retains already-calculated cells and observed execution facts in
the original private stage. First-claim evidence references immutable private enumeration
inputs and explicit pagination completion. It is an observed input snapshot, not an atomic
transaction across the plan directory and allocation CAS. Full plan bodies stay private;
the original allocation remains the sole assignment authority. Published audit artifacts
are deterministic hash-bound projections after the original data's disclosure age.

`veda verify-match` performs CPU statistical verification of three original public JSON
files. `veda audit` uses `S3Repository.load`, including its history, match/fence, model and
available disclosure-byte checks. It is GET-only and never acquires a writer lock, opens
SQLite, loads a wallet or changes a pointer. Missing historical evidence remains unknown.
The candidate implementation has CPU coverage; it is not a report of deployed evidence or
new GPU measurements. Private loser payloads remain private.

## Protocol and rules proposals awaiting explicit decisions

The following are design boundaries, not implemented changes or release prerequisites.
The active confidence remains 0.9975, promotion uses one exact batch, and `track_id` remains
`veda-sn120-v2-r10`. Infrastructure time limits are diagnostic/recovery deadlines: they do
not turn `SYSTEM_BLOCKED` into a miner loss or reset its submission, claim or opportunity.

| Proposal | Existing interface and proposed boundary | History and data impact |
| --- | --- | --- |
| External pre-seed commitment | `PlanCoordinator.create_next`, `BatchPlan` and disclosure verification currently bind private plan facts. To prove commitment before an unpredictable seed, an independently verifiable finalized commitment must fix the nonce/root and a future block before that block occurs. An operator-owned S3 timestamp alone is insufficient. The exact chain payload, finality ordering and authorized signer require a separate decision. | This changes generation provenance and its data contract. Existing READY bytes, original commitments and clocks cannot be re-labelled as externally precommitted. No new transaction is authorized here. |
| HF file commitment, first digest owner and HTTP buffer | `ModelSubmission.encode/decode` preserves VDA1 repo/revision bytes; its current competition maximum is 121 bytes. Appending a file hash changes encoding and can exceed that bound. A versioned payload/manifest reference and its byte budget must be chosen before editing the chain parser or admission binding. Canonical ownership must be validated in original S order; an unverified declared digest cannot reserve a model. A first-fetch interval needs an original-S clock and a defined distinction between provider failure and participant unavailability. | Old VDA1 records must decode byte-for-byte without invented hashes. VDR1 and the private route remain unchanged. No new wall-clock deadline may silently reassign ownership, discard a deferred task or consume a new opportunity. |
| Confidence 0.975 and a second test | `ScoringSpec`, `PairedTrackEvaluator`, `Inventory` and tournament validation need an explicit choice of second exact versus full-video measurement and the combined promotion rule. The existing historical confirmation allocation checks opposite slots and disjoint families. Both batches must already satisfy the original S boundary; a later generated batch cannot repair that eligibility. | A rules change needs its own accepted identity and statistical justification. Existing outcomes are not recomputed. Current exact/full scientific implementations are not evidence that a proposed two-test policy has been approved or measured. |
| Source-pool exclusions | Define exclusion criteria and provenance before changing the bound `SourcePool`, source-cache binding, Merkle root and `TrackSpec.source_commitments`. Publishing existing hotkey exclusion reasons is separate from changing dataset exclusions. | A changed source pool is a data-contract migration. Old source proofs, plans, READY bytes and C/R cannot be rewritten. Treatment of already-committed data must be decided explicitly. |

`S3Repository._recover` currently rejects a different specification under an active track ID.
Future rule activation must therefore specify a verified maintenance migration and continuous
immutable match numbering while preserving the fixed ID, prior policy/outcomes, S, claims,
champion tenure and opportunities. This task does not relax the digest checks, restart a
track or add another ledger. Compatible software maintenance keeps the existing competition
and data identity. Hardware flags and production-side dense checks remain recorded; current
GPU recomputation of an old authenticated cache is not restored as a consumer gate.

## Maintenance tooling

[`operations/preserve_runtime_assets.py`](operations/preserve_runtime_assets.py) preserves an
explicit runtime manifest in a locked, resumable off-GPU mirror. It is an operator tool, not a
service or publication authority. The mirror host needs SSH and `/usr/bin/rsync`. Its regression
tests remain in the normal suite; the real rsync integration case runs only where rsync is installed.

Historical research runners, one-off scripts and their results belong in private development
archives, not the maintained source tree. Removing a retired runner does not alter published
evidence, recorded failures, runtime-bound acceptance, or production tests.

### Native deployment preparation

Use [`operations/prepare_native_deployment.py`](operations/prepare_native_deployment.py) with
one operator-owned JSON configuration for each target. Do not copy a script for each address or
release. This configuration contains machine paths, pinned file references and Unix roles only;
it cannot override GPUs, capacity, scoring, track/data identity, start, history, DB or work.
Those facts come from the referenced `production.json`, specifications and manifests. Keep the
configuration outside Git; it must not contain secret values or a business-state snapshot.

```json
{
  "source_ssh": {
    "host": "source.example", "port": 22, "user": "root",
    "identity_file": "/operator/source-key", "known_hosts": "/operator/known-hosts"
  },
  "target_ssh": {
    "host": "target.example", "port": 22, "user": "root",
    "identity_file": "/operator/target-key", "known_hosts": "/operator/known-hosts"
  },
  "target_hostname": "expected-target-hostname",
  "runtime": {
    "python": "/opt/frozen/usr/bin/python3.12", "home": "/opt/frozen/usr",
    "site_packages": "/opt/frozen/usr/local/lib/python3.12/dist-packages"
  },
  "production": {"path": "/release/final/production.json", "sha256": "<full SHA-256>"},
  "accepted": {"path": "/release/final/accepted.json", "sha256": "<full SHA-256>"},
  "owner_program": {"path": "/release/control-owner.py", "sha256": "<full SHA-256>"},
  "roles": {
    "owner": {"uid": 0, "gid": 0, "secret_files": ["/run/veda-owner/control.env"]},
    "producer": {"uid": 2001, "gid": 2001, "secret_files": ["/run/veda-producer/role.env"]},
    "evaluator": {"uid": 2002, "gid": 2002, "secret_files": ["/run/veda-evaluator/role.env"]}
  }
}
```

Replace every example path, ID and digest with the actual prepared references. List **all**
role-readable secret files, including the owner's signing seed and worker credential-process
files; do not share files between Unix roles. Source SSH references describe the existing direct
source-to-target transfer; this preparer never connects to the source, opens a new transfer,
mints credentials or changes an in-flight transfer. Development hosts carry only orchestration
and small evidence, not model/data payloads or production control.

The deployment workflow is **inspect → stage → verify → prepare → switch → observe**.
The current entry implements only the first four, without starting any business process:

- `inspect` reads pinned configuration and the original data binding on the expected target.
- `stage` checks already-delivered required files and sizes; it does not claim SHA verification.
- `verify` uses the original full-byte runtime/asset checks, source-pool binding and capacity,
  competition/data binding and `ReleaseApproval`. The OCI descriptor records provenance, not a
  native Docker Engine check. No source, kernel, evidence or schema guard is bypassed.
- `prepare` includes those checks once and checks role-owned 0600 secret-file metadata and the
  pinned owner program without reading secret values or executing that program.

Normally run `python operations/prepare_native_deployment.py --config /operator/deployment.json
prepare` from this directory. Use the earlier phases separately for diagnosis, not as a reason
to repeat all verification. Each invocation inspects real files; there is no resume ledger or
cross-process stat/hash cache. Missing or changed bytes stop the current step, without rebuilding,
reproducing data, changing track or automatically retrying. An existing valid byte is reused,
not recopied. The tool never creates or opens SQLite or rewrites READY/C/R/claims.

`prepare` deliberately reports `ready_to_switch=false`: the production owner still must enforce
the fixed native launcher, worker UID/GID/group and FD/environment isolation, original writer
locks and whole-tree cleanup on owner death. A native pipe cannot reattach to an orphan writer.
Only an explicitly authorized switch may stop the old owner, establish the final DB/cursor/S/
claim and active-upload fence, and start the unique new owner. Never replace the latest database
or public cursor with an earlier snapshot when resuming or rolling back. Observe the real runtime
identity, cursor and original claims/READY commitments after switching. Existing authenticated
recover/history collection and publication fences supply history and champion facts; do not
hardcode a match count, winner, new start or data namespace into deployment scripts.

## Operational limits

Private state requires real conditional writes. Public Supabase uses one managed writer,
monotonic checks and readback; this is not distributed atomic publication. Unrevealed batch
payloads currently stay on the GPU: losing bytes permanently after READY can block their original
challenge, so component restore tests do not establish complete host-loss recovery.

Push alerts require a configured HTTPS receiver; logs are not delivery receipts. Fixed scoring
costs do not promise measured GPU latency. Broader ranking, rare-failure coverage, motion-metric
calibration, capacity and the real-miner reward lifecycle need continuing operational validation.
