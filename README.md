# Veda Subnet

Veda Subnet is a Bittensor competition for training sparse-attention selectors. It has three code
boundaries:

- `miner`: validates and submits a pinned Hugging Face model reference;
- `validator`: evaluates submissions, publishes the public tournament ledger, or independently
  applies public weights on-chain;
- `scoring`: validates selector weights and computes one deterministic authoritative score.

There is one evaluator process. It has object-storage write credentials but no Validator wallet.
Every Validator, including the one operated alongside the evaluator, runs the same separate
lightweight process: fetch `weights.json`, validate it against finalized chain state, and sign its
own `SetWeights` transaction. No service or direct Miner-to-Validator protocol exists.

## Competition flow

1. A Miner commits a Hugging Face `repo_id` and a complete 40-character commit SHA on-chain.
   Only that hotkey's first submission on the track is eligible; every later submission is
   terminally rejected, even when the first submission is malformed or fails evaluation.
   The evaluator verifies the signed old-contest activation history before the track starts.
   A valid signed old-contest activation permanently excludes that hotkey from Veda rewards,
   including an activation after its Veda submission. Historical matches and models remain
   unchanged; an excluded champion's weight falls back to UID0. Unrelated commitments do not
   change the selected Veda submission. Old-contest activations remain monitored after closing.
2. After the fixed publication grace period, the evaluator processes submissions by finalized
   `(block, extrinsic, event)` order. Before any model I/O, each admission-eligible submission
   atomically reserves one of `maximum_effective_challenges` season slots. Deterministic
   admission failures release the slot while consuming that hotkey's only submission.
   Infrastructure retries retain the reservation; every valid model entering exact scoring
   consumes its slot regardless of outcome. Later submissions are
   rejected before download when the frozen limit is full. The pinned revision must be anonymously
   downloadable.
3. The evaluator accepts exactly one `selector.safetensors` file. Its only metadata entry must be
   `hotkey`, set to the complete submitting hotkey. It rejects executable code, other metadata,
   unknown tensors, adapters, base-model patches, invalid shapes, and excess bytes.
4. A valid model waits for its scheduled exact-attention batch boundary. The finalized block hash
   at that boundary selects new private sources, noise, routes, steps and query tiles. It is not
   known when the model is committed. Each batch contains four independent videos in each of four
   strata, eight routes and three denoising steps: 384 paired cases in total. A later batch cannot
   be substituted to obtain a more favorable result. Submission admission closes
   `maximum_effective_challenges` generation intervals before the final batch, plus the hidden-data
   delay, so a worst-case FIFO queue retains one later confirmation boundary per eligible Miner.
5. UID 0 is the genesis champion without making a commit. Its identity is pinned to the official
   VEDA repository revision, filename and file SHA-256 in the track specification. Its baseline
   and every challenger are evaluated on the same new batch. The exact stage resamples complete
   video clusters rather than treating correlated routes as independent. A challenger that clears
   this gate receives a second, newly seeded batch and runs complete eight-step video inference at
   all three budgets beside the champion on GPUs 0 and 1. It replaces the champion only when the
   paired per-video lower confidence bound exceeds the frozen full-video gain threshold.
6. An immutable match record is published for every terminal item in the contiguous chain-order
   prefix. A failed or unfinished earlier submission can never be skipped to publish a later win.

Scoring covers several frozen block budgets. The authoritative score combines quality and
calibrated attention cost, so a selector must preserve quality while selecting fewer blocks. The
track specification pins every score-affecting rule and the cache-generation delay. Each match
pins the exact batch hash and private-source digest used for its decision. The evaluator covers
every layer and head derived from the fixed architecture. Kernel-required tiles are stored in the
hidden cache, always retained, and counted inside each block budget before predictor Top-k fills
the rest.

The current champion receives the full subnet weight. Before any successful challenge this is UID
0; after a valid win it is the winning Miner's hotkey. The evaluator checks the current finalized
hotkey-to-UID mapping and chain weight constraints before publishing. If a winning hotkey is no
longer registered, weight falls back to the registered VEDA baseline hotkey instead of leaving a
stale UID allocation behind. When neither hotkey is registered or the chain cannot accept a
single-recipient vector, `weights` is `null` and Validators keep their last finalized weights.

A selector that produces non-finite logits or sparse latents receives zero quality for the affected
case. This rule is applied independently to the challenger and incumbent, so a pathological model
cannot turn its own numerical failure into an infrastructure retry that blocks the FIFO queue.

Initial launch requires real H200 scorer/runtime parity, fixed self and degraded-model controls,
a two-stage FIFO challenge, measured budget costs, data isolation and durable recovery. Immutable
control summaries are published under `system_matches/` without affecting rewards. Broader
ten-batch stability, near-threshold power, proxy-ranking and human-quality studies continue after
launch; pending research is not marked passed. Bootstrap alpha is a calibration target, not a
proved finite-sample guarantee. Unit tests and intermediate GPU checks alone do not establish
release acceptance.

## Public object-storage state

The public contract uses one fixed prefix:

```text
tracks/main/
  track.json
  summary.json
  status.json
  weights.json
  runtime/
    <artifact_manifest_sha256>/manifest.json
  system_matches/
    <track_id>/
      000000000001.json
      ...
  events/
    <track_id>/track-aborted.json
  matches/
    <track_id>/
      000000000001.json
      000000000002.json
      ...
  batches/
    <sha256>/
      manifest.json
      part-00000
      ...
  batch_groups/
    <plan_digest>.json
  batch_assets/
    <sha256>/
      manifest.json
      part-00000
      ...
  batch_inputs/
    <sha256>/
      manifest.json
      part-00000
      ...
  models/
    <track_id>/<sha256>/
      manifest.json
      part-00000
      ...
```

- `track.json` is the stable pointer to the current canonical track specification and digest. An
  active `track_id` has one fixed specification; an upgrade uses a new `track_id`, while the URL
  remains unchanged.
- `system_matches/*.json` are immutable launch checks produced from real GPU panels. They compare
  the VEDA baseline with fixed control models, but never enter the Miner queue, tournament rank, or
  chain weights. A panel is attached only when its track contract, score-affecting source bytes,
  third-party environment and committed model/source assets match; rebuilding the same release
  packaging does not invalidate it. Panel-only selectors, runner bytes and raw results live under
  the manifest's `evidence/system-panel/` namespace. They remain content-addressed and recoverable
  but are excluded from the non-circular runtime digest because production scoring never reads them.
- `events/<track_id>/track-aborted.json` is the immutable stop fact created when system work remains
  unfinished after the close grace block. It survives loss of local state and prevents that track
  from resuming evaluation or weight refresh.
- `runtime/<artifact_manifest_sha256>/manifest.json` publishes the canonical hashes and sizes of
  every generator byte committed by the track, without exposing future private source samples.
- `matches/<track_id>/*.json` are immutable terminal match facts in chain order: public model reference,
  content hashes, aggregate result, incumbent, decision, and decision block.
- `batch_groups/<plan_digest>.json` reveals the finalized-block seed, source proofs, routes, samples,
  steps and immutable object mapping only after every linked challenge is terminal.
  `batches/<sha256>/manifest.json` publishes its exact QKV batch, while `batch_assets/` publishes
  every referenced source video and dense latent. `batch_inputs/` publishes the selected encoded
  input cache; its public mapping is derived from the same READY references bound into the exact
  batch. Each manifest verifies every bounded part and the reassembled file digest.
- `models/<track_id>/<sha256>/manifest.json` is the public durable copy of every selector that
  completes scoring, including challengers that do not replace the champion. A private copy of
  every validated selector is stored before scoring, so publication, independent replay and later
  matches remain recoverable even if the GPU host or original Hugging Face repository disappears.
- `summary.json` is a bounded current view containing the champion, top leaderboard and committed
  match count.
- `status.json` is the bounded live view used by the monitoring CLI; it includes UID 0, active
  entrants, queue positions, match numbers, scores, outcomes and track lifecycle state.
- `weights.json` is the small mutable artifact consumed by Validators.

Publication writes the immutable runtime manifest, current track, immutable batch/model files and
matches first, then the self-contained `weights.json` pointer, followed by `summary.json` and
`status.json`. A crash may leave display pointers behind a valid weight pointer, but cannot expose
weights before their immutable match facts. On restart the evaluator validates every committed
match, follows any contiguous orphaned matches, and restores them instead of evaluating them again.
Conflicting immutable bytes stop publication. A chain node behind the recovered public watermark
may keep scanning, but evaluation and publication wait until its finalized height catches up, so
public state cannot move backward.
The weight pointer binds the terminal match sequence and payload SHA-256, so a higher-block
publisher with a shorter or conflicting match prefix is fenced before it can change champion weights.
Mutable pointers use ETag conditional writes, and production startup probes both conditional create
and conditional replace. A backend that cannot fence two concurrent publishers is rejected.

Every new release to `tracks/main` requires an explicitly coordinated full track digest in
`VEDA_COORDINATED_RELEASE_DIGEST` and the matching runtime manifest. A missing approval or a
mismatched identity rejects publication before any write. Research experiments remain tracked in
the evidence inventory, with incomplete work explicitly marked as such; completing the entire
panel is not a startup requirement. The existing public v1 identity alone retains its established
publication behavior.

Initial operation can generate batches on demand with `veda validator produce --config ... --once`.
Continuous generation and broader quality studies can be enabled after the evaluator is operating;
there is no second data format or publication path.


Object storage is the durable source for public tournament facts. The local SQLite database and
downloaded model cache are disposable execution state: after loss, finalized chain submissions plus
verified published matches reconstruct completed work, and remaining submissions can be evaluated
again.

Private S3-compatible object storage preserves unrevealed plans, the committed source pool and
artifact manifest, validated selector archives, admission refusals, scoring stages and retry state.
Both production services verify conditional-write support for these authoritative facts before
starting. Fixed runtime assets are verified against the manifest and recovered from pinned upstream
revisions or verified off-host copies; duplicating every recoverable asset is not a startup gate.

Unrevealed videos, exact QKV, dense latents and selected per-batch inputs stay on the GPU in one
content-addressed directory shared by producer and evaluator. They are not mirrored to private S3
before scoring. After GPU loss, resume generation from the durable plan or record an interrupted
attempt and reevaluate unfinished work on verifiable replacement data. Never reuse an old hash for
different bytes, overwrite completed stage facts or published results, consume a Miner's chance,
or change FIFO order because local data was lost. Upload the actual used data through the existing
public disclosure paths only after its entire challenge group is terminal.

The manifest must cover a restorable OCI image, the exact producer/scorer source bundle and its
dependency lock. The source bundle is a ZIP whose `cli.py`, `miner/`, `scoring/`, `validator/` and
`miowtion/` entries must byte-match the modules actually loaded by the process. The canonical
dependency lock binds the Python executable and every installed distribution's metadata and current
files. Startup re-hashes both, so an unchanged archive cannot authorize changed ambient code or
packages. Release assembly calls `build_source_bundle()` and `write_dependency_lock()` from
`validator.production.environment`; both refuse overwrite, so verifier and builder share one
format. A canonical source-cache binding joins the source-pool Merkle root to the complete encoded
SampleCache index, text tensors and optional condition tensors. Producer startup verifies that
binding and exact source-ID coverage, so changing encoded input under an existing source ID is a
hard failure rather than a new hidden dataset with the old commitment. Each private source leaf
also contains an independent 256-bit nonce. The track publishes only its salted leaf digest,
partition and geometry. Batch selection ranks those public digests with the finalized-block seed;
after disclosure, the nonce, leaf and Merkle path prove both pool membership and that the selected
source was the deterministic unused candidate, without exposing future prompts or assets.

`Dockerfile.runtime` builds the restorable GPU runtime from a digest-pinned PyTorch/CUDA base and
pinned Miowtion/FlashAttention revisions. Build the source bundle and dependency lock inside that
image, then save the image itself as the OCI artifact covered by the same manifest.

Before a batch's complete challenge group becomes terminal, public artifacts exclude its hidden
arrays, per-sample metrics, local paths, worker and lease data, retry timing, error text, stack
traces, and credentials. Only stable error codes and retry counts are public. Use a public HTTPS
custom domain for reads; mutable objects are published with
`Cache-Control: no-store`, while immutable objects receive long-lived immutable caching.

## Commands

Install and test:

```bash
uv sync
uv run ruff check src tests
uv run pytest
```

Commit a Miner model reference with the wallet hotkey and wait for finalization:

```bash
uv run veda miner commit \
  --repo-id owner/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --wallet-name default \
  --hotkey default
```

Before uploading, bind the only allowed metadata field to the submitting hotkey:

```bash
uv run veda miner prepare \
  --model trained.safetensors \
  --output selector.safetensors \
  --hotkey-address 5...
```

The ownership metadata is checked separately from the canonical tensor digest. Consequently,
copying public weights and changing their hotkey still produces the same duplicate fingerprint.

The production netuid defaults to `120` and the network defaults to `finney`; both can be overridden.
Because only the hotkey's first commitment is eligible, the command sends exactly one transaction
and never retries it. Add `--dry-run` to validate the fields and inspect the encoded payload without
opening a chain connection. The model revision may be private at commit time. Anonymous access
must work when FIFO admission reaches it after the grace period; keep the fixed revision and model
file public until admission finishes and its durable archive is confirmed by the `SCORING` state.

Run the evaluator:

The public write endpoint must pass the conditional-create/replace probe and serve the fixed public
read prefix. The existing Supabase S3 endpoint fails that probe for candidate production; configuring
a private object store alone does not fix public publication, and changing a write endpoint does not
redirect the existing public URL. Resolve that deployment mapping and coordinate the accepted track
digest with the external weight-setter before launch. Keep the existing URL and digest unchanged
until that coordination is complete.

```bash
uv sync --extra storage
export STORAGE_URL=...
export ACCESS_KEY=...
export SECRET_ACCESS_KEY=...
export EVALUATOR_PRIVATE_STORAGE_URL=...
export EVALUATOR_PRIVATE_ACCESS_KEY=...
export EVALUATOR_PRIVATE_SECRET_ACCESS_KEY=...
# Optional HTTPS push notifications; failures are logged when this is unset.
# export VEDA_ALERT_WEBHOOK_URL=...

uv run --extra storage veda validator evaluate \
  --track-spec /data/track.json \
  --database /data/validator.db \
  --model-cache /data/models \
  --exact-cache /data/exact-batches \
  --production-config /data/production.json \
  --scoring-timeout-seconds 14400
```

AWS S3 and compatible providers use the same storage implementation and object paths. Set the
standard `AWS_DEFAULT_REGION` in both service environments when the provider requires a region;
it defaults to `auto` for existing deployments. Provider selection never changes the public read
URL or bypasses conditional-write and permission probes. Validators reading public weights do
not need storage credentials or a region setting.

The evaluator variables must identify a separate least-privilege storage role. Startup proves that the
role can use plan/stage CAS and read READY-selected objects, while reads of
`bootstrap/source_pool.json`, the `source-pool/` prefix, and the committed full source cache are
denied, including every `<manifest>.parts/part-*` object and its list prefix. Evaluator credentials
never fall back to the producer credentials.
Generate and verify the policy through the validator CLI before installing it on the backend:

```bash
veda validator evaluator-policy --bucket private --prefix evaluation/tracks/<track-digest> \
  --write evaluator-policy.json
veda validator evaluator-policy --bucket private --prefix evaluation/tracks/<track-digest> \
  --verify evaluator-policy.json
```

Policy generation refuses to overwrite an existing file; verification rejects extra grants.
This document check does not replace live credential denial probes against the target backend.
The production configuration also uses a separate `evaluator_artifact_root`; it must be mounted
without the producer artifact root, source-pool file, complete production/audit SampleCache, or
producer work tree. Mount only the producer's `work_dir/content` subdirectory read-only at the same
configured path in the evaluator, so it can read selected batch data without access to the full pool.

Failure events are logged when `VEDA_ALERT_WEBHOOK_URL` is unset; no external notification service
is required to start. To enable push notifications, configure it in both service environments as an
absolute HTTPS endpoint without URL credentials or fragments. Producer bootstrap,
scoring stages and complete producer attempts each have a four-hour wall-clock limit; bootstrap
failures emit a separate startup alert. Admission has five fast attempts; scoring and full-video
confirmation allow nine fast attempts with bounded exponential delays.
Production reserves each plan attempt in private object storage before GPU work. After five fast
attempts, it automatically probes at most once per hour, even across process or GPU-host loss,
and alerts on the first failure, the fifth failure and each failed probe. It stops once finalized
scanning passes the track close block. Admission and scoring alert on the first failure and when
their fast retry budget is exhausted. Waiting for a finalized batch boundary or normal producer lag
does not consume
an attempt; a batch still missing one complete generation interval after its scheduled block enters
bounded scoring retries. Exhaustion enters visible `SYSTEM_BLOCKED` cooldown without becoming a
Miner loss. Admission
automatically probes again every 30 minutes; scoring and confirmation every hour, with another
alert on a failed probe. Retry counts and deadlines survive local state loss. FIFO order is
preserved, and recovery resumes the original stage without an operator reset. To request an earlier
retry after repairing the cause:

```bash
veda validator retry <challenge-id> --database /data/validator.db \
  --production-config /data/production.json
```

Public publication failures are counted in the same SQLite database across restarts. The first
and fifth failure emit alerts through the configured channel. Five consecutive failures stop new
admission and scoring;
finalized chain scanning and immutable facts remain intact. Recovery probes continue at the poll
interval, and only successful publication of the complete continuous match prefix and its pointers
clears the gate. Reading the ledger successfully or skipping publication does not clear it.

After the track close grace block, unfinished system work first publishes an irreversible
`weights.json` fence with `track_aborted=true` and no weight plan, then the immutable
`TRACK_ABORTED` event. Validators stop submitting, and the evaluator stops evaluation and every
later pointer refresh for that track; the affected Miner is not marked as losing. Recovery requires
a new track ID. Commitments finalized after the close block are outside the track and are not added
to its ledger; already-scanned submissions beyond the frozen submission cutoff are terminally
recorded as `REJECTED_TRACK_CLOSED` without downloading or evaluating their models.

`--exact-cache` contains immutable NPZ batches named `<generation-block>-<sha256>.npz`. Every batch
binds its finalized block hash, source proofs, runtime artifacts and each case's planned
source/rollout, route, step/sigma and deterministic query selection. Admission replays restore their
archived result before contacting the submitted repository. Repository-access and model-validation
refusals are archived before local completion: recovery preserves the original reason, decision
block and downloaded file identity even if the repository later changes visibility. Failure to
archive a refusal defers work as an infrastructure fault instead of rejecting the participant.
The producer verifies
atomic conditional creates and replacements against that endpoint before starting. Local SQLite,
model downloads and work directories are disposable.

Private uploads and reused chunks are checked against their actual bytes before an archive is
acknowledged; HEAD metadata alone is insufficient. Public verification, private verification and
restoration share one bounded streaming verifier for per-part and complete SHA-256 checks.

Run the private data producer separately with the same production configuration:

```bash
export PRIVATE_STORAGE_URL=...
export PRIVATE_ACCESS_KEY=...
export PRIVATE_SECRET_ACCESS_KEY=...

uv run --extra storage veda validator produce --config /data/production.json
```

Run the producer and evaluator in separate service or container environments. Never inject the
producer's `PRIVATE_*` variables into the evaluator environment.

The fixed production allocation is GPUs 2-5 for streaming batch production, GPUs 0-1 for paired
full-video confirmations, and GPUs 6-7 left free for continuing calibration experiments.

If `--initial-model` is omitted, the evaluator restores the pinned UID 0 VEDA weights from the
public chunk manifest, verifies every chunk, the complete file SHA-256, the canonical tensor
digest, and then reruns it beside every challenger. Local model and batch caches are disposable.

Run any Validator, including the evaluator operator's independent Validator:

```bash
uv run veda validator run
```

This process needs no evaluator database, model cache, hidden data, GPU, or storage credentials. It
also needs no subnet, mechanism or digest arguments: it fetches `track.json` beside the fixed
`weights.json` URL on every cycle and validates both together. Wallet name, hotkey and network are
the only operator-specific settings.
It rejects artifacts for another track or subnet, future or stale snapshots, invalid normalization
or caps, insufficient UID count, and hotkey-to-UID mappings that differ from the finalized
metagraph.
