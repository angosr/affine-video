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
   A hotkey already in the old contest is rejected, while a hotkey's first recognized choice for
   the new contest remains fixed even if it later writes unrelated commitments.
2. After the fixed publication grace period, the evaluator processes submissions by finalized
   `(block, extrinsic, event)` order. The pinned revision must then be anonymously downloadable.
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
hotkey-to-UID mapping and chain weight constraints before publishing. When the champion is no
longer registered or a valid vector is impossible, `weights` is `null` and Validators keep their
last finalized weights.

The launch checks use ten consecutive 384-case batches on NVIDIA H200 hardware. They require the
VEDA incumbent to retain the title against MeanPool, shuffled-head and slightly perturbed VEDA
challengers in every batch, while the reverse VEDA-versus-shuffled positive control must promote
in every batch. The immutable summaries are published under `system_matches/`. Full-video
execution, cost calibration, clustered bootstrap stability,
malformed archives, non-finite weights, metadata impersonation, repacking and re-signing duplicate
weights are covered by real-GPU checks and the executable test suite.

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
  chain weights.
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
  every referenced source video and dense latent. Each manifest verifies every bounded part and the
  reassembled file digest.
- `models/<track_id>/<sha256>/manifest.json` is the public durable copy of every winning selector.
  A private copy of every validated selector is stored before scoring, so publication and later
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
Mutable pointers use ETag conditional writes, and production startup probes both conditional create
and conditional replace. A backend that cannot fence two concurrent publishers is rejected.

Object storage is the durable source for public tournament facts. The local SQLite database and
downloaded model cache are disposable execution state: after loss, finalized chain submissions plus
verified published matches reconstruct completed work, and remaining submissions can be evaluated
again.

Private R2 is the durable source for unrevealed plans, the committed source pool, the artifact
manifest, every runtime artifact, exact batches, validated selectors and recoverable scoring stages.
Both production services verify conditional-write support before starting, seed missing immutable
objects from a complete local runtime, and restore missing local bytes by committed size and SHA-256.
The GPU host therefore contains no unique tournament state.

The manifest must cover a restorable OCI image, the exact producer/scorer source bundle and its
dependency lock. A canonical source-cache binding joins the source-pool Merkle root to the complete
encoded SampleCache index, text tensors and optional condition tensors. Producer startup verifies
that binding and exact source-ID coverage, so changing encoded input under an existing source ID is
a hard failure rather than a new hidden dataset with the old commitment.

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
opening a chain connection. The model revision may be private at commit time, but it must be public
when its grace period expires.

Run the evaluator:

```bash
uv sync --extra storage
export STORAGE_URL=https://project.storage.supabase.co/storage/v1/s3
export ACCESS_KEY=...
export SECRET_ACCESS_KEY=...
export PRIVATE_STORAGE_URL=https://account.r2.cloudflarestorage.com
export PRIVATE_ACCESS_KEY=...
export PRIVATE_SECRET_ACCESS_KEY=...

uv run --extra storage veda validator evaluate \
  --track-spec /data/track.json \
  --database /data/validator.db \
  --model-cache /data/models \
  --exact-cache /data/exact-batches \
  --production-config /data/production.json \
  --scoring-timeout-seconds 14400
```

Set `VEDA_ALERT_WEBHOOK_URL` to an HTTPS push endpoint before continuous operation. Admission has
five attempts; scoring and full-video confirmation allow nine attempts with bounded exponential
delays, and scoring and producer workers have a four-hour wall-clock limit. Production retries a
failed plan at most five times and sends a push alert on the first and final failure. Admission and
scoring do the same. Waiting for a finalized batch boundary or normal producer lag does not consume
an attempt; a batch still missing one complete generation interval after its scheduled block enters
bounded scoring retries and sends the same first/final alerts. Scoring
exhaustion enters visible `SYSTEM_BLOCKED` state without becoming a Miner loss; after repairing the
cause, resume its original stage with
`veda validator retry <challenge-id> --database /data/validator.db`.

After the track close grace block, unfinished system work first publishes an irreversible
`weights.json` fence with `track_aborted=true` and no weight plan, then the immutable
`TRACK_ABORTED` event. Validators stop submitting, and the evaluator stops evaluation and every
later pointer refresh for that track; the affected Miner is not marked as losing. Recovery requires
a new track ID.

`--exact-cache` contains immutable NPZ batches named `<generation-block>-<sha256>.npz`. Every batch
binds its finalized block hash, source proofs, runtime artifacts and each case's planned
source/rollout, route, step/sigma and deterministic query selection. Private R2
stores the source pool, runtime-specific artifacts, every generated batch and every validated
selector together with its immutable challenge binding and recoverable stage results. Admission
replays restore this archive before contacting the submitted repository. The producer verifies
atomic conditional creates and replacements against that endpoint before starting. Local SQLite,
model downloads and work directories are disposable.

Run the private data producer separately with the same production configuration:

```bash
uv run --extra storage veda validator produce --config /data/production.json
```

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
