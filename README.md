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
4. A valid model waits for the earliest hidden exact-attention batch generated after its commitment
   boundary. The batch records its generation block, private-source digest, and content hash.
   Each batch samples unpredictable layer/head routes and covers every configured data stratum once
   per sampled route. Batches may serve multiple already-committed models, but no model may use an
   older batch.
5. UID 0 is the genesis champion without making a commit. Its identity is pinned to the official
   VEDA repository revision, filename and file SHA-256 in the track specification. Its baseline
   and every challenger are evaluated by the same scorer. For every match, the challenger and
   current champion are rerun on the same new batch. A challenger replaces the champion only when
   the lower confidence bound of their paired per-case score gain reaches
   `minimum_frontier_gain`.
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

Before live rewards, the rotating exact-cache score must be validated against full-video quality on
the target GPU. The production track must also pass ranking stability, cost calibration, baseline
headroom, and adversarial-submission experiments. Those are track-acceptance experiments rather
than a second runtime scoring system.

## Public object-storage state

The public contract uses one fixed prefix:

```text
tracks/main/
  track.json
  summary.json
  status.json
  weights.json
  system_matches/
    000000000001.json
    ...
  matches/
    veda-sn120-v1/
      000000000001.json
      000000000002.json
      ...
```

- `track.json` is the stable pointer to the current canonical track specification and digest. An
  active `track_id` has one fixed specification; an upgrade uses a new `track_id`, while the URL
  remains unchanged.
- `system_matches/*.json` are immutable launch checks produced from real GPU panels. They compare
  the VEDA baseline with fixed control models, but never enter the Miner queue, tournament rank, or
  chain weights.
- `matches/<track_id>/*.json` are immutable terminal match facts in chain order: public model reference,
  content hashes, aggregate result, incumbent, decision, and decision block.
- `summary.json` is a bounded current view containing the champion, top leaderboard and committed
  match count.
- `status.json` is the bounded live view used by the monitoring CLI; it includes UID 0, active
  entrants, queue positions, match numbers, scores and outcomes.
- `weights.json` is the small mutable artifact consumed by Validators.

Publication writes the current track and immutable matches first, then `summary.json`, then
`weights.json`.
A crash can therefore leave an extra immutable match but cannot expose weights for an uncommitted
summary. On restart the evaluator validates every committed match, follows any contiguous orphaned
matches, and restores them instead of evaluating them again. Conflicting immutable bytes stop
publication. A chain node behind the recovered public watermark may keep scanning, but evaluation
and publication wait until its finalized height catches up, so public state cannot move backward.

Object storage is the durable source for public tournament facts. The local SQLite database and
downloaded model cache are disposable execution state: after loss, finalized chain submissions plus
verified published matches reconstruct completed work, and remaining submissions can be evaluated
again.

Public artifacts deliberately exclude hidden samples, per-sample or per-layer metrics, local paths,
worker and lease data, retry timing, error text, stack traces, and credentials. Only stable error
codes are public. Use a public HTTPS custom domain for reads; mutable objects are published with
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

uv run --extra storage veda validator evaluate \
  --track-spec /data/track.json \
  --database /data/validator.db \
  --model-cache /data/models \
  --exact-cache /data/exact-batches \
  --archive-endpoint wss://archive.example \
  --scoring-timeout-seconds 1800
```

Keep the live UID 0 state current before the first evaluation track is published:

```bash
uv run --extra storage veda validator initialize
```

This loop publishes the immutable system checks plus `status.json` and `weights.json`. It verifies
that the pinned hotkey is still UID 0 at every finalized snapshot and stops writing as soon as the
evaluator publishes the immutable `track.json`.

`--exact-cache` is a directory populated with immutable NPZ batches named
`<generation-block>-<sha256>.npz`. Every batch contains the same generation block and its source
digest in validated metadata. Publish each file with one atomic rename only after it is complete.
Hidden arrays and source material are never published; their durable private copy must live outside
the disposable GPU host.

If `--initial-model` is omitted, the evaluator restores the pinned UID 0 VEDA weights from the
public chunk manifest, verifies every chunk, the complete file SHA-256, the canonical tensor
digest, and then reruns it beside every challenger. Local model and batch caches are disposable.

Run any Validator, including the evaluator operator's independent Validator:

```bash
uv run veda validator run
```

This process needs no evaluator database, model cache, hidden data, GPU, or storage write credentials. It
also needs no object-storage or track arguments: the public URL, subnet, and pinned track digest are
release defaults. Wallet name, hotkey, network, and public track fields remain optional overrides.
It rejects artifacts for another track or subnet, future or stale snapshots, invalid normalization
or caps, insufficient UID count, and hotkey-to-UID mappings that differ from the finalized
metagraph.
