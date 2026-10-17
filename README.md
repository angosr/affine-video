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
4. A valid model waits for its single scheduled exact-attention batch boundary. Boundaries occur at
   the interval fixed by `track.json`; the finalized block hash at that boundary seeds the public
   route/sample planner. The batch records that block hash, source digest, sample identifiers and
   content hash. Each batch covers every configured stratum once per sampled route. A later batch
   cannot be substituted to obtain a more favorable result.
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

The launch checks use full-video trajectories on NVIDIA H200 hardware. The VEDA baseline beat both
the MeanPool and shuffled-head controls on every tested geometry; those immutable measurements are
published under `system_matches/`. Exact-cache ranking, cost calibration, bootstrap stability,
malformed archives, non-finite weights, metadata impersonation, repacking and re-signing duplicate
weights are covered by the executable test suite. These checks validate the one runtime scorer;
they are not a second scoring path.

## Public object-storage state

The public contract uses one fixed prefix:

```text
tracks/main/
  track.json
  summary.json
  status.json
  weights.json
  system_matches/
    veda-sn120-v1/
      000000000001.json
      ...
  matches/
    veda-sn120-v1/
      000000000001.json
      000000000002.json
      ...
  batches/
    <sha256>.npz
  models/
    veda-sn120-v1/
      <sha256>.safetensors
```

- `track.json` is the stable pointer to the current canonical track specification and digest. An
  active `track_id` has one fixed specification; an upgrade uses a new `track_id`, while the URL
  remains unchanged.
- `system_matches/*.json` are immutable launch checks produced from real GPU panels. They compare
  the VEDA baseline with fixed control models, but never enter the Miner queue, tournament rank, or
  chain weights.
- `matches/<track_id>/*.json` are immutable terminal match facts in chain order: public model reference,
  content hashes, aggregate result, incumbent, decision, and decision block.
- `batches/<sha256>.npz` publishes the exact QKV batch after its first match finishes, allowing
  anyone to replay the score and verify its finalized-block seed.
- `models/<track_id>/<sha256>.safetensors` is the durable copy of every winning selector. The next
  match restores its incumbent from this archive, so deleting the original Hugging Face repository
  cannot stop the tournament.
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

Before a match completes, public artifacts exclude its hidden arrays, per-sample metrics, local
paths, worker and lease data, retry timing, error text, stack traces, and credentials. The used
batch is revealed only with the immutable match. Only stable error codes and retry counts are
public. Use a public HTTPS custom domain for reads; mutable objects are published with
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

uv run --extra storage veda validator evaluate \
  --track-spec /data/track.json \
  --database /data/validator.db \
  --model-cache /data/models \
  --exact-cache /data/exact-batches \
  --archive-endpoint wss://archive.example \
  --scoring-timeout-seconds 1800
```

`--exact-cache` is a directory populated with immutable NPZ batches named
`<generation-block>-<sha256>.npz`. Every batch contains that generation block, its finalized hash,
source digest and selected sample identifiers in validated metadata. The evaluator verifies the
hash against finalized chain state and publishes the used batch after the match. A lost GPU host
can deterministically regenerate an unpublished batch from the public generator and chain seed.

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
