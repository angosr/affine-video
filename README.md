# Veda Subnet

Veda Subnet is a Bittensor competition for training sparse-attention selectors. It has three code
boundaries:

- `miner`: validates and submits a pinned Hugging Face model reference;
- `validator`: evaluates submissions, publishes the public tournament ledger, or independently
  applies public weights on-chain;
- `scoring`: validates selector weights and runs deterministic staged evaluation.

There is one expensive evaluator process. It has R2 write credentials but no Validator wallet.
Every Validator, including the one operated alongside the evaluator, runs the same separate
lightweight process: fetch `weights.json`, validate it against finalized chain state, and sign its
own `SetWeights` transaction. No service or direct Miner-to-Validator protocol exists.

## Competition flow

1. A Miner commits a Hugging Face `repo_id` and a complete 40-character commit SHA on-chain.
2. After the fixed publication grace period, the evaluator processes submissions by finalized
   `(block, extrinsic, event)` order. The pinned revision must then be anonymously downloadable.
3. The evaluator accepts exactly one `selector.safetensors` file. It rejects executable code,
   metadata, unknown tensors, adapters, base-model patches, invalid shapes, and excess bytes.
4. Cheap screening and exact local scoring decide whether a model reaches the hidden full-video
   rollout. Only a successful rollout produces an authoritative score.
5. Terminal submissions are adjudicated in chain order. The first successful model becomes the
   initial champion. A later model replaces it only when
   `challenger_score - champion_score >= minimum_frontier_gain`.
6. An immutable match record is published for every terminal item in the contiguous chain-order
   prefix. A failed or unfinished earlier submission can never be skipped to publish a later win.

Scoring covers several frozen block budgets. The authoritative score combines quality and
calibrated attention cost, so a selector must preserve quality while selecting fewer blocks. The
track specification pins every score-affecting rule and asset hash.

Rewards use chronological frontier gains rather than raw leaderboard rank. Exact duplicates earn
nothing, sub-threshold changes earn nothing, and splitting the same sequence of improvements among
several hotkeys cannot increase total utility. Deregistered historical winners remain part of the
frontier calculation, so their removal cannot retroactively turn an earlier loser into a winner.
The resulting utilities are filtered to currently registered hotkeys and normalized under current
Bittensor count and weight-cap constraints. When no valid vector exists, `weights` is `null` and
Validators keep their last finalized weights.

The hidden rollout panel must be rotated between reward epochs. Before live rewards, the production
track must also pass real-GPU repeatability, ranking stability, cost calibration, baseline headroom,
and adversarial-submission experiments. CPU tests validate mechanics; they do not establish video
quality or target-GPU utilization.

## Public R2 state

Each frozen track uses one prefix:

```text
tracks/<track>/
  track.json
  summary.json
  weights.json
  matches/
    000000000001.json
    000000000002.json
    ...
```

- `track.json` is immutable and contains the canonical public track specification and digest.
- `matches/*.json` are immutable terminal match facts in chain order: public model reference,
  content hashes, aggregate result, incumbent, decision, and decision block.
- `summary.json` is a bounded current view containing the champion, top leaderboard and reward
  entries with their totals, and the committed match count.
- `weights.json` is the small mutable artifact consumed by Validators.

Publication writes the immutable track and matches first, then `summary.json`, then `weights.json`.
A crash can therefore leave an extra immutable match but cannot expose weights for an uncommitted
summary. On restart the evaluator validates every committed match, follows any contiguous orphaned
matches, and restores them instead of evaluating them again. Conflicting immutable bytes stop
publication. A chain node behind the recovered R2 watermark may keep scanning, but evaluation and
publication wait until its finalized height catches up, so public state cannot move backward.

R2 is the durable source for public tournament facts. The local SQLite database and downloaded
model cache are disposable execution state: after loss, finalized chain submissions plus verified
R2 matches reconstruct completed work, and remaining submissions can be evaluated again.

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

Preview a Miner payload locally:

```bash
uv run veda miner submit \
  --repo-id owner/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --netuid 1
```

Add `--execute`, wallet options, and the target network to sign, submit, wait for finalization, and
exit. The model revision may be private at submission time, but it must be public when its grace
period expires.

Run the evaluator:

```bash
uv sync --extra r2
export VEDA_R2_ACCESS_KEY_ID=...
export VEDA_R2_SECRET_ACCESS_KEY=...

uv run --extra r2 veda validator evaluate \
  --track-spec /data/track.json \
  --database /data/validator.db \
  --model-cache /data/models \
  --screen-cache /data/screen.npz \
  --exact-cache /data/exact.npz \
  --rollout-runner /opt/veda/rollout \
  --rollout-panel /data/rollout-panel \
  --archive-endpoint wss://archive.example \
  --r2-account-id 0123456789abcdef0123456789abcdef \
  --r2-bucket veda-public \
  --r2-prefix tracks/main
```

The rollout runner is a local executable started by the evaluator, not a network service. Omitting
both rollout arguments keeps qualifying submissions provisional and prevents them from receiving
weights.

Run any Validator, including the evaluator operator's independent Validator:

```bash
uv run veda validator run \
  --public-weights-url https://public.example/tracks/main/weights.json \
  --track-spec-digest <64-hex-track-digest> \
  --netuid 1 \
  --wallet-name default \
  --hotkey default
```

This process needs no evaluator database, model cache, hidden data, GPU, or R2 write credentials. It
rejects artifacts for another track or subnet, future or stale snapshots, invalid normalization or
caps, insufficient UID count, and hotkey-to-UID mappings that differ from the finalized metagraph.
