# Veda Subnet

Veda Subnet is a sparse-attention selector training competition on Bittensor. The repository has
three business boundaries:

- `miner`: one-shot model-reference submission;
- `validator`: owner evaluation, public R2 artifacts, and independent chain weight setting;
- `scoring`: deterministic model validation and evaluation.

The owner runs the only expensive evaluation loop. It scans finalized submissions, downloads and
scores models, then publishes public results and recommended weights to Cloudflare R2. Every
Validator, including the owner's Validator, runs the same lightweight loop: fetch weights from the
public R2 URL, validate them against current finalized chain state, and sign its own `set_weights`
transaction. The evaluation process never opens a Validator wallet.

The CPU evaluator, finalized-chain queue, challenge history, R2 artifact path, and independent
weight loop are implemented. A real Video DiT rollout backend and frozen production track/cache
bundle are still required before live rewards can be enabled.

## Submission and scoring

Each on-chain submission contains a Hugging Face `repo_id` and a complete commit SHA. The pinned
repository contains one `selector.safetensors` file with only Veda v1 tensors named
`layers.<layer>.proj_q` and `layers.<layer>.proj_k`. Shapes are frozen by the track as
`[num_heads, 3 * head_dim, head_dim]`. Metadata, biases, adapters, executable code, and extra
tensors are rejected.

A Miner may keep the revision private while submitting it. The track's positive
`publication_grace_blocks` delays admission by a fixed number of finalized blocks. The Miner must
make the file anonymously readable before that deadline. Evaluation then proceeds in finalized
chain order; an unavailable repository or revision receives a recorded participant failure and the
next submission proceeds. There is no ownership proof or commit-reveal phase. Reference and
canonical-weight deduplication give priority to the earlier chain position.

All submissions on a track use the same hidden panel. Screening covers every route and stratum and
only limits evaluation cost. Local scores are provisional; only a successful full rollout is
eligible for chain weight. Utility is the final-score improvement over earlier successful
submissions. Improvements from the same hotkey accumulate, exact copies receive no marginal
utility, and gains below the frozen `minimum_frontier_gain` receive no utility.

A provisional update cannot remove the same hotkey's previous confirmed result. It replaces that
result only after rollout success or deterministic participant failure. Infrastructure failure does
not permanently reserve a repository reference or model digest.

Before production, rollout scoring must be validated on real paired videos. The hidden rollout
panel must rotate and successful incumbents must be re-audited for each reward epoch; otherwise a
fixed public score becomes a black-box training oracle.

## Public R2 artifacts

The owner publishes two canonical JSON objects under one R2 prefix:

- `results.json`: leaderboard, weight-source history, and complete public challenge records;
- `weights.json`: the small hotkey/UID/weight vector consumed by Validators.

Both bind to `track_id`, `track_spec_digest`, `netuid`, and one finalized `source_block`.
`results.json` is uploaded first and `weights.json` last. Publication is skipped while the chain
scanner is behind, so an incomplete submission history cannot become a public weight vector. If
the current chain constraints cannot hold a valid vector, `weights` is `null`.

Validators reject a weight artifact when it targets another track or subnet, is stale or from the
future, violates current count/cap constraints, or contains a hotkey-to-UID mapping that no longer
matches the finalized metagraph. They also use the Validator hotkey's chain `last_update` and the
subnet rate limit to avoid duplicate or premature transactions.

Use an R2 custom domain for production. The publisher sets `Cache-Control: no-store`; do not add a
cache rule that overrides it for these mutable JSON objects. Cloudflare's `r2.dev` endpoint is
rate-limited and intended only for development.

## Commands

Run tests:

```bash
uv run pytest
```

Preview a Miner submission without opening a wallet or network connection:

```bash
uv run veda miner submit \
  --repo-id owner/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --netuid 1
```

Add `--execute`, `--wallet-name`, `--hotkey`, and optionally `--network` or `--wallet-path` to
submit, wait for finalization, and exit.

The owner installs the R2 publishing extra and supplies S3 credentials only through environment
variables:

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

An independent Validator needs no evaluation database, model cache, hidden data, GPU, or R2 write
credentials:

```bash
uv run veda validator run \
  --public-weights-url https://public.example/tracks/main/weights.json \
  --track-spec-digest <64-hex-track-digest> \
  --netuid 1 \
  --wallet-name default \
  --hotkey default
```

The local rollout runner is an executable, not a network service. Evaluation starts it directly,
sends one JSON request on stdin, and reads one JSON result from stdout. The track digest pins the
runner and hidden panel by SHA-256. Omitting both keeps qualifying results provisional; adding both
later automatically queues saved provisional results that meet the rollout threshold.

A successful runner returns exactly one object whose binding fields echo the request:

```json
{
  "status": "ok",
  "challenge_id": "...",
  "file_sha256": "...",
  "track_digest": "...",
  "final_score": 0.73,
  "attention_cost": 0.18,
  "metrics": {"video_samples": 4}
}
```

Deterministic model failures use `status: "participant_error"`; retryable GPU or environment
failures use `status: "temporary_error"`. Non-zero exits, malformed JSON, unknown fields, changed
hashes, or binding mismatches are treated as infrastructure failures rather than participant scores.
