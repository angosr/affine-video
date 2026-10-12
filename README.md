# Veda Subnet

Veda Subnet evaluates sparse-attention selector weights submitted through Bittensor.

The code has three business boundaries:

- `miner`: one-shot local model submission;
- `validator`: chain scanning, challenge state, ranking, and weights;
- `scoring`: deterministic model evaluation.

The CPU evaluator, finalized-chain queue, challenge history, and weight planner are implemented.
The repository does not yet contain a real Video DiT rollout backend or a frozen production
track/cache bundle, so it is not ready for a live competition and chain weight submission must
remain disabled outside test environments.

The submitted `selector.safetensors` contains only Veda v1 projection tensors named
`layers.<layer>.proj_q` and `layers.<layer>.proj_k`. Their shapes are fixed by the track as
`[num_heads, 3 * head_dim, head_dim]`. Scoring uses the same TripPool residual projection as
Veda: `x_hat = [mean|max|min] @ projection + mean`, followed by scaled Q/K dot products. Model
metadata, biases, adapters, executable code, and additional tensors are rejected.

All submissions on one track are compared on the same exact hidden panel. The sample covers every
selector route and scoring stratum. Screening runs over the complete lightweight cache and only
controls evaluation cost; it is not a ranking signal. Local scores remain provisional and receive
no chain weight until a full rollout succeeds. The independent rollout supplies the authoritative
final score; the fixed local panel is only a cost gate. Chain utility is the final-score improvement
over earlier successful submissions, so later behaviorally identical copies have no marginal
utility.

An already active `repo_id + revision` is rejected before downloading it again. Different
repositories that contain identical weights are rejected after canonical tensor hashing.

Before live rewards are enabled, the rollout score must be validated against real paired videos.
The rollout panel must rotate and successful incumbents must be re-audited before each reward epoch;
otherwise public scores turn a fixed hidden set into a black-box training oracle. The submission
flow also needs an ownership proof or commit-reveal step to prevent repository front-running.
Neither condition can be established by the current CPU-only tests or implemented by the current
single-phase submission payload.

Run the current test suite with `uv run pytest`.

Preview a Miner submission without accessing a wallet or network:

```bash
uv run veda miner submit \
  --repo-id owner/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --netuid 1
```

Add `--execute`, `--wallet-name`, `--hotkey`, and optionally `--network` or
`--wallet-path` to sign with the hotkey, wait for finalization, and exit.

Run one Validator cycle after a track and its hidden caches have been frozen:

```bash
uv run veda validator run \
  --track-spec /data/track_spec.json \
  --database /data/validator.db \
  --model-cache /data/models \
  --screen-cache /data/screen.npz \
  --exact-cache /data/exact.npz \
  --rollout-runner /opt/veda/rollout \
  --rollout-panel /data/rollout-panel \
  --archive-endpoint wss://your-archive-rpc.example \
  --once
```

The local rollout runner is not a service: Validator starts the executable directly, sends one
JSON request on stdin, and reads one JSON result from stdout. The track digest pins both the runner
file and the hidden rollout panel by SHA-256. Omitting both paths leaves qualifying results
provisional. Without `--once` the command continues polling. Chain weight submission is disabled by
default and is rejected unless rollout is configured; add `--submit-weights`, `--wallet-name`,
`--hotkey`, and optionally `--wallet-path` only on the signing Validator.

The runner request contains the challenge and model-file digests, the validated model path, the
pinned panel path, and the track's base-model revision, budgets, selector dimensions, and digest.
It must score the same frozen panel for every model; challenge identity must not affect sampling.
A successful runner writes exactly one JSON object:

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

The three binding fields must echo the request. A deterministic model failure uses the same fields
with `status: "participant_error"`, plus an uppercase `code` and a short `message`. A retryable GPU
or environment failure uses `status: "temporary_error"`. A non-zero process exit, malformed JSON,
unknown fields, changed hashes, or a mismatched binding is treated as infrastructure failure rather
than silently assigning a participant score.

