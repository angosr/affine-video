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

All submissions on one track are compared on the same exact hidden panel. The sample covers every
selector route and scoring stratum. Screening runs over the complete lightweight cache and only
controls evaluation cost; it is not a ranking signal. Local scores remain provisional and receive
no chain weight until a full rollout succeeds. The independent rollout supplies the authoritative
final score; the fixed local panel is only a cost gate. Chain utility is the final-score improvement
over earlier successful submissions, so later behaviorally identical copies have no marginal
utility.

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
  --archive-endpoint wss://your-archive-rpc.example \
  --once
```

Without `--once` the command continues polling. Chain weight submission is disabled by default;
add `--submit-weights`, `--wallet-name`, `--hotkey`, and optionally `--wallet-path` only on the
signing Validator.

