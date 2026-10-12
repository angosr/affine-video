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

