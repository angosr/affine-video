# VEDA — Sparse-Attention Training Competition

Train a better attention-block selector for video generation on Bittensor **SN120, mech 1**.
The video model and inference code are fixed; miners submit selector weights, not an inference
service. The current champion receives all weight allocated within this mechanism, not the
entire subnet's weight. UID0 starts as champion using the official VEDA selector.

- [Live track specification](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/track.json): authoritative model and scoring parameters.
- [Leaderboard and queue](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/status.json) · [Match summary](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/summary.json).
- [VEDA research](https://veda-sparse.github.io/) · [Reference model](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview).

## What to train

Improve the selector's query/key projections so it picks useful attention blocks under the
track's fixed block budgets. Scoring combines attention quality and calibrated cost; a candidate
must also preserve full-video quality to replace the champion. Submitting the unchanged reference
weights does not establish an improvement.

The reference repository contains the full video checkpoint, **not a ready-to-submit competition
file**. Export only the selector projections described below. Training methods and datasets are
up to you; this repository supplies the submission tool and evaluation implementation, not a
turnkey training pipeline.

## Submission format

Upload one model artifact, `selector.safetensors`, at the root of a Hugging Face **model** repository.
The current track requires:

| Field | Requirement |
| --- | --- |
| Tensor names | `layers.0.proj_q`, `layers.0.proj_k`, …, `layers.49.proj_q`, `layers.49.proj_k` |
| Tensor count | Exactly 100; no additional tensors |
| Shape and dtype | Every tensor is `(56, 384, 128)`, `float16`, with finite values |
| Metadata | Exactly `{"hotkey": "YOUR_FULL_SS58_HOTKEY"}` |
| Maximum file size | 560,000,000 bytes |

Do not include base-model weights, adapters or patches in this file. No training logs, training
configuration or executable code are required. The evaluator never runs code from your repository.
The [selector implementation](src/scoring/selector.py) defines the projection arithmetic;
the [model validator](src/scoring/model.py) enforces the file contract.

## Commands

### 1. Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/angosr/affine-video.git
cd affine-video
uv sync --locked
```

Your submitting hotkey must be registered on SN120. A hotkey activated in the original Affine
competition is ineligible for this competition; activating it there later also removes VEDA
reward eligibility. Registration alone is not an old-competition activation. No running miner
server, public endpoint or storage credentials are needed.

### 2. Bind your hotkey

Export your trained selector in the format above, then replace the placeholders:

```bash
uv run veda miner prepare \
  --model trained.safetensors \
  --output selector.safetensors \
  --hotkey-address YOUR_FULL_SS58_HOTKEY
```

This copies tensor values unchanged and sets ownership metadata. It refuses to overwrite an
existing output. **It does not check the complete tensor schema or competition eligibility.**

### 3. Upload to Hugging Face

```bash
uv run hf auth login
uv run hf upload YOUR_ACCOUNT/YOUR_MODEL selector.safetensors selector.safetensors
```

Ensure the repository is public when your finalized submission reaches FIFO admission.
Uploading to an existing private repository does not make it public. Changing hotkey
metadata does not change the duplicate-model fingerprint.

Obtain the upload's **complete 40-character commit SHA** from the repository's commit history.
Use that immutable revision, not `main`, a tag or a shortened hash. Do not modify the selected
file after choosing the revision.

### 4. Commit on-chain

**Only your hotkey's first VEDA submission on this track is eligible.** A malformed submission,
invalid model or lost challenge consumes that opportunity; later submissions cannot replace it.

```bash
uv run veda miner commit \
  --repo-id YOUR_ACCOUNT/YOUR_MODEL \
  --revision YOUR_40_CHARACTER_COMMIT_SHA \
  --wallet-name YOUR_WALLET \
  --hotkey YOUR_HOTKEY_NAME
```

The tool defaults to SN120 on Finney, signs with your hotkey and waits for finalization.
Add `--dry-run` first to inspect the encoded reference **without sending a transaction**;
this checks submission fields, not the model or eligibility. The command sends once and does
not automatically retry. If a transaction's outcome is uncertain, inspect its chain status
before submitting again.

When FIFO admission reaches your submission, the pinned HF revision and file must be
anonymously downloadable or the challenge fails. Admission can start as soon as the
commitment is finalized. Keep the repository public through admission and durable
archiving, confirmed by the `SCORING` state.

## Evaluation and rewards

1. Submissions enter a continuously open queue ordered by finalized block, extrinsic and event
   position. There is no submission deadline or competition-wide challenge limit. Each hotkey
   still has one submission opportunity.
2. Admission checks registration, competition eligibility, model format, hotkey binding and
   duplicate tensors. Re-signing an earlier model or copying the baseline is not a new model.
3. Challenger and champion run on the same hidden inventory batch: 16 videos across four strata,
   eight sampled layer/head routes and three denoising steps, giving 384 paired cases. The FIFO
   scorer claims the oldest available batch once; retries retain that allocation. Confidence
   bounds group correlated cases by video rather than treating all 384 cases as independent.
   Confidence is specified per comparison, not as a lifetime false-promotion guarantee.
4. Passing the attention-quality/cost gate claims a source-independent hidden batch for complete
   eight-step video inference at all three budgets. Only a challenger exceeding the frozen
   paired improvement thresholds in both stages becomes champion. Ties retain the champion.
5. Results are published in queue order. The champion receives this mechanism's full weight
   until replaced or no longer eligible; the registered UID0 baseline is the fallback.

Infrastructure failures defer and retry the original challenge; they do not count as miner
losses. Normal inventory waiting is not a failed attempt.
After 600 finalized blocks from submission (or the exact decision for confirmation), continued
inventory shortage raises a durable infrastructure retry/alert instead of waiting silently;
it still preserves the original opportunity and FIFO, and ready stock is claimed immediately.
Each batch belongs to one challenge/stage
and cannot be reused by a later model. Its data, committed secret and replay references are
disclosed only after that challenge finishes. Quality metrics, budgets and thresholds are
defined by the live track, not adjustable per submission. Scores from different batches are not
directly comparable; promotion uses the paired comparison with the incumbent.

The existing producer fills a small hidden inventory even without miners, pausing at the track's
batch-count and disk-byte limits. Generation combines a finalized block hash with a committed
private 256-bit secret; public block hashes alone cannot reconstruct unused inventory. Production
commitments have two fixed, family-disjoint halves; each challenge uses opposite halves for its
two stages. Confirmation may already exist before the exact decision. Lost unclaimed data is
CAS-invalidated and replenished; assigned hashes and archived results never change. Prompts may
recur with new secret-seeded rollouts; this is not an unlimited supply of new prompts.

## Watch the competition

```bash
uv run veda monitor
uv run veda monitor --watch 30
```

The read-only CLI shows UIDs, hotkey prefixes, champion, queue positions, match numbers, scores,
outcomes, inventory counts, waiting/computing activity and retry alerts. It needs no wallet, GPU
or storage key. `status.json` is the current
view; `summary.json` includes the committed match count and latest match. Immutable match records
use `matches/<track_id>/<12-digit-sequence>.json` under the same public `tracks/main/` prefix.

## Validators

Validators fetch the public [weights.json](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/weights.json),
validate it against the companion track and finalized metagraph, and submit mech-1 weights:

```bash
uv run veda validator run --wallet-name YOUR_WALLET --hotkey YOUR_HOTKEY_NAME
```

This requires a registered validator wallet, not a GPU, evaluator database or storage keys.
Track and mechanism settings come from the fixed public URLs. Existing Affine weight-setters
can consume the same file; do not run a second weight-setting process for the same wallet.

## Development and verification

Code boundaries are [`miner`](src/miner) (one-shot submission), [`scoring`](src/scoring)
(selector validation and metrics), and [`validator`](src/validator) (evaluation, publication
and independent weight setting). Public viewing uses `veda monitor`; evaluator-only operations
use `veda evaluator run`, `produce`, `policy`, `retry`, `leaderboard`, `challenges` and `challenge`
(see `uv run veda evaluator --help`). The evaluator and data producer run separately; miners
and ordinary validators do not run them.

```bash
uv run ruff check src tests
uv run pytest
```

GPU-dependent validation needs the frozen runtime in [`Dockerfile.runtime`](Dockerfile.runtime).
[Runtime verification](evidence/system-panel/README.md) links release identities, raw GPU evidence
and known operational limits. Research experiments are not additional miner submission requirements.
