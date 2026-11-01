# VEDA — Sparse-Attention Training Competition

Train a better attention-block selector for video generation on Bittensor **SN120, mech 1**.
The video model and inference code are fixed; miners submit selector weights, not an inference
service. The current champion receives all weight allocated within this mechanism, not the
entire subnet's weight. UID0 starts as champion using the official VEDA selector.

- [Live track specification](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/track.json): authoritative model and scoring parameters.
- [Leaderboard and queue](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/status.json) · [Match summary](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/summary.json).
- [VEDA research](https://veda-sparse.github.io/) · [Reference model](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview).

Repository [track.json](track.json) contains the exact bytes of the existing live rules,
`veda-sn120-v2-r10`, SHA-256 `a971c2714bf145f1953674492990207a6a61eecf61c093c487e179c114c24b63`.
The [immutable original specification](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/artifacts/a971c2714bf145f1953674492990207a6a61eecf61c093c487e179c114c24b63)
pins data origin `4e405540e88443dd21541f305d75821c1b3707129c6651dea842e9d1b485c1aa`.
Earlier repository templates remain historical Git versions. Synchronizing this file does
not change live rules, start, data or runtime identity; application release tags identify
the installed application commit, independently of later configuration/documentation commits.
Use `git show track-a971c2714bf1:track.json` to read the tagged rules. That configuration
tag targets the synchronization commit; `release-2026-11-01-1b95e40` targets the installed
application commit. Neither tag authenticates unrecorded historical execution details.

## What to train

Improve the selector's query/key projections so it picks useful attention blocks under the
track's fixed block budgets. Current promotion uses the paired exact attention-quality/cost
comparison; full-video confirmation is disabled. Submitting the unchanged reference weights
does not establish an improvement.

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

### 3. Stage an immutable Hugging Face revision

The SHA-bound VDA2 workflow is supported by the 2026-11-01 native release; check the
[changelog](CHANGELOG.md) and live runtime before using it on-chain.
Upload to an existing private repository and keep it private until the chain commitment
is finalized:

```bash
uv run hf auth login
uv run hf upload YOUR_ACCOUNT/YOUR_MODEL selector.safetensors selector.safetensors
```

Changing hotkey metadata does not change the duplicate-model fingerprint.

Obtain the upload's **complete 40-character commit SHA** from the repository's commit history.
Use that immutable revision, not `main`, a tag or a shortened hash. Do not modify the selected
file after choosing the revision.
Compute the complete SHA-256 of the prepared file, including its hotkey metadata:

```bash
sha256sum selector.safetensors
```

### 4. Commit on-chain, then make that revision public

**Each hotkey has one VEDA evaluation opportunity; maintenance does not reset it.** A malformed
submission, invalid model or lost challenge consumes that opportunity. Only an explicit
`SYSTEM_NOT_ADMITTED` outcome for missing eligible pre-submission data preserves it for a later
submission. Ordinary infrastructure retries are not permission to submit a replacement.

```bash
uv run veda miner commit \
  --repo-id YOUR_ACCOUNT/YOUR_MODEL \
  --revision YOUR_40_CHARACTER_COMMIT_SHA \
  --file-sha256 YOUR_64_CHARACTER_FILE_SHA256 \
  --wallet-name YOUR_WALLET \
  --hotkey YOUR_HOTKEY_NAME
```

The tool defaults to SN120 on Finney, signs with your hotkey and waits for finalization.
Add `--dry-run` first to inspect the encoded reference **without sending a transaction**;
this checks submission fields, not the model or eligibility. The command sends once and does
not automatically retry. If a transaction's outcome is uncertain, inspect its chain status
before submitting again.

After finalization, make that same repository/revision anonymously readable without
changing the file. The first actual anonymous HF metadata HTTP 401/403/404 observation starts one fixed
10-minute publication window. During that wait, no infrastructure failure budget is spent;
the original task is checked again at the deadline. Restarting, retrying or operator recovery
does not extend the window. Time alone does not reject a model: a later actual 401/403/404
after the deadline can fail the submission, while 429/5xx/timeouts remain infrastructure
failures under the original retry budget. Missing historical observation times stay unknown.
Once anonymous metadata succeeds, a CDN payload access error remains an infrastructure
failure, including 401/403/404 after the publication deadline.
Keep the revision public through admission and durable archiving, confirmed by `SCORING`.

Supplying `--file-sha256` selects VDA2: repository IDs are at most 71 ASCII bytes and the
payload is at most 128 bytes. The declared SHA must match the downloaded file before model
ownership is registered. Omitting this option preserves VDA1's original bytes, 96-byte
repository limit and 121-byte payload limit; existing miners and submissions need no
resubmission. The publication wait applies to future HF observations in either format;
historical outcomes and missing observation times are not rewritten.
Private uploads, VDU/VDR and the public weight format are unchanged.

## Evaluation and rewards

1. Submissions enter a continuously open queue ordered by finalized block, extrinsic and event
   position. There is no submission deadline or competition-wide challenge limit. Each hotkey
   still has one evaluation opportunity, subject to the explicit system-non-admission exception above.
2. Admission checks registration, competition eligibility, model format, hotkey binding and
   duplicate tensors. Re-signing an earlier model or copying the baseline is not a new model.
3. Challenger and champion are evaluated on the same data: 16 videos across four strata,
   eight sampled layer/head routes and three denoising steps, giving 384 paired cases. Retries
   retain the already-bound evaluation data. Data must have completed before the original
   finalized submission and still have been secret and not yet due for disclosure then.
   The same hidden data can serve multiple models; it is not produced separately for each model.

   The [2026-11-01 native release](CHANGELOG.md) (`36cc9ff`) excludes sources already revealed
   in verified public batch disclosures from future plans. New plans freeze the inputs they observed;
   later publication does not reselect an existing plan or change its validity window.
   Insufficient remaining sources in a stratum/slot stop generation rather than reuse
   public sources. Previously sealed data and allocations remain unchanged.
   Confidence bounds group correlated cases by video
   rather than treating all 384 cases as independent.
   Confidence is specified per comparison, not as a lifetime false-promotion guarantee.
4. Only a challenger whose exact paired lower confidence bound strictly exceeds the applicable
   threshold becomes champion. Confidence is 0.9975. The threshold decreases linearly from
   0.003 to 0.002 over the incumbent's first seven continuous days, then stays at 0.002.
   Incumbent tenure and the threshold are locked at the original finalized submission's chain
   time; losing challenges and maintenance do not restart tenure. A true winner starts a new
   tenure. Equality retains the incumbent. No full-video stage or second batch is requested.
5. Results are published in queue order without waiting for raw evaluation data publication.
   Raw inputs follow the live track's fixed delay from actual data completion; immediate results
   contain scores and non-revealing commitments, not prompts or reconstruction seeds.
   The champion receives this mechanism's full weight
   until replaced or no longer eligible; the registered UID0 baseline is the fallback.

Maintenance preserves confirmed champions, spent submission opportunities and original match
records; it does not rejudge past results under new rules.

Infrastructure failures defer and retry the original challenge; they do not count as miner
losses. Retries preserve already-bound evaluation data and archived stage results rather than
selecting replacement data. Quality metrics, budgets and thresholds are defined by the live track,
not adjustable per submission. Scores from different batches are not directly comparable;
promotion uses the paired comparison with the incumbent.
After the existing retry budget is exhausted, SYSTEM_BLOCKED waits for an explicit `veda evaluator retry` operation. Time alone does not restart model or GPU work. Recovery preserves the original submission, allocation and FIFO order; public blocked status has no automatic retry ETA.

Public immutable match records bind the submitted model, stage scores and evaluation data by
their recorded hashes. When replay data is available, reproduce a match using its original
frozen specification, runtime and bound inputs, not the latest working tree or a different
hardware execution path. Archived outcomes and data bindings are not rewritten by a retry or
maintenance release.

New evidence-enabled matches seal a small score audit before publication. Once the original
batch reaches its disclosure age, the existing publisher exposes the artifact referenced by
`stage_evidence.public_audit`. It contains paired cell scores, their grouping/order, bootstrap
parameters and recorded execution identity. Future question data and private model payloads
are excluded. Aggregate-only historical matches have unknown missing evidence; today's
runtime or a new experiment cannot fill it in.

Download the original public track envelope, match and referenced audit JSON, then run:

```bash
uv run veda verify-match --track track.json --match match.json --artifact audit.json
uv run veda verify-match --track track.json --match match.json --artifact audit.json \
  --batch-disclosure batch-group.json
uv run veda audit
```

`verify-match` checks the small JSON artifacts on CPU. Without the original
`batch_groups/<plan_digest>.json`, it explicitly reports `statistics-only`: self-reported
cells alone do not authenticate the original case design. With that disclosure, the match's
READY SHA authenticates the plan and every route/rollout/source/step/window in canonical
cache order. Track case count and geometry are always checked. This does not replay model
inference or verify tensor payloads; `model_inference_reproduced` remains false.
The credential-free `audit` command
uses the normal public ledger verifier, including available published payload bytes, which
may stream gigabytes of public model and disclosed evaluation data. Run full audit only on
an authorized, provisioned audit host, not as a routine development-machine check.
Neither command needs a wallet, GPU, private storage keys or evaluator database.
Statistical replay does not establish full model-inference replay when a private losing model
is intentionally unavailable.
The false-positive rate has not been empirically calibrated.

Full audit resolves each historical match's original rules once from
`artifacts/<track_spec_digest>`, authenticating its bytes by that digest. Missing original
rules are reported in `unavailable_score_audits`, never replaced with today's spec.
`verify-match --track` accepts this same immutable spec or the original track envelope.

The [2026-11-01 native release](CHANGELOG.md) (`1b95e40`) reports `disclosure_overdue` plan
digests separately from `disclosure_in_progress`.
These are operational observations from match-bound deadlines and the public inventory's
original retry projection, with observation time/source retained; they are not proof of a
complete private pool or of payload availability. A live upload lease is not a known failure.
The operator API `ReadyBatchSource.waive_disclosure` authenticates an original due READY plan
and records an explicit decision in its existing disclosure retry CAS. It preserves attempts,
errors and clocks, and suppresses subsequent automatic work and overdue reporting. It does
not cancel an upload already in flight or retract its bytes. A waiver never creates a
disclosure receipt or permits deletion of the original data. Its reason stays private;
an actual public receipt remains authoritative.
A pass with no known durable failure clears a stale error even while due uploads remain
fenced. This does not claim upload completion or refresh the successful disclosure
observation time; an actual exception still reports failure.

## Watch the competition

```bash
uv run veda monitor
uv run veda monitor --watch 30
```

The read-only CLI shows UIDs, hotkey prefixes, champion, queue positions, match numbers, scores,
outcomes, waiting/computing activity and retry alerts. It needs no wallet, GPU
or storage key. `status.json` is the current
view; `summary.json` includes the committed match count and latest match. Immutable match records
use `matches/<track_id>/<12-digit-sequence>.json` under the same public `tracks/main/` prefix.
Scores from different batches form no absolute-score ladder. Comparison views show the
champion and submission chronology, with each match's paired LCB and original threshold.
Public diagnostics use fixed safe summaries and recorded retry origins; raw exception text
and credentials remain private. Unknown historical retry or duplicate details stay unknown.

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

Code boundaries are [`miner`](src/miner) (submission), [`scoring`](src/scoring) (validation and
metrics), and [`validator`](src/validator) (evaluation, publication and independent weight setting).
The evaluator and data producer run separately; miners and ordinary validators do not run them.
Operator commands are documented by `uv run veda evaluator --help`.

```bash
uv run ruff check src tests
uv run pytest
```

GPU-dependent validation needs the frozen runtime described by
[`Dockerfile.runtime`](Dockerfile.runtime), or its verified native installation with the
actual installed lock and acceptance. A native installation does not imply a Docker Engine.
[Runtime verification](evidence/system-panel/README.md) explains evidence binding and operational
limits. Research experiments are not additional miner submission requirements.
The [changelog](CHANGELOG.md) separates unreleased source from the dated deployed baseline;
the live runtime identity remains authoritative for the current installation.
