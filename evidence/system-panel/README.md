# Runtime verification

The live track is `veda-sn120-v2-r1` on SN120/mech1. Public paths stay under
`tracks/main/`; there is one `weights.json`.

- Track digest: `cf747b59ae96a39782ee3660f0c8e3029003e784087fc37102bf9a36695e3eb3`.
- Frozen source: `364c0ca5e8ea0483cfc1417238cb19daa34dbb5d`.
- Source digest: `f86ddf56a3eeb4e0a37781a089f37961fb3542da5ac967d50f85ba6de1086eed`.
- Image: `sha256:6086b0bf747106c01f21341e7853cfee0996a4a8524af809f8af259d867a8f0b`.

## Release checks

The [inventory](acceptance/inventory.json) contains the three launch checks, not a seventeen-item
research completion target. Existing startup checks separately enforce runtime integrity,
storage capabilities, data isolation and track consistency.

| Check | Current frozen-runtime evidence |
| --- | --- |
| E2: numerical fidelity | 96 historical cases, three controls, 288 comparisons; zero mask mismatches; maximum sparse-output RSE 7.42e-18 |
| E3: two-stage execution | Sixteen independent videos per phase; self retains the champion, degraded challengers lose, the reference defeats a degraded incumbent; both gates are required |
| E16: determinism and errors | Actual CUDA tie-order regressions, six same-card repeat pairs and nine cross-card cells; identical scores and output hashes |

E3 uses an isolated accelerated chain fixture with real GPU computation, SQLite and off-GPU
TLS S3. It is not a real Miner commitment or a wallet transaction. E2/E16 retain their historical,
quarantined input identities rather than being relabeled as new competition batches.

[Public E2/E16 original logs, registrations, test reports and SHA-256 values](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/system_matches/veda-sn120-v2-r1/evidence/README.md)
are anonymously readable. All thirteen objects were independently read and matched byte-for-byte.
The E2 summary additionally references three private E3/audit logs: those are not in this public
package and must not be recursively disclosed. No private source pool, current audit inputs,
encoded text, QKV, latents, videos or credentials are included.

The restored source `validator.release` gate requires canonical acceptance of E2/E3/E16,
bound by the runtime manifest and matching scoring/runtime contracts. Missing, failed or
mismatched reports reject formal publication before storage writes. This source change has
not been installed into the already-running frozen image or published under its old digest.

## Operational evidence and limits

The deployed split is GPUs0–1 for evaluation, GPUs2–5 for generation, and GPUs6–7 for diagnostics.
At the actual 256GiB/16CPU pool limits, production regenerated sixteen videos with all48 file
hashes matching; confirmation reproduced all96 scores/output hashes. All six production GPUs
were observed computing concurrently without OOM. These are real pool checks, not sustained
capacity guarantees.

The current manifest's56 static assets, 97,296,695,944 bytes, have off-GPU copies, together with
the current configuration and role-credential sources. Eight obsolete image archives were
removed without changing the current assets. An empty-cache baseline restore recovered all
550,511,648 bytes from off-GPU private S3 with the correct full SHA. Killing the assembler mid-NPZ
and retrying reproduced the original10.30GB hash while pruning its incomplete file.

A fresh public-HF restore also recovered one605,429,308-byte fixed model file to an empty tree
without an original-model mount:144.48 seconds,4.19MB/s, independently rehashed. The completed
container and duplicate scratch were removed. Two small runtime packaging files have now been
backed up to the existing private cloud bucket and independently read back; all descriptor/part
anonymous reads were denied. The17GB cloud-image backup has not been performed. None of these
component results is described as a complete host-loss drill.

The single32MiB control-to-GPU SSH pilot measured0.404MB/s, while one complete anonymous40MiB
public-S3 read measured112MB/s. Different directions, cache states and object sizes prevent
extrapolating either into a full-restore promise. The original three-hour97GB transfer was not run.

No real Miner has completed the chain-to-reward lifecycle, and the external weight-setter's
actual pin and transaction remain unverified. Public system-match summaries do not affect rewards.
Private state uses verified CAS; public Supabase uses one managed writer, monotonic checks and
readback, not a distributed atomic-write guarantee. Proactive HTTPS alerts still require a real
receiver; log/status monitoring is not described as delivered push notification.

## Development diagnostics

The previous broad research inventory is preserved outside the source repository; its exact
reference is in the active inventory. It is not an additional release checklist.

Exploratory cost prediction and ten-percent overhead targets were not met in every context.
The failed fit was not adopted, and the fixed score-axis costs do not promise exact end-to-end
GPU latency. These are disclosed limitations, not fabricated passes or automatic launch blockers.
Fixed self/degraded/external controls have complete exact and video measurements; no operator
model training is required. Wider ranking, sample-power and perceptual calibration can continue
against actual operation rather than delaying launch for an idealized complete research program.

`launch/` retains replay and measurement tools used by existing evidence; they do not run as
services or authorize publication. `operations/` contains the maintained asset-preservation tool.
Raw experiments and intermediate outputs remain outside the repository.
