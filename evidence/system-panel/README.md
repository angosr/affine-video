# Runtime verification

The live track is `veda-sn120-v2-r2` on SN120/mech1, starting at block 9316700.
The original `tracks/main/` public pointers and single `weights.json` are unchanged.

- Track digest: `d1d4ced33ba2ec7478a96b8bdfc99daa0f3feaad6443ae3ff556e0f619f8d78b`.
- Frozen source: `c8685ad771a8edfebfe3de78cc11766f382f1676`.
- Source digest: `4df766931d59c20c6286b33609aa1288388752342588917f0b944b8e8ecc7c58`.
- Image: `sha256:b8f4c69b4b9536a37bd6bcb4f83185d8f9843016def1abd434805d2f324541d3`.

## Release checks

The deployed gate requires only E2/E3/E16 plus existing startup checks for runtime integrity,
storage capabilities, data isolation and track consistency. The [inventory](acceptance/inventory.json)
is an evidence index, not the canonical runtime authorization.

| Check | Evidence and scope |
| --- | --- |
| E2: numerical fidelity | Fresh H200 recapture: 96 historical cases, three controls, 288 rows; zero mask mismatches; rows identical to the previous image |
| E3: two-stage execution | Prior sixteen-video per-stage GPU/FIFO/real off-GPU S3 fixture, byte-identical scoring and tournament code; nine fresh candidate full-video cells reproduce original hashes and scores |
| E16: determinism and errors | Twelve fresh H200 backend tests, including tied-mask ordering; three self/repeat pairs have identical video scores and hashes |

[Public reports, original identities, raw GPU logs and registrations](https://iblosevwkuiootbbgnsy.storage.supabase.co/storage/v1/object/public/affine_video/tracks/main/system_matches/veda-sn120-v2-r2/evidence/README.md)
are anonymously readable. The canonical `accepted.json` is bound into the live runtime manifest
and verified by the installed gate. Only `validator/release.py` changed from the prior frozen
application; third-party execution bytes are unchanged. The full sixteen-video E3 was not rerun
in this maintenance image. Its chain fixture is accelerated and isolated, not a real Miner or
wallet transaction. Historical diagnostic sources are quarantined; current private evaluation
sources and exact caches are not included in these public logs.

## Operational limits

GPUs 0–1 evaluate, GPUs 2–5 generate, and GPUs 6–7 remain available for diagnostics. Previous actual
pool checks reproduce all 48 regenerated-file hashes and 96 full-video scores/output hashes at
the deployed resource limits. An empty-cache restore verifies all 550 MB of the baseline from
off-GPU private S3. Component results are not a whole-host recovery or sustained-capacity claim.

Private state uses verified CAS. Public Supabase uses one managed writer, monotonic checks and
readback, not distributed atomic publication. HTTPS push alerts require a real receiver;
log/status monitoring is not a delivered push notification. External weight-setter adoption and
a real Miner-to-reward lifecycle remain separately observable operational events.

Research experiments are not additional launch gates. Cost prediction and overhead targets were
not met in every exploratory context, and the failed fit was not adopted. Fixed scoring costs
do not promise exact GPU latency. Wider ranking, rare-failure power, motion-gate calibration and
full host recovery continue using operational feedback; failed or pending research is not passed
by relabeling it. No operator model training is required.

`launch/` retains measurement tools used by existing evidence; they do not run as services or
authorize publication. `operations/` holds the asset-preservation tool. Raw experiments and
intermediate outputs stay outside the repository.
