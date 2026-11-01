# Changelog

## Unreleased

- Derive structured disclosure overdue/upload observations through the existing audit
  repository. Preserve the difference between operational status and immutable public receipts.
- Record explicit operator disclosure waivers in the original retry CAS, preserving its
  attempt/error history and stopping automatic work without marking data as disclosed.

## 2026-11-01 — Native deployment of 36cc9ff

Operator-reported acceptance: application commit
`36cc9ffaaa17988bd7ef8b35f851e4910290ad32`, native installation.
Annotated tag: `release-2026-11-01-36cc9ff`, targeting that application commit.

| Installed identity | SHA-256 |
| --- | --- |
| Application source | `83923c182c7bf12f490bfaf0f393b9ff770bfdadc04e34fad38fac24974dd375` |
| Native dependency lock | `201360dcc3962e296da908a3ce0ee21ca6af4c7751ab1ac76467aed212dbfdfc` |
| Runtime manifest | `93ab97fc02c53eb25f6bd5ecc873c1598c42753ae40cb07df715580348b732fc` |
| Accepted release | `c9451c6e0ebc0d22130d46618b78b7e627afa22113b3073a2717d4964c668460` |
| Computational dependency environment | `760c0f5eb73df2da4916caade6bcff625416eee7f52167cb2f5a8dfe55d7bf16` |

- Exclude already disclosed source identities from future inventory plans. New plans bind
  their verified public disclosure inputs; cold recovery replays that frozen input instead
  of today's growing set. Preserve legacy plans, READY seals, allocations and disclosure clocks;
  insufficient remaining sources stop generation rather than reusing public sources.

At acceptance, public status/summary/weights had advanced to block 9410935 with the new
runtime identity; the database continued to 9410942 with 12 successful challenges and 12
stages. The six non-runtime configurations and original submission facts were unchanged.
No new natural v2 plan had been observed; installed support is not proof of a generated plan.
Track `veda-sn120-v2-r10`, data identity, original S, allocations, READY bytes and C/R remain
unchanged, with confidence 0.9975 and a single exact stage. E2/E3/E16 retain their original
measurement identities. OCI remains package provenance, not a new native Engine identity.

## 2026-11-01 — Native deployment of 34774f9

Operator-reported acceptance: application commit
`34774f9da8e375117f7f37dbe08766b3343cdb8f`, native installation.
Annotated tag: `release-2026-11-01-34774f9`, targeting that application commit.
The existing OCI identity remains dependency/package provenance, not a new Docker Engine deployment.

| Installed identity | SHA-256 |
| --- | --- |
| Application source | `c5bda78ed0215910407a7ece8152e9c99c7862f7076c5112dcf7994f8ce2c1de` |
| Native dependency lock | `486e2ca46509670c5255bc5a1a9ac770e4332440c60f17210fbbdb3109b70b5b` |
| Runtime manifest | `6c35276b2d4c936935f7fe2a6b8c8a439f13effb68699f35da31d20c2af85106` |
| Accepted release | `f3a82eac057bb6edc521b05330d7fc22d5d8c8b7eb5f8b3c9c00cfff8df7df8a` |
| Computational dependency environment | `760c0f5eb73df2da4916caade6bcff625416eee7f52167cb2f5a8dfe55d7bf16` |

Track `veda-sn120-v2-r10`, its original competition/data bindings, start, database,
READY bytes and C/R clocks remained unchanged. The six non-runtime configuration files
were byte-identical across maintenance. Original E2/E3/E16 reports retain their own frozen
source and hardware identities; this release does not relabel or repeat those experiments.

- Bound automatic infrastructure retries by the existing budget; preserve blocked tasks,
  original data-scoped operator recovery and FIFO, with no automatic public blocked retry ETA.
- Preserve the original VDA1 wire format and add opt-in VDA2 HF file commitments within
  the existing 128-byte payload limit. Bind the declaration through the original task,
  admission, archives, public match and authenticated recovery before assigning model ownership.
- Use the original admission retry CAS for a fixed ten-minute HF publication wait starting
  at the first actual anonymous HF metadata 401/403/404 observation. Waiting preserves the infrastructure
  failure budget, original submission and FIFO; reset and recovery retain the original clock.
  CDN payload errors after successful anonymous metadata retain infrastructure retry semantics.

Historical outcomes and missing evidence are not rewritten. Seed precommitments, protocol
anchors, statistical recalibration and future source exclusion are not part of this installed release.

## 2026-11-01 — Native deployment of 63ca340

Operator-reported acceptance: application commit
`63ca3400336f766bad9844c2acd57e423c1c69a3`, native installation; not a new Docker Engine
deployment. The existing OCI descriptor remains dependency/package provenance.
Annotated navigation tag: `release-2026-11-01-63ca340`, targeting the application commit
above rather than any later documentation commit.

| Installed identity | SHA-256 |
| --- | --- |
| Application source | `bb6141c3da179f2fb8f66091ef5784b30e7882ebf844b6266df1eed7a7ed4b80` |
| Runtime manifest | `64a4ae2946853a7a3808146995b5341080565c8ec34c3660ae73e70f1d1b8591` |
| Acceptance | `db51fcc52181e3692efef2ba806bfcf30f865f225f520f4945f8b6344a5e8a5d` |
| Installed native lock | `b51b2b29b1dbf52742f96a1955df599a6f4ebf3454aade093e61c365c93377fb` |
| Operator maintenance receipt | `8cdd778011889db7b9abc7a7529208acd4d48573b25b89fcc980de4c2d0e7a40` |

The operator verified the new public runtime identity at source block 9409592. At 15:46 CST,
the original database had continued to cursor 9409627 with 12 successful challenges and 12
stages, no challenge errors, eight unchanged READY commitments/C/R, and UID643 weight 1.
The original six non-payload configurations, DB, S, claims and data identity were preserved.
CDE remained unchanged. The original maintenance failure and strict integrity checks were
retained; no dependency source/RECORD or verification guard was relaxed.

The following capabilities are installed. No new natural evidence-enabled match had occurred
at acceptance: installation and CLI help checks do not establish capture, disclosure or
public statistical replay of a new real match. Historical missing evidence remains unknown.

- Seal per-match execution identity, calculated paired cells, statistical parameters and
  first-selection observations through the existing claim and stage authorities.
- Disclose one hash-bound public score audit after the original batch disclosure age.
  `veda verify-match` recomputes its scores, LCB and decision from the original public JSON.
  Missing historical cells, execution identity or first-allocation observations stay unknown.
- Reuse public audit receipts before reading private stage inputs. Candidate input evidence
  records complete observed enumeration and original plan facts, not a globally atomic pool.
- Bound cold public receipt checks per disclosure pass; warm history uses only the existing
  confirmed-byte cache and one unfinished audit is processed without a new retry ledger.
- Document each evidence authority, consumer, disclosure boundary and cleanup condition in
  the existing operations guide; remove the superseded set-only case check and duplicate
  SQLite evidence bodies after successful immutable stage archival.
- Add credential-free, read-only `veda audit` through `S3Repository.load`.
- Resolve each historical audit against its original hash-authenticated rule artifact,
  shared per spec digest. Report missing old rules as unavailable without substituting the
  current track. Preserve the original spec through the existing immutable publication path.
- Bind original case count, geometry and ordered route/rollout/source/step/window design
  when capturing or restoring stage evidence. Public `verify-match` reports statistics-only
  without the authenticated batch disclosure and never claims private model-inference replay.
- Present the champion and recent comparisons without ranking different batches by absolute
  scores. Show each comparison's LCB and original submission-locked threshold.
- Publish allowlisted diagnostic summaries, known duplicate references, recorded automatic
  and operator retry origins, and existing competition exclusion reasons.
- Reject a re-signed copy of the official baseline by canonical tensor digest without creating
  a synthetic challenge or assigning baseline ownership to a miner.

This maintenance preserves
the active rules, confidence 0.9975, exact-only promotion, original submissions, opportunities,
model privacy and original data commitments. No GPU experiment or historical re-evaluation
is part of these changes.

## Previous baseline (operator report, 2026-11-01)

- Application commit: `f8c40c04815d853a206493236672cc53aefaae87`.
- Application source digest:
  `822ffe95f3b310c8f2b5905787a34578ffdf7419ee294fa78a21ef27d48086d0`.
- The native runtime uses its installed dependency lock and approved runtime manifest;
  an OCI descriptor records package provenance, not a native Docker Engine identity.
- Authenticated READY consumption preserves original stored dense bytes and structural,
  SHA, plan, data-contract and submission gates. It does not recompute old dense data on
  the consuming GPU as an admission gate. Production and untrusted-cache validation remain.

This is a dated operational report, not a permanent claim about the current deployment.
Use the public `status.json` runtime identity and its bound manifest/acceptance for the live
release. Historical E2/E3/E16 reports retain their original frozen source and hardware;
new runtime releases do not rename those measurements or authenticate earlier executions.
