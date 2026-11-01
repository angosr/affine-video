# Changelog

## Unreleased

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
- Bind original case count, geometry and ordered route/rollout/source/step/window design
  when capturing or restoring stage evidence. Public `verify-match` reports statistics-only
  without the authenticated batch disclosure and never claims private model-inference replay.
- Present the champion and recent comparisons without ranking different batches by absolute
  scores. Show each comparison's LCB and original submission-locked threshold.
- Publish allowlisted diagnostic summaries, known duplicate references, recorded automatic
  and operator retry origins, and existing competition exclusion reasons.
- Reject a re-signed copy of the official baseline by canonical tensor digest without creating
  a synthetic challenge or assigning baseline ownership to a miner.

These source changes have not been deployed by this development session. They preserve
the active rules, confidence 0.9975, exact-only promotion, original submissions, opportunities,
model privacy and original data commitments. No GPU experiment or historical re-evaluation
is part of these changes.

## Operator-reported deployed baseline, 2026-11-01

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
