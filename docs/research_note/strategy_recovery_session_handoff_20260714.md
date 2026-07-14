# Strategy recovery cross-session handoff — 2026-07-14

## Resume identity

- Repository: `/home/hoky/Quants-agent/LuminaQuant`
- Main recovery branch: `recovery/strategy-plan-20260714`
- Committed implementation: `66c85d5da2edbe42c8e9f359ea59582dd814f997`
- Original private-main baseline: `1bd4405532271527d6c89ba7e3e55bf78c510eb6`
- Durable Ultragoal session id: `019f603a-0e73-7000-88a7-c94f42950c09`
- Durable state: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
- Audit ledger: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`
- Machine-readable state snapshot: [`strategy_recovery_resume_state_20260714.json`](strategy_recovery_resume_state_20260714.json)
- Run evidence root: `/home/hoky/quants-recovery-runs/20260714T105113Z`
- Writable market snapshot: `/home/hoky/quants-recovery-market/20260714T105113Z/market_parquet`

The aggregate objective remains:

> Complete the durable ultragoal plan in `.gjc/ultragoal/goals.json`, including later accepted/appended stories, under the original brief constraints; use `.gjc/ultragoal/ledger.jsonl` as the audit trail.

The session-specific paths above, not the unrelated legacy `.gjc/ultragoal/goals.json`, are authoritative for this run. Supply `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09` to native `gjc ultragoal` commands in a new session.

## Safety contract that remains binding

- Preserve all user work and keep every original data root read-only.
- Never use synthetic data, symbol substitution, pre-listing fill, forward-fill, shortened dates, missing-funding proxies, locked-OOS reselection, paper/testnet/live orders, or capital allocation.
- Never run Alpha-Max prelock or historical evaluation until its runbook alignment is committed and a complete authorized canonical source exists.
- Do not consume the quarantined Alpha source named below.
- Do not push either branch unless the user explicitly requests a push.
- Every story ends in PASS, scientific KILL, or an immutable external-blocker receipt.

## Durable goal state at handoff

| Goal | State | Evidence/meaning |
|---|---|---|
| G001 — D-01 inventory | complete | Source/snapshot inventories frozen; synthetic selected-root count zero; gaps disclosed. |
| G002 — D-01A/D-04/D-05 | superseded | Cumulatively replaced and verified by G008–G012. |
| G008 | superseded | D-04 separately approved; D-01A continued through blocker chain. |
| G009 | superseded | Replaced by stronger funding bijection/seal work. |
| G010 | superseded | Replaced by complete ownership work. |
| G011 | superseded | Automatically superseded by completed G012. |
| G012 — provider identity/receipt semantics | complete | Fresh strict quality gate, architect CLEAR/APPROVE, executor QA passed. |
| **G003 — Alpha-Max Rev5.15 alignment** | **active** | Resume here. Foundation is recorded below; alignment files remain unmodified. |
| G004 — exact R1/R2 replay/cost proof | pending | Start only after G003 checkpoint. |
| G005 — bounded data recovery and R-04/A-03 decisions | pending | Depends on validated data and G003/G004 infrastructure. |
| G006 — C-00 through C-05 follow-up cycle | pending | Starts only after R-04 and A-03 are terminal. |
| G007 — F-01/C-06 fresh-forward | pending | At most permitted champions; 30/60-day evidence; zero orders/capital. |

A user-requested cross-session stop was appended to the Ultragoal ledger while G003 remained active. The aggregate inline goal was then classified as human-blocked solely for the session transfer and paused; resume it in the new session before continuing G003.

## Completed foundational implementation

Commit `66c85d5da2edbe42c8e9f359ea59582dd814f997` contains:

- fail-closed research data contract validator;
- contract-wide official funding ownership/catalog reconciliation;
- canonical Binance provider identity and funding alias conflict rejection;
- exact pre-receipt schema, D-04, metric, count, digest, and seal validation;
- strict physical/repository OHLCV route parity;
- point-in-time symbol lifecycle/fold manifests;
- materializer `--help` root fix;
- adversarial tests for relabeling, wrappers, overlap, truncation, hash/digest drift, lifecycle forgery, and CLI boundaries.

Latest verified evidence before handoff:

- focused G012 gate: `125 passed`;
- full tracked suite: `4475 passed, 20 skipped, 3 xfailed`;
- Ruff, format check, and `git diff --check`: clean;
- real BTC June pre-append: expected `STOP`, exit `2`, 874 OHLCV interior gaps, 18 funding prefix gaps;
- architect: `agent://28-G012FinalArchitect` — CLEAR/CLEAR/CLEAR, APPROVE;
- executor QA: `agent://29-G012ExecutorQa` — passed;
- quality gate: `/home/hoky/quants-recovery-runs/20260714T105113Z/g012-quality-gate.json`;
- adversarial report: `/home/hoky/quants-recovery-runs/20260714T105113Z/g012-adversarial-test-report.json`.

## G003 state and exact resume point

A separate clean worktree already exists:

- Worktree: `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc`
- Branch: `recovery/alpha-max-rev515-alignment-20260714`
- HEAD/baseline: `629d91e5d4aac26911af65a4a5e15ebdcbded30f`
- Status at stop: clean; no alignment edits and no commit yet.
- Cancelled worker: `30-G003Rev515Runbook`; it stopped before editing.

The frozen Rev5.15 inputs were independently hash-checked and match the approved plan:

| Artifact | SHA-256 |
|---|---|
| `configs/research/alpha_max_portfolio_20260711_listing_aware.json` | `2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c` |
| `configs/research/alpha_max_contract_manifest_20260711_listing_aware.json` | `ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220` |
| `configs/research/alpha_max_official_availability_evidence_20260711.json` | `214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719` |
| `scripts/research/prepare_alpha_max_phase_roots.py` | `ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac` |

Runtime hashes that the aligned runbook must state:

- runtime contract: `b3859443c842cf8b04d04ed32923e6c6a8207af18e26f68a717ba623b4edfef9`;
- config payload: `b062e3805d94087cc18cd22634918815503f94dd73f8fa8ac1979e7aef535f85`;
- config file: `2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c`.

### G003 implementation still required

Delegate a bounded executor to edit only these two Alpha worktree files, then verify and commit them as the leader:

1. `docs/research_note/alpha_max_data_pc_runbook_20260711.md`
2. `docs/research_note/alpha_max_final_sha256_20260711.txt`

Required alignment:

- title/identity and baseline become Rev5.15 / commit `629d91e5d4aac26911af65a4a5e15ebdcbded30f`;
- make the four listing-aware artifacts and exact hashes above normative;
- correct the three runtime/config hashes;
- add a no-discretion phase-root preparer command using only authorized canonical `market_ohlcv_1s` plus `feature_points`, the listing-aware contract, and a new absent output root;
- replace every prelock config/contract command with the listing-aware paths;
- preserve immutable half-open dates:
  - warmup `[2022-12-31, 2024-01-01)`;
  - train `[2024-01-01, 2025-06-01)`;
  - purge `[2025-06-01, 2025-06-08)`;
  - validation `[2025-06-08, 2025-08-31)`;
  - embargo `[2025-08-31, 2025-09-07)`;
  - exposed historical `[2025-09-07, 2026-07-01)`;
- state TONUSDT official ownership exactly: raw `[2024-03-01T12:31:10Z, 2026-06-23T09:00:00Z)`, feature `[2024-03-01T16:00:00Z, 2026-06-23T09:00:00Z)`;
- require TON to fail original warmup/train admission; forbid GRAMUSDT substitution, synthetic warmup, synthesized listing-transition funding, date shifts, and post-delivery rows;
- retain exact structural checks: 68 prelock cells, 816 physical validation fold runs, 17 manifests per phase, 680 historical physical runs;
- update the final SHA manifest with the runbook hash and the four Rev5.15 artifacts.

Parent verification after the edit must include exact hash/date/path checks, focused Alpha config/preparer/runtime tests, cleanup, a full appropriate worktree gate, architect review, executor QA, then an Ultragoal G003 checkpoint. Commit the alignment on `recovery/alpha-max-rev515-alignment-20260714`; do not push without explicit instruction.

## Alpha source blocker — do not prepare phase roots

The only discovered 1s/funding candidate is explicitly quarantined and incomplete:

- candidate root: `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`;
- frozen inventory SHA-256: `74fe748ed4505824f79090e9d83ab43c20a997dbadbb136b29afa52943ca13aa`;
- warning: `/home/hoky/Quants-agent/LuminaQuant-data/DO_NOT_USE_alpha_max_20260711_listing_aware_source_UNAUTHORIZED_PARTIAL.txt`;
- warning SHA-256: `64f94db085ea673ff3f19b230bfd308bf3b0fad251e4c5feb2089f409e648990`;
- immutable blocker receipt: `/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-phase-preparation-blocker.json`.

Therefore phase-root preparation, prelock, and historical evaluation were not run. Do not read this source as performance-bearing input, materialize it, resume its backfill, or create phase roots from it. G003 can still PASS its alignment/commit portion while recording phase preparation as the external blocker.

## Remaining durable objectives

### G004 — Implement exact R1/R2 replay and strict cost-proof infrastructure

Add an immutable exactly-two-candidate frozen replay seam for the two named router IDs; require faithful per-fold actual-engine leaf manifests, point-in-time membership, `evaluation_mode` handler or `registry_simulator`, current-fold OOS exclusion, and `generic_fallback_proxy=0`. Add one frozen combined strict plus cost-realistic replacement profile and same-signal 10/15/20/30bp proof/gate path with complete funding, sqrt impact, exposure normalization, purge/embargo, DSR, SPA, PBO, MDD, liquidation, leave-best-fold-out, and dominance checks. Test fail-closed behavior and do not perform a new grid search.

### G005 — Recover bounded data and execute R-04 and A-03 scientific decisions

Using only validated owned intervals, repair continuous Router tails and actual funding/support data, create pre/post receipts, and use bounded repair manifests for any interior/prefix gaps. Execute the frozen R1/R2 proof once and record PASS/KILL for every candidate. After Alpha-Max alignment and phase preparation, execute its one-touch prelock/historical protocol once, verify sealed bundles and byte identity, and record PASS/KILL without date/candidate retuning. Continue independent executable work when one track has an external blocker.

The Alpha A-03 branch remains externally blocked until an authorized complete canonical source exists; continue the independent Router track rather than substituting data.

### G006 — Preregister and execute C-00 through C-05

Only after R-04 and A-03 are terminal, freeze `candidate-manifest.json`, `data-contract.json`, `trial-ledger.json`, and hashes for the approved bounded registry; validate Crypto/TradFi/metals contracts; run V-DIAG admission, standalone actual-engine walk-forwards, eligible V-PAIR/V-OVERLAY/V-COV ablations, and strict same-signal cost proofs. Preserve all rejects, whole-search trial counts, locked-OOS report-only semantics, prior-of-death, and exact STOP/KILL reasons. Add no new strategy implementation unless preregistered feature admission passes and no existing implementation exists.

### G007 — Run F-01 and C-06 to terminal evidence

Forward at most one Router/Alpha-Max champion per recovery track and at most one follow-up alpha leaf plus one risk overlay, only if prior gates pass. Freeze commit/config/manifest/universe/risk/cost hashes; start after the last previously viewed complete bar; record daily actual funding, hypothetical orders/fills, costs, reconciliation, kill switches, and hash drift with zero orders and zero capital. Maintain 30-day checkpoint and 60-day terminal PASS/KILL evidence; reset the clock on any permitted freeze change.

## New-session bootstrap

1. Read this handoff plus the three binding plan/runbook/audit documents.
2. Activate `/skill:ultragoal`.
3. For every native Ultragoal command, use `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09` so the prior durable state is selected.
4. Run `gjc ultragoal status --json` and confirm G003 is active, G012 is complete, G002/G008–G011 are superseded, and G004–G007 are pending.
5. Reconcile the inline goal tool: if no active goal exists, create the stable aggregate objective printed above; if it is paused, resume it; never create a competing goal.
6. Confirm both worktrees are clean at the commits/branches stated above.
7. Resume G003 at the two-file runbook alignment. Do not rerun completed G001/G012 work and do not consume the quarantined source.
