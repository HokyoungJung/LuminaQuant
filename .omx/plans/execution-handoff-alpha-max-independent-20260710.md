# Execution Handoff — Alpha Max Independent

## Authority

- Approved Ralplan revision: `5.14`
- Consensus gate: `.omx/plans/ralplan-consensus-alpha-max-independent-20260710.json`
- Consensus SHA-256: `3d1f3447d676801b6c73e4805f57d23550ef29f1489604bc95be79ab87951746`
- Baseline commit: `252910e54e280cc593365484cbc99d6ca87893f9`
- Isolated branch: `feat/alpha-max-20260710`
- Isolated worktree: `/home/hoky/Quants-agent-alpha-max-20260710`

## Outcome

Implement the approved return-first, portfolio-first crypto-perpetual alpha research package: daily trend persistence, daily near-52-week-high anchoring, 4h funding-harvest carry, and equal-weight/ERC/shrunk-HRP allocation variants. Preserve causal native-timeframe decisions, atomic 8h funding cashflows, fail-closed artifact provenance, exact 1,487-trial statistical-family accounting, and the frozen MDD policy (normal at or below 30%; soft band above 30% through 35% only under strict matched-row CAGR and Calmar superiority).

No local dataset collection or performance claim is required. Local CI proves implementation and integrity only. The data-bearing PC will run the real backtest and confirmation workflow.

## Architecture invariants

1. The approved plan, PRD, test specification, current-node registry, incumbent audit, and their exact hashes are normative.
2. Legacy/default portfolio behavior is byte/semantics neutral unless the new optional seams are explicitly used.
3. Artifact manifest/config parsing, hashing, and receipt creation use exactly one opened descriptor per artifact, reject unsafe targets, and bind consumer receipts to runner seals.
4. Generic manifest consumers accept arbitrary unique source IDs/counts; alpha-only code enforces the exact two-receipt contract.
5. Candidate identity and admitted execution membership remain distinct; the same 5–10 admitted symbols bind every runtime surface.
6. Funding is causal, UTC-boundary exact, membership-aware, cash-settled atomically, and disabled under the legacy `None` resolver.
7. Historical intervals are report-only; no exposed window may be called untouched. Real performance confirmation remains external.
8. Do not touch, stage, reset, clean, or copy unrelated `.omc` or shared-session source changes.

## Durable goals

### G001 — Establish isolated execution baseline and integrity plumbing

Create the exact isolated branch/worktree from the frozen baseline; carry only this session's approved planning artifacts. Implement `ArtifactReadReceipt`, generic manifest/source receipt propagation, alpha-only receipt enforcement seams, portfolio funding attribution/resolver seams, and regression tests without changing legacy behavior.

### G002 — Implement the three alpha sleeves and portfolio artifacts

Implement the daily trend, daily near-52-week-high, and 4h funding-harvest strategies using existing repository patterns. Add the frozen research-only configs/manifests/current-node artifacts and causal symbol/admission/funding contracts. Add focused unit/integration coverage.

### G003 — Implement experiment runner, selection statistics, and external replay package

Implement the raw-first experiment runner, 21-node current registry binding, prior+current 1,487-trial DSR accounting, PBO/SPA diagnostics, cost cells, MDD/soft-band selection policy, incumbent handling, report-only historical labeling, deterministic bundle/report outputs, and external-PC runbook. Tests must prove integrity and rejection paths without fabricating performance.

### G004 — Integrate, review, verify, commit, and push

Run targeted and full repository CI gates, clean only changed files, independently review code and architecture invariants, repair all blockers, commit using Lore protocol, and push `feat/alpha-max-20260710` to the configured remote. Report exact CI evidence and the explicit limitation that performance remains unverified until the data-bearing PC runs the bundle.

## Team staffing

Use three coordinated executor workers in the isolated leader worktree, with the leader owning Ultragoal and integration:

- Lane A: artifact receipt/security and generic portfolio integration.
- Lane B: strategy sleeves, funding chronology, configs/manifests.
- Lane C: experiment/statistics/reporting/tests and later independent verification support.

Workers must use task-level file ownership and notify the leader before shared-file edits. Keep one lane responsible for test evidence. Final code review requires distinct code-reviewer and architect evidence.

## Portable checkpoint update — 2026-07-11

The implementation has advanced beyond the original execution handoff. The portable, current continuation authority is:

- `docs/research_note/alpha_max_independent_checkpoint_20260711.md`
- `.omx/ultragoal/brief.md`
- `.omx/ultragoal/goals.json`
- `.omx/ultragoal/ledger.jsonl`
- this directory's Revision 5.14 Ralplan, PRD, test spec, registry, incumbent audit, consensus, architect review, and critic review.

G001 and G002 local implementation contracts are complete. G003 is partial: execution attribution, pure metrics/statistics/trial accounting, and the strict runtime-contract foundation exist; actual two-engine replay, matrix orchestration, selection/terminal logic, physically separated CLIs, data-PC bundle, and performance evidence remain. The stored Ultragoal status was not falsified after the Codex goal became `usageLimited`; use the checkpoint document's actual-status table while preserving the original goal audit.
