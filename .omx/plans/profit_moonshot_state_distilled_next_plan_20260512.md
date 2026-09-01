# RALPLAN-DR — Profit Moonshot State-Distilled Next Execution Plan

Generated: 2026-05-12 KST
Mode chain: `$ralplan` → `$team` → `$ralph`
Context snapshot: `.omx/context/profit-moonshot-state-distilled-next-20260512T102355Z.md`

## RALPLAN-DR Summary

### Principles
1. **Legality over score:** no valid live strategy may use fixed calendar month/day/hour entry rules.
2. **OOS quarantine:** train/validation choose; locked-OOS only gates/reports after candidate freeze.
3. **Mechanism interpretability:** each new family must map to observable market-state mechanics.
4. **Strict-vs-diagnostic separation:** zero-liquidation strict lane governs live promotion; nonfatal liquidation diagnostics are research-only.
5. **Small grid first:** narrow, interpretable grids beat broad opaque scans.

### Decision drivers
1. Avoid hidden calendar leakage while preserving hypothesis value from the rejected tuple.
2. Improve OOS economics over current-base reference without using OOS as selector.
3. Preserve memory and verification reliability under the 8 GiB contract.

### Viable options
- **Option A — Extend current state-distilled runner in place (chosen):** add bounded families, tests, and provenance to existing replay/liquidation scripts.
  - Pros: minimal integration risk; uses existing artifacts/tests; easiest to verify.
  - Cons: runner remains large; careful naming/validity tests needed.
- **Option B — Build a separate research runner:** isolate new logic in a new script.
  - Pros: cleaner exploratory surface.
  - Cons: duplicates panel/liquidation plumbing; higher risk of provenance mismatch.
- **Option C — Tune existing family only:** avoid new mechanics.
  - Pros: smallest diff.
  - Cons: insufficient mechanism coverage and likely weak OOS economics.

## ADR

### Decision
Implement Option A: extend `scripts/research/replay_profit_moonshot_fresh_start.py` and `scripts/research/run_profit_moonshot_liquidation_aware_validation.py` with tests-first guardrails, narrow new non-calendar families, explicit selection provenance, and separate strict/diagnostic lane reporting.

### Drivers
- Existing runner already has panel cache, split windows, candidate CSV/JSON/MD outputs, memory guard, and family filtering.
- Existing liquidation validator already has strict gates and current-base comparison logic.
- User requires final replay/liquidation artifacts in the existing alpha_v2 report tree.

### Alternatives rejected
- Separate runner: rejected due duplicate gate/provenance risk.
- Calendar/current-base retuning: rejected because current-base is invalid as live alpha.
- Locked-OOS ranking: rejected because it violates the selection contract.

### Consequences
- New code must include calendar-field tests so the large spec factory does not accidentally promote calendar fields.
- OOS may still fail deployable improvement; if so, report research result honestly and preserve diagnostics.

## Execution plan
1. **Tests first:** add tests for non-calendar family spec fields, selection provenance, strict/diagnostic lane metadata, and diagnostics.
2. **Feature arrays:** add only required market-state arrays: funding percentile/z-score, OI acceleration/percentile, beta residual z-score, cross-sectional dispersion ratio.
3. **Signal families:** implement narrow grids for:
   - `crowded_leadership_unwind_v2`,
   - `funding_oi_exhaustion_reversal`,
   - `beta_residual_reversion_spread`,
   - `dispersion_compression_state`,
   - regime-scaled variants via RV/market/funding downscaling.
4. **Selection provenance:** record nested/train-validation-only selection metadata in replay and liquidation payloads; never rank on OOS.
5. **Replay:** run new non-calendar allowlist and freeze candidate from train/validation evidence.
6. **Liquidation validation:** evaluate 1x-6x; report highest strict zero-liquidation leverage and diagnostic nonfatal 5x/6x separately.
7. **Docs/handoff:** write notepad, plan update, session handoff, and report directory handoff.
8. **Verification/delivery:** run targeted tests, focused pytest, full pytest, ruff, compileall, diff check, Lore commit, push, and CI check.

## Available agent types roster
- `explore`: fast repo lookup/mapping.
- `executor`: implementation and refactoring.
- `test-engineer`: targeted regression coverage and verification strategy.
- `verifier`: completion evidence and acceptance audit.
- `architect`: final read-only design/sign-off.
- `critic`: plan/quality challenge.

## Team staffing guidance
- Recommended launch: `omx team 3:executor "Profit moonshot state-distilled non-calendar next pass: tests-first implementation, narrow market-state families, strict/diagnostic liquidation reporting, and verification artifacts. Keep files scoped to replay/liquidation tests/scripts/docs; no calendar entry rules; locked-OOS report-only."`
- Lane 1 delivery: fresh-start replay families and arrays.
- Lane 2 regression/evidence: tests and selection/liquidation provenance.
- Lane 3 docs/report handoff: artifact summaries and verification checklist.
- Team verification path: leader runs final targeted/focused/full verification and owns commit/push/CI.

## Ralph follow-up guidance
Use Ralph as the persistent single-owner loop after team execution to integrate worker changes, run the full verification suite, handle failures, and produce final commit/CI evidence.


## RALPLAN iteration 2 revisions from Architect review
- Added factory-wide valid-family allowlist guardrail: every new valid family generated by the spec factory must prove no calendar month/day/hour entry fields or calendar-veto fields are active.
- Added calendar-proxy robustness requirements: month placebo, blocked-time/date-block concentration diagnostics, calendar-label permutation/equivalent status, and rolling/walk-forward status in artifacts.
- Added future-holdout policy: repeated locked-OOS observation means any live-promotion claim must require a fresh future holdout before deployment; current locked-OOS remains gate/report only.
- Added explicit risk-adjusted thresholds: OOS MDD <= 25%; OOS Sharpe, Sortino, smart Sortino, Calmar >= 1.0; OOS return and return/MDD beat current-base reference.
- Revised staffing emphasis: team execution still uses the tmux runtime, but leader will keep lane 2 dedicated to regression/provenance evidence and final Ralph verification will include independent architect/verifier sign-off.

## Team staffing revision
- Launch remains a tmux team because the user requested `$team`, but assignments must split by lane:
  1. Replay/family implementation lane (`executor`).
  2. Tests/provenance/liquidation lane (`executor` acting from test-engineer checklist because team runtime has one worker role in this session).
  3. Docs/report/handoff lane (`executor` with verification checklist ownership).
- If a role-specific native verifier is needed after team completion, Ralph will invoke a separate `verifier`/`architect` sign-off before completion.

## Execution result — 2026-05-12 KST

- Implemented the chosen in-place path with tests-first guardrails and team/Ralph execution.
- Replay new market-state allowlist: `184` specs, `0` train/validation-positive candidates, `0` replay survivors, `0` success candidates, peak RSS `257.844 MiB`.
- Portfolio diagnostic on prior valid leadership-unwind top18: `56,203` portfolio specs, `0` success candidates, peak RSS `954.422 MiB`; OOS-discovered diagnostic-best row is explicitly report-only. Full candidate CSV is stored as gzip plus a top-200 CSV mirror.
- Liquidation-aware validation: `62` candidate seeds, `366` integer results, `0` deployable candidates. Train/validation-selected candidate at `2x` is strict-liquidation safe but fails OOS economic gates. Highest strict zero-liquidation selection target remains the prior `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` at `4x`; locked-OOS report `+2.4722%` / MDD `2.5328%` / return-MDD `0.9761` does not beat the reference `+6.4281%` / `6.9169`.
- Diagnostic high-leverage lane is non-promotional: train/validation-selected portfolio at `5x` has `10` total liquidations, at `6x` has `17`; no wipeout, but strict promotion blocked.
- Result: strategy-validity/provenance improved, but strict deployable improvement remains `false`. Next pass should diagnose activation/threshold coverage before any broader grid expansion.

## Execution result — 2026-05-12 KST external-risk teacher pass

- Added lagged external market-state ingestion from FRED via `scripts/research/fetch_profit_moonshot_external_state.py` and joined it into replay/liquidation via `--external-state-csv`.
- Added non-calendar families: `calendar_teacher_state_similarity`, `calendar_teacher_state_fade`, and `state_distilled_external_risk_filter`.
- Teacher similarity/fade were not useful: similarity `972` specs and fade `324` specs both produced `0` survivors and no train/validation-positive improvement.
- External-risk filtered state-distilled replay: `1,728` specs, `565` train/validation-positive, `0` replay survivors under the legacy shadow-MDD gate, peak RSS `280.348 MiB`.
- Best train/validation-positive OOS diagnostic row after freeze (report-only, not selector): `fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp750_fl0_xr200`, vector train `+3.2248%`, validation `+1.8146%`, locked-OOS `+1.4128%`, OOS MDD `0.5320%`, Sharpe `3.6699`.
- Liquidation-aware train/validation-selected strict row: `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` at `4x`, train `+30.9030%`, validation `+12.4704%`, locked-OOS `+2.4852%`, MDD `2.5328%`, Sharpe `1.5096`, liquidation `0`, min margin buffers positive, strategy-validity pass.
- Decision: no live promotion. The new valid external-risk line still does not beat invalid current-base/calendar reference OOS `+6.4281%` and return/MDD `6.9169`; 5x/6x remains diagnostic-only because train/validation liquidations appear.
