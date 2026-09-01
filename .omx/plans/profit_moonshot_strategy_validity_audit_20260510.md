# RALPLAN-DR: Profit Moonshot Strategy Validity Audit, Retune, and Live Re-selection

## Outcome
Invalidate theoretically defective/live-inapplicable profit-moonshot rules, especially fixed calendar+fixed asset seasonality, then rerun candidate selection under practical live gates and produce a final deploy/no-deploy recommendation with artifacts, tests, commit/push, and CI evidence.

## RALPLAN-DR Summary

### Principles
1. **Alpha thesis before metrics:** performance cannot promote a rule whose live premise is data-mined or not economically/mechanistically defensible.
2. **No fixed calendar alpha without robust external evidence:** hard-coded month+asset rules are rejected/quarantined; calendar may only be allowed as a non-primary risk filter after explicit tests.
3. **Live deployability gates compose:** strategy-validity, integer leverage, liquidation/margin, locked-OOS firewall, memory, and operational tests must all pass.
4. **Selection remains train/validation-only:** locked-OOS is never used for tuning/ranking/tie-breaks, only report/gate evidence.
5. **No forced winner:** if all high-return rows are invalid, final output must say no live promotion and preserve/retain baseline.

### Decision Drivers
1. **Theoretical defect risk:** fixed TRX/ETH calendar rule passed because the strategy generator offered that rule, not because live-valid causal edge was proven.
2. **Ranking instability:** removing invalid families may reorder every candidate/portfolio/hybrid result, so all used artifacts must be re-audited.
3. **Production safety:** outputs must be reproducible, under 8 GiB, tested, committed, pushed, and CI-green before live handoff.

### Options

#### Option A — Hard validity gate + retune from existing dynamic candidates (recommended)
- Add an auditable strategy-validity classifier/gate.
- Reject calendar/fixed-symbol seasonality rows.
- Rebuild final selection and retune/reselect among remaining dynamic state-based rows already generated.
- Pros: fastest safe correction, minimal new overfit surface, directly addresses defect.
- Cons: may produce no deployable improvement if dynamic rows underperform.

#### Option B — Re-open broad alpha generation with calendar banned
- Generate new families and wider search excluding calendar.
- Pros: higher chance of finding a replacement winner.
- Cons: much higher data-mining risk, longer runtime, needs separate research validation.
- Invalidation: unsuitable for immediate live handoff without additional multi-period robustness.

#### Option C — Keep calendar row with warning only
- Pros: preserves high return.
- Cons: violates user concern and live-theory gate; unacceptable.
- Invalidation: rejected.

## ADR
- **Decision:** Implement Option A now. Treat calendar/fixed month+symbol alpha as promotion-blocking. Retune/reselect from existing dynamic, state-based candidates and report no-promotion if no row passes all gates.
- **Drivers:** live theoretical validity, leakage safety, memory-bound reproducibility, minimal new overfit.
- **Alternatives considered:** broad new alpha (deferred), warning-only calendar (rejected).
- **Consequences:** prior 5x winner will likely be rejected; final ranking may change or no candidate may be promoted.
- **Follow-ups:** if no deployable candidate remains, open a separate research plan for dynamic-only alpha expansion with walk-forward/year-by-year validation.

## Execution Plan

### Phase 1 — Audit/gate design
- Add a reusable audit script/module that loads fresh specs and latest artifacts.
- Candidate-level validity gate = pass only if every active sleeve has a defensible dynamic/state-based primary signal.
- Reject reasons include:
  - `calendar_fixed_month_alpha`,
  - `fixed_asset_calendar_target`,
  - `seasonality_without_robustness_evidence`,
  - `live_thesis_invalid_or_missing`.
- Explicitly mark the current calendar TRX/ETH winner as not deployable.

### Phase 2 — Wire final selection
- Extend `write_profit_moonshot_live_final_selection.py` with `strategy_validity` and `live_strategy_thesis` decision gates.
- `deployable_candidate` requires all existing gates plus strategy-validity pass.
- Markdown/JSON must show validity reasons for all rows.

### Phase 3 — Retune/reselect after invalidation
- Rebuild/audit existing candidate sources and filter out invalid calendar/fixed-seasonality rows.
- Re-run final selection using latest artifacts; if existing dynamic candidates already present in the current artifacts/CSV exist, evaluate integer leverage/liquidation/performance gates.
- Do **not** run new alpha generation or broad dynamic search in this handoff. If no existing dynamic-only candidate passes, emit `no_live_promotion` and open a separate research follow-up.

### Phase 4 — Reporting/handoff
- Write artifacts under:
  - `var/reports/profit_moonshot_20260501/live_final_selection_20260510/strategy_validity_audit/`
  - updated `final_decision/`
  - `.omx/notepad.md`
  - `.omx/plans/`
  - `docs/session_handoff_20260510_profit_moonshot_strategy_validity_audit.md`
- Include old vs new ranking, rejected rows, remaining candidates, and final deploy/no-deploy recommendation.

### Phase 5 — Verification and push
- Tests first for gate behavior, then implementation.
- Targeted tests:
  - strategy audit tests,
  - live final selection tests,
  - candidate hybrid/source leverage tests,
  - liquidation validation tests.
- Full verification: full pytest, ruff, compileall, git diff --check.
- Commit with Lore protocol, push to `private/main`, verify `ci` and `private-ci` green.

## Acceptance Criteria
- Calendar TRX/ETH current winner cannot be promoted.
- Every final row has strategy-validity metadata and rejection reasons where applicable.
- All final deployable rows, if any, pass strategy-validity, integer leverage, liquidation/margin, OOS MDD, return/R-MDD, and quality gates.
- If no row passes, recommendation is explicitly `no_live_promotion` / retain baseline.
- Memory evidence remains <8 GiB.
- Tests/lint/compileall/diff-check pass; push and CI/private-ci green.

## Available agent-types roster
- `executor`: code/tests/artifacts implementation.
- `test-engineer`: regression and gate tests.
- `architect`: strategy-validity design review and antithesis.
- `critic`: final plan/report adequacy review.
- `verifier`: evidence and CI validation.
- `git-master`: Lore commit/push hygiene.
- `explore`: fast artifact/code mapping.

## Team staffing guidance
- `worker-1/executor`: implement audit module + final-selection gate.
- `worker-2/test-engineer`: tests/fixtures for invalid calendar gate and dynamic pass cases.
- Leader: integrate, run heavy retune sequentially, final verification, commit/push/CI.
- Use only one heavy backtest/full pytest at a time under 8 GiB.

## Ralph guidance
Ralph should persist through failing tests/retune surprises until: artifacts exist, final recommendation is evidence-backed, verification is green, commit pushed, and CI/private-ci green.

## Goal-mode follow-up suggestion
This is an optimization/live-performance task; if converted to goal-mode, use `$performance-goal` rather than generic `$ultragoal`.

## Architect amendment accepted — source-aware validity taxonomy

Do **not** implement a blanket ban on any time/calendar/session field. Implement source-aware classification:

- **Invalid primary alpha by default:** `calendar_rotation` / `calendar_spread` where the primary entry is fixed month set plus fixed long/short asset identity, e.g. TRX long in Mar-May or ETH short in Jan-Feb, absent separate robust external/mechanistic evidence and out-of-family robustness tests.
- **Allowed if otherwise passing:** dynamic primary signals based on observable market state such as funding, open-interest, flow, residual z-score, trend/adaptive trend, volatility/compression, cross-sectional rank, or residual pair spread. Time/session/day filters may be allowed only as secondary risk/cadence filters, not primary alpha.
- **Required row metadata:** `strategy_validity.pass`, `primary_signal_type`, `primary_signal_evidence`, `rejection_reasons`, `audited_sleeves`, and `audit_sources`.
- **Audit closure:** include final selection JSON/MD, liquidation validation, candidate portfolio, candidate hybrid, merged candidate CSV, current-base/passing artifacts, and source rows feeding each row kind.

## Critic fixes — execution boundary hardening

1. **No broad new-alpha expansion in this handoff.** Remove/ignore any Phase 3 wording that suggests broad expansion. If no existing dynamic-only candidate passes all gates after invalidation, emit `no_live_promotion` / retain baseline and open a separate research plan. Do not invent a new search to force a winner.
2. **Calendar-primary taxonomy clarified.** Any `calendar_rotation` or `calendar_spread` where fixed month/window is the primary entry alpha is invalid for live promotion by default, whether the asset target is fixed or dynamically selected, unless a separate robust external/mechanistic thesis and out-of-family robustness suite is provided. Time/session/day filters are allowed only when the primary signal is non-calendar dynamic market state.
3. **Audit closure manifest required.** The audit artifact must list every audited source with path, artifact kind, row count, and source role, including final selection JSON/MD, liquidation validation, candidate portfolio, candidate hybrid, merged candidate CSV, current-base/passing artifacts, and every traced row/sleeve source used per final row kind.
4. **Closure/non-overblocking tests required.** Tests must cover closure manifest completeness; dynamic residual/funding/flow/cross-sectional/adaptive specs with optional secondary session filters pass; all calendar-primary specs fail without robust evidence.
5. **Final-selection wiring is fail-closed.** Final selection must either load an explicit strategy-validity audit artifact or run inline audit. Every row must contain `strategy_validity.pass`, `primary_signal_type`, `primary_signal_evidence`, `audited_sleeves`, and `audit_sources`; missing validity metadata blocks promotion.
