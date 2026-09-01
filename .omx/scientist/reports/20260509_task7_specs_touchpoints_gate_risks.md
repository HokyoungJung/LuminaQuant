# Task 7 — scripts/data/tests inspection and minimal specs

[OBJECTIVE] Inspect the current LuminaQuant profit-moonshot scripts, data artifacts, and tests; propose minimal implementable specs, expected touchpoints, and gate risks for the leader audit.

[DATA] Inspected core scripts/data surfaces: `scripts/research/replay_profit_moonshot_fresh_start.py`, `scripts/research/tune_profit_moonshot_fresh_portfolio.py`, `scripts/research/optuna_tune_profit_moonshot_calendar.py`, `scripts/collect_all_strategy_support_data.py`, `scripts/build_strategy_support_inventory.py`, `scripts/backfill_funding_fee_features.py`, `src/lumina_quant/data/support_inventory.py`, `src/lumina_quant/data/feature_points.py`.
[STAT:n] n=8 core script/data files, n=4 focused test files, n=102 focused `def test_` tests across `test_research_runner_feature_support.py` (74), `test_profit_moonshot_fresh_start_replay.py` (13), `test_profit_moonshot_fresh_portfolio_tuning.py` (8), and `test_strategy_support_collection_profiles.py` (7).
[STAT:n] Artifact base: all-family replay 6,805 specs / 300 successes; portfolio tuning 58,224 specs / 6,129 successes; funding-related replay 96 specs / 0 successes; H1/H2 improved candidates 0; H3/H4/H5 replay survivors 0 and portfolio improved candidates 0.

[FINDING] The lowest-risk implementation lane is incremental extension of existing replay/tuning/support-inventory surfaces, not a new subsystem.
[STAT:effect_size] Existing scripts already expose the required hooks: `FreshSpec` funding/OI/calendar-veto fields, `_candidate_signal` funding branches, `_calendar_veto_reason`, `_candidate_specs`, portfolio `LOCKBOX_POLICY`, memory guards, and support-inventory OI/funding counts.
[STAT:n] Touchpoint count for a minimal report-only/gating implementation is 3-5 files plus 2-4 focused tests, compared with 8 inspected core files; no new dependency is required.

[FINDING] Data-readiness gates are mandatory before any funding/OI regime transition can be promotion-eligible.
[STAT:effect_size] Current support inventory has 81,052 true-OI rows but first true OI starts `2026-03-07`/`2026-03-08`, after train (`2025`) and validation (`2026-01..2026-02`); liquidation rows total 0.
[STAT:n] n=5 symbols; BNB/TRX taker-flow rows are 0 in replay metadata, so support-data prechecks should explicitly label missing feature columns instead of allowing silent proxy promotion.

[FINDING] Existing artifact evidence points to calendar-conditioned gates/diagnostics rather than standalone new alphas.
[STAT:effect_size] Funding-related standalone replay success rate is 0/96 = 0.0%; calendar rotation success rate is 300/4,392 = 6.83%; H3/H4/H5 calendar-conditioned replay found 0/80 survivors and 0 improved portfolio candidates.
[STAT:n] H1/H2 bounded portfolio run evaluated 20,145 specs and found 0 improved/promoted candidates; all high-return MDD-failed rows remained `diagnostic_not_promoted`.

[FINDING] Test coverage is present but should be tightened around missing-data and promotion-boundary regressions before edits.
[STAT:n] Existing focused tests cover funding/OI signal shape, calendar veto/day-window behavior, calendar spread, cache reuse, memory guard, diagnostic quarantine, support-data profile flags, liquidation endpoint fallback, and OI chunking across 102 inspected tests.
[STAT:effect_size] Recommended missing checks: 5 small regressions — missing true-OI makes funding/OI promotion-ineligible; support inventory fails/labels train-val OI absence; regime-veto cannot use locked-OOS for selection; BNB/TRX missing taker-flow stays neutral; trip-starvation from veto filters fails promotion.

## Minimal implementable specs

1. `funding_oi_regime_transition_gate`
   - Type: diagnostic/calendar-veto gate, not standalone promotion.
   - Touchpoints: `replay_profit_moonshot_fresh_start.py` (`FreshSpec`, `_candidate_signal`, `_candidate_specs`); `support_inventory.py`; `test_profit_moonshot_fresh_start_replay.py`.
   - Gate: promotion-ineligible unless true OI exists in train and validation; locked-OOS remains report-only.

2. `calendar_veto_regime_label_extension`
   - Type: small H3-style veto extension using existing `_calendar_veto_reason`.
   - Touchpoints: `FreshSpec` veto fields, H3 candidate grid, calendar veto tests.
   - Gate: must retain train/validation return while reducing MDD; reject trip-starved variants.

3. `support_inventory_promotion_precheck`
   - Type: preflight/report guard.
   - Touchpoints: `collect_all_strategy_support_data.py`, `build_strategy_support_inventory.py`, `support_inventory.py`, `test_strategy_support_collection_profiles.py`.
   - Gate: fail or mark diagnostic-only when true-OI/funding/taker/liquidation coverage is absent in required splits.

4. `diagnostic_quarantine_regression`
   - Type: safety test/report lock.
   - Touchpoints: `tune_profit_moonshot_fresh_portfolio.py` lockbox labeling and `test_profit_moonshot_fresh_portfolio_tuning.py`.
   - Gate: OOS-ranked or MDD-failed diagnostics must remain `diagnostic_not_promoted`; selection basis stays train/validation-only.

[LIMITATION] This was a read/synthesis task; no new strategy code or heavy backtest was launched.
[LIMITATION] The report relies on existing artifacts generated before this worker run; if new data is backfilled, support coverage and pass rates must be recomputed.
[LIMITATION] Subagent probe was skipped because the Scientist role instructions require working alone; the skip is recorded for delegation-compliance evidence.

Artifacts:
- Summary JSON: `.omx/scientist/data/task7_inspection_summary.json`
- Figure: `.omx/scientist/figures/task7_test_coverage_touchpoints.svg`
