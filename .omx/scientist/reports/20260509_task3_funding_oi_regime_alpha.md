# Task 3 — funding/OI regime transition alpha

[OBJECTIVE] Evaluate whether a funding/open-interest regime-transition alpha is implementable and promotable for the profit-moonshot pipeline, using existing LuminaQuant artifacts only and preserving locked-OOS/report-only policy.

[DATA] Sources inspected: `fresh_start_overhaul_replay_latest.json`, `fresh_start_overhaul_replay_candidates.csv`, `passing_candidate_latest.json`, derivatives support inventory, and derivatives-flow squeeze candidate report.
[STAT:n] Replay sample: 6,805 specs over 11,769 joined panel rows, 5 symbols, split as train `2025-01-01..2025-12-31`, validation `2026-01-01..2026-02-28`, locked-OOS/report-only `2026-03-01..2026-05-08`.
[STAT:n] Feature coverage in replay metadata: 7,389 funding rows, 89,647 open-interest rows, 6,303,687 taker-flow rows; BNB/TRX taker-flow rows are 0.
[STAT:n] Support inventory confirms 8,539 funding rows and 81,052 open-interest rows across 5 symbols, with true OI first appearing only `2026-03-07`/`2026-03-08`; liquidation rows total 0.

[FINDING] Standalone funding/OI alpha is not promotable from the current evidence set because the true-OI signal starts after the train and validation windows.
[STAT:effect_size] True-OI train/validation coverage for the declared split is effectively 0 hours before validation end `2026-02-28`; OI begins `2026-03-07`/`2026-03-08`, inside locked-OOS only.
[STAT:n] n=5 symbols; total current true-OI rows=81,052 support-inventory rows / 89,647 replay feature rows, but these rows are not usable for train/validation selection without leakage or proxy substitution.

[FINDING] Existing funding-related replay families have zero pass rate under current gates.
[STAT:effect_size] Funding-related replay success rate = 0/96 = 0.0% across `funding_oi_carry_fade` (0/24), `funding_carry_fade` (0/36), and `funding_carry_momentum` (0/36).
[STAT:n] By comparison, `calendar_rotation` produced 300/4,392 successes (6.83%) in the same candidate CSV; all 300 replay successes were calendar-family.

[FINDING] The current `funding_oi_carry_fade` implementation is signal-starved for selection and loses in locked-OOS when it finally trades.
[STAT:effect_size] Best funding+OI row `fresh_funding_oi_fade_lb6_z125_f100_oi0`: train return 0.0000%, validation return 0.0000%, locked-OOS return -0.0921%, OOS Sharpe -4.0353, OOS MDD 0.1270%, 33 OOS round trips.
[STAT:n] n=24 funding+OI specs; survivor_count=0, success_count=0.

[FINDING] Prior derivatives-flow/OI proxy work is negative evidence for proxy-only promotion.
[STAT:effect_size] Latest-data v6 derivatives-flow run: train return -9.0995%, validation return -0.5571%, train Sharpe -0.0210, validation Sharpe -0.0121.
[STAT:n] n=9,382 train trades and 1,522 validation trades; selection_eligible=false; blocking reasons include train return below floor and validation return/sharpe/sortino not positive.

[FINDING] Minimal implementable spec should be a diagnostic/gating alpha, not a standalone promoted sleeve, until true OI is backfilled into train/validation.
[STAT:effect_size] The gap to current champion is material: current best passing portfolio locked-OOS return is +1.2181% with OOS MDD 0.1662% and Sharpe 6.7264, while the best funding+OI standalone row is -0.0921% OOS with Sharpe -4.0353.
[STAT:n] Candidate universe for the next safe test: use existing 5-symbol panel and require nonzero true-OI coverage in train and validation before promotion eligibility; otherwise emit diagnostics only.

## Minimal alpha spec

- Name: `funding_oi_regime_transition_gate`.
- Role: calendar-sleeve veto / diagnostic regime label, not standalone entry authority.
- Inputs: timestamp-aligned funding rate, true open interest, price return, realized volatility; liquidation/taker flow optional and reported as missing if unavailable.
- Regime labels:
  - `crowded_long_build`: positive funding above threshold and positive OI delta/z.
  - `crowded_short_build`: negative funding below threshold and positive OI delta/z.
  - `deleveraging_transition`: OI contraction after crowded build, especially with adverse price shock.
  - `neutral_or_missing`: missing true OI/funding or insufficient lookback.
- Entry use: veto calendar long/short entries when funding sign plus OI expansion conflicts with the proposed leg; allow diagnostics to score whether veto reduces MDD while retaining train/validation return.
- Promotion gate: no promotion until true OI exists in train and validation; locked-OOS remains report-only/gate-only.

[LIMITATION] This task did not launch a new heavy backtest; it synthesized existing committed artifacts to avoid duplicate heavy runs under the global <8 GiB memory guard.
[LIMITATION] Current OI rows begin after train/validation, so any OI alpha selected now would either leak OOS information or depend on proxy features that already showed negative train/validation evidence.
[LIMITATION] Liquidation rows are 0 in the support inventory; liquidation-exhaustion logic should remain disabled or diagnostic until data is materialized.

Artifacts:
- Stats JSON: `.omx/scientist/data/task3_funding_oi_regime_stats.json`
- Figure: `.omx/scientist/figures/task3_funding_oi_success_by_family.svg`
