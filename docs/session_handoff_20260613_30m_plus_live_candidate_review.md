# 2026-06-13 KST — 30m+ live-candidate review handoff

## Scope

- User intent: revisit the research completed a few days before the latest 1m scoreboard, keep only strategies using `30m` or higher bars, and identify the highest-return candidate that is practical enough for deployment work.
- Included bars/timeframes: `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`.
- Excluded: latest `1m` scoreboards and research-only low-timeframe artifacts.
- Main evidence families:
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_best_strategy_factory_20260601/best_strategy_final_recommendation_latest.json|md`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/existing_candidate_reuse_selector_20260609/existing_candidate_reuse_selector_latest.json|md`
  - `var/reports/strategy_factory/coverage_aware_30m_plus_20260610T123442Z/full_2634_run_chunk60_resample/strategy_factory_run_summary_latest.md`
  - Clean/new-alpha diagnostic summaries under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/*clean_new_alpha*`.

## Bottom line

There is **no real-money deployable strategy yet** in the recorded local research evidence. A repo-wide report audit found zero JSON artifacts with any of these true flags:

- `ready_for_real: true`
- `real_money_execution: true`
- `real_execution_allowed: true`

If the deployment meaning is narrowed to **paper/testnet/live-shadow candidate**, the highest-return 30m+ candidate is:

| Rank | Candidate | Status | OOS comp | Sharpe | Sortino | Hit | Max OOS MDD | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | paper/live-shadow challenger only | `+53.38%` | `2.07` | `15.31` | `5/10` | `18.80%` | Highest return. Uses train+validation-only fold selection, but the risk-capped rule was introduced after the research iteration, so it must be forward-shadowed before real money. |
| 2 | `cross_candidate_hybrid:hybrid_v3_5` | robust paper default | `+27.01%` | `1.24` | `6.33` | `5/10` | `13.72%` | The final recommendation keeps this as the robust full-run default until forward confirmation. |
| 3 | `profile_optuna:selected_train_validation_legal` | paper candidate | `+18.32%` | `0.80` | `2.07` | `5/10` | `18.80%` | Lower return and same max drawdown as the dynamic switch. |
| 4 | `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | defensive paper candidate | `+6.97%` | `0.66` | `1.84` | `4/10` | `7.32%` | Lower-risk fallback, not a highest-return choice. |

Operational decision: **do not send any of these to real-money execution yet**. For the next live-readiness branch, freeze the exact `dynamic_conviction_switch:t0.90_risk_capped_fallback` spec as the aggressive challenger and keep `cross_candidate_hybrid:hybrid_v3_5` as the conservative paper control/default.

## `dynamic_conviction_switch` / `cross_candidate_hybrid` details

Source: `best_strategy_final_recommendation_latest.json`.

- Timeframes in scope: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`.
- Cost assumption: `10bps` slippage/cost.
- Folds: `2025-09` through `2026-06`.
- Clean protocol: train + 2-month validation selection; locked OOS is report-only.
- Final recommendation fields:
  - `comp_winner`: `dynamic_conviction_switch:t0.90_risk_capped_fallback`
  - `robust_full_run_default`: `cross_candidate_hybrid:hybrid_v3_5`
  - `recommended_status`: `paper_shadow_challenger_only_for_dynamic_switch; robust default remains cross_candidate_hybrid:hybrid_v3_5 until forward confirmation`

Dynamic monthly selections were not one static leaf. They switched across strict-efficiency, cross-hybrid, and profile-optuna components using train+validation conviction:

| Fold | Selected component | Locked OOS | Locked MDD |
|---|---|---:|---:|
| 2025-09 | `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | `+0.14%` | `0.07%` |
| 2025-10 | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `-0.72%` | `1.39%` |
| 2025-11 | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `+12.20%` | `10.08%` |
| 2025-12 | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `-0.13%` | `1.63%` |
| 2026-01 | `cross_candidate_hybrid:hybrid_v3_5` | `+9.62%` | `3.92%` |
| 2026-02 | `profile_optuna:selected_optuna` | `+19.24%` | `18.80%` |
| 2026-03 | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `-2.65%` | `2.80%` |
| 2026-04 | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `-2.02%` | `4.57%` |
| 2026-05 | `profile_optuna:selected_optuna` | `+11.24%` | `16.08%` |
| 2026-06 | `profile_optuna:selected_optuna` | `-0.71%` | `1.04%` |

Risk notes to preserve:

1. The dynamic switch uses only train+validation inputs inside each fold, not current locked OOS.
2. The risk-capped fallback was introduced after the research iteration, so it is a forward-shadow challenger rather than real-money approval.
3. No candidate is positive every month; the dynamic switch wins by payoff asymmetry and fallback behavior, not by eliminating losses.

## Existing-candidate reuse selector

Source: `existing_candidate_reuse_selector_latest.json|md`.

The strongest variant is useful as a comparison candidate but is still post-failure research, not live approval:

| Variant | OOS comp | Approx annualized | Monthly MDD | Max OOS MDD | Hit | Sharpe approx | PF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `robust_balanced_v1_top1` | `+27.03%` | `+33.26%` | `2.72%` | `9.17%` | `7/10` | `2.23` | `7.92` |
| `robust_quality_v1_top1` | `+24.55%` | `+30.14%` | `3.10%` | n/a in this note | `7/10` | n/a in this note | `6.30` |
| `robust_top1` | `+22.14%` | `+27.12%` | `3.10%` | n/a in this note | `6/10` | n/a in this note | `4.95` |

All selected folds were `30m+` (`30m`, `1h`, `4h`). This makes `robust_balanced_v1_top1` a reasonable paper comparator, but it is explicitly marked as requiring fresh-forward confirmation.

## 30m+ Strategy Factory strict-pass run

Source: `strategy_factory_run_summary_latest.md`.

- Candidates evaluated: `2634`.
- Strict pass rows: `12`.
- Unique strict-pass parameterizations: `3`.
- Timeframes: `30m`, `1h`, `4h`, `1d`.

| Rank | Candidate | TF | Family | OOS return | Sharpe | DSR | PBO | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `pair_spread_4h_participation_btcusdt_bnbusdt_2.0_0.50` | `4h` | `market_neutral` | `+4.48%` | `2.409` | `0.007` | `0.250` | `14` |
| 2 | `pair_spread_1d_participation_btcusdt_ethusdt_1.5_0.33` | `1d` | `market_neutral` | `+2.97%` | `1.582` | `0.000` | `0.250` | `6` |
| 3 | `pair_spread_4h_participation_btcusdt_bnbusdt_1.8_0.45` | `4h` | `market_neutral` | `+3.14%` | `1.423` | `0.002` | `0.375` | `16` |

These are cleaner strict-pass ideas but not the highest-return branch.

## Clean/new-alpha diagnostics

The clean/new-alpha branch is valuable for avoiding OOS fitting, but the recorded 30m+ candidates are weaker or explicitly no-promotion:

- `indicator_kalman_ml_robust_selector_full_universe_20260609/clean_new_alpha_discovery_latest.md`: OOS comp `+22.14%`, annualized `+27.12%`, monthly equity MDD `3.10%`, max OOS MDD `9.89%`, hit `6/10`, Sharpe `1.70`; live plausibility `not_supported`; clean promotion eligible false.
- `codex_independent_flow_coverage_research_20260607/clean_new_alpha_core10_sparse_feature_asof_patch/clean_new_alpha_discovery_latest.md`: OOS comp `+7.69%`, annualized `+19.45%`, max OOS MDD `8.32%`, hit `3/5`, Sharpe `1.06`.
- `alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.md`: OOS comp `+6.86%`, annualized `+17.27%`, max OOS MDD `8.28%`, hit `3/5`, Sharpe `1.98`.
- `current_search_residual_only_overlay_v2_20260609/clean_new_alpha_discovery_latest.md`: OOS comp `+4.16%`, annualized `+17.73%`, monthly equity MDD `0.00%`, max OOS MDD `1.45%`, hit `3/3`, Sharpe `5.90`; live plausibility `not_supported`; clean promotion eligible false.

## Next gates before real-money deployment

1. Freeze challenger/control specs before observing any new unseen slice:
   - Challenger: `dynamic_conviction_switch:t0.90_risk_capped_fallback`.
   - Control/default: `cross_candidate_hybrid:hybrid_v3_5`.
2. Run fresh-forward paper/testnet/live-shadow with no selector changes.
3. Require 10/15/20bps cost stress and spread/slippage/fill telemetry.
4. Track reject/cancel/partial-fill/reconciliation gaps; fail closed on any unhandled execution discrepancy.
5. Re-check monthly loss tails: dynamic switch has `18.80%` max OOS MDD and two large profile-optuna-driven drawdown months.
6. Do not promote from `clean_promotion_eligible` alone. Promotion requires real execution telemetry plus fresh-forward evidence.

## Do-not-repeat notes

- Do not use the latest `1m` scoreboards for this decision; the user asked for the older 30m+ work.
- Do not present `dynamic_conviction_switch` as real-money ready; it is the top paper/shadow challenger.
- Do not keep mining the same locked OOS window to improve headline return.
- Do not synthesize missing feature history or fill telemetry.
