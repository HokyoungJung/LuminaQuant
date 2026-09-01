# Profit moonshot fresh portfolio tuning

Generated: `2026-05-12T11:32:36.885337Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is train/validation-stability primary; locked-OOS is report-only / gate-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.
- Locked-OOS gate label: `locked_oos_gate_only`.
- Diagnostic quarantine label: `diagnostic_not_promoted`.
- Current-base artifact: ``.
- Current-base status: `disabled`.
- Train/validation stability objective: `frozen_weighted_train_validation_score_v1` (current base `16.576134`).
- No-improvement lifecycle: `current_base_unavailable`.
- Stable-return floor: train, validation, and locked-OOS monthlyized return `>=2.00%`.
- Train buffer: post-leverage train monthlyized return `>=2.25%` and raw/unlevered train monthlyized return `>=1.00%`.
- Leverage policy: `train_val_monthly_return_budget` uses an integer train/validation-only grid; continuous floor-fitting leverage is diagnostic only.
- MDD budget: locked-OOS max drawdown `≤25.00%`.
- Quality floors: OOS Sharpe `≥2.0`, Sortino `≥3.0`, smart Sortino `≥3.0`, Calmar `≥1.0`.
- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>1.2181%`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_market_state_next_20260512/portfolio_tuning_leadership_unwind_top18/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `18`
- Portfolio specs evaluated: `56203`
- Combo cap per size: `4000`; skipped by size: `{'5': 4568}`
- Success candidates: `0`
- Peak RSS: `954.422 MiB`

## Selected by validation

- `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240`
- sleeves: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600, fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600, fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450, fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240`
- train: `+64.1052%`
- val: `+26.6771%`
- locked OOS: `+4.6759%`, Sharpe `1.256786`, MDD `+6.9705%`
- monthlyized train/val/OOS: `+4.2147%` / `+12.9749%` / `+2.1176%`; smart Sortino `1.191980`
- raw monthlyized train/val: `+1.6271%` / `+4.4929%`; leverage `3.000000`
- promotion status: `diagnostic_not_promoted` / failed gates: `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450`
- train: `+76.5899%`
- val: `+30.7065%`
- locked OOS: `+8.2808%`, Sharpe `1.870239`, MDD `+7.1067%`
- monthlyized locked OOS: `+3.7154%`; smart Sortino `1.742528`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | +8.2808% | +7.1067% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | +8.2808% | +7.1067% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | +8.1772% | +7.1022% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +7.1680% | +5.6502% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +7.1680% | +5.6502% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +7.1650% | +5.8687% | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +7.0273% | +5.8179% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +7.0273% | +5.8179% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +7.0209% | +5.8179% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | +6.9075% | +5.9476% | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +64.1052% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256786 | 1.191980 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret60_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +64.1051% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256786 | 1.191980 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +63.6294% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256585 | 1.192387 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret60_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +63.6293% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256585 | 1.192387 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +63.6130% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256786 | 1.191980 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret60_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +63.6129% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256786 | 1.191980 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +63.6294% | +26.6043% | +4.5723% | +2.0712% | +6.9659% | 1.234148 | 1.170175 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret60_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +63.6293% | +26.6043% | +4.5723% | +2.0712% | +6.9659% | 1.234148 | 1.170175 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +63.1372% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256585 | 1.192387 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret60_h120_ls590_ss100_tp240` | `train_val_monthly_return_budget` | 3.000000 | False | +63.1371% | +26.6771% | +4.6759% | +2.1176% | +6.9705% | 1.256585 | 1.192387 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 11 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +63.1372% | +26.6043% | +4.5723% | +2.0712% | +6.9659% | 1.234148 | 1.170175 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 12 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret60_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret180_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +63.1371% | +26.6043% | +4.5723% | +2.0712% | +6.9659% | 1.234148 | 1.170175 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 13 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret180_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +64.2166% | +26.9767% | +4.6759% | +2.1176% | +6.9705% | 1.256786 | 1.191980 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 14 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret180_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret120_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +63.7408% | +26.9767% | +4.6759% | +2.1176% | +6.9705% | 1.256585 | 1.192387 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
| 15 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z050_ret180_h120_ls590_ss100_tp240__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450` | `train_val_monthly_return_budget` | 3.000000 | False | +63.7244% | +26.9767% | +4.6759% | +2.1176% | +6.9705% | 1.256786 | 1.191980 | `oos_return_risk_beats_current_champion,oos_sharpe_high,oos_sortino_high,oos_smart_sortino_high` |
