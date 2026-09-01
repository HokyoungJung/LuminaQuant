# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T12:57:56.012826Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is train/validation-stability primary; locked-OOS is report-only / gate-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.
- Locked-OOS gate label: `locked_oos_gate_only`.
- Diagnostic quarantine label: `diagnostic_not_promoted`.
- Current-base artifact: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`.
- Current-base status: `loaded`.
- Train/validation stability objective: `frozen_weighted_train_validation_score_v1` (current base `16.576134`).
- No-improvement lifecycle: `no_improvement_current_base_retained`.
- Stable-return floor: train, validation, and locked-OOS monthlyized return `>=2.00%`.
- Train buffer: post-leverage train monthlyized return `>=2.25%` and raw/unlevered train monthlyized return `>=1.00%`.
- Leverage policy: `train_val_monthly_return_budget` uses an integer train/validation-only grid; continuous floor-fitting leverage is diagnostic only.
- MDD budget: locked-OOS max drawdown `≤25.00%`.
- Quality floors: OOS Sharpe `≥2.0`, Sortino `≥3.0`, smart Sortino `≥3.0`, Calmar `≥1.0`.
- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>1.2181%`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_20260509/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `30`
- Portfolio specs evaluated: `73465`
- Combo cap per size: `6000`; skipped by size: `{'4': 21405}`
- Success candidates: `0`
- Peak RSS: `1319.586 MiB`

## Selected by validation

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600, fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all, fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus`
- train: `+44.6833%`
- val: `+39.4561%`
- locked OOS: `+7.9515%`, Sharpe `3.613869`, MDD `+1.9971%`
- monthlyized train/val/OOS: `+3.1264%` / `+18.7177%` / `+3.5707%`; smart Sortino `4.204933`
- raw monthlyized train/val: `+0.6004%` / `+3.3403%`; leverage `6.000000`
- promotion status: `diagnostic_not_promoted` / failed gates: `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450`
- train: `+34.3638%`
- val: `+54.0014%`
- locked OOS: `+17.4297%`, Sharpe `5.540649`, MDD `+2.9400%`
- monthlyized locked OOS: `+7.6456%`; smart Sortino `6.029717`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `train_val_monthly_return_budget` | +17.4297% | +2.9400% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | +17.2735% | +2.9120% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus` | `train_val_monthly_return_budget` | +17.2167% | +2.4197% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | +17.1953% | +2.8980% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `train_val_monthly_return_budget` | +17.1714% | +2.4735% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | +17.1172% | +2.8840% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `train_val_monthly_return_budget` | +17.1033% | +3.0139% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls560_ss120_tp600` | `train_val_monthly_return_budget` | +16.9610% | +2.8560% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls540_ss120_tp600` | `train_val_monthly_return_budget` | +16.8048% | +2.8279% | `raw_train_monthly_return_gte_1pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,train_val_stability_beats_current_base,oos_return_risk_beats_current_base` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600` | `train_val_monthly_return_budget` | +16.7216% | +2.4813% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` | `train_val_monthly_return_budget` | 6.000000 | False | +44.6833% | +39.4561% | +7.9515% | +3.5707% | +1.9971% | 3.613869 | 4.204933 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` | `train_val_monthly_return_budget` | 6.000000 | False | +44.6478% | +39.4561% | +7.8731% | +3.5361% | +1.9689% | 3.632677 | 4.236842 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | 6.000000 | False | +44.6301% | +39.4561% | +7.8338% | +3.5189% | +1.9548% | 3.642268 | 4.251173 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 6.000000 | False | +44.6123% | +39.4561% | +7.7946% | +3.5016% | +1.9407% | 3.651986 | 4.266963 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `train_val_monthly_return_budget` | 6.000000 | False | +44.5768% | +39.4561% | +7.7161% | +3.4671% | +1.9124% | 3.671812 | 4.301685 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `train_val_monthly_return_budget` | 6.000000 | False | +44.5413% | +39.4561% | +7.6377% | +3.4325% | +1.8841% | 3.692171 | 4.334662 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600` | `train_val_monthly_return_budget` | 6.000000 | False | +44.5235% | +39.4561% | +7.5985% | +3.4152% | +1.8700% | 3.702556 | 4.351448 | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` | `train_val_monthly_return_budget` | 6.000000 | False | +44.5135% | +39.4561% | +7.7659% | +3.4890% | +1.9693% | 3.583026 | 4.176022 | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` | `train_val_monthly_return_budget` | 6.000000 | False | +44.4286% | +39.4561% | +7.6731% | +3.4481% | +1.9553% | 3.567224 | 4.158560 | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 6.000000 | False | +44.3437% | +39.4561% | +7.5803% | +3.4072% | +1.9414% | 3.551162 | 4.143245 | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 11 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | 3.000000 | False | +40.8710% | +38.8617% | +8.0301% | +3.6052% | +1.7844% | 3.856526 | 4.165678 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 12 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | 3.000000 | False | +40.8621% | +38.8617% | +8.0105% | +3.5966% | +1.7772% | 3.862776 | 4.174084 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 13 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 3.000000 | False | +40.8532% | +38.8617% | +7.9909% | +3.5880% | +1.7701% | 3.869073 | 4.182546 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 14 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | 3.000000 | False | +40.8444% | +38.8617% | +7.9713% | +3.5794% | +1.7630% | 3.875419 | 4.191065 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 15 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 3.000000 | False | +40.8355% | +38.8617% | +7.9516% | +3.5707% | +1.7559% | 3.881813 | 4.199640 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
