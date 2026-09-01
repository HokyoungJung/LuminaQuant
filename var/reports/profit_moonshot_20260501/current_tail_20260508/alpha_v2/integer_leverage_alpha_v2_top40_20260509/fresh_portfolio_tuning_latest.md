# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T13:20:58.562377Z`

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
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_top40_20260509/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `40`
- Portfolio specs evaluated: `158620`
- Combo cap per size: `12000`; skipped by size: `{'4': 79390}`
- Success candidates: `0`
- Peak RSS: `2523.523 MiB`

## Selected by validation

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600, fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0`
- train: `+38.4887%`
- val: `+24.3010%`
- locked OOS: `+6.7836%`, Sharpe `2.397948`, MDD `+3.9431%`
- monthlyized train/val/OOS: `+2.7510%` / `+11.8767%` / `+3.0554%`; smart Sortino `2.721914`
- raw monthlyized train/val: `+1.0111%` / `+4.1002%`; leverage `3.000000`
- promotion status: `diagnostic_not_promoted` / failed gates: `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600`
- train: `+65.0477%`
- val: `+48.1020%`
- locked OOS: `+18.4446%`, Sharpe `5.822772`, MDD `+2.2305%`
- monthlyized locked OOS: `+8.0712%`; smart Sortino `6.808936`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | +18.4446% | +2.2305% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss100_tp600` | `train_val_monthly_return_budget` | +18.2590% | +2.2084% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss100_tp600` | `train_val_monthly_return_budget` | +18.1662% | +2.1974% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss100_tp600` | `train_val_monthly_return_budget` | +18.0735% | +2.1864% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | +17.9496% | +2.1562% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp400_asiaus__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | +17.8725% | +2.2305% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h168_ls530_ss120_tp450__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0` | `train_val_monthly_return_budget` | +17.8449% | +3.3961% | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss100_tp600` | `train_val_monthly_return_budget` | +17.7640% | +2.1350% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss100_tp600` | `train_val_monthly_return_budget` | +17.7640% | +2.1350% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | +17.7640% | +2.1350% | `raw_train_monthly_return_gte_1pct,train_sortino_high,oos_return_risk_beats_current_base` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4887% | +24.3010% | +6.7836% | +3.0554% | +3.9431% | 2.397948 | 2.721914 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4354% | +24.3010% | +6.6659% | +3.0033% | +3.9062% | 2.372916 | 2.695040 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4709% | +24.3010% | +6.7444% | +3.0380% | +3.9308% | 2.389715 | 2.713599 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4176% | +24.3010% | +6.6267% | +2.9859% | +3.8938% | 2.364348 | 2.686255 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4621% | +24.3010% | +6.7248% | +3.0293% | +3.9246% | 2.385557 | 2.709650 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4532% | +24.3010% | +6.7052% | +3.0206% | +3.9185% | 2.381371 | 2.705157 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.4087% | +24.3010% | +6.6071% | +2.9772% | +3.8877% | 2.360022 | 2.681563 | `oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls530_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0493% | +23.9193% | +7.6331% | +3.4305% | +3.3191% | 2.731998 | 3.212523 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls540_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0514% | +23.9193% | +7.6722% | +3.4477% | +3.3146% | 2.742994 | 3.226481 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls560_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0556% | +23.9193% | +7.7503% | +3.4821% | +3.3056% | 2.764852 | 3.255523 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 11 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0598% | +23.9193% | +7.8284% | +3.5165% | +3.3063% | 2.786533 | 3.284205 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 12 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0619% | +23.9193% | +7.8674% | +3.5337% | +3.3103% | 2.797307 | 3.297272 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 13 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0640% | +23.9193% | +7.9065% | +3.5509% | +3.3142% | 2.808037 | 3.310280 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 14 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 3.000000 | False | +38.0682% | +23.9193% | +7.9846% | +3.5852% | +3.3221% | 2.829361 | 3.334211 | `oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 15 | `fresh_portfolio_train_val_monthly_return_budget_fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0` | `train_val_monthly_return_budget` | 5.000000 | False | +52.7685% | +23.2838% | +9.6565% | +4.3176% | +6.4950% | 2.237887 | 2.236403 | `raw_train_monthly_return_gte_1pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high,oos_return_risk_beats_current_base` |
