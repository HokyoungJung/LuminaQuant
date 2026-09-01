# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T08:17:26.072938Z`

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
- MDD budget: locked-OOS max drawdown `≤25.00%`.
- Quality floors: OOS Sharpe `≥2.0`, Sortino `≥3.0`, smart Sortino `≥3.0`, Calmar `≥1.0`.
- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>1.2181%`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/continued_optimization_20260509/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `30`
- Portfolio specs evaluated: `73465`
- Combo cap per size: `6000`; skipped by size: `{'4': 21405}`
- Success candidates: `0`
- Peak RSS: `1239.703 MiB`

## Selected by validation

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0`
- train: `+26.8207%`
- val: `+21.4285%`
- locked OOS: `+8.0880%`, Sharpe `4.008758`, MDD `+2.1151%`
- monthlyized train/val/OOS: `+2.0000%` / `+10.5353%` / `+3.6307%`; smart Sortino `5.122452`
- promotion status: `diagnostic_not_promoted` / failed gates: `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450`
- train: `+26.8207%`
- val: `+42.1478%`
- locked OOS: `+13.6038%`, Sharpe `5.504238`, MDD `+2.3647%`
- monthlyized locked OOS: `+6.0230%`; smart Sortino `6.148208`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `train_val_monthly_return_budget` | +13.6038% | +2.3647% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | +13.4851% | +2.3422% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | +13.4258% | +2.3309% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | +13.3664% | +2.3196% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls560_ss120_tp600` | `train_val_monthly_return_budget` | +13.2477% | +2.2970% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls540_ss120_tp600` | `train_val_monthly_return_budget` | +13.1289% | +2.2744% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls530_ss120_tp600` | `train_val_monthly_return_budget` | +13.0695% | +2.2631% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | +12.9502% | +2.8283% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | +12.8871% | +2.8103% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | +12.8240% | +2.7923% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.974212 | False | +26.8207% | +21.4285% | +8.0880% | +3.6307% | +2.1151% | 4.008758 | 5.122452 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_risk_beats_current_base` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.575239 | False | +26.8207% | +21.7126% | +6.0844% | +2.7454% | +1.7688% | 3.824188 | 4.898142 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.574507 | False | +26.8207% | +21.7064% | +6.0995% | +2.7521% | +1.7744% | 3.821833 | 4.892396 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.573043 | False | +26.8207% | +21.6941% | +6.1297% | +2.7655% | +1.7856% | 3.817044 | 4.881240 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.571583 | False | +26.8207% | +21.6817% | +6.1598% | +2.7789% | +1.7968% | 3.812155 | 4.870448 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.759667 | False | +26.8207% | +19.9815% | +6.0466% | +2.7286% | +1.9726% | 3.319359 | 4.184032 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.570853 | False | +26.8207% | +21.6756% | +6.1749% | +2.7856% | +1.8024% | 3.809675 | 4.864461 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.757843 | False | +26.8207% | +19.9608% | +6.0185% | +2.7161% | +1.9532% | 3.340159 | 4.216135 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.570124 | False | +26.8207% | +21.6694% | +6.1899% | +2.7922% | +1.8080% | 3.807172 | 4.858413 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.757501 | False | +26.8207% | +19.9569% | +6.0288% | +2.7207% | +1.9557% | 3.341265 | 4.217089 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 11 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.758035 | False | +26.8207% | +19.9629% | +6.0682% | +2.7382% | +1.9737% | 3.329394 | 4.197108 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 12 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.756819 | False | +26.8207% | +19.9491% | +6.0494% | +2.7299% | +1.9607% | 3.343411 | 4.219335 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 13 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600` | `train_val_monthly_return_budget` | 2.568669 | False | +26.8207% | +21.6572% | +6.2200% | +2.8056% | +1.8192% | 3.802099 | 4.846144 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 14 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.756138 | False | +26.8207% | +19.9414% | +6.0700% | +2.7390% | +1.9657% | 3.345468 | 4.221023 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |
| 15 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0` | `train_val_monthly_return_budget` | 1.755798 | False | +26.8207% | +19.9375% | +6.0803% | +2.7436% | +1.9682% | 3.346464 | 4.222314 | `train_sortino_high,oos_return_risk_beats_current_champion,oos_return_beats_current_base,oos_return_risk_beats_current_base` |

