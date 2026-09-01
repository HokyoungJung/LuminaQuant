# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T06:53:05.288517Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is validation-primary; locked-OOS is report-only / gate-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.
- Locked-OOS gate label: `locked_oos_gate_only`.
- Diagnostic quarantine label: `diagnostic_not_promoted`.
- Stable-return floor: train, validation, and locked-OOS monthlyized return `>=2.00%`.
- MDD budget: locked-OOS max drawdown `≤25.00%`.
- Quality floors: OOS Sharpe `≥2.0`, Sortino `≥3.0`, smart Sortino `≥3.0`, Calmar `≥1.0`.
- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>1.2181%`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `30`
- Portfolio specs evaluated: `73465`
- Combo cap per size: `6000`; skipped by size: `{'4': 21405}`
- Success candidates: `8`
- Peak RSS: `920.512 MiB`

## Selected by validation

- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450, fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600`
- train: `+26.8207%`
- val: `+49.0705%`
- locked OOS: `+12.6346%`, Sharpe `5.046915`, MDD `+2.7382%`
- monthlyized train/val/OOS: `+2.0000%` / `+22.8720%` / `+5.6072%`; smart Sortino `5.533180`
- promotion status: `diagnostic_not_promoted` / failed gates: `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion`

## Best success candidate

- Ranked by train/validation validation score after all locked-OOS gates pass.
- `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600, fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600`
- train: `+26.8207%`
- val: `+19.9713%`
- locked OOS: `+6.8582%`, Sharpe `5.653698`, MDD `+0.8198%`
- monthlyized train/val/OOS: `+2.0000%` / `+9.8490%` / `+3.0883%`; smart Sortino `7.153593`
- promotion status: `improved_success_candidate` / failed gates: ``

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
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `train_val_monthly_return_budget` | +13.6038% | +2.3647% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | +13.4851% | +2.3422% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | +13.4258% | +2.3309% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | +13.3664% | +2.3196% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus` | `train_val_monthly_return_budget` | +13.0935% | +2.0061% | `train_monthly_return_gte_2pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `train_val_monthly_return_budget` | +13.0483% | +2.1296% | `train_monthly_return_gte_2pct,train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | +12.9502% | +2.8283% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | +12.8871% | +2.8103% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | +12.8240% | +2.7923% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | +12.7609% | +2.7743% | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `train_val_monthly_return_budget` | 2.342733 | True | +26.8207% | +19.9713% | +6.8582% | +3.0883% | +0.8198% | 5.653698 | 7.153593 | `` |
| 2 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp400_asiaus` | `train_val_monthly_return_budget` | 3.587702 | True | +26.8207% | +20.1456% | +7.5279% | +3.3841% | +0.9671% | 5.625385 | 7.249358 | `` |
| 3 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `train_val_monthly_return_budget` | 2.339842 | True | +26.8207% | +19.9467% | +6.8859% | +3.1006% | +0.8234% | 5.652237 | 7.149394 | `` |
| 4 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600` | `train_val_monthly_return_budget` | 2.334081 | True | +26.8207% | +19.8975% | +6.9411% | +3.1250% | +0.8308% | 5.649335 | 7.141034 | `` |
| 5 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 2.328347 | True | +26.8207% | +19.8487% | +6.9960% | +3.1493% | +0.8380% | 5.646461 | 7.132727 | `` |
| 6 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `train_val_monthly_return_budget` | 2.325490 | True | +26.8207% | +19.8243% | +7.0234% | +3.1614% | +0.8417% | 5.645033 | 7.128593 | `` |
| 7 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `train_val_monthly_return_budget` | 2.322640 | True | +26.8207% | +19.8000% | +7.0507% | +3.1735% | +0.8453% | 5.643613 | 7.124473 | `` |
| 8 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` | `train_val_monthly_return_budget` | 3.510098 | True | +26.8207% | +19.6231% | +7.0238% | +3.1616% | +0.9492% | 5.386665 | 6.907750 | `` |
| 9 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 4.977446 | False | +26.8207% | +49.0705% | +12.6346% | +5.6072% | +2.7382% | 5.046915 | 5.533180 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 10 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 4.976801 | False | +26.8207% | +49.0641% | +12.6977% | +5.6344% | +2.7562% | 5.044211 | 5.524608 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 11 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | 4.976156 | False | +26.8207% | +49.0577% | +12.7609% | +5.6615% | +2.7743% | 5.041540 | 5.516093 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 12 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 4.975515 | False | +26.8207% | +49.0514% | +12.8240% | +5.6887% | +2.7923% | 5.038893 | 5.507618 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 13 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls590_ss120_tp600` | `train_val_monthly_return_budget` | 4.974870 | False | +26.8207% | +49.0451% | +12.8871% | +5.7158% | +2.8103% | 5.036286 | 5.499214 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 14 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp600` | `train_val_monthly_return_budget` | 4.974226 | False | +26.8207% | +49.0387% | +12.9502% | +5.7429% | +2.8283% | 5.033708 | 5.492176 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
| 15 | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp600` | `train_val_monthly_return_budget` | 4.872896 | False | +26.8207% | +48.6596% | +10.3221% | +4.6074% | +2.1289% | 4.477023 | 5.151872 | `train_sharpe_high,train_sortino_high,oos_return_risk_beats_current_champion` |
