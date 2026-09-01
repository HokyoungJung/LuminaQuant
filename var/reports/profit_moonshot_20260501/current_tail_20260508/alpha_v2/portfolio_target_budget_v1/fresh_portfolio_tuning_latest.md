# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T05:45:14.258695Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is validation-primary; locked-OOS is report-only / gate-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.
- Locked-OOS gate label: `locked_oos_gate_only`.
- Diagnostic quarantine label: `diagnostic_not_promoted`.
- Improved threshold: OOS return `>1.2181%` plus current-champion return/risk gates.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `30`
- Portfolio specs evaluated: `62970`
- Combo cap per size: `6000`; skipped by size: `{'4': 21405}`
- Success candidates: `38`
- Peak RSS: `746.188 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- train: `+11.7292%`
- val: `+13.5530%`
- locked OOS: `+2.1374%`, Sharpe `3.022856`, MDD `+0.6571%`
- promotion status: `diagnostic_not_promoted` / failed gates: `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow`

## Best success candidate

- Ranked by train/validation validation score after all locked-OOS gates pass.
- `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600`
- train: `+5.5019%`
- val: `+4.0000%`
- locked OOS: `+1.3397%`, Sharpe `5.477385`, MDD `+0.1774%`
- promotion status: `improved_success_candidate` / failed gates: ``

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600`
- train: `+12.1245%`
- val: `+11.4673%`
- locked OOS: `+3.9798%`, Sharpe `5.751805`, MDD `+0.5173%`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600` | `additive_sleeves` | +3.9798% | +0.5173% | `oos_mdd_beats_shadow` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600` | `additive_sleeves` | +3.9643% | +0.5154% | `oos_mdd_beats_shadow` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `additive_sleeves` | +3.9488% | +0.5135% | `oos_mdd_beats_shadow` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600` | `additive_sleeves` | +3.9179% | +0.5097% | `oos_mdd_beats_shadow` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `additive_sleeves` | +3.8870% | +0.5059% | `oos_mdd_beats_shadow` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `additive_sleeves` | +3.8715% | +0.5041% | `oos_mdd_beats_shadow` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600` | `additive_sleeves` | +3.8697% | +0.5056% | `oos_mdd_beats_shadow` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `additive_sleeves` | +3.8604% | +0.5513% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `additive_sleeves` | +3.8543% | +0.5036% | `oos_mdd_beats_shadow` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `additive_sleeves` | +3.8388% | +0.5015% | `oos_mdd_beats_shadow` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | locked OOS MDD | locked OOS Sharpe | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.5019% | +4.0000% | +1.3397% | +0.1774% | 5.477385 | `` |
| 2 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4970% | +4.0000% | +1.3343% | +0.1767% | 5.478457 | `` |
| 3 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4970% | +4.0000% | +1.3343% | +0.1767% | 5.478463 | `` |
| 4 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4920% | +4.0000% | +1.3288% | +0.1759% | 5.479538 | `` |
| 5 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4920% | +4.0000% | +1.3288% | +0.1759% | 5.479542 | `` |
| 6 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4870% | +4.0000% | +1.3234% | +0.1752% | 5.480630 | `` |
| 7 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4870% | +4.0000% | +1.3234% | +0.1752% | 5.480634 | `` |
| 8 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4820% | +4.0000% | +1.3180% | +0.1744% | 5.481729 | `` |
| 9 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4770% | +4.0000% | +1.3125% | +0.1737% | 5.482836 | `` |
| 10 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4721% | +4.0000% | +1.3071% | +0.1730% | 5.483945 | `` |
| 11 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `train_val_target_return_budget` | 0.351825 | True | +5.4621% | +4.0000% | +1.2962% | +0.1715% | 5.486186 | `` |
| 12 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `train_val_target_return_budget` | 0.469220 | True | +5.4183% | +4.0000% | +1.4244% | +0.1762% | 5.591380 | `` |
| 13 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `train_val_target_return_budget` | 0.469220 | True | +5.4117% | +4.0000% | +1.4171% | +0.1752% | 5.592956 | `` |
| 14 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `train_val_target_return_budget` | 0.469220 | True | +5.4050% | +4.0000% | +1.4099% | +0.1742% | 5.594539 | `` |
| 15 | `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600` | `train_val_target_return_budget` | 0.469220 | True | +5.3918% | +4.0000% | +1.3954% | +0.1723% | 5.597727 | `` |
