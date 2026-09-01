# Profit moonshot fresh portfolio tuning

Generated: `2026-05-08T11:08:15.232625Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is validation-primary; OOS is report-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `24`
- Portfolio specs evaluated: `10704`
- Combo cap per size: `600`; skipped by size: `{'3': 1424, '4': 10026, '5': 41904, '6': 133996}`
- Success candidates: `2`
- Peak RSS: `318.191 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0`
- sleeves: `fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0, fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0, fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h120_sc80_st0, fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0`
- train: `+25.7011%`
- val: `+14.0215%`
- locked OOS: `+5.0322%`, Sharpe `2.576801`, MDD `+2.3586%`
- success: `False` / failed gates: `oos_mdd_beats_shadow`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0`
- train: `+21.5812%`
- val: `+14.6314%`
- locked OOS: `+6.9653%`, Sharpe `3.464433`, MDD `+2.5542%`

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | locked OOS MDD | locked OOS Sharpe | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls540_ss120_tp450__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls530_ss120_tp450` | `validation_return_risk_weight` | 1.000000 | True | +0.6940% | +3.0370% | +0.8789% | +0.1760% | 5.617720 | `` |
| 2 | `fresh_portfolio_equal_weight_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls540_ss120_tp450__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls530_ss120_tp450` | `equal_weight` | 1.000000 | True | +0.6940% | +3.0370% | +0.8789% | +0.1760% | 5.617720 | `` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +21.5812% | +14.6314% | +6.9653% | +2.5542% | 3.464433 | `oos_mdd_beats_shadow` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +22.7372% | +14.4373% | +6.7953% | +1.8587% | 3.818637 | `oos_mdd_beats_shadow` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +24.5224% | +14.0215% | +6.7930% | +2.5423% | 3.398776 | `oos_mdd_beats_shadow` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +23.0567% | +14.3051% | +6.7925% | +3.3194% | 2.751436 | `oos_mdd_beats_shadow` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +22.6859% | +14.4996% | +6.7707% | +1.8587% | 3.805817 | `oos_mdd_beats_shadow` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +20.8512% | +14.7367% | +6.7528% | +3.3124% | 2.747275 | `oos_mdd_beats_shadow` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +20.9968% | +14.9434% | +6.6598% | +1.7979% | 3.829465 | `oos_mdd_beats_shadow` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls600_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +20.9816% | +14.9434% | +6.6268% | +1.8010% | 3.814944 | `oos_mdd_beats_shadow` |
| 11 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +24.2126% | +14.1109% | +6.6225% | +2.6313% | 3.005867 | `oos_mdd_beats_shadow` |
| 12 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +25.9978% | +13.6951% | +6.6202% | +3.3076% | 2.704790 | `oos_mdd_beats_shadow` |
| 13 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +24.1613% | +14.1733% | +6.5979% | +2.6313% | 2.995467 | `oos_mdd_beats_shadow` |
| 14 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +23.8849% | +14.1267% | +6.5963% | +2.5451% | 3.273948 | `oos_mdd_beats_shadow` |
| 15 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +21.4318% | +14.6314% | +6.5962% | +1.8490% | 3.731834 | `oos_mdd_beats_shadow` |
