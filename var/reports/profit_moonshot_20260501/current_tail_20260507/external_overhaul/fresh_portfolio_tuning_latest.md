# Profit moonshot fresh portfolio tuning

Generated: `2026-05-07T13:53:57.363791Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is validation-primary; OOS is report-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `24`
- Portfolio specs evaluated: `17104`
- Combo cap per size: `1000`; skipped by size: `{'3': 1024, '4': 9626, '5': 41504, '6': 133596}`
- Success candidates: `2`
- Peak RSS: `370.113 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0`
- sleeves: `fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0, fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0, fresh_calendar_rot_lethusdt_sweakest_lb168_thr100_h336_sc80_st0, fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h120_sc80_st0, fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0`
- train: `+25.7011%`
- val: `+14.0215%`
- locked OOS: `+5.0322%`, Sharpe `2.576801`, MDD `+2.3586%`
- success: `False` / failed gates: `oos_mdd_beats_shadow`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0`
- train: `+21.6466%`
- val: `+14.5425%`
- locked OOS: `+7.2344%`, Sharpe `3.568584`, MDD `+2.5541%`

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | locked OOS MDD | locked OOS Sharpe | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls540_ss120_tp450__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls530_ss120_tp450` | `validation_return_risk_weight` | 1.000000 | True | +0.6940% | +3.0370% | +0.8789% | +0.1760% | 5.617720 | `` |
| 2 | `fresh_portfolio_equal_weight_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls540_ss120_tp450__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls530_ss120_tp450` | `equal_weight` | 1.000000 | True | +0.6940% | +3.0370% | +0.8789% | +0.1760% | 5.617720 | `` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +21.6466% | +14.5425% | +7.2344% | +2.5541% | 3.568584 | `oos_mdd_beats_shadow` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +23.4318% | +14.1267% | +7.2321% | +3.2300% | 3.162191 | `oos_mdd_beats_shadow` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +21.5953% | +14.6049% | +7.2098% | +2.5541% | 3.557319 | `oos_mdd_beats_shadow` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +19.9062% | +15.0487% | +7.0989% | +2.4944% | 3.556481 | `oos_mdd_beats_shadow` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls600_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +19.8910% | +15.0487% | +7.0659% | +2.4975% | 3.542282 | `oos_mdd_beats_shadow` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +24.5877% | +13.9326% | +7.0621% | +2.5422% | 3.506220 | `oos_mdd_beats_shadow` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +22.7512% | +14.4107% | +7.0398% | +1.8593% | 3.904397 | `oos_mdd_beats_shadow` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +24.5364% | +13.9949% | +7.0375% | +2.5422% | 3.495230 | `oos_mdd_beats_shadow` |
| 11 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +20.3412% | +14.7367% | +7.0353% | +2.5451% | 3.498312 | `oos_mdd_beats_shadow` |
| 12 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr200_h336_sc80_st0` | `additive_sleeves` | 1.000000 | False | +21.6743% | +14.5161% | +7.0353% | +2.5451% | 3.498312 | `oos_mdd_beats_shadow` |
| 13 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls580_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +19.8758% | +15.0487% | +7.0329% | +2.5005% | 3.528042 | `oos_mdd_beats_shadow` |
| 14 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls560_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +19.8606% | +15.0487% | +6.9999% | +2.5036% | 3.513761 | `oos_mdd_beats_shadow` |
| 15 | `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls540_ss120_tp450` | `additive_sleeves` | 1.000000 | False | +19.8454% | +15.0487% | +6.9669% | +2.5067% | 3.499440 | `oos_mdd_beats_shadow` |
