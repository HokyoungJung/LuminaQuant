# Profit moonshot fresh portfolio tuning

Generated: `2026-05-08T12:33:24.829842Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is validation-primary; OOS is report-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `72`
- Portfolio specs evaluated: `58224`
- Combo cap per size: `3000`; skipped by size: `{'3': 56640, '4': 1025790, '5': 13988544, '6': 156235908}`
- Success candidates: `6129`
- Peak RSS: `652.676 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- train: `+20.4013%`
- val: `+19.4308%`
- locked OOS: `+3.9710%`, Sharpe `3.825533`, MDD `+0.8977%`
- success: `False` / failed gates: `oos_mdd_beats_shadow`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0`
- train: `+20.0276%`
- val: `+17.3732%`
- locked OOS: `+6.3563%`, Sharpe `5.969314`, MDD `+1.2530%`

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | locked OOS MDD | locked OOS Sharpe | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_equal_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `equal_weight` | 1.000000 | True | +3.5993% | +2.6755% | +1.2181% | +0.1662% | 6.726378 | `` |
| 2 | `fresh_portfolio_equal_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `equal_weight` | 1.000000 | True | +3.5864% | +2.6911% | +1.2119% | +0.1662% | 6.699390 | `` |
| 3 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.8883% | +2.6377% | +1.1923% | +0.1758% | 6.853414 | `` |
| 4 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_rot_ltrxusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.1900% | +2.7076% | +1.1858% | +0.1746% | 6.420248 | `` |
| 5 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss100_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.8814% | +2.6378% | +1.1847% | +0.1755% | 6.855015 | `` |
| 6 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +4.0218% | +2.7582% | +1.1839% | +0.1773% | 6.857293 | `` |
| 7 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss100_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.8779% | +2.6378% | +1.1809% | +0.1754% | 6.855738 | `` |
| 8 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +4.0184% | +2.7582% | +1.1803% | +0.1773% | 6.857801 | `` |
| 9 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr20_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.7522% | +2.7179% | +1.1788% | +0.1544% | 6.743364 | `` |
| 10 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss100_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.8744% | +2.6379% | +1.1771% | +0.1754% | 6.856407 | `` |
| 11 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +4.0149% | +2.7581% | +1.1766% | +0.1774% | 6.858260 | `` |
| 12 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr50_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.7400% | +2.7286% | +1.1743% | +0.1542% | 6.720817 | `` |
| 13 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +4.0115% | +2.7581% | +1.1730% | +0.1774% | 6.858662 | `` |
| 14 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss100_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +3.8675% | +2.6380% | +1.1696% | +0.1755% | 6.857581 | `` |
| 15 | `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` | `validation_return_risk_weight` | 1.000000 | True | +4.0081% | +2.7581% | +1.1693% | +0.1774% | 6.859017 | `` |
