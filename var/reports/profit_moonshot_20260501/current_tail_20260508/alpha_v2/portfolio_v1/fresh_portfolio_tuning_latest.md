# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T05:23:28.013589Z`

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
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `30`
- Portfolio specs evaluated: `52475`
- Combo cap per size: `6000`; skipped by size: `{'4': 21405}`
- Success candidates: `0`
- Peak RSS: `668.371 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- train: `+11.7292%`
- val: `+13.5530%`
- locked OOS: `+2.1374%`, Sharpe `3.022856`, MDD `+0.6571%`
- promotion status: `diagnostic_not_promoted` / failed gates: `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow`

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
| 1 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7292% | +13.5530% | +2.1374% | +0.6571% | 3.022856 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7262% | +13.5530% | +2.1309% | +0.6535% | 3.026282 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7232% | +13.5530% | +2.1244% | +0.6499% | 3.029736 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7203% | +13.5530% | +2.1178% | +0.6463% | 3.033219 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7173% | +13.5530% | +2.1113% | +0.6427% | 3.036733 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7173% | +13.5530% | +2.1113% | +0.6427% | 3.036732 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7144% | +13.5530% | +2.1047% | +0.6390% | 3.040275 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7144% | +13.5530% | +2.1047% | +0.6390% | 3.040274 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7114% | +13.5530% | +2.0982% | +0.6354% | 3.043847 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7114% | +13.5530% | +2.0982% | +0.6354% | 3.043846 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 11 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7114% | +13.5530% | +2.0982% | +0.6354% | 3.043846 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 12 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7085% | +13.5530% | +2.0917% | +0.6318% | 3.047450 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 13 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7084% | +13.5530% | +2.0917% | +0.6318% | 3.047449 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 14 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7084% | +13.5530% | +2.0917% | +0.6318% | 3.047448 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 15 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7055% | +13.5530% | +2.0851% | +0.6282% | 3.051083 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
