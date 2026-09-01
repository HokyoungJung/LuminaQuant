# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T06:16:01.036753Z`

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
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_return_quality_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `30`
- Portfolio specs evaluated: `62970`
- Combo cap per size: `6000`; skipped by size: `{'4': 21405}`
- Success candidates: `0`
- Peak RSS: `812.922 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600`
- train: `+11.7292%`
- val: `+13.5530%`
- locked OOS: `+2.1374%`, Sharpe `3.022856`, MDD `+0.6571%`
- monthlyized train/val/OOS: `+0.9286%` / `+6.7768%` / `+0.9745%`; smart Sortino `1.636249`
- promotion status: `diagnostic_not_promoted` / failed gates: `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600`
- train: `+12.1245%`
- val: `+11.4673%`
- locked OOS: `+3.9798%`, Sharpe `5.751805`, MDD `+0.5173%`
- monthlyized locked OOS: `+1.8056%`; smart Sortino `6.516153`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600` | `additive_sleeves` | +3.9798% | +0.5173% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600` | `additive_sleeves` | +3.9643% | +0.5154% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `additive_sleeves` | +3.9488% | +0.5135% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600` | `additive_sleeves` | +3.9179% | +0.5097% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls540_ss120_tp600` | `additive_sleeves` | +3.8870% | +0.5059% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `additive_sleeves` | +3.8715% | +0.5041% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600` | `additive_sleeves` | +3.8697% | +0.5056% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `additive_sleeves` | +3.8604% | +0.5513% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `additive_sleeves` | +3.8543% | +0.5036% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600` | `additive_sleeves` | +3.8388% | +0.5015% | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7292% | +13.5530% | +2.1374% | +0.9745% | +0.6571% | 3.022856 | 1.636249 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7262% | +13.5530% | +2.1309% | +0.9715% | +0.6535% | 3.026282 | 1.633540 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7232% | +13.5530% | +2.1244% | +0.9686% | +0.6499% | 3.029736 | 1.630834 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7203% | +13.5530% | +2.1178% | +0.9656% | +0.6463% | 3.033219 | 1.628132 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7173% | +13.5530% | +2.1113% | +0.9626% | +0.6427% | 3.036733 | 1.625432 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7173% | +13.5530% | +2.1113% | +0.9626% | +0.6427% | 3.036732 | 1.625433 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7144% | +13.5530% | +2.1047% | +0.9597% | +0.6390% | 3.040275 | 1.622738 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7144% | +13.5530% | +2.1047% | +0.9597% | +0.6390% | 3.040274 | 1.622739 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7114% | +13.5530% | +2.0982% | +0.9567% | +0.6354% | 3.043847 | 1.620046 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7114% | +13.5530% | +2.0982% | +0.9567% | +0.6354% | 3.043846 | 1.620048 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 11 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7114% | +13.5530% | +2.0982% | +0.9567% | +0.6354% | 3.043846 | 1.620049 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 12 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7085% | +13.5530% | +2.0917% | +0.9537% | +0.6318% | 3.047450 | 1.617358 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 13 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls540_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7084% | +13.5530% | +2.0917% | +0.9537% | +0.6318% | 3.047449 | 1.617360 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 14 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls530_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7084% | +13.5530% | +2.0917% | +0.9537% | +0.6318% | 3.047448 | 1.617361 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
| 15 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls580_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls560_ss120_tp600` | `additive_sleeves` | 1.000000 | False | +11.7055% | +13.5530% | +2.0851% | +0.9508% | +0.6282% | 3.051083 | 1.614674 | `train_monthly_return_gte_2pct,oos_monthly_return_gte_2pct,oos_return_risk_beats_current_champion,oos_smart_sortino_high` |
