# Profit moonshot fresh portfolio tuning

Generated: `2026-05-09T04:47:20.441628Z`

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
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/h3_h4_h5_calendar_conditioned/portfolio/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `18`
- Portfolio specs evaluated: `20145`
- Combo cap per size: `6000`; skipped by size: `{}`
- Success candidates: `0`
- Peak RSS: `389.789 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120`
- sleeves: `fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120, fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168, fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120, fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120`
- train: `+12.1482%`
- val: `+13.4822%`
- locked OOS: `+2.2229%`, Sharpe `3.265110`, MDD `+0.7026%`
- promotion status: `diagnostic_not_promoted` / failed gates: `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168`
- train: `+14.6610%`
- val: `+10.7938%`
- locked OOS: `+3.7780%`, Sharpe `5.286180`, MDD `+0.5981%`
- promotion status: `diagnostic_not_promoted`

## H6 diagnostic quarantine

- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.
- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.

| rank | name | mode | locked OOS | locked OOS MDD | failed gates |
|---:|---|---|---:|---:|---|
| 1 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168` | `additive_sleeves` | +3.7780% | +0.5981% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr150_h120` | `additive_sleeves` | +3.6872% | +0.4994% | `oos_mdd_beats_shadow` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr150_h120` | `additive_sleeves` | +3.6872% | +0.4994% | `oos_mdd_beats_shadow` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr150_h120` | `additive_sleeves` | +3.6565% | +0.4995% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr150_h120` | `additive_sleeves` | +3.6381% | +0.4994% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168` | `additive_sleeves` | +3.6347% | +0.5970% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168` | `additive_sleeves` | +3.6347% | +0.5970% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168__fresh_calendar_trx_veto_flow6_sethusdt_thr150_h120` | `additive_sleeves` | +3.6326% | +0.4994% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_fund100_sweakest_thr150_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168__fresh_calendar_trx_veto_flow6_sethusdt_thr150_h120` | `additive_sleeves` | +3.6326% | +0.4994% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168` | `additive_sleeves` | +3.6040% | +0.5973% | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | locked OOS MDD | locked OOS Sharpe | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +12.1482% | +13.4822% | +2.2229% | +0.7026% | 3.265110 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168` | `additive_sleeves` | 1.000000 | False | +12.0942% | +13.4822% | +2.2229% | +0.7026% | 3.265110 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +12.0162% | +13.4954% | +2.0797% | +0.7440% | 3.023703 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +12.0026% | +13.4822% | +2.3200% | +0.7026% | 3.419889 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168` | `additive_sleeves` | 1.000000 | False | +11.9487% | +13.4822% | +2.3200% | +0.7026% | 3.419889 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.9289% | +13.4822% | +2.1042% | +0.6821% | 3.017666 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.8956% | +13.4822% | +2.2939% | +0.7026% | 3.399258 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168` | `additive_sleeves` | 1.000000 | False | +11.8749% | +13.4822% | +2.1042% | +0.6821% | 3.017666 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +12.0628% | +13.5530% | +2.1998% | +0.6548% | 3.196083 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.8707% | +13.4954% | +2.1768% | +0.7440% | 3.176507 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 11 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.8416% | +13.4822% | +2.2939% | +0.7026% | 3.399258 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 12 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_flow6_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.8218% | +13.4822% | +2.0782% | +0.6821% | 3.011356 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 13 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168` | `additive_sleeves` | 1.000000 | False | +12.0088% | +13.5530% | +2.1998% | +0.6548% | 3.196083 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 14 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_fund100_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.9308% | +13.5662% | +2.0566% | +0.6962% | 2.960390 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
| 15 | `fresh_portfolio_additive_sleeves_fresh_calendar_trx_veto_rz10_sethusdt_thr150_h120__fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168__fresh_calendar_trx_veto_rz10_sethusdt_thr180_h120__fresh_calendar_trx_veto_rz15_sethusdt_thr180_h120` | `additive_sleeves` | 1.000000 | False | +11.7969% | +13.4954% | +1.9610% | +0.7235% | 2.780031 | `oos_return_risk_beats_current_champion,oos_mdd_beats_shadow` |
