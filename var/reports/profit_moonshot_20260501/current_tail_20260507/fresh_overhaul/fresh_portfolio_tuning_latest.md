# Profit moonshot fresh portfolio tuning

Generated: `2026-05-07T11:21:50.849897Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is validation-primary; OOS is report-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.

## Summary

- Candidate sleeves considered: `18`
- Portfolio specs evaluated: `25194`
- Success candidates: `0`
- Peak RSS: `2670.785 MiB`

## Selected by validation

- `fresh_portfolio_additive_sleeves_fresh_xs_mom_lb12_z175_h96__fresh_resid_flow_rev_fl6_lb6_z125_imb6__fresh_flow_mom_fl24_px24_imb6_h24__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175`
- sleeves: `fresh_xs_mom_lb12_z175_h96, fresh_resid_flow_rev_fl6_lb6_z125_imb6, fresh_flow_mom_fl24_px24_imb6_h24, fresh_resid_rev_rvcap3156_lb12_z175, fresh_resid_rev_rvcap4143_lb24_z175`
- train: `+0.2034%`
- val: `+0.3111%`
- OOS: `+0.1349%`, Sharpe `0.325600`, MDD `+0.1402%`
- success: `False` / failed gates: `oos_return_beats_incumbent,oos_sharpe_gt_1`

## Diagnostic best OOS (not selection authority)

- `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48`
- train: `+0.3895%`
- val: `+0.1018%`
- OOS: `+0.2661%`, Sharpe `0.480320`, MDD `+0.1963%`

## Top rows

| rank | name | success | train | val | OOS | OOS MDD | OOS Sharpe | failed gates |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3895% | +0.1018% | +0.2661% | +0.1963% | 0.480320 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 2 | `fresh_portfolio_additive_sleeves_fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_resid_flow_rev_fl12_lb48_z125_imb6__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.2992% | +0.0860% | +0.2546% | +0.1252% | 0.591243 | `oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 3 | `fresh_portfolio_additive_sleeves_fresh_xs_mom_lb12_z175_h96__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3005% | +0.2404% | +0.2531% | +0.1937% | 0.461921 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 4 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_flow_rev_fl12_lb48_z125_imb6__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3931% | +0.0744% | +0.2472% | +0.1871% | 0.467325 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 5 | `fresh_portfolio_additive_sleeves_fresh_xs_mom_lb12_z175_h96__fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3944% | +0.2288% | +0.2457% | +0.2392% | 0.384123 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 6 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_resid_flow_rev_fl12_lb48_z125_imb6__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3071% | +0.1070% | +0.2441% | +0.1273% | 0.610963 | `oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 7 | `fresh_portfolio_additive_sleeves_fresh_xs_mom_lb12_z175_h96__fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3084% | +0.2614% | +0.2427% | +0.1938% | 0.467929 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 8 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h24__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3674% | +0.1088% | +0.2419% | +0.1950% | 0.429630 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 9 | `fresh_portfolio_additive_sleeves_fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.2653% | +0.0773% | +0.2368% | +0.1325% | 0.554473 | `oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 10 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px12_imb6_h24__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.2944% | +0.1125% | +0.2367% | +0.1950% | 0.416900 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 11 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h24__fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.4612% | +0.0972% | +0.2345% | +0.2463% | 0.351154 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 12 | `fresh_portfolio_additive_sleeves_fresh_xs_mom_lb12_z175_h96__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_flow_rev_fl12_lb48_z125_imb6__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3042% | +0.2131% | +0.2342% | +0.1777% | 0.448665 | `oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 13 | `fresh_portfolio_additive_sleeves_fresh_flow_mom_fl24_px24_imb6_h24__fresh_flow_mom_fl24_px24_imb6_h48__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3752% | +0.1298% | +0.2315% | +0.1946% | 0.438184 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 14 | `fresh_portfolio_additive_sleeves_fresh_xs_mom_lb12_z175_h96__fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_resid_flow_rev_fl12_lb48_z125_imb6__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.2181% | +0.2457% | +0.2311% | +0.1443% | 0.534309 | `oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 15 | `fresh_portfolio_additive_sleeves_fresh_resid_rev_rvcap3156_lb12_z175__fresh_resid_rev_rvcap4143_lb24_z175__fresh_flow_mom_fl24_px6_imb6_h24__fresh_flow_mom_fl24_px12_imb6_h48__fresh_flow_mom_fl24_px6_imb6_h48` | False | +0.3082% | +0.0880% | +0.2299% | +0.2114% | 0.404761 | `oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |

