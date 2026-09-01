# Hybrid v3.5/v3.6 Optuna fixed-input comparison

- generated_at_utc: `2026-05-16T17:29:01.871204Z`
- fixed inputs: `A0 + P0 + E0 + S1 + S2 + S3 + S4`
- external method source: `/home/hoky/DeepLearning/ensemble_strategies/models/hybrid/v3_5.py`, `v3_6.py`
- selection/objective inputs: `train`, `validation` only
- locked-OOS: report/gate-only after candidate freeze; not used by Optuna objective/pruning/selection

## Split periods

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z`
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z`
- locked_oos: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z`

## Candidate inputs

| label | candidate | source | train | validation | locked-OOS |
|---|---|---|---:|---:|---:|
| A0 | `crypto_fx_alpha_zoo_state_calibrated` | `CryptoFxAlphaZooStateStrategy:alpha_zoo_conservative_exit:strict_6x` | +114.46% / MDD +28.27% / Sh 1.980 / Liq not_replayed / Buf not_replayed | +19.97% / MDD +13.67% / Sh 2.668 / Liq not_replayed / Buf not_replayed | +20.51% / MDD +6.79% / Sh 3.993 / Liq not_replayed / Buf not_replayed |
| P0 | `fresh_portfolio_train_val_monthly_return_budget_fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240` | `state_distilled_market_state_next_tuning:selected_by_validation` | +64.11% / MDD +17.75% / Sh 1.855 / Liq not_replayed / Buf not_replayed | +26.68% / MDD +5.37% / Sh 6.498 / Liq not_replayed / Buf not_replayed | +4.52% / MDD +6.97% / Sh 1.218 / Liq not_replayed / Buf not_replayed |
| E0 | `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` | `liquidation_aware_state_distilled_external_risk_filter_20260512` | +32.17% / MDD +7.89% / Sh 2.122 / Liq not_replayed / Buf not_replayed | +11.62% / MDD +2.79% / Sh 5.241 / Liq not_replayed / Buf not_replayed | +2.90% / MDD +2.36% / Sh 1.780 / Liq not_replayed / Buf not_replayed |
| S1 | `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` | `state_distilled_leadership_unwind_20260511` | +8.04% / MDD +2.31% / Sh 2.064 / Liq not_replayed / Buf not_replayed | +2.91% / MDD +0.77% / Sh 5.096 / Liq not_replayed / Buf not_replayed | +0.73% / MDD +0.60% / Sh 1.748 / Liq not_replayed / Buf not_replayed |
| S2 | `fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600` | `state_distilled_leadership_unwind_20260511` | +7.07% / MDD +2.86% / Sh 1.819 / Liq not_replayed / Buf not_replayed | +2.91% / MDD +0.77% / Sh 5.096 / Liq not_replayed / Buf not_replayed | +0.68% / MDD +0.60% / Sh 1.631 / Liq not_replayed / Buf not_replayed |
| S3 | `fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600` | `state_distilled_leadership_unwind_20260511` | +6.90% / MDD +2.86% / Sh 1.779 / Liq not_replayed / Buf not_replayed | +2.91% / MDD +0.77% / Sh 5.096 / Liq not_replayed / Buf not_replayed | +0.68% / MDD +0.60% / Sh 1.631 / Liq not_replayed / Buf not_replayed |
| S4 | `fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450` | `state_distilled_leadership_unwind_20260511` | +3.51% / MDD +3.27% / Sh 1.015 / Liq not_replayed / Buf not_replayed | +1.52% / MDD +0.90% / Sh 3.832 / Liq not_replayed / Buf not_replayed | +0.62% / MDD +0.81% / Sh 1.515 / Liq not_replayed / Buf not_replayed |

## Hybrid Optuna results

| model | TV score | train | validation | locked-OOS | OOS MDD | OOS Sharpe | OOS Sortino | OOS Calmar | OOS liquidation | OOS min buffer | deployable_success | rejection reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| hybrid_v3_5_optuna | 70.585 | +47.73% | +13.31% | +8.52% | +1.77% | 5.259 | 7.317 | 32.173 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |
| hybrid_v3_6_optuna | 85.548 | +49.52% | +12.49% | +7.79% | +1.75% | 4.860 | 5.991 | 29.200 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |

## Final weights

- hybrid_v3_5_optuna: A0=+39.60%, P0=+10.07%, E0=+10.07%, S1=+10.06%, S2=+10.06%, S3=+10.06%, S4=+10.06%
- hybrid_v3_6_optuna: A0=+9.92%, P0=+9.93%, E0=+40.03%, S1=+10.03%, S2=+10.03%, S3=+10.03%, S4=+10.01%
