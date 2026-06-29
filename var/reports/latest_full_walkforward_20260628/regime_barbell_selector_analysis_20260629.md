# Regime-aware barbell selector analysis

Shadow/research only. `ready_for_real=false`; current-fold OOS is not used for selector decisions.

## Contamination audit
- selector_uses_current_fold_oos_for_gate: `False`
- selector_uses_current_fold_oos_for_bull_sleeve_selection: `False`
- selector_weight_inputs: `train metrics, validation metrics, validation-end BTC/crypto regime features only`
- current_fold_oos_role: `evaluation_after_decision_only`
- new_strategy_real_money: `False`
- shadow_only: `True`
- post_oos_research_variant_present: `True`
- post_oos_research_variant_interpretation: `hypothesis_only_not_clean_scientific_evidence`

## Methodology validation audit
- per_fold_selector_decisions_walk_forward_valid: `True`
- current_fold_oos_used_by_selector_decisions: `False`
- prior_completed_oos_used_by_recency_weighting: `True`
- prior_completed_oos_is_lagged_live_available: `True`
- selector_family_and_hyperparameters_chosen_after_source_oos_review: `True`
- nested_walk_forward_methodology_validated: `False`
- methodology_status: `candidate_walk_forward_evaluated_not_methodology_walk_forward_validated`
- post_oos_research_selector_labels: `['regime_barbell_selector:v2_recency_weighted_raw_crisis_recovery_cash_guard_bull55', 'regime_barbell_selector:v2_recovery_cash_guard_fallback_mdd20_bull55_mixed30_bear0']`
- recency_weighted_selector_labels: `['regime_barbell_selector:v2_recency_weighted_raw_crisis_recovery_cash_guard_bull55']`
- clean_next_validation: `Run a rolling-origin/nested selector-method WF where the selector family, candidate sleeves, weights, guards, and recency half-life are chosen only from data and completed folds available before each evaluated fold; then lock the method for fresh forward shadow.`

## Baseline / selector comparison
| role | label | comp | positive folds | min OOS | latest OOS | max OOS MDD | monthly MDD | Sharpe | CVaR25 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source_baseline | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` | 69.08% | 4/10 | -4.46% | 6.07% | 27.69% | 5.09% | 1.80 | -1.71% |
| source_baseline | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2` | 56.47% | 4/10 | -4.46% | 6.07% | 25.01% | 5.09% | 1.92 | -1.71% |
| source_baseline | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | 34.39% | 3/10 | 0.00% | 0.00% | 27.69% | 0.00% | 1.12 | 0.00% |
| source_baseline | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled` | 22.39% | 5/10 | -4.32% | -4.32% | 27.69% | 9.39% | 0.78 | -3.38% |
| new_selector | `regime_barbell_selector:v2_recency_weighted_raw_crisis_recovery_cash_guard_bull55` | 82.45% | 6/10 | -0.05% | 6.07% | 27.69% | 0.05% | 2.13 | -0.01% |
| new_selector | `regime_barbell_selector:v2_recovery_cash_guard_fallback_mdd20_bull55_mixed30_bear0` | 68.85% | 6/10 | -0.05% | 6.07% | 25.01% | 0.05% | 2.39 | -0.01% |
| new_selector | `regime_barbell_selector:v1_raw_crisis_bull60_mixed35_bear10` | 60.73% | 6/10 | -9.45% | 4.38% | 27.69% | 9.57% | 1.59 | -3.43% |
| new_selector | `regime_barbell_selector:v1_return_floor_fallback_mdd20_bull70_mixed35_bear0` | 52.49% | 6/10 | -10.28% | 6.07% | 25.01% | 10.32% | 1.71 | -2.58% |
| new_selector | `regime_barbell_selector:v1_clean_bull_fallback_mdd20_bull55_mixed30_bear10` | 49.47% | 6/10 | -9.03% | 4.38% | 23.93% | 9.15% | 1.66 | -3.29% |
| new_selector | `regime_barbell_selector:v1_fallback_mdd20_bull65_mixed40_bear15` | 48.08% | 6/10 | -9.87% | 3.54% | 23.39% | 10.02% | 1.58 | -3.71% |
| new_strategy_sleeve_eval | `new_bull_bear_regime_rotation:bull_bear_regime_rotation_1d_daily_macro_ls_30_55_55` | 14.00% | 4/10 | -9.71% | 22.17% | 20.35% | 20.88% | 0.55 | -8.35% |
| new_strategy_sleeve_eval | `new_bull_bear_regime_rotation:bull_bear_regime_rotation_4h_macro_bear_capture_ls_36_56_54` | 6.36% | 5/10 | -11.68% | 15.14% | 17.19% | 20.30% | 0.37 | -10.18% |
| new_strategy_sleeve_eval | `new_bull_bear_regime_rotation:bull_bear_regime_rotation_1h_swing_core_ls_48_57_55` | -34.29% | 3/10 | -15.75% | 6.71% | 19.84% | 39.29% | -1.72 | -12.26% |
| new_strategy_sleeve_eval | `new_bull_bear_regime_rotation:bull_bear_regime_rotation_30m_exec_fast_ls_48_57_56` | -63.92% | 1/10 | -19.23% | -3.27% | 27.59% | 61.76% | -4.58 | -16.48% |

## Fold regime and OOS return table
| fold | regime | raw | fallback | clean | pos_base | selector_1 | selector_2 | selector_3 | selector_4 | selector_5 | selector_6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-09 | BEAR | 0.00% | 0.00% | 0.00% | 0.14% | 0.00% | 0.00% | -0.50% | 0.00% | -0.50% | -0.75% |
| 2025-10 | MIXED | 0.00% | 0.00% | 0.00% | 0.00% | 1.59% | 1.59% | 1.86% | 1.86% | 1.59% | 2.13% |
| 2025-11 | BEAR | 33.48% | 23.53% | 33.48% | 33.48% | 33.48% | 23.53% | 33.48% | 23.53% | 24.04% | 25.02% |
| 2025-12 | BEAR | 0.00% | 0.00% | 0.00% | -0.13% | 0.00% | 0.00% | -0.33% | 0.00% | -0.33% | -0.50% |
| 2026-01 | BEAR | 16.03% | 16.03% | 0.00% | 0.99% | 16.03% | 16.03% | 14.93% | 16.03% | 14.93% | 14.39% |
| 2026-02 | BEAR | 8.44% | 8.44% | 0.21% | 0.21% | 8.44% | 8.44% | 8.80% | 8.44% | 8.80% | 8.98% |
| 2026-03 | BEAR | -0.05% | -0.05% | 0.00% | -2.65% | -0.05% | -0.05% | -0.13% | -0.05% | -0.13% | -0.17% |
| 2026-04 | BULL | -4.46% | -4.46% | 0.00% | -3.18% | -0.00% | -0.00% | -9.45% | -10.28% | -9.03% | -9.87% |
| 2026-05 | BULL | -0.62% | -0.62% | 0.47% | 0.47% | 0.86% | 0.86% | 0.99% | 1.26% | 0.86% | 1.13% |
| 2026-06 | BEAR | 6.07% | 6.07% | 0.00% | -4.32% | 6.07% | 6.07% | 4.38% | 6.07% | 4.38% | 3.54% |

Selector labels:
- selector_1: `regime_barbell_selector:v2_recency_weighted_raw_crisis_recovery_cash_guard_bull55`
- selector_2: `regime_barbell_selector:v2_recovery_cash_guard_fallback_mdd20_bull55_mixed30_bear0`
- selector_3: `regime_barbell_selector:v1_raw_crisis_bull60_mixed35_bear10`
- selector_4: `regime_barbell_selector:v1_return_floor_fallback_mdd20_bull70_mixed35_bear0`
- selector_5: `regime_barbell_selector:v1_clean_bull_fallback_mdd20_bull55_mixed30_bear10`
- selector_6: `regime_barbell_selector:v1_fallback_mdd20_bull65_mixed40_bear15`

## Newly added strategy fold OOS table
| fold | new_1 | new_2 | new_3 | new_4 |
|---|---|---|---|---|
| 2025-09 | -3.37% | 2.03% | -2.94% | -5.64% |
| 2025-10 | -9.71% | -0.05% | 4.52% | -14.64% |
| 2025-11 | 23.08% | -1.87% | -2.15% | -15.58% |
| 2025-12 | 3.39% | -9.09% | -15.75% | -10.36% |
| 2026-01 | -8.29% | 15.52% | 5.55% | 2.22% |
| 2026-02 | -1.25% | 5.22% | -7.76% | -0.78% |
| 2026-03 | -6.00% | -11.68% | -5.95% | -13.62% |
| 2026-04 | -7.05% | -9.76% | -13.26% | -19.23% |
| 2026-05 | 6.24% | 4.81% | -7.30% | -13.51% |
| 2026-06 | 22.17% | 15.14% | 6.71% | -3.27% |

New strategy labels:
- new_1: `new_bull_bear_regime_rotation:bull_bear_regime_rotation_1d_daily_macro_ls_30_55_55`
- new_2: `new_bull_bear_regime_rotation:bull_bear_regime_rotation_4h_macro_bear_capture_ls_36_56_54`
- new_3: `new_bull_bear_regime_rotation:bull_bear_regime_rotation_1h_swing_core_ls_48_57_55`
- new_4: `new_bull_bear_regime_rotation:bull_bear_regime_rotation_30m_exec_fast_ls_48_57_56`

## Regime decision audit
| fold | decision | val BTC | last-val-month BTC | val breadth | last breadth | OOS BTC (analysis only) | reason |
|---|---|---:|---:|---:|---:|---:|---|
| 2025-09 | BEAR | 1.02% | -6.50% | 1.00 | 0.64 | 5.76% | last_validation_month_negative_or_breadth_breakdown |
| 2025-10 | MIXED | -1.51% | 5.76% | 0.64 | 0.73 | -3.86% | lagged_validation_regime_mixed_or_choppy |
| 2025-11 | BEAR | 1.65% | -3.86% | 0.27 | 0.09 | -17.52% | last_validation_month_negative_or_breadth_breakdown |
| 2025-12 | BEAR | -20.74% | -17.52% | 0.00 | 0.00 | -0.83% | last_validation_month_negative_or_breadth_breakdown |
| 2026-01 | BEAR | -19.99% | -0.83% | 0.00 | 0.36 | -10.30% | two_month_validation_drawdown_with_weak_breadth |
| 2026-02 | BEAR | -10.90% | -10.30% | 0.09 | 0.09 | -15.33% | last_validation_month_negative_or_breadth_breakdown |
| 2026-03 | BEAR | -23.72% | -15.33% | 0.00 | 0.00 | 2.24% | last_validation_month_negative_or_breadth_breakdown |
| 2026-04 | BULL | -13.68% | 2.24% | 0.09 | 0.45 | 12.27% | washout_recovery_bull_after_positive_last_validation_month |
| 2026-05 | BULL | 14.32% | 12.27% | 0.55 | 0.91 | -3.70% | last_validation_month_positive_with_breadth_and_non_negative_refit_ma_gap |
| 2026-06 | BEAR | 8.37% | -3.70% | 0.64 | 0.27 | -18.53% | last_validation_month_negative_or_breadth_breakdown |

## Notes
- This is a fast-path recombination/evaluation report; it does not replace a full fresh monthly WF rerun.
- Selector fold MDD uses a convex source-row MDD proxy because existing WF JSON does not carry intramonth equity curves.
- New BullBearRegimeRotationStrategy rows are evaluated from latest refreshed data as research/shadow sleeves only.
- The v2 selectors are explicitly flagged as post-OOS research variants; treat their results as hypotheses needing nested method WF or fresh forward shadow, not as clean evidence.
- No ready_for_real claim is made; fresh forward shadow/paper-testnet evidence remains required.
