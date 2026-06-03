# Risk-enhanced 69-asset walk-forward extended metrics
- generated: `2026-06-02T09:03:50.996117+00:00`
- OOS period: `2025-09-01T00:00:00` → `2026-06-01T06:30:00`
- latest data: `2026-06-01T06:30:00`
- Full-month OOS: `2025-09` through `2026-05`; `2026-06` is partial through `2026-06-01T06:30:00`.
- timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`; slippage: `10.0` bps
- Calmar1 = compounded OOS / max intra-fold OOS MDD. Calmar2 = annualized OOS / monthly equity MDD.

## Decision
- Clean 최고 comp는 여전히 `dynamic_conviction_switch:t0.85/t0.90/t0.95_risk_capped_fallback` = **53.38%**, Sharpe **2.07**, Sortino **15.31**, 하지만 max OOS MDD **18.80%**.
- MDD 15% 이하를 요구하면 clean 최고는 `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` = **32.94%**, Sharpe **1.45**, Sortino **10.16**, max OOS MDD **11.22%**.
- 성적을 더 끌어올리는 연구 overlay는 `risk_enhanced_blend:dyn085_70_aware_v36tv_30` = **47.60%**, Sharpe **1.98**, Sortino **16.23**, max OOS MDD **14.67%**. 단 이전 OOS를 본 뒤 추가한 후보라 같은 OOS에서는 clean 승격 불가, fresh forward shadow 필요.
- hit balance는 `risk_enhanced_blend:dyn085_50_aware_v35_50` = **39.31%**, hit **6/10**, max OOS MDD **14.96%**, PF **7.87**.

## Clean candidates by comp

| Candidate | Clean | OOS comp | Ann. | Hit | Min | Latest | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | Calmar1 | Calmar2 | VaR5 | CVaR25 | PF | Omega | Tail | Skew | Kurt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `True` | 53.38% | 67.08% | 5/10 | -2.65% | -0.71% | 18.80% | 4.62% | 2.07 | 15.31 | 2.84 | 14.53 | -2.37% | -1.80% | 8.40 | 8.40 | 6.79 | 0.71 | -0.96 |
| `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `True` | 53.38% | 67.08% | 5/10 | -2.65% | -0.71% | 18.80% | 4.62% | 2.07 | 15.31 | 2.84 | 14.53 | -2.37% | -1.80% | 8.40 | 8.40 | 6.79 | 0.71 | -0.96 |
| `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `True` | 53.38% | 67.08% | 5/10 | -2.65% | -0.71% | 18.80% | 4.62% | 2.07 | 15.31 | 2.84 | 14.53 | -2.37% | -1.80% | 8.40 | 8.40 | 6.79 | 0.71 | -0.96 |
| `dynamic_conviction_switch:t0.85_strict_fallback` | `True` | 43.73% | 54.55% | 4/10 | -8.56% | -0.71% | 18.80% | 2.11% | 1.62 | 4.24 | 2.33 | 25.81 | -5.62% | -3.77% | 4.27 | 4.27 | 2.86 | 0.39 | -0.91 |
| `dynamic_conviction_switch:t0.90_strict_fallback` | `True` | 43.73% | 54.55% | 4/10 | -8.56% | -0.71% | 18.80% | 2.11% | 1.62 | 4.24 | 2.33 | 25.81 | -5.62% | -3.77% | 4.27 | 4.27 | 2.86 | 0.39 | -0.91 |
| `dynamic_conviction_switch:t0.95_strict_fallback` | `True` | 43.73% | 54.55% | 4/10 | -8.56% | -0.71% | 18.80% | 2.11% | 1.62 | 4.24 | 2.33 | 25.81 | -5.62% | -3.77% | 4.27 | 4.27 | 2.86 | 0.39 | -0.91 |
| `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `True` | 39.53% | 49.15% | 5/10 | -2.65% | 0.00% | 18.80% | 4.62% | 1.69 | 10.82 | 2.10 | 10.64 | -2.37% | -1.80% | 7.54 | 7.54 | 6.79 | 1.15 | -0.12 |
| `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `True` | 32.94% | 40.73% | 5/10 | -3.19% | -0.78% | 11.22% | 4.46% | 1.45 | 10.16 | 2.94 | 9.12 | -3.12% | -2.65% | 4.10 | 4.10 | 4.76 | 1.43 | 1.23 |
| `dynamic_conviction_switch:t1.00_strict_fallback` | `True` | 30.54% | 37.68% | 4/10 | -8.56% | -0.17% | 18.80% | 2.11% | 1.26 | 3.11 | 1.62 | 17.83 | -5.62% | -3.77% | 3.55 | 3.55 | 2.86 | 0.76 | -0.29 |
| `cross_candidate_hybrid:hybrid_v3_5` | `True` | 27.01% | 33.23% | 5/10 | -4.31% | -0.84% | 13.72% | 5.26% | 1.24 | 6.33 | 1.97 | 6.31 | -3.98% | -3.21% | 3.18 | 3.18 | 3.79 | 1.28 | 0.66 |

## Research overlays by comp

| Candidate | Clean | OOS comp | Ann. | Hit | Min | Latest | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | Calmar1 | Calmar2 | VaR5 | CVaR25 | PF | Omega | Tail | Skew | Kurt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `risk_enhanced_blend:dyn085_70_aware_v36tv_30` | `False` | 47.60% | 59.55% | 5/10 | -2.35% | -0.73% | 14.67% | 4.54% | 1.98 | 16.23 | 3.24 | 13.10 | -2.30% | -1.86% | 7.23 | 7.23 | 6.73 | 0.98 | -0.10 |
| `risk_enhanced_blend:dyn085_60_aware_v36tv_40` | `False` | 45.60% | 56.96% | 5/10 | -2.46% | -0.74% | 14.14% | 4.53% | 1.93 | 16.93 | 3.22 | 12.59 | -2.31% | -1.95% | 6.63 | 6.63 | 6.69 | 1.08 | 0.21 |
| `risk_enhanced_blend:dyn085_50_aware_v36tv_50` | `False` | 43.56% | 54.33% | 5/10 | -2.58% | -0.75% | 13.66% | 4.51% | 1.87 | 17.10 | 3.19 | 12.05 | -2.31% | -2.05% | 6.09 | 6.09 | 6.65 | 1.17 | 0.51 |
| `risk_enhanced_blend:dyn085_60_aware_v36tv_30_strict_growth_10` | `False` | 42.81% | 53.36% | 5/10 | -2.49% | -0.68% | 13.11% | 4.41% | 1.95 | 15.04 | 3.26 | 12.09 | -2.26% | -1.85% | 6.76 | 6.76 | 6.28 | 0.98 | -0.10 |
| `risk_enhanced_blend:dyn085_60_aware_v36_40` | `False` | 40.97% | 50.99% | 5/10 | -2.72% | -0.61% | 14.04% | 3.98% | 1.91 | 16.62 | 2.92 | 12.80 | -2.10% | -1.80% | 6.03 | 6.03 | 6.72 | 0.91 | -0.32 |
| `risk_enhanced_blend:dyn085_50_aware_v35_50` | `False` | 39.31% | 48.86% | 6/10 | -3.69% | -0.71% | 14.96% | 3.86% | 1.83 | 7.65 | 2.63 | 12.64 | -2.35% | -1.67% | 7.87 | 7.87 | 6.08 | 1.08 | 0.02 |
| `risk_enhanced_blend:dyn100_60_aware_v36tv_40` | `False` | 37.64% | 46.72% | 5/10 | -2.46% | -0.31% | 13.26% | 4.53% | 1.72 | 12.78 | 2.84 | 10.32 | -2.31% | -1.95% | 6.09 | 6.09 | 6.39 | 1.45 | 1.24 |
| `validation_selector:validation_calmar_mdd12` | `False` | 23.23% | 28.48% | 4/10 | -4.03% | -0.61% | 14.29% | 7.07% | 1.09 | 6.24 | 1.63 | 4.03 | -3.84% | -3.60% | 2.54 | 2.54 | 3.91 | 1.11 | -0.05 |
| `validation_selector:validation_sharpe_mdd10` | `False` | 17.31% | 21.12% | 5/10 | -6.06% | -0.61% | 9.86% | 6.06% | 0.95 | 2.89 | 1.76 | 3.48 | -4.90% | -3.74% | 2.33 | 2.33 | 2.68 | 0.97 | -0.19 |
| `validation_selector:validation_utility_mdd15` | `False` | -4.47% | -5.35% | 4/10 | -16.45% | -0.71% | 18.16% | 26.58% | 0.01 | 0.01 | -0.25 | -0.20 | -12.85% | -10.89% | 1.01 | 1.01 | 1.17 | 0.27 | -0.74 |

## All MDD <= 15% by comp

| Candidate | Clean | OOS comp | Ann. | Hit | Min | Latest | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | Calmar1 | Calmar2 | VaR5 | CVaR25 | PF | Omega | Tail | Skew | Kurt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `risk_enhanced_blend:dyn085_70_aware_v36tv_30` | `False` | 47.60% | 59.55% | 5/10 | -2.35% | -0.73% | 14.67% | 4.54% | 1.98 | 16.23 | 3.24 | 13.10 | -2.30% | -1.86% | 7.23 | 7.23 | 6.73 | 0.98 | -0.10 |
| `risk_enhanced_blend:dyn085_60_aware_v36tv_40` | `False` | 45.60% | 56.96% | 5/10 | -2.46% | -0.74% | 14.14% | 4.53% | 1.93 | 16.93 | 3.22 | 12.59 | -2.31% | -1.95% | 6.63 | 6.63 | 6.69 | 1.08 | 0.21 |
| `risk_enhanced_blend:dyn085_50_aware_v36tv_50` | `False` | 43.56% | 54.33% | 5/10 | -2.58% | -0.75% | 13.66% | 4.51% | 1.87 | 17.10 | 3.19 | 12.05 | -2.31% | -2.05% | 6.09 | 6.09 | 6.65 | 1.17 | 0.51 |
| `risk_enhanced_blend:dyn085_60_aware_v36tv_30_strict_growth_10` | `False` | 42.81% | 53.36% | 5/10 | -2.49% | -0.68% | 13.11% | 4.41% | 1.95 | 15.04 | 3.26 | 12.09 | -2.26% | -1.85% | 6.76 | 6.76 | 6.28 | 0.98 | -0.10 |
| `risk_enhanced_blend:dyn085_60_aware_v36_40` | `False` | 40.97% | 50.99% | 5/10 | -2.72% | -0.61% | 14.04% | 3.98% | 1.91 | 16.62 | 2.92 | 12.80 | -2.10% | -1.80% | 6.03 | 6.03 | 6.72 | 0.91 | -0.32 |
| `risk_enhanced_blend:dyn085_50_aware_v35_50` | `False` | 39.31% | 48.86% | 6/10 | -3.69% | -0.71% | 14.96% | 3.86% | 1.83 | 7.65 | 2.63 | 12.64 | -2.35% | -1.67% | 7.87 | 7.87 | 6.08 | 1.08 | 0.02 |
| `risk_enhanced_blend:dyn100_60_aware_v36tv_40` | `False` | 37.64% | 46.72% | 5/10 | -2.46% | -0.31% | 13.26% | 4.53% | 1.72 | 12.78 | 2.84 | 10.32 | -2.31% | -1.95% | 6.09 | 6.09 | 6.39 | 1.45 | 1.24 |
| `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `True` | 32.94% | 40.73% | 5/10 | -3.19% | -0.78% | 11.22% | 4.46% | 1.45 | 10.16 | 2.94 | 9.12 | -3.12% | -2.65% | 4.10 | 4.10 | 4.76 | 1.43 | 1.23 |
| `cross_candidate_hybrid:hybrid_v3_5` | `True` | 27.01% | 33.23% | 5/10 | -4.31% | -0.84% | 13.72% | 5.26% | 1.24 | 6.33 | 1.97 | 6.31 | -3.98% | -3.21% | 3.18 | 3.18 | 3.79 | 1.28 | 0.66 |
| `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `True` | 25.74% | 31.63% | 5/10 | -5.39% | -0.83% | 13.81% | 7.81% | 1.16 | 4.23 | 1.86 | 4.05 | -5.02% | -4.17% | 2.81 | 2.81 | 3.09 | 1.03 | -0.01 |

## Clean MDD <= 15% by comp

| Candidate | Clean | OOS comp | Ann. | Hit | Min | Latest | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | Calmar1 | Calmar2 | VaR5 | CVaR25 | PF | Omega | Tail | Skew | Kurt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `True` | 32.94% | 40.73% | 5/10 | -3.19% | -0.78% | 11.22% | 4.46% | 1.45 | 10.16 | 2.94 | 9.12 | -3.12% | -2.65% | 4.10 | 4.10 | 4.76 | 1.43 | 1.23 |
| `cross_candidate_hybrid:hybrid_v3_5` | `True` | 27.01% | 33.23% | 5/10 | -4.31% | -0.84% | 13.72% | 5.26% | 1.24 | 6.33 | 1.97 | 6.31 | -3.98% | -3.21% | 3.18 | 3.18 | 3.79 | 1.28 | 0.66 |
| `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `True` | 25.74% | 31.63% | 5/10 | -5.39% | -0.83% | 13.81% | 7.81% | 1.16 | 4.23 | 1.86 | 4.05 | -5.02% | -4.17% | 2.81 | 2.81 | 3.09 | 1.03 | -0.01 |
| `dynamic_aware_hybrid:hybrid_v3_5` | `True` | 25.12% | 30.86% | 6/10 | -5.62% | -0.71% | 13.84% | 5.62% | 1.22 | 3.29 | 1.81 | 5.50 | -5.53% | -4.13% | 2.89 | 2.89 | 2.44 | 0.76 | -0.36 |
| `dynamic_aware_hybrid:hybrid_v3_6` | `True` | 22.50% | 27.57% | 6/10 | -3.81% | -0.45% | 11.12% | 3.81% | 1.31 | 4.89 | 2.02 | 7.24 | -3.69% | -3.52% | 2.99 | 2.99 | 3.14 | 0.92 | 0.08 |
| `cross_candidate_hybrid:hybrid_v3_6` | `True` | 22.07% | 27.04% | 5/10 | -4.01% | -0.71% | 13.85% | 7.22% | 1.07 | 6.27 | 1.59 | 3.75 | -3.71% | -3.41% | 2.64 | 2.64 | 4.01 | 1.14 | -0.07 |
| `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `True` | 19.72% | 24.10% | 4/10 | -5.84% | -0.77% | 12.73% | 10.31% | 0.91 | 3.33 | 1.55 | 2.34 | -5.35% | -4.28% | 2.28 | 2.28 | 2.80 | 1.35 | 1.01 |
| `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | `True` | 6.97% | 8.42% | 4/10 | -3.58% | -0.17% | 7.32% | 3.70% | 0.66 | 1.84 | 0.95 | 2.27 | -3.00% | -2.29% | 1.98 | 1.98 | 2.47 | 1.65 | 2.08 |
| `individual_robust:hybrid_v3_5` | `True` | 1.42% | 1.71% | 4/10 | -5.24% | -1.85% | 8.40% | 13.22% | 0.17 | 0.74 | 0.17 | 0.13 | -5.13% | -4.72% | 1.18 | 1.18 | 2.38 | 2.04 | 3.18 |
| `strict_efficiency:hybrid_v3_5` | `True` | -1.31% | -1.56% | 3/10 | -4.64% | -0.03% | 6.20% | 5.36% | -0.06 | -0.15 | -0.21 | -0.29 | -3.76% | -3.27% | 0.95 | 0.95 | 1.64 | 1.39 | 1.51 |

## Fold winners
| Fold | Best | OOS | OOS MDD | Risk overlays | Selectors |
| --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `individual_robust:hybrid_v3_5` | 3.10% | 5.79% | 7 | 3 |
| `2025-10` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | 6.41% | 7.04% | 7 | 3 |
| `2025-11` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 12.20% | 10.08% | 7 | 3 |
| `2025-12` | `dynamic_aware_hybrid:hybrid_v3_5` | 0.20% | 4.64% | 7 | 3 |
| `2026-01` | `relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna` | 16.03% | 9.34% | 7 | 3 |
| `2026-02` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | 20.90% | 5.88% | 7 | 3 |
| `2026-03` | `cross_candidate_hybrid:hybrid_v3_5` | 2.63% | 3.14% | 7 | 3 |
| `2026-04` | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | -0.40% | 1.65% | 7 | 3 |
| `2026-05` | `profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna` | 16.58% | 9.96% | 7 | 3 |
| `2026-06` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 0.00% | 0.00% | 7 | 3 |

## Audits
- `metric_reconciliation`: `{"candidate_count": 64, "metrics_reconciled": true, "mismatches": []}`
- `dynamic_self_feed_audit`: `{"no_same_month_dynamic_self_feeding": true, "rule": "same_fold_dynamic_switch_label_oos_utility_or_oracle_rank_not_used", "violations": []}`
- `bridge_protocol_audit`: `{"current_fold_oos_used_for_bridge_weighting": false, "manifest_frozen_before_bridge_evaluation": true, "post_oos_expansion_for_same_protocol": false, "same_month_dynamic_self_feeding": false}`
- `online_weight_audit`: `{"fully_lagged_online_weights": true, "rule": "month_m_weights_use_only_completed_months_before_m", "violating_months": []}`
- `promotability`: `{"if_false_recommendation": "fresh_forward_shadow_required_before_promotion", "promotable": false, "promotion_hard_stop_pass": false, "promotion_hard_stop_reasons": ["blocked_non_clean_research_variant"]}`

## Guardrail
Research overlays are not real-money deployable from this same backtest. They need fresh forward/paper fills and slippage telemetry before promotion.
