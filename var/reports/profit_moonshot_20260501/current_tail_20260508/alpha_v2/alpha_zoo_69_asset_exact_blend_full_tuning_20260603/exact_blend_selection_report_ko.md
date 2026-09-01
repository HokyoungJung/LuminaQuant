# 69-Asset Monthly Refit WF: Exact Relaxed/Dynamic Blend Selection Report

- 생성: `2026-06-03T11:28:50.811187Z` / 완료: `2026-06-03T11:28:50.862470Z`
- 데이터: `2025-01-01T00:00:00` ~ `2026-06-01T06:30:00`, universe `69` assets, timeframes `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- 프로토콜: monthly day-1 refit, train expanding, validation previous 2 calendar months, locked OOS next 1 calendar month, slippage `10.0bps`
- Trials/seed: `{'asset_trials': 12, 'hybrid_trials': 192, 'profile_trials': 72, 'seed': 20260602}`; runner peak RSS `1121.3 MiB`
- 누수 감사: bridge={'current_fold_oos_used_for_bridge_weighting': False, 'manifest_frozen_before_bridge_evaluation': True, 'post_oos_expansion_for_same_protocol': False, 'same_month_dynamic_self_feeding': False}, dynamic={'no_same_month_dynamic_self_feeding': True, 'rule': 'same_fold_dynamic_switch_label_oos_utility_or_oracle_rank_not_used', 'violations': []}, online={'fully_lagged_online_weights': True, 'rule': 'month_m_weights_use_only_completed_months_before_m', 'violating_months': []}, metric_reconciliation={'candidate_count': 80, 'metrics_reconciled': True, 'mismatches': []}

## Exact bar-level relaxed/dynamic blend candidates
| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF/Omega | Hit | Worst month | Latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_relaxed_dynamic_blend:relaxed70_dynamic30 | N | 122.36% | 160.90% | 16.66% | 14.66% | 1.74 | 8.97 | 6.33 | 6/10 | -8.42% | -0.17% |
| fixed_relaxed_dynamic_blend:relaxed60_dynamic40 | N | 111.75% | 146.03% | 16.19% | 14.35% | 1.75 | 8.43 | 6.00 | 6/10 | -8.61% | -0.20% |
| fixed_relaxed_dynamic_blend:relaxed50_dynamic50 | N | 101.45% | 131.74% | 15.73% | 14.05% | 1.75 | 7.66 | 5.66 | 5/10 | -8.81% | -0.23% |
| fixed_relaxed_dynamic_blend:relaxed40_dynamic60 | N | 91.47% | 118.03% | 15.26% | 13.77% | 1.75 | 7.07 | 5.30 | 5/10 | -9.02% | -0.26% |
| fixed_relaxed_dynamic_blend:relaxed30_dynamic70 | N | 81.81% | 104.90% | 14.83% | 13.50% | 1.73 | 6.43 | 4.94 | 5/10 | -9.24% | -0.29% |

## Clean candidates by compounded OOS return
| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF/Omega | Hit | Worst month | Latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relaxed_efficiency:hybrid_v3_5 | Y | 156.03% | 209.00% | 19.75% | 15.66% | 1.69 | 10.48 | 7.04 | 5/10 | -8.41% | -0.09% |
| relaxed_efficiency:selected_optuna | Y | 60.18% | 76.01% | 24.27% | 31.69% | 1.12 | 2.32 | 2.70 | 5/10 | -22.55% | -0.09% |
| cross_candidate_hybrid:hybrid_v3_6_train_validation_fit | Y | 55.03% | 69.23% | 19.78% | 21.06% | 1.33 | 3.25 | 2.92 | 5/10 | -13.65% | -0.44% |
| dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit | Y | 54.76% | 68.89% | 16.75% | 12.76% | 1.53 | 4.43 | 3.87 | 5/10 | -9.97% | -0.38% |
| cross_candidate_hybrid:hybrid_v3_5_train_validation_fit | Y | 53.99% | 67.88% | 26.22% | 14.08% | 1.49 | 4.21 | 3.74 | 6/10 | -10.11% | -0.40% |
| cross_candidate_hybrid:hybrid_v3_5 | Y | 51.20% | 64.23% | 26.41% | 14.85% | 1.41 | 3.92 | 3.36 | 6/10 | -10.73% | -0.44% |
| cross_candidate_hybrid:hybrid_v3_6 | Y | 50.68% | 63.56% | 27.34% | 19.49% | 1.30 | 3.04 | 3.03 | 6/10 | -13.23% | -0.39% |
| dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit | Y | 47.12% | 58.93% | 16.55% | 13.35% | 1.37 | 4.17 | 3.37 | 5/10 | -9.60% | -0.26% |
| dynamic_aware_hybrid:hybrid_v3_5 | Y | 46.09% | 57.59% | 20.25% | 10.56% | 1.36 | 5.87 | 3.36 | 5/10 | -7.74% | -0.58% |
| dynamic_conviction_switch:t0.85_strict_fallback | Y | 44.31% | 55.29% | 29.29% | 24.09% | 1.11 | 4.23 | 2.52 | 4/10 | -10.73% | -0.32% |
| dynamic_conviction_switch:t0.90_strict_fallback | Y | 44.31% | 55.29% | 29.29% | 24.09% | 1.11 | 4.23 | 2.52 | 4/10 | -10.73% | -0.32% |
| dynamic_conviction_switch:t0.95_strict_fallback | Y | 44.31% | 55.29% | 29.29% | 24.09% | 1.11 | 4.23 | 2.52 | 4/10 | -10.73% | -0.32% |

## All top candidates by compounded OOS return
| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF/Omega | Hit | Worst month | Latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relaxed_efficiency:hybrid_v3_5 | Y | 156.03% | 209.00% | 19.75% | 15.66% | 1.69 | 10.48 | 7.04 | 5/10 | -8.41% | -0.09% |
| fixed_relaxed_dynamic_blend:relaxed70_dynamic30 | N | 122.36% | 160.90% | 16.66% | 14.66% | 1.74 | 8.97 | 6.33 | 6/10 | -8.42% | -0.17% |
| fixed_relaxed_dynamic_blend:relaxed60_dynamic40 | N | 111.75% | 146.03% | 16.19% | 14.35% | 1.75 | 8.43 | 6.00 | 6/10 | -8.61% | -0.20% |
| fixed_relaxed_dynamic_blend:relaxed50_dynamic50 | N | 101.45% | 131.74% | 15.73% | 14.05% | 1.75 | 7.66 | 5.66 | 5/10 | -8.81% | -0.23% |
| fixed_relaxed_dynamic_blend:relaxed40_dynamic60 | N | 91.47% | 118.03% | 15.26% | 13.77% | 1.75 | 7.07 | 5.30 | 5/10 | -9.02% | -0.26% |
| fixed_relaxed_dynamic_blend:relaxed30_dynamic70 | N | 81.81% | 104.90% | 14.83% | 13.50% | 1.73 | 6.43 | 4.94 | 5/10 | -9.24% | -0.29% |
| mdd30_barbell_blend:dyn085_70_strict_growth_30_x1_50 | N | 70.16% | 189.55% | 31.57% | 14.41% | 1.52 | 6.15 | 4.57 | 2/6 | -14.41% | -0.34% |
| relaxed_efficiency:selected_optuna | Y | 60.18% | 76.01% | 24.27% | 31.69% | 1.12 | 2.32 | 2.70 | 5/10 | -22.55% | -0.09% |
| mdd30_risk_scaled:cross_v35_x1_50 | N | 56.88% | 194.71% | 23.51% | 15.86% | 1.77 | 0.00 | 4.51 | 4/5 | -15.86% | -15.86% |
| cross_candidate_hybrid:hybrid_v3_6_train_validation_fit | Y | 55.03% | 69.23% | 19.78% | 21.06% | 1.33 | 3.25 | 2.92 | 5/10 | -13.65% | -0.44% |
| dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit | Y | 54.76% | 68.89% | 16.75% | 12.76% | 1.53 | 4.43 | 3.87 | 5/10 | -9.97% | -0.38% |
| cross_candidate_hybrid:hybrid_v3_5_train_validation_fit | Y | 53.99% | 67.88% | 26.22% | 14.08% | 1.49 | 4.21 | 3.74 | 6/10 | -10.11% | -0.40% |

## Monthly OOS return distribution
| Fold | Relaxed v3.5 | Dynamic-aware v3.5 TV-fit | Blend 70/30 | Blend 60/40 | Blend 50/50 |
| --- | --- | --- | --- | --- | --- |
| 2025-09 | -0.78% | 2.65% | 0.24% | 0.59% | 0.93% |
| 2025-10 | 0.39% | -0.47% | 0.15% | 0.07% | -0.01% |
| 2025-11 | 63.19% | 22.76% | 50.03% | 45.85% | 41.77% |
| 2025-12 | -2.59% | -3.48% | -2.82% | -2.90% | -2.99% |
| 2026-01 | 16.60% | 24.60% | 19.00% | 19.80% | 20.60% |
| 2026-02 | 47.58% | 7.43% | 34.77% | 30.64% | 26.59% |
| 2026-03 | -8.41% | -3.10% | -6.81% | -6.28% | -5.75% |
| 2026-04 | -7.91% | -9.97% | -8.42% | -8.61% | -8.81% |
| 2026-05 | 11.52% | 9.88% | 11.18% | 11.04% | 10.88% |
| 2026-06 | -0.09% | -0.38% | -0.17% | -0.20% | -0.23% |

## Reconstructed exposure breadth for exact blends
| Blend | Unique assets used across folds | Avg active/fold | Active range | Avg gross | Gross range |
| --- | --- | --- | --- | --- | --- |
| 60/40 | 29 | 13.7 | 10~29 | 1.92 | 1.35~2.58 |
| 70/30 | 29 | 13.7 | 10~29 | 2.15 | 1.51~2.97 |

## Selection call
- 성능 극대화만 보면 `fixed_relaxed_dynamic_blend:relaxed70_dynamic30`이 OOS comp 122.36%로 최고입니다.
- 리스크/실전 후보 균형은 `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`이 더 낫습니다: comp 111.75%, max OOS MDD 16.19%, monthly equity MDD 14.35%, Sharpe 1.75, Hit 6/10.
- 단, exact blend 후보들은 OOS 리뷰 이후 추가한 조합이라 현재 창에서는 clean promotion이 아니라 `fresh_forward_shadow_required_before_promotion`입니다.
- 현재 창에서 clean 후보만 고르면 `relaxed_efficiency:hybrid_v3_5`가 comp 156.03%로 최상위이나 max OOS MDD 19.75%와 hit 5/10이라 실전 안정성은 60/40 shadow보다 떨어집니다.
