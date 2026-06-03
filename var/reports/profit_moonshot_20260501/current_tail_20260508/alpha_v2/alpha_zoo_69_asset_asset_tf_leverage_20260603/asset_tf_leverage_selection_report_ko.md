# Asset/Timeframe Leverage Scaling WF Report

- 기간: `2025-09-01T00:00:00` ~ `2026-06-01T06:30:00`
- 조건: 69 assets, timeframes `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`, monthly day-1 refit, validation 2 months, slippage `10.0bps`
- 추가 축: source sleeve의 `symbol x timeframe x integer_leverage` 위에 train/validation-only post-allocation multiplier를 튜닝
- 런타임: peak RSS `1085.0 MiB`, 완료 `2026-06-03T13:21:32.027800Z`
- 감사: dynamic={'no_same_month_dynamic_self_feeding': True, 'rule': 'same_fold_dynamic_switch_label_oos_utility_or_oracle_rank_not_used', 'violations': []}, reconciliation={'candidate_count': 28, 'metrics_reconciled': True, 'mismatches': []}

## Asset/timeframe leverage 후보 성과
| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Worst | Latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asset_timeframe_leverage:hybrid_v3_6 | Y | 21.78% | 26.67% | 21.97% | 21.19% | 0.72 | 1.67 | 1.79 | 4/10 | -15.87% | -0.57% |
| asset_timeframe_leverage:selected_optuna | Y | 18.23% | 22.25% | 29.60% | 27.75% | 0.60 | 1.18 | 1.65 | 4/10 | -22.90% | -0.31% |
| asset_timeframe_leverage:hybrid_v3_5 | Y | 17.82% | 21.75% | 29.60% | 27.75% | 0.60 | 1.11 | 1.68 | 4/10 | -22.90% | -0.31% |
| asset_timeframe_leverage:asset_tf_leverage_growth_mdd16_gross6_core22 | Y | 14.17% | 17.23% | 26.61% | 23.40% | 0.50 | 1.92 | 1.65 | 3/10 | -12.26% | -0.63% |
| asset_timeframe_leverage:static_guarded | Y | 13.86% | 16.86% | 26.61% | 23.61% | 0.50 | 1.90 | 1.64 | 3/10 | -12.26% | -0.63% |
| asset_timeframe_leverage:asset_tf_leverage_balanced_mdd12_gross4_core16 | Y | 3.88% | 4.67% | 23.23% | 18.17% | 0.28 | 0.78 | 1.24 | 3/10 | -8.95% | -0.92% |
| asset_timeframe_leverage:selected_train_validation_legal | Y | -7.25% | -8.64% | 29.60% | 32.66% | 0.05 | 0.09 | 1.04 | 3/10 | -22.90% | -0.31% |

## 같은 run 내 clean 상위
| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Worst | Latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_candidate_hybrid:hybrid_v3_6 | Y | 61.61% | 77.89% | 27.57% | 14.83% | 1.55 | 4.28 | 4.47 | 7/10 | -9.48% | -0.57% |
| cross_candidate_hybrid:hybrid_v3_6_train_validation_fit | Y | 54.21% | 68.17% | 28.28% | 16.83% | 1.40 | 4.01 | 3.76 | 5/10 | -9.97% | -0.47% |
| cross_candidate_hybrid:hybrid_v3_5_train_validation_fit | Y | 52.06% | 65.35% | 25.76% | 13.88% | 1.38 | 3.95 | 3.53 | 6/10 | -10.69% | -0.41% |
| cross_candidate_hybrid:hybrid_v3_5 | Y | 48.42% | 60.62% | 26.05% | 14.49% | 1.32 | 3.46 | 3.40 | 6/10 | -11.38% | -0.47% |
| profile_optuna:hybrid_v3_6 | Y | 35.90% | 44.50% | 29.29% | 24.45% | 0.92 | 2.47 | 2.28 | 3/10 | -16.14% | -0.32% |
| asset_timeframe_leverage:hybrid_v3_6 | Y | 21.78% | 26.67% | 21.97% | 21.19% | 0.72 | 1.67 | 1.79 | 4/10 | -15.87% | -0.57% |
| asset_timeframe_leverage:selected_optuna | Y | 18.23% | 22.25% | 29.60% | 27.75% | 0.60 | 1.18 | 1.65 | 4/10 | -22.90% | -0.31% |
| asset_timeframe_leverage:hybrid_v3_5 | Y | 17.82% | 21.75% | 29.60% | 27.75% | 0.60 | 1.11 | 1.68 | 4/10 | -22.90% | -0.31% |
| profile_optuna:selected_optuna | Y | 16.35% | 19.93% | 28.74% | 23.34% | 0.61 | 1.21 | 1.75 | 4/10 | -16.14% | -0.32% |
| asset_timeframe_leverage:asset_tf_leverage_growth_mdd16_gross6_core22 | Y | 14.17% | 17.23% | 26.61% | 23.40% | 0.50 | 1.92 | 1.65 | 3/10 | -12.26% | -0.63% |
| asset_timeframe_leverage:static_guarded | Y | 13.86% | 16.86% | 26.61% | 23.61% | 0.50 | 1.90 | 1.64 | 3/10 | -12.26% | -0.63% |
| meta_portfolio:validation_calmar_top5_capped | Y | 9.63% | 11.67% | 31.89% | 22.55% | 0.48 | 0.97 | 1.46 | 3/10 | -13.73% | -0.32% |

## 기존 final 후보와 비교 기준
| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Worst | Latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relaxed_efficiency:hybrid_v3_5 | Y | 156.03% | 209.00% | 19.75% | 15.66% | 1.69 | 10.48 | 7.04 | 5/10 | -8.41% | -0.09% |
| fixed_relaxed_dynamic_blend:relaxed70_dynamic30 | N | 122.36% | 160.90% | 16.66% | 14.66% | 1.74 | 8.97 | 6.33 | 6/10 | -8.42% | -0.17% |
| fixed_relaxed_dynamic_blend:relaxed60_dynamic40 | N | 111.75% | 146.03% | 16.19% | 14.35% | 1.75 | 8.43 | 6.00 | 6/10 | -8.61% | -0.20% |
| dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit | Y | 54.76% | 68.89% | 16.75% | 12.76% | 1.53 | 4.43 | 3.87 | 5/10 | -9.97% | -0.38% |

## 월별 OOS 분포
| Fold | asset_tf v3.6 | asset_tf v3.5 | balanced | growth |
| --- | --- | --- | --- | --- |
| 2025-09 | 8.25% | 5.12% | 6.90% | 6.31% |
| 2025-10 | -6.11% | -2.84% | -8.95% | -7.21% |
| 2025-11 | 27.89% | 35.14% | 16.97% | 41.62% |
| 2025-12 | -5.55% | -7.03% | -0.70% | -1.08% |
| 2026-01 | 18.25% | 17.27% | 12.27% | 7.86% |
| 2026-02 | 6.44% | 8.38% | -6.44% | -4.14% |
| 2026-03 | -5.28% | -2.91% | -4.55% | -3.69% |
| 2026-04 | -0.53% | -3.18% | -2.07% | -4.84% |
| 2026-05 | -15.87% | -22.90% | -5.56% | -12.26% |
| 2026-06 | -0.57% | -0.31% | -0.92% | -0.63% |

## Exposure breadth
| 후보 | Unique assets | Avg active/fold | Active range | Avg gross | Gross range |
| --- | --- | --- | --- | --- | --- |
| asset_timeframe_leverage:hybrid_v3_6 | 22 | 10.6 | 8~18 | 2.66 | 1.59~4.00 |
| asset_timeframe_leverage:hybrid_v3_5 | 22 | 10.6 | 8~18 | 2.75 | 1.65~4.00 |

## 결론
- Rebalancing/leverage는 source 단계에서 일부 튜닝되어 있었고, 이번에 asset/timeframe post-allocation multiplier 축을 clean하게 추가 검증했다.
- 하지만 새 축 단독 최고 `asset_timeframe_leverage:hybrid_v3_6`은 OOS comp 21.78%, MDD 21.97%, Sharpe 0.72로 기존 final 후보를 대체하지 못한다.
- 따라서 이 축은 최종 코어에 바로 넣기보다, 노출/레버리지 진단 및 forward monitor 보조 후보로 두는 것이 맞다.
- 기존 선택 후보는 유지: 실전 균형 shadow는 `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`, clean-only 최고는 `relaxed_efficiency:hybrid_v3_5`.
