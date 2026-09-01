# Alpha Zoo top-seed Hybrid v3.5/v3.6 cost validation

- generated_at_utc: `2026-05-19T10:00:01.512339Z`
- real_money_execution_attempted: `false`
- hybrid objective/fitting inputs: `train`, `validation` only
- locked-OOS role: gate/report-only after seed/hybrid freeze
- cost scenarios: round-trip slippage/fee `5bps` and `10bps`

## Seed universe

- deduped seed count: `16`

- 1. `alpha_zoo_fast_residual 6x/0.175` via `live_oos_calmar, live_oos_return`
- 2. `alpha_zoo_fast_residual 7x/0.15` via `live_oos_calmar, live_oos_return`
- 3. `alpha_zoo_fast_residual 5x/0.2` via `live_oos_calmar, live_oos_return`
- 4. `alpha_zoo_fast_residual 6x/0.05` via `live_oos_sharpe, live_oos_smart_sortino, live_oos_sortino`
- 5. `alpha_zoo_fast_residual 3x/0.1` via `live_oos_sharpe`
- 6. `alpha_zoo_fast_residual 4x/0.1` via `live_oos_sharpe`
- 7. `alpha_zoo_fast_residual 5x/0.05` via `live_oos_smart_sortino, live_oos_sortino`
- 8. `alpha_zoo_fast_residual 2x/0.125` via `live_oos_smart_sortino, live_oos_sortino`
- 9. `alpha_zoo_quality_single_pair 7x/0.2` via `filtered_validation_return, live_full_compound`
- 10. `alpha_zoo_quality_single_pair 7x/0.175` via `live_full_compound`
- 11. `alpha_zoo_quality_single_pair 6x/0.2` via `live_full_compound`
- 12. `alpha_zoo_high_confidence_single_pair 7x/0.2` via `filtered_balanced_score, filtered_oos_calmar, filtered_oos_return`
- 13. `alpha_zoo_high_confidence_single_pair 7x/0.175` via `filtered_balanced_score, filtered_oos_calmar, filtered_oos_return`
- 14. `alpha_zoo_high_confidence_single_pair 6x/0.2` via `filtered_balanced_score, filtered_oos_calmar, filtered_oos_return`
- 15. `alpha_zoo_quality_single_pair 10x/0.15` via `filtered_validation_return`
- 16. `alpha_zoo_high_confidence_long_only 7x/0.2` via `filtered_validation_return`

## Cost metrics

| cost bps | model | role | split | return | MDD | Sharpe | Sortino | Smart Sortino | Calmar | events | liq | wipeout | min buffer | OOS gate |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | `hybrid_v3_5_seed_union` | hybrid_seed_union | train | +49.19% | +30.09% | 1.0815 | 0.8728 | 0.6710 | 1.6349 | 3574 | 0 | 0 | 8575.1074 | True |
| 5 | `hybrid_v3_5_seed_union` | hybrid_seed_union | validation | +21.16% | +12.33% | 2.3486 | 2.0157 | 1.7943 | 9.5787 | 872 | 0 | 0 | 13977.1680 | True |
| 5 | `hybrid_v3_5_seed_union` | hybrid_seed_union | locked_oos | +3.29% | +15.39% | 0.9450 | 0.8602 | 0.7455 | 1.8827 | 471 | 0 | 0 | 17127.5409 | True |
| 5 | `hybrid_v3_6_seed_union` | hybrid_seed_union | train | -7.98% | +35.42% | 0.0561 | 0.0402 | 0.0297 | -0.2252 | 3574 | 0 | 0 | 6762.4075 | False |
| 5 | `hybrid_v3_6_seed_union` | hybrid_seed_union | validation | +8.21% | +10.70% | 1.2272 | 1.0572 | 0.9550 | 3.5291 | 872 | 0 | 0 | 8102.8769 | False |
| 5 | `hybrid_v3_6_seed_union` | hybrid_seed_union | locked_oos | -2.90% | +12.13% | -0.8589 | -0.6274 | -0.5596 | -1.7021 | 471 | 0 | 0 | 9394.4171 | False |
| 5 | `seed_alpha_zoo_fast_residual_6x_0p175` | seed_universe | train | -80.10% | +85.71% | -2.1469 | -1.3791 | -0.7426 | -0.9345 | 3102 | 0 | 0 | 8514.1183 | True |
| 5 | `seed_alpha_zoo_fast_residual_6x_0p175` | seed_universe | validation | -4.27% | +22.45% | 0.0228 | 0.0155 | 0.0127 | -0.7235 | 767 | 0 | 0 | 9066.5742 | True |
| 5 | `seed_alpha_zoo_fast_residual_6x_0p175` | seed_universe | locked_oos | +7.63% | +15.47% | 1.5356 | 1.0791 | 0.9345 | 5.0511 | 391 | 0 | 0 | 9383.4608 | True |
| 5 | `seed_alpha_zoo_fast_residual_7x_0p15` | seed_universe | train | -80.10% | +85.71% | -2.1469 | -1.3791 | -0.7426 | -0.9345 | 3102 | 0 | 0 | 8514.1183 | True |
| 5 | `seed_alpha_zoo_fast_residual_7x_0p15` | seed_universe | validation | -4.27% | +22.45% | 0.0228 | 0.0155 | 0.0127 | -0.7235 | 767 | 0 | 0 | 9066.5742 | True |
| 5 | `seed_alpha_zoo_fast_residual_7x_0p15` | seed_universe | locked_oos | +7.63% | +15.47% | 1.5356 | 1.0791 | 0.9345 | 5.0511 | 391 | 0 | 0 | 9383.4608 | True |
| 5 | `seed_alpha_zoo_fast_residual_5x_0p2` | seed_universe | train | -78.28% | +84.20% | -2.1455 | -1.3782 | -0.7482 | -0.9297 | 3102 | 0 | 0 | 8584.8746 | True |
| 5 | `seed_alpha_zoo_fast_residual_5x_0p2` | seed_universe | validation | -3.88% | +21.47% | 0.0220 | 0.0150 | 0.0123 | -0.6914 | 767 | 0 | 0 | 9111.0230 | True |
| 5 | `seed_alpha_zoo_fast_residual_5x_0p2` | seed_universe | locked_oos | +7.31% | +14.78% | 1.5352 | 1.0788 | 0.9399 | 5.0097 | 391 | 0 | 0 | 9412.8198 | True |
| 5 | `seed_alpha_zoo_fast_residual_6x_0p05` | seed_universe | train | -33.93% | +40.66% | -2.1251 | -1.3652 | -0.9705 | -0.8343 | 3102 | 0 | 0 | 9575.4624 | True |
| 5 | `seed_alpha_zoo_fast_residual_6x_0p05` | seed_universe | validation | -0.34% | +6.82% | 0.0106 | 0.0072 | 0.0068 | -0.2017 | 767 | 0 | 0 | 9733.3069 | True |
| 5 | `seed_alpha_zoo_fast_residual_6x_0p05` | seed_universe | locked_oos | +2.36% | +4.61% | 1.5305 | 1.0747 | 1.0273 | 4.3688 | 391 | 0 | 0 | 9823.8459 | True |
| 5 | `seed_alpha_zoo_fast_residual_3x_0p1` | seed_universe | train | -33.93% | +40.66% | -2.1251 | -1.3652 | -0.9705 | -0.8343 | 3102 | 0 | 0 | 9575.4624 | True |
| 5 | `seed_alpha_zoo_fast_residual_3x_0p1` | seed_universe | validation | -0.34% | +6.82% | 0.0106 | 0.0072 | 0.0068 | -0.2017 | 767 | 0 | 0 | 9733.3069 | True |
| 5 | `seed_alpha_zoo_fast_residual_3x_0p1` | seed_universe | locked_oos | +2.36% | +4.61% | 1.5305 | 1.0747 | 1.0273 | 4.3688 | 391 | 0 | 0 | 9823.8459 | True |
| 5 | `seed_alpha_zoo_fast_residual_4x_0p1` | seed_universe | train | -42.93% | +50.44% | -2.1280 | -1.3670 | -0.9087 | -0.8511 | 3102 | 0 | 0 | 9433.9498 | True |
| 5 | `seed_alpha_zoo_fast_residual_4x_0p1` | seed_universe | validation | -0.61% | +9.02% | 0.0123 | 0.0083 | 0.0076 | -0.2743 | 767 | 0 | 0 | 9644.4092 | True |
| 5 | `seed_alpha_zoo_fast_residual_4x_0p1` | seed_universe | locked_oos | +3.12% | +6.12% | 1.5311 | 1.0753 | 1.0133 | 4.4654 | 391 | 0 | 0 | 9765.1279 | True |
| 5 | `seed_alpha_zoo_fast_residual_5x_0p05` | seed_universe | train | -29.02% | +35.15% | -2.1237 | -1.3642 | -1.0094 | -0.8255 | 3102 | 0 | 0 | 9646.2186 | True |
| 5 | `seed_alpha_zoo_fast_residual_5x_0p05` | seed_universe | validation | -0.23% | +5.70% | 0.0098 | 0.0067 | 0.0063 | -0.1654 | 767 | 0 | 0 | 9777.7558 | True |
| 5 | `seed_alpha_zoo_fast_residual_5x_0p05` | seed_universe | locked_oos | +1.98% | +3.85% | 1.5301 | 1.0744 | 1.0345 | 4.3201 | 391 | 0 | 0 | 9853.2049 | True |
| 5 | `seed_alpha_zoo_fast_residual_2x_0p125` | seed_universe | train | -29.02% | +35.15% | -2.1237 | -1.3642 | -1.0094 | -0.8255 | 3102 | 0 | 0 | 9646.2186 | True |
| 5 | `seed_alpha_zoo_fast_residual_2x_0p125` | seed_universe | validation | -0.23% | +5.70% | 0.0098 | 0.0067 | 0.0063 | -0.1654 | 767 | 0 | 0 | 9777.7558 | True |
| 5 | `seed_alpha_zoo_fast_residual_2x_0p125` | seed_universe | locked_oos | +1.98% | +3.85% | 1.5301 | 1.0744 | 1.0345 | 4.3201 | 391 | 0 | 0 | 9853.2049 | True |
| 5 | `seed_alpha_zoo_quality_single_pair_7x_0p2` | seed_universe | train | +65.46% | +66.46% | 1.0034 | 0.4959 | 0.2979 | 0.9849 | 1416 | 0 | 0 | 8018.8244 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_7x_0p2` | seed_universe | validation | +6.96% | +25.34% | 0.7394 | 0.3501 | 0.2793 | 1.2415 | 334 | 0 | 0 | 9002.3812 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_7x_0p2` | seed_universe | locked_oos | -3.59% | +28.64% | -0.2622 | -0.1335 | -0.1037 | -0.8710 | 183 | 0 | 0 | 9177.9477 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_7x_0p175` | seed_universe | train | +61.69% | +61.22% | 1.0034 | 0.4955 | 0.3073 | 1.0077 | 1416 | 0 | 0 | 8266.4714 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_7x_0p175` | seed_universe | validation | +6.64% | +22.50% | 0.7379 | 0.3493 | 0.2851 | 1.3259 | 334 | 0 | 0 | 9127.0835 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_7x_0p175` | seed_universe | locked_oos | -2.97% | +25.53% | -0.2654 | -0.1351 | -0.1076 | -0.8268 | 183 | 0 | 0 | 9280.7042 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_6x_0p2` | seed_universe | train | +61.02% | +60.42% | 1.0034 | 0.4954 | 0.3088 | 1.0099 | 1416 | 0 | 0 | 8301.8495 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_6x_0p2` | seed_universe | validation | +6.58% | +22.09% | 0.7376 | 0.3492 | 0.2860 | 1.3373 | 334 | 0 | 0 | 9144.8982 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_6x_0p2` | seed_universe | locked_oos | -2.89% | +25.07% | -0.2659 | -0.1353 | -0.1082 | -0.8203 | 183 | 0 | 0 | 9295.3838 | False |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p2` | seed_universe | train | -30.57% | +54.32% | -0.3009 | -0.1013 | -0.0657 | -0.5629 | 821 | 0 | 0 | 8018.8244 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p2` | seed_universe | validation | +5.42% | +16.72% | 0.6908 | 0.2402 | 0.2058 | 1.4296 | 161 | 0 | 0 | 9002.3812 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p2` | seed_universe | locked_oos | +11.03% | +14.02% | 2.3179 | 1.0267 | 0.9004 | 9.0905 | 106 | 0 | 0 | 9177.9477 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p175` | seed_universe | train | -25.81% | +49.15% | -0.2993 | -0.1007 | -0.0675 | -0.5251 | 821 | 0 | 0 | 8266.4714 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p175` | seed_universe | validation | +5.02% | +14.76% | 0.6908 | 0.2402 | 0.2093 | 1.4916 | 161 | 0 | 0 | 9127.0835 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p175` | seed_universe | locked_oos | +9.69% | +12.37% | 2.3176 | 1.0265 | 0.9135 | 8.6401 | 106 | 0 | 0 | 9280.7042 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_6x_0p2` | seed_universe | train | -25.14% | +48.37% | -0.2991 | -0.1007 | -0.0678 | -0.5197 | 821 | 0 | 0 | 8301.8495 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_6x_0p2` | seed_universe | validation | +4.95% | +14.47% | 0.6908 | 0.2402 | 0.2098 | 1.5001 | 161 | 0 | 0 | 9144.8982 | True |
| 5 | `seed_alpha_zoo_high_confidence_single_pair_6x_0p2` | seed_universe | locked_oos | +9.50% | +12.13% | 2.3175 | 1.0265 | 0.9155 | 8.5772 | 106 | 0 | 0 | 9295.3838 | True |
| 5 | `seed_alpha_zoo_quality_single_pair_10x_0p15` | seed_universe | train | -40.30% | +75.55% | 0.0365 | 0.0154 | 0.0088 | -0.5334 | 1416 | 9 | 0 | 7877.3119 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_10x_0p15` | seed_universe | validation | +7.08% | +26.92% | 0.7403 | 0.3505 | 0.2762 | 1.1898 | 334 | 0 | 0 | 8931.1227 | False |
| 5 | `seed_alpha_zoo_quality_single_pair_10x_0p15` | seed_universe | locked_oos | -3.96% | +30.37% | -0.2603 | -0.1325 | -0.1017 | -0.8952 | 183 | 0 | 0 | 9119.2297 | False |
| 5 | `seed_alpha_zoo_high_confidence_long_only_7x_0p2` | seed_universe | train | +18.00% | +44.97% | 0.5884 | 0.1596 | 0.1101 | 0.4004 | 486 | 0 | 0 | 8224.4970 | True |
| 5 | `seed_alpha_zoo_high_confidence_long_only_7x_0p2` | seed_universe | validation | +26.20% | +14.15% | 2.7832 | 0.7849 | 0.6876 | 11.1216 | 104 | 0 | 0 | 9093.9048 | True |
| 5 | `seed_alpha_zoo_high_confidence_long_only_7x_0p2` | seed_universe | locked_oos | +3.82% | +10.98% | 0.9839 | 0.4009 | 0.3612 | 3.1182 | 73 | 0 | 0 | 9177.9477 | True |
| 5 | `reference_fast_residual_7x_0p15` | reference_fast_residual_7x_0p15 | train | -80.10% | +85.71% | -2.1469 | -1.3791 | -0.7426 | -0.9345 | 3102 | 0 | 0 | 8514.1183 | True |
| 5 | `reference_fast_residual_7x_0p15` | reference_fast_residual_7x_0p15 | validation | -4.27% | +22.45% | 0.0228 | 0.0155 | 0.0127 | -0.7235 | 767 | 0 | 0 | 9066.5742 | True |
| 5 | `reference_fast_residual_7x_0p15` | reference_fast_residual_7x_0p15 | locked_oos | +7.63% | +15.47% | 1.5356 | 1.0791 | 0.9345 | 5.0511 | 391 | 0 | 0 | 9383.4608 | True |
| 5 | `reference_strict_zero_fast_residual_6x_0p10` | reference_strict_zero_fast_residual_6x_0p10 | train | -57.95% | +65.74% | -2.1338 | -1.3707 | -0.8270 | -0.8816 | 3102 | 0 | 0 | 9150.9248 | True |
| 5 | `reference_strict_zero_fast_residual_6x_0p10` | reference_strict_zero_fast_residual_6x_0p10 | validation | -1.40% | +13.31% | 0.0155 | 0.0105 | 0.0093 | -0.4181 | 767 | 0 | 0 | 9466.6138 | True |
| 5 | `reference_strict_zero_fast_residual_6x_0p10` | reference_strict_zero_fast_residual_6x_0p10 | locked_oos | +4.59% | +9.07% | 1.5325 | 1.0765 | 0.9870 | 4.6547 | 391 | 0 | 0 | 9647.6919 | True |
| 10 | `hybrid_v3_5_seed_union` | hybrid_seed_union | train | +47.75% | +24.87% | 1.4017 | 1.1823 | 0.9469 | 1.9202 | 3574 | 0 | 0 | 8773.9574 | False |
| 10 | `hybrid_v3_5_seed_union` | hybrid_seed_union | validation | +18.91% | +8.82% | 3.3543 | 3.3591 | 3.0868 | 11.5758 | 872 | 0 | 0 | 13394.9568 | False |
| 10 | `hybrid_v3_5_seed_union` | hybrid_seed_union | locked_oos | -2.82% | +10.26% | -1.0720 | -0.7922 | -0.7185 | -1.9613 | 471 | 0 | 0 | 16659.9165 | False |
| 10 | `hybrid_v3_6_seed_union` | hybrid_seed_union | train | -9.07% | +33.09% | -0.0082 | -0.0064 | -0.0048 | -0.2742 | 3574 | 0 | 0 | 7018.5404 | False |
| 10 | `hybrid_v3_6_seed_union` | hybrid_seed_union | validation | -7.11% | +13.13% | -0.8975 | -0.7484 | -0.6615 | -1.9719 | 872 | 0 | 0 | 7882.4723 | False |
| 10 | `hybrid_v3_6_seed_union` | hybrid_seed_union | locked_oos | -6.22% | +14.38% | -2.0012 | -1.5180 | -1.3272 | -2.7550 | 471 | 0 | 0 | 7701.2479 | False |
| 10 | `seed_alpha_zoo_fast_residual_6x_0p175` | seed_universe | train | -96.10% | +96.37% | -4.6313 | -2.9417 | -1.4980 | -0.9971 | 3102 | 0 | 0 | 8514.1183 | False |
| 10 | `seed_alpha_zoo_fast_residual_6x_0p175` | seed_universe | validation | -35.94% | +37.71% | -2.6005 | -1.7425 | -1.2654 | -2.2180 | 767 | 0 | 0 | 9066.5742 | False |
| 10 | `seed_alpha_zoo_fast_residual_6x_0p175` | seed_universe | locked_oos | -12.44% | +24.21% | -2.1690 | -1.4847 | -1.1953 | -2.6754 | 391 | 0 | 0 | 9383.4608 | False |
| 10 | `seed_alpha_zoo_fast_residual_7x_0p15` | seed_universe | train | -96.10% | +96.37% | -4.6313 | -2.9417 | -1.4980 | -0.9971 | 3102 | 0 | 0 | 8514.1183 | False |
| 10 | `seed_alpha_zoo_fast_residual_7x_0p15` | seed_universe | validation | -35.94% | +37.71% | -2.6005 | -1.7425 | -1.2654 | -2.2180 | 767 | 0 | 0 | 9066.5742 | False |
| 10 | `seed_alpha_zoo_fast_residual_7x_0p15` | seed_universe | locked_oos | -12.44% | +24.21% | -2.1690 | -1.4847 | -1.1953 | -2.6754 | 391 | 0 | 0 | 9383.4608 | False |
| 10 | `seed_alpha_zoo_fast_residual_5x_0p2` | seed_universe | train | -95.40% | +95.71% | -4.6296 | -2.9407 | -1.5025 | -0.9967 | 3102 | 0 | 0 | 8584.8746 | False |
| 10 | `seed_alpha_zoo_fast_residual_5x_0p2` | seed_universe | validation | -34.44% | +36.16% | -2.6014 | -1.7428 | -1.2800 | -2.2681 | 767 | 0 | 0 | 9111.0230 | False |
| 10 | `seed_alpha_zoo_fast_residual_5x_0p2` | seed_universe | locked_oos | -11.84% | +23.19% | -2.1694 | -1.4849 | -1.2054 | -2.7101 | 391 | 0 | 0 | 9412.8198 | False |
| 10 | `seed_alpha_zoo_fast_residual_6x_0p05` | seed_universe | train | -58.51% | +59.52% | -4.6065 | -2.9253 | -1.8339 | -0.9831 | 3102 | 0 | 0 | 9575.4624 | False |
| 10 | `seed_alpha_zoo_fast_residual_6x_0p05` | seed_universe | validation | -11.15% | +11.86% | -2.6138 | -1.7458 | -1.5608 | -3.2155 | 767 | 0 | 0 | 9733.3069 | False |
| 10 | `seed_alpha_zoo_fast_residual_6x_0p05` | seed_universe | locked_oos | -3.50% | +7.50% | -2.1742 | -1.4873 | -1.3835 | -3.2515 | 391 | 0 | 0 | 9823.8459 | False |
| 10 | `seed_alpha_zoo_fast_residual_3x_0p1` | seed_universe | train | -58.51% | +59.52% | -4.6065 | -2.9253 | -1.8339 | -0.9831 | 3102 | 0 | 0 | 9575.4624 | False |
| 10 | `seed_alpha_zoo_fast_residual_3x_0p1` | seed_universe | validation | -11.15% | +11.86% | -2.6138 | -1.7458 | -1.5608 | -3.2155 | 767 | 0 | 0 | 9733.3069 | False |
| 10 | `seed_alpha_zoo_fast_residual_3x_0p1` | seed_universe | locked_oos | -3.50% | +7.50% | -2.1742 | -1.4873 | -1.3835 | -3.2515 | 391 | 0 | 0 | 9823.8459 | False |
| 10 | `seed_alpha_zoo_fast_residual_4x_0p1` | seed_universe | train | -69.32% | +70.28% | -4.6098 | -2.9274 | -1.7191 | -0.9863 | 3102 | 0 | 0 | 9433.9498 | False |
| 10 | `seed_alpha_zoo_fast_residual_4x_0p1` | seed_universe | validation | -14.72% | +15.63% | -2.6120 | -1.7454 | -1.5095 | -3.0480 | 767 | 0 | 0 | 9644.4092 | False |
| 10 | `seed_alpha_zoo_fast_residual_4x_0p1` | seed_universe | locked_oos | -4.67% | +9.89% | -2.1735 | -1.4869 | -1.3531 | -3.1682 | 391 | 0 | 0 | 9765.1279 | False |
| 10 | `seed_alpha_zoo_fast_residual_5x_0p05` | seed_universe | train | -51.83% | +52.82% | -4.6049 | -2.9243 | -1.9136 | -0.9814 | 3102 | 0 | 0 | 9646.2186 | False |
| 10 | `seed_alpha_zoo_fast_residual_5x_0p05` | seed_universe | validation | -9.33% | +9.94% | -2.6147 | -1.7461 | -1.5882 | -3.3039 | 767 | 0 | 0 | 9777.7558 | False |
| 10 | `seed_alpha_zoo_fast_residual_5x_0p05` | seed_universe | locked_oos | -2.91% | +6.28% | -2.1746 | -1.4874 | -1.3995 | -3.2939 | 391 | 0 | 0 | 9853.2049 | False |
| 10 | `seed_alpha_zoo_fast_residual_2x_0p125` | seed_universe | train | -51.83% | +52.82% | -4.6049 | -2.9243 | -1.9136 | -0.9814 | 3102 | 0 | 0 | 9646.2186 | False |
| 10 | `seed_alpha_zoo_fast_residual_2x_0p125` | seed_universe | validation | -9.33% | +9.94% | -2.6147 | -1.7461 | -1.5882 | -3.3039 | 767 | 0 | 0 | 9777.7558 | False |
| 10 | `seed_alpha_zoo_fast_residual_2x_0p125` | seed_universe | locked_oos | -2.91% | +6.28% | -2.1746 | -1.4874 | -1.3995 | -3.2939 | 391 | 0 | 0 | 9853.2049 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_7x_0p2` | seed_universe | train | -38.61% | +80.90% | -0.1315 | -0.0643 | -0.0356 | -0.4773 | 1416 | 0 | 0 | 8018.8244 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_7x_0p2` | seed_universe | validation | -15.28% | +29.47% | -0.7091 | -0.3333 | -0.2574 | -1.6638 | 334 | 0 | 0 | 9002.3812 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_7x_0p2` | seed_universe | locked_oos | -15.25% | +34.18% | -2.1214 | -1.0693 | -0.7969 | -2.1277 | 183 | 0 | 0 | 9177.9477 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_7x_0p175` | seed_universe | train | -32.08% | +76.06% | -0.1322 | -0.0646 | -0.0367 | -0.4218 | 1416 | 0 | 0 | 8266.4714 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_7x_0p175` | seed_universe | validation | -13.04% | +26.27% | -0.7104 | -0.3339 | -0.2644 | -1.6494 | 334 | 0 | 0 | 9127.0835 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_7x_0p175` | seed_universe | locked_oos | -13.32% | +30.61% | -2.1248 | -1.0705 | -0.8196 | -2.2041 | 183 | 0 | 0 | 9280.7042 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_6x_0p2` | seed_universe | train | -31.16% | +75.28% | -0.1323 | -0.0647 | -0.0369 | -0.4139 | 1416 | 0 | 0 | 8301.8495 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_6x_0p2` | seed_universe | validation | -12.73% | +25.80% | -0.7106 | -0.3339 | -0.2655 | -1.6469 | 334 | 0 | 0 | 9144.8982 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_6x_0p2` | seed_universe | locked_oos | -13.04% | +30.08% | -2.1252 | -1.0707 | -0.8231 | -2.2154 | 183 | 0 | 0 | 9295.3838 | False |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p2` | seed_universe | train | -60.94% | +68.73% | -1.2472 | -0.4177 | -0.2476 | -0.8867 | 821 | 0 | 0 | 8018.8244 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p2` | seed_universe | validation | -5.75% | +18.75% | -0.2930 | -0.1008 | -0.0849 | -1.1413 | 161 | 0 | 0 | 9002.3812 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p2` | seed_universe | locked_oos | +3.02% | +16.87% | 0.7972 | 0.3504 | 0.2999 | 1.5607 | 106 | 0 | 0 | 9177.9477 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p175` | seed_universe | train | -55.15% | +63.46% | -1.2457 | -0.4170 | -0.2551 | -0.8690 | 821 | 0 | 0 | 8266.4714 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p175` | seed_universe | validation | -4.79% | +16.57% | -0.2930 | -0.1008 | -0.0865 | -1.0903 | 161 | 0 | 0 | 9127.0835 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_7x_0p175` | seed_universe | locked_oos | +2.74% | +14.92% | 0.7970 | 0.3503 | 0.3049 | 1.5856 | 106 | 0 | 0 | 9280.7042 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_6x_0p2` | seed_universe | train | -54.28% | +62.65% | -1.2454 | -0.4170 | -0.2564 | -0.8664 | 821 | 0 | 0 | 8301.8495 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_6x_0p2` | seed_universe | validation | -4.65% | +16.26% | -0.2930 | -0.1008 | -0.0867 | -1.0828 | 161 | 0 | 0 | 9144.8982 | True |
| 10 | `seed_alpha_zoo_high_confidence_single_pair_6x_0p2` | seed_universe | locked_oos | +2.70% | +14.63% | 0.7969 | 0.3503 | 0.3056 | 1.5889 | 106 | 0 | 0 | 9295.3838 | True |
| 10 | `seed_alpha_zoo_quality_single_pair_10x_0p15` | seed_universe | train | -79.22% | +88.48% | -0.9751 | -0.4119 | -0.2185 | -0.8953 | 1416 | 9 | 0 | 7877.3119 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_10x_0p15` | seed_universe | validation | -16.59% | +31.24% | -0.7083 | -0.3330 | -0.2537 | -1.6695 | 334 | 0 | 0 | 8931.1227 | False |
| 10 | `seed_alpha_zoo_quality_single_pair_10x_0p15` | seed_universe | locked_oos | -16.35% | +36.15% | -2.1194 | -1.0686 | -0.7849 | -2.0857 | 183 | 0 | 0 | 9119.2297 | False |
| 10 | `seed_alpha_zoo_high_confidence_long_only_7x_0p2` | seed_universe | train | -16.03% | +54.54% | -0.1599 | -0.0433 | -0.0280 | -0.2939 | 486 | 0 | 0 | 8224.4970 | False |
| 10 | `seed_alpha_zoo_high_confidence_long_only_7x_0p2` | seed_universe | validation | +17.35% | +15.53% | 1.9771 | 0.5571 | 0.4822 | 5.8959 | 104 | 0 | 0 | 9093.9048 | False |
| 10 | `seed_alpha_zoo_high_confidence_long_only_7x_0p2` | seed_universe | locked_oos | -1.35% | +13.48% | -0.1179 | -0.0484 | -0.0426 | -0.7529 | 73 | 0 | 0 | 9177.9477 | False |
| 10 | `reference_fast_residual_7x_0p15` | reference_fast_residual_7x_0p15 | train | -96.10% | +96.37% | -4.6313 | -2.9417 | -1.4980 | -0.9971 | 3102 | 0 | 0 | 8514.1183 | False |
| 10 | `reference_fast_residual_7x_0p15` | reference_fast_residual_7x_0p15 | validation | -35.94% | +37.71% | -2.6005 | -1.7425 | -1.2654 | -2.2180 | 767 | 0 | 0 | 9066.5742 | False |
| 10 | `reference_fast_residual_7x_0p15` | reference_fast_residual_7x_0p15 | locked_oos | -12.44% | +24.21% | -2.1690 | -1.4847 | -1.1953 | -2.6754 | 391 | 0 | 0 | 9383.4608 | False |
| 10 | `reference_strict_zero_fast_residual_6x_0p10` | reference_strict_zero_fast_residual_6x_0p10 | train | -83.43% | +84.17% | -4.6164 | -2.9324 | -1.5922 | -0.9911 | 3102 | 0 | 0 | 9150.9248 | False |
| 10 | `reference_strict_zero_fast_residual_6x_0p10` | reference_strict_zero_fast_residual_6x_0p10 | validation | -21.62% | +22.87% | -2.6085 | -1.7445 | -1.4198 | -2.7477 | 767 | 0 | 0 | 9466.6138 | False |
| 10 | `reference_strict_zero_fast_residual_6x_0p10` | reference_strict_zero_fast_residual_6x_0p10 | locked_oos | -7.05% | +14.52% | -2.1721 | -1.4863 | -1.2978 | -3.0073 | 391 | 0 | 0 | 9647.6919 | False |

## Hybrid final weights

- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 6x/0.175=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 7x/0.15=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 5x/0.2=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 6x/0.05=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 3x/0.1=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 4x/0.1=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 5x/0.05=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_fast_residual 2x/0.125=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_quality_single_pair 7x/0.2=+2.32%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_quality_single_pair 7x/0.175=+2.32%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_quality_single_pair 6x/0.2=+70.03%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_high_confidence_single_pair 7x/0.2=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_high_confidence_single_pair 7x/0.175=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_high_confidence_single_pair 6x/0.2=+1.92%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_quality_single_pair 10x/0.15=+2.32%
- hybrid_v3_5_seed_union cost=5bps alpha_zoo_high_confidence_long_only 7x/0.2=+1.87%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 6x/0.175=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 7x/0.15=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 5x/0.2=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 6x/0.05=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 3x/0.1=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 4x/0.1=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 5x/0.05=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_fast_residual 2x/0.125=+1.78%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_quality_single_pair 7x/0.2=+2.18%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_quality_single_pair 7x/0.175=+2.18%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_quality_single_pair 6x/0.2=+2.18%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_high_confidence_single_pair 7x/0.2=+1.77%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_high_confidence_single_pair 7x/0.175=+1.77%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_high_confidence_single_pair 6x/0.2=+1.77%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_quality_single_pair 10x/0.15=+72.18%
- hybrid_v3_6_seed_union cost=5bps alpha_zoo_high_confidence_long_only 7x/0.2=+1.75%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 6x/0.175=+1.77%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 7x/0.15=+1.77%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 5x/0.2=+1.77%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 6x/0.05=+1.78%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 3x/0.1=+1.78%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 4x/0.1=+1.78%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 5x/0.05=+1.78%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_fast_residual 2x/0.125=+1.78%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_quality_single_pair 7x/0.2=+2.12%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_quality_single_pair 7x/0.175=+2.12%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_quality_single_pair 6x/0.2=+2.12%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_high_confidence_single_pair 7x/0.2=+1.83%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_high_confidence_single_pair 7x/0.175=+1.83%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_high_confidence_single_pair 6x/0.2=+1.83%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_quality_single_pair 10x/0.15=+2.12%
- hybrid_v3_5_seed_union cost=10bps alpha_zoo_high_confidence_long_only 7x/0.2=+71.80%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 6x/0.175=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 7x/0.15=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 5x/0.2=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 6x/0.05=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 3x/0.1=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 4x/0.1=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 5x/0.05=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_fast_residual 2x/0.125=+1.77%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_quality_single_pair 7x/0.2=+2.14%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_quality_single_pair 7x/0.175=+2.14%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_quality_single_pair 6x/0.2=+2.14%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_high_confidence_single_pair 7x/0.2=+1.83%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_high_confidence_single_pair 7x/0.175=+1.83%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_high_confidence_single_pair 6x/0.2=+1.83%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_quality_single_pair 10x/0.15=+72.14%
- hybrid_v3_6_seed_union cost=10bps alpha_zoo_high_confidence_long_only 7x/0.2=+1.80%
