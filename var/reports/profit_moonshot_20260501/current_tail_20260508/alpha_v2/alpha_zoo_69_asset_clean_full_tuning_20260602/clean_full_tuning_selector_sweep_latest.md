# Clean selector sweep over existing walk-forward rows

- generated: `2026-06-02T15:29:53.731880Z`
- selector count: `50`
- folds: `10`
- note: diagnostic only: selectors use train/validation rows only and exclude post-OOS research variants; choosing a new selector by historical OOS still requires fresh-forward confirmation

## Existing clean baseline by comp

| Rank | Candidate | Family | OOS comp | Hit | Max OOS MDD | Sharpe | Sortino | PF |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `relaxed_efficiency:hybrid_v3_5` | `relaxed_efficiency` | 156.03% | 5/10 | 19.75% | 1.69 | 10.48 | 7.04 |
| 2 | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | 60.18% | 5/10 | 24.27% | 1.12 | 2.32 | 2.70 |
| 3 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | 55.03% | 5/10 | 19.78% | 1.33 | 3.25 | 2.92 |
| 4 | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 54.76% | 5/10 | 16.75% | 1.53 | 4.43 | 3.87 |
| 5 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | 53.99% | 6/10 | 26.22% | 1.49 | 4.21 | 3.74 |
| 6 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | 51.20% | 6/10 | 26.41% | 1.41 | 3.92 | 3.36 |
| 7 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | 50.68% | 6/10 | 27.34% | 1.30 | 3.04 | 3.03 |
| 8 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | 47.12% | 5/10 | 16.55% | 1.37 | 4.17 | 3.37 |
| 9 | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | 46.09% | 5/10 | 20.25% | 1.36 | 5.87 | 3.36 |
| 10 | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 44.31% | 4/10 | 29.29% | 1.11 | 4.23 | 2.52 |

## Diagnostic selector sweep by OOS comp

| Rank | Selector | OOS comp | Hit | Max OOS MDD | Min OOS | Latest | Sharpe | Sortino | PF |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `clean_selector_sweep:dynamic_aware_only_utility_mdd15` | 68.46% | 5/10 | 20.25% | -9.97% | -0.58% | 1.69 | 5.35 | 4.41 |
| 2 | `clean_selector_sweep:dynamic_aware_only_utility_mdd18` | 68.46% | 5/10 | 20.25% | -9.97% | -0.58% | 1.69 | 5.35 | 4.41 |
| 3 | `clean_selector_sweep:dynamic_aware_only_utility_mdd22` | 68.46% | 5/10 | 20.25% | -9.97% | -0.58% | 1.69 | 5.35 | 4.41 |
| 4 | `clean_selector_sweep:dynamic_aware_only_utility_mdd30` | 68.46% | 5/10 | 20.25% | -9.97% | -0.58% | 1.69 | 5.35 | 4.41 |
| 5 | `clean_selector_sweep:stable_mdd22` | 62.64% | 5/10 | 28.74% | -13.23% | 0.07% | 1.30 | 4.00 | 3.28 |
| 6 | `clean_selector_sweep:stable_mdd30` | 62.64% | 5/10 | 28.74% | -13.23% | 0.07% | 1.30 | 4.00 | 3.28 |
| 7 | `clean_selector_sweep:clean_hybrid_only_utility_mdd15` | 61.37% | 6/10 | 20.25% | -13.65% | -0.39% | 1.51 | 3.37 | 3.57 |
| 8 | `clean_selector_sweep:clean_hybrid_only_utility_mdd18` | 61.37% | 6/10 | 20.25% | -13.65% | -0.39% | 1.51 | 3.37 | 3.57 |
| 9 | `clean_selector_sweep:clean_hybrid_only_utility_mdd22` | 61.37% | 6/10 | 20.25% | -13.65% | -0.39% | 1.51 | 3.37 | 3.57 |
| 10 | `clean_selector_sweep:clean_hybrid_only_utility_mdd30` | 61.37% | 6/10 | 20.25% | -13.65% | -0.39% | 1.51 | 3.37 | 3.57 |
| 11 | `clean_selector_sweep:utility_mdd10` | 44.75% | 5/10 | 20.25% | -13.65% | -0.32% | 1.28 | 2.75 | 3.09 |
| 12 | `clean_selector_sweep:dynamic_switch_only_utility_mdd15` | 44.31% | 4/10 | 29.29% | -10.73% | -0.32% | 1.11 | 4.23 | 2.52 |
| 13 | `clean_selector_sweep:dynamic_switch_only_utility_mdd18` | 44.31% | 4/10 | 29.29% | -10.73% | -0.32% | 1.11 | 4.23 | 2.52 |
| 14 | `clean_selector_sweep:dynamic_switch_only_utility_mdd22` | 44.31% | 4/10 | 29.29% | -10.73% | -0.32% | 1.11 | 4.23 | 2.52 |
| 15 | `clean_selector_sweep:dynamic_switch_only_utility_mdd30` | 44.31% | 4/10 | 29.29% | -10.73% | -0.32% | 1.11 | 4.23 | 2.52 |
| 16 | `clean_selector_sweep:sharpeproxy_mdd10` | 43.47% | 5/10 | 26.22% | -7.74% | -0.25% | 1.30 | 5.25 | 3.21 |
| 17 | `clean_selector_sweep:sharpeproxy_mdd12` | 43.47% | 5/10 | 26.22% | -7.74% | -0.25% | 1.30 | 5.25 | 3.21 |
| 18 | `clean_selector_sweep:sharpeproxy_mdd15` | 43.47% | 5/10 | 26.22% | -7.74% | -0.25% | 1.30 | 5.25 | 3.21 |
| 19 | `clean_selector_sweep:sharpeproxy_mdd18` | 43.47% | 5/10 | 26.22% | -7.74% | -0.25% | 1.30 | 5.25 | 3.21 |
| 20 | `clean_selector_sweep:sharpeproxy_mdd22` | 43.47% | 5/10 | 26.22% | -7.74% | -0.25% | 1.30 | 5.25 | 3.21 |

## Best diagnostic selector choices: `clean_selector_sweep:dynamic_aware_only_utility_mdd15`

| Fold | Selected clean candidate | Family | Val | Val MDD | OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 58.81% | 4.18% | 2.65% | 5.41% |
| `2025-10` | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 52.08% | 6.74% | -0.47% | 13.19% |
| `2025-11` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | 100.56% | 10.32% | 22.84% | 14.86% |
| `2025-12` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | 113.57% | 8.88% | -3.39% | 7.71% |
| `2026-01` | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 175.66% | 7.48% | 24.60% | 3.15% |
| `2026-02` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | 100.74% | 8.37% | 19.63% | 15.08% |
| `2026-03` | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 105.55% | 6.63% | -3.10% | 7.95% |
| `2026-04` | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 90.69% | 6.45% | -9.97% | 10.98% |
| `2026-05` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | 137.72% | 7.85% | 7.46% | 20.25% |
| `2026-06` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | 156.81% | 6.25% | -0.58% | 0.60% |

## Clean interpretation

- These selector rules are fold-clean: they select only from train/validation metrics and exclude post-OOS research rows.
- However, picking a new selector because it ranks well on this already-reviewed OOS window would still be OOS-mining.
- Therefore the existing clean baseline remains the promotion candidate unless this selector family is frozen and validated on fresh-forward data.
