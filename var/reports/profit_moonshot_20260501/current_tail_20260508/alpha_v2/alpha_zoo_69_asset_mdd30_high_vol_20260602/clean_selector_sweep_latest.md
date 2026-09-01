# Clean selector sweep over existing walk-forward rows

- generated: `2026-06-02T11:55:03.283498Z`
- selector count: `50`
- folds: `10`
- note: diagnostic only: selectors use train/validation rows only and exclude post-OOS research variants; choosing a new selector by historical OOS still requires fresh-forward confirmation

## Existing clean baseline by comp

| Rank | Candidate | Family | OOS comp | Hit | Max OOS MDD | Sharpe | Sortino | PF |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | 18.80% | 2.07 | 15.31 | 8.40 |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | 18.80% | 2.07 | 15.31 | 8.40 |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | 18.80% | 2.07 | 15.31 | 8.40 |
| 4 | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 43.73% | 4/10 | 18.80% | 1.62 | 4.24 | 4.27 |
| 5 | `dynamic_conviction_switch:t0.90_strict_fallback` | `dynamic_conviction_switch` | 43.73% | 4/10 | 18.80% | 1.62 | 4.24 | 4.27 |
| 6 | `dynamic_conviction_switch:t0.95_strict_fallback` | `dynamic_conviction_switch` | 43.73% | 4/10 | 18.80% | 1.62 | 4.24 | 4.27 |
| 7 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | 39.53% | 5/10 | 18.80% | 1.69 | 10.82 | 7.54 |
| 8 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | 32.94% | 5/10 | 11.22% | 1.45 | 10.16 | 4.10 |
| 9 | `dynamic_conviction_switch:t1.00_strict_fallback` | `dynamic_conviction_switch` | 30.54% | 4/10 | 18.80% | 1.26 | 3.11 | 3.55 |
| 10 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | 27.01% | 5/10 | 13.72% | 1.24 | 6.33 | 3.18 |

## Diagnostic selector sweep by OOS comp

| Rank | Selector | OOS comp | Hit | Max OOS MDD | Min OOS | Latest | Sharpe | Sortino | PF |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `clean_selector_sweep:dynamic_switch_only_utility_mdd15` | 43.73% | 4/10 | 18.80% | -8.56% | -0.71% | 1.62 | 4.24 | 4.27 |
| 2 | `clean_selector_sweep:dynamic_switch_only_utility_mdd18` | 43.73% | 4/10 | 18.80% | -8.56% | -0.71% | 1.62 | 4.24 | 4.27 |
| 3 | `clean_selector_sweep:dynamic_switch_only_utility_mdd22` | 43.73% | 4/10 | 18.80% | -8.56% | -0.71% | 1.62 | 4.24 | 4.27 |
| 4 | `clean_selector_sweep:dynamic_switch_only_utility_mdd30` | 43.73% | 4/10 | 18.80% | -8.56% | -0.71% | 1.62 | 4.24 | 4.27 |
| 5 | `clean_selector_sweep:strict_or_dynamic_utility_mdd15` | 43.41% | 4/10 | 18.80% | -4.61% | -0.71% | 1.65 | 8.57 | 4.12 |
| 6 | `clean_selector_sweep:strict_or_dynamic_utility_mdd18` | 43.41% | 4/10 | 18.80% | -4.61% | -0.71% | 1.65 | 8.57 | 4.12 |
| 7 | `clean_selector_sweep:strict_or_dynamic_utility_mdd22` | 43.41% | 4/10 | 18.80% | -4.61% | -0.71% | 1.65 | 8.57 | 4.12 |
| 8 | `clean_selector_sweep:strict_or_dynamic_utility_mdd30` | 43.41% | 4/10 | 18.80% | -4.61% | -0.71% | 1.65 | 8.57 | 4.12 |
| 9 | `clean_selector_sweep:calmar_mdd10` | 31.55% | 4/10 | 12.56% | -3.61% | -0.71% | 1.51 | 8.94 | 3.77 |
| 10 | `clean_selector_sweep:calmar_mdd12` | 31.55% | 4/10 | 12.56% | -3.61% | -0.71% | 1.51 | 8.94 | 3.77 |
| 11 | `clean_selector_sweep:calmar_mdd15` | 31.55% | 4/10 | 12.56% | -3.61% | -0.71% | 1.51 | 8.94 | 3.77 |
| 12 | `clean_selector_sweep:calmar_mdd18` | 31.55% | 4/10 | 12.56% | -3.61% | -0.71% | 1.51 | 8.94 | 3.77 |
| 13 | `clean_selector_sweep:calmar_mdd22` | 31.55% | 4/10 | 12.56% | -3.61% | -0.71% | 1.51 | 8.94 | 3.77 |
| 14 | `clean_selector_sweep:calmar_mdd30` | 31.55% | 4/10 | 12.56% | -3.61% | -0.71% | 1.51 | 8.94 | 3.77 |
| 15 | `clean_selector_sweep:sharpeproxy_mdd10` | 31.35% | 4/10 | 12.56% | -3.61% | -0.71% | 1.50 | 8.82 | 3.72 |
| 16 | `clean_selector_sweep:sharpeproxy_mdd12` | 31.35% | 4/10 | 12.56% | -3.61% | -0.71% | 1.50 | 8.82 | 3.72 |
| 17 | `clean_selector_sweep:sharpeproxy_mdd15` | 31.35% | 4/10 | 12.56% | -3.61% | -0.71% | 1.50 | 8.82 | 3.72 |
| 18 | `clean_selector_sweep:sharpeproxy_mdd18` | 31.35% | 4/10 | 12.56% | -3.61% | -0.71% | 1.50 | 8.82 | 3.72 |
| 19 | `clean_selector_sweep:sharpeproxy_mdd22` | 31.35% | 4/10 | 12.56% | -3.61% | -0.71% | 1.50 | 8.82 | 3.72 |
| 20 | `clean_selector_sweep:sharpeproxy_mdd30` | 31.35% | 4/10 | 12.56% | -3.61% | -0.71% | 1.50 | 8.82 | 3.72 |

## Best diagnostic selector choices: `clean_selector_sweep:dynamic_switch_only_utility_mdd15`

| Fold | Selected clean candidate | Family | Val | Val MDD | OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 20.66% | 11.13% | -8.56% | 10.62% |
| `2025-10` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 1.73% | 2.55% | -0.72% | 1.39% |
| `2025-11` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 26.81% | 7.58% | 12.20% | 10.08% |
| `2025-12` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 1.92% | 2.50% | -0.13% | 1.63% |
| `2026-01` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 80.04% | 5.03% | 9.62% | 3.92% |
| `2026-02` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 108.23% | 11.29% | 19.24% | 18.80% |
| `2026-03` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 15.06% | 8.04% | -0.10% | 5.76% |
| `2026-04` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 5.79% | 3.74% | -2.02% | 4.57% |
| `2026-05` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 43.70% | 6.48% | 11.24% | 16.08% |
| `2026-06` | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 86.60% | 8.87% | -0.71% | 1.04% |

## Clean interpretation

- These selector rules are fold-clean: they select only from train/validation metrics and exclude post-OOS research rows.
- However, picking a new selector because it ranks well on this already-reviewed OOS window would still be OOS-mining.
- Therefore the existing clean baseline remains the promotion candidate unless this selector family is frozen and validated on fresh-forward data.
