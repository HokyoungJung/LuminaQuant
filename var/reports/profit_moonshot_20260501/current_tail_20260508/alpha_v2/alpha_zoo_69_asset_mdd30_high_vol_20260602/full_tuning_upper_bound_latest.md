# Full-tuning upper-bound diagnostics

- generated: `2026-06-02T12:23:50.452966Z`
- OOS period: `2025-09-01T00:00:00` → `2026-06-01T06:30:00`
- note: full-tuning upper bound only: scenarios use locked-OOS oracle selection and are not deployable. Use this as a ceiling/diagnostic, not as clean live evidence.

## Scenario ranking

| Rank | Scenario | Deployable | OOS oracle | OOS comp | Ann. approx | Hit | Max OOS MDD | Min OOS | Latest | Sharpe | Sortino | PF |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `oracle_all_candidates_max_return` | `False` | `True` | 132.98% | 175.92% | 8/10 | 23.30% | -0.40% | 0.00% | 3.12 | 0.00 | 231.61 |
| 2 | `oracle_all_candidates_mdd30_max_return` | `False` | `True` | 132.98% | 175.92% | 8/10 | 23.30% | -0.40% | 0.00% | 3.12 | 0.00 | 231.61 |
| 3 | `oracle_all_candidates_mdd20_max_return` | `False` | `True` | 118.89% | 156.02% | 8/10 | 19.72% | -0.40% | 0.00% | 3.36 | 0.00 | 212.04 |
| 4 | `oracle_clean_only_max_return` | `False` | `True` | 106.20% | 138.31% | 8/10 | 10.08% | -0.40% | 0.00% | 3.37 | 0.00 | 194.74 |
| 5 | `oracle_clean_only_mdd30_max_return` | `False` | `True` | 106.20% | 138.31% | 8/10 | 10.08% | -0.40% | 0.00% | 3.37 | 0.00 | 194.74 |
| 6 | `oracle_clean_only_mdd20_max_return` | `False` | `True` | 106.20% | 138.31% | 8/10 | 10.08% | -0.40% | 0.00% | 3.37 | 0.00 | 194.74 |
| 7 | `oracle_clean_only_mdd20_calmar` | `False` | `True` | 95.08% | 122.97% | 8/10 | 9.96% | -0.40% | 0.00% | 3.30 | 0.00 | 179.16 |

## Clean baseline by comp

| Rank | Candidate | Family | OOS comp | Hit | Max OOS MDD | Sharpe | Sortino | PF | Clean |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | 18.80% | 2.07 | 15.31 | 8.40 | `True` |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | 18.80% | 2.07 | 15.31 | 8.40 | `True` |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | 18.80% | 2.07 | 15.31 | 8.40 | `True` |
| 4 | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | 43.73% | 4/10 | 18.80% | 1.62 | 4.24 | 4.27 | `True` |
| 5 | `dynamic_conviction_switch:t0.90_strict_fallback` | `dynamic_conviction_switch` | 43.73% | 4/10 | 18.80% | 1.62 | 4.24 | 4.27 | `True` |
| 6 | `dynamic_conviction_switch:t0.95_strict_fallback` | `dynamic_conviction_switch` | 43.73% | 4/10 | 18.80% | 1.62 | 4.24 | 4.27 | `True` |
| 7 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | 39.53% | 5/10 | 18.80% | 1.69 | 10.82 | 7.54 | `True` |
| 8 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | 32.94% | 5/10 | 11.22% | 1.45 | 10.16 | 4.10 | `True` |

## Monthly choices: `oracle_all_candidates_max_return`

| Fold | Candidate | Family | Clean | Research | Val | Val MDD | OOS | OOS MDD |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `individual_robust:hybrid_v3_5` | `individual_robust` | `True` | `False` | 25.41% | 4.37% | 3.10% | 5.79% |
| `2025-10` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `False` | 46.78% | 4.36% | 6.41% | 7.04% |
| `2025-11` | `mdd30_risk_scaled:dyn085_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 41.12% | 11.25% | 18.00% | 14.79% |
| `2025-12` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | `True` | `False` | 60.44% | 13.01% | 0.20% | 4.64% |
| `2026-01` | `relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna` | `relaxed_efficiency` | `True` | `False` | 32.08% | 13.75% | 16.03% | 9.34% |
| `2026-02` | `mdd30_risk_scaled:cross_v35_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 157.38% | 14.92% | 29.77% | 20.07% |
| `2026-03` | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `False` | 53.31% | 3.66% | 2.63% | 3.14% |
| `2026-04` | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 2.16% | 4.25% | -0.40% | 1.65% |
| `2026-05` | `mdd30_risk_scaled:dyn085_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 70.24% | 9.64% | 16.69% | 23.30% |
| `2026-06` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 0.01% | 0.07% | 0.00% | 0.00% |

## Monthly choices: `oracle_all_candidates_mdd30_max_return`

| Fold | Candidate | Family | Clean | Research | Val | Val MDD | OOS | OOS MDD |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `individual_robust:hybrid_v3_5` | `individual_robust` | `True` | `False` | 25.41% | 4.37% | 3.10% | 5.79% |
| `2025-10` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `False` | 46.78% | 4.36% | 6.41% | 7.04% |
| `2025-11` | `mdd30_risk_scaled:dyn085_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 41.12% | 11.25% | 18.00% | 14.79% |
| `2025-12` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | `True` | `False` | 60.44% | 13.01% | 0.20% | 4.64% |
| `2026-01` | `relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna` | `relaxed_efficiency` | `True` | `False` | 32.08% | 13.75% | 16.03% | 9.34% |
| `2026-02` | `mdd30_risk_scaled:cross_v35_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 157.38% | 14.92% | 29.77% | 20.07% |
| `2026-03` | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `False` | 53.31% | 3.66% | 2.63% | 3.14% |
| `2026-04` | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 2.16% | 4.25% | -0.40% | 1.65% |
| `2026-05` | `mdd30_risk_scaled:dyn085_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 70.24% | 9.64% | 16.69% | 23.30% |
| `2026-06` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 0.01% | 0.07% | 0.00% | 0.00% |

## Monthly choices: `oracle_all_candidates_mdd20_max_return`

| Fold | Candidate | Family | Clean | Research | Val | Val MDD | OOS | OOS MDD |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `individual_robust:hybrid_v3_5` | `individual_robust` | `True` | `False` | 25.41% | 4.37% | 3.10% | 5.79% |
| `2025-10` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `False` | 46.78% | 4.36% | 6.41% | 7.04% |
| `2025-11` | `mdd30_risk_scaled:dyn085_x1_50` | `mdd30_risk_scaled` | `False` | `True` | 41.12% | 11.25% | 18.00% | 14.79% |
| `2025-12` | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | `True` | `False` | 60.44% | 13.01% | 0.20% | 4.64% |
| `2026-01` | `relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna` | `relaxed_efficiency` | `True` | `False` | 32.08% | 13.75% | 16.03% | 9.34% |
| `2026-02` | `mdd30_barbell_blend:dyn085_70_strict_growth_30_x1_50` | `mdd30_barbell_blend` | `False` | `True` | 129.03% | 12.22% | 22.03% | 19.72% |
| `2026-03` | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `False` | 53.31% | 3.66% | 2.63% | 3.14% |
| `2026-04` | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 2.16% | 4.25% | -0.40% | 1.65% |
| `2026-05` | `profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna` | `profile_optuna` | `True` | `False` | 29.53% | 13.31% | 16.58% | 9.96% |
| `2026-06` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 0.01% | 0.07% | 0.00% | 0.00% |

## Interpretation

- The scenario rows are maximum-performance diagnostics because the selected candidate is chosen by locked-OOS performance.
- `oracle_clean_only_*` shows that even if the source candidates are clean, choosing among them with OOS is still not clean for deployment.
- The deployable reference remains the best pre-registered clean candidate unless a new selection rule is frozen before fresh-forward validation.
