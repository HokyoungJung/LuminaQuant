# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-04T11:48:17.720249Z`
- latest available data: `2026-06-01T06:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `12` / `72` / `192`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.
- recomputed from existing rows: `True`
- source JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_full_tuning_walkforward_latest.json`
- source sha256: `563aff7f59174a7ebb6b53f9164eb1feb0cf67881e7f203aecb06987024fa58f`
- recompute interpretation: `governance/ranking repair only; not a fresh no-nested Optuna search`

## Fold schedule

| Fold | Refit | Train | Validation | OOS |
| --- | --- | --- | --- | --- |
| `2025-09` | `2025-09-01T00:00:00` | `2025-01-01T00:00:00 → 2025-06-30T23:30:00` | `2025-07-01T00:00:00 → 2025-08-31T23:30:00` | `2025-09-01T00:00:00 → 2025-09-30T23:30:00` |
| `2025-10` | `2025-10-01T00:00:00` | `2025-01-01T00:00:00 → 2025-07-31T23:30:00` | `2025-08-01T00:00:00 → 2025-09-30T23:30:00` | `2025-10-01T00:00:00 → 2025-10-31T23:30:00` |
| `2025-11` | `2025-11-01T00:00:00` | `2025-01-01T00:00:00 → 2025-08-31T23:30:00` | `2025-09-01T00:00:00 → 2025-10-31T23:30:00` | `2025-11-01T00:00:00 → 2025-11-30T23:30:00` |
| `2025-12` | `2025-12-01T00:00:00` | `2025-01-01T00:00:00 → 2025-09-30T23:30:00` | `2025-10-01T00:00:00 → 2025-11-30T23:30:00` | `2025-12-01T00:00:00 → 2025-12-31T23:30:00` |
| `2026-01` | `2026-01-01T00:00:00` | `2025-01-01T00:00:00 → 2025-10-31T23:30:00` | `2025-11-01T00:00:00 → 2025-12-31T23:30:00` | `2026-01-01T00:00:00 → 2026-01-31T23:30:00` |
| `2026-02` | `2026-02-01T00:00:00` | `2025-01-01T00:00:00 → 2025-11-30T23:30:00` | `2025-12-01T00:00:00 → 2026-01-31T23:30:00` | `2026-02-01T00:00:00 → 2026-02-28T23:30:00` |
| `2026-03` | `2026-03-01T00:00:00` | `2025-01-01T00:00:00 → 2025-12-31T23:30:00` | `2026-01-01T00:00:00 → 2026-02-28T23:30:00` | `2026-03-01T00:00:00 → 2026-03-31T23:30:00` |
| `2026-04` | `2026-04-01T00:00:00` | `2025-01-01T00:00:00 → 2026-01-31T23:30:00` | `2026-02-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-04-30T23:30:00` |
| `2026-05` | `2026-05-01T00:00:00` | `2025-01-01T00:00:00 → 2026-02-28T23:30:00` | `2026-03-01T00:00:00 → 2026-04-30T23:30:00` | `2026-05-01T00:00:00 → 2026-05-31T23:30:00` |
| `2026-06` | `2026-06-01T00:00:00` | `2025-01-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-05-31T23:30:00` | `2026-06-01T00:00:00 → 2026-06-01T06:30:00` |

## Raw aggregate ranking (diagnostic only)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 122.36% | 6/10 | -8.42% | -0.17% | 1.74 | 8.97 | 16.66% |
| 2 | `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 111.75% | 6/10 | -8.61% | -0.20% | 1.75 | 8.43 | 16.19% |
| 3 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 53.99% | 6/10 | -10.11% | -0.40% | 1.49 | 4.21 | 26.22% |
| 4 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 51.20% | 6/10 | -10.73% | -0.44% | 1.41 | 3.92 | 26.41% |
| 5 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 50.68% | 6/10 | -13.23% | -0.39% | 1.30 | 3.04 | 27.34% |
| 6 | `relaxed_efficiency:hybrid_v3_5` | `relaxed_efficiency` | `True` | `` | `False` | 156.03% | 5/10 | -8.41% | -0.09% | 1.69 | 10.48 | 19.75% |
| 7 | `fixed_relaxed_dynamic_blend:relaxed50_dynamic50` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 101.45% | 5/10 | -8.81% | -0.23% | 1.75 | 7.66 | 15.73% |
| 8 | `fixed_relaxed_dynamic_blend:relaxed40_dynamic60` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 91.47% | 5/10 | -9.02% | -0.26% | 1.75 | 7.07 | 15.26% |
| 9 | `fixed_relaxed_dynamic_blend:relaxed30_dynamic70` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 81.81% | 5/10 | -9.24% | -0.29% | 1.73 | 6.43 | 14.83% |
| 10 | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | `True` | `` | `False` | 60.18% | 5/10 | -22.55% | -0.09% | 1.12 | 2.32 | 24.27% |
| 11 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 55.03% | 5/10 | -13.65% | -0.44% | 1.33 | 3.25 | 19.78% |
| 12 | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 54.76% | 5/10 | -9.97% | -0.38% | 1.53 | 4.43 | 16.75% |

## Clean-promotion ranking (current recommendation set)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `relaxed_efficiency:hybrid_v3_5` | `relaxed_efficiency` | `True` | `` | `False` | 156.03% | 5/10 | -8.41% | -0.09% | 1.69 | 10.48 | 19.75% |
| 2 | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | `True` | `` | `False` | 60.18% | 5/10 | -22.55% | -0.09% | 1.12 | 2.32 | 24.27% |
| 3 | `strict_efficiency:static_guarded` | `strict_efficiency` | `True` | `` | `False` | 40.37% | 5/10 | -26.40% | -0.53% | 0.84 | 1.57 | 29.16% |
| 4 | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | `True` | `` | `False` | 33.82% | 5/10 | -22.55% | -0.09% | 0.81 | 1.57 | 25.79% |
| 5 | `strict_efficiency:hybrid_v3_6` | `strict_efficiency` | `True` | `` | `False` | 32.13% | 5/10 | -5.89% | -0.17% | 1.03 | 5.48 | 15.89% |
| 6 | `individual_robust:hybrid_v3_6` | `individual_robust` | `True` | `` | `False` | 26.77% | 5/10 | -7.46% | -0.31% | 1.09 | 3.25 | 13.78% |
| 7 | `relaxed_efficiency:balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna` | `relaxed_efficiency` | `True` | `` | `False` | 13.19% | 5/10 | -13.05% | 0.00% | 0.55 | 1.62 | 32.62% |
| 8 | `strict_efficiency:selected_train_validation_legal` | `strict_efficiency` | `True` | `` | `False` | 11.40% | 5/10 | -10.22% | -0.02% | 0.54 | 1.27 | 17.35% |
| 9 | `strict_efficiency:selected_optuna` | `strict_efficiency` | `True` | `` | `False` | 11.23% | 5/10 | -10.22% | -0.17% | 0.53 | 1.27 | 17.35% |
| 10 | `individual_robust:individual_robust_balanced_mdd10_gross3_core10` | `individual_robust` | `True` | `` | `False` | 9.65% | 5/10 | -11.13% | -0.00% | 0.49 | 0.97 | 14.98% |
| 11 | `strict_efficiency:hybrid_v3_5` | `strict_efficiency` | `True` | `` | `False` | 7.21% | 5/10 | -10.29% | -0.02% | 0.39 | 0.77 | 17.35% |
| 12 | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `` | `False` | 30.24% | 4/10 | -11.13% | -0.15% | 0.99 | 2.60 | 31.69% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 122.36% | 6/10 | -8.42% | -0.17% | 1.74 | 8.97 | 16.66% |
| 2 | `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 111.75% | 6/10 | -8.61% | -0.20% | 1.75 | 8.43 | 16.19% |
| 3 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 53.99% | 6/10 | -10.11% | -0.40% | 1.49 | 4.21 | 26.22% |
| 4 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 51.20% | 6/10 | -10.73% | -0.44% | 1.41 | 3.92 | 26.41% |
| 5 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 50.68% | 6/10 | -13.23% | -0.39% | 1.30 | 3.04 | 27.34% |
| 6 | `fixed_relaxed_dynamic_blend:relaxed50_dynamic50` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 101.45% | 5/10 | -8.81% | -0.23% | 1.75 | 7.66 | 15.73% |
| 7 | `fixed_relaxed_dynamic_blend:relaxed40_dynamic60` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 91.47% | 5/10 | -9.02% | -0.26% | 1.75 | 7.07 | 15.26% |
| 8 | `fixed_relaxed_dynamic_blend:relaxed30_dynamic70` | `fixed_relaxed_dynamic_blend` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 81.81% | 5/10 | -9.24% | -0.29% | 1.73 | 6.43 | 14.83% |
| 9 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 55.03% | 5/10 | -13.65% | -0.44% | 1.33 | 3.25 | 19.78% |
| 10 | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 54.76% | 5/10 | -9.97% | -0.38% | 1.53 | 4.43 | 16.75% |
| 11 | `mdd30_high_vol_gate:breakout_barbell_blend` | `mdd30_high_vol_gate` | `False` | `nested_hybrid_dependency,post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 51.32% | 5/10 | -18.90% | -0.36% | 1.10 | 2.44 | 31.69% |
| 12 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | `False` | `nested_hybrid_dependency` | `False` | 47.12% | 5/10 | -9.60% | -0.26% | 1.37 | 4.17 | 16.55% |

## Best clean candidate monthly OOS detail: `relaxed_efficiency:hybrid_v3_5`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 55.30% | -0.78% | 8.27% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.1943136293975687, "balanced_mdd12_gross5_69_asset_relaxed_effici...` |
| `2025-10` | 55.59% | 0.39% | 14.90% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.010534808477121842, "balanced_mdd12_gross5_69_asset_relaxed_effi...` |
| `2025-11` | 86.61% | 63.19% | 5.71% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.10335794023187309, "balanced_mdd12_gross5_69_asset_relaxed_effic...` |
| `2025-12` | 126.06% | -2.59% | 9.44% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.0304693670943202, "balanced_mdd12_gross5_69_asset_relaxed_effici...` |
| `2026-01` | 106.72% | 16.60% | 5.75% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.07520542559571883, "balanced_mdd12_gross5_69_asset_relaxed_effic...` |
| `2026-02` | 80.06% | 47.58% | 19.75% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.8263932073331357, "balanced_mdd12_gross5_69_asset_relaxed_effici...` |
| `2026-03` | 64.10% | -8.41% | 11.71% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.29877731485635683, "balanced_mdd12_gross5_69_asset_relaxed_effic...` |
| `2026-04` | 55.20% | -7.91% | 13.18% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.0710949167452735, "balanced_mdd12_gross5_69_asset_relaxed_effici...` |
| `2026-05` | 64.80% | 11.52% | 12.58% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.18944328021339357, "balanced_mdd12_gross5_69_asset_relaxed_effic...` |
| `2026-06` | 151.79% | -0.09% | 0.05% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.07437006293889148, "balanced_mdd12_gross5_69_asset_relaxed_effic...` |

### Best candidate extended metrics

- OOS comp: `156.03%`
- hit rate: `5/10`
- monthly Sharpe / Sortino approx: `1.69` / `10.48`
- 5% monthly VaR / 25% CVaR: `-8.19%` / `-6.30%`
- avg gain / avg loss: `27.85%` / `-3.96%`
- gain/loss ratio: `7.04`
- max loss streak: `2`
- mean/min validation: `84.62%` / `55.20%`

## Timeframe coverage

| Timeframe | Symbols with rows | Symbols skipped | Median rows | Latest |
| --- | ---: | ---: | ---: | --- |
| `30m` | 69 | 0 | 2674.0 | `2026-06-01T06:30:00` |
| `1h` | 69 | 0 | 1337.0 | `2026-06-01T06:00:00` |
| `2h` | 69 | 0 | 668.0 | `2026-06-01T04:00:00` |
| `4h` | 69 | 0 | 333.0 | `2026-06-01T00:00:00` |
| `6h` | 69 | 0 | 222.0 | `2026-06-01T00:00:00` |
| `8h` | 69 | 0 | 166.0 | `2026-05-31T16:00:00` |
| `12h` | 69 | 0 | 110.0 | `2026-05-31T12:00:00` |
| `1d` | 69 | 0 | 55.0 | `2026-05-31T00:00:00` |

## Interpretation guardrails

- This is still research/paper-testnet evidence, not real-money approval.
- The latest OOS month can be partial when the data feed ends before month-end.
- If a candidate has a negative validation fold or low OOS consistency, prefer shadow monitoring over allocation.
