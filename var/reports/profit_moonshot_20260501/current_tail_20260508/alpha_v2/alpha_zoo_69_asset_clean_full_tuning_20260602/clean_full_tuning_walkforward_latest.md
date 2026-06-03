# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-02T15:29:41.598366Z`
- latest available data: `2026-06-01T06:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `12` / `72` / `192`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.

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

## Aggregate ranking

| Rank | Candidate | Family | Clean | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 53.99% | 6/10 | -10.11% | -0.40% | 1.49 | 4.21 | 26.22% |
| 2 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `False` | 51.20% | 6/10 | -10.73% | -0.44% | 1.41 | 3.92 | 26.41% |
| 3 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `True` | `False` | 50.68% | 6/10 | -13.23% | -0.39% | 1.30 | 3.04 | 27.34% |
| 4 | `relaxed_efficiency:hybrid_v3_5` | `relaxed_efficiency` | `True` | `False` | 156.03% | 5/10 | -8.41% | -0.09% | 1.69 | 10.48 | 19.75% |
| 5 | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | `True` | `False` | 60.18% | 5/10 | -22.55% | -0.09% | 1.12 | 2.32 | 24.27% |
| 6 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 55.03% | 5/10 | -13.65% | -0.44% | 1.33 | 3.25 | 19.78% |
| 7 | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `True` | 54.76% | 5/10 | -9.97% | -0.38% | 1.53 | 4.43 | 16.75% |
| 8 | `mdd30_high_vol_gate:breakout_barbell_blend` | `mdd30_high_vol_gate` | `False` | `False` | 51.32% | 5/10 | -18.90% | -0.36% | 1.10 | 2.44 | 31.69% |
| 9 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `False` | 47.12% | 5/10 | -9.60% | -0.26% | 1.37 | 4.17 | 16.55% |
| 10 | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | `True` | `False` | 46.09% | 5/10 | -7.74% | -0.58% | 1.36 | 5.87 | 20.25% |
| 11 | `dynamic_aware_hybrid:hybrid_v3_6` | `dynamic_aware_hybrid` | `True` | `False` | 41.41% | 5/10 | -10.45% | -0.17% | 1.27 | 3.55 | 22.00% |
| 12 | `strict_efficiency:static_guarded` | `strict_efficiency` | `True` | `False` | 40.37% | 5/10 | -26.40% | -0.53% | 0.84 | 1.57 | 29.16% |

## Best candidate monthly OOS detail: `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 60.83% | 2.15% | 5.96% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.6634804601065519, "meta_portfolio:validation_stability_top8_equal": 0.0121083545493472...` |
| `2025-10` | 59.08% | 1.26% | 13.53% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.011026202731841904, "meta_portfolio:validation_stability_top8_equal": 0.01094059710753...` |
| `2025-11` | 105.07% | 22.64% | 14.26% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.006398915672219168, "meta_portfolio:validation_stability_top8_equal": 0.16830786868036...` |
| `2025-12` | 120.36% | -3.18% | 7.96% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.08535165375112788, "meta_portfolio:validation_stability_top8_equal": 0.086329305038977...` |
| `2026-01` | 182.39% | 23.15% | 3.29% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.3480532934526293, "meta_portfolio:validation_stability_top8_equal": 0.0440642834899703...` |
| `2026-02` | 108.85% | 16.11% | 14.42% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_stability_top8_equal": 0.12344484604520314, "profile_optuna:hybrid_v3_5": 0.2970819420344332, "profile_optuna...` |
| `2026-03` | 126.11% | -4.41% | 9.56% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.13884224266339507, "meta_portfolio:validation_stability_top8_equal": 0.139727759669770...` |
| `2026-04` | 97.50% | -10.11% | 11.45% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.6502250890997933, "meta_portfolio:validation_inverse_mdd_top10_capped": 0.032396656899...` |
| `2026-05` | 162.34% | 2.45% | 26.22% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.036176374947488744, "meta_portfolio:validation_stability_top8_equal": 0.03571912310077...` |
| `2026-06` | 147.36% | -0.40% | 0.51% | `hybrid_v3_5_optuna_three_profile_blend` / `{"meta_portfolio:validation_calmar_top5_capped": 0.36738737226335083, "meta_portfolio:validation_inverse_mdd_top10_capped": 0.06691437240...` |

### Best candidate extended metrics

- OOS comp: `53.99%`
- hit rate: `6/10`
- monthly Sharpe / Sortino approx: `1.49` / `4.21`
- 5% monthly VaR / 25% CVaR: `-7.55%` / `-5.90%`
- avg gain / avg loss: `11.29%` / `-4.52%`
- gain/loss ratio: `2.50`
- max loss streak: `2`
- mean/min validation: `116.99%` / `59.08%`

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
