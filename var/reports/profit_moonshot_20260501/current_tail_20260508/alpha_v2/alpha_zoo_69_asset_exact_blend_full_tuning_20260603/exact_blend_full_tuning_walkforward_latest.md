# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-03T11:28:50.811187Z`
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
| 1 | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` | `fixed_relaxed_dynamic_blend` | `False` | `False` | 122.36% | 6/10 | -8.42% | -0.17% | 1.74 | 8.97 | 16.66% |
| 2 | `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` | `fixed_relaxed_dynamic_blend` | `False` | `False` | 111.75% | 6/10 | -8.61% | -0.20% | 1.75 | 8.43 | 16.19% |
| 3 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 53.99% | 6/10 | -10.11% | -0.40% | 1.49 | 4.21 | 26.22% |
| 4 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `False` | 51.20% | 6/10 | -10.73% | -0.44% | 1.41 | 3.92 | 26.41% |
| 5 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `True` | `False` | 50.68% | 6/10 | -13.23% | -0.39% | 1.30 | 3.04 | 27.34% |
| 6 | `relaxed_efficiency:hybrid_v3_5` | `relaxed_efficiency` | `True` | `False` | 156.03% | 5/10 | -8.41% | -0.09% | 1.69 | 10.48 | 19.75% |
| 7 | `fixed_relaxed_dynamic_blend:relaxed50_dynamic50` | `fixed_relaxed_dynamic_blend` | `False` | `False` | 101.45% | 5/10 | -8.81% | -0.23% | 1.75 | 7.66 | 15.73% |
| 8 | `fixed_relaxed_dynamic_blend:relaxed40_dynamic60` | `fixed_relaxed_dynamic_blend` | `False` | `False` | 91.47% | 5/10 | -9.02% | -0.26% | 1.75 | 7.07 | 15.26% |
| 9 | `fixed_relaxed_dynamic_blend:relaxed30_dynamic70` | `fixed_relaxed_dynamic_blend` | `False` | `False` | 81.81% | 5/10 | -9.24% | -0.29% | 1.73 | 6.43 | 14.83% |
| 10 | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | `True` | `False` | 60.18% | 5/10 | -22.55% | -0.09% | 1.12 | 2.32 | 24.27% |
| 11 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 55.03% | 5/10 | -13.65% | -0.44% | 1.33 | 3.25 | 19.78% |
| 12 | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `True` | 54.76% | 5/10 | -9.97% | -0.38% | 1.53 | 4.43 | 16.75% |

## Best candidate monthly OOS detail: `fixed_relaxed_dynamic_blend:relaxed70_dynamic30`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 56.39% | 0.24% | 7.42% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2025-10` | 54.58% | 0.15% | 14.13% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2025-11` | 91.65% | 50.03% | 7.50% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2025-12` | 120.90% | -2.82% | 8.71% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2026-01` | 125.91% | 19.00% | 4.44% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2026-02` | 86.62% | 34.77% | 16.66% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2026-03` | 75.88% | -6.81% | 9.62% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2026-04` | 66.06% | -8.42% | 11.99% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2026-05` | 70.75% | 11.18% | 12.65% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |
| `2026-06` | 154.50% | -0.17% | 0.22% | `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` / `{"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 0.3, "relaxed_efficiency:hybrid_v3_5": 0.7}` |

### Best candidate extended metrics

- OOS comp: `122.36%`
- hit rate: `6/10`
- monthly Sharpe / Sortino approx: `1.74` / `8.97`
- 5% monthly VaR / 25% CVaR: `-7.70%` / `-6.02%`
- avg gain / avg loss: `19.23%` / `-4.56%`
- gain/loss ratio: `4.22`
- max loss streak: `2`
- mean/min validation: `90.32%` / `54.58%`

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
