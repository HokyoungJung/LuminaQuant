# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-02T07:15:59.632464Z`
- latest available data: `2026-06-01T06:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `6` / `24` / `96`
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
| 1 | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | `True` | `False` | 25.12% | 6/10 | -5.62% | -0.71% | 1.22 | 3.29 | 13.84% |
| 2 | `dynamic_aware_hybrid:hybrid_v3_6` | `dynamic_aware_hybrid` | `True` | `False` | 22.50% | 6/10 | -3.81% | -0.45% | 1.31 | 4.89 | 11.12% |
| 3 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 4 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 5 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 6 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 39.53% | 5/10 | -2.65% | 0.00% | 1.69 | 10.82 | 18.80% |
| 7 | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `True` | 32.94% | 5/10 | -3.19% | -0.78% | 1.45 | 10.16 | 11.22% |
| 8 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | `True` | 27.01% | 5/10 | -4.31% | -0.84% | 1.24 | 6.33 | 13.72% |
| 9 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `True` | `False` | 25.74% | 5/10 | -5.39% | -0.83% | 1.16 | 4.23 | 13.81% |
| 10 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `True` | `False` | 22.07% | 5/10 | -4.01% | -0.71% | 1.07 | 6.27 | 13.85% |
| 11 | `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | `True` | `False` | 21.87% | 5/10 | -5.63% | -0.70% | 1.06 | 3.46 | 15.39% |
| 12 | `profile_optuna:selected_train_validation_legal` | `profile_optuna` | `True` | `False` | 18.32% | 5/10 | -8.46% | -0.71% | 0.80 | 2.07 | 18.80% |

## Best candidate monthly OOS detail: `dynamic_aware_hybrid:hybrid_v3_5`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 34.06% | -1.35% | 3.62% | `hybrid_v3_5_optuna_three_profile_blend` / `{"cross_candidate_hybrid:hybrid_v3_5": 0.7945996802215437, "cross_candidate_hybrid:hybrid_v3_5_train_validation_fit": 0.01834773566516939...` |
| `2025-10` | 44.52% | 1.01% | 8.06% | `hybrid_v3_5_optuna_three_profile_blend` / `{"cross_candidate_hybrid:hybrid_v3_5": 0.004460256364037764, "cross_candidate_hybrid:hybrid_v3_5_train_validation_fit": 0.004448618799935...` |
| `2025-11` | 32.53% | -5.62% | 9.72% | `hybrid_v3_5_optuna_three_profile_blend` / `{"cross_candidate_hybrid:hybrid_v3_6": 0.3768373917514856, "cross_candidate_hybrid:hybrid_v3_6_train_validation_fit": 0.0055987242642166,...` |
| `2025-12` | 60.44% | 0.20% | 4.64% | `hybrid_v3_5_optuna_three_profile_blend` / `{"dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.01272205726026693, "dynamic_conviction_switch:t0.90_risk_capped_fallback": 0.0...` |
| `2026-01` | 63.72% | 9.76% | 3.10% | `hybrid_v3_5_optuna_three_profile_blend` / `{"cross_candidate_hybrid:hybrid_v3_5": 0.02982372411619716, "cross_candidate_hybrid:hybrid_v3_5_train_validation_fit": 0.0263886314691301...` |
| `2026-02` | 97.87% | 16.56% | 10.52% | `hybrid_v3_5_optuna_three_profile_blend` / `{"cross_candidate_hybrid:hybrid_v3_5": 0.005532997107771664, "cross_candidate_hybrid:hybrid_v3_5_train_validation_fit": 0.140478992074227...` |
| `2026-03` | 43.79% | 2.32% | 2.50% | `hybrid_v3_5_optuna_three_profile_blend` / `{"dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.02874936096246721, "dynamic_conviction_switch:t0.90_risk_capped_fallback": 0.0...` |
| `2026-04` | 57.65% | -5.42% | 9.43% | `hybrid_v3_5_optuna_three_profile_blend` / `{"dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.008884341643193239, "dynamic_conviction_switch:t0.90_risk_capped_fallback": 0....` |
| `2026-05` | 40.25% | 8.01% | 13.84% | `hybrid_v3_5_optuna_three_profile_blend` / `{"dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.4404322689018978, "dynamic_conviction_switch:t0.90_risk_capped_fallback": 0.00...` |
| `2026-06` | 76.48% | -0.71% | 0.87% | `hybrid_v3_5_optuna_three_profile_blend` / `{"dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.6955597462061197, "dynamic_conviction_switch:t0.85_strict_fallback": 0.0058504...` |

### Best candidate extended metrics

- OOS comp: `25.12%`
- hit rate: `6/10`
- monthly Sharpe / Sortino approx: `1.22` / `3.29`
- 5% monthly VaR / 25% CVaR: `-5.53%` / `-4.13%`
- avg gain / avg loss: `6.31%` / `-3.28%`
- gain/loss ratio: `1.93`
- max loss streak: `1`
- mean/min validation: `55.13%` / `32.53%`

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
