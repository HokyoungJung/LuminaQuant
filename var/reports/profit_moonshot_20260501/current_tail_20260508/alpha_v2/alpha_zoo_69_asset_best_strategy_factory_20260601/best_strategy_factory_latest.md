# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-01T15:17:59.770659Z`
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

| Rank | Candidate | Family | Clean | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | 39.53% | 5/10 | -2.65% | 0.00% | 1.69 | 10.82 | 18.80% |
| 5 | `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | `True` | 27.01% | 5/10 | -4.31% | -0.84% | 1.24 | 6.33 | 13.72% |
| 6 | `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | `True` | 25.74% | 5/10 | -5.39% | -0.83% | 1.16 | 4.23 | 13.81% |
| 7 | `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | `True` | 22.07% | 5/10 | -4.01% | -0.71% | 1.07 | 6.27 | 13.85% |
| 8 | `profile_optuna:selected_train_validation_legal` | `profile_optuna` | `True` | 18.32% | 5/10 | -8.46% | -0.71% | 0.80 | 2.07 | 18.80% |
| 9 | `profile_optuna:selected_optuna` | `profile_optuna` | `True` | 14.66% | 5/10 | -8.46% | -0.71% | 0.68 | 1.74 | 18.80% |
| 10 | `profile_optuna:hybrid_v3_5` | `profile_optuna` | `True` | 9.78% | 5/10 | -12.10% | -0.37% | 0.49 | 0.96 | 18.80% |
| 11 | `profile_optuna:hybrid_v3_6` | `profile_optuna` | `True` | -0.24% | 5/10 | -9.62% | -0.71% | 0.10 | 0.16 | 16.66% |
| 12 | `profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna` | `profile_optuna` | `True` | -20.50% | 5/10 | -15.35% | -0.49% | -0.92 | -1.16 | 31.33% |

## Best candidate monthly OOS detail: `dynamic_conviction_switch:t0.85_risk_capped_fallback`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 0.01% | 0.14% | 0.07% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-10` | 1.73% | -0.72% | 1.39% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-11` | 26.81% | 12.20% | 10.08% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-12` | 1.92% | -0.13% | 1.63% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-01` | 80.04% | 9.62% | 3.92% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"cross_candidate_hybrid:hybrid_v3_5": 1.0}` |
| `2026-02` | 108.23% | 19.24% | 18.80% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"profile_optuna:selected_optuna": 1.0}` |
| `2026-03` | 1.82% | -2.65% | 2.80% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-04` | 5.79% | -2.02% | 4.57% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-05` | 43.70% | 11.24% | 16.08% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"profile_optuna:selected_optuna": 1.0}` |
| `2026-06` | 86.60% | -0.71% | 1.04% | `dynamic_conviction_switch_t0.85_risk_capped_fallback` / `{"profile_optuna:selected_optuna": 1.0}` |

### Best candidate extended metrics

- OOS comp: `53.38%`
- hit rate: `5/10`
- monthly Sharpe / Sortino approx: `2.07` / `15.31`
- 5% monthly VaR / 25% CVaR: `-2.37%` / `-1.80%`
- avg gain / avg loss: `10.49%` / `-1.25%`
- gain/loss ratio: `8.40`
- max loss streak: `2`
- mean/min validation: `35.67%` / `0.01%`

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
