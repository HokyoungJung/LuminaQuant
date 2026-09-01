# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-02T11:44:00.733804Z`
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
| 1 | `individual_robust:hybrid_v3_5` | `individual_robust` | `True` | `False` | 3.10% | 1/1 | 3.10% | 3.10% | 0.00 | 0.00 | 5.79% |
| 2 | `individual_robust:selected_optuna` | `individual_robust` | `True` | `False` | 3.10% | 1/1 | 3.10% | 3.10% | 0.00 | 0.00 | 5.79% |
| 3 | `relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | `relaxed_efficiency` | `True` | `False` | 2.40% | 1/1 | 2.40% | 2.40% | 0.00 | 0.00 | 9.27% |
| 4 | `profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna` | `profile_optuna` | `True` | `False` | 1.62% | 1/1 | 1.62% | 1.62% | 0.00 | 0.00 | 16.44% |
| 5 | `individual_robust:hybrid_v3_6` | `individual_robust` | `True` | `False` | 0.83% | 1/1 | 0.83% | 0.83% | 0.00 | 0.00 | 6.26% |
| 6 | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `False` | 0.51% | 1/1 | 0.51% | 0.51% | 0.00 | 0.00 | 7.09% |
| 7 | `mdd30_risk_scaled:dyn085_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `False` | 0.20% | 1/1 | 0.20% | 0.20% | 0.00 | 0.00 | 0.11% |
| 8 | `mdd30_risk_scaled:dyn085_x1_50` | `mdd30_risk_scaled` | `False` | `False` | 0.20% | 1/1 | 0.20% | 0.20% | 0.00 | 0.00 | 0.11% |
| 9 | `mdd30_risk_scaled:dyn100_x1_50` | `mdd30_risk_scaled` | `False` | `False` | 0.20% | 1/1 | 0.20% | 0.20% | 0.00 | 0.00 | 0.11% |
| 10 | `mdd30_risk_scaled:dyn085_x1_25` | `mdd30_risk_scaled` | `False` | `False` | 0.17% | 1/1 | 0.17% | 0.17% | 0.00 | 0.00 | 0.09% |
| 11 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 0.14% | 1/1 | 0.14% | 0.14% | 0.00 | 0.00 | 0.07% |
| 12 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 0.14% | 1/1 | 0.14% | 0.14% | 0.00 | 0.00 | 0.07% |

## Best candidate monthly OOS detail: `individual_robust:hybrid_v3_5`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 25.41% | 3.10% | 5.79% | `hybrid_v3_5_optuna_three_profile_blend` / `{"individual_robust_balanced_mdd10_gross3_core10": 0.7121442456627092, "individual_robust_growth_mdd14_gross5_core14": 0.1447888430853744...` |

### Best candidate extended metrics

- OOS comp: `3.10%`
- hit rate: `1/1`
- monthly Sharpe / Sortino approx: `0.00` / `0.00`
- 5% monthly VaR / 25% CVaR: `3.10%` / `3.10%`
- avg gain / avg loss: `3.10%` / `0.00%`
- gain/loss ratio: `0.00`
- max loss streak: `0`
- mean/min validation: `25.41%` / `25.41%`

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
