# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-02T05:07:22.685733Z`
- latest available data: `2026-06-01T06:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `6` / `24` / `48`
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
| 1 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.33% | 5/10 | -2.65% | 0.00% | 2.07 | 13.82 | 19.20% |
| 2 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.33% | 5/10 | -2.65% | 0.00% | 2.07 | 13.82 | 19.20% |
| 3 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 52.20% | 5/10 | -2.65% | -0.73% | 2.02 | 15.08 | 19.20% |
| 4 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 52.20% | 5/10 | -2.65% | -0.73% | 2.02 | 15.08 | 19.20% |
| 5 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `True` | `True` | 30.85% | 5/10 | -5.01% | -0.58% | 1.27 | 5.91 | 14.25% |
| 6 | `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_riskcap` | `hybrid_oracle_bridge` | `True` | `False` | 18.39% | 5/10 | -7.62% | -0.73% | 0.81 | 2.50 | 19.20% |
| 7 | `profile_optuna:selected_optuna` | `profile_optuna` | `True` | `False` | 10.05% | 5/10 | -9.30% | -0.73% | 0.50 | 1.16 | 19.20% |
| 8 | `profile_optuna:hybrid_v3_5` | `profile_optuna` | `True` | `False` | 9.06% | 5/10 | -9.30% | -0.23% | 0.47 | 1.04 | 19.20% |
| 9 | `profile_optuna:selected_train_validation_legal` | `profile_optuna` | `True` | `False` | -2.47% | 5/10 | -16.68% | -0.73% | 0.08 | 0.13 | 31.33% |
| 10 | `profile_optuna:hybrid_v3_6` | `profile_optuna` | `True` | `False` | -10.58% | 5/10 | -13.91% | -0.73% | -0.34 | -0.49 | 22.35% |
| 11 | `profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna` | `profile_optuna` | `True` | `False` | -20.50% | 5/10 | -15.35% | -0.49% | -0.92 | -1.16 | 31.33% |
| 12 | `dynamic_conviction_switch:t0.95_strict_fallback` | `dynamic_conviction_switch` | `True` | `False` | 43.44% | 4/10 | -8.56% | -0.17% | 1.61 | 4.16 | 19.20% |

## Best candidate monthly OOS detail: `dynamic_conviction_switch:t0.95_risk_capped_fallback`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 0.01% | 0.14% | 0.07% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-10` | 1.73% | -0.72% | 1.39% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-11` | 26.81% | 12.20% | 10.08% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-12` | 1.92% | -0.13% | 1.63% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-01` | 79.51% | 10.25% | 4.02% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"cross_candidate_hybrid:hybrid_v3_5": 1.0}` |
| `2026-02` | 111.17% | 20.19% | 19.20% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"profile_optuna:selected_optuna": 1.0}` |
| `2026-03` | 1.82% | -2.65% | 2.80% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-04` | 5.79% | -2.02% | 4.57% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-05` | 41.22% | 8.91% | 18.03% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"profile_optuna:selected_optuna": 1.0}` |
| `2026-06` | 0.01% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.95_risk_capped_fallback` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |

### Best candidate extended metrics

- OOS comp: `53.33%`
- hit rate: `5/10`
- monthly Sharpe / Sortino approx: `2.07` / `13.82`
- 5% monthly VaR / 25% CVaR: `-2.37%` / `-1.80%`
- avg gain / avg loss: `10.34%` / `-1.38%`
- gain/loss ratio: `7.48`
- max loss streak: `2`
- mean/min validation: `27.00%` / `0.01%`

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
