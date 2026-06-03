# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-02T09:02:09.299477Z`
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
| 1 | `risk_enhanced_blend:dyn085_50_aware_v35_50` | `risk_enhanced_blend` | `False` | `False` | 39.31% | 6/10 | -3.69% | -0.71% | 1.83 | 7.65 | 14.96% |
| 2 | `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | `True` | `False` | 25.12% | 6/10 | -5.62% | -0.71% | 1.22 | 3.29 | 13.84% |
| 3 | `dynamic_aware_hybrid:hybrid_v3_6` | `dynamic_aware_hybrid` | `True` | `False` | 22.50% | 6/10 | -3.81% | -0.45% | 1.31 | 4.89 | 11.12% |
| 4 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 5 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 6 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 53.38% | 5/10 | -2.65% | -0.71% | 2.07 | 15.31 | 18.80% |
| 7 | `risk_enhanced_blend:dyn085_70_aware_v36tv_30` | `risk_enhanced_blend` | `False` | `False` | 47.60% | 5/10 | -2.35% | -0.73% | 1.98 | 16.23 | 14.67% |
| 8 | `risk_enhanced_blend:dyn085_60_aware_v36tv_40` | `risk_enhanced_blend` | `False` | `False` | 45.60% | 5/10 | -2.46% | -0.74% | 1.93 | 16.93 | 14.14% |
| 9 | `risk_enhanced_blend:dyn085_50_aware_v36tv_50` | `risk_enhanced_blend` | `False` | `False` | 43.56% | 5/10 | -2.58% | -0.75% | 1.87 | 17.10 | 13.66% |
| 10 | `risk_enhanced_blend:dyn085_60_aware_v36tv_30_strict_growth_10` | `risk_enhanced_blend` | `False` | `False` | 42.81% | 5/10 | -2.49% | -0.68% | 1.95 | 15.04 | 13.11% |
| 11 | `risk_enhanced_blend:dyn085_60_aware_v36_40` | `risk_enhanced_blend` | `False` | `False` | 40.97% | 5/10 | -2.72% | -0.61% | 1.91 | 16.62 | 14.04% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `False` | 39.53% | 5/10 | -2.65% | 0.00% | 1.69 | 10.82 | 18.80% |

## Best candidate monthly OOS detail: `risk_enhanced_blend:dyn085_50_aware_v35_50`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 15.88% | -0.60% | 1.82% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2025-10` | 21.60% | 0.28% | 4.42% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2025-11` | 30.05% | 3.07% | 6.92% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2025-12` | 28.59% | 0.07% | 2.38% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2026-01` | 71.74% | 9.70% | 3.51% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2026-02` | 103.47% | 18.04% | 14.29% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2026-03` | 21.11% | -0.18% | 2.25% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2026-04` | 29.45% | -3.69% | 6.07% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2026-05` | 41.98% | 9.63% | 14.96% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |
| `2026-06` | 81.49% | -0.71% | 0.95% | `risk_enhanced_blend:dyn085_50_aware_v35_50` / `{"dynamic_aware_hybrid:hybrid_v3_5": 0.5, "dynamic_conviction_switch:t0.85_risk_capped_fallback": 0.5}` |

### Best candidate extended metrics

- OOS comp: `39.31%`
- hit rate: `6/10`
- monthly Sharpe / Sortino approx: `1.83` / `7.65`
- 5% monthly VaR / 25% CVaR: `-2.35%` / `-1.67%`
- avg gain / avg loss: `6.80%` / `-1.30%`
- gain/loss ratio: `5.25`
- max loss streak: `2`
- mean/min validation: `44.54%` / `15.88%`

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
