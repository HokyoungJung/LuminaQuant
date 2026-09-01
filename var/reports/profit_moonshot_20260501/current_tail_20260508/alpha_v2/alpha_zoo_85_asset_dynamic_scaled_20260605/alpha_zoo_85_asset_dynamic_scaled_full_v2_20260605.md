# Expanded-universe monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-05T14:07:26.394605Z`
- requested symbols: `85`
- loaded symbols with bars: `85`
- missing symbols held for future monitoring/backfill: `0`
- latest available data: `2026-06-05T12:00:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `6` / `24` / `48`
- source symbol workers: `4`
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
| `2026-06` | `2026-06-01T00:00:00` | `2025-01-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-05-31T23:30:00` | `2026-06-01T00:00:00 → 2026-06-05T12:00:00` |

## Raw aggregate ranking (diagnostic only)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 6 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 7 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 8 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |

## Clean-promotion ranking (current recommendation set)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 23.41% | 5/10 | -3.18% | 0.00% | 0.91 | 5.24 | 23.59% |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 6 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 7 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 8 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 18.46% | 5/10 | -3.18% | 0.00% | 0.87 | 4.14 | 19.29% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 13.17% | 5/10 | -3.18% | 0.00% | 0.80 | 2.96 | 14.79% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `validation_selector:validation_calmar_mdd12` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -16.97% | 3/10 | -15.06% | -0.03% | -0.57 | -0.97 | 20.26% |
| 2 | `validation_selector:validation_sharpe_mdd10` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -21.31% | 3/10 | -15.06% | -0.03% | -0.76 | -1.25 | 20.26% |
| 3 | `validation_selector:validation_utility_mdd15` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -29.68% | 3/10 | -15.35% | -0.64% | -0.93 | -1.85 | 26.62% |
| 4 | `mdd30_risk_scaled:relaxed_aggressive_val_mdd30_cap1_75` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -35.08% | 2/9 | -21.84% | -1.07% | -1.20 | -1.82 | 27.66% |
| 5 | `mdd30_risk_scaled:relaxed_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -40.34% | 2/9 | -20.61% | -1.25% | -1.43 | -2.08 | 37.85% |
| 6 | `mdd30_high_vol_gate:breakout_barbell_blend` | `mdd30_high_vol_gate` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -44.27% | 2/9 | -17.51% | -0.39% | -2.37 | -3.71 | 20.69% |
| 7 | `mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 17.32% |
| 8 | `mdd30_risk_scaled:profile_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 17.32% |
| 9 | `mdd30_barbell_blend:profile_aggressive_70_strict_balanced_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 12.76% | 1/2 | -4.17% | 17.67% | 1.51 | 0.00 | 15.29% |
| 10 | `mdd30_barbell_blend:relaxed_aggressive_70_strict_growth_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -19.46% | 1/5 | -16.00% | -0.84% | -0.89 | -1.70 | 21.09% |
| 11 | `mdd30_risk_scaled:strict_aggressive_x1_25` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -20.81% | 1/4 | -17.53% | -0.50% | -2.23 | -2.07 | 18.19% |
| 12 | `mdd30_risk_scaled:relaxed_growth_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -39.15% | 1/9 | -20.42% | -0.22% | -1.31 | -2.44 | 26.48% |

## Best clean candidate monthly OOS detail: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 0.01% | 0.14% | 0.07% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.0}` |
| `2025-10` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled_cash` / `{}` |
| `2025-11` | 70.70% | 28.71% | 23.59% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 2.5}` |
| `2025-12` | 1.92% | -0.13% | 1.63% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-01` | 2.45% | 0.99% | 2.15% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.25}` |
| `2026-02` | 12.51% | 0.26% | 8.44% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 2.5}` |
| `2026-03` | 1.82% | -2.65% | 2.80% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-04` | 1.56% | -3.18% | 3.45% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.25}` |
| `2026-05` | 2.86% | 0.47% | 2.28% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-06` | 0.01% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |

### Best candidate extended metrics

- OOS comp: `23.41%`
- hit rate: `5/10`
- monthly Sharpe / Sortino approx: `0.91` / `5.24`
- 5% monthly VaR / 25% CVaR: `-2.94%` / `-1.99%`
- avg gain / avg loss: `6.11%` / `-1.99%`
- gain/loss ratio: `3.07`
- max loss streak: `2`
- mean/min validation: `9.39%` / `0.00%`

## Timeframe coverage

| Timeframe | Symbols with rows | Symbols skipped | Median rows | Latest |
| --- | ---: | ---: | ---: | --- |
| `30m` | 85 | 0 | 1234.0 | `2026-06-05T12:00:00` |
| `1h` | 85 | 0 | 617.0 | `2026-06-05T11:00:00` |
| `2h` | 85 | 0 | 308.0 | `2026-06-05T10:00:00` |
| `4h` | 85 | 0 | 153.0 | `2026-06-05T08:00:00` |
| `6h` | 85 | 0 | 102.0 | `2026-06-05T06:00:00` |
| `8h` | 85 | 0 | 76.0 | `2026-06-05T00:00:00` |
| `12h` | 85 | 0 | 50.0 | `2026-06-05T00:00:00` |
| `1d` | 85 | 0 | 25.0 | `2026-06-04T00:00:00` |

## Interpretation guardrails

- This is still research/paper-testnet evidence, not real-money approval.
- The latest OOS month can be partial when the data feed ends before month-end.
- If a candidate has a negative validation fold or low OOS consistency, prefer shadow monitoring over allocation.
