# Expanded-universe monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-05T15:49:23.534932Z`
- requested symbols: `85`
- loaded symbols with bars: `85`
- missing symbols held for future monitoring/backfill: `0`
- latest available data: `2026-06-05T12:00:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `8` / `32` / `64`
- source symbol workers: `8`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.
- recomputed from existing rows: `True`
- source JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v4_20260605.json`
- source sha256: `48aab7060c57dd4b235f99f67f43d16a4687e6be0f9133286fd82abbf492ec47`
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
| `2026-06` | `2026-06-01T00:00:00` | `2025-01-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-05-31T23:30:00` | `2026-06-01T00:00:00 → 2026-06-05T12:00:00` |

## Raw aggregate ranking (diagnostic only)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 2 | `dynamic_conviction_switch:t0.90_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 3 | `dynamic_conviction_switch:t0.95_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 4 | `dynamic_conviction_switch:t1.00_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 5 | `dynamic_conviction_switch:t0.85_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 6 | `dynamic_conviction_switch:t0.90_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 7 | `dynamic_conviction_switch:t0.95_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 8 | `dynamic_conviction_switch:t1.00_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |

## Clean-promotion ranking (current recommendation set)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 2 | `dynamic_conviction_switch:t0.90_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 3 | `dynamic_conviction_switch:t0.95_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 4 | `dynamic_conviction_switch:t1.00_strict_fallback_val_mdd12_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 12.45% | 4/10 | -14.40% | 0.00% | 0.59 | 0.91 | 1.66 | 16.97% |
| 5 | `dynamic_conviction_switch:t0.85_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 6 | `dynamic_conviction_switch:t0.90_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 7 | `dynamic_conviction_switch:t0.95_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 8 | `dynamic_conviction_switch:t1.00_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 11.46% | 4/10 | -19.42% | 0.00% | 0.51 | 0.68 | 1.58 | 22.58% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.90% | 3/10 | -19.42% | 0.00% | 0.50 | 0.00 | 1.80 | 22.58% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `validation_selector:validation_sharpe_mdd10` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -3.74% | 3/10 | -7.64% | 0.00% | -0.08 | -0.20 | 0.94 | 22.60% |
| 2 | `mdd30_barbell_blend:strict_aggressive_70_strict_balanced_30_x1_25` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -4.37% | 0/1 | -4.37% | -4.37% | 0.00 | 0.00 | 0.00 | 10.41% |
| 3 | `mdd30_barbell_blend:profile_growth_60_strict_growth_40_x1_25` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -10.58% | 0/1 | -10.58% | -10.58% | 0.00 | 0.00 | 0.00 | 12.61% |
| 4 | `validation_selector:validation_calmar_mdd12` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -14.47% | 3/10 | -13.13% | -0.20% | -0.37 | -0.86 | 0.78 | 22.60% |
| 5 | `mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -20.00% | 0/1 | -20.00% | -20.00% | 0.00 | 0.00 | 0.00 | 27.34% |
| 6 | `mdd30_risk_scaled:profile_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -20.00% | 0/1 | -20.00% | -20.00% | 0.00 | 0.00 | 0.00 | 27.34% |
| 7 | `mdd30_risk_scaled:profile_growth_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -22.71% | 1/4 | -17.60% | -2.29% | -1.24 | -2.20 | 0.40 | 27.41% |
| 8 | `mdd30_risk_scaled:relaxed_growth_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -28.71% | 2/9 | -19.42% | -0.22% | -0.89 | -1.45 | 0.49 | 33.96% |
| 9 | `mdd30_risk_scaled:strict_aggressive_x1_25` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -32.01% | 0/4 | -17.53% | -6.54% | -4.36 | -4.36 | 0.00 | 18.19% |
| 10 | `validation_selector:validation_utility_mdd15` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -42.50% | 2/10 | -16.99% | -1.53% | -1.60 | -3.13 | 0.35 | 22.60% |
| 11 | `mdd30_high_vol_gate:breakout_barbell_blend` | `mdd30_high_vol_gate` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -46.12% | 3/10 | -19.84% | -1.80% | -1.38 | -3.13 | 0.40 | 30.01% |
| 12 | `mdd30_barbell_blend:relaxed_aggressive_70_strict_growth_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -49.31% | 1/5 | -29.66% | -13.25% | -2.22 | -5.37 | 0.24 | 35.13% |

## Best clean candidate monthly OOS detail: `dynamic_conviction_switch:t0.85_strict_fallback_val_mdd12_scaled`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 106.22% | -14.40% | 16.97% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna": 1.1}` |
| `2025-10` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled_cash` / `{}` |
| `2025-11` | 41.12% | 18.00% | 14.79% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.5}` |
| `2025-12` | 2.73% | -1.01% | 1.53% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-01` | 11.19% | 10.55% | 4.73% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-02` | 14.15% | 9.75% | 13.93% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 2.0}` |
| `2026-03` | 16.11% | -3.85% | 7.67% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.5}` |
| `2026-04` | 14.04% | -4.05% | 6.42% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-05` | 2.86% | 0.47% | 2.28% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-06` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_strict_fallback_val_mdd12_scaled_cash` / `{}` |

### Best candidate extended metrics

- OOS comp: `12.45%`
- hit rate: `4/10`
- monthly Sharpe / Sortino approx: `0.59` / `0.91`
- profit factor / omega(0): `1.66` / `1.66`
- 5% monthly VaR / 25% CVaR: `-9.74%` / `-7.43%`
- avg gain / avg loss: `9.69%` / `-5.83%`
- gain/loss ratio: `1.66`
- max loss streak: `2`
- mean/min validation: `20.84%` / `0.00%`

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
