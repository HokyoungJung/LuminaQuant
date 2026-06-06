# Expanded-universe monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-06T09:46:44.506177Z`
- requested symbols: `85`
- loaded symbols with bars: `85`
- missing symbols held for future monitoring/backfill: `0`
- latest available data: `2026-06-06T08:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `6` / `24` / `48`
- source symbol workers: `8`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.
- recomputed from existing rows: `True`
- source JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606.json`
- source sha256: `340f955250f345df2c8cf621d95990874b32a4ccc5d6fc74fefa129e2c591757`
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
| `2026-06` | `2026-06-01T00:00:00` | `2025-01-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-05-31T23:30:00` | `2026-06-01T00:00:00 → 2026-06-06T08:30:00` |

## Raw aggregate ranking (diagnostic only)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 61.40% | 4/10 | -3.86% | -3.34% | 1.61 | 50.87 | 8.55 | 29.13% |
| 2 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd25_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 61.40% | 4/10 | -3.86% | -3.34% | 1.61 | 50.87 | 8.55 | 29.13% |
| 3 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap140` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 59.99% | 4/10 | -3.59% | -3.34% | 1.60 | 105.04 | 8.70 | 27.69% |
| 4 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd15_cap125` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 57.79% | 4/10 | -3.34% | -3.34% | 1.58 | 158.71 | 8.92 | 27.69% |
| 5 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 56.15% | 4/10 | -6.59% | -6.59% | 1.53 | 6.09 | 6.58 | 27.69% |
| 6 | `regime_opportunity_leaf_switch:strict30_relaxed15_cap150` | `regime_opportunity_leaf_switch` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 39.34% | 5/10 | -17.00% | 3.84% | 1.00 | 1.67 | 2.39 | 29.13% |
| 7 | `regime_opportunity_leaf_switch:strict30_relaxed15_cap125` | `regime_opportunity_leaf_switch` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 38.67% | 5/10 | -15.29% | 3.84% | 1.00 | 1.73 | 2.43 | 27.69% |
| 8 | `regime_opportunity_leaf_switch:strict30_relaxed20_cap150` | `regime_opportunity_leaf_switch` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 36.83% | 5/10 | -18.87% | 3.84% | 0.91 | 1.43 | 2.18 | 29.13% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |

## Clean-promotion ranking (current recommendation set)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 5 | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | `strict_efficiency` | `True` | `` | `True` | 32.74% | 4/9 | -14.22% | 49.10% | 0.84 | 2.58 | 2.95 | 14.77% |
| 6 | `relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | `relaxed_efficiency` | `True` | `` | `False` | 31.62% | 4/10 | -13.99% | 64.80% | 0.70 | 2.89 | 2.13 | 26.47% |
| 7 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 8 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 9 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 10 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 11 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.82% | 5/10 | -3.18% | -0.07% | 0.94 | 6.19 | 5.84 | 27.69% |
| 12 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.82% | 5/10 | -3.18% | -0.07% | 0.94 | 6.19 | 5.84 | 27.69% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 61.40% | 4/10 | -3.86% | -3.34% | 1.61 | 50.87 | 8.55 | 29.13% |
| 2 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd25_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 61.40% | 4/10 | -3.86% | -3.34% | 1.61 | 50.87 | 8.55 | 29.13% |
| 3 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap140` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 59.99% | 4/10 | -3.59% | -3.34% | 1.60 | 105.04 | 8.70 | 27.69% |
| 4 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd15_cap125` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 57.79% | 4/10 | -3.34% | -3.34% | 1.58 | 158.71 | 8.92 | 27.69% |
| 5 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 56.15% | 4/10 | -6.59% | -6.59% | 1.53 | 6.09 | 6.58 | 27.69% |
| 6 | `regime_opportunity_leaf_switch:strict30_relaxed15_cap150` | `regime_opportunity_leaf_switch` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 39.34% | 5/10 | -17.00% | 3.84% | 1.00 | 1.67 | 2.39 | 29.13% |
| 7 | `regime_opportunity_leaf_switch:strict30_relaxed15_cap125` | `regime_opportunity_leaf_switch` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 38.67% | 5/10 | -15.29% | 3.84% | 1.00 | 1.73 | 2.43 | 27.69% |
| 8 | `regime_opportunity_leaf_switch:strict30_relaxed20_cap150` | `regime_opportunity_leaf_switch` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 36.83% | 5/10 | -18.87% | 3.84% | 0.91 | 1.43 | 2.18 | 29.13% |
| 9 | `mdd30_barbell_blend:relaxed_aggressive_70_strict_growth_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 34.31% | 2/5 | -16.00% | 65.35% | 0.99 | 5.93 | 2.39 | 21.09% |
| 10 | `mdd30_risk_scaled:relaxed_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 26.21% | 3/9 | -20.61% | 108.88% | 0.64 | 3.13 | 2.02 | 37.85% |
| 11 | `mdd30_risk_scaled:relaxed_aggressive_val_mdd30_cap1_75` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 23.95% | 3/9 | -21.84% | 88.90% | 0.62 | 2.69 | 1.91 | 27.66% |
| 12 | `row_level_leaf_selector:validation_return_mdd25` | `row_level_leaf_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 22.11% | 5/10 | -15.35% | 64.80% | 0.59 | 3.56 | 1.70 | 26.62% |

## Best clean candidate monthly OOS detail: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |
| `2025-10` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |
| `2025-11` | 85.56% | 33.48% | 27.69% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 3.0}` |
| `2025-12` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |
| `2026-01` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |
| `2026-02` | 15.01% | 0.21% | 10.08% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 3.0}` |
| `2026-03` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |
| `2026-04` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |
| `2026-05` | 2.86% | 0.47% | 2.28% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-06` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled_cash` / `{}` |

### Best candidate extended metrics

- OOS comp: `34.39%`
- hit rate: `3/10`
- monthly Sharpe / Sortino approx: `1.12` / `∞`
- profit factor / omega(0): `∞` / `∞`
- 5% monthly VaR / 25% CVaR: `0.00%` / `0.00%`
- avg gain / avg loss: `11.39%` / `0.00%`
- gain/loss ratio: `∞`
- max loss streak: `0`
- mean/min validation: `10.34%` / `0.00%`

## Timeframe coverage

| Timeframe | Symbols with rows | Symbols skipped | Median rows | Latest |
| --- | ---: | ---: | ---: | --- |
| `30m` | 85 | 0 | 1478.0 | `2026-06-06T08:30:00` |
| `1h` | 85 | 0 | 739.0 | `2026-06-06T08:00:00` |
| `2h` | 85 | 0 | 369.0 | `2026-06-06T06:00:00` |
| `4h` | 85 | 0 | 184.0 | `2026-06-06T04:00:00` |
| `6h` | 85 | 0 | 122.0 | `2026-06-06T00:00:00` |
| `8h` | 85 | 0 | 92.0 | `2026-06-06T00:00:00` |
| `12h` | 85 | 0 | 60.0 | `2026-06-05T12:00:00` |
| `1d` | 85 | 0 | 30.0 | `2026-06-05T00:00:00` |

## Interpretation guardrails

- This is still research/paper-testnet evidence, not real-money approval.
- The latest OOS month can be partial when the data feed ends before month-end.
- If a candidate has a negative validation fold or low OOS consistency, prefer shadow monitoring over allocation.
