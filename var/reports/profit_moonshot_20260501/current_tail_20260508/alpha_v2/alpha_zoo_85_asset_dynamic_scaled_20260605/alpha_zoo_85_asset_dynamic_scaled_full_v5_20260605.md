# Expanded-universe monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-05T15:49:24.230615Z`
- requested symbols: `85`
- loaded symbols with bars: `85`
- missing symbols held for future monitoring/backfill: `0`
- latest available data: `2026-06-05T12:00:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `10` (`2025-09` → `2026-06`)
- trials: asset/profile/hybrid = `6` / `24` / `48`
- source symbol workers: `8`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.
- recomputed from existing rows: `True`
- source JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.json`
- source sha256: `092493fd74287b9354ed6f822020f45f3ba81ae21c33e58cb81de3ad21176eb9`
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
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 6 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 7 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 8 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |

## Clean-promotion ranking (current recommendation set)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 34.39% | 3/10 | 0.00% | 0.00% | 1.12 | ∞ | ∞ | 27.69% |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 6 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 7 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 8 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 29.65% | 3/10 | 0.00% | 0.00% | 1.13 | ∞ | ∞ | 23.59% |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |
| 10 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |
| 11 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |
| 12 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_mdd30_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 27.92% | 5/10 | -3.18% | 0.00% | 0.94 | 6.24 | 5.92 | 27.69% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 25.33 | 17.32% |
| 2 | `mdd30_risk_scaled:profile_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 25.33 | 17.32% |
| 3 | `mdd30_barbell_blend:profile_aggressive_70_strict_balanced_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 12.76% | 1/2 | -4.17% | 17.67% | 1.51 | 0.00 | 4.24 | 15.29% |
| 4 | `mdd30_barbell_blend:strict_aggressive_70_strict_balanced_30_x1_25` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -2.81% | 0/1 | -2.81% | -2.81% | 0.00 | 0.00 | 0.00 | 8.48% |
| 5 | `mdd30_barbell_blend:profile_growth_60_strict_growth_40_x1_25` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -10.84% | 0/2 | -10.43% | -0.46% | -2.67 | -2.67 | 0.00 | 12.65% |
| 6 | `validation_selector:validation_calmar_mdd12` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -16.97% | 3/10 | -15.06% | -0.03% | -0.57 | -0.97 | 0.66 | 20.26% |
| 7 | `mdd30_barbell_blend:relaxed_aggressive_70_strict_growth_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -19.46% | 1/5 | -16.00% | -0.84% | -0.89 | -1.70 | 0.51 | 21.09% |
| 8 | `mdd30_risk_scaled:strict_aggressive_x1_25` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -20.81% | 1/4 | -17.53% | -0.50% | -2.23 | -2.07 | 0.03 | 18.19% |
| 9 | `validation_selector:validation_sharpe_mdd10` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -21.31% | 3/10 | -15.06% | -0.03% | -0.76 | -1.25 | 0.58 | 20.26% |
| 10 | `validation_selector:validation_utility_mdd15` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -29.68% | 3/10 | -15.35% | -0.64% | -0.93 | -1.85 | 0.53 | 26.62% |
| 11 | `mdd30_risk_scaled:relaxed_aggressive_val_mdd30_cap1_75` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -35.08% | 2/9 | -21.84% | -1.07% | -1.20 | -1.82 | 0.39 | 27.66% |
| 12 | `mdd30_risk_scaled:profile_growth_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -37.06% | 0/3 | -23.29% | -0.97% | -4.15 | -4.15 | 0.00 | 37.83% |

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
