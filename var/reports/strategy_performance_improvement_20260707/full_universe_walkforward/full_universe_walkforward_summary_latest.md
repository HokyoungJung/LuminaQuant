# Expanded-universe monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-07-08T11:47:52.322760Z`
- requested symbols: `110`
- loaded symbols with bars: `110`
- missing symbols held for future monitoring/backfill: `0`
- latest available data: `2026-07-04T06:30:00`
- allowed timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage/cost proxy: `10.0` bps
- folds: `11` (`2025-09` → `2026-07`)
- trials: asset/profile/hybrid = `6` / `24` / `48`
- source symbol workers: `2`
- selection/refit input: train + 2M validation only; OOS month is evaluated after frozen fold params.
- locked-OOS rankings/tables: diagnostic-only; not consumed for selection, weighting, sizing, or clean/live promotion.

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
| `2026-06` | `2026-06-01T00:00:00` | `2025-01-01T00:00:00 → 2026-03-31T23:30:00` | `2026-04-01T00:00:00 → 2026-05-31T23:30:00` | `2026-06-01T00:00:00 → 2026-06-30T23:30:00` |
| `2026-07` | `2026-07-01T00:00:00` | `2025-01-01T00:00:00 → 2026-04-30T23:30:00` | `2026-05-01T00:00:00 → 2026-06-30T23:30:00` | `2026-07-01T00:00:00 → 2026-07-04T06:30:00` |

## Raw aggregate ranking (diagnostic only)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 63.36% | 4/11 | -8.59% | -8.59% | 1.50 | 4.48 | 5.11 | 27.69% |
| 2 | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 51.18% | 4/11 | -8.59% | -8.59% | 1.52 | 3.69 | 4.39 | 25.01% |
| 3 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 46.31% | 3/11 | -2.51% | -0.91% | 1.31 | 13.38 | 11.71 | 27.69% |
| 4 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd12_cap110_trimmed` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 39.14% | 3/11 | -2.78% | -0.91% | 1.17 | 10.52 | 9.58 | 27.69% |
| 5 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd15_cap125` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 38.46% | 3/11 | -3.18% | -1.15% | 1.15 | 9.43 | 8.16 | 27.69% |
| 6 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap140` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 37.83% | 3/11 | -3.59% | -1.29% | 1.13 | 8.41 | 7.20 | 27.69% |
| 7 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 37.36% | 3/11 | -3.86% | -1.38% | 1.12 | 7.83 | 6.66 | 27.69% |
| 8 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd25_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 37.36% | 3/11 | -3.86% | -1.38% | 1.12 | 7.83 | 6.66 | 27.69% |
| 9 | `mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 25.33 | 17.32% |
| 10 | `mdd30_risk_scaled:profile_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 25.33 | 17.32% |
| 11 | `mdd30_barbell_blend:profile_aggressive_70_strict_balanced_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 12.76% | 1/2 | -4.17% | 17.67% | 1.51 | 0.00 | 4.24 | 15.29% |
| 12 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 7.99% | 3/11 | -16.71% | -16.71% | 0.38 | 0.00 | 1.76 | 23.59% |

## Clean candidate OOS diagnostics (not a promotion decision)

These rows are locked-OOS diagnostics for clean-eligible candidates. Promotion still requires the hard-stop gate and fresh forward/shadow evidence; this table is not an input to fold selection, weighting, sizing, or live routing.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 7.99% | 3/11 | -16.71% | -16.71% | 0.38 | 0.00 | 1.76 | 23.59% |
| 2 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 7.99% | 3/11 | -16.71% | -16.71% | 0.38 | 0.00 | 1.76 | 23.59% |
| 3 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 7.99% | 3/11 | -16.71% | -16.71% | 0.38 | 0.00 | 1.76 | 23.59% |
| 4 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 7.99% | 3/11 | -16.71% | -16.71% | 0.38 | 0.00 | 1.76 | 23.59% |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 6.97% | 3/11 | -14.05% | -14.05% | 0.37 | 0.00 | 1.73 | 19.29% |
| 6 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 6.97% | 3/11 | -14.05% | -14.05% | 0.37 | 0.00 | 1.73 | 19.29% |
| 7 | `dynamic_conviction_switch:t0.95_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 6.97% | 3/11 | -14.05% | -14.05% | 0.37 | 0.00 | 1.73 | 19.29% |
| 8 | `dynamic_conviction_switch:t1.00_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | `True` | `` | `False` | 6.97% | 3/11 | -14.05% | -14.05% | 0.37 | 0.00 | 1.73 | 19.29% |
| 9 | `strict_efficiency:static_guarded` | `strict_efficiency` | `True` | `` | `False` | 0.55% | 3/11 | -8.56% | -0.91% | 0.12 | 0.24 | 1.11 | 11.81% |
| 10 | `tradfi_intraday_session_v1:open_impulse_close_top10_mdd15` | `tradfi_intraday_session_v1` | `True` | `` | `False` | 0.48% | 2/11 | -0.30% | -0.30% | 0.79 | 0.00 | 2.61 | 0.36% |
| 11 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate` | `dynamic_conviction_switch` | `True` | `` | `False` | 0.15% | 3/11 | -11.34% | -11.34% | 0.09 | 0.00 | 1.14 | 10.88% |
| 12 | `dynamic_conviction_switch:t0.90_risk_capped_fallback_val_ret02_calmar80_gate` | `dynamic_conviction_switch` | `True` | `` | `False` | 0.15% | 3/11 | -11.34% | -11.34% | 0.09 | 0.00 | 1.14 | 10.88% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | PF | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 63.36% | 4/11 | -8.59% | -8.59% | 1.50 | 4.48 | 5.11 | 27.69% |
| 2 | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 51.18% | 4/11 | -8.59% | -8.59% | 1.52 | 3.69 | 4.39 | 25.01% |
| 3 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 46.31% | 3/11 | -2.51% | -0.91% | 1.31 | 13.38 | 11.71 | 27.69% |
| 4 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd12_cap110_trimmed` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 39.14% | 3/11 | -2.78% | -0.91% | 1.17 | 10.52 | 9.58 | 27.69% |
| 5 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd15_cap125` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 38.46% | 3/11 | -3.18% | -1.15% | 1.15 | 9.43 | 8.16 | 27.69% |
| 6 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap140` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 37.83% | 3/11 | -3.59% | -1.29% | 1.13 | 8.41 | 7.20 | 27.69% |
| 7 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 37.36% | 3/11 | -3.86% | -1.38% | 1.12 | 7.83 | 6.66 | 27.69% |
| 8 | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd25_cap150` | `lagged_shadow_leaf_router` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 37.36% | 3/11 | -3.86% | -1.38% | 1.12 | 7.83 | 6.66 | 27.69% |
| 9 | `mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 25.33 | 17.32% |
| 10 | `mdd30_risk_scaled:profile_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 15.69% | 1/2 | -0.65% | 16.44% | 2.26 | 0.00 | 25.33 | 17.32% |
| 11 | `mdd30_barbell_blend:profile_aggressive_70_strict_balanced_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 12.76% | 1/2 | -4.17% | 17.67% | 1.51 | 0.00 | 4.24 | 15.29% |
| 12 | `factor_regime_router_v1:release_lag_not_encoded_diagnostic_reject` | `factor_regime_router_v1` | `False` | `requires_fresh_forward_shadow` | `False` | 0.00% | 0/11 | 0.00% | 0.00% | 0.00 | 0.00 | 0.00 | 0.00% |

## Best clean candidate monthly OOS detail: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2025-10` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2025-11` | 70.70% | 28.71% | 23.59% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 2.5}` |
| `2025-12` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2026-01` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2026-02` | 12.51% | 0.26% | 8.44% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 2.5}` |
| `2026-03` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2026-04` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2026-05` | 2.86% | 0.47% | 2.28% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` / `{"strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}` |
| `2026-06` | 0.00% | 0.00% | 0.00% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled_cash` / `{}` |
| `2026-07` | 180.50% | -16.71% | 16.03% | `dynamic_conviction_switch_t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` / `{"profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna": 1.5}` |

### Best candidate extended metrics

- OOS comp: `7.99%`
- hit rate: `3/11`
- monthly Sharpe / Sortino approx: `0.38` / `0.00`
- profit factor / omega(0): `1.76` / `1.76`
- 5% monthly VaR / 25% CVaR: `-8.35%` / `-2.09%`
- avg gain / avg loss: `9.81%` / `-16.71%`
- gain/loss ratio: `0.59`
- max loss streak: `1`
- mean/min validation: `24.23%` / `0.00%`

## Timeframe coverage

| Timeframe | Symbols with rows | Symbols skipped | Median rows | Latest |
| --- | ---: | ---: | ---: | --- |
| `30m` | 110 | 0 | 2212.0 | `2026-07-04T06:30:00` |
| `1h` | 110 | 0 | 1106.0 | `2026-07-04T06:00:00` |
| `2h` | 110 | 0 | 552.0 | `2026-07-04T04:00:00` |
| `4h` | 110 | 0 | 275.0 | `2026-07-04T00:00:00` |
| `6h` | 110 | 0 | 183.0 | `2026-07-04T00:00:00` |
| `8h` | 110 | 0 | 137.0 | `2026-07-03T16:00:00` |
| `12h` | 110 | 0 | 90.0 | `2026-07-03T12:00:00` |
| `1d` | 110 | 0 | 45.0 | `2026-07-03T00:00:00` |

## Interpretation guardrails

- This is still research/paper-testnet evidence, not real-money approval.
- The latest OOS month can be partial when the data feed ends before month-end.
- If a candidate has a negative validation fold or low OOS consistency, prefer shadow monitoring over allocation.
