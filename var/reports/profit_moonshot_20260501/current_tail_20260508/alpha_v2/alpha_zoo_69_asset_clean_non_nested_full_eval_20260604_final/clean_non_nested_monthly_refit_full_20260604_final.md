# 69-asset monthly-refit walk-forward: 2M validation / 1M OOS

- generated: `2026-06-04T13:18:51.185028Z`
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

## Raw aggregate ranking (diagnostic only)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `profile_optuna:selected_optuna` | `profile_optuna` | `True` | `` | `False` | 10.05% | 5/10 | -9.30% | -0.73% | 0.50 | 1.16 | 19.20% |
| 2 | `profile_optuna:hybrid_v3_5` | `profile_optuna` | `True` | `` | `False` | 9.06% | 5/10 | -9.30% | -0.23% | 0.47 | 1.04 | 19.20% |
| 3 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 4 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 5 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 6 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 7 | `profile_optuna:selected_train_validation_legal` | `profile_optuna` | `True` | `` | `False` | -2.47% | 5/10 | -16.68% | -0.73% | 0.08 | 0.13 | 31.33% |
| 8 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `True` | `` | `False` | -7.88% | 5/10 | -10.70% | -0.84% | -0.47 | -0.65 | 13.33% |
| 9 | `profile_optuna:hybrid_v3_6` | `profile_optuna` | `True` | `` | `False` | -10.58% | 5/10 | -13.91% | -0.73% | -0.34 | -0.49 | 22.35% |
| 10 | `profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna` | `profile_optuna` | `True` | `` | `False` | -20.50% | 5/10 | -15.35% | -0.49% | -0.92 | -1.16 | 31.33% |
| 11 | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.47% | 4/9 | -8.56% | -0.17% | 0.70 | 1.26 | 10.62% |
| 12 | `dynamic_conviction_switch:t0.90_strict_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.47% | 4/9 | -8.56% | -0.17% | 0.70 | 1.26 | 10.62% |

## Clean-promotion ranking (current recommendation set)

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `profile_optuna:selected_optuna` | `profile_optuna` | `True` | `` | `False` | 10.05% | 5/10 | -9.30% | -0.73% | 0.50 | 1.16 | 19.20% |
| 2 | `profile_optuna:hybrid_v3_5` | `profile_optuna` | `True` | `` | `False` | 9.06% | 5/10 | -9.30% | -0.23% | 0.47 | 1.04 | 19.20% |
| 3 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 4 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 5 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 6 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 8.61% | 5/9 | -2.65% | 0.00% | 0.80 | 2.65 | 10.08% |
| 7 | `profile_optuna:selected_train_validation_legal` | `profile_optuna` | `True` | `` | `False` | -2.47% | 5/10 | -16.68% | -0.73% | 0.08 | 0.13 | 31.33% |
| 8 | `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | `True` | `` | `False` | -7.88% | 5/10 | -10.70% | -0.84% | -0.47 | -0.65 | 13.33% |
| 9 | `profile_optuna:hybrid_v3_6` | `profile_optuna` | `True` | `` | `False` | -10.58% | 5/10 | -13.91% | -0.73% | -0.34 | -0.49 | 22.35% |
| 10 | `profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna` | `profile_optuna` | `True` | `` | `False` | -20.50% | 5/10 | -15.35% | -0.49% | -0.92 | -1.16 | 31.33% |
| 11 | `dynamic_conviction_switch:t0.85_strict_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.47% | 4/9 | -8.56% | -0.17% | 0.70 | 1.26 | 10.62% |
| 12 | `dynamic_conviction_switch:t0.90_strict_fallback` | `dynamic_conviction_switch` | `True` | `` | `False` | 10.47% | 4/9 | -8.56% | -0.17% | 0.70 | 1.26 | 10.62% |

## Demoted nested/historical ranking

These rows may remain useful diagnostics, but they are not current clean-promotion evidence.

| Rank | Candidate | Family | Clean | Reasons | Hard-stop | OOS comp | OOS pos | Min OOS | Latest OOS | Sharpe | Sortino | Max OOS MDD |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `validation_selector:validation_calmar_mdd12` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -17.44% | 3/10 | -9.83% | -1.12% | -0.84 | -1.87 | 17.68% |
| 2 | `validation_selector:validation_sharpe_mdd10` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -22.59% | 3/10 | -10.62% | -1.12% | -1.13 | -2.17 | 17.68% |
| 3 | `mdd30_risk_scaled:relaxed_aggressive_val_mdd30_cap1_75` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -23.57% | 3/8 | -21.84% | -0.28% | -0.68 | -0.99 | 27.66% |
| 4 | `validation_selector:validation_utility_mdd15` | `validation_selector` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -27.44% | 3/10 | -15.35% | -1.12% | -1.03 | -1.91 | 26.62% |
| 5 | `mdd30_risk_scaled:relaxed_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -30.80% | 3/8 | -20.61% | -0.24% | -0.96 | -1.40 | 37.85% |
| 6 | `mdd30_risk_scaled:relaxed_growth_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -28.90% | 2/8 | -20.42% | 7.38% | -0.82 | -1.58 | 26.48% |
| 7 | `mdd30_high_vol_gate:validation_breakout_or_defensive_scaled` | `mdd30_high_vol_gate` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -33.86% | 2/9 | -18.87% | -0.74% | -1.21 | -1.97 | 23.81% |
| 8 | `mdd30_high_vol_gate:breakout_barbell_blend` | `mdd30_high_vol_gate` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -35.04% | 2/9 | -17.51% | -0.91% | -1.85 | -2.77 | 20.14% |
| 9 | `mdd30_risk_scaled:profile_aggressive_val_mdd30_cap1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 14.86% | 1/3 | -0.72% | -0.72% | 1.76 | 351.45 | 22.10% |
| 10 | `mdd30_risk_scaled:profile_aggressive_x1_50` | `mdd30_risk_scaled` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 14.86% | 1/3 | -0.72% | -0.72% | 1.76 | 351.45 | 22.10% |
| 11 | `mdd30_barbell_blend:profile_aggressive_70_strict_balanced_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | 12.76% | 1/2 | -4.17% | 17.67% | 1.51 | 0.00 | 15.29% |
| 12 | `mdd30_barbell_blend:relaxed_aggressive_70_strict_growth_30_x1_50` | `mdd30_barbell_blend` | `False` | `post_oos_research_variant,requires_fresh_forward_shadow` | `False` | -14.50% | 1/4 | -16.00% | -0.24% | -0.64 | -1.19 | 21.09% |

## Best clean candidate monthly OOS detail: `profile_optuna:selected_optuna`

| Fold | Val | OOS | OOS MDD | Weights/source |
| --- | ---: | ---: | ---: | --- |
| `2025-09` | 41.99% | -1.97% | 5.39% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.330367793054126, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.28986429700571...` |
| `2025-10` | 51.96% | -8.91% | 14.22% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.13681516376455938, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.048460547396...` |
| `2025-11` | 25.71% | 1.67% | 6.93% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.09087450168324118, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.089174931113...` |
| `2025-12` | 66.04% | -6.68% | 9.51% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.21587240206595765, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.219627670309...` |
| `2026-01` | 64.41% | 7.87% | 5.71% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.21192617262938152, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.202044934071...` |
| `2026-02` | 111.17% | 20.19% | 19.20% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.38440122368972446, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.064932203127...` |
| `2026-03` | 61.79% | 2.18% | 4.71% | `hybrid_v3_6_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.7224745589886503, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.1339902683201...` |
| `2026-04` | 49.62% | -9.30% | 12.48% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.3095393953172158, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.1358736945727...` |
| `2026-05` | 41.22% | 8.91% | 18.03% | `hybrid_v3_5_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.28434575305878895, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.280957501738...` |
| `2026-06` | 83.53% | -0.73% | 0.91% | `hybrid_v3_6_optuna_three_profile_blend` / `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.2425859823467736, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.4052520400295...` |

### Best candidate extended metrics

- OOS comp: `10.05%`
- hit rate: `5/10`
- monthly Sharpe / Sortino approx: `0.50` / `1.16`
- 5% monthly VaR / 25% CVaR: `-9.13%` / `-8.30%`
- avg gain / avg loss: `8.16%` / `-5.52%`
- gain/loss ratio: `1.48`
- max loss streak: `2`
- mean/min validation: `59.74%` / `25.71%`

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
