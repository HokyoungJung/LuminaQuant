# Alpha Zoo 69-Asset Best Strategy Final Recommendation v2

- source: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_best_strategy_factory_20260601/best_strategy_factory_latest.json`
- timeframes: `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`
- slippage: `10.0` bps
- folds: `2025-09` → `2026-06`
- clean fold protocol: train+2M validation selection; locked OOS report-only inside each fold

## Decision

- **High-comp paper/shadow challenger:** `dynamic_conviction_switch:t0.90_risk_capped_fallback`
- **Robust full-run default:** `cross_candidate_hybrid:hybrid_v3_5`
- **Real-money:** blocked; needs forward paper/fill/slippage telemetry.

## Comparison

| Candidate | Family | OOS comp | Hit | Min OOS | Max OOS MDD | Sharpe | Sortino | Min Val |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | -2.65% | 18.80% | 2.07 | 15.31 | 0.01% |
| `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | 27.01% | 5/10 | -4.31% | 13.72% | 1.24 | 6.33 | 24.56% |
| `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | `cross_candidate_hybrid` | 25.74% | 5/10 | -5.39% | 13.81% | 1.16 | 4.23 | 27.32% |
| `cross_candidate_hybrid:hybrid_v3_6` | `cross_candidate_hybrid` | 22.07% | 5/10 | -4.01% | 13.85% | 1.07 | 6.27 | 31.35% |
| `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | 19.72% | 4/10 | -5.84% | 12.73% | 0.91 | 3.33 | 33.22% |
| `profile_optuna:selected_train_validation_legal` | `profile_optuna` | 18.32% | 5/10 | -8.46% | 18.80% | 0.80 | 2.07 | 29.14% |
| `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | `strict_efficiency` | 6.97% | 4/10 | -3.58% | 7.32% | 0.66 | 1.84 | 0.01% |

## Dynamic switch monthly choices

| Fold | Selected | Val | OOS | OOS MDD |
| --- | --- | ---: | ---: | ---: |
| `2025-09` | `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 0.01% | 0.14% | 0.07% |
| `2025-10` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.73% | -0.72% | 1.39% |
| `2025-11` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 26.81% | 12.20% | 10.08% |
| `2025-12` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.92% | -0.13% | 1.63% |
| `2026-01` | `cross_candidate_hybrid:hybrid_v3_5` | 80.04% | 9.62% | 3.92% |
| `2026-02` | `profile_optuna:selected_optuna` | 108.23% | 19.24% | 18.80% |
| `2026-03` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.82% | -2.65% | 2.80% |
| `2026-04` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 5.79% | -2.02% | 4.57% |
| `2026-05` | `profile_optuna:selected_optuna` | 43.70% | 11.24% | 16.08% |
| `2026-06` | `profile_optuna:selected_optuna` | 86.60% | -0.71% | 1.04% |

## Caveats

- Dynamic switch uses train+validation only inside each fold, not current locked OOS.
- However the risk-capped rule was introduced after this research iteration, so it is a forward-shadow challenger rather than real-money approval.
- No candidate is positive every month; dynamic improves comp by payoff asymmetry and defensive fallback, not by eliminating losses.
