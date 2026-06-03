# Dynamic-aware 69-asset hybrid final recommendation

- generated: `2026-06-02T07:17:51.173791+00:00`
- source walk-forward: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_dynamic_aware_hybrid_20260602/dynamic_aware_walkforward_latest.json`
- latest data: `2026-06-01T06:30:00`
- folds/timeframes/slippage: `10` / `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d` / `10.0` bps
- refit protocol: monthly day-1, expanding train, previous 2 calendar months validation, next month locked OOS; OOS not used for selection.

## Decision

- **Real-money live: blocked**. Artifact promotability remains false.
- **Primary paper-shadow candidate:** `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` because it is the best OOS comp under max OOS MDD <= 15% in this run and improves over robust-default reference, but it still does not beat the high-comp dynamic challenger.
- **Stability shadow:** `dynamic_aware_hybrid:hybrid_v3_6` because hit rate is 6/10 and max OOS MDD is low, but comp is lower.
- **High-comp monitor only:** `dynamic_conviction_switch:t0.85/t0.90/t0.95_risk_capped_fallback` because comp is ~53%, but max OOS MDD is ~18.8% and hard-stop promotion fails.

## Selected comparison

| Candidate | Family | OOS comp | Hit | Min OOS | Latest OOS | Max OOS MDD | Sharpe | Sortino | Hard-stop flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | -2.65% | -0.71% | 18.80% | 2.07 | 15.31 | `False` |
| `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | -2.65% | -0.71% | 18.80% | 2.07 | 15.31 | `False` |
| `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `dynamic_conviction_switch` | 53.38% | 5/10 | -2.65% | -0.71% | 18.80% | 2.07 | 15.31 | `False` |
| `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | `dynamic_aware_hybrid` | 32.94% | 5/10 | -3.19% | -0.78% | 11.22% | 1.45 | 10.16 | `True` |
| `dynamic_aware_hybrid:hybrid_v3_5` | `dynamic_aware_hybrid` | 25.12% | 6/10 | -5.62% | -0.71% | 13.84% | 1.22 | 3.29 | `False` |
| `dynamic_aware_hybrid:hybrid_v3_6` | `dynamic_aware_hybrid` | 22.50% | 6/10 | -3.81% | -0.45% | 11.12% | 1.31 | 4.89 | `False` |
| `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | `dynamic_aware_hybrid` | 21.87% | 5/10 | -5.63% | -0.70% | 15.39% | 1.06 | 3.46 | `False` |
| `cross_candidate_hybrid:hybrid_v3_5` | `cross_candidate_hybrid` | 27.01% | 5/10 | -4.31% | -0.84% | 13.72% | 1.24 | 6.33 | `True` |
| `cross_candidate_hybrid:hybrid_v3_6_train_validation_fit` | `cross_candidate_hybrid` | 19.72% | 4/10 | -5.84% | -0.77% | 12.73% | 0.91 | 3.33 | `False` |
| `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_riskcap` | `hybrid_oracle_bridge` | 22.76% | 4/10 | -5.42% | -0.71% | 18.80% | 1.00 | 3.95 | `False` |
| `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1` | `hybrid_oracle_bridge` | 11.94% | 4/10 | -8.89% | -0.73% | 19.20% | 0.57 | 1.42 | `False` |
| `hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_hedge` | `hybrid_oracle_bridge` | 9.46% | 4/10 | -9.78% | -0.73% | 19.20% | 0.48 | 1.28 | `False` |

## Fold winners

| Fold | Best candidate | OOS | OOS MDD | Candidates | Dynamic-aware candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `individual_robust:hybrid_v3_5` | 3.10% | 5.79% | 54 | 4 |
| `2025-10` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | 6.41% | 7.04% | 54 | 4 |
| `2025-11` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 12.20% | 10.08% | 54 | 4 |
| `2025-12` | `dynamic_aware_hybrid:hybrid_v3_5` | 0.20% | 4.64% | 54 | 4 |
| `2026-01` | `relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna` | 16.03% | 9.34% | 53 | 4 |
| `2026-02` | `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit` | 20.90% | 5.88% | 54 | 4 |
| `2026-03` | `cross_candidate_hybrid:hybrid_v3_5` | 2.63% | 3.14% | 54 | 4 |
| `2026-04` | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | -0.40% | 1.65% | 54 | 4 |
| `2026-05` | `profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna` | 16.58% | 9.96% | 54 | 4 |
| `2026-06` | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 0.00% | 0.00% | 53 | 4 |

## Dynamic-aware composition summary


### `dynamic_aware_hybrid:hybrid_v3_5`
- dynamic inputs: `dynamic_conviction_switch:t0.85_risk_capped_fallback`, `dynamic_conviction_switch:t0.90_risk_capped_fallback`, `dynamic_conviction_switch:t0.95_risk_capped_fallback`
| Avg-weight top input | Avg weight | Max weight | Months >1% |
| --- | ---: | ---: | ---: |
| `profile_optuna:selected_train_validation_legal` | 16.70% | 83.59% | 5 |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | 16.18% | 69.56% | 6 |
| `cross_candidate_hybrid:hybrid_v3_5` | 8.34% | 79.46% | 2 |
| `profile_optuna:hybrid_v3_5` | 7.18% | 70.14% | 2 |
| `cross_candidate_hybrid:hybrid_v3_6` | 3.82% | 37.68% | 1 |
| `meta_portfolio:validation_stability_top8_equal` | 2.71% | 26.24% | 1 |
| `cross_candidate_hybrid:hybrid_v3_5_train_validation_fit` | 1.90% | 14.05% | 3 |
| `dynamic_conviction_switch:t0.90_risk_capped_fallback` | 1.30% | 2.98% | 4 |

### `dynamic_aware_hybrid:hybrid_v3_6`
- dynamic inputs: `dynamic_conviction_switch:t0.85_risk_capped_fallback`, `dynamic_conviction_switch:t0.90_risk_capped_fallback`, `dynamic_conviction_switch:t0.95_risk_capped_fallback`
| Avg-weight top input | Avg weight | Max weight | Months >1% |
| --- | ---: | ---: | ---: |
| `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 17.46% | 90.56% | 5 |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | 14.17% | 59.77% | 8 |
| `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 8.56% | 75.97% | 3 |
| `individual_robust:selected_train_validation_legal` | 7.78% | 58.30% | 6 |
| `profile_optuna:selected_train_validation_legal` | 5.68% | 38.89% | 6 |
| `individual_robust:hybrid_v3_6` | 4.35% | 35.53% | 4 |
| `dynamic_conviction_switch:t0.90_risk_capped_fallback` | 2.37% | 5.95% | 6 |
| `dynamic_conviction_switch:t0.95_risk_capped_fallback` | 2.37% | 5.95% | 6 |

### `dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit`
- dynamic inputs: `dynamic_conviction_switch:t0.85_risk_capped_fallback`, `dynamic_conviction_switch:t0.90_risk_capped_fallback`, `dynamic_conviction_switch:t0.95_risk_capped_fallback`
| Avg-weight top input | Avg weight | Max weight | Months >1% |
| --- | ---: | ---: | ---: |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | 27.76% | 72.61% | 10 |
| `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 15.54% | 77.42% | 4 |
| `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 5.68% | 27.46% | 5 |
| `profile_optuna:selected_train_validation_legal` | 5.47% | 36.43% | 7 |
| `dynamic_conviction_switch:t0.90_risk_capped_fallback` | 2.26% | 4.53% | 7 |
| `dynamic_conviction_switch:t0.95_risk_capped_fallback` | 2.26% | 4.53% | 7 |
| `individual_robust:selected_train_validation_legal` | 2.19% | 4.46% | 8 |
| `profile_optuna:selected_optuna` | 2.19% | 4.29% | 7 |

## Audits

- `metric_reconciliation`: `{'metrics_reconciled': True}`
- `dynamic_self_feed_audit`: `{'no_same_month_dynamic_self_feeding': True}`
- `bridge_protocol_audit`: `{'current_fold_oos_used_for_bridge_weighting': False}`
- `online_weight_audit`: `{'fully_lagged_online_weights': True}`
- `protocol_freeze_report`: `{'frozen_before_first_oos_evaluation': True, 'oos_used_for_protocol_expansion': False}`
- `promotability`: `{'promotable': False, 'promotion_hard_stop_pass': False}`

## Guardrail

This is research/paper-testnet evidence only. The last OOS fold is partial (`2026-06-01T00:00:00` to `2026-06-01T06:30:00`), so forward shadow/fill/slippage telemetry is required before any real allocation.
