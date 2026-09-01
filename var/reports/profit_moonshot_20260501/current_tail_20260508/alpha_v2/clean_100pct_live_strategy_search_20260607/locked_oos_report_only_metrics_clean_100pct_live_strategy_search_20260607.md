# Locked OOS report-only metrics aggregation

- Generated UTC: `2026-06-07T07:11:42Z`
- Ultragoal: `G005-locked-oos-report-only-and-metrics-a`
- Policy: locked OOS is attached only after train/validation choices and is report-only.

| Candidate | Final label | Ann approx | OOS comp | Max OOS MDD | Gate result |
| --- | --- | ---: | ---: | ---: | --- |
| `clean_input_meta_selector` | `shadow_freeze_only` | 110.46% | 85.91% | 19.29% | fail_current_promotion_due_post_oos_selector_grid_ranking |
| `relaxed_efficiency_hybrid_v3_5_69_asset_historical_incumbent` | `paper_control` | 160.90% | 122.36% | 16.66% | not_current_promotion_evidence_under_no_nested_oos_mining |
| `strict_no_leak_best_single_10bps` | `paper_control` | n/a | 54.56% | 30.63% | fails_real_money_due_drawdown_tail_and_missing_15bps_paper_telemetry |
| `dynamic_conviction_switch_85_symbol_baseline` | `paper_control` | 42.57% | 34.39% | 27.69% | below_100pct_and_cost_telemetry_missing |
| `clean_new_alpha_discovery_full` | `rejected` | 3.01% | 2.51% | 8.77% | clean_promotion_eligible_false_and_low_return |
| `clean_new_alpha_discovery_feature_bounded` | `rejected` | -0.57% | -0.24% | 8.32% | clean_promotion_eligible_false_and_negative_return |

## Summary

- Clean 100%+ report-label candidate found: **False**
- Historical 100%+ shadow label exists: **True**
- Real-money candidate found: **False**
- Small-sleeve candidate found: **False**
