# Alpha Zoo sample-guarded alpha discovery

Generated: `2026-05-21T13:22:01.478460Z`

This artifact is paper/testnet research only. `ready_for_real=false` and `real_money_execution=false`.
Locked-OOS is attached only after train+validation profile ranking freezes.

## Decision

- Status: `no_new_paper_promotion_shadow_shortlist`
- Paper candidate count: `0`
- Shadow/thin sample count: `1694`
- Reject/quarantine count: `891`
- Primary cost: `10.0` bps round-trip

## Top train+validation-ranked candidates

| Rank | Status | Model | Val return | Train return | OOS return | Trades T/V/O | Reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 8.8975% | 42.7972% | -0.6059% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |
| 2 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 8.8975% | 42.7972% | -0.6059% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |
| 3 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 7.5724% | 45.4812% | -0.2253% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 4 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 7.5724% | 45.4812% | -0.2253% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 5 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 27.3844% | 4.5832% | -4.2507% | 476/103/56 | train_validation_return_ratio_0.1674_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 6 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 27.3844% | 4.5832% | -4.2507% | 476/103/56 | train_validation_return_ratio_0.1674_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 7 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 8.4744% | 40.5768% | -0.5767% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |
| 8 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 8.4744% | 40.5768% | -0.5767% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |
| 9 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 7.2127% | 43.0156% | -0.2146% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 10 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 7.2127% | 43.0156% | -0.2146% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 11 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 26.0085% | 4.7950% | -4.0374% | 476/103/56 | train_validation_return_ratio_0.1844_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 12 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 26.0085% | 4.7950% | -4.0374% | 476/103/56 | train_validation_return_ratio_0.1844_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 13 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p15alloc` | 7.6279% | 36.1869% | -0.5183% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |
| 14 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_6p0x_0p15alloc` | 7.6279% | 36.1869% | -0.5183% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |
| 15 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_2p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p175alloc` | 7.4162% | 35.1002% | -0.5038% | 63/14/5 | train_trade_event_count_63_below_80, validation_trade_event_count_14_below_30, locked_oos_trade_event_count_5_below_20 |

## Baseline lanes preserved

| Role | Model | Leverage | Allocation |
| --- | --- | ---: | ---: |
| active | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` | 7.0 | 0.200 |
| balanced | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` | 6.0 | 0.175 |
| validation_return_leader | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` | 5.0 | 0.200 |
| validation_efficiency_reference | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` | 4.0 | 0.175 |
