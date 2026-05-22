# Alpha Zoo sample-guarded alpha discovery

Generated: `2026-05-22T10:30:13.295045Z`

This artifact is paper/testnet research only. `ready_for_real=false` and `real_money_execution=false`.
Locked-OOS is attached only after train+validation profile ranking freezes.

## Decision

- Status: `no_new_paper_promotion_shadow_shortlist`
- Paper candidate count: `0`
- Shadow/thin sample count: `252`
- Reject/quarantine count: `724`
- Primary cost: `10.0` bps round-trip

## Top train+validation-ranked candidates

| Rank | Status | Model | Val return | Train return | OOS return | Trades T/V/O | Reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 7.5724% | 45.4812% | -0.2253% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 2 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 7.5724% | 45.4812% | -0.2253% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 3 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 27.3844% | 4.5832% | -4.2507% | 476/103/56 | train_validation_return_ratio_0.1674_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 4 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 27.3844% | 4.5832% | -4.2507% | 476/103/56 | train_validation_return_ratio_0.1674_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 5 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 7.2127% | 43.0156% | -0.2146% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 6 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 7.2127% | 43.0156% | -0.2146% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 7 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 26.0085% | 4.7950% | -4.0374% | 476/103/56 | train_validation_return_ratio_0.1844_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 8 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 26.0085% | 4.7950% | -4.0374% | 476/103/56 | train_validation_return_ratio_0.1844_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 9 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p15alloc` | 6.4929% | 38.1786% | -0.1931% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 10 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_6p0x_0p15alloc` | 6.4929% | 38.1786% | -0.1931% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 11 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p175alloc` | 6.3129% | 36.9888% | -0.1878% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 12 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_5p0x_0p175alloc` | 6.3129% | 36.9888% | -0.1878% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |
| 13 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p15alloc` | 23.2769% | 5.0897% | -3.6139% | 476/103/56 | train_validation_return_ratio_0.2187_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 14 | reject_or_quarantine | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p15alloc` | 23.2769% | 5.0897% | -3.6139% | 476/103/56 | train_validation_return_ratio_0.2187_below_0.50, locked_oos_return_not_positive, primary_10bps_promotion_gate_failed |
| 15 | shadow_only_thin_sample | `fresh_tv10_filter_abs_score_ge_3_alpha_zoo_conservative_exit_carry_forward_old_split_selected_7p0x_0p15alloc` | 7.5724% | 24.6980% | -0.2253% | 42/8/1 | train_trade_event_count_42_below_80, validation_trade_event_count_8_below_30, locked_oos_trade_event_count_1_below_20 |

## Baseline lanes preserved

| Role | Model | Leverage | Allocation |
| --- | --- | ---: | ---: |
| active | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` | 7.0 | 0.200 |
| balanced | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` | 6.0 | 0.175 |
| validation_return_leader | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` | 5.0 | 0.200 |
| validation_efficiency_reference | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` | 4.0 | 0.175 |
