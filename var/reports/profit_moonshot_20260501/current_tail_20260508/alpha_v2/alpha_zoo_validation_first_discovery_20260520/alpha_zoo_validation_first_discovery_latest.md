# Alpha Zoo 10bps validation-first discovery

Generated: `2026-05-20T11:54:25.904605Z`

Locked-OOS is gate/report-only after the validation-first ranking is frozen.
Real-money execution remains disabled.

## Selected paper/testnet candidates

| Role | Model | Val return | Val MDD | Train return | Locked-OOS return | Paper | Real |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| validation_return_leader | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` | 0.5986% | 10.8490% | 34.5152% | 1.4956% | `True` | `False` |
| validation_efficiency_reference | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` | 0.5561% | 7.6999% | 24.8845% | 1.1432% | `True` | `False` |

## High-validation quarantine

These rows were found by validation ranking but are not paper candidates because locked-OOS/promotion gates fail.

| Rank | Model | Val return | Locked-OOS return | Gate reasons |
| ---: | --- | ---: | ---: | --- |
| 1 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 27.3844% | -4.2507% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 2 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 27.3844% | -4.2507% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 3 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 26.0085% | -4.0374% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 4 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 26.0085% | -4.0374% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 5 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p15alloc` | 23.2769% | -3.6139% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 6 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p15alloc` | 23.2769% | -3.6139% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 7 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p175alloc` | 22.5983% | -3.5087% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 8 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_5p0x_0p175alloc` | 22.5983% | -3.5087% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 9 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_4p0x_0p2alloc` | 20.5728% | -3.1945% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
| 10 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_4p0x_0p2alloc` | 20.5728% | -3.1945% | locked_oos_calmar_non_positive;locked_oos_return_non_positive;locked_oos_sharpe_non_positive;locked_oos_smart_sortino_non_positive;locked_oos_sortino_non_positive;train_calmar_not_above_validation;train_sharpe_not_above_validation;train_smart_sortino_not_above_validation;train_sortino_not_above_validation;train_total_return_not_above_validation |
