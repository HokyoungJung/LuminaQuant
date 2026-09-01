# Alpha Zoo 10bps four-lane paper-forward and shadow discovery

Generated: `2026-05-20T12:58:02.265127Z`

Real-money execution remains disabled. Locked-OOS is gate/report-only.

## Four paper/testnet lanes

| Lane | Model | Lev/Alloc | Val return | Val MDD | Train return | Locked-OOS | Notional/equity | Paper | Real |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| active | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` | 7.0x/0.200 | 0.4724% | 14.9117% | 45.6916% | 1.8382% | 140.0% | `True` | `False` |
| balanced | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` | 6.0x/0.175 | 0.5942% | 11.3653% | 36.0268% | 1.5464% | 105.0% | `True` | `False` |
| validation_return_leader | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` | 5.0x/0.200 | 0.5986% | 10.8490% | 34.5152% | 1.4956% | 100.0% | `True` | `False` |
| validation_efficiency_reference | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` | 4.0x/0.175 | 0.5561% | 7.6999% | 24.8845% | 1.1432% | 70.0% | `True` | `False` |

## Shadow strategy findings

Paper-forward should compare four quality_single_pair lanes; frozen 10bps data does not justify promoting conservative_exit or side/family filters because their validation edge fails locked-OOS gates.

### Top conservative-exit rescue hypotheses

| Rank | Model | Val return | Train return | Locked-OOS | Status |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p175alloc` | 27.3844% | 4.5832% | -4.2507% | shadow_only_locked_oos_negative |
| 2 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p175alloc` | 27.3844% | 4.5832% | -4.2507% | shadow_only_locked_oos_negative |
| 3 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p2alloc` | 26.0085% | 4.7950% | -4.0374% | shadow_only_locked_oos_negative |
| 4 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_5p0x_0p2alloc` | 26.0085% | 4.7950% | -4.0374% | shadow_only_locked_oos_negative |
| 5 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p15alloc` | 23.2769% | 5.0897% | -3.6139% | shadow_only_locked_oos_negative |
| 6 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_6p0x_0p15alloc` | 23.2769% | 5.0897% | -3.6139% | shadow_only_locked_oos_negative |
| 7 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p175alloc` | 22.5983% | 5.1364% | -3.5087% | shadow_only_locked_oos_negative |
| 8 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_5p0x_0p175alloc` | 22.5983% | 5.1364% | -3.5087% | shadow_only_locked_oos_negative |
| 9 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_4p0x_0p2alloc` | 20.5728% | 5.2114% | -3.1945% | shadow_only_locked_oos_negative |
| 10 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_4p0x_0p2alloc` | 20.5728% | 5.2114% | -3.1945% | shadow_only_locked_oos_negative |
| 11 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_6p0x_0p125alloc` | 19.2312% | 5.2069% | -2.9864% | shadow_only_locked_oos_negative |
| 12 | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_conservative_exit_carry_forward_old_split_selected_5p0x_0p15alloc` | 19.2312% | 5.2069% | -2.9864% | shadow_only_locked_oos_negative |
