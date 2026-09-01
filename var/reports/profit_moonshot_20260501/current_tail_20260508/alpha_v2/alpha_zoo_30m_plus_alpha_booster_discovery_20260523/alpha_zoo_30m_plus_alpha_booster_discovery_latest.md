# Alpha Zoo 30m+ booster discovery

Generated: `2026-05-23T10:44:48.240241Z`

Research/paper-testnet only. Locked-OOS is gate/report-only after train+validation ranking freeze.

- Candidates evaluated: `63450`
- Paper candidate gate pass: `46`
- Preferred booster target pass: `0`
- `ready_for_real=false`
- `real_money_execution=false`
- Runner peak RSS MiB: `2248.61328125`

## Top train+validation-ranked rows

| Rank | Symbol | TF | Family | Train | Val | OOS | RPT train/val/OOS | Decision | Reasons |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | SOLUSDT | 1h | relative_strength_chandelier_breakout | 25.1681% | 25.5505% | -5.8294% | 17.48/92.57/-38.86 | no_promotion_shadow_or_reject | train_return_below_validation_return; train_validation_return_ratio_0.9850_below_1.00; locked_oos_return_not_positive |
| 2 | ETHUSDT | 2h | relative_strength_chandelier_breakout | 44.8764% | 22.7217% | -7.7371% | 45.61/118.34/-64.48 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-64.476_not_above_10.000 |
| 3 | ETHUSDT | 2h | relative_strength_chandelier_breakout | 39.2912% | 21.8270% | -7.8276% | 39.93/113.68/-65.23 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-65.230_not_above_10.000 |
| 4 | ETHUSDT | 2h | relative_strength_chandelier_breakout | 36.0674% | 22.2335% | -13.6144% | 39.55/123.52/-126.06 | no_promotion_shadow_or_reject | locked_oos_trade_event_count_18_below_20; locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-126.060_not_above_10.000 |
| 5 | SOLUSDT | 1h | relative_strength_chandelier_breakout | 21.1760% | 20.9945% | -4.8691% | 17.65/91.28/-38.95 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-38.953_not_above_10.000 |
| 6 | SOLUSDT | 1h | relative_strength_chandelier_breakout | 21.1760% | 20.9945% | -4.8691% | 17.65/91.28/-38.95 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-38.953_not_above_10.000 |
| 7 | SOLUSDT | 1h | relative_strength_chandelier_breakout | 21.1271% | 24.3062% | -8.3188% | 14.31/84.40/-44.72 | no_promotion_shadow_or_reject | train_return_below_validation_return; train_validation_return_ratio_0.8692_below_1.00; locked_oos_return_not_positive |
| 8 | SOLUSDT | 2h | relative_strength_chandelier_breakout | 23.3246% | 22.8363% | -2.5992% | 16.61/77.67/-12.74 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-12.741_not_above_10.000 |
| 9 | SOLUSDT | 2h | relative_strength_chandelier_breakout | 23.2482% | 22.8363% | -2.5992% | 16.56/77.67/-12.74 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-12.741_not_above_10.000 |
| 10 | SOLUSDT | 2h | relative_strength_chandelier_breakout | 29.5918% | 20.9853% | 3.0255% | 23.26/89.68/33.62 | no_promotion_shadow_or_reject | locked_oos_trade_event_count_15_below_20 |
| 11 | ETHUSDT | 2h | relative_strength_chandelier_breakout | 36.6528% | 18.6979% | -6.4807% | 44.70/116.86/-64.81 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-64.807_not_above_10.000 |
| 12 | ETHUSDT | 2h | relative_strength_chandelier_breakout | 36.6528% | 18.6979% | -6.4807% | 44.70/116.86/-64.81 | no_promotion_shadow_or_reject | locked_oos_return_not_positive; locked_oos_return_per_turnover_proxy_bps_-64.807_not_above_10.000 |

## Paper/testnet candidates

| Rank | Model | Symbol | TF | Family | Train | Val | OOS | Booster target |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 91 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | false |
| 92 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_5p0x_0p1_3beb48bb` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | false |
| 95 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_4p0x_0p125_f89f6f75` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 35.3241% | 16.0919% | 4.2373% | false |
| 96 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_5p0x_0p1_e0f89ffc` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 35.3241% | 16.0919% | 4.2373% | false |
| 138 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p15_76d4e65b` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 33.7755% | 14.4603% | 3.8287% | false |
| 140 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_3p0x_0p15_8757709e` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 31.9037% | 14.4603% | 3.8287% | false |
| 351 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p5_rel0p005_trail2p0_hold12_4p0x_0p15_f9efb529` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 37.0531% | 11.8881% | 3.4531% | false |
| 352 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p5_rel0p005_trail3p0_hold12_4p0x_0p15_d8f15883` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 36.9682% | 11.8881% | 3.4531% | false |
| 417 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p1_3ac49851` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 22.5430% | 9.5896% | 2.5823% | false |
| 418 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_2p0x_0p15_3fa94945` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 22.5430% | 9.5896% | 2.5823% | false |
| 429 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_3p0x_0p1_5e18fb88` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 21.3984% | 9.5896% | 2.5823% | false |
| 430 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_2p0x_0p15_56095a3e` | SOLUSDT | 2h | relative_strength_chandelier_breakout | 21.3984% | 9.5896% | 2.5823% | false |
