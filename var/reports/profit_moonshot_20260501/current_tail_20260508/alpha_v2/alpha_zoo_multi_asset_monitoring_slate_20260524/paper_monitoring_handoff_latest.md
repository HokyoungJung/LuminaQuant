# Paper/testnet handoff — multi-asset monitoring slate

- Status: `paper_testnet_candidates_available`
- Candidate count: `136`
- Symbol count: `3`
- `ready_for_real=false`
- `real_money_execution=false`
- Monitor all paper symbols together; do not cherry-pick only one or two lanes.

## ETHUSDT

| Rank | Model | TF | Family | Train | Val | OOS report | RPT train/val/OOS |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 120 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_adx20_3p0x_0p1_236cf336` | 1h | debounced_momentum_hysteresis_efficiency_repair | 11.6397% | 8.3334% | 1.4279% | 15.52/47.08/16.41 |
| 121 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_adx20_2p0x_0p15_5f977da5` | 1h | debounced_momentum_hysteresis_efficiency_repair | 11.6397% | 8.3334% | 1.4279% | 15.52/47.08/16.41 |
| 122 | `debrepair_debounced_efficiency_repair_2h_ethusdt_long_short_lb6_e0p02_x0p005_hold24_cool0_trend_strength2_3p0x_0p15_c6ebae40` | 2h | debounced_momentum_hysteresis_efficiency_repair | 17.0562% | 7.8921% | 2.3678% | 18.22/38.97/23.92 |
| 123 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_none_3p0x_0p1_7865a8ae` | 1h | debounced_momentum_hysteresis_efficiency_repair | 10.7183% | 8.3334% | 1.6427% | 14.29/47.08/18.88 |
| 124 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_none_2p0x_0p15_3a8c496f` | 1h | debounced_momentum_hysteresis_efficiency_repair | 10.7183% | 8.3334% | 1.6427% | 14.29/47.08/18.88 |
| 125 | `debrepair_debounced_efficiency_repair_2h_ethusdt_long_short_lb6_e0p02_x0p005_hold24_cool0_trend_strength2_4p0x_0p1_d3b31f6d` | 2h | debounced_momentum_hysteresis_efficiency_repair | 15.4440% | 7.0605% | 2.1139% | 18.56/39.23/24.02 |
| 126 | `debrepair_debounced_efficiency_repair_2h_ethusdt_long_short_lb6_e0p02_x0p005_hold24_cool0_trend_strength2_2p0x_0p15_67f318f8` | 2h | debounced_momentum_hysteresis_efficiency_repair | 11.9812% | 5.3611% | 1.5992% | 19.20/39.71/24.23 |
| 127 | `debrepair_debounced_efficiency_repair_2h_ethusdt_long_short_lb6_e0p02_x0p005_hold24_cool0_trend_strength2_3p0x_0p1_b7768f90` | 2h | debounced_momentum_hysteresis_efficiency_repair | 11.9812% | 5.3611% | 1.5992% | 19.20/39.71/24.23 |
| 128 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5` | 2h | relative_residual_reclaim | 16.8301% | 4.7367% | 4.8120% | 18.29/29.60/37.02 |
| 129 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_5p0x_0p1_cf067261` | 2h | relative_residual_reclaim | 16.8301% | 4.7367% | 4.8120% | 18.29/29.60/37.02 |
| 130 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_2p0x_0p15_1a9aa250` | 2h | relative_residual_reclaim | 10.1047% | 2.8751% | 2.8746% | 18.31/29.95/36.85 |
| 131 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_3p0x_0p1_06295c43` | 2h | relative_residual_reclaim | 10.1047% | 2.8751% | 2.8746% | 18.31/29.95/36.85 |
| 132 | `debrepair_debounced_efficiency_repair_1h_ethusdt_short_only_lb6_e0p02_x0p005_hold18_cool12_adx20_4p0x_0p1_7117e59a` | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.0129% | 3.1316% | 1.0420% | 18.63/11.18/10.02 |
| 133 | `debrepair_debounced_efficiency_repair_1h_ethusdt_short_only_lb6_e0p02_x0p005_hold18_cool12_adx20_3p0x_0p1_6f810d7a` | 1h | debounced_momentum_hysteresis_efficiency_repair | 15.7263% | 2.4086% | 0.7882% | 18.59/11.47/10.10 |
| 134 | `debrepair_debounced_efficiency_repair_1h_ethusdt_short_only_lb6_e0p02_x0p005_hold18_cool12_adx20_2p0x_0p15_1bb7d246` | 1h | debounced_momentum_hysteresis_efficiency_repair | 15.7263% | 2.4086% | 0.7882% | 18.59/11.47/10.10 |

## SOLUSDT

| Rank | Model | TF | Family | Train | Val | OOS report | RPT train/val/OOS |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d` | 1h | debounced_momentum_hysteresis_efficiency_repair | 28.2198% | 16.9294% | 2.4704% | 24.69/59.72/21.11 |
| 2 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | 2h | relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | 30.96/60.72/31.39 |
| 3 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_5p0x_0p1_3beb48bb` | 2h | relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | 30.96/60.72/31.39 |
| 4 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_4p0x_0p125_f89f6f75` | 2h | relative_strength_chandelier_breakout | 35.3241% | 16.0919% | 4.2373% | 28.95/60.72/31.39 |
| 5 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_5p0x_0p1_e0f89ffc` | 2h | relative_strength_chandelier_breakout | 35.3241% | 16.0919% | 4.2373% | 28.95/60.72/31.39 |
| 6 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_4p0x_0p1_edcf6277` | 1h | debounced_momentum_hysteresis_efficiency_repair | 25.3406% | 15.0157% | 2.2043% | 24.94/59.59/21.19 |
| 7 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p15_76d4e65b` | 2h | relative_strength_chandelier_breakout | 33.7755% | 14.4603% | 3.8287% | 31.02/60.63/31.51 |
| 8 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p15_d6eac828` | 1h | debounced_momentum_hysteresis_efficiency_repair | 26.3570% | 15.1313% | 1.7643% | 23.43/55.12/17.82 |
| 9 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_3p0x_0p15_8757709e` | 2h | relative_strength_chandelier_breakout | 31.9037% | 14.4603% | 3.8287% | 29.06/60.63/31.51 |
| 10 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool12_none_3p0x_0p15_6ff58bc8` | 1h | debounced_momentum_hysteresis_efficiency_repair | 24.5154% | 13.6239% | 2.4704% | 23.69/51.31/21.11 |
| 11 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_4p0x_0p1_44b8ef4f` | 1h | debounced_momentum_hysteresis_efficiency_repair | 23.9228% | 13.4658% | 1.5867% | 23.92/55.19/18.03 |
| 12 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool6_none_3p0x_0p15_ef4179cd` | 1h | debounced_momentum_hysteresis_efficiency_repair | 23.3233% | 13.8100% | 2.4704% | 21.07/48.71/21.11 |
| 13 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool12_none_4p0x_0p1_ffd0f1b7` | 1h | debounced_momentum_hysteresis_efficiency_repair | 22.0625% | 12.1097% | 2.2043% | 23.98/51.31/21.19 |
| 14 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool6_none_4p0x_0p1_e5d4ce1a` | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.0497% | 12.2825% | 2.2043% | 21.39/48.74/21.19 |
| 15 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_2p0x_0p15_bfc43af2` | 1h | debounced_momentum_hysteresis_efficiency_repair | 19.3343% | 11.2087% | 1.6656% | 25.37/59.31/21.35 |
| 16 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p1_2d4f59e0` | 1h | debounced_momentum_hysteresis_efficiency_repair | 19.3343% | 11.2087% | 1.6656% | 25.37/59.31/21.35 |
| 17 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p5_rel0p005_trail2p0_hold12_4p0x_0p15_f9efb529` | 2h | relative_strength_chandelier_breakout | 37.0531% | 11.8881% | 3.4531% | 26.62/44.03/22.14 |
| 18 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p5_rel0p005_trail3p0_hold12_4p0x_0p15_d8f15883` | 2h | relative_strength_chandelier_breakout | 36.9682% | 11.8881% | 3.4531% | 26.56/44.03/22.14 |
| 19 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p025_x0p0_hold36_cool0_adx20_3p0x_0p15_c2ecca50` | 1h | debounced_momentum_hysteresis_efficiency_repair | 32.0204% | 11.2265% | 3.1624% | 33.88/48.92/29.28 |
| 20 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold18_cool0_adx20_3p0x_0p15_a9f9a6d7` | 1h | debounced_momentum_hysteresis_efficiency_repair | 25.9410% | 11.7165% | 1.8327% | 15.50/28.30/13.58 |

## TRXUSDT

| Rank | Model | TF | Family | Train | Val | OOS report | RPT train/val/OOS |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 135 | `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_adx15_3p0x_0p15_cca555d7` | 4h | volatility_adjusted_trend_persistence | 13.4914% | 2.9576% | 1.1825% | 15.07/12.64/10.51 |
| 136 | `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_none_3p0x_0p15_c555caef` | 4h | volatility_adjusted_trend_persistence | 13.2777% | 2.9576% | 1.1825% | 14.83/12.64/10.51 |
