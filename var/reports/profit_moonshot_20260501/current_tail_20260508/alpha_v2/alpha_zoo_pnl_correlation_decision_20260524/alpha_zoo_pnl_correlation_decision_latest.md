# Alpha Zoo PnL Correlation Decision

Generated: `2026-05-24T10:58:14.311758Z`

## Decision method

- Compute per-strategy PnL return streams from replayed paper/testnet candidates.
- Align timestamps across 30m/1h/2h/4h/6h strategies; missing bars are filled with 0 PnL before Pearson correlation.
- Rank only by train+validation monitoring score; locked-OOS is report-only after selection freeze.
- Greedy-select candidates if max abs train+validation corr <= 0.70 and max abs validation corr <= 0.75 versus already selected candidates.
- Keep `ready_for_real=false` and `real_money_execution=false` for every artifact and candidate.

## Summary

- Paper universe replayed: 136 candidates
- PnL capture count: 166 candidates; missing: 0
- Corr-diversified selected candidates: 11
- High-correlation clusters at |corr| >= 0.85: 14
- Decision: **do_not_adopt_all_paper_candidates; use corr-diversified paper/testnet-only subset**

## Selected corr-diversified paper/testnet-only slate

| Rank | Strategy | Train | Val | OOS report-only | Max train+val corr to prior | Max val corr to prior |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d` SOLUSDT 1h debounced_momentum_hysteresis_efficiency_repair | 28.2198% | 16.9294% | 2.4704% | 0.000 | 0.000 |
| 2 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` SOLUSDT 2h relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | 0.017 | 0.018 |
| 3 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p15_d6eac828` SOLUSDT 1h debounced_momentum_hysteresis_efficiency_repair | 26.3570% | 15.1313% | 1.7643% | 0.167 | 0.282 |
| 4 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold12_cool6_none_4p0x_0p1_b27a86b8` SOLUSDT 1h debounced_momentum_hysteresis_efficiency_repair | 24.2933% | 8.5915% | 1.2826% | 0.675 | 0.731 |
| 5 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_adx20_3p0x_0p1_236cf336` ETHUSDT 1h debounced_momentum_hysteresis_efficiency_repair | 11.6397% | 8.3334% | 1.4279% | 0.229 | 0.243 |
| 6 | `debrepair_debounced_efficiency_repair_2h_ethusdt_long_short_lb6_e0p02_x0p005_hold24_cool0_trend_strength2_3p0x_0p15_c6ebae40` ETHUSDT 2h debounced_momentum_hysteresis_efficiency_repair | 17.0562% | 7.8921% | 2.3678% | 0.327 | 0.350 |
| 7 | `a30fb_voladj_trend_6h_solusdt_lb6_z1p5_hold8_cool2_adx20_low_vol_q75_3p0x_0p1_e295f93a` SOLUSDT 6h volatility_adjusted_trend_persistence | 14.4540% | 5.5469% | 1.8230% | 0.027 | 0.016 |
| 8 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5` ETHUSDT 2h relative_residual_reclaim | 16.8301% | 4.7367% | 4.8120% | 0.159 | 0.227 |
| 9 | `debrepair_debounced_efficiency_repair_4h_solusdt_long_short_lb12_e0p02_x0p005_hold12_cool6_low_vol_q55_adx15_2p0x_0p15_de16b11e` SOLUSDT 4h debounced_momentum_hysteresis_efficiency_repair | 7.8735% | 5.2349% | 2.3500% | 0.067 | 0.044 |
| 10 | `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_adx15_3p0x_0p15_cca555d7` TRXUSDT 4h volatility_adjusted_trend_persistence | 13.4914% | 2.9576% | 1.1825% | 0.065 | 0.129 |
| 11 | `debrepair_debounced_efficiency_repair_1h_ethusdt_short_only_lb6_e0p02_x0p005_hold18_cool12_adx20_4p0x_0p1_7117e59a` ETHUSDT 1h debounced_momentum_hysteresis_efficiency_repair | 21.0129% | 3.1316% | 1.0420% | 0.549 | 0.594 |

## Portfolio comparison

| Portfolio | Count | Gross notional unscaled | Val equal-weight return | OOS equal-weight return | Val unscaled return | OOS unscaled return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_paper_candidates | 136 | 49.80x | 8.4824% | 2.5290% | -100.0000% | -35.0490% |
| corr_diversified_selected | 11 | 4.50x | 8.9654% | 2.3304% | 134.0623% | 27.0598% |

## Selected train+validation correlation matrix (excerpt)

| ID | Strategy |
| --- | --- |
| S1 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d` SOLUSDT 1h debounced_momentum_hysteresis_efficiency_repair |
| S2 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` SOLUSDT 2h relative_strength_chandelier_breakout |
| S3 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p15_d6eac828` SOLUSDT 1h debounced_momentum_hysteresis_efficiency_repair |
| S4 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold12_cool6_none_4p0x_0p1_b27a86b8` SOLUSDT 1h debounced_momentum_hysteresis_efficiency_repair |
| S5 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_adx20_3p0x_0p1_236cf336` ETHUSDT 1h debounced_momentum_hysteresis_efficiency_repair |
| S6 | `debrepair_debounced_efficiency_repair_2h_ethusdt_long_short_lb6_e0p02_x0p005_hold24_cool0_trend_strength2_3p0x_0p15_c6ebae40` ETHUSDT 2h debounced_momentum_hysteresis_efficiency_repair |
| S7 | `a30fb_voladj_trend_6h_solusdt_lb6_z1p5_hold8_cool2_adx20_low_vol_q75_3p0x_0p1_e295f93a` SOLUSDT 6h volatility_adjusted_trend_persistence |
| S8 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5` ETHUSDT 2h relative_residual_reclaim |
| S9 | `debrepair_debounced_efficiency_repair_4h_solusdt_long_short_lb12_e0p02_x0p005_hold12_cool6_low_vol_q55_adx15_2p0x_0p15_de16b11e` SOLUSDT 4h debounced_momentum_hysteresis_efficiency_repair |
| S10 | `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_adx15_3p0x_0p15_cca555d7` TRXUSDT 4h volatility_adjusted_trend_persistence |
| S11 | `debrepair_debounced_efficiency_repair_1h_ethusdt_short_only_lb6_e0p02_x0p005_hold18_cool12_adx20_4p0x_0p1_7117e59a` ETHUSDT 1h debounced_momentum_hysteresis_efficiency_repair |

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 1.000 | -0.017 | 0.167 | 0.675 | 0.112 | -0.013 | -0.002 | -0.047 | -0.005 | -0.012 | 0.521 |
| S2 | -0.017 | 1.000 | -0.009 | -0.018 | -0.024 | 0.327 | 0.027 | 0.159 | -0.008 | 0.010 | -0.028 |
| S3 | 0.167 | -0.009 | 1.000 | 0.150 | 0.229 | 0.026 | -0.005 | 0.048 | -0.016 | -0.011 | 0.163 |
| S4 | 0.675 | -0.018 | 0.150 | 1.000 | 0.129 | 0.011 | 0.005 | 0.005 | -0.011 | -0.011 | 0.549 |
| S5 | 0.112 | -0.024 | 0.229 | 0.129 | 1.000 | -0.012 | 0.018 | 0.018 | 0.021 | -0.000 | 0.167 |
| S6 | -0.013 | 0.327 | 0.026 | 0.011 | -0.012 | 1.000 | 0.018 | 0.104 | -0.016 | 0.023 | -0.003 |
| S7 | -0.002 | 0.027 | -0.005 | 0.005 | 0.018 | 0.018 | 1.000 | 0.008 | 0.067 | 0.019 | 0.005 |
| S8 | -0.047 | 0.159 | 0.048 | 0.005 | 0.018 | 0.104 | 0.008 | 1.000 | 0.013 | 0.026 | 0.003 |
| S9 | -0.005 | -0.008 | -0.016 | -0.011 | 0.021 | -0.016 | 0.067 | 0.013 | 1.000 | 0.065 | 0.003 |
| S10 | -0.012 | 0.010 | -0.011 | -0.011 | -0.000 | 0.023 | 0.019 | 0.026 | 0.065 | 1.000 | -0.016 |
| S11 | 0.521 | -0.028 | 0.163 | 0.549 | 0.167 | -0.003 | 0.005 | 0.003 | 0.003 | -0.016 | 1.000 |

## Top absolute train+validation correlation pairs

| Abs corr | Corr | Left | Right |
| ---: | ---: | --- | --- |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_5p0x_0p1_3beb48bb` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p15_76d4e65b` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p1_3ac49851` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_2p0x_0p15_3fa94945` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_5p0x_0p1_3beb48bb` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p15_76d4e65b` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_5p0x_0p1_3beb48bb` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p1_3ac49851` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_5p0x_0p1_3beb48bb` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_2p0x_0p15_3fa94945` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_4p0x_0p125_f89f6f75` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_5p0x_0p1_e0f89ffc` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_4p0x_0p125_f89f6f75` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_3p0x_0p1_5e18fb88` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_4p0x_0p125_f89f6f75` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_2p0x_0p15_56095a3e` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_5p0x_0p1_e0f89ffc` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_3p0x_0p1_5e18fb88` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_5p0x_0p1_e0f89ffc` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail2p0_hold18_2p0x_0p15_56095a3e` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p15_76d4e65b` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p1_3ac49851` |
| 1.0000 | 1.0000 | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_3p0x_0p15_76d4e65b` | `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_2p0x_0p15_3fa94945` |
| 1.0000 | 1.0000 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p15_d6eac828` | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_4p0x_0p1_44b8ef4f` |

## Guardrails

- ready_for_real=false
- real_money_execution=false
- locked-OOS used for selection=false
- locked-OOS role: gate/report-only after train+validation correlation ranking freeze
