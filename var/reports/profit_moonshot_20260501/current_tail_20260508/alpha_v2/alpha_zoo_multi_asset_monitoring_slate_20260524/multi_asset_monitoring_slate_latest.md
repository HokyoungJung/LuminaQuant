# Alpha Zoo multi-asset monitoring slate

Generated: `2026-05-24T08:12:32.023761Z`

Combined paper/testnet-only monitoring view across all recent Alpha Zoo discovery
artifacts. Locked-OOS remains gate/report-only; monitoring priority uses train+
validation evidence only. Real-money execution is disabled.

## Summary

- Candidate rows normalized: `2488`
- Symbols covered by matrix: `14`
- Symbols with paper candidates: `3`
- Paper/testnet monitor candidates: `136`
- Status counts: `{'coverage_blocked_shadow': 249, 'paper_testnet_monitor': 136, 'shadow_watchlist_no_promotion': 2103}`
- `ready_for_real=false`
- `real_money_execution=false`
- Runner peak RSS MiB: `117.141`

## Asset monitoring matrix

| Symbol | Group | Paper | Shadow | Coverage blocked | Best paper | Best shadow | Action |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| ADAUSDT | crypto_high_beta_alt | 0 | 0 | 0 | `-` | `-` | source_coverage_only_no_candidate_rows |
| AVAXUSDT | crypto_high_beta_alt | 0 | 0 | 16 | `-` | `a30fb_asset_diverse_rank_chandelier_2h_avaxusdt_lb12_top1_mom0p005_breadth0p4_adx18p0_trail2p0_hold12_4p0x_0p125_56a81e2b` | shadow_monitor_or_extend_data_before_paper_review |
| BNBUSDT | crypto_exchange_beta | 0 | 147 | 0 | `-` | `a30fb_booster_mh_consensus_2h_bnbusdt_s6_l24_thr0p005_adx18p0_hold18_cool2_4p0x_0p15_64149974` | shadow_monitor_or_extend_data_before_paper_review |
| BTCUSDT | crypto_major | 0 | 13 | 0 | `-` | `a30fb_voladj_trend_4h_btcusdt_lb12_z1p0_hold4_cool0_low_vol_q70_3p0x_0p15_6ee8568c` | shadow_monitor_or_extend_data_before_paper_review |
| DOGEUSDT | crypto_high_beta_alt | 0 | 0 | 128 | `-` | `a30fb_asset_diverse_rank_chandelier_1h_dogeusdt_lb12_top1_mom0p015_breadth0p4_adx12p0_trail2p0_hold18_4p0x_0p125_20ad5aa3` | shadow_monitor_or_extend_data_before_paper_review |
| ETHUSDT | crypto_major | 15 | 761 | 0 | `debrepair_debounced_efficiency_repair_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold48_cool6_adx20_3p0x_0p1_236cf336` | `debrepair_debounced_efficiency_repair_4h_ethusdt_long_short_lb12_e0p02_x0p005_hold48_cool6_low_vol_q65_3p0x_0p15_12b53e05` | monitor_all_paper_testnet_candidates_for_symbol |
| SOLUSDT | crypto_high_beta_alt | 119 | 1138 | 0 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d` | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p02_x0p005_hold48_cool0_none_3p0x_0p15_852b9c04` | monitor_all_paper_testnet_candidates_for_symbol |
| TONUSDT | crypto_high_beta_alt | 0 | 0 | 0 | `-` | `-` | source_coverage_only_no_candidate_rows |
| TRXUSDT | crypto_payment_alt | 2 | 44 | 0 | `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_adx15_3p0x_0p15_cca555d7` | `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_low_vol_q70_3p0x_0p15_144859db` | monitor_all_paper_testnet_candidates_for_symbol |
| XAGUSDT | precious_metal_proxy | 0 | 0 | 0 | `-` | `-` | source_coverage_only_no_candidate_rows |
| XAUUSDT | precious_metal_proxy | 0 | 0 | 1 | `-` | `a30fb_asset_diverse_rank_chandelier_6h_xauusdt_lb12_top1_mom0p005_breadth0p4_adx12p0_trail2p0_hold8_4p0x_0p125_0d025903` | shadow_monitor_or_extend_data_before_paper_review |
| XPDUSDT | precious_metal_proxy | 0 | 0 | 0 | `-` | `-` | source_coverage_only_no_candidate_rows |
| XPTUSDT | precious_metal_proxy | 0 | 0 | 0 | `-` | `-` | source_coverage_only_no_candidate_rows |
| XRPUSDT | crypto_payment_alt | 0 | 0 | 104 | `-` | `a30fb_asset_diverse_rank_chandelier_1h_xrpusdt_lb12_top1_mom0p005_breadth0p4_adx12p0_trail2p0_hold18_4p0x_0p125_b92ab84a` | shadow_monitor_or_extend_data_before_paper_review |

## Paper/testnet candidates by monitoring rank

| Rank | Symbol | Group | TF | Family | Train | Val | OOS report | RPT train/val/OOS |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 28.2198% | 16.9294% | 2.4704% | 24.69/59.72/21.11 |
| 2 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | 30.96/60.72/31.39 |
| 3 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 37.4602% | 16.0919% | 4.2373% | 30.96/60.72/31.39 |
| 4 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 35.3241% | 16.0919% | 4.2373% | 28.95/60.72/31.39 |
| 5 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 35.3241% | 16.0919% | 4.2373% | 28.95/60.72/31.39 |
| 6 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 25.3406% | 15.0157% | 2.2043% | 24.94/59.59/21.19 |
| 7 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 33.7755% | 14.4603% | 3.8287% | 31.02/60.63/31.51 |
| 8 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 26.3570% | 15.1313% | 1.7643% | 23.43/55.12/17.82 |
| 9 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 31.9037% | 14.4603% | 3.8287% | 29.06/60.63/31.51 |
| 10 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 24.5154% | 13.6239% | 2.4704% | 23.69/51.31/21.11 |
| 11 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 23.9228% | 13.4658% | 1.5867% | 23.92/55.19/18.03 |
| 12 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 23.3233% | 13.8100% | 2.4704% | 21.07/48.71/21.11 |
| 13 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 22.0625% | 12.1097% | 2.2043% | 23.98/51.31/21.19 |
| 14 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.0497% | 12.2825% | 2.2043% | 21.39/48.74/21.19 |
| 15 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 19.3343% | 11.2087% | 1.6656% | 25.37/59.31/21.35 |
| 16 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 19.3343% | 11.2087% | 1.6656% | 25.37/59.31/21.35 |
| 17 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 37.0531% | 11.8881% | 3.4531% | 26.62/44.03/22.14 |
| 18 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 36.9682% | 11.8881% | 3.4531% | 26.56/44.03/22.14 |
| 19 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 32.0204% | 11.2265% | 3.1624% | 33.88/48.92/29.28 |
| 20 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 25.9410% | 11.7165% | 1.8327% | 15.50/28.30/13.58 |
| 21 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 36.1228% | 11.0347% | 4.8086% | 34.01/50.04/46.46 |
| 22 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 32.0292% | 11.4117% | 1.8327% | 19.24/27.56/13.58 |
| 23 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 34.2182% | 11.0347% | 4.8086% | 31.95/50.04/46.46 |
| 24 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.5430% | 9.5896% | 2.5823% | 31.05/60.31/31.88 |
| 25 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.5430% | 9.5896% | 2.5823% | 31.05/60.31/31.88 |
| 26 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 28.5631% | 10.0039% | 2.8181% | 34.00/49.04/29.36 |
| 27 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 18.6111% | 10.1171% | 1.2176% | 24.81/55.28/18.45 |
| 28 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 18.6111% | 10.1171% | 1.2176% | 24.81/55.28/18.45 |
| 29 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 21.3984% | 9.5896% | 2.5823% | 29.23/60.31/31.88 |
| 30 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 21.3984% | 9.5896% | 2.5823% | 29.23/60.31/31.88 |
| 31 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 31.0463% | 9.9341% | 2.8995% | 26.76/44.15/22.30 |
| 32 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 31.0463% | 9.9341% | 2.8995% | 26.76/44.15/22.30 |
| 33 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 30.9787% | 9.9341% | 2.8995% | 26.71/44.15/22.30 |
| 34 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 30.9787% | 9.9341% | 2.8995% | 26.71/44.15/22.30 |
| 35 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 23.2101% | 10.4227% | 1.6343% | 15.60/28.32/13.62 |
| 36 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 28.4910% | 10.1548% | 1.6343% | 19.25/27.59/13.62 |
| 37 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 16.9081% | 9.0780% | 1.6656% | 24.50/51.29/21.35 |
| 38 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 16.9081% | 9.0780% | 1.6656% | 24.50/51.29/21.35 |
| 39 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 15.9731% | 11.0278% | 1.4144% | 14.55/38.90/12.09 |
| 40 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 16.2179% | 9.2182% | 1.6656% | 21.98/48.77/21.35 |
| 41 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 16.2179% | 9.2182% | 1.6656% | 21.98/48.77/21.35 |
| 42 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 27.9946% | 8.9518% | 2.6193% | 26.81/44.21/22.39 |
| 43 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 27.9351% | 8.9518% | 2.6193% | 26.76/44.21/22.39 |
| 44 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 16.0778% | 10.5026% | 2.4045% | 10.45/24.83/17.81 |
| 45 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 30.5625% | 8.6625% | 4.3352% | 32.65/39.29/48.17 |
| 46 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 17.1668% | 10.5762% | 1.3040% | 15.38/38.53/13.17 |
| 47 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 14.6126% | 9.8393% | 1.2668% | 14.97/39.04/12.18 |
| 48 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.5222% | 7.5369% | 2.1241% | 34.16/49.26/29.50 |
| 49 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.5222% | 7.5369% | 2.1241% | 34.16/49.26/29.50 |
| 50 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 28.5457% | 8.5917% | 4.4845% | 30.50/37.44/45.30 |
| 51 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 27.2761% | 7.7367% | 3.8556% | 32.78/39.47/48.20 |
| 52 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 14.5831% | 9.3572% | 2.1416% | 10.66/24.89/17.85 |
| 53 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 15.8438% | 9.4601% | 1.1779% | 15.97/38.77/13.39 |
| 54 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 23.9488% | 7.3819% | 3.2164% | 33.83/50.22/46.61 |
| 55 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 23.9488% | 7.3819% | 3.2164% | 33.83/50.22/46.61 |
| 56 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 10.6730% | 8.6351% | 2.0304% | 10.11/30.62/22.56 |
| 57 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 10.6730% | 8.6351% | 2.0304% | 10.11/30.62/22.56 |
| 58 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.7911% | 7.3819% | 3.2164% | 31.92/50.22/46.61 |
| 59 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.7911% | 7.3819% | 3.2164% | 31.92/50.22/46.61 |
| 60 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 24.2933% | 8.5915% | 1.2826% | 14.81/21.70/10.02 |
| 61 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.3586% | 7.6299% | 1.2334% | 19.24/27.64/13.70 |
| 62 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 21.3586% | 7.6299% | 1.2334% | 19.24/27.64/13.70 |
| 63 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 25.5426% | 7.6824% | 3.9878% | 30.70/37.66/45.32 |
| 64 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 24.7011% | 9.1128% | 5.1461% | 20.25/30.89/32.16 |
| 65 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 24.7011% | 9.1128% | 5.1461% | 20.25/30.89/32.16 |
| 66 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 17.5975% | 7.8264% | 1.2334% | 15.77/28.36/13.70 |
| 67 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 17.5975% | 7.8264% | 1.2334% | 15.77/28.36/13.70 |
| 68 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 17.2376% | 8.6119% | 1.2826% | 10.46/21.75/10.02 |
| 69 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.7632% | 9.1128% | 5.1461% | 18.51/30.89/32.16 |
| 70 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.7632% | 9.1128% | 5.1461% | 18.51/30.89/32.16 |
| 71 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 22.5308% | 8.2559% | 4.6461% | 20.52/31.10/32.26 |
| 72 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 20.1268% | 8.4255% | 1.2668% | 20.13/32.41/12.18 |
| 73 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 9.2888% | 7.5780% | 3.0449% | 10.83/44.32/39.04 |
| 74 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 9.2888% | 7.5780% | 3.0449% | 10.83/44.32/39.04 |
| 75 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 9.2550% | 7.5780% | 3.0449% | 10.79/44.32/39.04 |
| 76 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 9.2550% | 7.5780% | 3.0449% | 10.79/44.32/39.04 |
| 77 | SOLUSDT | crypto_high_beta_alt | 2h | relative_strength_chandelier_breakout | 20.8164% | 8.2559% | 4.6461% | 18.80/31.10/32.26 |
| 78 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 20.5743% | 5.8551% | 2.8947% | 32.97/39.83/48.24 |
| 79 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 20.5743% | 5.8551% | 2.8947% | 32.97/39.83/48.24 |
| 80 | SOLUSDT | crypto_high_beta_alt | 1h | debounced_momentum_hysteresis_efficiency_repair | 11.5480% | 7.4309% | 0.9643% | 15.78/39.32/12.36 |

## Guardrails

- Paper/testnet-only monitoring handoff; real-money execution remains prohibited.
- Replay/live notional parity, realized BBO spread, realized fee/slippage/all-in
  round-trip cost, liquidation-inclusive MDD, and account wipeout must be
  recorded for every monitored symbol and candidate.
- Existing four `quality_single_pair` baseline lanes are preserved unchanged.
