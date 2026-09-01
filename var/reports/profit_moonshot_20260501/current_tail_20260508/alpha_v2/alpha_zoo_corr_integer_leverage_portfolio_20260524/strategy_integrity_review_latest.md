# Integer-Leverage Strategy Integrity Review

- status: `pass`
- paper profiles: `3`
- unique strategy sleeves checked: `6`
- model id source: `derived_from_frozen_corr_decision_artifact_not_hardcoded_allowlist`

## Checks

- calendar/date rule check: `pass`
- 10bps cost check: `pass` (10.0bps round-trip friction proxy)
- locked-OOS policy: `pass`
- live-level status: `paper_testnet_review_only`

## Strategy sleeves

| Model | Symbol | TF | Family | Side | Calendar hits |
| --- | --- | --- | --- | --- | --- |
| `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5` | ETHUSDT | 2h | relative_residual_reclaim | long_short | [] |
| `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26` | SOLUSDT | 2h | relative_strength_chandelier_breakout | long_short | [] |
| `a30fb_voladj_trend_4h_trxusdt_lb6_z1p5_hold12_cool2_adx15_3p0x_0p15_cca555d7` | TRXUSDT | 4h | volatility_adjusted_trend_persistence | long_short | [] |
| `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p15_d6eac828` | SOLUSDT | 1h | debounced_momentum_hysteresis_efficiency_repair | long_short | [] |
| `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d` | SOLUSDT | 1h | debounced_momentum_hysteresis_efficiency_repair | short_only | [] |
| `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold12_cool6_none_4p0x_0p1_b27a86b8` | SOLUSDT | 1h | debounced_momentum_hysteresis_efficiency_repair | short_only | [] |
