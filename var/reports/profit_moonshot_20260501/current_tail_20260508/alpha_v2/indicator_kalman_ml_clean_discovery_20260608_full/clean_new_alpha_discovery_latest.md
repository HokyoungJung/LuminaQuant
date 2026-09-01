# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-08T14:19:45.022908Z`
- pre-registered search hash: `ee6ecd539a6b5a8c078bd0e39f22ef0bb483d10e1ecbf97236c51b9a6fb087e8`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `-8.77%`
- annualized approx: `-10.43%`
- monthly equity MDD: `28.82%`
- max OOS MDD: `16.70%`
- positive folds: `4/10`
- Sharpe approx: `-0.16`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | `az69_stdridge_4h_solusdt_lb48_alpha1p0_thr0p001_hold4_lev4_d2544303` | `standardized_indicator_ridge_directional` | 43.97% | 24.86% | 9.28% | 3.04% |
| `2025-10` | `az69_stdridge_4h_solusdt_lb48_alpha1p0_thr0p0005_hold8_lev4_787d7390` | `standardized_indicator_ridge_directional` | 34.17% | 28.29% | 14.45% | 6.15% |
| `2025-11` | `az69_stdridge_4h_solusdt_lb48_alpha10p0_thr0p0005_hold8_lev4_fd626457` | `standardized_indicator_ridge_directional` | 75.85% | 27.12% | -0.39% | 10.49% |
| `2025-12` | `az69_kalmanvoltrend_4h_adausdt_lb48_q0p002_r0p02_sz0p25_vol0p028_hold4_lev4_25a0d6ec` | `indicator_kalman_volatility_trend` | 47.40% | 18.69% | -6.97% | 11.02% |
| `2026-01` | `az69_leadlag_4h_solusdt_avaxusdt_lb12_thr0p01_lag0p25_hold4_lev4_9f73c134` | `cross_asset_lead_lag_momentum` | 43.53% | 23.21% | 9.71% | 1.97% |
| `2026-02` | `az69_leadlag_4h_solusdt_avaxusdt_lb12_thr0p01_lag0p25_hold4_lev4_9f73c134` | `cross_asset_lead_lag_momentum` | 63.11% | 18.95% | 0.79% | 8.28% |
| `2026-03` | `az69_stdridge_4h_dogeusdt_lb24_alpha10p0_thr0p0005_hold8_lev4_22369ed8` | `standardized_indicator_ridge_directional` | 94.31% | 21.54% | -16.11% | 16.70% |
| `2026-04` | `az69_stdridge_4h_avaxusdt_lb24_alpha1p0_thr0p001_hold4_lev4_880a5f55` | `standardized_indicator_ridge_directional` | 127.77% | 23.83% | -13.45% | 14.95% |
| `2026-05` | `az69_vwapatrbbrev_4h_avaxusdt_lb24_vz1p25_bb2p0_atr0p028_hold8_lev4_2025e3f7` | `indicator_vwap_atr_bollinger_reversion` | 12.62% | 11.84% | -1.60% | 3.59% |
| `2026-06` | `az69_drvolmom_4h_tonusdt_lb12_mom0p006_vol0p028_cr24_crret0p055_hold4_lev4_8c8c4a30` | `deep_research_vol_managed_momentum_crash_gate` | 18.48% | 42.80% | -0.37% | 4.35% |
