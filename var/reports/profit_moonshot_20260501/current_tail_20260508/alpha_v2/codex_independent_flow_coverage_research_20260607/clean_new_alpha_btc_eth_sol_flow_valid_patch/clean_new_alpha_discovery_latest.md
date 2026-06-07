# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-07T11:27:01.115061Z`
- pre-registered search hash: `1d421663b0f9f785a18d69e5068c81f1816005598f5a378bfeb697239f2488f6`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `-4.06%`
- annualized approx: `-9.46%`
- monthly equity MDD: `8.62%`
- max OOS MDD: `8.32%`
- positive folds: `2/5`
- Sharpe approx: `-0.68`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-02` | `az69_leadlag_4h_btcusdt_ethusdt_lb24_thr0p01_lag0p25_hold8_lev4_274796ed` | `cross_asset_lead_lag_momentum` | 31.63% | 9.83% | 0.11% | 3.49% |
| `2026-03` | `az69_leadlag_4h_btcusdt_ethusdt_lb24_thr0p01_lag0p25_hold8_lev4_274796ed` | `cross_asset_lead_lag_momentum` | 26.89% | 14.06% | -5.28% | 8.32% |
| `2026-04` | `az69_leadlag_1h_btcusdt_solusdt_lb12_thr0p01_lag0p5_hold8_lev2_cae238e2` | `cross_asset_lead_lag_momentum` | 1.59% | 3.99% | -3.42% | 3.55% |
| `2026-05` | `az69_absorb_1h_solusdt_lb48_vz1p5_wick0p45_hold4_lev4_82153506` | `volume_absorption_reversal` | 3.22% | 4.84% | -0.10% | 0.83% |
| `2026-06` | `az69_squeeze_4h_ethusdt_lb72_q0p15_br12_hold4_lev2_03b53284` | `volatility_squeeze_breakout` | 5.90% | 0.93% | 4.87% | 0.55% |
