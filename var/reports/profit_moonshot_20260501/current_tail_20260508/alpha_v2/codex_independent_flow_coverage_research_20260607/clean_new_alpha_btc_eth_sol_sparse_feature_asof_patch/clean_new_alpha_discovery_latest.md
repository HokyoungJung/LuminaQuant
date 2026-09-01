# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-07T11:40:24.739626Z`
- pre-registered search hash: `1ad1e7560baf9c68152b31351efb8e5bf42e5c4d54a94439b5f4754a5794cf7f`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `3.75%`
- annualized approx: `9.23%`
- monthly equity MDD: `5.28%`
- max OOS MDD: `8.32%`
- positive folds: `3/5`
- Sharpe approx: `0.65`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-02` | `az69_leadlag_4h_btcusdt_ethusdt_lb24_thr0p01_lag0p25_hold8_lev4_274796ed` | `cross_asset_lead_lag_momentum` | 31.63% | 9.83% | 0.11% | 3.49% |
| `2026-03` | `az69_leadlag_4h_btcusdt_ethusdt_lb24_thr0p01_lag0p25_hold8_lev4_274796ed` | `cross_asset_lead_lag_momentum` | 26.89% | 14.06% | -5.28% | 8.32% |
| `2026-04` | `az69_flowexhaust_4h_ethusdt_lb6_flow0p15_ret0p008_fundcap0p0003_vol0p008_hold8_lev4_f90ccd3e` | `feature_taker_flow_exhaustion_reversal` | 6.74% | 13.13% | 5.16% | 3.83% |
| `2026-05` | `az69_flowexhaust_4h_ethusdt_lb6_flow0p15_ret0p008_fundcap0p0003_vol0p008_hold8_lev4_f90ccd3e` | `feature_taker_flow_exhaustion_reversal` | 6.32% | 19.43% | -0.79% | 1.08% |
| `2026-06` | `az69_squeeze_4h_ethusdt_lb72_q0p15_br12_hold4_lev2_03b53284` | `volatility_squeeze_breakout` | 5.90% | 0.93% | 4.87% | 0.55% |
