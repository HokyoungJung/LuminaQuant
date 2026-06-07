# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-07T06:13:13.852816Z`
- pre-registered search hash: `4a6fee0f540f5d9ce15158beaf6b7c91ad89600cb5d76e1c4bfa0e33008b81b7`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `-0.24%`
- annualized approx: `-0.57%`
- monthly equity MDD: `8.72%`
- max OOS MDD: `8.32%`
- positive folds: `3/5`
- Sharpe approx: `0.04`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-02` | `az69_leadlag_4h_solusdt_avaxusdt_lb12_thr0p01_lag0p25_hold4_lev4_9f73c134` | `cross_asset_lead_lag_momentum` | 63.11% | 18.95% | 0.79% | 8.28% |
| `2026-03` | `az69_leadlag_4h_btcusdt_ethusdt_lb24_thr0p01_lag0p25_hold8_lev4_274796ed` | `cross_asset_lead_lag_momentum` | 26.89% | 14.06% | -5.28% | 8.32% |
| `2026-04` | `az69_leadlag_4h_btcusdt_avaxusdt_lb6_thr0p01_lag0p25_hold4_lev4_19b0a1b8` | `cross_asset_lead_lag_momentum` | 10.07% | 9.94% | -3.63% | 5.00% |
| `2026-05` | `az69_leadlag_1h_solusdt_xrpusdt_lb12_thr0p02_lag0p25_hold4_lev4_07bb10f1` | `cross_asset_lead_lag_momentum` | 18.88% | 4.76% | 0.29% | 0.81% |
| `2026-06` | `az69_leadlag_4h_btcusdt_avaxusdt_lb12_thr0p02_lag0p5_hold8_lev4_e3af3f45` | `cross_asset_lead_lag_momentum` | 54.49% | 4.26% | 8.11% | 1.33% |
