# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-07T05:58:15.162556Z`
- pre-registered search hash: `4a6fee0f540f5d9ce15158beaf6b7c91ad89600cb5d76e1c4bfa0e33008b81b7`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `1.92%`
- annualized approx: `7.89%`
- monthly equity MDD: `1.54%`
- max OOS MDD: `2.05%`
- positive folds: `1/3`
- Sharpe approx: `0.89`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-04` | `az69_leadlag_4h_solusdt_bnbusdt_lb12_thr0p02_lag0p25_hold4_lev4_8c3ff680` | `cross_asset_lead_lag_momentum` | 6.83% | 5.27% | -1.44% | 2.05% |
| `2026-05` | `az69_absorb_1h_solusdt_lb48_vz1p5_wick0p45_hold4_lev4_82153506` | `volume_absorption_reversal` | 3.22% | 4.84% | -0.10% | 0.83% |
| `2026-06` | `az69_leadlag_4h_btcusdt_trxusdt_lb12_thr0p01_lag0p5_hold8_lev4_5ace7e24` | `cross_asset_lead_lag_momentum` | 18.14% | 6.13% | 3.51% | 0.73% |
