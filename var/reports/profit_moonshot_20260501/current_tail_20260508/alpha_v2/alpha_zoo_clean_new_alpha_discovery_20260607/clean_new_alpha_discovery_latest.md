# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-08T13:32:34.006678Z`
- pre-registered search hash: `d92dce2b046441bcf1a7a7ebfa5499844b418a52d6d2416fefad491458967312`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `6.86%`
- annualized approx: `17.27%`
- monthly equity MDD: `1.16%`
- max OOS MDD: `8.28%`
- positive folds: `3/5`
- Sharpe approx: `1.98`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-02` | `az69_leadlag_4h_solusdt_avaxusdt_lb12_thr0p01_lag0p25_hold4_lev4_9f73c134` | `cross_asset_lead_lag_momentum` | 63.11% | 18.95% | 0.79% | 8.28% |
| `2026-03` | `az69_drvolmom_4h_xrpusdt_lb48_mom0p012_vol0p018_cr24_crret0p055_hold8_lev4_6001fd08` | `deep_research_vol_managed_momentum_crash_gate` | 28.53% | 18.97% | 5.13% | 5.73% |
| `2026-04` | `az69_drvolmom_4h_xrpusdt_lb48_mom0p012_vol0p018_cr24_crret0p055_hold8_lev4_6001fd08` | `deep_research_vol_managed_momentum_crash_gate` | 41.40% | 13.68% | 2.03% | 3.43% |
| `2026-05` | `az69_drvolmom_4h_xrpusdt_lb48_mom0p012_vol0p018_cr24_crret0p035_hold8_lev4_11fbbf9f` | `deep_research_vol_managed_momentum_crash_gate` | 22.64% | 11.54% | -0.79% | 4.42% |
| `2026-06` | `az69_drvolmom_4h_tonusdt_lb12_mom0p006_vol0p028_cr24_crret0p055_hold4_lev4_8c8c4a30` | `deep_research_vol_managed_momentum_crash_gate` | 18.48% | 42.80% | -0.37% | 4.35% |
