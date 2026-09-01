# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-09T12:36:36.900722Z`
- pre-registered search hash: `c1e3482dfefb4c0591eb80c5c343650693f7e6f2a54db285c0f4e418d1e4db05`
- selection policy: `robust_train_validation_v1`
- enabled families: `1`
- integer leverages: `[2]`
- fold workers: `3`
- simulation backend: `rust`
- candidate cap sort: `eligible_first_active_train_validation_selection_score`
- candidate rows retained/written: `1500`/`200`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `0.61%`
- annualized approx: `2.46%`
- monthly equity MDD: `1.11%`
- max OOS MDD: `2.10%`
- positive folds: `2/3`
- Sharpe approx: `0.56`

## Live realism diagnostics

- live plausibility: `not_supported`
- mean validation return: `1.93%`
- mean locked-OOS return: `0.21%`
- positive locked-OOS fold share: `0.67`
- min validation trade events: `20`
- max validation Sharpe: `4.74`
- blockers: `continuous_position_state_across_split_boundaries`, `continuous_position_state_split_simulation_not_live_equivalent`, `fresh_forward_required_before_promotion`, `robust_selector_is_post_failure_diagnostic_requires_fresh_forward`, `selected_rows_not_ready_for_real_money`, `some_validation_samples_below_30_trade_events`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-04` | `az69_vwapkalmanpullback_1h_solusdt_lb96_q0p002_r0p02_trend0p25_pb0p25_atrx2p5_vol0p028_hold8_lev2_c51c2adb` | `indicator_vwap_kalman_pullback_continuation` | 4.35% | 2.55% | -1.11% | 2.10% |
| `2026-05` | `az69_vwapkalmanpullback_1h_solusdt_lb48_q0p002_r0p08_trend0p5_pb0p25_atrx2p5_vol0p018_hold4_lev2_7a584ccc` | `indicator_vwap_kalman_pullback_continuation` | 3.09% | 1.53% | 1.48% | 0.62% |
| `2026-06` | `az69_vwapkalmanpullback_1h_solusdt_lb48_q0p002_r0p08_trend0p5_pb0p25_atrx2p5_vol0p018_hold4_lev2_7a584ccc` | `indicator_vwap_kalman_pullback_continuation` | 4.44% | 1.70% | 0.26% | 0.32% |
