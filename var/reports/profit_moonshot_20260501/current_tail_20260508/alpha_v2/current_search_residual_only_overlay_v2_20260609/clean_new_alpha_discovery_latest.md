# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-09T13:34:31.243714Z`
- pre-registered search hash: `4dd982a04779707f11d4530059f314ebe965cdee32fbd5a92a87a946ca3c7be7`
- selection policy: `robust_train_validation_v1`
- enabled families: `7`
- integer leverages: `[2]`
- fold workers: `3`
- simulation backend: `rust`
- candidate cap sort: `eligible_first_active_train_validation_selection_score`
- candidate rows retained/written: `3000`/`200`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`

## Aggregate selected fold result

- OOS comp: `4.16%`
- annualized approx: `17.73%`
- monthly equity MDD: `0.00%`
- max OOS MDD: `1.45%`
- positive folds: `3/3`
- Sharpe approx: `5.90`

## Live realism diagnostics

- live plausibility: `not_supported`
- mean validation return: `4.67%`
- mean locked-OOS return: `1.37%`
- positive locked-OOS fold share: `1.00`
- min validation trade events: `46`
- max validation Sharpe: `4.72`
- blockers: `continuous_position_state_across_split_boundaries`, `continuous_position_state_split_simulation_not_live_equivalent`, `fresh_forward_required_before_promotion`, `robust_selector_is_post_failure_diagnostic_requires_fresh_forward`, `selected_rows_not_ready_for_real_money`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-04` | `az69_leadlag_1h_solusdt_bnbusdt_lb12_thr0p02_lag0p25_hold8_lev2_f88eb163` | `cross_asset_lead_lag_momentum` | 2.99% | 3.56% | 0.95% | 0.77% |
| `2026-05` | `az69_xsvamom_1h_bnbusdt_lb24_vw48_thr0p5_stress0p05_vol0p02_hold8_lev2_b2e98dfb` | `cross_sectional_vol_adjusted_momentum` | 1.86% | 5.14% | 0.86% | 1.45% |
| `2026-06` | `az69_xsvamom_1h_bnbusdt_lb24_vw48_thr0p5_stress0p05_vol0p02_hold8_lev2_b2e98dfb` | `cross_sectional_vol_adjusted_momentum` | 2.58% | 5.30% | 2.30% | 1.18% |
