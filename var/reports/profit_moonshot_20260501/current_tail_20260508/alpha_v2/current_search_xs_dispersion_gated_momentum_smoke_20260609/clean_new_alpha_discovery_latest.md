# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-09T13:31:28.187989Z`
- pre-registered search hash: `4dd982a04779707f11d4530059f314ebe965cdee32fbd5a92a87a946ca3c7be7`
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

- OOS comp: `0.69%`
- annualized approx: `2.79%`
- monthly equity MDD: `0.56%`
- max OOS MDD: `1.56%`
- positive folds: `2/3`
- Sharpe approx: `1.13`

## Live realism diagnostics

- live plausibility: `not_supported`
- mean validation return: `5.36%`
- mean locked-OOS return: `0.23%`
- positive locked-OOS fold share: `0.67`
- min validation trade events: `54`
- max validation Sharpe: `5.15`
- blockers: `continuous_position_state_across_split_boundaries`, `continuous_position_state_split_simulation_not_live_equivalent`, `fresh_forward_required_before_promotion`, `robust_selector_is_post_failure_diagnostic_requires_fresh_forward`, `selected_rows_not_ready_for_real_money`, `validation_sharpe_too_high_for_live_assumption_without_forward_fill_telemetry`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-04` | `az69_xsdispmom_1h_solusdt_lb48_vw48_thr1p0_dw48_dq0p6_stress0p05_vol0p012_hold4_lev2_2f43f273` | `cross_sectional_dispersion_gated_momentum` | 2.80% | 7.63% | -0.56% | 1.56% |
| `2026-05` | `az69_xsdispmom_1h_bnbusdt_lb24_vw48_thr0p5_dw96_dq0p8_stress0p05_vol0p02_hold8_lev2_bad9f4b8` | `cross_sectional_dispersion_gated_momentum` | 8.23% | 3.64% | 0.80% | 1.36% |
| `2026-06` | `az69_xsdispmom_1h_bnbusdt_lb24_vw24_thr1p0_dw48_dq0p6_stress0p025_vol0p02_hold4_lev2_5d81f229` | `cross_sectional_dispersion_gated_momentum` | 5.55% | 4.80% | 0.45% | 1.25% |
