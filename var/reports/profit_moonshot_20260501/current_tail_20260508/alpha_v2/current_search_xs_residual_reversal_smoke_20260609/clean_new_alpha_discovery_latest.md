# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-09T13:27:34.686030Z`
- pre-registered search hash: `b04872b2d911cfdac856771e0e582b9e6d184ed9357f8b1fda59e6673eaeeb22`
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

- OOS comp: `-0.25%`
- annualized approx: `-1.49%`
- monthly equity MDD: `0.25%`
- max OOS MDD: `0.98%`
- positive folds: `0/2`
- Sharpe approx: `-2.45`

## Live realism diagnostics

- live plausibility: `not_supported`
- mean validation return: `1.48%`
- mean locked-OOS return: `-0.12%`
- positive locked-OOS fold share: `0.00`
- min validation trade events: `13`
- max validation Sharpe: `4.62`
- blockers: `continuous_position_state_across_split_boundaries`, `continuous_position_state_split_simulation_not_live_equivalent`, `fresh_forward_required_before_promotion`, `robust_selector_is_post_failure_diagnostic_requires_fresh_forward`, `selected_rows_not_ready_for_real_money`, `some_validation_samples_below_30_trade_events`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2026-05` | `az69_xsresrev_1h_solusdt_lb12_vw48_z1p75_tz1p25_stress0p025_vol0p02_hold4_lev2_c27db3a5` | `cross_sectional_residual_reversal` | 0.16% | 0.28% | -0.25% | 0.98% |
| `2026-06` | `az69_xsresrev_1h_solusdt_lb24_vw48_z1p25_tz1p25_stress0p05_vol0p02_hold8_lev2_0bab561e` | `cross_sectional_residual_reversal` | 2.37% | 2.69% | 0.00% | 0.00% |
