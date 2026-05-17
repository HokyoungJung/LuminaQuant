# Common-split Alpha Zoo vs Hybrid v3.5/v3.6

- generated_at_utc: `2026-05-17T07:18:22.856696Z`
- baseline_parent: `80a557c133930f51748ec20c4e582aa0d6f678de`
- Alpha Zoo old split: **historical only**, not used for common-split selection.
- locked-OOS: gate/report-only after candidate freeze.
- return/MDD: diagnostic/report-only, not a hard promotion gate.

## Common split

- locked_oos: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z`; unique timestamps `1593`, rows `7965`
- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z`; unique timestamps `8760`, rows `43800`
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z`; unique timestamps `1416`, rows `7080`

## Candidate split performance

| candidate | split | period | active | return | MDD | return/MDD diag | Sharpe | Sortino | smart Sortino | Calmar | trades | liq | min buffer | deployable | rejection reasons |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| alpha_zoo_strict_6x_old_split_historical_only | train | 2025-01-01T00:00:00Z → 2025-10-22T04:00:00Z | None → None | +68.88% | +29.57% | 2.330 | 1.569 | 1.920 | 1.482 | 2.330 | 1779 | 0 | 9049.126 | False | historical_old_split_only_not_common_split_selection |
| alpha_zoo_strict_6x_old_split_historical_only | validation | 2025-10-22T05:00:00Z → 2026-01-28T06:00:00Z | None → None | +30.12% | +9.56% | 3.151 | 1.552 | 2.096 | 1.913 | 3.151 | 524 | 0 | 9527.696 | False | historical_old_split_only_not_common_split_selection |
| alpha_zoo_strict_6x_old_split_historical_only | locked_oos | 2026-01-28T07:00:00Z → 2026-05-06T23:00:00Z | None → None | +41.10% | +13.67% | 3.007 | 2.143 | 2.842 | 2.500 | 3.007 | 540 | 0 | 9572.449 | False | historical_old_split_only_not_common_split_selection |
| alpha_zoo_strict_6x_common_split_carry_forward_old_selected | train | 2025-01-01T00:00:00Z → 2025-12-31T23:00:00Z | 2025-01-02T00:00:00 → 2025-12-31T15:00:00 | +114.46% | +29.57% | 3.872 | 2.062 | 2.566 | 1.981 | 3.872 | 2143 | 0 | 9049.126 | True |  |
| alpha_zoo_strict_6x_common_split_carry_forward_old_selected | validation | 2026-01-01T00:00:00Z → 2026-02-28T23:00:00Z | 2026-01-01T11:00:00 → 2026-02-28T10:00:00 | +19.97% | +13.67% | 1.461 | 1.252 | 1.557 | 1.370 | 1.461 | 335 | 0 | 9572.449 | True |  |
| alpha_zoo_strict_6x_common_split_carry_forward_old_selected | locked_oos | 2026-03-01T00:00:00Z → 2026-05-06T23:00:00Z | 2026-03-01T01:00:00 → 2026-05-06T23:00:00 | +20.51% | +6.79% | 3.022 | 1.772 | 2.579 | 2.415 | 3.022 | 365 | 0 | 9643.448 | True |  |
| alpha_zoo_strict_6x_common_split_reselected:alpha_zoo_conservative_exit | train | 2025-01-01T00:00:00Z → 2025-12-31T23:00:00Z | 2025-01-02T00:00:00 → 2025-12-31T15:00:00 | +114.46% | +29.57% | 3.872 | 2.062 | 2.566 | 1.981 | 3.872 | 2143 | 0 | 9049.126 | True |  |
| alpha_zoo_strict_6x_common_split_reselected:alpha_zoo_conservative_exit | validation | 2026-01-01T00:00:00Z → 2026-02-28T23:00:00Z | 2026-01-01T11:00:00 → 2026-02-28T10:00:00 | +19.97% | +13.67% | 1.461 | 1.252 | 1.557 | 1.370 | 1.461 | 335 | 0 | 9572.449 | True |  |
| alpha_zoo_strict_6x_common_split_reselected:alpha_zoo_conservative_exit | locked_oos | 2026-03-01T00:00:00Z → 2026-05-06T23:00:00Z | 2026-03-01T01:00:00 → 2026-05-06T23:00:00 | +20.51% | +6.79% | 3.022 | 1.772 | 2.579 | 2.415 | 3.022 | 365 | 0 | 9643.448 | True |  |
| hybrid_v3_5_optuna_common_split | train | 2025-01-01T00:00:00Z → 2025-12-31T23:00:00Z | 2025-01-01T00:00:00Z → 2025-12-31T23:00:00Z | +47.73% | +11.04% | 4.322 | 2.706 | 2.932 | 2.641 | 4.322 | 7514 | 0 | 9932.439 | True |  |
| hybrid_v3_5_optuna_common_split | validation | 2026-01-01T00:00:00Z → 2026-02-28T23:00:00Z | 2026-01-01T00:00:00Z → 2026-02-28T23:00:00Z | +13.31% | +2.26% | 5.884 | 5.391 | 7.610 | 7.442 | 51.558 | 1302 | 0 | 14594.054 | True |  |
| hybrid_v3_5_optuna_common_split | locked_oos | 2026-03-01T00:00:00Z → 2026-05-06T23:00:00Z | 2026-03-01T00:00:00Z → 2026-05-06T23:00:00Z | +8.52% | +1.77% | 4.828 | 5.259 | 7.317 | 7.190 | 32.173 | 1467 | 0 | 16587.500 | True |  |
| hybrid_v3_6_optuna_common_split | train | 2025-01-01T00:00:00Z → 2025-12-31T23:00:00Z | 2025-01-01T00:00:00Z → 2025-12-31T23:00:00Z | +49.52% | +7.69% | 6.436 | 2.898 | 2.999 | 2.785 | 6.436 | 7514 | 0 | 9847.515 | True |  |
| hybrid_v3_6_optuna_common_split | validation | 2026-01-01T00:00:00Z → 2026-02-28T23:00:00Z | 2026-01-01T00:00:00Z → 2026-02-28T23:00:00Z | +12.49% | +1.54% | 8.138 | 7.002 | 8.681 | 8.550 | 69.800 | 1302 | 0 | 14879.249 | True |  |
| hybrid_v3_6_optuna_common_split | locked_oos | 2026-03-01T00:00:00Z → 2026-05-06T23:00:00Z | 2026-03-01T00:00:00Z → 2026-05-06T23:00:00Z | +7.79% | +1.75% | 4.455 | 4.860 | 5.991 | 5.888 | 29.200 | 1467 | 0 | 16664.270 | True |  |

## Selection provenance and locked-OOS audit

- locked-OOS contamination violation: `False`
- violation reasons: `none`
- Hybrid input universe: `A0 + P0 + E0 + S1 + S2 + S3 + S4`.
- Hybrid live promotion possible: `True`.

## Strict zero-liquidation integer leverage lane

| leverage | deployable | strict_safe | OOS return | OOS MDD | liq | min buffer |
|---:|---|---|---:|---:|---:|---:|
| 1.0 | False | True | +3.24% | +1.16% | 0 | 9841.521 |
| 2.0 | True | True | +6.56% | +2.30% | 0 | 9683.042 |
| 3.0 | True | True | +9.94% | +3.44% | 0 | 9524.563 |
| 4.0 | True | True | +13.39% | +4.57% | 0 | 9366.084 |
| 5.0 | True | True | +16.92% | +5.68% | 0 | 9207.605 |
| 6.0 | True | True | +20.51% | +6.79% | 0 | 9049.126 |

## Diagnostic nonfatal 5x/6x lane

Diagnostic only; separated from live promotion.

- 5.0x: promotion_allowed `False`, total_liquidations `0`, min_buffer `9207.605`
- 6.0x: promotion_allowed `False`, total_liquidations `0`, min_buffer `9049.126`

## Decision

- best common-split strict candidate: `crypto_fx_alpha_zoo_state_calibrated`
- hybrid v3.5/v3.6 live promotion possible: `True`
- memory peak RSS MiB: `769.102`
- research history/source ledger update: `not_regenerated_no_new_global_source_family_or_chronology_ledger_change`
