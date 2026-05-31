# 69-asset relaxed live-efficiency repair Optuna

Generated: `2026-05-31T04:31:25.361986Z`

## Policy

- policy id: `material_positive_tradfi_low_sample_mdd_guard_20260531`
- train < validation is allowed only when both train and validation returns are >= 2% and MDD guard passes.
- TradFi/material-positive low-sample rows are allowed as warnings under the same MDD and strict 10bps RPT guards.
- Gross/concentration pressure is optimized/penalized; it is not a hard rejection while MDD stays below the relaxed guard.
- 10bps return-per-turnover gate remains strict on train and validation.
- No test set or locked-OOS is used for selection; real-money execution remains disabled.

## Candidate expansion versus strict pass

- strict gate-ok rows: `18`
- relaxed gate-ok rows: `30`
- newly admitted rows: `16`
- relaxed unique gate-ok symbols: `19`
- newly admitted symbols: `COINUSDT, COPPERUSDT, CRCLUSDT, ETHUSDT, GOOGLUSDT, INTCUSDT, METAUSDT, MSTRUSDT, PLTRUSDT, TONUSDT, XPDUSDT`

## Selected relaxed portfolio

- profile: `aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna`
- train / validation: `555.8771%` / `518.4857%`
- train / validation MDD: `27.7065%` / `21.2478%`
- RPT bps train / validation: `96.70` / `286.89`
- 20bps stress train / validation proxy: `498.3937%` / `500.4133%`
- gross notional: `7.2541x`
- final weights: `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 1.0}`
- selection reasons: `[]`

## Selected relaxed Optuna hybrid

- profile: `hybrid_v3_5_optuna_three_profile_blend`
- train / validation: `284.5998%` / `373.8607%`
- train / validation MDD: `23.9019%` / `16.3433%`
- RPT bps train / validation: `58.07` / `250.46`
- gross notional: `6.8713x`
- final weights: `{"aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna": 0.8879998972843969, "balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna": 0.05299226965092312, "growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna": 0.04772476342463325}`
- selection reasons: `[]`

## Repaired relaxed profiles

| Profile | Sleeves | Gross | Train | Validation | Val MDD | RPT T/V bps | Relaxed share | Paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna` | 7 | 2.79x | 127.6411% | 102.7096% | 8.1747% | 132.65/191.96 | 29.56% | true |
| `growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna` | 11 | 5.88x | 274.9878% | 296.8675% | 18.4082% | 49.65/292.93 | 79.56% | true |
| `aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | 12 | 7.25x | 555.8771% | 518.4857% | 21.2478% | 96.70/286.89 | 55.19% | true |

## Strict reference

- available: `True`
- path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_efficiency_repair_optuna_20260530/alpha_zoo_69_asset_efficiency_repair_optuna_latest.json`
- strict selected train/validation: `119.3799%` / `79.7120%`
- strict selected MDD train/validation: `16.6872%` / `7.4789%`

## Governance

- primary round-trip cost bps: `10.0`
- return-per-turnover threshold bps: `10.0`
- ready_for_real: `false`
- real_money_execution: `false`
- runner peak RSS MiB: `893.31`
