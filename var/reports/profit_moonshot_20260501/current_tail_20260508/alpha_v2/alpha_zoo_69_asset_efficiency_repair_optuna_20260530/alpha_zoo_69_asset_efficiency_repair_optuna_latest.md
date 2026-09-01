# 69-asset live-efficiency repair Optuna

Generated: `2026-05-30T16:17:37.479914Z`

## Purpose

- Repairs the 69-asset per-profile artifact for live/paper efficiency rather than selecting only three assets.
- Every source asset/profile remains individually tuned; this pass retunes portfolio sleeves and hybrid weights with stronger 10bps RPT, sample, turnover, concentration, and 15/20bps stress constraints.
- No locked test set is used; train and latest 8-week validation remain the only selection inputs.
- Symbol/timeframe rows without train-split bars are excluded from parameter fitting, sleeve allocation, hybrid selection, and live promotion; they remain watch/research coverage only until a future refit has train data.
- Real-money execution remains disabled.

## Train eligibility

- train-eligible symbols: `32`
- train-ineligible symbols: `37`
- exclusion policy: `exclude_symbol_timeframes_without_train_rows_from_parameter_fit_allocation_selection_and_live_promotion`
- warmup scope: `train_split_only`
- train-ineligible symbol list: `QQQUSDT, SPYUSDT, SOXLUSDT, AAPLUSDT, TSMUSDT, MUUSDT, SNDKUSDT, MSFTUSDT, AVGOUSDT, BABAUSDT, AMDUSDT, QCOMUSDT, USARUSDT, LITEUSDT, ORCLUSDT, DISUSDT, UBERUSDT, CSCOUSDT, HDUSDT, MRVLUSDT, CRWVUSDT, WMTUSDT, JPMUSDT, VUSDT, BRKBUSDT, FLNCUSDT, DRAMUSDT, RKLBUSDT, CBRSUSDT, NBISUSDT, WDCUSDT, ARMUSDT, BEUSDT, COHRUSDT, SPCXUSDT, OPENAIUSDT, QNTXUSDT`

## Selected legal portfolio

- profile: `balanced_mdd12_gross5_69_asset_efficiency_repair_optuna`
- train / validation: `119.3799%` / `79.7120%`
- train / validation MDD: `16.6872%` / `7.4789%`
- RPT bps train / validation: `108.53` / `157.53`
- 20bps stress train / validation proxy: `108.3799%` / `74.6520%`
- gross notional: `2.2000x`
- final weights: `{"balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 1.0}`
- selection reasons: `[]`

## Selected Optuna hybrid for paper/testnet live handoff

- profile: `hybrid_v3_6_optuna_three_profile_blend`
- train / validation: `96.5913%` / `68.1871%`
- train / validation MDD: `12.5785%` / `7.8678%`
- RPT bps train / validation: `42.30` / `149.64`
- gross notional: `2.5042x`
- final weights: `{"aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna": 0.1823047019724131, "balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 0.6162420536737696, "growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 0.1809761276157996}`
- selection reasons: `[]`

## Repaired profiles

| Profile | Sleeves | Gross | Train | Validation | Val MDD | RPT T/V bps | 20bps stress T/V | Low-eff | Paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 5 | 2.20x | 119.3799% | 79.7120% | 7.4789% | 108.53/157.53 | 108.3799%/74.6520% | 0.00% | true |
| `growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 6 | 1.60x | 68.7714% | 9.8661% | 3.6407% | 24.28/30.25 | 40.4465%/6.6041% | 62.64% | true |
| `aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | 7 | 3.80x | 316.1673% | 69.8299% | 15.5162% | 66.94/78.55 | 268.9373%/60.9399% | 28.95% | true |

## Governance

- primary round-trip cost bps: `10.0`
- return-per-turnover threshold bps: `10.0`
- ready_for_real: `false`
- real_money_execution: `false`
- runner peak RSS MiB: `870.61`
