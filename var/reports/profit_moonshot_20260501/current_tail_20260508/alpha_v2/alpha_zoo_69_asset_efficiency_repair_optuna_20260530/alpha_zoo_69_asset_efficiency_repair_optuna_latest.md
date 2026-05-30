# 69-asset live-efficiency repair Optuna

Generated: `2026-05-30T09:54:51.210338Z`

## Purpose

- Repairs the 69-asset per-profile artifact for live/paper efficiency rather than selecting only three assets.
- Every source asset/profile remains individually tuned; this pass retunes portfolio sleeves and hybrid weights with stronger 10bps RPT, sample, turnover, concentration, and 15/20bps stress constraints.
- No locked test set is used; train and latest 8-week validation remain the only selection inputs.
- Real-money execution remains disabled.

## Selected legal portfolio

- profile: `hybrid_v3_5_optuna_three_profile_blend`
- train / validation: `295.9880%` / `172.7926%`
- train / validation MDD: `17.7072%` / `6.0984%`
- RPT bps train / validation: `76.65` / `125.01`
- 20bps stress train / validation proxy: `251.9612%` / `157.7868%`
- gross notional: `7.3257x`
- final weights: `{"aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna": 0.25092248001570466, "balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 0.40468881726998696, "growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 0.2328544319198214}`
- selection reasons: `[]`

## Repaired profiles

| Profile | Sleeves | Gross | Train | Validation | Val MDD | RPT T/V bps | 20bps stress T/V | Low-eff | Paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 45 | 5.00x | 130.5230% | 107.5429% | 6.3080% | 41.65/99.50 | 99.1834%/96.7342% | 49.63% | true |
| `growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 44 | 7.12x | 155.8516% | 149.4461% | 11.8001% | 31.65/106.15 | 106.6022%/135.3674% | 64.68% | true |
| `aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | 57 | 10.00x | 285.9853% | 278.9450% | 10.8603% | 52.59/138.11 | 231.6002%/258.7483% | 73.21% | false |

## Governance

- primary round-trip cost bps: `10.0`
- return-per-turnover threshold bps: `10.0`
- ready_for_real: `false`
- real_money_execution: `false`
- runner peak RSS MiB: `911.34`
