# 69-asset per-profile Optuna hybrid refit

Generated: `2026-05-30T08:31:32.642085Z`

## Correction

- This run rebuilds the three hybrid source profiles across the 69-symbol universe instead of applying one shared parameter set or only optimizing final stream weights.
- Every symbol/profile pair is Optuna-tuned over family, timeframe, side, entry/exit, hold/cooldown, and integer leverage.
- Domain anchors are tracked beyond BTC: BTC, ETH, SOL, SPY, QQQ, XAU, XAG, and crude proxy anchors are used to penalize single-benchmark clones and profile-level anchor concentration.
- Each rebuilt source profile then gets its own Optuna sleeve-allocation pass, and the final blend reuses the existing v3.5/v3.6 Optuna hybrid engine.
- No live or real-money execution is enabled; `ready_for_real=false` and `real_money_execution=false` remain invariant.

## Selected hybrid

- profile: `hybrid_v3_5_optuna_three_profile_blend`
- version: `v3_5`
- train / validation: `301.5592%` / `715.7005%`
- train / validation MDD: `11.4754%` / `10.5453%`
- RPT bps train / validation: `100.36` / `438.69`
- gross notional: `7.7496x`
- final weights: `{"aggressive_mdd30_gross10_69_asset_profile_optuna": 0.20077133723954163, "balanced_mdd12_gross5_69_asset_profile_optuna": 0.1853495578056165, "growth_mdd20_gross8_69_asset_profile_optuna": 0.2871522107847317}`
- selection reasons: `['train_return_3.0156_below_validation_return_7.1570']`

## Selected train/validation-legal portfolio

- selected legal profile: `hybrid_static_train_dominance_guarded_three_profile_blend`
- train / validation: `160.3316%` / `150.0726%`
- selection reasons: `[]`

## Rebuilt source profiles

| Profile | Sleeves | Gross | Train | Validation | Val MDD | RPT T/V bps | Paper | Top symbol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `balanced_mdd12_gross5_69_asset_profile_optuna` | 22 | 5.00x | 160.3316% | 150.0726% | 7.5634% | 69.40/152.18 | true | `XPTUSDT 10.73%` |
| `growth_mdd20_gross8_69_asset_profile_optuna` | 30 | 8.00x | 174.8674% | 1433.8164% | 17.6281% | 38.25/594.86 | false | `AAPLUSDT 9.56%` |
| `aggressive_mdd30_gross10_69_asset_profile_optuna` | 45 | 10.00x | 218.5344% | 1497.4011% | 18.1911% | 43.72/642.94 | false | `MUUSDT 7.93%` |

## Governance

- search method: `optuna_tpe_per_asset_profile_then_v35_v36_hybrid`
- asset trials/profile: `24`
- profile allocation trials: `96`
- hybrid trials/version: `160`
- runner peak RSS MiB: `1018.78`
- ready_for_real: `false`
- real_money_execution: `false`
