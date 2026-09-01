# Integer-Leverage Optuna Hybrid Decision

Generated: `2026-05-24T14:24:41.957095Z`

## Method correction

- This artifact replaces the prior coarse 5% grid as the optimization decision surface.
- Optuna/TPESampler tunes v3.5- and v3.6-style hybrid parameters, matching the existing `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` pattern.
- v3.5 mapping: warmup-learned default profile + rolling return/error weights + high-vol boost + bias/exposure dampening.
- v3.6 mapping: v3.5 mechanics plus online default-profile refresh from rolling scores.
- Objective/learning/selection inputs: train + validation only. locked-OOS is report-only after frozen Optuna params.

## Comparison

| Profile | Version | Optimizer | Weights/avg TV weights | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | RPT T/V/OOS bps | Paper candidate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_mdd12_gross5` | `source_profile` | `source_profile` | `{"balanced_mdd12_gross5": 1.0}` | 1.00x | 74.6685% | 33.2153% | 5.5300% | 11.6134% | 7.2003% | 30.91/57.02/22.21 | true |
| `growth_mdd20_gross8` | `source_profile` | `source_profile` | `{"growth_mdd20_gross8": 1.0}` | 3.90x | 262.3353% | 71.6291% | 23.3695% | 19.9983% | 9.2371% | 32.16/37.72/23.35 | true |
| `aggressive_mdd30_gross10_shadow` | `source_profile` | `source_profile` | `{"aggressive_mdd30_gross10_shadow": 1.0}` | 4.90x | 438.4462% | 117.4976% | 27.5772% | 29.4044% | 12.3630% | 38.81/44.16/21.87 | true |
| `hybrid_mdd20_three_profile_blend` | `grid_baseline` | `coarse_5pct_grid_baseline_not_selected_by_this_runner` | `{"aggressive_mdd30_gross10_shadow": 0.15, "balanced_mdd12_gross5": 0.15, "growth_mdd20_gross8": 0.7}` | 3.61x | 262.3642% | 72.5692% | 21.2977% | 19.9330% | 9.0718% | 33.78/39.96/22.97 | true |
| `hybrid_v3_5_optuna_three_profile_blend` | `v3_5` | `optuna_tpe` | `{"aggressive_mdd30_gross10_shadow": 0.781093636345445, "balanced_mdd12_gross5": 0.10913956140451388, "growth_mdd20_gross8": 0.10976680225004104}` | 4.36x | 611.5025% | 138.3170% | 20.8319% | 18.9796% | 10.5735% | 83.39/79.17/25.29 | true |
| `hybrid_v3_6_optuna_three_profile_blend` | `v3_6` | `optuna_tpe` | `{"aggressive_mdd30_gross10_shadow": 0.32021147589191945, "balanced_mdd12_gross5": 0.3309978420797652, "growth_mdd20_gross8": 0.34879068202831537}` | 3.26x | 296.4869% | 85.9099% | 11.5273% | 15.7399% | 8.4448% | 51.05/62.78/17.91 | true |

## Selected Optuna hybrid

- profile: `hybrid_v3_5_optuna_three_profile_blend`
- hybrid version: `v3_5`
- avg train+validation weights: `{"aggressive_mdd30_gross10_shadow": 0.781093636345445, "balanced_mdd12_gross5": 0.10913956140451388, "growth_mdd20_gross8": 0.10976680225004104}`
- final weights: `{"aggressive_mdd30_gross10_shadow": 0.5726993181554131, "balanced_mdd12_gross5": 0.07983098667432496, "growth_mdd20_gross8": 0.08067156944866473}`
- train/validation/OOS report-only: `611.5025%` / `138.3170%` / `20.8319%`
- validation MDD / OOS MDD: `18.9796%` / `10.5735%`
- RPT bps train/validation/OOS: `83.39` / `79.17` / `25.29`
- selection reasons: `[]`
- report-only OOS gate reasons: `[]`

## Governance

- primary round-trip cost bps: `10.0`
- return-per-turnover threshold bps: `10.0`
- ready_for_real: `false`
- real_money_execution: `false`
- real_execution_allowed: `false`
- locked-OOS used for selection: `False`
