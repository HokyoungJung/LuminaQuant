# Integer-Leverage Optuna Hybrid Decision

Generated: `2026-05-28T12:17:14.082473Z`

## Method correction

- This artifact replaces the prior coarse 5% grid as the optimization decision surface.
- Optuna/TPESampler tunes v3.5- and v3.6-style hybrid parameters, matching the existing `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` pattern.
- v3.5 mapping: warmup-learned default profile + rolling return/error weights + high-vol boost + bias/exposure dampening.
- v3.6 mapping: v3.5 mechanics plus online default-profile refresh from rolling scores.
- Standard live refit inputs: Optuna learns/optimizes on train only; validation is a recent holdout for scoring/selection; after selection, the frozen parameter set is final-refit on train+validation.
- No locked test/OOS split is reserved for live final refit; live runtime uses frozen artifacts and still blocks real-money execution.

## Standard live-refit split

- train: `2025-01-01T00:00:00Z` → `2026-04-02T10:00:00Z`
- validation: `2026-04-02T11:00:00Z` → `2026-05-28T10:00:00Z`
- selection fit inputs: `['train']`
- final refit inputs: `['train', 'validation']`
- locked-OOS enabled: `False`

## Comparison

| Profile | Version | Optimizer | Weights/avg TV weights | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | RPT T/V/OOS bps | Paper candidate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_mdd12_gross5` | `source_profile` | `source_profile` | `{"balanced_mdd12_gross5": 1.0}` | 1.00x | 139.4112% | 2.7436% | 0.0000% | 7.2003% | 0.0000% | 46.32/9.11/0.00 | true |
| `growth_mdd20_gross8` | `source_profile` | `source_profile` | `{"growth_mdd20_gross8": 1.0}` | 3.90x | 554.0523% | 36.0101% | 0.0000% | 9.2371% | 0.0000% | 54.98/28.83/0.00 | true |
| `aggressive_mdd30_gross10_shadow` | `source_profile` | `source_profile` | `{"aggressive_mdd30_gross10_shadow": 1.0}` | 4.90x | 1157.0744% | 34.6545% | 0.0000% | 12.3630% | 0.0000% | 82.63/22.40/0.00 | true |
| `hybrid_mdd20_three_profile_blend` | `grid_baseline` | `coarse_5pct_grid_baseline_not_selected_by_this_runner` | `{"aggressive_mdd30_gross10_shadow": 0.1, "balanced_mdd12_gross5": 0.1, "growth_mdd20_gross8": 0.8}` | 3.71x | 558.2484% | 32.2913% | 0.0000% | 9.1032% | 0.0000% | 57.18/27.27/0.00 | false |
| `hybrid_v3_5_optuna_three_profile_blend` | `v3_5` | `optuna_tpe` | `{"aggressive_mdd30_gross10_shadow": 0.9142036046718318, "balanced_mdd12_gross5": 0.04274976787133024, "growth_mdd20_gross8": 0.04304662745683793}` | 4.69x | 3447.4699% | 38.0717% | 0.0000% | 7.4789% | 0.0000% | 368.22/36.77/0.00 | true |
| `hybrid_v3_6_optuna_three_profile_blend` | `v3_6` | `optuna_tpe` | `{"aggressive_mdd30_gross10_shadow": 0.25513669666835037, "balanced_mdd12_gross5": 0.34683758458980146, "growth_mdd20_gross8": 0.39802571874184817}` | 3.15x | 553.3179% | 24.8750% | 0.0000% | 5.7827% | 0.0000% | 83.63/33.75/0.00 | true |

## Selected Optuna hybrid

- profile: `hybrid_v3_5_optuna_three_profile_blend`
- hybrid version: `v3_5`
- avg train+validation weights: `{"aggressive_mdd30_gross10_shadow": 0.9142036046718318, "balanced_mdd12_gross5": 0.04274976787133024, "growth_mdd20_gross8": 0.04304662745683793}`
- final weights: `{"aggressive_mdd30_gross10_shadow": 0.914203604671838, "balanced_mdd12_gross5": 0.041881045072758326, "growth_mdd20_gross8": 0.04391535025540365}`
- train/validation/OOS report-only: `3447.4699%` / `38.0717%` / `0.0000%`
- validation MDD / OOS MDD: `7.4789%` / `0.0000%`
- RPT bps train/validation/OOS: `368.22` / `36.77` / `0.00`
- selection reasons: `[]`
- report-only OOS gate reasons: `[]`

## Governance

- primary round-trip cost bps: `10.0`
- return-per-turnover threshold bps: `10.0`
- ready_for_real: `false`
- real_money_execution: `false`
- real_execution_allowed: `false`
- locked-OOS used for selection: `False`
