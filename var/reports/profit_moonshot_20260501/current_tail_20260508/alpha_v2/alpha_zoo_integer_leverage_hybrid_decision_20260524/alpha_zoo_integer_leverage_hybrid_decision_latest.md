# Integer-Leverage Three-Profile Hybrid Decision

Generated: `2026-05-24T13:28:52.336644Z`

## Method

- Source profiles: strict balanced, relaxed growth, relaxed aggressive from the frozen integer-leverage artifact.
- Reconstructs the 10bps-costed profile PnL streams; no order execution.
- Searches three-profile weights on a 5% grid with at least 10% allocated to each source profile.
- Selects the hybrid using train+validation score only with a 20% validation-MDD target.
- locked-OOS is attached after the hybrid weights are frozen as gate/report-only evidence.

## Four-profile comparison

| Profile | Kind | Weights | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | RPT T/V/OOS bps | Paper candidate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_mdd12_gross5` | source_integer_leverage_profile | `{"balanced_mdd12_gross5": 1.0}` | 1.00x | 74.6685% | 33.2153% | 5.5300% | 11.6134% | 7.2003% | 30.91/57.02/22.21 | true |
| `growth_mdd20_gross8` | source_integer_leverage_profile | `{"growth_mdd20_gross8": 1.0}` | 3.90x | 262.3353% | 71.6291% | 23.3695% | 19.9983% | 9.2371% | 32.16/37.72/23.35 | true |
| `aggressive_mdd30_gross10_shadow` | source_integer_leverage_profile | `{"aggressive_mdd30_gross10_shadow": 1.0}` | 4.90x | 438.4462% | 117.4976% | 27.5772% | 29.4044% | 12.3630% | 38.81/44.16/21.87 | true |
| `hybrid_mdd20_three_profile_blend` | hybrid_train_validation_selected | `{"aggressive_mdd30_gross10_shadow": 0.15, "balanced_mdd12_gross5": 0.15, "growth_mdd20_gross8": 0.7}` | 3.61x | 262.3642% | 72.5692% | 21.2977% | 19.9330% | 9.0718% | 33.78/39.96/22.97 | true |

## Selected hybrid

- profile: `hybrid_mdd20_three_profile_blend`
- weights: `{"aggressive_mdd30_gross10_shadow": 0.15, "balanced_mdd12_gross5": 0.15, "growth_mdd20_gross8": 0.7}`
- train/validation/OOS report-only: `262.3642%` / `72.5692%` / `21.2977%`
- validation MDD / OOS MDD: `19.9330%` / `9.0718%`
- report-only OOS gate reasons: `[]`

## Governance

- primary round-trip cost bps: `10.0`
- return-per-turnover threshold bps: `10.0`
- ready_for_real: `false`
- real_money_execution: `false`
- real_execution_allowed: `false`
- locked-OOS used for selection: `False`
