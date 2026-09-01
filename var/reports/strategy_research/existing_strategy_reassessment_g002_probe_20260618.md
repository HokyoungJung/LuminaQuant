# Existing strategy reassessment smoke manifest

- generated: `2026-06-18T13:29:28.977510Z`
- strategy rows: `3`
- skipped: `0`
- tier counts: `{'live_default': 1, 'research_only': 2}`
- selection input: `registry metadata + future train/validation bounded smoke only`
- locked-OOS/current top evidence: `report context only`
- full-WF promotion list: `empty until bounded smoke + strict gates pass`
- real-money: `false`

## Strategy audit rows

| Strategy | Tier | Runnable | Required TF | Required features | Promotion | Audit flags | Rejection reasons |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `MeanReversionStdStrategy` | `live_default` | yes | `none` | `none` | `not_promoted_requires_smoke_and_full_wf` | `live_default_registry_tier_requires_recheck_before_new_promotion` | `fresh_forward_required_before_promotion`, `requires_bounded_smoke_metrics`, `requires_full_wf_metrics` |
| `DiversifiedMultiFactorEnsembleStrategy` | `research_only` | yes | `none` | `funding_rate`, `open_interest`, `mark_price`, `index_price` | `not_promoted_requires_smoke_and_full_wf` | `requires_feature_lookup`, `research_only_tier` | `fresh_forward_required_before_promotion`, `requires_bounded_smoke_metrics`, `requires_full_wf_metrics`, `research_only_until_cost_realistic_wf_passes` |
| `CrossSectionalFundingMomentumCarryStrategy` | `research_only` | yes | `none` | `funding_rate` | `not_promoted_requires_smoke_and_full_wf` | `requires_feature_lookup`, `research_only_tier` | `fresh_forward_required_before_promotion`, `requires_bounded_smoke_metrics`, `requires_full_wf_metrics`, `research_only_until_cost_realistic_wf_passes` |

## Current benchmark/control evidence

| Role | Model | Clean | OOS comp | MDD | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `raw_shadow_rank_1` | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` | no | 79.42% | 27.69% | `shadow_only_requires_fresh_forward` |
| `risk_trimmed_shadow` | `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2` | no | 64.42% | 18.46% | `preferred_shadow_watch_if_drawdown_matters` |
| `best_clean_paper_baseline` | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | yes | 34.39% | 27.69% | `paper_baseline_only_not_real_money` |
| `lower_mdd_clean_scaled` | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | yes | 29.65% | 23.59% | `clean_but_cash_heavy_not_hard_stop_pass` |
| `best_clean_under_15pct_mdd_bucket` | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate` | yes | 12.97% | 10.08% | `defensive_benchmark_not_core` |

## Promotion outputs

- survivor list: `[]`
- full-WF promotion list: `[]`
- reason: `bounded smoke metrics have not been run yet`
