# Alpha Zoo clean-input meta-selector research

- generated: `2026-06-07T05:21:54.412161Z`
- source: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606.json`
- freeze manifest: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json`
- fold choice inputs: `train + validation only`
- locked-OOS use: `grid ranking/report only`, not per-fold selection
- evidence class cap: `shadow-freeze-only`
- status: `post-OOS research / fresh-forward shadow required / real-money false`

## Best selector

- deployment label: `shadow-freeze-only`
- OOS comp: `85.91%`
- annualized approx: `110.46%`
- monthly equity MDD: `6.32%`
- max bar OOS MDD: `19.29%`
- positive folds: `5/10`
- Sharpe approx: `1.28`
- params: `{"calmar_weight": 0.05, "family_group": "dynamic_strict_relaxed", "mdd_weight": 2.0, "return_weight": 8.0, "spike_penalty_weight": 2.0, "train_mdd_cap": 0.4, "validation_mdd_cap": 0.25, "validation_spike_cap": 0.15}`

## Fold choices

| Fold | Candidate | Family | Selector score | OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: |
| `2025-09` | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | 2.4159 | -6.10% | 6.64% |
| `2025-10` | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | 0.6366 | -0.23% | 6.35% |
| `2025-11` | `dynamic_conviction_switch:t1.00_strict_fallback_val_ret02_calmar80_gate_val_mdd15_scaled` | `dynamic_conviction_switch` | 3.0299 | 23.53% | 19.29% |
| `2025-12` | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | 2.9714 | -3.35% | 3.01% |
| `2026-01` | `relaxed_efficiency:selected_optuna` | `relaxed_efficiency` | 6.6317 | 5.10% | 6.77% |
| `2026-02` | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | 5.1372 | -1.03% | 11.71% |
| `2026-03` | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | 1.6673 | 0.39% | 3.26% |
| `2026-04` | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | 2.0295 | -4.46% | 10.06% |
| `2026-05` | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | 6.1716 | 1.09% | 15.42% |
| `2026-06` | `relaxed_efficiency:selected_train_validation_legal` | `relaxed_efficiency` | 7.5976 | 64.80% | 6.69% |

## Guardrail

This artifact is not clean promotion evidence. It is a bounded way to identify a selector formula to freeze before future fresh-forward evaluation.
The grid ranking uses the historical locked-OOS window diagnostically, so the label is capped at `shadow-freeze-only` even though each fold choice uses only train/validation fields.
