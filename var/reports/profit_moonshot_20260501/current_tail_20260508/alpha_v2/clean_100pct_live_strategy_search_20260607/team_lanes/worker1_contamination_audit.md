# Worker-1 contamination / eligibility audit

- Generated UTC: `2026-06-07T07:02:35Z`
- Worker: `worker-1`
- Task: `1` / active Ultragoal evidence: `G002-contamination-and-eligibility-audit`
- Output directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/team_lanes`
- Real-money approval blocked: **true**
- Coordination protocol: coordinated - handoff, dirty repo read-only boundary, lane output ownership, and Ultragoal non-mutation checked.
- Boundary: `/home/hoky/Quants-agent/LuminaQuant` was inspected read-only; writes are only in this clean worktree lane directory.

## Source handoff

- Worktree relative handoff `.omx/plans/ralplan-handoff-clean-100pct-live-strategy-search-20260607.json`: **missing**.
- Leader clean copy used: `/home/hoky/Quants-agent/LuminaQuant-clean-100pct-20260607/.omx/plans/ralplan-handoff-clean-100pct-live-strategy-search-20260607.json` (sha256 `032e5917cd234fae5d5ca484f65d80e978c0cc550bda11c0e7ebf1c01c27956c`)
- Main dirty read-only copy also exists: `/home/hoky/Quants-agent/LuminaQuant/.omx/plans/ralplan-handoff-clean-100pct-live-strategy-search-20260607.json`

## Verdict

No inspected existing artifact is acceptable current real-money or historical-locked-OOS promotion evidence. `clean_input_meta_selector` is the only inspected 100%+ annualized label candidate, but it is explicitly **rejected for current promotion** and capped at `shadow_freeze_only` because its selector-grid ranking used historical locked-OOS context. Strict/no-leak and 85-symbol artifacts can remain controls; new-alpha discovery artifacts are rejected for current promotion.

## Hard-gate summary

- `no_nested_oos_mining`: prior locked-OOS/history is contamination/control context only; no existing artifact may promote a candidate in this lane.
- `execution_cost_gate`: 10bps evidence exists for controls; strict no-leak has 20bps diagnostic; 15bps and paper fill telemetry are missing/blocking for real money.
- `theory_plausibility_gate`: family plausibility remains a downstream theory-lane dependency; no arbitrary parameter/OOS threshold promotion accepted here.
- `100pct_reporting_label`: post-evaluation label only; not selector objective, Optuna target, rerun trigger, or promotion filter.

## Classification table

| Artifact / candidate | Label | Current promotion decision | Metrics snapshot | Blockers | Evidence |
| --- | --- | --- | --- | --- | --- |
| `clean_input_meta_selector` | `shadow_freeze_only` | rejected | oos_compounded_return_pct=85.91, annualized_oos_return_approx_pct=110.46, monthly_equity_mdd_pct=6.32, max_bar_oos_mdd_pct=19.29, hit_rate=5/10 | post_oos_selector_grid_ranking_uses_historical_locked_oos; fresh_forward_required_before_promotion; paper_fill_telemetry_absent; 15bps_stress_artifact_absent | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.json`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.md`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json` |
| `strict_no_leak_best_single_10bps` | `eligible_control` | no_real_money; paper/control only | 10bps_total_return_pct=54.56, 10bps_mdd_pct=30.63, 10bps_sharpe=1.26, 10bps_pf=1.2096, 10bps_positive_months=6/10, 20bps_total_return_pct=27.1, 20bps_mdd_pct=43.63 | drawdown too high for real sleeve; 20bps stress MDD tail risk; 15bps/paper-fill telemetry missing; eligible universe only 10/85 symbols | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_strict_no_leak_20260606/strict_no_leak_selector_latest.md`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_strict_no_leak_20260606/strict_no_leak_selector_latest.json` |
| `dynamic_conviction_switch_85_symbol_baseline` | `eligible_control` | paper baseline/monitor only | oos_compounded_return_pct=34.39, annualized_oos_return_approx_pct=42.57, max_oos_mdd_pct=27.69, hit_rate=3/10, cost_bps=10.0 | below 100pct reporting label; sparse positive folds; high MDD; no 15bps/20bps/paper fill telemetry in this artifact | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.json`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.md` |
| `relaxed_efficiency_hybrid_v3_5_69_asset_incumbent` | `eligible_control` | historical incumbent/control only; no new promotion from this audit | oos_compounded_return_pct=156.03, annualized_oos_return_approx_pct=209.0, max_oos_mdd_pct=19.75, hit_rate=5/10, cost_bps=10.0 | existing historical locked-OOS artifact cannot be current promotion evidence; no fresh-forward/paper fill telemetry for real money; 15bps/20bps hard-gate proof incomplete here | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.json`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.md` |
| `profile_optuna_69_asset_clean_non_nested_full_eval` | `eligible_control` | paper/control only | top_oos_compounded_return_pct=10.05, top_annualized_return_approx_pct=12.18, max_oos_mdd_pct=19.2, hit_rate=5/10, cost_bps=10.0 | below 100pct reporting label; promotability=false; no fresh-forward/paper fill telemetry | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.json`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.md` |
| `lagged_shadow_leaf_router_and_demoted_post_oos_variants` | `shadow_freeze_only` | rejected for current promotion | example_oos_compounded_return_pct=61.4, example_annualized_return_approx_pct=77.62, example_max_bar_mdd_pct=29.13, example_hit_rate=4/10 | post-OOS family/variant status; fresh-forward required; 10bps proxy only | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.md`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606.json` |
| `clean_new_alpha_discovery_full_feature_bounded_smoke` | `rejected` | rejected as current alpha/promotion source | full_annualized_pct=3.01, feature_bounded_annualized_pct=-0.57, smoke_annualized_pct=7.89 | low/negative returns; explicit clean_promotion_eligible=false; same-window promotion blocked; future hypothesis source only | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_smoke/clean_new_alpha_discovery_latest.json` |
| `deep_research_live_status_and_best_strategy_conclusion` | `diagnostic_only` | diagnostic summary only | decision=no_real_money_deployment_now, fresh_search_status=no fresh clean candidate beats incumbent | not primary manifest-bound run evidence; real-money blocked by fresh-forward and telemetry gaps | `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_live_status_20260606/current_verdict_latest.md`; `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_best_strategy_clean_oos_20260607/deep_research_best_strategy_clean_oos_20260607.md` |

## Artifact inventory

- `.omx/plans/ralplan-handoff-clean-100pct-live-strategy-search-20260607.json` (planning_or_handoff, sha256 `032e5917cd234fae…`)
- `.omx/plans/prd-clean-100pct-live-strategy-search-20260607.md` (planning_or_handoff, sha256 `babc17edcd67440c…`)
- `.omx/plans/test-spec-clean-100pct-live-strategy-search-20260607.md` (planning_or_handoff, sha256 `07171613f65c7687…`)
- `.omx/context/clean-100pct-live-strategy-search-20260607T064554Z.md` (planning_or_handoff, sha256 `0a0b1d7e930eb13b…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/execution_context_g001.json` (existing_evidence_readonly, sha256 `de39c58cf861317c…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/execution_context_g001.md` (existing_evidence_readonly, sha256 `fdb0e578ed7d8bd5…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.json` (existing_evidence_readonly, sha256 `0371c1d0578fa148…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.md` (existing_evidence_readonly, sha256 `2d79fe35dacf5225…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json` (existing_evidence_readonly, sha256 `bd26dcd511633764…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_strict_no_leak_20260606/strict_no_leak_selector_latest.json` (existing_evidence_readonly, sha256 `03cbea6ea0ff5a20…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_strict_no_leak_20260606/strict_no_leak_selector_latest.md` (existing_evidence_readonly, sha256 `fef9b2f8cbdd41b1…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.json` (existing_evidence_readonly, sha256 `cd7f4ced043cf406…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.md` (existing_evidence_readonly, sha256 `e627bb859c3fdfd6…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.json` (existing_evidence_readonly, sha256 `5338aa0cc6ef66aa…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.md` (existing_evidence_readonly, sha256 `5e2f34500096de4f…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.json` (existing_evidence_readonly, sha256 `39687d7a210098d6…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.md` (existing_evidence_readonly, sha256 `13c2786d0047cc5f…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.json` (existing_evidence_readonly, sha256 `83094ae79f946d81…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.md` (existing_evidence_readonly, sha256 `5c31d19ad39e749c…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json` (existing_evidence_readonly, sha256 `99cb2b44152146d7…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.md` (existing_evidence_readonly, sha256 `8bd4bd119b2f3fc7…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json` (existing_evidence_readonly, sha256 `2a9e639401c5822e…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.md` (existing_evidence_readonly, sha256 `e389a45a1f72860f…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_smoke/clean_new_alpha_discovery_latest.json` (existing_evidence_readonly, sha256 `c44a38af8618954c…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_smoke/clean_new_alpha_discovery_latest.md` (existing_evidence_readonly, sha256 `2e02394b28a14abc…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_live_status_20260606/current_verdict_latest.md` (existing_evidence_readonly, sha256 `d6318f44a128f11a…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_live_status_20260606/live_search_status_latest.md` (existing_evidence_readonly, sha256 `ae821a833b91af93…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_best_strategy_clean_oos_20260607/deep_research_best_strategy_clean_oos_20260607.json` (existing_evidence_readonly, sha256 `34e7848ceab8a24f…`)
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_best_strategy_clean_oos_20260607/deep_research_best_strategy_clean_oos_20260607.md` (existing_evidence_readonly, sha256 `ab3d165ca565b0b5…`)

## Subagent evidence

- Subagents spawned: 2 (`Repository map probe` `019ea0df-0f47-70f3-ad19-798a4471baee` / Helmholtz; `Review probe` `019ea0e0-16b6-71c1-8ee1-3e420230e12d` / Hegel).
- Subagent model: requested `gpt-5.4-mini` per task contract; available spawn surface exposed no model or `agent_type` parameter, so `agent_type: explore` was passed in the child prompts.
- Serial searches before spawn: 2.
- Findings integrated: map probe key paths, artifact classes, boundary risks, and minimum JSON/MD fields. Review probe timed out on first wait and is pending final check before task transition.

## Git status snapshots

### Main dirty repo (read-only)

```text
## private-main...private/main [ahead 1]
 M docs/research_note/research_note.md
 M scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py
 M tests/test_alpha_zoo_clean_new_alpha_discovery.py
 M var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json
 M var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.md
 M var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json
 M var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.md
?? .gjc/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_bestlane_rerun_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_efficiency_specialist_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_fresh_69_relaxed_seed_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_fresh_69_rematch_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_fresh_69_teacher_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_fresh_85_clean_focus_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_fresh_85_full_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_fresh_85_teacher_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_leaf_optuna_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_live_status_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_probe_69_teacher_1fold_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_probe_69_teacher_3fold_20260606/
?? var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_relaxed_specialist_20260606/
```

### Worker worktree before artifact write

```text
## HEAD (no branch)
```

## Stop condition for this lane

Task 1 evidence is ready for leader integration after final verification of JSON parse, path boundary, and a final review-probe/transition update. Real-money approval remains blocked.
