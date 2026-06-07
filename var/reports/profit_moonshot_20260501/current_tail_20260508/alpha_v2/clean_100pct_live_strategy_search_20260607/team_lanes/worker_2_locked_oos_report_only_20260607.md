# Task 3 — Locked OOS report-only lane

generated_at_utc: `2026-06-07T06:58:00.986100Z`
team: `clean-100pct-live-tar-4a549153`
worker: `worker-2`

## Decision
- Locked OOS remains `report/gate only after train-validation freeze`.
- Locked OOS may not select, tune, prune, fit parameters, trigger reruns, or promote a candidate.
- The `100%+ annualized` threshold is a post-evaluation reporting label only, not a selector objective.
- Real-money approval remains blocked without fresh-forward and paper fill telemetry evidence.
- Task 3 produced a report-only lane artifact; no code or runner behavior was changed.

## Source artifacts checked
- primary_clean_discovery: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json`
- feature_bounded_clean_discovery: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json`
- shadow_meta_selector: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_research_latest.json`
- shadow_meta_selector_freeze_manifest: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json`

## Primary clean discovery report-only metrics
- generated: `2026-06-07T06:09:00.709499Z`
- pre-registered search-space hash: `4a6fee0f540f5d9ce15158beaf6b7c91ad89600cb5d76e1c4bfa0e33008b81b7`
- selection inputs: `['train', 'validation']`
- locked-OOS selection/objective/fitting/pruning flags: `False` / `False` / `False` / `False`
- OOS compounded / annualized approx: `2.51%` / `3.01%`
- OOS positive folds: `5/10`
- OOS min return / max MDD / latest return: `-6.47%` / `8.77%` / `8.11%`
- label blockers: `['continuous_position_state_across_split_boundaries', 'fresh_forward_required_before_promotion']`

## Feature-bounded clean discovery report-only metrics
- generated: `2026-06-07T06:13:13.852816Z`
- OOS compounded / annualized approx: `-0.24%` / `-0.57%`
- OOS positive folds: `3/5`
- label blockers: `['continuous_position_state_across_split_boundaries', 'fresh_forward_required_before_promotion']`

## 100%+ shadow example remains non-promotable
- meta-selector annualized OOS report-only label: `110.46%`
- deployment label / evidence class: `shadow-freeze-only` / `shadow-freeze-only`
- locked OOS used for selector grid ranking: `True`
- blockers: `['post_oos_selector_grid_ranking_uses_historical_locked_oos', 'fresh_forward_required_before_promotion']`
- Interpretation: a 100%+ report-only historical label does not become promotion evidence when post-OOS grid ranking contaminated the selector; it is capped at shadow-freeze-only.

## Code/test policy evidence
- `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py` — Docstring and optimization payload rank train/validation first, then attach locked-OOS report/gate fields; flags set uses_locked_oos_for_selection/objective/parameter_fitting/pruning false.
- `tests/test_alpha_zoo_clean_new_alpha_discovery.py` — Tests mutate locked-OOS report fields and assert score/selection are unchanged; synthetic run asserts locked-OOS selection false and real money false.
- `tests/test_alpha_zoo_clean_meta_selector_research.py` — Tests show selector fold choice ignores locked OOS but post-OOS selector-grid ranking demotes to shadow-freeze-only and bans locked_oos in manifest selection fields.
- `src/lumina_quant/alpha_zoo/optuna_hybrid_config.py` — Loader validation raises if Optuna/integer artifacts set locked-OOS discovery/objective/fitting/pruning/selection flags to anything other than false.
- `src/lumina_quant/alpha_zoo/crypto_fx_factors.py` — Factor screen ranks only train/validation and returns locked_oos as report-only split stats with uses_locked_oos_for_selection false.

## Handoff alignment
- Required sequence item covered: `locked_oos_report_only`.
- Final integrated report should still state `found`/`not_found` only after contamination, manifest/search, cost, theory, and label lanes are integrated.
- This lane does not approve real money and does not mutate `.omx/ultragoal`.

## Subagent probe integrated
- Subagents spawned: `1` (`019ea0dd-b8f9-73c2-bed9-8ab99b7117be`, change-slice probe).
- Subagent model requested: `gpt-5.4-mini` (tool schema inherited current model; no explicit model field exposed).
- Findings integrated: safest slice is report-only; code edits unnecessary/risky; worker-2 should only write worker-owned lane files; source code, `.omx/ultragoal`, docs, and peer lane files are shared-file hazards.
- Additional anchors integrated: `src/lumina_quant/optimization/search_policy.py`, `tests/test_optimization_search_policy.py`, `src/lumina_quant/alpha_zoo/factor_card.py`, and `scripts/research/run_alpha_zoo_clean_meta_selector_research.py`.

## Additional policy anchors
- `src/lumina_quant/optimization/search_policy.py` — `LOCKED_OOS_SEARCH_FLAGS` centralizes `uses_locked_oos_for_selection/objective/pruning/parameter_fitting=false`.
- `tests/test_optimization_search_policy.py` — Optuna policy payload defaults are asserted to keep all locked-OOS search flags false.
- `src/lumina_quant/alpha_zoo/factor_card.py` — factor cards fail closed with `locked_oos_used_for_selection` if selected splits include `locked_oos`/`oos`; provenance records `locked_oos_role=gate_report_only`.
- `scripts/research/run_alpha_zoo_clean_meta_selector_research.py` — the 110%+ shadow example explicitly records `uses_locked_oos_for_selector_grid_ranking=true`, `post_oos_research_variant=true`, `ready_for_real=false`, and `deployment_label=shadow-freeze-only`.

## Scan/caveat
- Focused scan found no current clean-process `100pct`/`100%` annualized selector or Optuna objective; annualized fields in clean discovery/meta-selector are aggregate/report rendering.
- Caveat: historical/control config `configs/research/bridge-protocol-manifest-oos-oracle-hybrid-v1-20260602.json` contains OOS promotion thresholds; this lane does not claim those historical configs are current clean promotion inputs.
