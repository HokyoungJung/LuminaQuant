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

