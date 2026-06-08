# Planner Stage 1 - LuminaQuant Clean New-Alpha Discovery After Post-OOS Selector Rejection

## Summary
Plan a research-only execution effort that discovers genuinely new alphas without trusting post-OOS meta-selector/grid outcomes. The current clean mechanical anchor remains the latest acceptable candidate at +34.39% OOS / 27.69% MDD. The post-OOS meta-selector +85.91% artifact and raw lagged-shadow +61.40% result are not acceptable as clean promotion evidence; they may be treated only as rejected/shadow hypotheses requiring fresh-forward confirmation. Real-money remains blocked.

Repository evidence inspected:
- scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py already defines monthly folds, train/validation/locked_oos splits, clean promotion reports, no-nested candidate filters, lagged-shadow router quarantine, and candidate/fold aggregation.
- scripts/research/run_alpha_zoo_clean_meta_selector_research.py is explicitly post-OOS selector research and must not be used as authoritative final evidence.
- src/lumina_quant/optimization/search_policy.py centralizes LOCKED_OOS_SEARCH_FLAGS, bounded-grid metadata, and Optuna policy helpers.
- scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py has train+validation-only objective comments and recent warmup/hot-path optimization surface.
- docs/session_handoff_20260604_nested_hybrid_eval_perf.md records no-nested cleanup and evaluator hot-path optimization, and demotes nested/post-OOS/fresh-forward-only winners.

Task classification: broad research initiative with performance-optimization track. Stop at pending approval; no source edits or execution in this planner stage.

## RALPLAN-DR short mode
### Principles
1. Locked-OOS is report/gate only after a train+validation freeze; it is never an objective, selector, pruning, threshold, grid, Optuna, correlation, or parameter-fitting input.
2. New alpha must be new evidence, not a better re-labeling of post-OOS selector artifacts; post-OOS and lagged-shadow results are hypotheses only.
3. No nested/hybrid contamination: hybrid/selector/bridge/meta rows may be outputs, but cannot become downstream material for another clean candidate.
4. Train/validation/fresh-forward evidence is required before paper/testnet consideration; OOS alone cannot promote.
5. Runtime work must preserve metric equivalence and search-policy metadata; speedups without checksum/equivalence evidence are not actionable.

### Top decision drivers
1. Research trust: user explicitly rejected post-OOS selector trust, so discovery must create pre-registered, train/validation-frozen candidates plus fresh-forward shadow.
2. Evidence quality versus runtime: broad full walk-forward is expensive, so bounded smoke, cache/perf improvements, and artifact invariants must unlock more trials without weakening hygiene.
3. Promotion safety: real-money blocked; paper/testnet only after strict split, sample, cost, MDD, liquidation, and contamination gates.

### Viable options
#### Option A - Clean discovery ladder + fresh-forward shadow + measured runtime optimization (recommended)
Pros:
- Directly answers the rejection by separating invention, train/validation freeze, locked-OOS gate, and fresh-forward evidence.
- Reuses existing monthly walk-forward runner, no-nested policy, search-policy metadata, and optimization hot-path surfaces.
- Allows post-OOS selector and lagged-shadow artifacts to seed hypotheses only, not promotion authority.
- Produces CI-testable contamination/audit invariants.
Cons:
- Slower to produce headline returns than accepting the +85.91% selector artifact.
- May produce no promotion candidate; that is acceptable if rejection reasons are precise.
Invalidation rationale for rejected alternatives: Option A is chosen because it is the only option that can yield genuinely new, clean evidence under the user's trust constraint while still improving runtime.

#### Option B - Accept post-OOS selector output as shadow benchmark only and optimize execution around it
Pros:
- Fastest path to compare against high reported returns.
- Useful for defining target behavior and failure diagnostics.
Cons:
- Does not satisfy the user's core objection; post-OOS meta-selector cannot be clean evidence.
- High risk of reintroducing OOS-mined thresholds through artifacts, grids, or selector labels.
Invalidation: reject as execution recommendation; retain only as a quarantined benchmark/hypothesis source with clean_promotion_eligible=false and requires_fresh_forward_shadow=true.

#### Option C - Full greenfield alpha engine rewrite before research reruns
Pros:
- Could improve runtime and architecture long-term.
- Opportunity to enforce split hygiene from first principles.
Cons:
- High implementation risk, delays discovery, and may duplicate existing Alpha Zoo/monthly-refit infrastructure.
- Does not by itself produce better clean alpha evidence.
Invalidation: reject for this cycle; only consider after measured bottlenecks show existing runners cannot meet throughput needs.

## In scope
- New or revised research runners/artifacts for clean alpha discovery that pre-register search spaces and freeze candidates from train+validation before locked-OOS gate/report.
- Explicit quarantine of run_alpha_zoo_clean_meta_selector_research.py outputs and any lagged-shadow/post-OOS artifacts as non-clean evidence.
- Fresh-forward shadow stage using data strictly after the frozen research window; no selector/threshold revision from fresh-forward results.
- Runtime optimization in existing backtest/evaluation hot paths where equivalence can be proven.
- Tests and artifact invariants for locked-OOS exclusion, no nested material, freshness/provenance, and performance equivalence.

## Out of scope
- Real-money execution, live order routing, credentials, or ready_for_real=true.
- Promotion based on +85.91% post-OOS meta-selector or +61.40% raw lagged-shadow alone.
- Nested/hybrid stacking or use of selector/meta/hybrid rows as material inputs to downstream clean candidates.
- OOS-mined grids, thresholds, correlations, Optuna enqueue choices, pruning, or tie-breaks.
- Full architecture rewrite unless Architect later blocks the recommended path.

## File-level changes likely required
### Research runners
- scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py
  - Add or strengthen clean discovery lanes that emit pre-freeze candidate manifests, selection inputs, contamination flags, and fresh-forward-required statuses.
  - Ensure any new candidate family uses leaf-only material and preserves no-nested filters already present.
  - Add fresh-forward shadow artifact mode or a companion handoff hook that never feeds back into selection.
- scripts/research/run_alpha_zoo_clean_meta_selector_research.py
  - Re-label outputs as post-OOS/shadow-only if not already explicit in artifact fields; block any clean_promotion_eligible=true from this lineage.
- scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py
  - Continue warmup/hot-path optimization only with train+validation objective preservation and split-policy metadata.
  - Add benchmarks/provenance if execution needs more throughput for new searches.
- Potential new focused runner under scripts/research, e.g. run_alpha_zoo_clean_new_alpha_discovery.py, only if extending monthly-refit directly would overmix post-OOS and clean discovery responsibilities.

### Shared optimization/runtime
- src/lumina_quant/optimization/search_policy.py
  - Extend policy payload with explicit fresh_forward_role, post_oos_selector_trusted=false, pre_registered_search_space_sha256, and contamination flags if needed.
- src/lumina_quant/optimization/fast_eval.py, frozen_dataset.py, native_backend.py, walkers.py, threading_control.py
  - Touch only for measured bottlenecks with equivalence checks.

### Tests
- tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
  - Add invariants for clean discovery freeze, no nested material, locked-OOS gate/report-only, and fresh-forward non-feedback.
- tests/test_alpha_zoo_clean_meta_selector_research.py
  - Assert selector outputs remain shadow/non-clean and cannot promote even if reported OOS is high.
- tests/test_alpha_zoo_integer_leverage_optuna_hybrid_decision.py
  - Preserve train+validation-only Optuna objective and warmup optimization equivalence.
- Add focused tests for any new runner: artifact schema, policy flags, pre-registration hash, candidate freeze, fresh-forward gate, and real-money false.

### Docs/artifacts
- docs/research_note/research_note.md and/or a new docs/session_handoff new_alpha note after execution.
- Artifact path under var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/ with a distinct clean-new-alpha directory and latest/timestamped JSON/MD/CSV/log outputs.

## Sequencing and dependencies
1. Freeze research doctrine in artifact schema before adding alpha breadth.
   - Define required top-level fields: selection_policy, locked_oos_policy, post_oos_selector_trusted=false, fresh_forward_policy, pre_registered_search_space, nested_material_policy, real_money_execution=false, ready_for_real=false, source_lineage, output_paths, candidate_freeze_sha256.
   - Define row fields: family, label, source lineage, train metrics, validation metrics, locked-OOS metrics, fresh-forward metrics if available, split trade counts, cost/RPT, MDD, liquidation/wipeout, contamination flags, rejection reasons.
2. Build clean candidate discovery lanes using only train+validation.
   - Prefer leaf strategy families and simple auditable market-state rules already present in Alpha Zoo infrastructure.
   - Keep post-OOS meta-selector winners only in a quarantined benchmark table.
   - Require bounded grids/Optuna configs to be recorded through optimization_search_policy_payload or equivalent metadata.
3. Add locked-OOS gate/report stage after freeze.
   - Candidate ranking must be immutable before locked-OOS metrics are attached.
   - Locked-OOS can only reject or label; it cannot re-rank or update thresholds.
4. Add fresh-forward shadow stage.
   - Use data strictly later than the frozen locked-OOS/report window.
   - Report pass/fail, stability, turnover, cost, MDD, and sample counts.
   - Do not feed fresh-forward outcomes back into same-run selection; any repair requires a new pre-registered run lineage.
5. Continue runtime optimization in parallel only where behavior is locked by tests/checksums.
   - Benchmark monthly-refit candidate evaluation, period metrics, Optuna warmup, return-stream assembly, and report rendering.
   - Implement only measured changes; every speed claim needs equivalence checksum/metric preservation.
6. Generate artifacts only after implementation approval.
   - First run: bounded smoke with low trials/folds to validate artifact invariants.
   - Second run: full clean discovery under resource limits.
   - Third run: fresh-forward shadow after candidate freeze.
7. Review path after Planner approval.
   - Architect review should validate split/lineage architecture, contamination boundaries, and whether to extend monthly-refit or create a new runner.
   - Critic review should follow Architect to verify that locked-OOS, post-OOS selector, nested/hybrid, and fresh-forward feedback boundaries are airtight.

## Acceptance criteria
- No artifact or row uses locked-OOS for objective, selection, pruning, parameter fitting, threshold tuning, correlation selection, enqueue choice, or tie-break.
- Post-OOS meta-selector +85.91% and raw lagged-shadow +61.40% are present only as quarantined/shadow references, never as clean promotion evidence.
- Candidate ranking is frozen from train+validation before locked-OOS metrics are attached; artifact includes freeze hash/provenance.
- Fresh-forward evidence is required for any paper/testnet recommendation and cannot alter same-run candidate selection.
- No nested/hybrid contamination: downstream clean material excludes hybrid, selector, bridge, meta, dynamic switch, validation selector, MDD gate/portfolio, selected_optuna, selected_train_validation_legal, static_guarded, and hybrid-like labels/profile IDs.
- Primary promotion gate uses realistic cost policy already established in repo conventions, with 10bps round-trip as the main assumption unless Architect explicitly approves another baseline.
- Any deployability field remains paper/testnet only: ready_for_real=false, real_money_execution=false.
- Runtime optimization claims include benchmark command, baseline/current timings, and equivalence checksum or exact metric preservation.
- The latest clean mechanical candidate (+34.39% OOS / 27.69% MDD) remains the clean anchor until a new candidate passes train, validation, locked-OOS report gate, and fresh-forward shadow.

## Verification plan for execution phase
Do not run in this planner stage. After approval, execution verification should include:
- Targeted pytest for modified/new runner tests, especially locked-OOS exclusion and non-promotion of post-OOS selector artifacts.
- Artifact invariant inspection script or tests confirming all policy flags and lineage fields.
- Smoke artifact generation with minimal folds/trials and JSON/MD/CSV existence checks.
- Full clean discovery artifact generation under memory guard after smoke passes.
- Fresh-forward shadow artifact generation using frozen candidate manifest.
- Benchmark for any runtime change with metric checksum/equivalence.
- Ruff/format/static checks for changed files.

## Risks and mitigations
- Risk: OOS leakage re-enters through grid names, selector labels, Optuna enqueue trials, or correlation filters.
  - Mitigation: mandatory policy fields, tests that perturb locked-OOS and assert unchanged selection, and row-level contamination flags.
- Risk: Post-OOS artifacts remain tempting because headline returns are high.
  - Mitigation: artifact sections must separate clean rankings, quarantined shadow references, and fresh-forward-only hypotheses.
- Risk: Fresh-forward becomes a second OOS mining loop.
  - Mitigation: fresh-forward can only validate or reject the frozen candidate; repairs require a new pre-registered lineage and cannot reuse the same fresh-forward window as optimizer input.
- Risk: Runtime optimization changes metrics.
  - Mitigation: checksum/equivalence tests before accepting any speedup.
- Risk: No new candidate beats the clean anchor.
  - Mitigation: no-promotion is a valid outcome if rejection reasons are exact and the next pre-registered search direction is documented.

## Handoff guidance
Recommended option: Option A. Send to Architect review before implementation. Architect should decide whether clean-new-alpha discovery belongs as an extension of run_alpha_zoo_69_asset_monthly_refit_walkforward.py or as a separate focused runner that consumes monthly-refit leaf streams. Critic review should follow Architect to verify that locked-OOS, post-OOS selector, nested/hybrid, and fresh-forward feedback boundaries are airtight. Use executor only after approval for bounded implementation slices; do not use team/ultragoal unless the approved execution expands into a durable multi-session research campaign.
