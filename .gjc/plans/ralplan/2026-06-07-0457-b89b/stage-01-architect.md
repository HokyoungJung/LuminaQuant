## Summary
Architectural status: WATCH. Code-review recommendation: APPROVE. The Planner artifact is architecturally sound for research-only new-alpha discovery because it separates train/validation invention, candidate freeze, locked-OOS report/gate, and fresh-forward validation, while keeping real-money blocked. The only material watch item is lane separation: execution must not let post-OOS/lagged-shadow hypotheses become clean material by label, artifact dependency, or search-space backfilling.

## Analysis
- Spec compliance: The Planner directly encodes the requested doctrine: locked-OOS report/gate-only, post-OOS selector distrust, no nested/hybrid contamination, fresh-forward requirement, and bounded runtime/optimization. Its recommended Option A is the only option that can create new evidence without treating the rejected +85.91% selector or +61.40% lagged-shadow result as promotion evidence.
- No post-OOS trust: scripts/research/run_alpha_zoo_clean_meta_selector_research.py:1-9 explicitly describes the selector as post-OOS, shadow-only, fresh-forward-required, and non-promotable. Lines 231-266 record uses_locked_oos_for_selector_grid_ranking=true, post_oos_research_variant=true, requires_fresh_forward_shadow=true, and clean_promotion_eligible=false. The Planner correctly quarantines this lineage.
- No locked-OOS objective/selection: src/lumina_quant/optimization/search_policy.py:17-22 centralizes false locked-OOS search flags, while scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py:949-952 and 1120-1127 keep objective/selection train+validation-only. The Planner freeze-then-attach-OOS sequence matches these boundaries.
- No nested contamination: scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py:1815-1850 disables old nested hybrid passes. Lines 3084-3114 exclude non-leaf, post-OOS, fresh-forward-required, and source-post-OOS rows from clean downstream material. Lines 4812-4849 compute clean_promotion_eligible from those contamination flags. The Planner no-nested acceptance criteria are aligned with this architecture.
- Fresh-forward requirement: Existing code has shadow/fresh-forward flags and promotion hard-stops, but the actual fresh-forward shadow stage is primarily a planned addition. That is acceptable for this planning stage, provided execution persists fresh-forward window identity, source candidate freeze hash, and a no-feedback assertion before any paper/testnet recommendation.
- Bounded compute/optimization: search_policy.py:61-101 requires bounded grid justification/caps and audit metadata. Optuna artifacts record trial count, search space, and policy at run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py:1047-1065. The Planner smoke/full/fresh-forward sequence and equivalence-check requirement are consistent with the repository optimization boundary.

## Root Cause
The underlying defect being planned around is evidence contamination, not lack of model complexity: OOS-reviewed selector families and nested portfolio rows can look profitable while being epistemically unusable for promotion. The right fix is protocol architecture: immutable pre-registration, leaf-only material, frozen train/validation selection, OOS report-only attachment, and fresh-forward confirmation.

## Findings
- LOW: Planner artifact /home/hoky/Quants-agent/LuminaQuant/.gjc/plans/ralplan/2026-06-07-0457-b89b/stage-01-planner.md, Option A / sequencing. The plan allows post-OOS selector and lagged-shadow artifacts to seed hypotheses. Impact: even without row-level current-OOS reads, human-selected search spaces can inherit OOS-mining bias and later be mislabeled clean. Fix: require any hypothesis derived from rejected post-OOS artifacts to remain post_oos_research_variant=true and clean_promotion_eligible=false until a new pre-registered lineage is frozen and validated on a strictly later fresh-forward window.
- LOW: Planner artifact, file-level change guidance. Extending run_alpha_zoo_69_asset_monthly_refit_walkforward.py could further mix clean discovery, shadow diagnostics, recompute, and post-OOS research responsibilities. Impact: boundary dilution and harder auditability. Fix: prefer a thin focused clean-new-alpha runner once changes exceed small schema additions, but reuse monthly-refit split/evaluation primitives and shared search_policy rather than duplicating split logic.

## Recommendations
1. Approve Option A with the lane-separation constraint above.
2. Make pre_registered_search_space_sha256, candidate_freeze_sha256, fresh_forward_window_id, fresh_forward_no_feedback=true, post_oos_selector_trusted=false, and contamination flags mandatory before any generated artifact can be considered for paper/testnet routing.
3. Critic should specifically attack hypothesis-derived search spaces, OOS tie-breaks, and fresh-forward feedback loops before implementation approval.
4. Execution should route bounded implementation slices only after consensus approval. No product-source edits or tests/builds occurred in this Architect stage.

## Architectural Status
WATCH

## Code Review Recommendation
APPROVE

## Trade-offs
- Extend monthly-refit runner: reuses existing split, aggregation, no-nested, and lagged-shadow audit machinery. Cost: risks mixing clean and post-OOS/shadow responsibilities in one large file. Synthesis: accept for small schema/freeze additions only.
- New focused clean discovery runner: clearer clean-vs-shadow boundary and easier artifact audit. Cost: risks duplicated split logic and drift from tested monthly-refit semantics. Synthesis: preferred if discovery lane is more than a narrow extension; consume shared/monthly-refit primitives.
- Purist greenfield forward-only protocol: strongest antithesis because it minimizes human OOS-mining bias from rejected artifacts. Cost: slowest path, discards useful audited infrastructure, and may delay evidence. Synthesis: not required now, but any post-OOS-inspired hypothesis must remain quarantined until fresh-forward.

Strongest steelman antithesis: because the team has already seen the +85.91% and +61.40% historical OOS outcomes, even train/validation-only formulas chosen after that review can encode OOS-mined priors. A maximally pure program would ignore those artifacts entirely, freeze an economically motivated leaf-only search universe, and wait for future data. The Planner synthesis is acceptable because it makes those artifacts shadow hypotheses only and requires fresh-forward confirmation before promotion.
