# Pending Approval Plan — Clean New-Alpha Discovery Without Post-OOS Trust

Status: pending approval
Run: 2026-06-07-0457-b89b

## Decision
Proceed with Option A: clean discovery ladder + fresh-forward shadow + measured runtime optimization. Do not treat post-OOS meta-selector (+85.91%) or lagged-shadow (+61.40%) artifacts as clean evidence. Use them only as quarantined hypothesis references. The implementation phase must discover or pre-register genuinely new alpha search spaces using train/validation only, then attach locked-OOS as report/gate only, then require strictly later fresh-forward shadow before any paper/testnet recommendation.

## Drivers
1. The user explicitly rejected post-OOS trust; historical OOS-derived selector formulas cannot be promoted.
2. Current clean anchor is weak (+34.39% OOS / 27.69% MDD), so new alpha discovery is necessary.
3. Full WF runtime is expensive; optimization must be measured and equivalence-checked, not used to weaken split hygiene.

## Principles
- Locked-OOS is never objective, selector, pruning, fitting, threshold, tie-break, enqueue, or correlation input.
- Post-OOS and lagged-shadow artifacts are hypotheses only.
- No nested/hybrid/selector/meta rows as clean downstream material.
- Candidate freeze must precede locked-OOS attachment.
- Fresh-forward validates or rejects only; it does not repair the same lineage.
- ready_for_real and real_money_execution remain false.

## Chosen execution shape
1. Define artifact/schema guards: selection policy, locked-OOS policy, post_oos_selector_trusted=false, fresh_forward_policy, pre_registered_search_space_sha256, contamination flags, freeze hash, lineage.
2. Implement a clean new-alpha discovery lane, preferably a focused runner if extending the monthly runner would mix post-OOS research and clean discovery.
3. Keep post-OOS meta-selector and lagged-shadow outputs in quarantined benchmark tables only.
4. Add tests that perturb locked-OOS and prove selection is unchanged.
5. Add no-nested material tests for hybrid/selector/bridge/meta/dynamic/selected/static labels.
6. Generate smoke artifact first, then full clean discovery under memory/time limits.
7. Freeze candidate manifest, then run fresh-forward shadow on strictly later data without feedback into selection.
8. Continue runtime optimization only with benchmark plus checksum/equivalence proof.

## Likely files
- `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`
- `scripts/research/run_alpha_zoo_clean_meta_selector_research.py`
- `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py`
- possible new `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`
- `src/lumina_quant/optimization/search_policy.py`
- `tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py`
- `tests/test_alpha_zoo_clean_meta_selector_research.py`
- `tests/test_alpha_zoo_integer_leverage_optuna_hybrid_decision.py`
- new focused tests for clean new-alpha discovery artifact invariants

## Acceptance criteria
- No artifact or row uses locked-OOS for objective, selection, pruning, fitting, threshold tuning, correlation selection, enqueue choice, or tie-break.
- Candidate ranking is frozen from train+validation before locked-OOS metrics are attached; freeze hash/provenance is recorded.
- Post-OOS meta-selector and lagged-shadow outputs are never clean_promotion_eligible.
- Fresh-forward evidence is required and cannot alter same-run selection.
- No nested/hybrid contamination in clean material.
- Runtime speedups include benchmark command, baseline/current timing, and equivalence checksum or exact metric preservation.
- New result either beats the clean anchor honestly or records no-promotion with precise rejection reasons.

## Alternatives considered
### Option B — Accept post-OOS selector output as shadow benchmark only and optimize around it
Rejected for promotion because it does not satisfy the user's trust objection. Retained only as quarantined benchmark/hypothesis source.

### Option C — Full greenfield alpha engine rewrite before reruns
Rejected for this cycle because it delays discovery and duplicates existing WF infrastructure. Reconsider only if measured bottlenecks prove current runners cannot meet throughput needs.

## Architect review
Architecture status: WATCH. Verdict: APPROVE. Constraint: keep post-OOS/lagged-shadow-inspired hypotheses quarantined until separately pre-registered and frozen before a strictly later fresh-forward window.

## Critic review
Verdict: OKAY. Constraint: runtime optimization only with checksum/equivalence proof; never label historical OOS selector results clean.

## Consequences
- Implementation must be slower to approve but more trustworthy.
- High historical OOS numbers remain useful but non-promotable.
- No-promotion remains an acceptable result.

## Follow-ups
- Execute through an approved execution skill after user approval.
- Prefer ultragoal for durable goal tracking; team only if tmux-based worker coordination is explicitly needed.
