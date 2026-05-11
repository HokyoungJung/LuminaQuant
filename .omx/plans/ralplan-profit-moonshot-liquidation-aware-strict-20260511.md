# RALPLAN — Profit moonshot liquidation-aware strict validation — 2026-05-11

## RALPLAN-DR summary
### Principles
1. Strict safety beats apparent performance: liquidation count must be zero for promotion.
2. Preserve selection integrity: train/validation decide; locked-OOS is report-only/gate-only.
3. Preserve lineage: `77f10d5` and `02f4520` stay as immutable comparison anchors.
4. Prefer minimal reversible changes over broad research restarts.
5. Verification evidence must be fresh and command-backed.

### Decision drivers
1. User explicitly requires `liquidation count > 0` or `margin buffer <= 0` to block promoted success.
2. Current code default has drifted to a tiny-liquidation-tolerant retune path.
3. Existing strict artifact already found forced current-base `5x` unsafe; current work must prevent later tolerant defaults from contradicting that policy.

### Viable options
- **Option A — restore strict defaults while keeping tolerant diagnostics explicit.**
  - Pros: satisfies current request, minimal code churn, keeps historical tolerant artifacts readable.
  - Cons: older tolerant reruns must pass explicit nonzero tolerance flags.
- **Option B — delete all tolerance support.**
  - Pros: strongest policy clarity.
  - Cons: wider diff, could break historical diagnostics/tests unrelated to this task.
- **Option C — leave code unchanged and only document strict result.**
  - Pros: no code risk.
  - Cons: fails the current acceptance criterion because future default runs can still promote with liquidation tolerance.

Chosen option: **A**.

## ADR
- **Decision:** Make liquidation-aware promotion strict by default and make the pass-under-8GiB validator reject positive liquidation evidence for promoted candidates regardless of tolerance metadata.
- **Drivers:** strict user acceptance criteria, preservation of train/validation selection, baseline lineage, low-risk patch size.
- **Alternatives considered:** deleting tolerance entirely (too broad), documentation-only (insufficient).
- **Why chosen:** aligns default behavior with deployment safety while preserving older research-diagnostic capability behind explicit flags/tests.
- **Consequences:** future tolerant experiments must be clearly opt-in and cannot masquerade as strict deployment promotion.
- **Follow-ups:** if exact Binance bracket credentials become available, replace scalar fallback with symbol bracket lookup while keeping the strict gate.

## Execution plan
1. Add/adjust tests first for strict CLI defaults and validator rejection of any positive liquidation count despite tolerance metadata.
2. Patch `run_profit_moonshot_liquidation_aware_validation.py` defaults/policy to strict zero-liquidation promotion.
3. Patch `validate_profit_moonshot_pass_under_8gb.py` to reject positive or missing liquidation/margin evidence for promoted success regardless of artifact allowance.
4. Run strict replay into `var/reports/.../alpha_v2/liquidation_aware_strict_20260511/` and inspect policy/results.
5. Write handoff artifacts and notepad entry.
6. Run targeted/full verification under memory guard, then Lore commit/push/CI check.

## Available agent types roster
- `executor` — implementation and targeted test updates (medium reasoning).
- `test-engineer` — regression coverage and full verification plan (medium reasoning).
- `architect` — read-only final sign-off (high reasoning).
- `verifier` — CI/push evidence audit (high reasoning).

## Team staffing guidance
- Recommended coordinated team: `omx team 2:executor "strict liquidation-aware default/gate audit for profit moonshot; one lane reviews tests, one lane reviews implementation risks"`.
- Delivery lane: leader owns code edits to avoid shared-file conflicts.
- Evidence lane: worker/reviewer checks acceptance criteria and reports gaps.
- Final sign-off lane: architect/verifier after local green.

## Team verification path
- `omx team status <team>` until workers terminal, then shutdown only after `pending=0`, `in_progress=0`, `failed=0`.
- Local gates: targeted pytest → strict replay → full pytest → ruff → compileall → diff-check → git/CI evidence.

## Goal-mode follow-up suggestions
- `$performance-goal` is the best fit if future work optimizes memory/runtime of liquidation-aware replay.
- `$ultragoal` is appropriate for durable multi-step implementation tracking.
- `$autoresearch-goal` is only appropriate for new market/research hypothesis validation, not this strict implementation gate.
