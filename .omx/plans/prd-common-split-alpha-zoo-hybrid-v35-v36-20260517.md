# PRD — Common-split Alpha Zoo vs Hybrid v3.5/v3.6 (2026-05-17)

## Objective
Recompare existing Alpha Zoo strict 6x and fixed-input hybrid v3.5/v3.6 Optuna candidates on one explicit common split, preserving `private/main@80a557c133930f51748ec20c4e582aa0d6f678de` as baseline parent.

## Split contract
- train: `2025-01-01T00:00:00Z` through `2025-12-31T23:00:00Z`
- validation: `2026-01-01T00:00:00Z` through `2026-02-28T23:00:00Z`
- locked-OOS: `2026-03-01T00:00:00Z` through `2026-05-06T23:00:00Z`

## Functional requirements
- Apply the common split explicitly in runner artifacts; old Alpha Zoo fractional split is historical-only.
- Rebuild Alpha Zoo screen, calibration, carry-forward replay, reselected replay, strict integer leverage grid, and diagnostic 5x/6x lane on the common split.
- Re-run fixed-input hybrid v3.5/v3.6 Optuna with exactly `A0 + P0 + E0 + S1 + S2 + S3 + S4` and train+validation-only objective/selection.
- Treat locked-OOS as gate/report-only after candidate freeze; any objective/pruning/selection use invalidates live promotion.
- Keep strict deploy lane separate from diagnostic nonfatal high-leverage lane.
- Report split min/max timestamps, return, MDD, return/MDD diagnostic, Sharpe, Sortino, smart Sortino, Calmar, trade count, liquidation count, minimum margin buffer, deployable success, and rejection reasons.

## Output artifacts
- `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`
- `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py`
- `docs/session_handoff_20260517_common_split_alpha_zoo_hybrid_v35_v36.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/`

## Acceptance
Alpha Zoo/hybrid candidates are fairly comparable on the common split, locked-OOS non-use is auditable, memory remains below 8 GiB, required verification passes, Lore commit is pushed to `private/main`, and `ci/private-ci` is green.
