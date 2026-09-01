# PRD — Profit moonshot liquidation-aware strict validation — 2026-05-11

## Objective
Validate whether the current-base sleeve tuple at integer `5x` is deployable under a strict liquidation-aware Binance USDⓈ-M-style margin model. Preserve the green handoff head `77f10d54174628c24f1a6bbba34a74505a2a40b5` and performance baseline `02f4520cf906f48089b8852c2651a0f1e4bd0c1c` as historical comparison anchors.

## Scope
- Enforce strict promotion semantics: any liquidation count `> 0` in train/validation/OOS or any split minimum margin buffer `<= 0` blocks promoted success.
- Keep locked-OOS out of selection. OOS may only be report-only/gate-only after train/validation selection.
- Compare the liquidation-aware current-base reference (`2.3427334297703024x`) with forced integer `5x` and the `1x..6x` integer grid.
- Record margin model assumptions, fees/slippage/funding/stress/liquidation reserves, split liquidation counts, minimum margin buffer, and minimum margin ratio.
- Write result/handoff artifacts under `.omx/plans`, `docs/`, `.omx/notepad.md`, and `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_*`.

## Non-goals
- Do not promote liquidation-tolerant candidates for this task.
- Do not use locked-OOS for ranking, leverage fitting, or selection.
- Do not reset history to the baseline commits; preserve them as ancestors and evidence anchors.

## Acceptance criteria
- Tests prove intrabar adverse high/low breaches produce liquidation events.
- Tests prove split liquidation count, minimum margin buffer, and minimum margin ratio are recorded.
- Tests prove locked-OOS is selection-excluded.
- Tests prove liquidation count `> 0` or margin buffer `<= 0` blocks promoted success and the validator cannot override that with tolerance metadata.
- Tests prove missing liquidation/margin evidence cannot pass strict promoted-success validation; absence of evidence is not evidence of zero liquidation.
- Strict replay completes under 8 GiB.
- Final artifact states whether `5x` is deployable and why.
- Targeted tests, full pytest, ruff, compileall, and `git diff --check` pass before commit/push.
- Lore commit is pushed to `private/main` and GitHub Actions `ci` + `private-ci` are green.
