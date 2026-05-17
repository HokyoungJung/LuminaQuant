# Context snapshot — common-split Alpha Zoo vs hybrid v3.5/v3.6

## Task statement
Run `$ralplan $team $ralph` continuation for `/home/hoky/Quants-agent/LuminaQuant`: preserve private/main baseline `80a557c133930f51748ec20c4e582aa0d6f678de`, then fairly compare existing Alpha Zoo strict 6x (`CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit`) against fixed-input hybrid v3.5/v3.6 Optuna candidates on the same common split.

## Desired outcome
New common-split artifacts under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/`, updated research note/handoff/notepad/plan as needed, all required local validations green, Lore commit pushed to `private/main`, and GitHub Actions `ci/private-ci` green with commit hash/link reported.

## Known facts/evidence
- Repository synced to `private-main` and reset to `private/main` at `80a557c133930f51748ec20c4e582aa0d6f678de`; working tree clean at intake.
- Existing Alpha Zoo split was older/fractional and must be treated historical only.
- Existing fixed-input hybrid v3.5/v3.6 artifact already uses fresh common split periods but must be rerun with common-split Alpha Zoo calibration/selection inputs for a fair comparison.
- OMX explore mapped core files: `scripts/research/replay_crypto_fx_alpha_zoo_state.py`, `scripts/research/run_crypto_fx_alpha_zoo_screen.py`, `scripts/research/calibrate_crypto_fx_edges.py`, `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`, `src/lumina_quant/alpha_zoo/crypto_fx_factors.py`, `src/lumina_quant/research/crypto_fx_alpha_zoo_real_data.py`.

## Constraints
- Common split fixed exactly: train `2025-01-01T00:00:00Z`–`2025-12-31T23:00:00Z`; validation `2026-01-01T00:00:00Z`–`2026-02-28T23:00:00Z`; locked-OOS `2026-03-01T00:00:00Z`–`2026-05-06T23:00:00Z`.
- Locked-OOS cannot be used for objective/pruning/selection/tie-break before candidate freeze.
- No calendar/month/day/hour entry rule. Current-base/calendar tuple is hypothesis_reference_only.
- Return/MDD diagnostic/report-only, not a hard promotion gate.
- Strict deploy lane separate from diagnostic nonfatal 5x/6x lane.
- Strict lane promotion disallowed if liquidation_count > 0 or min margin buffer <= 0. OOS MDD <=25%; OOS return must beat invalid current-base reference; Sharpe/Sortino/smart Sortino/Calmar must be positive/good; memory <8 GiB.
- Hybrid inputs fixed to A0 + P0 + E0 + S1 + S2 + S3 + S4; no hybrid-inside-hybrid or same-family strategy output input.
- Hybrid v3.5/v3.6 implementation must remain faithful to `/home/hoky/DeepLearning/ensemble_strategies`: v3.5 fixed default + rolling weights/high-vol boost + Optuna; v3.6 v3.5 core plus online dynamic default-candidate refresh only.

## Unknowns/open questions
- Whether common-split Alpha Zoo reselected candidate remains `alpha_zoo_conservative_exit` or another grid row after recalibration/reselection.
- Whether common-split hybrid Optuna remains non-promotable due missing integrated margin replay and/or OOS gates.
- Whether global research history/source ledger needs regeneration; likely no if no new source family is added, but final artifact must state decision.

## Likely codebase touchpoints
- New/updated common-split runner under `scripts/research/`.
- Existing hybrid runner `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` may be reused with common Alpha Zoo replay/calibration paths.
- Tests under `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` and Alpha Zoo state/validation tests may need additions for common-split invariants.
- Docs: `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md`, new `docs/session_handoff_20260517_common_split_alpha_zoo_hybrid_v35_v36.md`, `.omx/notepad.md`, maybe `.omx/plans/*`.
