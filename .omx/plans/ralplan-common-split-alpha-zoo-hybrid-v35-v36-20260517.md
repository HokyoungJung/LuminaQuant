# RALPLAN — Common-split Alpha Zoo vs Hybrid v3.5/v3.6 (2026-05-17)

## Decision
Proceed with an additive, manifest-driven common-split runner and report bundle. Preserve `private/main@80a557c133930f51748ec20c4e582aa0d6f678de` as the baseline parent.

## Principles
1. Fixed common split is the only comparison authority.
2. Locked-OOS is gate/report-only after freeze; never objective/pruning/selection/tie-break.
3. Strict deploy lane is separate from diagnostic 5x/6x.
4. Hybrid inputs remain exactly `A0 + P0 + E0 + S1 + S2 + S3 + S4` with no hybrid-inside-hybrid.
5. Keep changes additive and reuse existing replay/calibration/Optuna logic.

## Exact common split
- train: `2025-01-01T00:00:00Z` through `2025-12-31T23:00:00Z` inclusive
- validation: `2026-01-01T00:00:00Z` through `2026-02-28T23:00:00Z` inclusive
- locked_oos: `2026-03-01T00:00:00Z` through `2026-05-06T23:00:00Z` inclusive

## Implementation surfaces
- Add new thin orchestration runner: `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`.
- Add minimal tests: `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py` plus existing policy tests if needed.
- Runner may use existing primitives directly instead of `run_crypto_fx_alpha_zoo_screen.build_screen_payload` because that path currently uses fractional splits.
- Reuse without duplicating core algorithms:
  - `src/lumina_quant/alpha_zoo/crypto_fx_factors.py`
  - `src/lumina_quant/research/crypto_fx_alpha_zoo_real_data.py`
  - `scripts/research/calibrate_crypto_fx_edges.py`
  - `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
  - `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`

## Required comparison rows
1. Historical old-split Alpha Zoo best: reference-only.
2. Common-split Alpha Zoo carry-forward strict 6x: old selected params replayed on common split.
3. Common-split Alpha Zoo reselected replay: grid selection train/validation-only.
4. Common-split hybrid v3.5 Optuna.
5. Common-split hybrid v3.6 Optuna.
6. Strict zero-liquidation integer leverage 1x..6x.
7. Diagnostic nonfatal 5x/6x lane.

## Acceptance criteria
- `common_split_manifest` includes exact boundaries, row counts, timestamp-index hash, input artifact hashes, baseline parent, commands/provenance, output paths.
- Manifest row counts/hash match Alpha screen/ledger/calibration/replay and hybrid split periods.
- Split-bounded labels prevent train/validation forward labels from crossing into later splits.
- OOS poison tests do not change train/validation selected factors/calibrations or Optuna best result under same seed.
- Each candidate reports timestamp min/max, return, MDD, return/MDD diagnostic, Sharpe, Sortino, smart Sortino, Calmar, trade count, liquidation count, min margin buffer, deployable_success, rejection reasons.
- Any locked-OOS use for objective/pruning/selection/tie-break invalidates live promotion.
- Memory peak RSS < 8192 MiB.

## Verification path
Run the exact user-required commands:
- `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q`
- `uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q`
- `uv run --extra dev pytest -q`
- `uv run --extra dev ruff check .`
- `uv run --extra dev python -m compileall -q src scripts tests`
- `git diff --check`
- `git diff --cached --check`
Then run final artifact audit, Lore commit, push to `private/main`, and verify GitHub Actions `ci/private-ci` green. If `gh` auth/workflow visibility is blocked after push, record exact blocker and do not claim green.

## 2026-05-17 KST addendum — mixed allocator integrated margin replay

Follow-up resolved the prior hybrid live-promotion blocker. `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` now performs post-freeze account-level integrated margin replay for the mixed A0+P0+E0+S1+S2+S3+S4 allocator and emits split liquidation count/minimum margin buffer. `run_common_split_alpha_zoo_hybrid_v35_v36.py` now propagates hybrid deployability instead of hardcoding the missing-replay blocker.

Updated decision: hybrid v3.5/v3.6 are live-promotion-capable under strict integrated margin evidence, but Alpha Zoo strict 6x remains the common-split performance leader. Research history/source ledger remains unchanged because no new global source family or chronology/source-ledger input was added.
