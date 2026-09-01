# Test spec — Common-split Alpha Zoo vs Hybrid v3.5/v3.6 (2026-05-17)

## Regression tests
- Exact calendar boundary filtering and split labels.
- Split-bounded forward labels must not cross train/validation/OOS boundaries.
- Factor screening and edge calibration must be invariant to locked-OOS poisoning.
- Calibration locked-OOS record count must remain zero.
- Timestamp index hash must use unique split timestamps.
- Hybrid A0 reconstruction must apply the Alpha replay common-split manifest instead of fractional splits.
- Alpha row extraction must use top-level strict integer grid results.
- Hybrid candidate rows must expose fixed split periods as effective active windows.

## Artifact checks
- Manifest periods/row counts match the common split.
- Old Alpha split rows are historical-only and non-promotable.
- Carry-forward and reselected Alpha rows are distinct provenance rows.
- Hybrid Optuna reports `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_selection=false`.
- Strict integer leverage table covers 1x..6x; strict lane rejects liquidation count >0 or min buffer <=0.
- Diagnostic 5x/6x lane has `promotion_allowed=false` and remains separate.
- Peak RSS <8192 MiB.

## Required verification commands
```bash
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

Post-deslop verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_post_deslop_20260517T054855Z.log`.
