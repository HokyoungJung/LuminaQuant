# Test spec — Profit moonshot liquidation-aware strict validation — 2026-05-11

## Unit/behavior locks
1. `tests/test_profit_moonshot_liquidation_aware_validation.py`
   - intrabar low/high crossing liquidation threshold emits liquidation event.
   - `_split_margin_summary` records `liquidation_count`, `minimum_margin_buffer`, `minimum_margin_ratio`, and buffer positivity.
   - `_select_train_validation_leverage` ignores locked-OOS and only uses train/validation safety/score.
   - `_liquidation_safe_for_promotion` is strict by default: count `>0` or buffer `<=0` blocks success.
   - CLI parser defaults to strict zero-liquidation allowance.
   - If a tolerance object is constructed for research diagnostics, strict promotion still remains the default behavior unless explicitly and separately invoked by older artifacts.

2. `tests/test_profit_moonshot_pass_under_8gb_validator.py` and/or liquidation validator tests
   - promoted candidates with liquidation count `>0` fail the candidate return-quality contract even when artifact metadata advertises a positive `allowed_total_liquidations`.
   - promoted candidates with non-positive margin buffer fail.
   - promoted candidates missing liquidation count or minimum margin buffer evidence fail strict validation.

## Replay/evidence
- Run `scripts/research/run_profit_moonshot_liquidation_aware_validation.py` into a strict `liquidation_aware_*` output directory with `--allowed-total-liquidations 0 --allowed-split-liquidations 0` (or strict defaults).
- Confirm policy fields: `promotion_requires_liquidation_count_zero=true`, `uses_locked_oos_for_selection=false`, and `liquidation_tolerance.allowed_total_liquidations=0`.
- Confirm train/validation/OOS split metrics and memory summary under 8 GiB.

## Required verification commands
- Targeted tests for liquidation validation, portfolio tuning, and validator.
- Full `uv run --extra dev pytest -q` with memory evidence.
- `uv run --extra dev ruff check .`
- `python3 -m compileall -q src scripts tests`
- `git diff --check`
- GitHub Actions `ci` and `private-ci` green on pushed Lore commit.
