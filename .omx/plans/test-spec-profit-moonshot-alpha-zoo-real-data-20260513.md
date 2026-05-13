# Test Spec — Profit Moonshot Alpha Zoo Real-Data Calibration (2026-05-13)

## Targeted regression tests
- `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q`
- Add/extend tests when adapter/reporting behavior changes, especially source coverage, calendar guard, locked-OOS exclusion, and calibration bucket blocking.
- Include an artifact-level calibration test proving locked-OOS ledger rows are excluded from bucket estimation, with `locked_oos_calibration_record_count=0` in the output.
- Include an adapter/reporting test proving real current-tail input reports observed/imputed coverage and fails closed on missing required OHLCV coverage instead of silently treating defaults as real data.

## Focused integration tests
- `uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q`

## Full verification
- `uv run --extra dev pytest -q`
- `uv run --extra dev ruff check .`
- `uv run --extra dev python -m compileall -q src scripts tests`
- `git diff --check`
- `git diff --cached --check`

## Research validation commands
- Run the Alpha Zoo real-data screen against current-tail data and write reports under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_*`.
- Run candidate ledger/triple-barrier generation and edge calibration using train/validation only.
- Run stateful calibrated replay and liquidation-aware validation with integer leverage 1x..6x.

## Evidence checklist
- Strategy/factor/calibration actually used.
- Train/validation selection provenance and candidate freeze evidence.
- Locked-OOS excluded from selection and used gate/report-only after freeze.
- Locked-OOS excluded from edge calibration inputs by record counts, not metadata-only assertion.
- Real-data source coverage reports observed/imputed columns per source/symbol.
- Strict zero-liquidation lane results.
- Diagnostic 5x/6x nonfatal lane results.
- Current-base/calendar teacher excluded from promotion target.
- Peak RSS < 8 GiB.
- Updated plan/research note/handoff/report artifact paths.
- Commit hash and GitHub Actions `ci`/`private-ci` links.
