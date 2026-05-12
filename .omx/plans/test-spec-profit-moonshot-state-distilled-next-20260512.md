# Test Spec — Profit Moonshot State-Distilled Non-Calendar Next Pass

Generated: 2026-05-12 KST

## Tests to add before implementation
- `tests/test_profit_moonshot_fresh_start_replay.py`
  - valid non-calendar family specs must have no `calendar_*` month fields, no fixed `entry_days_of_month`, and no fixed `entry_hours`.
  - new strategy family specs must be generated with names/families for crowded leadership unwind v2, funding/OI exhaustion reversal, beta residual reversion, and dispersion compression state.
  - representative signal tests for each family must use identical datetimes/months where applicable and depend only on market-state arrays.
- `tests/test_profit_moonshot_liquidation_aware_validation.py`
  - candidate/leverage selection ranks on train/validation only and records nested/provenance metadata.
  - strict lane and diagnostic nonfatal lane are represented separately.
  - strict promotion fails if any split liquidation count > 0 or min margin buffer <= 0.
  - diagnostic nonfatal lane records liquidation count/event drawdown/equity loss/recovery and is not live promotion.

## Required verification commands
1. Targeted tests:
   - `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py`
2. Focused profit-moonshot suite:
   - `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_strategy_validity_audit.py`
3. Static checks:
   - `uv run --extra dev ruff check .`
   - `python3 -m compileall -q scripts tests src`
   - `git diff --check`
4. Full suite:
   - `/usr/bin/time -v uv run --extra dev pytest -q`
5. Research execution evidence:
   - fresh replay with new non-calendar family allowlist.
   - liquidation-aware validation with 1x-6x retune/evaluation, strict and diagnostic lanes.
6. Delivery:
   - Lore commit.
   - `git push private HEAD:main` or configured private/main push.
   - GitHub Actions `ci` and `private-ci` green for pushed commit.

## RALPLAN iteration 2 added tests/checks
- Factory-wide allowlist test: `VALID_NON_CALENDAR_FAMILIES_NEXT` from `_candidate_specs` must have empty `calendar_long_months`, `calendar_short_months`, `entry_days_of_month`, `entry_hours`, `calendar_long_symbol`, `calendar_short_symbol`, and calendar-veto fields equal to zero.
- Signal placebo tests: representative new non-calendar signals must return identical decisions for identical state in different months.
- Provenance artifact checks:
  - replay payload includes `selection_provenance` with train/validation selector fields and locked-OOS report-only status.
  - replay payload includes `calendar_proxy_diagnostics` with month placebo, blocked-time concentration, rolling/walk-forward status, and future-holdout policy fields.
  - liquidation payload separates `strict_deploy_lane` from `diagnostic_nonfatal_lane`.
- Metric thresholds are explicit in tests where gate payloads are built: MDD <= 25%, Sharpe/Sortino/smart Sortino/Calmar >= 1.0, return and return/MDD beat reference for deployable success.
