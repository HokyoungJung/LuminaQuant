# Test spec — Profit moonshot liquidation-aware 5x validation — 2026-05-10

## Required pre-implementation tests
1. Intrabar adverse breach:
   - Long position with low below liquidation threshold emits a liquidation event.
   - Short position with high above liquidation threshold emits a liquidation event.
2. Split metrics:
   - Validation summarizes per split liquidation count, minimum margin buffer, and minimum margin ratio.
3. Selection integrity:
   - Integer grid selection uses train/validation liquidation and margin evidence only.
   - Poisoned OOS evidence can fail/report gate but cannot change selected leverage.
4. Promotion hard gate:
   - Candidate/gates with liquidation count > 0 or minimum margin buffer <= 0 cannot become a promoted success.
   - Final validator rejects an otherwise strong promoted candidate when liquidation evidence is unsafe.

## Verification commands
- Targeted tests: `uv run --extra dev pytest -q tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py`
- Full pytest: `uv run --extra dev pytest -q`
- Ruff: `uv run --extra dev ruff check .`
- Compileall: `python3 -m compileall -q src scripts tests`
- Whitespace: `git diff --check`

## Heavy replay command
`/usr/bin/time -v uv run --extra dev python scripts/research/run_profit_moonshot_liquidation_aware_validation.py --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_5x_20260510`
