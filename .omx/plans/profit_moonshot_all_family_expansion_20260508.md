# Profit moonshot all-family expansion plan/result — 2026-05-08

## Objective
Include new strategy families in the latest-tail replay/tuning path, improve return evidence beyond Sharpe-only reporting, and preserve <8 GiB + locked-OOS report-only gates.

## Changes made
- Added strategy families in `scripts/research/replay_profit_moonshot_fresh_start.py`: residual momentum, cross-sectional Sharpe reversal, funding carry momentum, adaptive trend fade, compression breakout fade.
- Expanded TRX calendar take-profit grid around Optuna-supported parameters.
- Added family-balanced candidate pool support via `--family-quota` in `scripts/research/tune_profit_moonshot_fresh_portfolio.py`.
- Added regression tests for new signals/spec families and family-balanced sleeve pool.

## Result
- Best passing portfolio OOS return: 1.2181%; train 3.5993%; validation 2.6755%; OOS MDD 0.1662%; OOS Sharpe 6.7264.
- New non-calendar families: evaluated, zero promoted.
- Calendar rotation remains the only passing family.

## Verification
- 1193 passed in 269.27s (0:04:29) via uv run --extra dev pytest -q
- All checks passed via uv run --extra dev ruff check .
- python3 -m compileall -q src scripts tests passed
- git diff --check passed

## Artifacts
- `var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/passing_candidate_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/SESSION_HANDOFF_20260508_ALL_FAMILY_EXPANSION.md`
- `docs/session_handoff_20260508_profit_moonshot_all_family_expansion.md`
