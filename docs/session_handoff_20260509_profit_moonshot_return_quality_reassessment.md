# Session handoff — profit moonshot return-quality reassessment — 2026-05-09

## Current status
The previous alpha-v2 target-budget candidate is downgraded to historical/diagnostic. It is no longer a success because the user clarified that MDD up to ~25% is acceptable, but the strategy must deliver stable ~2% monthly return with strong Sharpe/Sortino/smart Sortino/Calmar.

## Promotion rule now in force
Report `improved=true` only if train/validation/OOS monthlyized return are each `>= +2.0%`, locked-OOS MDD `<= 25%`, OOS total return and return/risk beat the current champion, and OOS Sharpe `>=2`, Sortino `>=3`, smart Sortino `>=3`, Calmar `>=1`. Locked-OOS stays report-only/gate-only; selection remains train/validation only.

## Evidence from reassessment
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_return_quality_v1/fresh_portfolio_tuning_latest.json`
- Specs evaluated: `62,970`.
- Success candidates: `0`.
- Peak RSS: `812.921875 MiB`; `/usr/bin/time` max RSS `832432 KB`.
- Old alpha-v2 candidate: locked-OOS `+1.3397%` total but only `+0.6121%` monthlyized; train monthlyized `+0.4474%`; smart Sortino `2.0594` under the local return-quality penalty.
- Diagnostic best OOS in the replay reached OOS monthlyized `+1.8056%`, still below the `+2.0%` floor and train monthlyized only `+0.9583%`.

## Changed surfaces
- `scripts/research/tune_profit_moonshot_fresh_portfolio.py`: adds return-quality constants, monthlyized return, smart Sortino, and stricter promotion gates.
- `scripts/research/validate_profit_moonshot_pass_under_8gb.py`: validates candidate return quality and rejects stale pass artifacts.
- Tests: `tests/test_profit_moonshot_fresh_portfolio_tuning.py`, `tests/test_profit_moonshot_pass_under_8gb_validator.py` lock the new gates.
- State: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` and `var/reports/.../alpha_v2/passing_candidate_latest.json` are marked superseded.

## Verification completed so far
- `uv run --extra dev ruff check scripts/research/tune_profit_moonshot_fresh_portfolio.py scripts/research/validate_profit_moonshot_pass_under_8gb.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py` -> passed.
- `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py` -> `13 passed in 0.10s`.
- Mission validator against alpha-v2 result now intentionally fails (`candidate_return_quality_contract=false`) until a new candidate satisfies the 2% monthly contract.

## Next step
Run the next alpha-search lane under the same contract. Do not treat low-MDD/low-return rows as sufficient; search for stable return first, then enforce the quality metrics and locked-OOS gate.
