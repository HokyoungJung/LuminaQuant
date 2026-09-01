# Session handoff — profit moonshot integer leverage audit — 2026-05-09

## Status
Integer-leverage and raw-train audit work is implemented and replayed. No new candidate is promoted; the current base remains the reference because all wider integer-leverage challengers failed at least one stricter quality/current-base gate.

## Key files changed
- `scripts/research/tune_profit_moonshot_fresh_portfolio.py`
  - integer leverage grid for `train_val_monthly_return_budget`,
  - raw/unlevered train/validation diagnostics,
  - stricter train buffer/raw/integer promotion gates,
  - current-base sleeve anchoring and forced base-combo evaluation.
- `scripts/research/validate_profit_moonshot_pass_under_8gb.py`
  - validator now rejects new promoted candidates with weak raw train/validation or fractional leverage.
- `tests/test_profit_moonshot_fresh_portfolio_tuning.py`
  - locks integer-grid behavior, raw floor-fit quarantine, and non-monotonic leverage selection.
- `tests/test_profit_moonshot_pass_under_8gb_validator.py`
  - locks validator rejection of weak raw-train and fractional-leverage candidates.

## Main evidence
- Audit summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_audit_20260509.json` and `.md`.
- Wider top40 run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_top40_20260509/fresh_portfolio_tuning_latest.json`.
- Top40 replay scale: `158,620` portfolio specs, `40` sleeves, `0` success candidates.
- Top40 memory: payload peak `2523.5234 MiB`; `/usr/bin/time` max RSS `2,584,088 KiB`; under the `<8 GiB` guard.
- Mutex evidence: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/leader_integer_leverage_alpha_v2_top40_mutex_20260509.json` (`status=completed`, `overlap_check=passed`).

## What the leverage audit found
- Selected-by-validation top40 row uses integer leverage `3`, with train monthly `+2.7510%`, validation monthly `+11.8767%`, raw train `+1.0111%`, and OOS return `+6.7836%`; it is still rejected because OOS return/risk does not beat the current base.
- Best OOS diagnostic uses integer leverage `6`, with OOS return `+18.4446%`, MDD `+2.2305%`, and return/risk `8.2694`; it is rejected because raw train is only `+0.8615%`, train Sortino fails, and OOS return/risk is still below current base.
- Forced current-base integer row selected `5x`, producing OOS return `+14.6371%`, MDD `+1.6919%`, return/risk `8.6514`, but it fails raw train (`+0.9075%`) and train Sortino (`1.4814`).

## Decision boundary
The suspicious “more leverage always looks better” result is not accepted as a promoted improvement. Higher leverage can boost OOS return, but it is quarantined unless raw train, train Sortino, integer leverage, train/validation stability, and current-base OOS gates all pass.

## Verification status
- Targeted tests: `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py` -> `26 passed in 0.11s`.
- Focused ruff: `uv run --extra dev ruff check scripts/research/tune_profit_moonshot_fresh_portfolio.py scripts/research/validate_profit_moonshot_pass_under_8gb.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_profit_moonshot_pass_under_8gb_validator.py` -> passed.
- Full pytest: `uv run --extra dev pytest -q` -> `1220 passed in 338.07s (0:05:38)`; `/usr/bin/time` max RSS `2,673,972 KiB`.
- Repo ruff: `uv run --extra dev ruff check .` -> `All checks passed`.
- Compileall: `python3 -m compileall -q src scripts tests` -> pass.
- Whitespace: `git diff --check` -> pass.
- Mission result is updated with integer-leverage audit evidence; validator/CI finalization remains after source push.

## Next steps
Run final full verification, update `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json`, commit with Lore protocol, push to `private/main`, and verify both `ci` and `private-ci` green.

## Final CI / validator evidence
- Lore commit `6055d691b07fac1d54362c51157935d628f3f129` pushed to `private/main`.
- GitHub Actions: `private-ci` run `25602373489` success; `ci` run `25602373501` success.
- Final mission validator: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/validation_latest.json` -> `status=passed`, `passed=true`.
