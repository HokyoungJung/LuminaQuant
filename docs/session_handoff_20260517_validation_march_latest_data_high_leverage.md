# Session handoff — 2026-05-17 latest-data March-validation high-leverage Alpha Zoo

## Status

Completed local data refresh, March-extended validation split, isolated high-leverage Alpha Zoo tuning/replay, and live-decision artifact generation.

## Data and split

- Data refresh cutoff: `2026-05-17T10:59:59Z`.
- Joined panel: `var/cache/profit_moonshot_fresh_start/joined_panel_76f825ffea81c04f2fe41fbf.parquet`.
- Actual panel max: `2026-05-17T10:00:00Z`.
- Split:
  - train `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z`
  - validation `2026-01-01T00:00:00Z` ~ `2026-03-31T23:00:00Z`
  - locked-OOS `2026-04-01T00:00:00Z` ~ `2026-05-17T10:00:00Z`

## Selection result

Train+validation-only ranking froze candidates before locked-OOS. Locked-OOS was gate/report-only.

Live-promoted high-leverage candidate:

- Strategy: `CryptoFxAlphaZooStateStrategy`
- Candidate params: `alpha_zoo_fast_residual`
- Leverage/margin: isolated `7x`
- Target allocation: `0.15`
- Locked-OOS: return `+30.5357%`, MDD `11.3027%`, Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, liquidation `0`, account-wipeout `0`.

Strict zero-liquidation fallback lane for the same params:

- `6x` at `0.10` allocation: locked-OOS return `+16.7783%`, MDD `6.5951%`, liquidation `0`, min margin buffer `9150.924760`, deployable under strict gate.

## Key files

- Runner: `scripts/research/run_alpha_zoo_validation_march_high_leverage.py`
- Runtime leverage cap: `src/lumina_quant/configuration/validate.py` now allows live exchange leverage up to `20x`.
- Tests updated: `tests/test_live_config_source_validation.py`, `tests/test_live_selection_infer.py`.
- Main artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json`
- Data refresh report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/data_refresh_latest.json`

## Follow-up

Verification is complete locally. Remaining closeout: Lore-commit/push to `private/main` and confirm `ci`/`private-ci`.

## Verification

Passed on 2026-05-17 UTC:

- Live/source validation targeted suite: `27 passed`.
- Required Alpha Zoo suite: `24 passed`.
- Required moonshot validation suite: `74 passed`.
- Full pytest: `1329 passed in 80.27s`.
- `uv run --extra dev ruff check .`: passed.
- `uv run --extra dev python -m compileall -q src scripts tests`: passed.
- `git diff --check` and `git diff --cached --check`: passed.

Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/local_verification_validation_march_high_leverage_20260517T120700Z.log`.

Live-readiness preflight for the isolated `7x` decision artifact passed in paper/testnet mode with a supplied paper Postgres DSN placeholder and stale threshold override: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`. Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_readiness_preflight_7x_latest.json`.

Post-staging CSV LF/runner-lineterminator verification re-ran clean: full pytest `1329 passed in 55.01s`, required Alpha Zoo suite `24 passed`, moonshot suite `74 passed`, live/source targeted suite `27 passed`, ruff/compileall/diff checks passed.
