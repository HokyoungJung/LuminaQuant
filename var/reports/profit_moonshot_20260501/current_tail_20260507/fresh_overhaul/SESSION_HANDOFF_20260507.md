# Session handoff — profit moonshot fresh reset (2026-05-07)

## Current state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch: `private-main`
- Data refreshed through complete cutoff `2026-05-06T23:59:59Z` from `binance_raw_aggtrades`.
- Memory evidence:
  - data refresh peak RSS: `4793.78515625 MiB`, workers `1`
  - coverage inventory peak RSS from `/usr/bin/time -v`: `3254400 KB`
  - fresh replay peak RSS: `2547.13671875 MiB`
- Targeted tests passed: `uv run pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_strategy_support_inventory.py tests/test_raw_taker_flow_feature_backfill.py` → `5 passed`.

## Important correction from crash recovery

The crashed fresh-start replay was invalid because it read only sparse `timeframe=1h` materialized partitions, producing only `47` joined hourly rows and starving train/val. The new loader in `scripts/research/replay_profit_moonshot_fresh_start.py` now chunks monthly/daily raw-first 1s data into hourly bars and fills incomplete monthly days from committed daily materialized 1s partitions.

## Latest results

- Fresh-start replay specs evaluated: `1219`
- Families: residual reversion, cross momentum, funding-carry fade, taker-flow momentum, taker-flow exhaustion, residual+flow confirmation, compression breakout.
- Replay survivors: `0`
- Success candidates: `0`
- Decision: **do not promote or trade any fresh candidate**.

Key artifact:

```text
var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul/RISK_STRATEGY_RESET_20260507.md
```

## Next useful work

1. Do not keep tuning stale incumbent/context-wrapper variants.
2. Run split-wise predictive diagnostics first; no more indicator spray.
3. Backfill train/val open-interest if OI will be used; otherwise keep OI out of selectable paths.
4. Only run one live-equivalent full backtest after a stateful replay survivor exists.
5. Keep one heavy process at a time under 8 GiB.

## Resume prompt

```text
cd /home/hoky/Quants-agent/LuminaQuant

Continue from:
var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul/SESSION_HANDOFF_20260507.md
var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul/RISK_STRATEGY_RESET_20260507.md

Do not reuse stale incumbent/context-wrapper assumptions except as baseline gates. Start data-first: use current refreshed raw-first data through 2026-05-06, run split-wise feature diagnostics for taker-flow/funding/vol/residuals, backfill or exclude OI, then generate only a small candidate set whose train and validation evidence is positive before OOS reporting. One heavy run at a time, workers=1, total session memory under 8GB, and record RSS evidence.
```

## Additional continuation in this session

- Added multi-sleeve fresh portfolio tuner:
  - `scripts/research/tune_profit_moonshot_fresh_portfolio.py`
  - output: `fresh_portfolio_tuning_latest.md/json/csv`
- Portfolio tuning evaluated `25194` validation-primary portfolio specs from `18` train+val-positive sleeves.
- Success candidates remained `0`; selected-by-validation portfolio had train `+0.2034%`, val `+0.3111%`, OOS `+0.1349%`, OOS Sharpe `0.325600`.
- Diagnostic best OOS portfolio reached OOS `+0.2661%`, Sharpe `0.480320`, but is not selection authority and still fails incumbent/Sharpe gates.

## CI fixes made

- Ruff now excludes `var/` artifacts via `pyproject.toml` so report scratch scripts do not fail source lint.
- Hardcoded parameter audit baseline refreshed to current strategy state (`567` baselined, `0` new).
- Profit reboot unit fixture updated to include the current OOS success gate contract.

## Verified green locally

- `uv run ruff check .`
- `bash scripts/ci/architecture_gate_live_data.sh`
- `bash scripts/ci/architecture_gate_market_window_contract.sh`
- `bash scripts/ci/architecture_gate_binance_native.sh`
- `uv run python scripts/check_architecture.py`
- `uv run python scripts/audit_hardcoded_params.py`
- `uv run python scripts/verify_docs.py`
- CI raw-first pytest subset: `76 passed`
- Dashboard: `npm install`, `npm run lint`, `npm run test`, `npm run typecheck`, `npm run build`
- Benchmark smoke + 8GB gate: peak RSS `184.45 MiB < 7372.80 MiB`
- GPU contract: `verify_polars_gpu_runtime.py --mode auto` passed; GPU tests `24 passed`
- Full pytest: `1168 passed, 1262 warnings`
