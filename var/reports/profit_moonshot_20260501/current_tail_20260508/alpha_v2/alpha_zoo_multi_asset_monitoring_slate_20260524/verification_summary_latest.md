# Multi-asset monitoring slate verification

Generated: 2026-05-24 KST/UTC session

## Artifact checks

- `multi_asset_monitoring_slate_latest.json` invariant check passed.
- Candidate rows normalized: `2488`.
- Symbols covered by monitoring matrix: `14`.
- Paper/testnet monitor candidates: `136` across `3` symbols (`ETHUSDT`, `SOLUSDT`, `TRXUSDT`).
- Monitoring matrix keeps source-coverage-only / shadow symbols (`ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `DOGEUSDT`, `TONUSDT`, `XAGUSDT`, `XAUUSDT`, `XPDUSDT`, `XPTUSDT`, `XRPUSDT`) rather than monitoring only one or two leaders.
- No-real-money guard passed: `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.
- Locked-OOS usage flags: discovery/selection/objective/pruning/parameter fitting all `false`; locked-OOS is gate/report-only.
- Primary cost/RPT threshold: `10bps`.

## Local verification

- Runner: `/usr/bin/time -v uv run python scripts/research/run_alpha_zoo_multi_asset_monitoring_slate.py`
  - Max RSS: `139092 KiB` (<8 GiB)
- Artifact invariant inline check: passed.
- Targeted pytest: `27 passed in 0.42s`
  - Max RSS: `207504 KiB` (<8 GiB)
- `uv run ruff check .`: passed.
- `uv run python -m compileall -q src scripts tests`: passed.
- `uv run python scripts/audit_hardcoded_params.py`: `new=0`.
- `git diff --check && git diff --cached --check`: passed.
- Full pytest: `1427 passed in 65.82s (0:01:05)`
  - Max RSS: `2759596 KiB` (<8 GiB)
