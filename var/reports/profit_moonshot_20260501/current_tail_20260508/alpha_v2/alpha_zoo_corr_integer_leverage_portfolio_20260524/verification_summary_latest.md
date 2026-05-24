# Integer-Leverage Correlation Portfolio Verification

Generated: 2026-05-24T12:44:46Z

## Artifact invariant

- Artifact: `alpha_zoo_corr_integer_leverage_portfolio_latest.json`
- Paper/testnet candidate profiles: `3`
  - Strict promotion candidate: `balanced_mdd12_gross5`, active integer leverage `SOLUSDT=2`, `TRXUSDT=1`, gross `1.00x`, train / validation / locked-OOS report-only `+74.6685%` / `+33.2153%` / `+5.5300%`, validation MDD `11.6134%`, locked-OOS MDD `7.2003%`, RPT `30.91/57.02/22.21bps`, trades `945/229/100`.
  - Relaxed paper/testnet candidate: `growth_mdd20_gross8`, integer leverage `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `3.90x`, train / validation / locked-OOS `+262.3353%` / `+71.6291%` / `+23.3695%`, validation MDD `19.9983%`, locked-OOS MDD `9.2371%`, RPT `32.16/37.72/23.35bps`, trades `1255/292/124`.
  - Relaxed paper/testnet candidate: `aggressive_mdd30_gross10_shadow`, integer leverage `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `4.90x`, train / validation / locked-OOS `+438.4462%` / `+117.4976%` / `+27.5772%`, validation MDD `29.4044%`, locked-OOS MDD `12.3630%`, RPT `38.81/44.16/21.87bps`, trades `1539/360/158`.
- All paper/testnet candidates have positive train/validation/locked-OOS report-only returns, train >= validation, required trade counts, split RPT > `10bps`, integer leverage maps, and locked-OOS liquidation/account-wipeout `0/0`.
- Governance: `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`; locked-OOS selection/discovery/objective/pruning/fitting flags all false.
- Runner max RSS: `6,396,380 KiB` by `/usr/bin/time -v` (`runner_peak_rss_mib=6246.46` in artifact), below the 8 GiB cap.

## Strategy integrity / live-level review

- Integrity artifact: `strategy_integrity_review_latest.json|md`.
- Status: `pass`.
- Calendar/date-rule check: `pass`; checked the six active sleeves and found no calendar/date token hits.
- Source-code calendar feature grep: `pass` for `dt.day`, `dt.weekday`, `dt.dayofweek`, `dt.month`, `dt.hour`, `day_of_week`, `weekday`, `month_end`, `hour_of_day`, and `time_of_day` across the source strategy runners.
- Hardcoded model-ID grep: `pass`; selected model IDs are not hardcoded in the integer-leverage runner and are derived from the frozen corr-decision artifact.
- Cost check: `pass`; the runner records primary `10bps` all-in round-trip execution friction and RPT threshold `avg BBO spread 2bps * 5 = 10bps`. This is a backtest friction proxy, not live fill-derived slippage telemetry.
- Live-level status: paper/testnet-review only; runner executes no orders and requires replay/live notional parity plus realized BBO/fill/cost/liquidation/account-wipeout telemetry before any future real-money review.

## Local verification

Log: `local_verification_integer_leverage_20260524T124342Z.log`

- Artifact invariant check passed.
- Code-level calendar feature grep passed.
- Hardcoded model-ID grep passed.
- Targeted tests: `14 passed` (`test_alpha_zoo_corr_integer_leverage_portfolio.py`, `test_alpha_zoo_pnl_correlation_decision.py`, `test_alpha_zoo_multi_asset_monitoring_slate.py`).
- `ruff check .` passed.
- `python -m compileall -q src scripts tests` passed.
- Hardcoded parameter audit: `total=567 new=0 baselined=567`.
- `git diff --check` and `git diff --cached --check` passed.
- Full pytest: `1436 passed in 67.59s`; max RSS `2,745,132 KiB`, below 8 GiB.

## Post integrity/docs sanity verification

Additional post-documentation sanity verification passed after recording the strategy-integrity review and candidate-set correction:

- Log: `local_verification_integer_leverage_post_integrity_docs_20260524T125141Z.log`
- Artifact invariant: `pass` (`ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`, integrity review `pass`, cost/RPT threshold `10bps`, 3 paper/testnet candidate profiles)
- Targeted integer-leverage tests: `5 passed`
- Ruff on runner/test: passed
- Compileall on runner/test: passed
- Hardcoded-parameter audit: `total=567 new=0 baselined=567`
- `git diff --check` and `git diff --cached --check`: passed
