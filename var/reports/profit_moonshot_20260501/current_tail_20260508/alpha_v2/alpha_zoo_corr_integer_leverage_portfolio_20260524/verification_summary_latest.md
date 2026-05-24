# Integer-Leverage Correlation Portfolio Verification

Generated: 2026-05-24T12:27:21Z

## Artifact invariant

- Artifact: `alpha_zoo_corr_integer_leverage_portfolio_latest.json`
- Selected strict profile: `balanced_mdd12_gross5`
- Active integer leverage map: `SOLUSDT=2`, `TRXUSDT=1`
- Selected candidate count: `4`
- Gross notional fraction: `1.00x`
- Train / validation / locked-OOS report-only returns: `+74.6685%` / `+33.2153%` / `+5.5300%`
- Validation / locked-OOS report-only MDD: `11.6134%` / `7.2003%`
- Trade events train / validation / locked-OOS: `945` / `229` / `100`
- RPT proxy train / validation / locked-OOS: `30.91bps` / `57.02bps` / `22.21bps`
- Liquidation/account wipeout: `0/0` across gate/report splits
- Governance: `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`; locked-OOS selection/discovery/objective/pruning/fitting flags all false.
- Runner max RSS: `6,492,344 KiB` by `/usr/bin/time -v` (`runner_peak_rss_mib=6340.18` in artifact), below the 8 GiB cap.

## Relaxed shadow context

- Growth relaxed shadow: `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `3.90x`, train / validation / locked-OOS `+262.3353%` / `+71.6291%` / `+23.3695%`, validation MDD `19.9983%`; not strict promotion.
- Aggressive relaxed shadow: `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `4.90x`, train / validation / locked-OOS `+438.4462%` / `+117.4976%` / `+27.5772%`, validation MDD `29.4044%`; not strict promotion.

## Local verification

Log: `local_verification_integer_leverage_20260524T122617Z.log`

- Artifact invariant check passed.
- Targeted tests: `13 passed` (`test_alpha_zoo_corr_integer_leverage_portfolio.py`, `test_alpha_zoo_pnl_correlation_decision.py`, `test_alpha_zoo_multi_asset_monitoring_slate.py`).
- `ruff check .` passed.
- `python -m compileall -q src scripts tests` passed.
- Hardcoded parameter audit: `total=567 new=0 baselined=567`.
- `git diff --check` and `git diff --cached --check` passed.
- Full pytest: `1435 passed in 64.41s`; max RSS `2,755,288 KiB`, below 8 GiB.
