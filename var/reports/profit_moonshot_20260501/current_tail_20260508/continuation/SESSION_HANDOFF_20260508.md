# Profit Moonshot latest-tail continuation handoff — 2026-05-08

## Status
- Latest available tail refreshed from Binance raw aggTrades/materialized OHLCV through `2026-05-08T10:53:10Z`; all five symbols reached `2026-05-08T10:53:09Z`.
- Locked-OOS/report-only policy preserved; OOS end date was `2026-05-08`.
- 8 GiB guard preserved: refresh peak `4102.4 MiB`, replay `2129.2 MiB`, portfolio `318.2 MiB`, Optuna `243.6 MiB`.

## Results
- Replay: `63` success candidates / `4941` specs.
  - Top replay candidate `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss80_tp180`: train `1.9449%`, val `0.5464%`, locked-OOS `1.0074%`, OOS MDD `0.1532%`, OOS Sharpe `7.9499`.
- Portfolio tuning: `2` success candidates / `10704` specs.
  - First pass candidate `fresh_portfolio_validation_return_risk_weight_fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls540_ss120_tp450__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls530_ss120_tp450`: train `0.6940%`, val `3.0370%`, locked-OOS `0.8789%`, OOS MDD `0.1760%`, OOS Sharpe `5.6177`.
  - Diagnostic highest OOS return `fresh_portfolio_additive_sleeves_fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr50_h120_sc80_st0__fresh_calendar_rot_lstrongest_sethusdt_lb336_thr100_h120_sc80_st0__fresh_calendar_rot_ltrxusdt_sethusdt_lb168_thr100_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr200_h336_sc80_st0__fresh_calendar_rot_lbtcusdt_sweakest_lb168_thr100_h336_sc80_st0` is **not** selected because it fails the shadow-MDD gate; keep it research-only.
- Optuna calendar tuning: `2` success candidates / `24` trials.
  - Selected Optuna candidate `optuna_calendar_trx_sethusdt_t0_thr150_h120_ls620_ss100_tp180`: train `2.2558%`, val `0.6824%`, locked-OOS `1.0074%`, OOS MDD `0.1532%`, OOS Sharpe `7.9499`.

## Artifacts
- Passing bundle: `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/passing_candidate_latest.json`
- Replay: `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/fresh_start_overhaul_replay_latest.json`, `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/fresh_start_overhaul_replay_candidates.csv`
- Portfolio: `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/fresh_portfolio_tuning_latest.json`, `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/fresh_portfolio_tuning_candidates.csv`
- Optuna: `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/calendar_optuna_latest.json`, `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/calendar_optuna_trials.csv`
- Mission validator copy: `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/mission_validation_latest.json`

## Verification
- Targeted profit-moonshot tests: `16 passed in 0.34s`.
- Full local tests: `uv run --extra dev pytest -q` → `1191 passed in 283.20s (0:04:43)`.
- Full ruff: `uv run --extra dev ruff check .` → `All checks passed!`.
- Compile check: `python3 -m compileall -q src scripts tests` → passed.
- Targeted py_compile: `python3 -m py_compile scripts/research/replay_profit_moonshot_fresh_start.py scripts/research/tune_profit_moonshot_fresh_portfolio.py scripts/research/optuna_tune_profit_moonshot_calendar.py scripts/research/validate_profit_moonshot_pass_under_8gb.py` → passed.
- `uv run python scripts/research/validate_profit_moonshot_pass_under_8gb.py --output-path var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/mission_validation_latest.json --markdown-path var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/mission_validation_latest.md` → `passed`.

## Next
- Do not promote diagnostic high-return portfolio sleeves unless the shadow-MDD gate is redesigned and re-approved.
- If continuing, run a wider Optuna sweep or portfolio search with the same `--oos-end-date` discipline and one heavy process at a time.
