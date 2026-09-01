# Context Snapshot: Profit Moonshot Strategy Validity Audit

- Created: 2026-05-10T10:06:09Z
- CWD: `/home/hoky/Quants-agent/LuminaQuant`
- Branch/HEAD: `private-main` @ `69578fd30be9709d7018785abff365b26d1f300f`

## Task statement
User requests `$ralplan $team $ralph`: audit not only final candidates but all used/previous strategy artifacts for theoretical/live-trading flaws like fixed calendar/seasonality rules, then retune/optimize/retest/re-rank under practical live constraints and report final deployable results.

## Desired outcome
A live-ready selection process and report that rejects theoretically defective/data-mined rules, especially fixed month+fixed asset calendar alpha without robust thesis, reruns candidate selection/backtests under those gates, and commits/pushes verified artifacts.

## Known facts/evidence
- Current pushed HEAD: `69578fd... Require live-integer leverage before promotion`; CI/private-ci green for this commit.
- Current final winner is direct candidate `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_5x`.
- That winner is a `calendar_rotation` family derived from `scripts/research/replay_profit_moonshot_fresh_start.py`, with fixed `calendar_long_symbol=TRXUSDT`, `calendar_short_symbol=ETHUSDT`, long months `(3,4,5)`, short months `(1,2)`, 168h lookback and take-profit.
- User correctly challenged that crypto calendar fixed month/symbol seasonality is a serious live-validity defect absent robust exogenous thesis.
- Existing final-selection gates cover integer leverage, liquidation, locked-OOS selection firewall, and performance, but not alpha-thesis/data-mining validity.
- Split windows: train 2025-01-01..2025-12-31; validation 2026-01-01..2026-02-28; locked-OOS 2026-03-01..2026-05-09 report/gate only.
- Universe/panel symbols: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, TRX/USDT.

## Constraints
- Live deployability is required; theoretical/practical defects must block promotion even if metrics look good.
- Locked-OOS remains gate/report-only, never selection.
- Integer leverage only for live promotion.
- Conservative liquidation/margin model remains required.
- Total process memory must stay below 8 GiB.
- Must run tests/lint/compileall/diff checks; commit with Lore protocol; push to private/main; verify GitHub Actions green.

## Unknowns/open questions
- Which existing historical artifacts should be included in “all used” audit beyond current `live_final_selection_20260510` artifacts.
- Whether any non-calendar dynamic candidates remain deployable after validity gates.
- Whether retuning from current candidate CSV without calendar family produces a superior live candidate or no-promote outcome.

## Likely codebase touchpoints
- `scripts/research/replay_profit_moonshot_fresh_start.py` — spec family definitions and fixed calendar rules.
- `scripts/research/tune_profit_moonshot_fresh_portfolio.py` — candidate portfolio ranking/selection.
- `scripts/research/run_profit_moonshot_liquidation_aware_validation.py` — liquidation-aware replay and retune seed selection.
- `scripts/research/write_profit_moonshot_live_final_selection.py` — final promotion gates/report.
- New likely script: `scripts/research/audit_profit_moonshot_strategy_validity.py` or similar.
- Tests under `tests/test_profit_moonshot_*`.
- Artifacts under `var/reports/profit_moonshot_20260501/live_final_selection_20260510/` and prior alpha_v2/current_tail dirs.
