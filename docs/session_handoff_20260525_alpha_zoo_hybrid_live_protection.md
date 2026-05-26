# Session Handoff — Alpha Zoo Hybrid Paper/Testnet Live Protection

Date: 2026-05-25 KST; amended 2026-05-26 KST for limit-first execution
Scope: `private-main` / `private/main` private source-of-truth

## Current conclusion

The Alpha Zoo Optuna hybrid live path is saved as a **paper/testnet-only** candidate. It is not approved for real-money execution.

Hard safety flags remain:

- `ready_for_real=false`
- `real_money_execution=false`
- `real_execution_allowed=false`

## What is now implemented

- `AlphaZooOptunaHybridLiveStrategy` remains the selected runtime handoff for the frozen Optuna v3.5 hybrid artifact.
- Entry, short-entry, reduce-only exit, and risk-flatten parent orders are now limit-first: `LMT`, `one_tick_worse`, BUY reference +1 tick, SELL reference -1 tick. Market parent orders require explicit `allow_market_orders=true`.
- Entry signals carry component-level stop/trailing metadata for paper/local intrabar guard handling.
- After an entry order is confirmed filled, the paper/testnet execution path can submit Binance USD-M Futures conditional algo protection:
  - `STOP`
  - `TAKE_PROFIT`
  - side-aware one-tick-worse limit `price` plus `GTC`
  - endpoint family: `POST /fapi/v1/algoOrder`
- Market-style `STOP_MARKET` / `TAKE_PROFIT_MARKET` remains optional only behind explicit market opt-in and separate review.
- The exchange request payload is allowlisted to Binance-supported fields. Internal parent/protection telemetry keys are retained in local reconciliation records and are not forwarded to the exchange.
- Asset-generic behavior is regression-tested across selected frozen source assets:
  - `ETHUSDT`
  - `SOLUSDT`
  - `TRXUSDT`

## Primary artifacts and docs

- Paper/testnet handoff JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.json`
- Paper/testnet handoff MD: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.md`
- Operator runbook: `docs/live-readiness/04-paper-trading-runbook.md`
- Readiness checklist update: `docs/live-readiness/01-live-trading-checklist.md`
- Paper readiness update: `docs/live-readiness/02-paper-trading-readiness.md`
- Deployment safety note: `docs/DEPLOYMENT.md`
- Research history: `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md`
- README summaries: `README.md`, `README_KR.md`

## Verification evidence

Implementation commit `6a01168b` was pushed to `private/main` and passed CI:

- local `ruff check .`
- architecture check
- `compileall`
- hardcoded-parameter audit `new=0`
- `git diff --check`
- full pytest `1460 passed`
- max RSS `2,724,568 KiB` (<8 GiB)
- GitHub Actions `private-ci` success: `26398200366`
- GitHub Actions `ci` success: `26398200362`

The current docs/README pass records the same conclusion and should be committed as a documentation-only follow-up.

## Remaining blockers

- No actual Binance testnet conditional algo fill sample has been collected yet.
- Protective sibling cancellation/reconciliation after one leg triggers still needs paper/testnet evidence.
- Queue priority remains proxy-only; exact exchange queue position is not available.
- Real-money execution remains blocked until at least two continuous weeks of paper/testnet fill, BBO spread, slippage, reject/timeout, reconciliation, and protective-order telemetry validate the 10bps/replay-live parity assumptions.
