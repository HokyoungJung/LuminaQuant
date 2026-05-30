# Alpha Zoo Optuna Hybrid Paper/Testnet Live Decision

- strategy_name: `AlphaZooOptunaHybridLiveStrategy`
- selected_mode: `alpha_zoo_integer_leverage_optuna_hybrid`
- selected_profile_id: `hybrid_v3_5_optuna_three_profile_blend`
- symbols: `BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, TRX/USDT`
- target_allocation_mode: `notional_fraction`
- target_allocation source: `SignalEvent.metadata.target_allocation`
- risk_caps: `{"max_order_notional_pct": 1.247444, "max_order_value": 0.0, "max_symbol_exposure_pct": 1.427227, "max_total_margin_pct": 3.520744, "max_total_notional_pct": 3.520744}`
- intrabar protection: `paper_local_or_simulated_component_exit_plus_paper_testnet_exchange_algo_limit_stop`
- limit order contract: `entry/exit LMT one_tick_worse; market optional only with explicit opt-in`
- unfilled-order policy: `cancel, reconcile, no chase, no market fallback`
- slippage guard: `skip/cancel on guard breach; no high-slippage fallback`
- paper/testnet exchange protection: `STOP/TAKE_PROFIT limit after entry fill`
- asset applicability: `ETHUSDT, SOLUSDT, TRXUSDT`
- microstructure telemetry: `required before real-money review`
- ready_for_real: `false`
- real_money_execution: `false`
- real_execution_allowed: `false`
- primary round-trip cost: `10bps`

This is a paper/testnet handoff artifact only; it is not a real-money approval.

## Real-money blockers

- paper_testnet_artifacts_only: ready_for_real, real_money_execution, and real_execution_allowed are false
- no_exchange_paper_fill_telemetry: realized BBO spread, fees, slippage, rejects, partial fills, and cancels are not observed yet
- backtest_cost_is_proxy: 10bps round-trip friction is enforced in replay and gates but is not a live measured all-in cost
- fail_closed_allocation: decision target_allocation is 0.0 and live sizing depends on SignalEvent.metadata.target_allocation
- live_market_orders_disabled_by_default: any market-style execution requires an explicit allow_market_orders=true override and a separate review
- strict_slippage_guard_requires_bbo_telemetry: orders are skipped when BBO is missing or spread/slippage guard breaches

## Known limitations

- The selected v3.5 Optuna blend is dominated by the aggressive source profile, so independent-alpha diversification is limited.
- Validation MDD is near the relaxed 20% label and exceeds the strict 12% promotion cap.
- locked-OOS remains gate/report-only; it is not a parameter-fitting or selection surface.
- Alpha decisions use completed 1h/2h/4h bars; paper/testnet exchange-side STOP/TAKE_PROFIT limit protection is supported after entry fill, but real-money remains unapproved.
- Default one_tick_worse limit prices are marketable limits for fast fill control, not a maker-fee guarantee; realized fee_bps must be measured in paper/testnet.
- Paper/testnet liquidity can diverge from real exchange liquidity, funding, fees, and liquidation mechanics.
- Frozen-artifact replay avoids online learning; stale artifacts or regime drift require a new research/paper review.

## Paper/testnet validation requirements

- realized BBO spread and all-in round-trip cost by symbol/timeframe
- order reject, timeout, cancel, and partial-fill rates
- replay/live notional parity from SignalEvent metadata to submitted order notional
- position reconciliation drift and stale-data blocks/recoveries
- liquidation-inclusive MDD and account-wipeout telemetry
- minimum 2 weeks paper/testnet observation before any real-money review

## Intrabar / microstructure contract

- Entry signals attach stop-loss and chandelier-style trailing protection where available.
- Entry, short-entry, and reduce-only exit orders are limit-first (`LMT`) with `one_tick_worse` pricing by default.
- BUY limits are one exchange tick above the reference; SELL limits are one exchange tick below it; `same_price` and `one_tick_better` remain config options.
- If a limit order does not fill, the runtime cancels on timeout, reconciles any partial fill, and waits for the next valid completed-bar signal instead of chasing.
- Missing-BBO, high-spread, and high-slippage submissions are skipped/canceled; market fallback remains disabled by default.
- Paper/local simulation can emit component-level EXIT signals from intrabar guard breaches.
- Paper/testnet can submit Binance USD-M conditional algo `STOP`/`TAKE_PROFIT` limit protection after entry fill.
- Algo-order request payloads are whitelisted to Binance-supported fields; internal parent/protection telemetry is not sent to the exchange.
- Real-money exchange-side protective orders are still not approved without separate telemetry review.
- Selected-source asset applicability is verified for ETHUSDT, SOLUSDT, and TRXUSDT.
- Exact queue priority is unavailable from the exchange; use BBO/depth/fill-latency proxies.
