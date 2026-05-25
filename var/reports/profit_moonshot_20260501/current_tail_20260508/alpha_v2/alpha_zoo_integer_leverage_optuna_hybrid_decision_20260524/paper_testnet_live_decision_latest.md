# Alpha Zoo Optuna Hybrid Paper/Testnet Live Decision

- strategy_name: `AlphaZooOptunaHybridLiveStrategy`
- selected_mode: `alpha_zoo_integer_leverage_optuna_hybrid`
- selected_profile_id: `hybrid_v3_5_optuna_three_profile_blend`
- symbols: `BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, TRX/USDT`
- target_allocation_mode: `notional_fraction`
- target_allocation source: `SignalEvent.metadata.target_allocation`
- risk_caps: `{"max_order_notional_pct": 1.247444, "max_order_value": 0.0, "max_symbol_exposure_pct": 1.427227, "max_total_margin_pct": 3.520744, "max_total_notional_pct": 3.520744}`
- intrabar protection: `paper_local_or_simulated_component_exit`
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

## Known limitations

- The selected v3.5 Optuna blend is dominated by the aggressive source profile, so independent-alpha diversification is limited.
- Validation MDD is near the relaxed 20% label and exceeds the strict 12% promotion cap.
- locked-OOS remains gate/report-only; it is not a parameter-fitting or selection surface.
- Alpha decisions use completed 1h/2h/4h bars; intrabar exits are paper/local simulated guards, not approved real exchange-side protective orders.
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
- Paper/local simulation can emit component-level EXIT signals from intrabar guard breaches.
- Real exchange-side protective orders are still not approved without explicit exchange order support.
- Exact queue priority is unavailable from the exchange; use BBO/depth/fill-latency proxies.
