# 69-Asset Efficiency-Repair Paper/Testnet Live Decision

- strategy_name: `AlphaZooOptunaHybridLiveStrategy`
- selected_mode: `alpha_zoo_69_asset_efficiency_repair_optuna_hybrid`
- selected_profile_id: `hybrid_v3_6_optuna_three_profile_blend`
- symbol_count: `69`
- selected_source_symbol_count: `13`
- selected_source_symbols: `ADAUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, COPPERUSDT, CRCLUSDT, DOGEUSDT, SOLUSDT, TONUSDT, XAGUSDT, XAUUSDT, XPTUSDT, XRPUSDT`
- live_final_gross_notional: `2.338904x`
- historical_train_validation_gross_notional: `2.504196x`
- target_allocation_mode: `notional_fraction`
- target_allocation source: `SignalEvent.metadata.target_allocation`
- risk_caps: `{"max_order_notional_pct": 0.419745, "max_order_value": 0.0, "max_symbol_exposure_pct": 0.485734, "max_total_margin_pct": 2.588904, "max_total_notional_pct": 2.588904}`
- limit order contract: `LMT one_tick_worse by default; market optional only by explicit opt-in`
- unfilled-order policy: `cancel, reconcile, do not chase, no market fallback`
- slippage guard: `missing-BBO/guard breach skip or cancel; no high-slippage fallback`
- ready_for_real: `false`
- real_money_execution: `false`
- real_execution_allowed: `false`
- primary round-trip cost: `10bps`

This is a paper/testnet handoff artifact only; it is not a real-money approval.

## No-fill / slippage policy

```json
{
  "after_cancel_action": "skip_until_next_completed_bar_unless_signal_revalidates",
  "market_fallback_allowed": false,
  "max_chase_attempts": 0,
  "partial_fill_action": "keep_filled_cancel_remainder_on_timeout",
  "repeated_timeout_action": "freeze_symbol_and_require_operator_review",
  "resubmit_requires": [
    "same_component_signal_still_active",
    "fresh_completed_bar_or_same_bar_revalidation",
    "spread_within_slippage_guard",
    "notional_and_position_caps_unchanged"
  ],
  "timeout_action": "cancel_reconcile_revalidate_signal"
}
```

```json
{
  "limit_price_mode": "one_tick_worse",
  "limit_price_offset_ticks": 1,
  "market_fallback_allowed": false,
  "max_bbo_spread_bps_at_submit": 4.0,
  "max_estimated_one_way_slippage_bps": 5.0,
  "max_realized_one_way_slippage_bps": 5.0,
  "max_realized_round_trip_cost_bps": 10.0,
  "on_missing_bbo_snapshot": "do_not_submit_no_market_fallback",
  "on_open_order_breach": "cancel_open_order_no_market_fallback",
  "on_pre_submit_breach": "do_not_submit",
  "on_realized_breach": "freeze_symbol_and_review",
  "paper_testnet_measurement_required": true,
  "require_bbo_snapshot": true
}
```

## Real-money blockers

- paper_testnet_artifacts_only: ready_for_real, real_money_execution, and real_execution_allowed are false
- no_exchange_paper_fill_telemetry: realized BBO spread, fees, slippage, rejects, partial fills, and cancels are not observed yet
- backtest_cost_is_proxy: 10bps round-trip friction is enforced in replay and gates but is not a live measured all-in cost
- fail_closed_allocation: decision target_allocation is 0.0 and live sizing depends on SignalEvent.metadata.target_allocation
- live_market_orders_disabled_by_default: any market-style execution requires an explicit allow_market_orders=true override and a separate review
- strict_slippage_guard_requires_bbo_telemetry: orders are skipped when BBO is missing or spread/slippage guard breaches

## Known limitations

- The 69-asset efficiency repair expands the live watch universe and needs exchange paper/testnet fill evidence for every active asset group.
- Final live weights differ from historical train+validation average weights; the artifact records both gross figures for replay/live parity review.
- No locked test set is used in this final-refit mode; validation is the latest 8 complete weeks and real promotion requires forward telemetry.
- Alpha decisions use completed 30m/1h/2h/4h bars; paper/testnet exchange-side STOP/TAKE_PROFIT limit protection is supported after entry fill, but real-money remains unapproved.
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
