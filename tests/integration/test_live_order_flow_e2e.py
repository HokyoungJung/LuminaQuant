"""C7 — end-to-end live order-flow scenarios on the scripted fake exchange.

Wires the REAL ``LiveExecutionHandler`` + live ``Portfolio`` + ``RiskManager``
under the real event-dispatch path (``process_event`` routing identical to
``LiveTrader.run()``) with ONLY the exchange faked. Covers the three historical
live seam bugs the current ``SimpleNamespace`` unit tests could not see:

* retry double-send (5xx-then-success on submit),
* flatten latch not clearing once flat (fill-during-flatten),
* SIGNAL/ORDER-before-FILL ordering across a normal round-trip,

plus restart-mid-position state recovery (M1) and duplicate executionReport
dedup on the user-stream fill path.
"""

from __future__ import annotations

import pytest

from lumina_quant.core.events import OrderEvent
from lumina_quant.live.binance_user_stream import BinanceUserStreamClient

from tests.integration.live_harness import (
    RetryableExchangeError,
    ScriptedFakeExchange,
    build_live_trader,
    drive_events,
    open_long_position,
)


def _count(seq: list[str], value: str) -> int:
    return sum(1 for item in seq if item == value)


# --------------------------------------------------------------------------- #
# Scenario 1: normal entry/exit round-trip                                    #
# --------------------------------------------------------------------------- #
def test_normal_entry_exit_round_trip():
    exchange = ScriptedFakeExchange()
    trader, exchange, audit, _notifier = build_live_trader(exchange=exchange)

    trader.events.put(OrderEvent("BTC/USDT", "MKT", 1.0, "BUY"))
    result = drive_events(trader)
    assert result["errors"] == []
    assert _count(result["processed"], "ORDER") == 1
    assert _count(result["processed"], "FILL") == 1
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)

    trader.events.put(OrderEvent("BTC/USDT", "MKT", 1.0, "SELL", reduce_only=True))
    result = drive_events(trader)
    assert result["errors"] == []
    assert _count(result["processed"], "FILL") == 1
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(0.0)

    # exactly one exchange submit per leg (no double-send on the happy path)
    assert len(exchange.submit_calls) == 2
    assert len(audit.fills) == 2


# --------------------------------------------------------------------------- #
# Scenario 2: fill-during-flatten converges to flat, latch clears             #
# --------------------------------------------------------------------------- #
def test_fill_during_flatten_converges_and_latch_clears():
    exchange = ScriptedFakeExchange()
    trader, exchange, audit, _notifier = build_live_trader(exchange=exchange)

    open_long_position(trader, "BTC/USDT", qty=1.0, price=100.0)
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)
    # Default config is HEDGE: the flatten reads side-aware legs from the exchange.
    exchange.position_legs = {"BTC/USDT": {"LONG": 1.0, "SHORT": 0.0}}

    # Force a hard-drawdown FLATTEN: equity 20% under day-start, hard tier at 10%.
    trader.portfolio.day_start_equity = 10_000.0
    trader.portfolio.current_holdings["total"] = 8_000.0
    trader.risk_manager.hard_drawdown_flatten_pct = 0.10

    # 1st risk-guard tick queues the reduce-only flatten order.
    trader._evaluate_risk_guards()
    assert trader._flatten_inflight is True
    assert trader._flatten_attempts == 1

    # Driving the queue fills the reduce-only exit -> position goes flat.
    result = drive_events(trader)
    assert result["errors"] == []
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(0.0)

    # 2nd tick: breach persists but positions are gone -> the latch MUST clear and
    # NO FLATTEN_FAILED may be raised (the historical flatten-latch bug).
    trader._evaluate_risk_guards()
    assert trader._flatten_inflight is False
    assert trader._flatten_attempts == 0
    assert "FLATTEN_ALL_TRIGGERED" in audit.reasons()
    assert "FLATTEN_FAILED" not in audit.reasons()


# --------------------------------------------------------------------------- #
# Scenario 3: restart mid-position restores positions + kill-switch (M1)       #
# --------------------------------------------------------------------------- #
def test_restart_mid_position_restores_state_and_kill_switch():
    store: dict = {}
    exchange_a = ScriptedFakeExchange()
    trader_a, _ex, _audit, _notif = build_live_trader(exchange=exchange_a, state_store=store)

    open_long_position(trader_a, "BTC/USDT", qty=1.0, price=100.0)
    for _ in range(3):
        trader_a.risk_manager.record_loss(realized_pnl=-5.0)
    trader_a._hard_halt_active = True
    trader_a._hard_halt_reason = "manual_test_halt"
    trader_a._save_state()
    assert store.get("state")  # payload persisted

    # "Restart": a fresh trader over the same on-disk state.
    exchange_b = ScriptedFakeExchange()
    trader_b, _ex, _audit, _notif = build_live_trader(exchange=exchange_b, state_store=store)

    assert trader_b.portfolio.current_positions.get("BTC/USDT") == pytest.approx(1.0)
    # The consecutive-loss counter survives the restart (halt cannot re-arm to 0).
    assert trader_b.risk_manager._consecutive_loss_count == 3
    assert trader_b._hard_halt_active is True
    assert trader_b.portfolio.trading_frozen is True


def test_foreign_state_is_refused_in_real_mode():
    store: dict = {}
    trader_a, _ex_a, _audit_a, _notif_a = build_live_trader(
        exchange=ScriptedFakeExchange(),
        mode="real",
        state_store=store,
        strategy_name="AlphaOne",
    )
    open_long_position(trader_a, "BTC/USDT", qty=1.0, price=100.0)
    trader_a._save_state()
    assert store.get("state")

    # A DIFFERENT strategy on the same account/mode must NOT inherit the positions
    # (M1 state-fingerprint identity check, enforced in real mode).
    with pytest.raises(RuntimeError, match="fingerprint rejected"):
        build_live_trader(
            exchange=ScriptedFakeExchange(),
            mode="real",
            state_store=store,
            strategy_name="AlphaTwo",
        )


# --------------------------------------------------------------------------- #
# Scenario 4: duplicate executionReport is counted exactly once               #
# --------------------------------------------------------------------------- #
def test_duplicate_execution_report_fills_once():
    # ``ingest_user_stream_event`` is the user-stream fill-projection seam and runs
    # independently of the configured order_state_source; use polling mode so no
    # (non-daemon) websocket thread is spawned by the constructor.
    exchange = ScriptedFakeExchange()
    trader, exchange, _audit, _notifier = build_live_trader(exchange=exchange)

    raw = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_700_000_000_000,
        "o": {
            "s": "BTCUSDT",
            "i": 987654321,
            "c": "LQ-dup-test",
            "x": "TRADE",
            "X": "FILLED",
            "l": 1.0,
            "z": 1.0,
            "L": 100.0,
            "t": 55,
            "S": "BUY",
            "ps": "LONG",
            "R": False,
        },
    }
    normalized = BinanceUserStreamClient.parse_message(raw)
    assert normalized is not None and normalized["event_type"] == "executionReport"

    trader.execution_handler.ingest_user_stream_event(dict(normalized))
    trader.execution_handler.ingest_user_stream_event(dict(normalized))

    fills = []
    while not trader.events.empty():
        event = trader.events.get_nowait()
        if str(getattr(event, "type", "")).upper() == "FILL":
            fills.append(event)
    assert len(fills) == 1
    assert fills[0].quantity == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Scenario 5: 5xx-then-success on submit must NOT double-send                  #
# --------------------------------------------------------------------------- #
def test_retryable_5xx_on_submit_does_not_double_send(monkeypatch):
    monkeypatch.setattr("lumina_quant.live.execution_live.time.sleep", lambda *_a, **_k: None)

    exchange = ScriptedFakeExchange()
    exchange.submit_faults.append(RetryableExchangeError(status_code=502))
    trader, exchange, audit, _notifier = build_live_trader(exchange=exchange)

    trader.events.put(OrderEvent("BTC/USDT", "MKT", 1.0, "BUY"))
    result = drive_events(trader)

    assert result["errors"] == []
    # Two submit ATTEMPTS were made (fail, then success) but only ONE order exists.
    assert len(exchange.submit_calls) == 2
    assert len(exchange.orders) == 1
    # And exactly one net fill => the position is 1.0, not a double-sent 2.0.
    assert trader.portfolio.current_positions["BTC/USDT"] == pytest.approx(1.0)
    assert len(audit.fills) == 1
