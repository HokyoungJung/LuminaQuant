from __future__ import annotations

import queue
from datetime import datetime

from lumina_quant.core.events import OrderEvent
from lumina_quant.live.execution_live import LiveExecutionHandler


class _Bars:
    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)


class _Config:
    EXCHANGE_ID = "BINANCE"
    MARKET_TYPE = "future"
    TAKER_FEE_RATE = 0.0004
    ORDER_TIMEOUT = 10
    ORDER_STATE_SOURCE = "user_stream"
    RECONCILIATION_POLL_FALLBACK_ENABLED = False


class _Exchange:
    @staticmethod
    def execute_order(**kwargs):
        _ = kwargs
        return {
            "id": "ord-1",
            "status": "open",
            "filled": 0.0,
            "amount": 1.0,
            "price": 100.0,
        }

    @staticmethod
    def fetch_open_orders(symbol=None):
        _ = symbol
        return []

    @staticmethod
    def fetch_order(order_id, symbol=None):
        _ = (order_id, symbol)
        return {}

    @staticmethod
    def cancel_order(order_id, symbol=None):
        _ = (order_id, symbol)
        return True


def test_user_stream_projection_updates_state_and_emits_fill_delta():
    events = queue.Queue()
    handler = LiveExecutionHandler(events, _Bars(), _Config, _Exchange())

    states = []
    handler.set_order_state_callback(states.append)

    order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY")
    handler.execute_order(order)
    assert "ord-1" in handler.tracked_orders

    handler.ingest_user_stream_event(
        {
            "event_type": "executionReport",
            "exchange_ts_ms": 1_700_000_000_100,
            "symbol": "BTCUSDT",
            "order_id": "ord-1",
            "client_order_id": order.client_order_id,
            "exec_type": "TRADE",
            "order_status": "PARTIALLY_FILLED",
            "cum_fill_qty": 0.4,
            "last_fill_qty": 0.4,
            "last_fill_price": 100.0,
            "trade_id": 1,
            "side": "BUY",
        }
    )
    assert handler.tracked_orders["ord-1"]["state"] == "PARTIAL"

    handler.ingest_user_stream_event(
        {
            "event_type": "executionReport",
            "exchange_ts_ms": 1_700_000_000_200,
            "symbol": "BTCUSDT",
            "order_id": "ord-1",
            "client_order_id": order.client_order_id,
            "exec_type": "TRADE",
            "order_status": "FILLED",
            "cum_fill_qty": 1.0,
            "last_fill_qty": 0.6,
            "last_fill_price": 101.0,
            "trade_id": 2,
            "side": "BUY",
        }
    )

    assert "ord-1" not in handler.tracked_orders

    fills = []
    while not events.empty():
        evt = events.get_nowait()
        if getattr(evt, "type", "") == "FILL":
            fills.append(evt)

    assert len(fills) == 2
    assert round(sum(float(item.quantity) for item in fills), 8) == 1.0
    assert any(item.status == "PARTIALLY_FILLED" for item in fills)
    assert any(item.status == "FILLED" for item in fills)
    assert any(str(payload.get("state")) == "PARTIAL" for payload in states)
    assert any(str(payload.get("state")) == "FILLED" for payload in states)
