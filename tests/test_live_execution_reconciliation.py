"""Live execution reconciliation tests."""

from __future__ import annotations

import queue
from datetime import datetime
from typing import cast

from lumina_quant.core.protocols import ExchangeInterface
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
    ORDER_TIMEOUT = 5


class _Exchange:
    def __init__(self):
        self._open_orders = []

    def execute_order(self, **kwargs):
        _ = kwargs
        return {"id": "order-x", "status": "open", "filled": 0.0, "amount": 1.0, "price": 100.0}

    def fetch_order(self, order_id, symbol):
        _ = (order_id, symbol)
        return {
            "id": "order-1",
            "status": "closed",
            "filled": 1.0,
            "amount": 1.0,
            "average": 100.0,
            "price": 100.0,
        }

    def fetch_open_orders(self, symbol=None):
        _ = symbol
        return list(self._open_orders)

    @staticmethod
    def cancel_order(order_id, symbol=None):
        _ = (order_id, symbol)
        return True

    @staticmethod
    def get_balance(currency):
        _ = currency
        return 1000.0

    @staticmethod
    def get_all_positions():
        return {}


class _FailOpenOrdersExchange(_Exchange):
    @staticmethod
    def fetch_open_orders(symbol=None):
        _ = symbol
        raise RuntimeError("network-down")


def test_rehydrate_and_reconcile_terminal_order_emits_fill():
    events = queue.Queue()
    exchange = _Exchange()
    handler = LiveExecutionHandler(events, _Bars(), _Config, cast(ExchangeInterface, exchange))
    handler.rehydrate_orders(
        {
            "order-1": {
                "state": "OPEN",
                "symbol": "BTC/USDT",
                "client_order_id": "LQ-test-1",
                "last_filled": 0.0,
                "created_at": 1700000000.0,
                "metadata": {
                    "symbol": "BTC/USDT",
                    "direction": "BUY",
                    "order_type": "MKT",
                    "quantity": 1.0,
                    "position_side": "LONG",
                    "reduce_only": False,
                    "client_order_id": "LQ-test-1",
                },
            }
        }
    )

    assert "order-1" in handler.tracked_orders

    records = handler.reconcile_open_orders()

    assert "order-1" not in handler.tracked_orders
    assert any(item["reason"] == "TERMINAL_RESOLVED" for item in records)
    fill = events.get_nowait()
    assert fill.symbol == "BTC/USDT"
    assert fill.direction == "BUY"
    assert float(fill.quantity) == 1.0


def test_reconcile_open_orders_rehydrates_unknown_exchange_open_order():
    events = queue.Queue()
    exchange = _Exchange()
    exchange._open_orders = [
        {
            "id": "ext-1",
            "symbol": "BTC/USDT",
            "status": "open",
            "filled": 0.2,
            "amount": 1.0,
            "price": 100.0,
            "side": "buy",
            "type": "limit",
            "clientOrderId": "LQ-ext-1",
        }
    ]
    handler = LiveExecutionHandler(events, _Bars(), _Config, cast(ExchangeInterface, exchange))

    assert "ext-1" not in handler.tracked_orders
    records = handler.reconcile_open_orders()

    assert "ext-1" in handler.tracked_orders
    assert handler.exchange_open_order_count() == 1
    assert handler.exchange_open_snapshot_ready() is True
    assert any(item["reason"] == "UNTRACKED_OPEN_ORDER_REHYDRATED" for item in records)


def test_reconcile_open_orders_marks_snapshot_unready_on_fetch_failure():
    events = queue.Queue()
    exchange = _FailOpenOrdersExchange()
    handler = LiveExecutionHandler(events, _Bars(), _Config, cast(ExchangeInterface, exchange))

    _ = handler.reconcile_open_orders()
    assert handler.exchange_open_snapshot_ready() is False
