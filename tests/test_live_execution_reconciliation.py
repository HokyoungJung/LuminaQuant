"""Live execution reconciliation tests."""

from __future__ import annotations

import queue
from datetime import datetime
from typing import cast

from lumina_quant.interfaces import ExchangeInterface
from lumina_quant.live_execution import LiveExecutionHandler


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
