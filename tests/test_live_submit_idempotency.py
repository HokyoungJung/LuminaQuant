"""Idempotent order-submission retry (2026-07-03 audit fix #3a).

A plain retry loop double-sends on the documented Binance "unknown outcome"
path: the matching engine accepts the order but the HTTP response fails with
a retryable error, and a clientOrderId becomes reusable once the original
order fills. The handler must query the exchange by client id before ever
resubmitting, adopting the accepted order when found.
"""

import queue
import unittest
from datetime import datetime

from lumina_quant.core.events import OrderEvent
from lumina_quant.live.execution_live import LiveExecutionHandler


class MockBars:
    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    @staticmethod
    def get_market_spec(symbol):
        _ = symbol
        return {"price_tick_size": 0.1}


class MockConfig:
    EXCHANGE_ID = "BINANCE"
    MARKET_TYPE = "future"
    TAKER_FEE_RATE = 0.0004
    ORDER_TIMEOUT = 2
    MODE = "paper"


class RequestTimeout(Exception):
    """Name matters: _is_retryable_exception matches by class name."""


class UnknownOutcomeExchange:
    """Accepts the order server-side, then fails the submit response once."""

    def __init__(self, *, accept_before_failing=True):
        self.accept_before_failing = accept_before_failing
        self.accepted_orders = []
        self.submit_calls = 0

    def execute_order(self, *, symbol, type, side, quantity, price=None, params=None):
        self.submit_calls += 1
        client_id = (params or {}).get("newClientOrderId")
        if self.submit_calls == 1:
            if self.accept_before_failing:
                self.accepted_orders.append(
                    {
                        "id": f"EX-{self.submit_calls}",
                        "clientOrderId": client_id,
                        "symbol": symbol,
                        "status": "open",
                        "filled": 0.0,
                        "amount": quantity,
                    }
                )
            raise RequestTimeout("simulated 5xx after matching-engine accept")
        self.accepted_orders.append(
            {
                "id": f"EX-{self.submit_calls}",
                "clientOrderId": client_id,
                "symbol": symbol,
                "status": "open",
                "filled": 0.0,
                "amount": quantity,
            }
        )
        return dict(self.accepted_orders[-1])

    def fetch_order(self, order_id, symbol=None, params=None):
        _ = (order_id, symbol)
        wanted = str((params or {}).get("origClientOrderId") or "")
        for order in self.accepted_orders:
            if str(order.get("clientOrderId") or "") == wanted:
                return dict(order)
        raise ValueError("order not found")

    def fetch_open_orders(self, symbol=None):
        _ = symbol
        return [dict(order) for order in self.accepted_orders]


class TestSubmitIdempotency(unittest.TestCase):
    def _handler(self, exchange):
        events = queue.Queue()
        return LiveExecutionHandler(events, MockBars(), MockConfig, exchange)

    def test_unknown_outcome_adopts_accepted_order_instead_of_resubmitting(self):
        exchange = UnknownOutcomeExchange(accept_before_failing=True)
        handler = self._handler(exchange)
        order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY")
        handler.execute_order(order)
        # The exchange holds exactly ONE order: the retry adopted the accepted
        # order via clientOrderId lookup instead of double-sending.
        self.assertEqual(len(exchange.accepted_orders), 1)
        self.assertEqual(exchange.submit_calls, 1)
        self.assertIn("EX-1", handler.tracked_orders)

    def test_true_failure_resubmits_once_lookup_finds_nothing(self):
        exchange = UnknownOutcomeExchange(accept_before_failing=False)
        handler = self._handler(exchange)
        order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY")
        handler.execute_order(order)
        # First attempt truly failed (nothing accepted) -> lookup finds nothing
        # -> resubmission is safe and produces exactly one live order.
        self.assertEqual(exchange.submit_calls, 2)
        self.assertEqual(len(exchange.accepted_orders), 1)
        self.assertIn("EX-2", handler.tracked_orders)


if __name__ == "__main__":
    unittest.main()
