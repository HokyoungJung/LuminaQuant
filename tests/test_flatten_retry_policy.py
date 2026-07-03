"""FLATTEN kill-switch retry/escalation policy (2026-07-03 audit fix #3b)."""

import time
import unittest
from types import SimpleNamespace

from lumina_quant.live.trader import LiveTrader


def _state(*, started_ago, attempts, position=1.0, retry_seconds=30.0, max_retries=3):
    return SimpleNamespace(
        config=SimpleNamespace(
            FLATTEN_RETRY_SECONDS=retry_seconds,
            FLATTEN_MAX_RETRIES=max_retries,
        ),
        _flatten_started_at=(time.time() - started_ago) if started_ago is not None else None,
        _flatten_attempts=attempts,
        portfolio=SimpleNamespace(current_positions={"BTC/USDT": position}),
        symbol_list=["BTC/USDT"],
    )


class TestFlattenRetryDue(unittest.TestCase):
    def test_due_when_breach_persists_and_positions_remain(self):
        state = _state(started_ago=31.0, attempts=1)
        self.assertTrue(LiveTrader._flatten_retry_due(state))

    def test_not_due_before_retry_interval(self):
        state = _state(started_ago=5.0, attempts=1)
        self.assertFalse(LiveTrader._flatten_retry_due(state))

    def test_not_due_after_max_retries(self):
        state = _state(started_ago=120.0, attempts=3)
        self.assertFalse(LiveTrader._flatten_retry_due(state))

    def test_not_due_when_flat(self):
        state = _state(started_ago=120.0, attempts=1, position=0.0)
        self.assertFalse(LiveTrader._flatten_retry_due(state))

    def test_not_due_without_inflight_flatten(self):
        state = _state(started_ago=None, attempts=0)
        self.assertFalse(LiveTrader._flatten_retry_due(state))


class TestMarketEscalationGuard(unittest.TestCase):
    """The execution market-order guard admits ONLY gated flatten escalations."""

    def _handler(self, *, escalate_enabled):
        import queue
        from datetime import datetime

        from lumina_quant.live.execution_live import LiveExecutionHandler

        class Bars:
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

        class Config:
            EXCHANGE_ID = "BINANCE"
            MARKET_TYPE = "future"
            TAKER_FEE_RATE = 0.0004
            ORDER_TIMEOUT = 2
            MODE = "paper"
            ALLOW_MARKET_ORDERS = False
            FLATTEN_ESCALATE_TO_MARKET = escalate_enabled

        class Exchange:
            def execute_order(self, *, symbol, type, side, quantity, price=None, params=None):
                _ = (symbol, side, quantity, price, params)
                assert type == "market"
                return {"id": "EX-1", "status": "closed", "filled": quantity, "amount": quantity}

            def fetch_order(self, *args, **kwargs):
                raise ValueError("not found")

            def fetch_open_orders(self, symbol=None):
                _ = symbol
                return []

        return LiveExecutionHandler(queue.Queue(), Bars(), Config, Exchange())

    def test_plain_market_order_still_blocked(self):
        from lumina_quant.core.events import OrderEvent

        handler = self._handler(escalate_enabled=True)
        order = OrderEvent("BTC/USDT", "MKT", 1.0, "SELL")
        with self.assertRaises(RuntimeError):
            handler.execute_order(order)

    def test_gated_flatten_escalation_passes(self):
        from lumina_quant.core.events import OrderEvent

        handler = self._handler(escalate_enabled=True)
        order = OrderEvent(
            "BTC/USDT",
            "MKT",
            1.0,
            "SELL",
            reduce_only=True,
            metadata={"risk_flatten_escalation": True},
        )
        handler.execute_order(order)  # must not raise

    def test_escalation_metadata_without_config_gate_is_blocked(self):
        from lumina_quant.core.events import OrderEvent

        handler = self._handler(escalate_enabled=False)
        order = OrderEvent(
            "BTC/USDT",
            "MKT",
            1.0,
            "SELL",
            reduce_only=True,
            metadata={"risk_flatten_escalation": True},
        )
        with self.assertRaises(RuntimeError):
            handler.execute_order(order)


if __name__ == "__main__":
    unittest.main()
