import queue
import unittest
from datetime import datetime, timedelta

from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.core.events import MarketEvent, OrderEvent


class _MockBars:
    def __init__(self):
        self.latest = {}

    def set_latest_bar(self, symbol, bar):
        self.latest[symbol] = bar

    def get_latest_bar_value(self, symbol, val_type):
        bar = self.latest.get(symbol)
        if bar is None:
            return 0.0
        idx = {
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }[val_type]
        return float(bar[idx])

    def get_latest_bar_datetime(self, symbol):
        bar = self.latest.get(symbol)
        if bar is None:
            return None
        return bar[0]


class _MockConfig:
    RANDOM_SEED = 42
    COMMISSION_RATE = 0.0
    TAKER_FEE_RATE = 0.0
    SLIPPAGE_RATE = 0.0
    SPREAD_RATE = 0.0


class TestExecutionProtectiveOrders(unittest.TestCase):
    def setUp(self):
        self.events = queue.Queue()
        self.bars = _MockBars()
        self.handler = SimulatedExecutionHandler(self.events, self.bars, _MockConfig())
        self.symbol = "BTC/USDT"
        self.t0 = datetime(2026, 1, 1, 0, 0, 0)

    def _push_market(self, dt, o, h, low_price, c, v=10000.0):
        bar = (dt, float(o), float(h), float(low_price), float(c), float(v))
        self.bars.set_latest_bar(self.symbol, bar)
        event = MarketEvent(dt, self.symbol, o, h, low_price, c, v)
        self.handler.check_open_orders(event)

    def test_take_profit_bracket_fills_and_cancels_sibling(self):
        self.handler.execute_order(
            OrderEvent(
                symbol=self.symbol,
                order_type="MKT",
                quantity=1.0,
                direction="BUY",
                position_side="LONG",
                stop_loss=99.0,
                take_profit=101.0,
            )
        )

        self._push_market(self.t0, 100.0, 100.2, 99.8, 100.1)
        first_fill = self.events.get_nowait()
        self.assertEqual(first_fill.direction, "BUY")
        self.assertEqual(first_fill.status, "FILLED")

        active_types = sorted(order["type"] for order in self.handler.active_orders)
        self.assertEqual(active_types, ["STOP", "TAKE_PROFIT"])

        self._push_market(self.t0 + timedelta(seconds=1), 100.9, 101.2, 100.7, 101.0)
        second_fill = self.events.get_nowait()
        self.assertEqual(second_fill.direction, "SELL")
        self.assertEqual(second_fill.status, "FILLED")
        self.assertEqual(len(self.handler.active_orders), 0)

    def test_stop_loss_bracket_fills_and_cancels_sibling(self):
        self.handler.execute_order(
            OrderEvent(
                symbol=self.symbol,
                order_type="MKT",
                quantity=1.0,
                direction="BUY",
                position_side="LONG",
                stop_loss=99.0,
                take_profit=101.0,
            )
        )

        self._push_market(self.t0, 100.0, 100.1, 99.9, 100.0)
        _ = self.events.get_nowait()

        self._push_market(self.t0 + timedelta(seconds=1), 99.4, 99.6, 98.8, 99.1)
        stop_fill = self.events.get_nowait()
        self.assertEqual(stop_fill.direction, "SELL")
        self.assertEqual(stop_fill.status, "FILLED")
        self.assertEqual(len(self.handler.active_orders), 0)


if __name__ == "__main__":
    unittest.main()
