import os
import queue
import sys
import unittest
from datetime import datetime

# Add Parent Dir to Path to import lumina_quant
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.core.events import MarketEvent, OrderEvent


# Mock classes
class MockBars:
    def __init__(self):
        self.data = {}  # symbol -> list of bars
        self.latest = {}  # symbol -> current bar

    def get_latest_bar_value(self, symbol, val_type):
        if symbol not in self.latest:
            return 0.0
        # Mock bar: (datetime, open, high, low, close, volume)
        # indices: 0, 1, 2, 3, 4, 5
        bar = self.latest[symbol]
        if val_type == "open":
            return bar[1]
        elif val_type == "high":
            return bar[2]
        elif val_type == "low":
            return bar[3]
        elif val_type == "close":
            return bar[4]
        elif val_type == "volume":
            return bar[5]
        return 0.0

    def get_latest_bar_datetime(self, symbol):
        return self.latest[symbol][0]

    def set_latest_bar(self, symbol, bar):
        self.latest[symbol] = bar


class MockConfig:
    COMMISSION_RATE = 0.0
    SLIPPAGE_RATE = 0.0
    SPREAD_RATE = 0.0  # Zero spread for easy price check


class TestLookAheadBias(unittest.TestCase):
    def setUp(self):
        self.events = queue.Queue()
        self.bars = MockBars()
        self.config = MockConfig()
        self.handler = SimulatedExecutionHandler(self.events, self.bars, self.config)

    def test_market_order_filling(self):
        symbol = "BTCUSDT"

        # --- TIME T=1 ---
        # Strategy sees Bar 1 and decides to BUY
        # Bar 1: Open=100, Close=110
        bar1 = (datetime(2023, 1, 1, 10, 0), 100.0, 115.0, 95.0, 110.0, 1000.0)
        self.bars.set_latest_bar(symbol, bar1)

        # Strategy sends ORDER
        order = OrderEvent(symbol, "MKT", 1.0, "BUY")

        # Handler processes Order at T=1
        self.handler.execute_order(order)

        # CHECK: Queue should be empty (Order NOT filled yet)
        self.assertTrue(
            self.events.empty(),
            "Order filled immediately at T=1! Look-Ahead Bias detected.",
        )

        # CHECK: Order should be pending
        self.assertEqual(len(self.handler.active_orders), 1)
        self.assertEqual(self.handler.active_orders[0]["status"], "PENDING")
        print("\n[Pass] Order NOT filled at T=1 (Close=110)")

        # --- TIME T=2 ---
        # Market moves to Bar 2
        # Bar 2: Open=112, Close=120
        bar2 = (datetime(2023, 1, 1, 11, 0), 112.0, 125.0, 111.0, 120.0, 1000.0)
        self.bars.set_latest_bar(symbol, bar2)

        # Market Event T=2 triggers processing
        mkt_event = MarketEvent(bar2[0], symbol, bar2[1], bar2[2], bar2[3], bar2[4], bar2[5])
        self.handler.check_open_orders(mkt_event)

        # CHECK: Queue should have FillEvent now
        self.assertFalse(self.events.empty(), "Order NOT filled at T=2 Open.")

        fill = self.events.get()
        print(
            f"[Info] Fill Price: {fill.fill_cost} (Unit Price approx {fill.fill_cost / fill.quantity})"
        )

        # KEY ASSERTION: Price should be Bar 2 OPEN (112), NOT Bar 1 CLOSE (110)
        # Note: Slippage/Spread might affect it slightly, but we set them to 0 in MockConfig.
        # Wait, SimulatedExecutionHandler uses random slippage if not mocked out?
        # It imports random inside the method.
        # But if we check if price is closer to 112 than 110.

        unit_price = fill.fill_cost / fill.quantity
        diff_open = abs(unit_price - 112.0)
        diff_close = abs(unit_price - 110.0)

        print(f"[Check] Diff from Next Open (112): {diff_open:.4f}")
        print(f"[Check] Diff from Prev Close (110): {diff_close:.4f}")

        # If random slippage is applied, it might drift.
        # But 1bps-5bps is small. 112 * 0.0005 = 0.056.
        # So price should be 112 +/- 0.06.
        # 110 is far away.

        self.assertTrue(diff_open < 1.0, "Fill Price is NOT close to Next Open!")
        self.assertTrue(diff_close > 1.0, "Fill Price IS close to previous Close! Look-Ahead Bias?")

        print("[Pass] Order filled at T=2 Open. No Look-Ahead confirmed.")


if __name__ == "__main__":
    unittest.main()
