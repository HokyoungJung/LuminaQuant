import queue
import unittest
from datetime import datetime

from lumina_quant.events import SignalEvent
from lumina_quant.portfolio import Portfolio


class MockBars:
    symbol_list = ["BTC/USDT"]

    def get_latest_bar_value(self, symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    def get_market_spec(self, symbol):
        _ = symbol
        return {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 5.0}


class MockConfig:
    INITIAL_CAPITAL = 10000.0
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.1
    MAX_DAILY_LOSS_PCT = 0.03
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    DEFAULT_STOP_LOSS_PCT = 0.01


class TestPortfolioSizing(unittest.TestCase):
    def test_risk_based_order_generation(self):
        p = Portfolio(MockBars(), queue.Queue(), datetime(2026, 1, 1), MockConfig)
        signal = SignalEvent(
            strategy_id=1,
            symbol="BTC/USDT",
            datetime=datetime(2026, 1, 1),
            signal_type="LONG",
            stop_loss=99.0,
        )
        order = p.generate_order_from_signal(signal)
        self.assertIsNotNone(order)
        self.assertGreater(order.quantity, 0.0)
        self.assertEqual(order.direction, "BUY")
        self.assertEqual(order.position_side, "LONG")


if __name__ == "__main__":
    unittest.main()
