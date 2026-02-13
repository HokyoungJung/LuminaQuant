import queue
import unittest

from strategies.rsi_strategy import RsiStrategy


class MockBars:
    symbol_list = ["BTC/USDT"]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        _ = (symbol, val_type, N)
        return [100.0, 101.0, 102.0, 103.0, 104.0]


class TestStrategyStateCompat(unittest.TestCase):
    def test_rsi_state_roundtrip(self):
        strategy = RsiStrategy(MockBars(), queue.Queue())
        strategy.bought["BTC/USDT"] = "LONG"

        state = strategy.get_state()
        self.assertEqual(state["bought"]["BTC/USDT"], "LONG")

        strategy.bought["BTC/USDT"] = "OUT"
        strategy.set_state(state)
        self.assertEqual(strategy.bought["BTC/USDT"], "LONG")


if __name__ == "__main__":
    unittest.main()
