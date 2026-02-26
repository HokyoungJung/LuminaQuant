import queue
import unittest
from dataclasses import dataclass

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.moving_average import MovingAverageCrossStrategy
from lumina_quant.strategies.rsi_strategy import RsiStrategy


@dataclass
class _BarStore:
    symbol_list: list[str]

    def __post_init__(self):
        self._latest_close = dict.fromkeys(self.symbol_list)
        self._latest_time = dict.fromkeys(self.symbol_list)

    def set_bar(self, symbol, time_index, close_price):
        self._latest_time[symbol] = time_index
        self._latest_close[symbol] = float(close_price)

    def get_latest_bar_value(self, symbol, value_type):
        _ = value_type
        value = self._latest_close.get(symbol)
        return float(value) if value is not None else 0.0

    def get_latest_bar_datetime(self, symbol):
        return self._latest_time.get(symbol)


def _collect_signal_types(events):
    out = []
    while not events.empty():
        out.append(str(events.get().signal_type))
    return out


class TestStrategyLongShortSupport(unittest.TestCase):
    def test_rsi_strategy_emits_short_when_enabled(self):
        bars = _BarStore(["BTC/USDT"])
        events = queue.Queue()
        strategy = RsiStrategy(
            bars,
            events,
            rsi_period=2,
            oversold=30,
            overbought=70,
            allow_short=True,
        )

        price_path = [100.0, 101.0, 102.0, 80.0, 120.0]
        for idx, price in enumerate(price_path):
            bars.set_bar("BTC/USDT", idx, price)
            event = MarketEvent(idx, "BTC/USDT", price, price, price, price, 1000.0)
            strategy.calculate_signals(event)

        signal_types = _collect_signal_types(events)
        self.assertIn("LONG", signal_types)
        self.assertIn("SHORT", signal_types)

    def test_moving_average_strategy_emits_short_when_enabled(self):
        bars = _BarStore(["BTC/USDT"])
        events = queue.Queue()
        strategy = MovingAverageCrossStrategy(
            bars,
            events,
            short_window=2,
            long_window=3,
            allow_short=True,
        )

        price_path = [100.0, 99.0, 98.0, 105.0, 95.0]
        for idx, price in enumerate(price_path):
            bars.set_bar("BTC/USDT", idx, price)
            event = MarketEvent(idx, "BTC/USDT", price, price, price, price, 1000.0)
            strategy.calculate_signals(event)

        signal_types = _collect_signal_types(events)
        self.assertIn("SHORT", signal_types)
        self.assertIn("LONG", signal_types)


if __name__ == "__main__":
    unittest.main()
