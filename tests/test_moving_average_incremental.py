import queue
import unittest
from dataclasses import dataclass

from lumina_quant.events import MarketEvent
from strategies.moving_average import MovingAverageCrossStrategy


@dataclass
class _BarStore:
    symbol_list: list[str]

    def __post_init__(self):
        self._prices = {symbol: [] for symbol in self.symbol_list}

    def append_price(self, symbol, price):
        self._prices[symbol].append(float(price))

    def get_latest_bars_values(self, symbol, val_type, N=1):
        _ = val_type
        return self._prices[symbol][-N:]

    def get_latest_bar_value(self, symbol, val_type):
        _ = val_type
        return self._prices[symbol][-1] if self._prices[symbol] else 0.0


def _expected_signals(closes, short_window, long_window):
    state = "OUT"
    out = []
    for i in range(len(closes)):
        if i + 1 < long_window:
            continue
        short = sum(closes[i - short_window + 1 : i + 1]) / float(short_window)
        long = sum(closes[i - long_window + 1 : i + 1]) / float(long_window)
        if short > long and state == "OUT":
            out.append((i, "LONG"))
            state = "LONG"
        elif short < long and state == "LONG":
            out.append((i, "EXIT"))
            state = "OUT"
    return out


class TestMovingAverageIncremental(unittest.TestCase):
    def test_incremental_matches_reference_signals(self):
        symbol = "BTC/USDT"
        short_window = 3
        long_window = 5
        closes = [100, 101, 102, 103, 104, 103, 102, 101, 100, 101, 102, 103]

        bars = _BarStore([symbol])
        events = queue.Queue()
        strategy = MovingAverageCrossStrategy(
            bars,
            events,
            short_window=short_window,
            long_window=long_window,
        )

        for idx, close in enumerate(closes):
            bars.append_price(symbol, close)
            event = MarketEvent(idx, symbol, close, close, close, close, 1.0)
            strategy.calculate_signals(event)

        actual = []
        while not events.empty():
            signal = events.get()
            actual.append((int(signal.datetime), str(signal.signal_type)))

        expected = _expected_signals(closes, short_window, long_window)
        self.assertEqual(expected, actual)

    def test_state_roundtrip_preserves_signal_sequence(self):
        symbol = "BTC/USDT"
        closes = [10, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15]

        # Full run baseline
        full_bars = _BarStore([symbol])
        full_events = queue.Queue()
        full_strategy = MovingAverageCrossStrategy(
            full_bars, full_events, short_window=3, long_window=5
        )
        for idx, close in enumerate(closes):
            full_bars.append_price(symbol, close)
            full_strategy.calculate_signals(
                MarketEvent(idx, symbol, close, close, close, close, 1.0)
            )

        full_signals = []
        while not full_events.empty():
            signal = full_events.get()
            full_signals.append((int(signal.datetime), str(signal.signal_type)))

        # Split run with state transfer in the middle
        split_index = 6
        bars_a = _BarStore([symbol])
        events_a = queue.Queue()
        strategy_a = MovingAverageCrossStrategy(bars_a, events_a, short_window=3, long_window=5)

        for idx, close in enumerate(closes[:split_index]):
            bars_a.append_price(symbol, close)
            strategy_a.calculate_signals(MarketEvent(idx, symbol, close, close, close, close, 1.0))

        state = strategy_a.get_state()

        bars_b = _BarStore([symbol])
        events_b = queue.Queue()
        strategy_b = MovingAverageCrossStrategy(bars_b, events_b, short_window=3, long_window=5)
        for close in closes[:split_index]:
            bars_b.append_price(symbol, close)
        strategy_b.set_state(state)

        for idx, close in enumerate(closes[split_index:], start=split_index):
            bars_b.append_price(symbol, close)
            strategy_b.calculate_signals(MarketEvent(idx, symbol, close, close, close, close, 1.0))

        split_signals = []
        while not events_a.empty():
            signal = events_a.get()
            split_signals.append((int(signal.datetime), str(signal.signal_type)))
        while not events_b.empty():
            signal = events_b.get()
            split_signals.append((int(signal.datetime), str(signal.signal_type)))

        self.assertEqual(full_signals, split_signals)


if __name__ == "__main__":
    unittest.main()
