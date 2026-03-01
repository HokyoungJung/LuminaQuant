import math
import queue
import unittest
from dataclasses import dataclass

from lumina_quant.core.events import MarketEvent
from lumina_quant.symbols import canonical_symbol
from lumina_quant.strategies.pair_trading_zscore import PairTradingZScoreStrategy


@dataclass
class _PairBarStore:
    symbol_list: list[str]

    def __post_init__(self):
        canonical = [canonical_symbol(symbol) for symbol in self.symbol_list]
        self._latest_close = dict.fromkeys(canonical)
        self._latest_time = dict.fromkeys(canonical)
        self.symbol_list = canonical

    def set_bar(self, symbol, time_index, close_price):
        token = canonical_symbol(symbol)
        self._latest_time[token] = time_index
        self._latest_close[token] = float(close_price)

    def get_latest_bar_value(self, symbol, val_type):
        _ = val_type
        value = self._latest_close.get(canonical_symbol(symbol))
        return float(value) if value is not None else 0.0

    def get_latest_bar_datetime(self, symbol):
        return self._latest_time.get(canonical_symbol(symbol))


def _build_pair_prices(length=320):
    prices = []
    for idx in range(length):
        y_price = 100.0 + (0.08 * idx)
        spread_noise = 0.08 * math.sin(idx / 5.0)
        if 220 <= idx <= 236:
            spread_noise += 1.6
        if 280 <= idx <= 295:
            spread_noise -= 1.7
        x_price = y_price + spread_noise
        prices.append((idx, x_price, y_price))
    return prices


def _run_strategy(prices, split=None):
    symbol_x = "XAU/USDT"
    symbol_y = "XAG/USDT"
    params = {
        "lookback_window": 30,
        "hedge_window": 60,
        "entry_z": 1.5,
        "exit_z": 0.25,
        "stop_z": 4.5,
        "min_correlation": -1.0,
        "max_hold_bars": 120,
        "cooldown_bars": 0,
        "symbol_x": symbol_x,
        "symbol_y": symbol_y,
    }

    def _feed(strategy, bars, events, rows):
        for ts, x_price, y_price in rows:
            bars.set_bar(symbol_x, ts, x_price)
            bars.set_bar(symbol_y, ts, y_price)
            event = MarketEvent(ts, symbol_x, x_price, x_price, x_price, x_price, 1.0)
            strategy.calculate_signals(event)

        out = []
        while not events.empty():
            signal = events.get()
            out.append((int(signal.datetime), str(signal.symbol), str(signal.signal_type)))
        return out

    if split is None:
        bars = _PairBarStore([symbol_x, symbol_y])
        events = queue.Queue()
        strategy = PairTradingZScoreStrategy(bars, events, **params)
        return _feed(strategy, bars, events, prices)

    rows_a = prices[:split]
    rows_b = prices[split:]

    bars_a = _PairBarStore([symbol_x, symbol_y])
    events_a = queue.Queue()
    strategy_a = PairTradingZScoreStrategy(bars_a, events_a, **params)
    signals_a = _feed(strategy_a, bars_a, events_a, rows_a)
    state = strategy_a.get_state()

    bars_b = _PairBarStore([symbol_x, symbol_y])
    events_b = queue.Queue()
    strategy_b = PairTradingZScoreStrategy(bars_b, events_b, **params)
    strategy_b.set_state(state)
    signals_b = _feed(strategy_b, bars_b, events_b, rows_b)
    return [*signals_a, *signals_b]


class TestPairTradingZScore(unittest.TestCase):
    def test_generates_pair_entry_and_exit_signals(self):
        signals = _run_strategy(_build_pair_prices())
        self.assertGreater(len(signals), 0)

        signal_types = [signal_type for _, _, signal_type in signals]
        self.assertIn("LONG", signal_types)
        self.assertIn("SHORT", signal_types)
        self.assertIn("EXIT", signal_types)

        exit_symbols = {symbol for _, symbol, signal_type in signals if signal_type == "EXIT"}
        self.assertEqual(exit_symbols, {"XAU/USDT", "XAG/USDT"})

    def test_state_roundtrip_preserves_signal_sequence(self):
        prices = _build_pair_prices()
        full_signals = _run_strategy(prices)
        split_signals = _run_strategy(prices, split=190)
        self.assertEqual(full_signals, split_signals)


if __name__ == "__main__":
    unittest.main()
