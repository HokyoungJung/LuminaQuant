import queue
import unittest
from dataclasses import dataclass

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.alpha101_formula import Alpha101FormulaStrategy


@dataclass
class _BarsMock:
    symbol_list: list[str]

    def __post_init__(self):
        self._rows = {
            symbol: {
                "datetime": None,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
            }
            for symbol in self.symbol_list
        }

    def set_bar(self, symbol, time_index, open_price, high_price, low_price, close_price, volume):
        self._rows[symbol] = {
            "datetime": time_index,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }

    def get_latest_bar_value(self, symbol, value_type):
        return self._rows[symbol][value_type]

    def get_latest_bar_datetime(self, symbol):
        return self._rows[symbol]["datetime"]


def _build_rows(length=80):
    rows = []
    for idx in range(length):
        if idx < 24:
            body = 0.05
        elif idx < 48:
            body = 0.35
        elif idx < 64:
            body = -0.35
        else:
            body = 0.0
        open_price = 100.0 + (0.15 * idx)
        close_price = open_price + body
        high_price = max(open_price, close_price) + 0.08
        low_price = min(open_price, close_price) - 0.08
        volume = 1000.0 + (5.0 * idx)
        rows.append((idx, open_price, high_price, low_price, close_price, volume))
    return rows


def _run_strategy(rows, split=None):
    params = {
        "alpha_id": 101,
        "history_window": 24,
        "score_window": 8,
        "entry_z": 0.9,
        "exit_z": 0.15,
        "signal_sign": 1.0,
        "stop_loss_pct": 0.03,
        "allow_short": True,
        "alpha_param_overrides": {"alpha101.101.const.001": 0.01},
    }

    def _feed(strategy, bars, events, chunk):
        for ts, open_price, high_price, low_price, close_price, volume in chunk:
            bars.set_bar("BTC/USDT", ts, open_price, high_price, low_price, close_price, volume)
            strategy.calculate_signals(
                MarketEvent(
                    ts,
                    "BTC/USDT",
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                )
            )
        out = []
        while not events.empty():
            signal = events.get()
            out.append((int(signal.datetime), str(signal.symbol), str(signal.signal_type)))
        return out

    if split is None:
        bars = _BarsMock(["BTC/USDT"])
        events = queue.Queue()
        strategy = Alpha101FormulaStrategy(bars, events, **params)
        return _feed(strategy, bars, events, rows)

    rows_a = rows[:split]
    rows_b = rows[split:]
    bars_a = _BarsMock(["BTC/USDT"])
    events_a = queue.Queue()
    strategy_a = Alpha101FormulaStrategy(bars_a, events_a, **params)
    signals_a = _feed(strategy_a, bars_a, events_a, rows_a)
    state = strategy_a.get_state()

    bars_b = _BarsMock(["BTC/USDT"])
    events_b = queue.Queue()
    strategy_b = Alpha101FormulaStrategy(bars_b, events_b, **params)
    strategy_b.set_state(state)
    signals_b = _feed(strategy_b, bars_b, events_b, rows_b)
    return [*signals_a, *signals_b]


class TestAlpha101FormulaStrategy(unittest.TestCase):
    def test_generates_long_short_and_exit_signals(self):
        signals = _run_strategy(_build_rows())
        self.assertGreater(len(signals), 0)
        signal_types = [signal_type for _, _, signal_type in signals]
        self.assertIn("LONG", signal_types)
        self.assertIn("SHORT", signal_types)
        self.assertIn("EXIT", signal_types)

    def test_state_roundtrip_preserves_signal_sequence(self):
        rows = _build_rows()
        full_signals = _run_strategy(rows)
        split_signals = _run_strategy(rows, split=48)
        self.assertEqual(full_signals, split_signals)


if __name__ == "__main__":
    unittest.main()
