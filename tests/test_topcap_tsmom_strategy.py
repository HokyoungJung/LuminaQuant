import queue
import unittest
from dataclasses import dataclass

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.topcap_tsmom import TopCapTimeSeriesMomentumStrategy


@dataclass
class _MomentumBarStore:
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


def _build_price_rows(length=140):
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
    ]
    prices = {
        "BTC/USDT": 40000.0,
        "ETH/USDT": 2500.0,
        "BNB/USDT": 350.0,
        "SOL/USDT": 100.0,
        "XRP/USDT": 1.0,
        "ADA/USDT": 0.8,
    }

    rows = []
    split_point = int(length // 2)
    for idx in range(length):
        if idx < split_point:
            step = {
                "BTC/USDT": 0.0010,
                "ETH/USDT": 0.0030,
                "BNB/USDT": 0.0025,
                "SOL/USDT": 0.0020,
                "XRP/USDT": -0.0025,
                "ADA/USDT": -0.0020,
            }
        else:
            step = {
                "BTC/USDT": -0.0008,
                "ETH/USDT": -0.0028,
                "BNB/USDT": -0.0020,
                "SOL/USDT": -0.0023,
                "XRP/USDT": 0.0028,
                "ADA/USDT": 0.0022,
            }

        frame = {}
        for symbol in symbols:
            prices[symbol] = prices[symbol] * (1.0 + step[symbol])
            frame[symbol] = prices[symbol]
        rows.append((idx, frame))
    return symbols, rows


def _run_strategy(rows, symbols, split=None):
    params = {
        "lookback_bars": 8,
        "rebalance_bars": 2,
        "signal_threshold": 0.01,
        "stop_loss_pct": 0.08,
        "max_longs": 2,
        "max_shorts": 2,
        "min_price": 0.1,
        "btc_regime_ma": 0,
        "btc_symbol": "BTC/USDT",
    }

    def _feed(strategy, bars, events, chunk):
        for time_index, frame in chunk:
            for symbol in symbols:
                price = frame[symbol]
                bars.set_bar(symbol, time_index, price)
                event = MarketEvent(time_index, symbol, price, price, price, price, 1000.0)
                strategy.calculate_signals(event)

        out = []
        while not events.empty():
            signal = events.get()
            out.append((int(signal.datetime), str(signal.symbol), str(signal.signal_type)))
        return out

    if split is None:
        bars = _MomentumBarStore(symbols)
        events = queue.Queue()
        strategy = TopCapTimeSeriesMomentumStrategy(bars, events, **params)
        return _feed(strategy, bars, events, rows)

    chunk_a = rows[:split]
    chunk_b = rows[split:]

    bars_a = _MomentumBarStore(symbols)
    events_a = queue.Queue()
    strategy_a = TopCapTimeSeriesMomentumStrategy(bars_a, events_a, **params)
    signals_a = _feed(strategy_a, bars_a, events_a, chunk_a)
    saved_state = strategy_a.get_state()

    bars_b = _MomentumBarStore(symbols)
    events_b = queue.Queue()
    strategy_b = TopCapTimeSeriesMomentumStrategy(bars_b, events_b, **params)
    strategy_b.set_state(saved_state)
    signals_b = _feed(strategy_b, bars_b, events_b, chunk_b)
    return [*signals_a, *signals_b]


class TestTopCapTimeSeriesMomentumStrategy(unittest.TestCase):
    def test_generates_long_short_and_exit(self):
        symbols, rows = _build_price_rows()
        signals = _run_strategy(rows, symbols)
        self.assertGreater(len(signals), 0)

        signal_types = [signal_type for _, _, signal_type in signals]
        self.assertIn("LONG", signal_types)
        self.assertIn("SHORT", signal_types)
        self.assertIn("EXIT", signal_types)

    def test_state_roundtrip_preserves_signal_sequence(self):
        symbols, rows = _build_price_rows()
        full_signals = _run_strategy(rows, symbols)
        split_signals = _run_strategy(rows, symbols, split=80)
        self.assertEqual(full_signals, split_signals)


if __name__ == "__main__":
    unittest.main()
