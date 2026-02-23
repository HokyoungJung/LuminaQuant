import queue
from datetime import datetime, timedelta

from strategies.lag_convergence import LagConvergenceStrategy
from strategies.mean_reversion_std import MeanReversionStdStrategy
from strategies.rolling_breakout import RollingBreakoutStrategy
from strategies.vwap_reversion import VwapReversionStrategy


class _BarsMock:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)
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

    def set_bar(self, symbol, dt, open_price, high_price, low_price, close_price, volume):
        self._rows[symbol] = {
            "datetime": dt,
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


def _market_event(symbol, dt, close_price, high_price=None, low_price=None, volume=1.0):
    return type(
        "MarketEvent",
        (),
        {
            "type": "MARKET",
            "symbol": symbol,
            "time": dt,
            "datetime": dt,
            "close": float(close_price),
            "high": float(high_price if high_price is not None else close_price),
            "low": float(low_price if low_price is not None else close_price),
            "volume": float(volume),
        },
    )


def test_rolling_breakout_emits_long_signal():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = RollingBreakoutStrategy(
        bars,
        events,
        lookback_bars=5,
        breakout_buffer=0.0,
        allow_short=False,
    )

    start = datetime(2026, 1, 1)
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (101, 106, 100, 105),
    ]
    for idx, (open_price, high_price, low_price, close_price) in enumerate(rows):
        current_dt = start + timedelta(minutes=idx)
        bars.set_bar("BTC/USDT", current_dt, open_price, high_price, low_price, close_price, 10)
        strategy.calculate_signals(
            _market_event("BTC/USDT", current_dt, close_price, high_price, low_price, 10)
        )

    assert not events.empty()
    signal = events.get_nowait()
    assert signal.signal_type == "LONG"


def test_mean_reversion_std_emits_long_signal():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = MeanReversionStdStrategy(
        bars,
        events,
        window=8,
        entry_z=1.0,
        exit_z=0.3,
        allow_short=False,
    )

    start = datetime(2026, 1, 1)
    closes = [100, 102, 98, 101, 99, 100, 102, 98, 90]
    for idx, close_price in enumerate(closes):
        current_dt = start + timedelta(minutes=idx)
        bars.set_bar("BTC/USDT", current_dt, close_price, close_price, close_price, close_price, 10)
        strategy.calculate_signals(_market_event("BTC/USDT", current_dt, close_price, volume=10))

    assert not events.empty()
    signal = events.get_nowait()
    assert signal.signal_type == "LONG"


def test_vwap_reversion_emits_long_signal():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = VwapReversionStrategy(
        bars,
        events,
        window=8,
        entry_dev=0.05,
        exit_dev=0.01,
        allow_short=False,
    )

    start = datetime(2026, 1, 1)
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 85]
    for idx, close_price in enumerate(closes):
        current_dt = start + timedelta(minutes=idx)
        bars.set_bar("BTC/USDT", current_dt, close_price, close_price, close_price, close_price, 10)
        strategy.calculate_signals(_market_event("BTC/USDT", current_dt, close_price, volume=10))

    assert not events.empty()
    signal = events.get_nowait()
    assert signal.signal_type == "LONG"


def test_lag_convergence_emits_pair_entry_signals():
    bars = _BarsMock(["BTC/USDT", "ETH/USDT"])
    events = queue.Queue()
    strategy = LagConvergenceStrategy(
        bars,
        events,
        symbol_x="BTC/USDT",
        symbol_y="ETH/USDT",
        lag_bars=1,
        entry_threshold=0.05,
        exit_threshold=0.01,
        stop_threshold=0.2,
    )

    start = datetime(2026, 1, 1)
    bars.set_bar("BTC/USDT", start, 100, 100, 100, 100, 10)
    bars.set_bar("ETH/USDT", start, 100, 100, 100, 100, 10)
    strategy.calculate_signals(_market_event("BTC/USDT", start, 100, volume=10))

    later = start + timedelta(minutes=1)
    bars.set_bar("BTC/USDT", later, 120, 120, 120, 120, 10)
    bars.set_bar("ETH/USDT", later, 100, 100, 100, 100, 10)
    strategy.calculate_signals(_market_event("BTC/USDT", later, 120, volume=10))

    assert events.qsize() == 2
    first = events.get_nowait()
    second = events.get_nowait()
    assert {first.signal_type, second.signal_type} == {"SHORT", "LONG"}
