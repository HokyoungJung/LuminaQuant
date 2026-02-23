from __future__ import annotations

import queue
from datetime import datetime, timedelta

from strategies.candidate_regime_breakout import RegimeBreakoutCandidateStrategy
from strategies.candidate_vol_compression_reversion import VolatilityCompressionReversionStrategy


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


def test_regime_breakout_candidate_emits_long_signal():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = RegimeBreakoutCandidateStrategy(
        bars,
        events,
        lookback_window=20,
        slope_window=10,
        volatility_fast_window=8,
        volatility_slow_window=24,
        range_entry_threshold=0.65,
        max_volatility_ratio=2.5,
        allow_short=False,
    )

    start = datetime(2026, 1, 1)
    for idx in range(40):
        dt = start + timedelta(minutes=idx)
        close = 100.0 + idx * 0.4
        bars.set_bar("BTC/USDT", dt, close - 0.1, close + 0.5, close - 0.5, close, 20.0)
        strategy.calculate_signals(_market_event("BTC/USDT", dt, close, close + 0.5, close - 0.5, 20))

    assert not events.empty()
    signal = events.get_nowait()
    assert signal.signal_type == "LONG"


def test_volatility_compression_reversion_emits_long_signal():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = VolatilityCompressionReversionStrategy(
        bars,
        events,
        z_window=16,
        fast_vol_window=8,
        slow_vol_window=24,
        compression_threshold=3.0,
        entry_z=0.8,
        exit_z=0.3,
        allow_short=False,
    )

    start = datetime(2026, 1, 1)
    closes = [100.0] * 30 + [95.0]
    for idx, close in enumerate(closes):
        dt = start + timedelta(minutes=idx)
        bars.set_bar("BTC/USDT", dt, close, close + 0.2, close - 0.2, close, 10.0)
        strategy.calculate_signals(_market_event("BTC/USDT", dt, close, close + 0.2, close - 0.2, 10))

    assert not events.empty()
    signal = events.get_nowait()
    assert signal.signal_type == "LONG"
