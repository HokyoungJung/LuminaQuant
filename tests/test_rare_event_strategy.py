from __future__ import annotations

import queue
from datetime import datetime, timedelta

from lumina_quant.strategies.rare_event_score import RareEventScoreStrategy


class _BarsMock:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)
        self._rows = {
            symbol: {
                "datetime": None,
                "close": 0.0,
            }
            for symbol in self.symbol_list
        }

    def set_close(self, symbol, dt, close_price):
        self._rows[symbol] = {
            "datetime": dt,
            "close": float(close_price),
        }

    def get_latest_bar_value(self, symbol, value_type):
        return self._rows[symbol][value_type]

    def get_latest_bar_datetime(self, symbol):
        return self._rows[symbol]["datetime"]


def _market_event(symbol, dt, close_price):
    return type(
        "MarketEvent",
        (),
        {
            "type": "MARKET",
            "symbol": symbol,
            "time": dt,
            "datetime": dt,
            "close": float(close_price),
        },
    )


def _drain_signal_types(events):
    out = []
    while not events.empty():
        out.append(str(events.get_nowait().signal_type))
    return out


def test_rare_event_strategy_emits_long_on_rare_down_streak():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = RareEventScoreStrategy(
        bars,
        events,
        history_bars=128,
        trend_rolling_window=12,
        local_extremum_window=48,
        entry_score=0.90,  # keep test deterministic: streak condition drives entry
        exit_score=0.95,
        entry_streak=4,
        allow_short=False,
    )

    start = datetime(2026, 1, 1, 0, 0, 0)
    closes = [100.0 + ((idx % 2) * 0.15) - ((idx % 3) * 0.05) for idx in range(60)]
    closes.extend([99.0, 97.0, 94.0, 90.0, 86.0, 83.0])

    for idx, close in enumerate(closes):
        dt = start + timedelta(minutes=idx)
        bars.set_close("BTC/USDT", dt, close)
        strategy.calculate_signals(_market_event("BTC/USDT", dt, close))

    assert "LONG" in _drain_signal_types(events)


def test_rare_event_strategy_emits_short_on_rare_up_streak():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = RareEventScoreStrategy(
        bars,
        events,
        history_bars=128,
        trend_rolling_window=12,
        local_extremum_window=48,
        entry_score=0.90,
        exit_score=0.95,
        entry_streak=4,
        allow_short=True,
    )

    start = datetime(2026, 1, 1, 0, 0, 0)
    closes = [100.0 + ((idx % 2) * 0.12) - ((idx % 4) * 0.04) for idx in range(60)]
    closes.extend([101.0, 103.0, 106.0, 110.0, 115.0, 121.0])

    for idx, close in enumerate(closes):
        dt = start + timedelta(minutes=idx)
        bars.set_close("BTC/USDT", dt, close)
        strategy.calculate_signals(_market_event("BTC/USDT", dt, close))

    assert "SHORT" in _drain_signal_types(events)
