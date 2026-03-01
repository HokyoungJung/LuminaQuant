from __future__ import annotations

import queue
from dataclasses import dataclass

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.candidate_vol_compression_reversion import (
    VolatilityCompressionReversionStrategy,
)
from lumina_quant.strategies.composite_trend import CompositeTrendStrategy
from lumina_quant.strategies.leadlag_spillover import LeadLagSpilloverStrategy
from lumina_quant.strategies.perp_crowding_carry import PerpCrowdingCarryStrategy


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
                "funding_rate": 0.0,
                "open_interest": 0.0,
                "liquidation_long_notional": 0.0,
                "liquidation_short_notional": 0.0,
            }
            for symbol in self.symbol_list
        }

    def set_bar(self, symbol: str, row: dict):
        self._rows[symbol].update(row)

    def get_latest_bar_value(self, symbol: str, value_type: str):
        return self._rows[symbol].get(value_type)

    def get_latest_bar_datetime(self, symbol: str):
        return self._rows[symbol].get("datetime")


def _feed_basic_series(strategy, bars: _BarsMock, symbol: str, n: int = 220) -> None:
    for idx in range(n):
        close = 100.0 + (0.06 * idx)
        row = {
            "datetime": idx,
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0 + (idx * 2.5),
        }
        bars.set_bar(symbol, row)
        strategy.calculate_signals(
            MarketEvent(
                time=idx,
                symbol=symbol,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
        )


def test_composite_trend_state_roundtrip():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = CompositeTrendStrategy(bars, events)

    _feed_basic_series(strategy, bars, "BTC/USDT", n=240)
    state = strategy.get_state()

    clone = CompositeTrendStrategy(_BarsMock(["BTC/USDT"]), queue.Queue())
    clone.set_state(state)

    assert clone.get_state() == state


def test_volcomp_reversion_state_roundtrip():
    bars = _BarsMock(["ETH/USDT"])
    events = queue.Queue()
    strategy = VolatilityCompressionReversionStrategy(bars, events)

    _feed_basic_series(strategy, bars, "ETH/USDT", n=260)
    state = strategy.get_state()

    clone = VolatilityCompressionReversionStrategy(_BarsMock(["ETH/USDT"]), queue.Queue())
    clone.set_state(state)

    assert clone.get_state() == state


def test_leadlag_state_roundtrip():
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    bars = _BarsMock(symbols)
    events = queue.Queue()
    strategy = LeadLagSpilloverStrategy(bars, events)

    for idx in range(260):
        for s_idx, symbol in enumerate(symbols):
            close = 100.0 + (s_idx * 5.0) + (0.05 * idx)
            row = {
                "datetime": idx,
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 900.0 + (s_idx * 30.0),
            }
            bars.set_bar(symbol, row)
            strategy.calculate_signals(
                MarketEvent(
                    time=idx,
                    symbol=symbol,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
            )

    state = strategy.get_state()
    clone = LeadLagSpilloverStrategy(_BarsMock(symbols), queue.Queue())
    clone.set_state(state)

    assert clone.get_state() == state


def test_perp_crowding_state_roundtrip():
    bars = _BarsMock(["BTC/USDT"])
    events = queue.Queue()
    strategy = PerpCrowdingCarryStrategy(bars, events)

    for idx in range(220):
        close = 100.0 + (0.04 * idx)
        row = {
            "datetime": idx,
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1100.0,
            "funding_rate": 0.0002 + (0.00005 * ((idx % 17) / 17.0)),
            "open_interest": 1_000_000 + (idx * 5_000),
            "liquidation_long_notional": 120_000 + (idx * 20),
            "liquidation_short_notional": 110_000 + (idx * 15),
        }
        bars.set_bar("BTC/USDT", row)
        event = type(
            "PerpMarketEvent",
            (),
            {
                "type": "MARKET",
                "time": idx,
                "datetime": idx,
                "symbol": "BTC/USDT",
                "close": row["close"],
                "funding_rate": row["funding_rate"],
                "open_interest": row["open_interest"],
                "liquidation_long_notional": row["liquidation_long_notional"],
                "liquidation_short_notional": row["liquidation_short_notional"],
            },
        )
        strategy.calculate_signals(event)

    state = strategy.get_state()
    clone = PerpCrowdingCarryStrategy(_BarsMock(["BTC/USDT"]), queue.Queue())
    clone.set_state(state)

    assert clone.get_state() == state
