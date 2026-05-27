from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.timeframe_pair_zscore_reversion import (
    TimeframePairZScoreReversionStrategy,
)
from lumina_quant.timeframe_aggregator import TimeframeAggregator


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


def _event(symbol: str, when: datetime, close: float) -> MarketEvent:
    return MarketEvent(
        time=when,
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _step(strategy, aggregator, when: datetime, x_close: float, y_close: float) -> None:
    x_event = _event("XAU/USDT", when, x_close)
    y_event = _event("XAG/USDT", when, y_close)
    aggregator.update_from_1s_batch("XAU/USDT", [x_event])
    aggregator.update_from_1s_batch("XAG/USDT", [y_event])
    strategy.calculate_signals_window(x_event, aggregator)


def test_timeframe_pair_zscore_reversion_uses_completed_pair_bars_with_caps() -> None:
    queue = _Queue()
    strategy = TimeframePairZScoreReversionStrategy(
        SimpleNamespace(symbol_list=["XAU/USDT", "XAG/USDT"]),
        queue,
        symbol_x="XAU/USDT",
        symbol_y="XAG/USDT",
        timeframe="1h",
        lookback_window=12,
        hedge_window=24,
        entry_z=3.0,
        exit_z=0.4,
        stop_z=8.0,
        min_correlation=-1.0,
        max_hold_bars=3,
        target_allocation=0.021,
        max_order_value=310.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 80})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset in range(60):
        when = start + timedelta(hours=offset)
        y_close = 100.0 + (0.04 * offset)
        spread = 0.002 * math.sin(offset / 3.0)
        if offset == 50:
            spread = -0.050
        x_close = y_close * math.exp(spread)
        _step(strategy, aggregator, when, x_close, y_close)

    entries = [item for item in queue.items if item.signal_type in {"LONG", "SHORT"}]
    assert len(entries) >= 2
    assert entries[0].symbol == "XAU/USDT"
    assert entries[0].signal_type == "LONG"
    assert entries[1].symbol == "XAG/USDT"
    assert entries[1].signal_type == "SHORT"
    assert entries[0].strength == 0.021
    assert entries[0].metadata["target_allocation"] == 0.021
    assert entries[0].metadata["max_symbol_exposure_pct"] == 0.021
    assert entries[0].metadata["max_order_value"] == 310.0
    assert entries[0].metadata["timeframe"] == "1h"

    strategy.calculate_signals_window(
        _event("XAU/USDT", start + timedelta(hours=59, minutes=1), x_close),
        aggregator,
    )
    assert len([item for item in queue.items if item.signal_type in {"LONG", "SHORT"}]) == len(
        entries
    )
