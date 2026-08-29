from __future__ import annotations

from queue import SimpleQueue

import pytest

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.indicators.oscillators import internal_bar_strength
from lumina_quant.strategies.registry import get_strategy_tier, resolve_strategy_class
from lumina_quant.strategies.trend_gated_ibs_reversion import (
    TrendGatedIbsReversionStrategy,
)


class _Bars:
    symbol_list = ["BTC/USDT"]


def _bar(time: int, open_: float, high: float, low: float, close: float) -> MarketEvent:
    return MarketEvent(time, "BTC/USDT", open_, high, low, close, 1.0)


def test_internal_bar_strength_edges() -> None:
    assert internal_bar_strength(10.0, 8.0, 8.5) == pytest.approx(0.25)
    assert internal_bar_strength(10.0, 10.0, 10.0) is None
    assert internal_bar_strength(10.0, 8.0, 11.0) is None
    assert internal_bar_strength(float("inf"), 8.0, 9.0) is None
    assert internal_bar_strength(10.0, 8.0, float("nan")) is None


def test_no_lookahead_entry_exit_state_and_registry() -> None:
    events: SimpleQueue = SimpleQueue()
    strategy = TrendGatedIbsReversionStrategy(_Bars(), events)

    for index in range(20):
        close = 100.0 + index
        open_ = close - 0.5 if index < 19 else close + 1.0
        strategy.calculate_signals(_bar(index, open_, close + 1.0, close - 1.0, close))
    assert events.empty()

    strategy.calculate_signals(_bar(20, 120.5, 121.0, 119.8, 120.0))
    entry = events.get_nowait()
    assert entry.signal_type == "LONG"
    assert entry.stop_loss == pytest.approx(116.4)
    assert entry.metadata["lagged_sma"] == pytest.approx(109.5)
    assert entry.metadata["target_allocation"] == pytest.approx(0.10)
    assert entry.metadata["max_order_value"] == pytest.approx(500.0)

    restored_events: SimpleQueue = SimpleQueue()
    restored = TrendGatedIbsReversionStrategy(_Bars(), restored_events)
    restored.set_state(strategy.get_state())
    assert restored.get_state() == strategy.get_state()

    restored.calculate_signals(
        MarketWindowEvent(
            time=21,
            window_seconds=86400,
            bars_1s={"BTC/USDT": ((21, 120.0, 123.0, 120.0, 122.5, 1.0),)},
        )
    )
    exit_signal = restored_events.get_nowait()
    assert exit_signal.signal_type == "EXIT"
    assert exit_signal.metadata["reason"] == "ibs_rebound"

    assert resolve_strategy_class("TrendGatedIbsReversionStrategy") is (
        TrendGatedIbsReversionStrategy
    )
    assert get_strategy_tier("TrendGatedIbsReversionStrategy") == "research_only"


@pytest.mark.parametrize("target_allocation", [0.0, -0.1])
def test_nonpositive_target_allocation_never_enters(target_allocation: float) -> None:
    events: SimpleQueue = SimpleQueue()
    strategy = TrendGatedIbsReversionStrategy(
        _Bars(), events, sma_window=2, target_allocation=target_allocation
    )

    strategy.calculate_signals(_bar(0, 100.0, 101.0, 99.0, 100.0))
    strategy.calculate_signals(_bar(1, 103.0, 103.0, 101.0, 102.0))
    strategy.calculate_signals(_bar(2, 103.0, 106.0, 102.0, 103.0))

    assert events.empty()
    assert strategy._state["BTC/USDT"].mode == "OUT"
