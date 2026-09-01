from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.cross_crypto_slow_diffusion import CrossCryptoSlowDiffusionStrategy
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


def _step(strategy, aggregator, when, btc_close, eth_close) -> None:
    btc = _event("BTC/USDT", when, btc_close)
    eth = _event("ETH/USDT", when, eth_close)
    aggregator.update_from_1s_batch("BTC/USDT", [btc])
    aggregator.update_from_1s_batch("ETH/USDT", [eth])
    strategy.calculate_signals_window(eth, aggregator)


def test_cross_crypto_slow_diffusion_uses_completed_hourly_leader_move_once() -> None:
    queue = _Queue()
    strategy = CrossCryptoSlowDiffusionStrategy(
        SimpleNamespace(symbol_list=["BTC/USDT", "ETH/USDT"]),
        queue,
        lag_bars=2,
        leader_abs_ret_min=0.015,
        target_underreaction_cap=999.0,
        max_hold_bars=2,
        target_allocation=0.008,
        max_order_value=175.0,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    _step(strategy, aggregator, start + timedelta(hours=0), 100.0, 100.0)
    _step(strategy, aggregator, start + timedelta(hours=1), 100.0, 100.0)
    _step(strategy, aggregator, start + timedelta(hours=2), 102.0, 100.5)
    _step(strategy, aggregator, start + timedelta(hours=3), 102.2, 100.7)

    assert len(queue.items) == 1
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.symbol == "ETH/USDT"
    assert signal.strength == 0.008
    assert signal.metadata["reason"] == "leader_up_target_lag_long"
    assert signal.metadata["max_order_value"] == 175.0

    strategy.calculate_signals_window(
        _event("ETH/USDT", start + timedelta(hours=3, minutes=1), 100.7), aggregator
    )
    assert len(queue.items) == 1

    _step(strategy, aggregator, start + timedelta(hours=4), 102.1, 100.8)
    assert len(queue.items) == 1
    _step(strategy, aggregator, start + timedelta(hours=5), 102.0, 100.9)
    assert len(queue.items) == 2
    assert queue.items[1].signal_type == "EXIT"
    assert queue.items[1].metadata["reason"] == "max_hold_exit"
