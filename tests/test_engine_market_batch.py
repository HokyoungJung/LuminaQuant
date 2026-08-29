from types import SimpleNamespace

import pytest

from lumina_quant.core.engine import TradingEngine
from lumina_quant.core.events import MarketBatchEvent, MarketEvent, MarketWindowEvent


def _bar(symbol: str) -> MarketEvent:
    return MarketEvent(1, symbol, 1.0, 1.0, 1.0, 1.0, 1.0)


def _engine(strategy):
    return TradingEngine(
        events=None,
        data_handler=SimpleNamespace(_feature_lookup=None),
        strategy=strategy,
        portfolio=SimpleNamespace(update_timeindex=lambda event: None),
        execution_handler=SimpleNamespace(),
    )


def test_market_batch_hook_is_once_and_fallback_stays_per_bar():
    bars = (_bar("BTC/USDT"), _bar("ETH/USDT"))
    event = MarketBatchEvent(time=1, bars=bars)

    class BatchStrategy:
        def __init__(self):
            self.received = []

        def should_process_market_event(self, bar):
            return bar.symbol == "ETH/USDT"

        def calculate_signals_batch(self, batch):
            self.received.append(batch)

    batch_strategy = BatchStrategy()
    batch_engine = _engine(batch_strategy)
    batch_engine.handle_market_batch_event(event)

    assert batch_engine.market_events == 2
    assert len(batch_strategy.received) == 1
    assert batch_strategy.received[0].bars == (bars[1],)

    fallback_calls = []
    fallback_engine = _engine(SimpleNamespace(calculate_signals=fallback_calls.append))
    fallback_engine.handle_market_batch_event(event)

    assert fallback_engine.market_events == 2
    assert fallback_calls == list(bars)


def test_market_window_internal_type_error_does_not_fall_back_to_legacy_dispatch():
    class BrokenWindowStrategy:
        def __init__(self):
            self.window_calls = 0
            self.legacy_calls = 0

        def calculate_signals_window(self, event, aggregator):
            self.window_calls += 1
            raise TypeError("strategy implementation failure")

        def calculate_signals(self, event):
            self.legacy_calls += 1

    strategy = BrokenWindowStrategy()
    with pytest.raises(TypeError, match="strategy implementation failure"):
        _engine(strategy).handle_market_window_event(
            MarketWindowEvent(time=1, window_seconds=60, bars_1s={})
        )
    assert strategy.window_calls == 1
    assert strategy.legacy_calls == 0


def test_market_guard_failure_propagates_without_defaulting_to_processing():
    class BrokenGuardStrategy:
        def should_process_market_event(self, event):
            raise RuntimeError("guard implementation failure")

        def calculate_signals(self, event):
            raise AssertionError("must not be called")

    with pytest.raises(RuntimeError, match="guard implementation failure"):
        _engine(BrokenGuardStrategy()).handle_market_event(_bar("BTC/USDT"))
