from types import SimpleNamespace

from lumina_quant.core.engine import TradingEngine
from lumina_quant.core.events import MarketBatchEvent, MarketEvent


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
