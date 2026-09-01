from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.core.engine import TradingEngine
from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.strategy import Strategy


class _ContextStrategy(Strategy):
    preferred_contract = "context"
    required_inputs = ("market_window",)
    required_features = ("feature_points",)

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def calculate_signals(self, event):
        self.calls.append(("market", event))

    def calculate_signals_context(self, context):
        self.calls.append(("context", context))


class _WindowStrategy(Strategy):
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def calculate_signals(self, event):
        self.calls.append(("market", event))

    def calculate_signals_window(self, event, aggregator):
        self.calls.append(("window", aggregator))


class _OhlcvWindowStrategy(_WindowStrategy):
    required_inputs = ("ohlcv",)


class _AggregatorWindowStrategy(_WindowStrategy):
    uses_timeframe_aggregator = True
    required_timeframes = ("20s",)


class _FeatureStrategy(Strategy):
    required_features = ("funding_rate",)

    def calculate_signals(self, event):
        _ = event


class _Engine(TradingEngine):
    def on_fill(self, event):
        _ = event


def _build_engine(strategy: Strategy) -> _Engine:
    data_handler = SimpleNamespace(_feature_lookup=SimpleNamespace(db_path="data/market_parquet"))
    execution_handler = SimpleNamespace(exchange="exchange")
    return _Engine(
        events=[],
        data_handler=data_handler,
        strategy=strategy,
        portfolio=SimpleNamespace(update_timeindex=lambda _event: None),
        execution_handler=execution_handler,
    )


def _event() -> MarketWindowEvent:
    return MarketWindowEvent(
        time=1_700_000_000_000,
        window_seconds=5,
        bars_1s={
            "BTC/USDT": ((1_700_000_000_000, 10.0, 11.0, 9.0, 10.5, 100.0),),
        },
    )


def test_engine_prefers_context_callback_when_strategy_requests_context():
    strategy = _ContextStrategy()
    engine = _build_engine(strategy)

    engine.handle_market_window_event(_event())

    assert len(strategy.calls) == 1
    call_type, context = strategy.calls[0]
    assert call_type == "context"
    assert context.feature_lookup is not None
    assert context.feature_lookup.db_path == "data/market_parquet"
    assert context.provider_metadata["data_handler_class"] == "SimpleNamespace"


def test_engine_preserves_window_callback_for_legacy_strategies():
    strategy = _WindowStrategy()
    engine = _build_engine(strategy)

    engine.handle_market_window_event(_event())

    assert len(strategy.calls) == 1
    assert strategy.calls[0][0] == "window"
    assert strategy.calls[0][1] is None
    assert engine.timeframe_aggregator is None


def test_engine_treats_market_window_bars_as_ohlcv_input():
    strategy = _OhlcvWindowStrategy()
    engine = _build_engine(strategy)

    engine.handle_market_window_event(_event())

    assert len(strategy.calls) == 1
    assert strategy.calls[0][0] == "window"


def test_engine_builds_aggregator_only_for_explicit_aggregator_users():
    strategy = _AggregatorWindowStrategy()
    engine = _build_engine(strategy)

    engine.handle_market_window_event(_event())

    assert len(strategy.calls) == 1
    assert strategy.calls[0][0] == "window"
    assert strategy.calls[0][1] is engine.timeframe_aggregator
    assert engine.timeframe_aggregator is not None


def test_engine_fails_fast_when_required_features_are_unavailable():
    strategy = _FeatureStrategy()
    engine = _build_engine(strategy)
    engine.data_handler._feature_lookup = None

    try:
        engine.handle_market_window_event(_event())
    except RuntimeError as exc:
        assert "required_features" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected missing required_features to fail fast")
