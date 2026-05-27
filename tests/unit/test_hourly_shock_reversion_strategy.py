from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.hourly_shock_reversion import HourlyShockReversionStrategy
from lumina_quant.timeframe_aggregator import TimeframeAggregator


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


class _FeatureLookup:
    def __init__(self, values: dict[str, float | None]) -> None:
        self.values = dict(values)

    def sum_between(self, _symbol, field, *, start_timestamp_ms, end_timestamp_ms):
        assert start_timestamp_ms <= end_timestamp_ms
        return self.values.get(field)


def _event(when: datetime, close: float) -> MarketEvent:
    return _event_for_symbol("ETH/USDT", when, close)


def _event_for_symbol(symbol: str, when: datetime, close: float) -> MarketEvent:
    return MarketEvent(
        time=when,
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _step(strategy, aggregator, when: datetime, close: float) -> None:
    event = _event(when, close)
    aggregator.update_from_1s_batch("ETH/USDT", [event])
    strategy.calculate_signals_window(event, aggregator)


def _step_with_features(
    strategy,
    aggregator,
    when: datetime,
    close: float,
    feature_lookup,
) -> None:
    event = _event(when, close)
    aggregator.update_from_1s_batch("ETH/USDT", [event])
    strategy.calculate_signals_window(event, aggregator, feature_lookup=feature_lookup)


def _step_symbols(strategy, aggregator, when: datetime, closes: dict[str, float]) -> None:
    events = {symbol: [_event_for_symbol(symbol, when, close)] for symbol, close in closes.items()}
    aggregator.update_from_1s_batch(events)
    strategy.calculate_signals_window(events["ETH/USDT"][0], aggregator)


def test_hourly_shock_reversion_fades_completed_negative_shock_once() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        max_hold_bars=48,
        target_allocation=0.008,
        max_order_value=175.0,
        stop_loss_pct=0.02,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0, 99.0]):
        _step(strategy, aggregator, start + timedelta(hours=offset), close)

    assert len(queue.items) == 1
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.symbol == "ETH/USDT"
    assert signal.strength == 0.008
    assert signal.price == 99.0
    assert signal.stop_loss == 99.0 * 0.98
    assert signal.metadata["reason"] == "negative_shock_reversion_long"
    assert signal.metadata["shock_return"] < -0.006
    assert signal.metadata["max_order_value"] == 175.0

    strategy.calculate_signals_window(
        _event(start + timedelta(hours=5, minutes=1), 99.0),
        aggregator,
    )
    assert len(queue.items) == 1


def test_hourly_shock_reversion_shorts_positive_shock_and_consumes_exit_bar() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        max_hold_bars=2,
        target_allocation=0.008,
        max_order_value=175.0,
        stop_loss_pct=0.02,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0]):
        _step(strategy, aggregator, start + timedelta(hours=offset), close)

    assert [item.signal_type for item in queue.items] == ["SHORT", "EXIT"]
    entry, exit_signal = queue.items
    assert entry.price == 101.0
    assert entry.stop_loss == 101.0 * 1.02
    assert entry.metadata["reason"] == "positive_shock_reversion_short"
    assert exit_signal.metadata["reason"] == "max_hold_exit"


def test_hourly_shock_reversion_respects_excluded_entry_hours() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        max_hold_bars=48,
        excluded_entry_hours_utc="4",
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0, 99.0]):
        _step(strategy, aggregator, start + timedelta(hours=offset), close)

    assert queue.items == []


def test_hourly_shock_reversion_requires_taker_flow_confirmation() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        flow_confirmation_lookback_bars=1,
        flow_imbalance_min=0.10,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)
    buy_dominant = _FeatureLookup({"taker_buy_quote_volume": 70.0, "taker_sell_quote_volume": 30.0})

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0, 99.0]):
        _step_with_features(
            strategy, aggregator, start + timedelta(hours=offset), close, buy_dominant
        )

    assert queue.items == []


def test_hourly_shock_reversion_accepts_confirming_taker_flow() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        flow_confirmation_lookback_bars=1,
        flow_imbalance_min=0.10,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)
    sell_dominant = _FeatureLookup(
        {"taker_buy_quote_volume": 30.0, "taker_sell_quote_volume": 70.0}
    )

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0, 99.0]):
        _step_with_features(
            strategy, aggregator, start + timedelta(hours=offset), close, sell_dominant
        )

    assert len(queue.items) == 1
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.metadata["flow_imbalance"] == -0.4
    assert signal.metadata["flow_source"] == "quote_volume"


def test_hourly_shock_reversion_counterguard_blocks_long_during_btc_downtrend() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["BTC/USDT", "ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        max_hold_bars=48,
        regime_symbol="BTC/USDT",
        regime_lookback_bars=4,
        counterguard_return_threshold=0.015,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset, (eth_close, btc_close) in enumerate(
        zip([100.0, 100.0, 100.0, 100.0, 99.0, 99.0], [100.0, 100.0, 100.0, 100.0, 98.0, 98.0])
    ):
        _step_symbols(
            strategy,
            aggregator,
            start + timedelta(hours=offset),
            {"ETH/USDT": eth_close, "BTC/USDT": btc_close},
        )

    assert queue.items == []
