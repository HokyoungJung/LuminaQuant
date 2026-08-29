"""Timeframe alignment must treat naive bar times as UTC regardless of host TZ.

Regression for the legacy-backtest gate: ``TimeframeGatedStrategy`` tests
``event_ms % timeframe_ms == 0`` and ``_event_time_to_ms`` used to interpret
naive datetimes in the host's local timezone, so on a KST host every 1d/4h bar
was dropped and daily strategies silently produced zero signals.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from lumina_quant.backtesting.backtest import TimeframeGatedStrategy
from lumina_quant.core.engine import TradingEngine, _event_time_to_ms, _warmup_time_to_ms
from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.event_clock import normalize_timestamp_ns


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def calculate_signals(self, event) -> None:
        self.seen.append(event)


@pytest.fixture(params=["UTC", "Asia/Seoul", "America/New_York"])
def host_tz(request):
    if not hasattr(time, "tzset"):  # pragma: no cover - non-POSIX
        pytest.skip("time.tzset unavailable")
    previous = os.environ.get("TZ")
    os.environ["TZ"] = request.param
    time.tzset()
    try:
        yield request.param
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_naive_datetime_is_utc_and_matches_warmup_coercion(host_tz) -> None:
    naive = datetime(2026, 3, 1)
    aware = datetime(2026, 3, 1, tzinfo=UTC)
    assert _event_time_to_ms(naive) == _event_time_to_ms(aware) == 1_772_323_200_000
    assert _event_time_to_ms(naive) == _warmup_time_to_ms(naive)
    assert _event_time_to_ms(naive) % 86_400_000 == 0
    # Epoch inputs are untouched (seconds are promoted to ms).
    assert _event_time_to_ms(1_772_323_200) == 1_772_323_200_000
    assert _event_time_to_ms(1_772_323_200_000) == 1_772_323_200_000
    assert _event_time_to_ms(None) is None


@pytest.mark.parametrize("timeframe,hours", [("1d", 24), ("4h", 4), ("1h", 1)])
def test_gate_processes_aligned_naive_bars_on_any_host_timezone(host_tz, timeframe, hours) -> None:
    inner = _Recorder()
    gate = TimeframeGatedStrategy(inner, timeframe)
    aligned = SimpleNamespace(type="MARKET", time=datetime(2026, 3, 1, 0), symbol="BTC/USDT")
    next_bucket = SimpleNamespace(
        type="MARKET",
        time=datetime(2026, 3, 1, hours % 24) if hours < 24 else datetime(2026, 3, 2),
        symbol="BTC/USDT",
    )
    off_bucket = SimpleNamespace(type="MARKET", time=datetime(2026, 3, 1, 0, 30), symbol="BTC/USDT")
    assert gate.should_process_market_event(aligned) is True
    assert gate.should_process_market_event(aligned) is False  # same bucket, same symbol
    assert gate.should_process_market_event(off_bucket) is False
    assert gate.should_process_market_event(next_bucket) is True
    other_symbol = SimpleNamespace(type="MARKET", time=datetime(2026, 3, 1, 0), symbol="ETH/USDT")
    assert gate.should_process_market_event(other_symbol) is True


def test_event_identity_timestamp_is_host_timezone_independent(host_tz) -> None:
    naive = datetime(2026, 3, 1)
    aware = datetime(2026, 3, 1, tzinfo=UTC)
    assert normalize_timestamp_ns(naive) == normalize_timestamp_ns(aware)
    assert normalize_timestamp_ns(naive) == 1_772_323_200_000_000_000
    assert normalize_timestamp_ns("2026-03-01T00:00:00") == normalize_timestamp_ns(aware)
    assert normalize_timestamp_ns("2026-03-01T00:00:00Z") == normalize_timestamp_ns(aware)


def test_event_and_warmup_time_coercion_share_strict_utc_contract(host_tz) -> None:
    expected = 1_772_323_200_123
    values = (
        datetime(2026, 3, 1, 0, 0, 0, 123_000),
        "2026-03-01T00:00:00.123Z",
        "2026-03-01T09:00:00.123+09:00",
        1_772_323_200.123,
        expected,
    )
    for value in values:
        assert _event_time_to_ms(value) == expected
        assert _warmup_time_to_ms(value) == expected

    for value in (True, False, float("inf"), float("-inf"), float("nan")):
        assert _event_time_to_ms(value) is None
        assert _warmup_time_to_ms(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1.001, 1001),
        (-1.001, -1001),
        (datetime(1970, 1, 1, 0, 0, 1, 1000), 1001),
        (datetime(1969, 12, 31, 23, 59, 58, 999000), -1001),
        ("1970-01-01T00:00:01.001Z", 1001),
        ("1969-12-31T23:59:58.999+00:00", -1001),
    ),
)
def test_low_epoch_fractional_milliseconds_are_exact(value, expected) -> None:
    assert _event_time_to_ms(value) == expected
    assert _warmup_time_to_ms(value) == expected


def test_iso_cadence_is_timezone_independent_and_once_per_bucket(host_tz) -> None:
    calls: list[object] = []
    strategy = SimpleNamespace(
        decision_cadence_seconds=60,
        calculate_signals=calls.append,
    )
    engine = TradingEngine(
        events=None,
        data_handler=SimpleNamespace(_feature_lookup=None),
        strategy=strategy,
        portfolio=SimpleNamespace(update_timeindex=lambda event: None),
        execution_handler=SimpleNamespace(),
    )

    for value in (
        "2026-03-01T00:00:00Z",
        "2026-03-01T09:00:30+09:00",
        "2026-03-01T00:01:00Z",
    ):
        engine.handle_market_window_event(
            MarketWindowEvent(time=value, window_seconds=60, bars_1s={})
        )

    assert len(calls) == 2


@pytest.mark.parametrize("cadence", (True, "60", 1.5))
def test_invalid_cadence_declaration_fails_closed(cadence) -> None:
    engine = TradingEngine(
        events=None,
        data_handler=SimpleNamespace(_feature_lookup=None),
        strategy=SimpleNamespace(
            decision_cadence_seconds=cadence,
            calculate_signals=lambda _event: None,
        ),
        portfolio=SimpleNamespace(update_timeindex=lambda _event: None),
        execution_handler=SimpleNamespace(),
    )
    with pytest.raises(TypeError, match="decision_cadence_seconds"):
        engine.handle_market_window_event(MarketWindowEvent(time=1, window_seconds=60, bars_1s={}))


def test_active_cadence_rejects_unparseable_timestamp() -> None:
    engine = TradingEngine(
        events=None,
        data_handler=SimpleNamespace(_feature_lookup=None),
        strategy=SimpleNamespace(
            decision_cadence_seconds=60,
            calculate_signals=lambda _event: None,
        ),
        portfolio=SimpleNamespace(update_timeindex=lambda _event: None),
        execution_handler=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="decision timestamp"):
        engine.handle_market_window_event(
            MarketWindowEvent(time="not-a-time", window_seconds=60, bars_1s={})
        )


def test_aggregator_declaration_and_state_failures_propagate() -> None:
    def broken_timeframes():
        raise RuntimeError("declaration failed")

    engine = TradingEngine(
        events=None,
        data_handler=SimpleNamespace(_feature_lookup=None),
        strategy=SimpleNamespace(
            uses_timeframe_aggregator=True,
            required_timeframes=broken_timeframes,
        ),
        portfolio=SimpleNamespace(),
        execution_handler=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="declaration failed"):
        engine._ensure_timeframe_aggregator()

    engine.timeframe_aggregator = SimpleNamespace(
        get_state=lambda: (_ for _ in ()).throw(RuntimeError("state failed"))
    )
    with pytest.raises(RuntimeError, match="state failed"):
        engine.get_engine_state()
