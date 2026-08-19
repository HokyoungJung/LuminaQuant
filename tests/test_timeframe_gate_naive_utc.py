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
from lumina_quant.core.engine import _event_time_to_ms, _warmup_time_to_ms
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
