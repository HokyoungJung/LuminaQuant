"""Incremental-sync cursors and materializer bounds must not depend on the host timezone.

Stored OHLCV ``datetime`` columns are tz-naive UTC wall times.  Reading the
cursor with a bare ``datetime.timestamp()`` shifted it by the host UTC offset:
east of UTC the next sync re-fetched hours of overlap, west of UTC it started
AFTER the true last bar and left a silent gap.  Same class of bug for the
raw-first materializer's ``end_date`` coercion.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import polars as pl
import pytest

from lumina_quant.market_data import MarketDataRepository
from lumina_quant.services.materialize_from_raw import _coerce_timestamp_ms

_EXPECTED_MS = int(datetime(2026, 3, 2, tzinfo=UTC).timestamp() * 1000)


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


def test_last_timestamp_cursors_are_utc_on_any_host(host_tz, tmp_path) -> None:
    repo = MarketDataRepository(str(tmp_path / "data"))
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 3, 1), datetime(2026, 3, 2)],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    repo.upsert_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1d", rows=frame)
    assert repo.get_last_timestamp_ms(exchange="binance", symbol="BTC/USDT", timeframe="1d") == (
        _EXPECTED_MS
    )
    rows_1s = pl.DataFrame(
        {
            "datetime": [datetime(2026, 3, 1, 23, 59, 59), datetime(2026, 3, 2)],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    repo.upsert_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1s", rows=rows_1s)
    assert repo.get_last_ohlcv_1s_timestamp_ms(exchange="binance", symbol="BTC/USDT") == (
        _EXPECTED_MS
    )


def test_materializer_bound_coercion_is_utc_on_any_host(host_tz) -> None:
    assert _coerce_timestamp_ms("2026-03-02") == _EXPECTED_MS
    assert _coerce_timestamp_ms("2026-03-02T00:00:00") == _EXPECTED_MS
    assert _coerce_timestamp_ms("2026-03-02T00:00:00Z") == _EXPECTED_MS
    assert _coerce_timestamp_ms(datetime(2026, 3, 2)) == _EXPECTED_MS
    assert _coerce_timestamp_ms(datetime(2026, 3, 2, tzinfo=UTC)) == _EXPECTED_MS
    assert _coerce_timestamp_ms(_EXPECTED_MS) == _EXPECTED_MS
    assert _coerce_timestamp_ms(_EXPECTED_MS // 1000) == _EXPECTED_MS
    assert _coerce_timestamp_ms(None) is None
    assert _coerce_timestamp_ms("not-a-date") is None


def test_timeutil_helpers_and_remaining_sites_are_utc_on_any_host(host_tz, tmp_path) -> None:
    from lumina_quant.live.readiness_policy import _parse_utc
    from lumina_quant.optimization.frozen_dataset import _as_ns_epoch
    from lumina_quant.timeframe_aggregator import TimeframeAggregator
    from lumina_quant.utils.timeutil import as_utc, utc_epoch_ms, utc_epoch_seconds

    naive = datetime(2026, 3, 2)
    aware = datetime(2026, 3, 2, tzinfo=UTC)
    assert as_utc(naive) == aware and as_utc(aware) == aware
    assert utc_epoch_seconds(naive) == utc_epoch_seconds(aware) == _EXPECTED_MS / 1000
    assert utc_epoch_ms(naive) == utc_epoch_ms(aware) == _EXPECTED_MS
    assert utc_epoch_ms("2026-03-02") == utc_epoch_ms("2026-03-02T00:00:00Z") == _EXPECTED_MS
    assert utc_epoch_ms(_EXPECTED_MS // 1000) == utc_epoch_ms(_EXPECTED_MS) == _EXPECTED_MS
    assert (
        utc_epoch_ms(None) is None
        and utc_epoch_ms("garbage") is None
        and utc_epoch_ms(True) is None
    )
    # Frozen-dataset split bounds and readiness stamps follow the same convention.
    assert _as_ns_epoch(naive) == _as_ns_epoch("2026-03-02T00:00:00") == _EXPECTED_MS * 1_000_000
    assert (
        _parse_utc("2026-03-02T00:00:00") == aware and _parse_utc("2026-03-02T00:00:00Z") == aware
    )
    # Timeframe aggregator bucket coercion (naive datetime path).
    assert TimeframeAggregator._coerce_timestamp_ms(naive) == _EXPECTED_MS
    assert TimeframeAggregator._coerce_timestamp_ms(_EXPECTED_MS) == _EXPECTED_MS


def test_raw_first_materializer_window_bounds_are_utc_on_any_host(host_tz, tmp_path) -> None:
    """The per-partition window bounds must match the raw timestamp_ms slice exactly."""
    import inspect

    from lumina_quant.services import materialize_from_raw as mod

    source = inspect.getsource(mod)
    assert ".timestamp() * 1000" not in source  # no host-local epoch conversions remain
    assert "utc_epoch_ms(dt_min)" in source and "utc_epoch_ms(dt_max)" in source
