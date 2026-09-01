from __future__ import annotations

import queue
from datetime import UTC, datetime, timedelta

import pytest

from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.data_windowed_parquet import (
    HistoricParquetWindowedDataHandler,
    RawPoint,
)


def _rows(
    start: datetime, closes: list[float]
) -> list[tuple[datetime, float, float, float, float, float]]:
    return [
        (
            start + timedelta(seconds=idx),
            close,
            close + 1.0,
            close - 1.0,
            close,
            10.0 + idx,
        )
        for idx, close in enumerate(closes)
    ]


def _handler(rows, **kwargs):
    return HistoricParquetWindowedDataHandler(
        queue.Queue(),
        "/unused",
        ["BTC/USDT"],
        data_dict={"BTC/USDT": rows},
        backtest_poll_seconds=1,
        backtest_window_seconds=8,
        market_window_parity_v2_enabled=False,
        **kwargs,
    )


class _InjectedLookup:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def get_latest(self, symbol: str, field: str, *, timestamp_ms: int | None):
        self.calls.append((symbol, field, timestamp_ms))
        return 0.125


def test_windowed_feature_kwargs_all_none_preserve_off_baseline():
    rows = _rows(datetime(2026, 1, 1, tzinfo=UTC), [100.0, 101.0])
    omitted = _handler(rows)
    explicit_none = _handler(rows, feature_db_path=None, feature_exchange=None, feature_lookup=None)

    assert omitted._feature_lookup.db_path == explicit_none._feature_lookup.db_path == ""
    assert omitted._feature_lookup.exchange == explicit_none._feature_lookup.exchange == "binance"

    omitted.update_bars()
    explicit_none.update_bars()

    assert omitted.last_emitted_timestamp_ms == explicit_none.last_emitted_timestamp_ms
    assert omitted.get_latest_bar("BTC/USDT") == explicit_none.get_latest_bar("BTC/USDT")
    assert omitted.get_latest_bar_value("BTC/USDT", "funding_rate") == 0.0
    assert explicit_none.get_latest_bar_value("BTC/USDT", "funding_rate") == 0.0


def test_windowed_injected_feature_lookup_is_identity_and_disables_parent_ambient(monkeypatch):
    calls = []
    original = HistoricCSVDataHandler._build_feature_lookup

    def recording_build_feature_lookup(*, db_path, exchange, start_date=None, end_date=None):
        calls.append((db_path, exchange, start_date, end_date))
        return original(
            db_path=db_path,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )

    monkeypatch.setattr(
        HistoricCSVDataHandler,
        "_build_feature_lookup",
        staticmethod(recording_build_feature_lookup),
    )
    rows = _rows(datetime(2026, 1, 1, tzinfo=UTC), [100.0])
    injected = _InjectedLookup()

    handler = _handler(rows, feature_lookup=injected)

    assert calls == [("", "binance", None, None)]
    assert handler._feature_lookup is injected
    handler.update_bars()
    assert handler.get_latest_bar_value("BTC/USDT", "funding_rate") == 0.125
    assert injected.calls == [("BTC/USDT", "funding_rate", handler.last_emitted_timestamp_ms)]

    with pytest.raises(ValueError, match="feature_lookup"):
        _handler(rows, feature_db_path="/nonempty", feature_lookup=injected)


def test_get_latest_raw_point_uses_only_emitted_completed_positive_close():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    rows = _rows(start, [100.0, 101.0, 102.0])
    handler = _handler(rows)

    assert handler.get_latest_raw_point("BTC/USDT", "close", timestamp_ms=start_ms + 1_000) is None

    handler.update_bars()
    assert handler.get_latest_raw_point("BTC/USDT", "close", timestamp_ms=start_ms) is None
    assert handler.get_latest_raw_point("BTC/USDT", "close", timestamp_ms=start_ms + 999) is None
    assert handler.get_latest_raw_point(
        "BTC/USDT", "close", timestamp_ms=start_ms + 1_000
    ) == RawPoint(
        value=100.0,
        row_timestamp_ms=start_ms,
        close_timestamp_ms=start_ms + 1_000,
    )
    assert (
        handler.get_latest_raw_point("BTC/USDT", "mark_price", timestamp_ms=start_ms + 1_000)
        is None
    )
    assert handler.get_latest_raw_point("ETH/USDT", "close", timestamp_ms=start_ms + 1_000) is None

    handler.update_bars()
    assert handler.get_latest_raw_point(
        "BTC/USDT", "close", timestamp_ms=start_ms + 1_000
    ) == RawPoint(
        value=100.0,
        row_timestamp_ms=start_ms,
        close_timestamp_ms=start_ms + 1_000,
    )
    assert handler.get_latest_raw_point(
        "BTC/USDT", "close", timestamp_ms=start_ms + 2_000
    ) == RawPoint(
        value=101.0,
        row_timestamp_ms=start_ms + 1_000,
        close_timestamp_ms=start_ms + 2_000,
    )


def test_get_latest_raw_point_skips_nonfinite_nonpositive_and_future_rows():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    rows = _rows(start, [100.0, 0.0, float("nan"), -1.0, 104.0])
    handler = _handler(rows)

    for _ in range(4):
        handler.update_bars()

    assert handler.get_latest_raw_point(
        "BTC/USDT", "close", timestamp_ms=start_ms + 4_000
    ) == RawPoint(
        value=100.0,
        row_timestamp_ms=start_ms,
        close_timestamp_ms=start_ms + 1_000,
    )
    assert handler.get_latest_raw_point("BTC/USDT", "open", timestamp_ms=start_ms + 4_000) is None

    handler.update_bars()
    assert handler.get_latest_raw_point(
        "BTC/USDT", "close", timestamp_ms=start_ms + 5_000
    ) == RawPoint(
        value=104.0,
        row_timestamp_ms=start_ms + 4_000,
        close_timestamp_ms=start_ms + 5_000,
    )


def test_get_latest_raw_point_reconstructs_timestamps_without_mutating_handler_state():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    rows = _rows(start, [100.0, 101.0])
    handler = _handler(rows)
    handler.update_bars()
    handler.update_bars()
    timestamp_rows = handler._window_row_timestamps_ms["BTC/USDT"]
    timestamp_rows.pop()
    before = tuple(timestamp_rows)

    point = handler.get_latest_raw_point("BTC/USDT", "close", timestamp_ms=start_ms + 2_000)

    assert point == RawPoint(
        value=101.0,
        row_timestamp_ms=start_ms + 1_000,
        close_timestamp_ms=start_ms + 2_000,
    )
    assert tuple(handler._window_row_timestamps_ms["BTC/USDT"]) == before
