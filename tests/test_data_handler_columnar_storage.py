"""Storage-shape regression test for the audited memory optimization.

The backtest data handler used to materialize the whole OHLCV history as one
Python tuple per row (~6.7x more memory than a packed numpy layout). This
locks in the columnar replacement: consumer-visible tuple shape/values must
stay identical while the underlying per-symbol storage is columnar, not a
tuple of per-row Python tuples.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from lumina_quant.backtesting.data import (
    _ColumnarBarRows,
    _EpochTimestamps,
    _MsTimestampArray,
    HistoricCSVDataHandler,
)
from lumina_quant.backtesting.data_windowed_parquet import (
    _EpochMsWindowRows,
    HistoricParquetWindowedDataHandler,
)


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _frame(start: datetime, n: int) -> pl.DataFrame:
    datetimes = [start + timedelta(minutes=i) for i in range(n)]
    base = [100.0 + i * 0.01 for i in range(n)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": base,
            "high": [v + 0.5 for v in base],
            "low": [v - 0.5 for v in base],
            "close": [v + 0.25 for v in base],
            "volume": [1000.0 + i for i in range(n)],
        }
    )


def test_dataframe_backed_symbol_uses_columnar_storage():
    start = datetime(2024, 1, 1)
    n = 50
    df = _frame(start, n)
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
    )

    stored = handler.symbol_rows["BTC/USDT"]
    assert isinstance(stored, _ColumnarBarRows)
    # Real tz-naive, microsecond-precision datetimes must take the compact
    # epoch-encoded fast path rather than a plain per-row object list.
    assert isinstance(stored._timestamps, _EpochTimestamps)
    assert len(stored) == n


def test_prefrozen_tuple_rows_are_not_converted_to_columnar():
    """Callers may hand in already-materialized row tuples (chunked DB mode).

    That path must keep storing the caller's raw tuple object unchanged
    (see test_data_handler_prefrozen.py for the identity guarantee some
    callers rely on) rather than being folded into columnar storage.
    """
    start = datetime(2024, 1, 1)
    rows = tuple(
        (start + timedelta(minutes=i), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0)
        for i in range(5)
    )
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": rows},
    )

    assert handler.symbol_rows["BTC/USDT"] is rows
    assert not isinstance(handler.symbol_rows["BTC/USDT"], _ColumnarBarRows)


def test_columnar_storage_round_trips_bars_identically():
    start = datetime(2024, 1, 1)
    n = 30
    df = _frame(start, n)
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
    )

    expected_rows = tuple(df.iter_rows(named=False))

    for i in range(n):
        handler.update_bars()
        bar = handler.get_latest_bar("BTC/USDT")
        expected = expected_rows[i]

        assert bar[0] == expected[0]
        assert isinstance(bar[0], datetime)
        for col, idx in handler.col_idx.items():
            if col == "datetime":
                continue
            assert bar[idx] == expected[idx]
            assert isinstance(bar[idx], float)

        assert handler.get_latest_bar_datetime("BTC/USDT") == expected[0]
        assert handler.get_latest_bar_value("BTC/USDT", "close") == expected[4]

    history = handler.get_latest_bars("BTC/USDT", N=5)
    assert [row[0] for row in history] == [row[0] for row in expected_rows[-5:]]
    assert handler.get_latest_bars_values("BTC/USDT", "volume", N=5) == [
        row[5] for row in expected_rows[-5:]
    ]


def test_columnar_storage_matches_csv_loaded_symbol(tmp_path):
    """The plain CSV-loading path (no data_dict) also uses columnar storage.

    Self-contained: writes its own CSV — repo data/ files are absent in the
    sanitized public-CI checkout, so tests must never depend on them.
    """
    rows = [
        "datetime,open,high,low,close,volume",
        "2024-01-01T00:00:00.000000,100.1,100.9,99.5,100.4,462",
        "2024-01-02T00:00:00.000000,100.4,101.2,99.9,100.8,681",
        "2024-01-03T00:00:00.000000,100.8,101.5,100.2,101.1,523",
    ]
    (tmp_path / "BTCUSDT.csv").write_text("\n".join(rows) + "\n")

    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir=str(tmp_path),
        symbol_list=["BTCUSDT"],
    )
    assert isinstance(handler.symbol_rows["BTCUSDT"], _ColumnarBarRows)
    assert len(handler.symbol_rows["BTCUSDT"]) == 3

    bar = handler.symbol_rows["BTCUSDT"][0]
    assert isinstance(bar[0], datetime)
    assert all(isinstance(v, float) for v in bar[1:])
    assert bar[handler.col_idx["close"]] == 100.4


# --------------------------------------------------------------------------- #
# N1: numeric lossless contract (nulls / integer dtypes / wide integers)
# --------------------------------------------------------------------------- #


def _frame_with(columns: dict[str, list]) -> pl.DataFrame:
    n = len(next(iter(columns.values())))
    base = {
        "datetime": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)],
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1000.0 + i for i in range(n)],
    }
    base.update(columns)
    return pl.DataFrame(base)


def _load(df: pl.DataFrame):
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
    )
    return handler.symbol_rows["BTC/USDT"]


def test_null_bar_falls_back_to_legacy_object_rows_preserving_none():
    """A null OHLCV field must NOT be silently coerced to NaN in columnar.

    Legacy per-row tuples kept a Python ``None`` (crashing loudly at the first
    ``float()``); the columnar float64 pack would turn it into a silent NaN.
    The lossless gate rejects that frame, keeping legacy object rows.
    """
    df = _frame_with({"open": [100.0, None, 102.0]})
    stored = _load(df)

    assert not isinstance(stored, _ColumnarBarRows)
    # The null survives as None (not NaN), reproducing legacy loud-failure.
    assert stored[1][1] is None
    with pytest.raises((TypeError, ValueError)):
        float(stored[1][1])
    # Non-null rows keep their exact values.
    assert stored[0][1] == 100.0
    assert stored[2][1] == 102.0


def test_integer_volume_packs_losslessly_into_columnar():
    """Small Int64 columns round-trip exactly through float64 -> columnar."""
    df = _frame_with({"volume": [1000, 1001, 1002]})
    assert df.schema["volume"] == pl.Int64
    stored = _load(df)

    assert isinstance(stored, _ColumnarBarRows)
    for i, expected in enumerate((1000.0, 1001.0, 1002.0)):
        assert stored[i][5] == expected
        assert isinstance(stored[i][5], float)


def test_wide_integer_beyond_float64_precision_falls_back():
    """An Int64 value past float64's exact range must not lose precision.

    ``float(2**53 + 1) == 2**53`` -- packing would corrupt it, so the frame
    falls back to legacy object rows preserving the exact integer.
    """
    big = 2**53 + 1
    df = _frame_with({"volume": [1000, big, 1002]})
    stored = _load(df)

    assert not isinstance(stored, _ColumnarBarRows)
    assert stored[1][5] == big  # exact int preserved
    assert float(stored[1][5]) != stored[1][5]  # float would have lost it


def test_mixed_integer_and_float_columns_pack_when_lossless():
    """Mixed dtypes (Int64 open + Float64 rest), all small, stay columnar."""
    df = _frame_with({"open": [100, 101, 102]})
    assert df.schema["open"] == pl.Int64
    assert df.schema["close"] == pl.Float64
    stored = _load(df)

    assert isinstance(stored, _ColumnarBarRows)
    assert stored[0][1] == 100.0
    assert isinstance(stored[0][1], float)


def test_genuine_nan_float_stays_on_columnar_fast_path():
    """A real NaN *float* (distinct from a null) is byte-identical either way."""
    df = _frame_with({"open": [100.0, float("nan"), 102.0]})
    assert df.schema["open"] == pl.Float64
    assert df["open"].null_count() == 0
    stored = _load(df)

    assert isinstance(stored, _ColumnarBarRows)
    assert np.isnan(stored[1][1])


# --------------------------------------------------------------------------- #
# X3: vectorized int64 epoch-ms timestamps + searchsorted skip
# --------------------------------------------------------------------------- #


def test_epoch_symbol_timestamps_are_int64_array_byte_identical():
    start = datetime(2024, 1, 1)
    n = 40
    df = _frame(start, n)
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
    )
    ts = handler.symbol_timestamps_ms["BTC/USDT"]

    assert isinstance(ts, _MsTimestampArray)
    assert isinstance(ts, np.ndarray)
    assert ts.dtype == np.int64
    # Byte-identical to the per-row _bar_time_ms path it replaces.
    expected = [handler._bar_time_ms(start + timedelta(minutes=i)) for i in range(n)]
    assert [int(v) for v in ts] == expected


def test_timestamp_array_survives_truthiness_or_idiom():
    """backtest._check_warmup does `for ts in ts_list or ()` -- must not raise."""
    df = _frame(datetime(2024, 1, 1), 10)
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
    )
    ts = handler.symbol_timestamps_ms["BTC/USDT"]
    collected = [int(v) for v in (ts or ())]
    assert len(collected) == 10
    empty = np.array([], dtype=np.int64).view(_MsTimestampArray)
    assert (empty or ()) == ()


def test_skip_to_timestamp_searchsorted_matches_bisect():
    start = datetime(2024, 1, 1)
    n = 60
    df = _frame(start, n)
    handler = HistoricCSVDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
    )
    ts = [int(v) for v in handler.symbol_timestamps_ms["BTC/USDT"]]
    target = ts[25] + 1  # lands between row 25 and 26
    moved = handler.skip_to_timestamp_ms(target)

    from bisect import bisect_left

    assert moved == bisect_left(ts, target)
    assert handler.symbol_index["BTC/USDT"] == bisect_left(ts, target)


# --------------------------------------------------------------------------- #
# X2: windowed handler serves columnar rows lazily (no full re-materialization)
# --------------------------------------------------------------------------- #


def test_windowed_handler_serves_columnar_rows_lazily():
    start = datetime(2024, 1, 1)
    n = 30
    df = _frame(start, n)
    handler = HistoricParquetWindowedDataHandler(
        events=_Queue(),
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
        backtest_poll_seconds=5,
        backtest_window_seconds=5,
    )
    stored = handler.symbol_rows["BTC/USDT"]

    # Lazy columnar view, not a re-boxed tuple-of-tuples.
    assert isinstance(stored, _EpochMsWindowRows)
    assert len(stored) == n

    # Byte-identical to the eager-frozen (epoch_ms, o, h, l, c, v) shape.
    expected_rows = list(df.iter_rows(named=False))
    for i in range(n):
        expected = expected_rows[i]
        row = stored[i]
        assert row[0] == handler._bar_time_ms(expected[0])
        assert isinstance(row[0], int)
        for col in range(1, 6):
            assert row[col] == expected[col]
            assert isinstance(row[col], float)


def test_windowed_handler_emits_identical_window_bars():
    start = datetime(2024, 1, 1)
    n = 12
    df = _frame(start, n)
    events = _Queue()
    handler = HistoricParquetWindowedDataHandler(
        events=events,
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        data_dict={"BTC/USDT": df},
        backtest_poll_seconds=60,
        backtest_window_seconds=60,
    )
    while handler.continue_backtest:
        handler.update_bars()

    assert events.items, "windowed handler emitted no MARKET_WINDOW events"
    # Every emitted 1s bar row must carry the exact epoch_ms + float OHLCV.
    seen = 0
    for event in events.items:
        for bars in event.bars_1s.values():
            for bar in bars:
                assert isinstance(bar[0], int)
                assert all(isinstance(v, float) for v in bar[1:])
                seen += 1
    assert seen > 0
