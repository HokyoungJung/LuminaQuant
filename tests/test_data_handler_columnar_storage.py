"""Storage-shape regression test for the audited memory optimization.

The backtest data handler used to materialize the whole OHLCV history as one
Python tuple per row (~6.7x more memory than a packed numpy layout). This
locks in the columnar replacement: consumer-visible tuple shape/values must
stay identical while the underlying per-symbol storage is columnar, not a
tuple of per-row Python tuples.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from lumina_quant.backtesting.data import (
    _ColumnarBarRows,
    _EpochTimestamps,
    HistoricCSVDataHandler,
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
