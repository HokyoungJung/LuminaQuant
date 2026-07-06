import heapq
import os
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import deque
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader
from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.data.feature_points import FeaturePointLookup
from lumina_quant.market_data import resolve_symbol_csv_path

_EPOCH_ORIGIN = datetime(1970, 1, 1)
_EPOCH_UNIT_TO_TIMEDELTA_KWARG = {"us": "microseconds", "ms": "milliseconds"}
# Integer divisor turning a native epoch (in the datetime column's own unit)
# into epoch milliseconds. Proven byte-identical to `_bar_time_ms` of the
# reconstructed datetime for every non-negative epoch (see test coverage).
_EPOCH_MS_DIVISOR = {"microseconds": 1000, "milliseconds": 1}
_NUMERIC_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class _MsTimestampArray(np.ndarray):
    """int64 epoch-ms timestamps that stay truthy like the legacy list.

    Storing per-symbol millisecond timestamps as a packed int64 array (audit
    X3) removes the PyLong-list overhead (~40 B/row), but a bare ``ndarray``
    raises ``ValueError`` on the ``timestamps or ()`` truthiness idiom some
    consumers use. Overriding ``__bool__`` with a size test keeps every existing
    consumer working unchanged while indexing / iteration / ``searchsorted``
    behave exactly like any int64 array and the stored integers are identical.
    """

    def __bool__(self) -> bool:
        return self.size > 0


class _EpochTimestamps:
    """Epoch-encoded datetime column, materializing `datetime.datetime` lazily.

    Only constructed once `_try_encode_timestamps` has proven, row by row,
    that reconstructing from the encoded int64 epoch reproduces the exact
    original ``datetime.datetime`` values -- so this is never used unless it
    is behaviorally indistinguishable from keeping the original objects.
    """

    __slots__ = ("_epoch", "_timedelta_kwarg")

    def __init__(self, epoch: np.ndarray, timedelta_kwarg: str) -> None:
        self._epoch = epoch
        self._timedelta_kwarg = timedelta_kwarg

    def __len__(self) -> int:
        return len(self._epoch)

    def __getitem__(self, idx):
        return _EPOCH_ORIGIN + timedelta(**{self._timedelta_kwarg: int(self._epoch[idx])})


def _try_encode_timestamps(series: pl.Series, materialized: list[Any]) -> _EpochTimestamps | None:
    """Return a compact epoch-backed timestamp column, or None if unsafe.

    Requires a tz-naive `Datetime` dtype at "us"/"ms" precision (the only
    units representable exactly via `timedelta`), and then proves the
    encoding lossless by reconstructing every value and comparing it against
    the values Polars itself already materialized. Any dtype/precision this
    can't prove identical for (tz-aware, `Date`-only, "ns" precision, etc.)
    returns None so the caller falls back to keeping the plain object list.
    """
    dtype = series.dtype
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is not None:
        return None
    timedelta_kwarg = _EPOCH_UNIT_TO_TIMEDELTA_KWARG.get(dtype.time_unit)
    if timedelta_kwarg is None:
        return None
    try:
        epoch = series.dt.epoch(time_unit=dtype.time_unit).to_numpy().astype(np.int64, copy=False)
    except Exception:
        return None

    candidate = _EpochTimestamps(epoch, timedelta_kwarg)
    try:
        for i, original in enumerate(materialized):
            if candidate[i] != original:
                return None
    except OverflowError, ValueError:
        return None
    return candidate


def _numeric_columns_pack_lossless(df: Any) -> bool:
    """Return True iff every OHLCV numeric column packs into float64 losslessly.

    The columnar store reconstructs each numeric field as ``float(packed)`` from
    a single packed ``float64`` array. That is byte-identical to the legacy
    per-row object tuples only when nothing is silently corrupted on the way in.
    Two corruptions are guarded here, mirroring the timestamp lossless contract
    in ``_try_encode_timestamps``:

    * **null bars** -- Polars coerces a null to ``NaN`` in ``to_numpy``, whereas
      ``iter_rows`` materialized a Python ``None`` (which crashes loudly at the
      first ``float()``). Any null in a numeric column rejects the fast path.
    * **wide integers** -- an ``Int64`` value beyond float64's exact-integer
      range loses precision when packed. Non-``Float64`` numeric columns are
      accepted only when every value round-trips through ``float`` to the exact
      value Polars materialized; anything else keeps the legacy object rows.

    ``Float64`` columns (the production shape) are always safe once null-free, so
    they skip the per-value check -- and genuine ``NaN`` *floats* (distinct from
    nulls) are preserved exactly, matching legacy behavior.
    """
    for column in _NUMERIC_OHLCV_COLUMNS:
        series = df[column]
        if series.null_count() != 0:
            return False
        if series.dtype == pl.Float64:
            continue
        try:
            packed = series.to_numpy()
            for original, value in zip(series.to_list(), packed, strict=True):
                if float(value) != original:
                    return False
        except TypeError, ValueError:
            return False
    return True


class _ColumnarBarRows:
    """Columnar OHLCV row storage with lazy per-row tuple materialization.

    A per-symbol history used to be kept as one Python tuple per row
    (``(datetime, open, high, low, close, volume)``), which boxes five
    Python floats per row on top of the tuple's own overhead. This stores
    the five numeric fields as a single packed float64 numpy array, and the
    datetime/timestamp column as either a compact epoch-encoded column (when
    proven lossless -- see `_try_encode_timestamps`) or a plain Python list
    preserving the exact original objects. `__getitem__`/`__iter__`
    reconstruct the original row tuple on demand so every existing consumer
    of `symbol_rows[s]` (indexing, `len()`, iteration) keeps working
    unmodified.
    """

    __slots__ = ("_numeric", "_timestamps")

    def __init__(self, timestamps: list[Any] | _EpochTimestamps, numeric: np.ndarray) -> None:
        self._timestamps = timestamps
        self._numeric = numeric

    @classmethod
    def from_frame(cls, df: Any) -> _ColumnarBarRows | None:
        """Build columnar storage from a normalized OHLCV Polars frame.

        Column order follows `col_idx`: datetime, open, high, low, close, volume.
        Returns ``None`` when the numeric columns cannot be packed into float64
        without silent corruption (nulls or precision-losing wide integers), so
        the caller falls back to keeping the legacy per-row object tuples.
        """
        if not _numeric_columns_pack_lossless(df):
            return None
        n = df.height
        numeric = np.empty((n, 5), dtype=np.float64)
        numeric[:, 0] = df["open"].to_numpy()
        numeric[:, 1] = df["high"].to_numpy()
        numeric[:, 2] = df["low"].to_numpy()
        numeric[:, 3] = df["close"].to_numpy()
        numeric[:, 4] = df["volume"].to_numpy()

        materialized = df["datetime"].to_list()
        timestamps = _try_encode_timestamps(df["datetime"], materialized)
        return cls(timestamps if timestamps is not None else materialized, numeric)

    def __len__(self) -> int:
        return len(self._timestamps)

    def __bool__(self) -> bool:
        return len(self._timestamps) > 0

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self._timestamps))))
        n = len(self._timestamps)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("row index out of range")
        row = self._numeric[idx]
        return (
            self._timestamps[idx],
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
        )

    def __iter__(self):
        for i in range(len(self._timestamps)):
            yield self[i]


class DataHandler(ABC):
    """DataHandler abstract base class."""

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> tuple | None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> list[tuple]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_datetime(self, symbol: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars_values(self, symbol: str, val_type: str, N: int = 1) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def update_bars(self) -> None:
        raise NotImplementedError


class HistoricCSVDataHandler(DataHandler):
    """Historic data handler with timestamp-ordered merge and skip-ahead support."""

    def __init__(
        self,
        events,
        csv_dir,
        symbol_list,
        start_date=None,
        end_date=None,
        data_dict=None,
        feature_db_path: str | None = None,
        feature_exchange: str | None = None,
    ):
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date
        self.max_lookback = 5000  # Memory Cap (Safety)
        self.data_dict = data_dict  # Pre-loaded data support
        self._strict_data_dict = data_dict is not None
        self._single_symbol = len(symbol_list) == 1
        self._frame_loader = OHLCVFrameLoader(start_date=self.start_date, end_date=self.end_date)
        self._feature_lookup = self._build_feature_lookup(
            db_path=feature_db_path,
            exchange=feature_exchange,
            start_date=start_date,
            end_date=end_date,
        )

        # Value is `_ColumnarBarRows` for CSV/parquet-loaded symbols, or the raw
        # prefrozen tuple-of-tuples when a caller supplies pre-materialized rows
        # (identity of that tuple is preserved -- see test_data_handler_prefrozen).
        self.symbol_rows: dict[str, Any] = {}
        # list[int] for prefrozen / non-epoch symbols; a packed int64
        # `_MsTimestampArray` for columnar epoch-encoded symbols (audit X3).
        self.symbol_timestamps_ms: dict[str, Any] = {}
        self.symbol_index: dict[str, int] = {}
        self.next_bar: dict[str, tuple[Any, ...]] = {}
        self.finished_symbols = set()
        self._epoch_ms_prefrozen_symbols: set[str] = set()
        self._bar_heap: list[tuple[Any, int, str]] = []
        self._heap_seq = 0
        self.last_emitted_timestamp_ms: int | None = None

        self.latest_symbol_data = {s: deque(maxlen=self.max_lookback) for s in symbol_list}
        self.continue_backtest = True

        # Column Index Mapping for Speed
        self.col_idx = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }

        self._open_convert_csv_files()

    @staticmethod
    def _build_feature_lookup(
        *,
        db_path: str | None,
        exchange: str | None,
        start_date: Any = None,
        end_date: Any = None,
    ) -> FeaturePointLookup:
        resolved_db_path = db_path
        resolved_exchange = exchange
        if resolved_db_path is None or resolved_exchange is None:
            try:
                from lumina_quant.configuration import get_default_runtime_config

                _rt = get_default_runtime_config()
                _rt_storage = _rt.storage
            except Exception:
                _rt_storage = None
            if resolved_db_path is None:
                resolved_db_path = str(
                    getattr(_rt_storage, "market_data_parquet_path", "data/market_parquet")
                    or "data/market_parquet"
                )
            if resolved_exchange is None:
                resolved_exchange = str(
                    getattr(_rt_storage, "market_data_exchange", "binance") or "binance"
                )
        return FeaturePointLookup(
            db_path=resolved_db_path,
            exchange=str(resolved_exchange or "binance"),
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _bar_time_ms(bar_time: Any) -> int | None:
        if bar_time is None:
            return None
        if isinstance(bar_time, (int, float)):
            numeric = int(bar_time)
            if abs(numeric) < 100_000_000_000:
                return numeric * 1000
            return numeric
        if isinstance(bar_time, datetime):
            dt = bar_time if bar_time.tzinfo is not None else bar_time.replace(tzinfo=UTC)
            return int(dt.astimezone(UTC).timestamp() * 1000)
        try:
            dt = datetime.fromisoformat(str(bar_time).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.astimezone(UTC).timestamp() * 1000)
        except Exception:
            return None

    def _open_convert_csv_files(self):
        """Opens the CSV files using Polars and materializes tuple rows per symbol."""
        combined_data = self.data_dict if self.data_dict else {}

        for s in self.symbol_list:
            try:
                # Load from Memory or Disk
                if s in combined_data:
                    preloaded = combined_data[s]
                    if self._is_prefrozen_rows(preloaded):
                        rows = tuple(preloaded)
                        if self._is_epoch_ms_ohlcv_rows(rows):
                            self._epoch_ms_prefrozen_symbols.add(s)
                    else:
                        df = self._frame_loader.normalize(preloaded)
                        if df is None:
                            print(
                                "Warning: Missing or invalid OHLCV columns in "
                                f"{s}. Required: {self._frame_loader.columns}"
                            )
                            self.finished_symbols.add(s)
                            continue
                        rows = self._rows_from_frame(df)
                else:
                    if self._strict_data_dict:
                        # When explicit in-memory data_dict is supplied (chunked DB mode),
                        # never fallback to CSV to avoid hidden full-history loads.
                        self.finished_symbols.add(s)
                        continue
                    csv_path = self._resolve_symbol_csv_path(s)
                    if not os.path.exists(csv_path):
                        print(f"Warning: Data file not found for {s} at {csv_path}")
                        self.finished_symbols.add(s)
                        continue
                    df = self._frame_loader.load_csv(csv_path)
                    if df is None:
                        print(
                            "Warning: Missing or invalid OHLCV columns in "
                            f"{s}. Required: {self._frame_loader.columns}"
                        )
                        self.finished_symbols.add(s)
                        continue
                    rows = self._rows_from_frame(df)

                if not rows:
                    self.finished_symbols.add(s)
                    continue

                self.symbol_rows[s] = rows
                self.symbol_index[s] = 0
                self.next_bar[s] = rows[0]

                if s in self._epoch_ms_prefrozen_symbols:
                    timestamps_ms = [int(row[0]) for row in rows]
                else:
                    timestamps_ms = self._build_timestamps_ms(rows)
                self.symbol_timestamps_ms[s] = timestamps_ms
                self._push_heap(s, rows[0])
            except Exception as e:
                print(f"Dataset Load Error for {s}: {e}")
                self.finished_symbols.add(s)

        if not self.next_bar:
            self.continue_backtest = False

    @staticmethod
    def _rows_from_frame(df: Any) -> Any:
        """Columnar rows for a normalized frame, or legacy object tuples.

        Falls back to ``df.iter_rows`` (the exact pre-columnar materialization,
        preserving ``None`` for nulls and full integer precision) when the frame
        cannot be packed into float64 losslessly -- see
        ``_numeric_columns_pack_lossless``.
        """
        columnar = _ColumnarBarRows.from_frame(df)
        if columnar is not None:
            return columnar
        return tuple(df.iter_rows(named=False))

    def _build_timestamps_ms(self, rows: Any):
        """Per-row epoch-ms timestamps for a symbol's loaded rows.

        For columnar rows whose datetime column proved losslessly epoch-encoded
        (``_EpochTimestamps``), derive milliseconds directly from the packed
        int64 epoch buffer (audit X3) instead of materializing every row tuple,
        rebuilding a ``datetime`` and converting back. The result is an int64
        array (byte-identical integers, ~5x smaller than the PyLong list) that
        still iterates and compares like the legacy list; ``skip_to_timestamp_ms``
        consumes it via ``searchsorted``. Any layout this cannot prove identical
        for (non-epoch datetime encodings, legacy object rows, negative epochs)
        keeps the exact per-row path.
        """
        if isinstance(rows, _ColumnarBarRows):
            ts = rows._timestamps
            if isinstance(ts, _EpochTimestamps):
                divisor = _EPOCH_MS_DIVISOR.get(ts._timedelta_kwarg)
                epoch = ts._epoch
                if divisor is not None and epoch.size and int(epoch.min()) >= 0:
                    return (epoch // divisor).astype(np.int64).view(_MsTimestampArray)
        return [self._bar_time_ms(row[0]) or 0 for row in rows]

    def _resolve_symbol_csv_path(self, symbol):
        return resolve_symbol_csv_path(self.csv_dir, symbol)

    @staticmethod
    def _is_prefrozen_rows(value) -> bool:
        if not isinstance(value, (list, tuple)):
            return False
        if len(value) == 0:
            return True
        first = value[0]
        return isinstance(first, tuple) and len(first) >= 6

    @staticmethod
    def _is_epoch_ms_ohlcv_rows(rows) -> bool:
        if not isinstance(rows, (list, tuple)) or len(rows) == 0:
            return False
        first = rows[0]
        if not isinstance(first, tuple) or len(first) < 6:
            return False
        if isinstance(first[0], datetime):
            return False
        try:
            value = int(first[0])
        except Exception:
            return False
        return abs(value) >= 100_000_000_000

    def _push_heap(self, symbol, bar):
        heapq.heappush(self._bar_heap, (bar[0], self._heap_seq, symbol))
        self._heap_seq += 1

    def _rebuild_heap(self):
        self._bar_heap = []
        for symbol, bar in self.next_bar.items():
            self._push_heap(symbol, bar)

    def _advance_symbol(self, symbol: str) -> None:
        rows = self.symbol_rows.get(symbol)
        if not rows:
            self.next_bar.pop(symbol, None)
            self.finished_symbols.add(symbol)
            return

        idx = int(self.symbol_index.get(symbol, 0)) + 1
        if idx >= len(rows):
            self.symbol_index[symbol] = len(rows)
            self.next_bar.pop(symbol, None)
            self.finished_symbols.add(symbol)
            return

        self.symbol_index[symbol] = idx
        nxt = rows[idx]
        self.next_bar[symbol] = nxt
        self._push_heap(symbol, nxt)

    def skip_to_timestamp_ms(self, target_ts_ms: int | None) -> int:
        """Skip internal cursors to the first row >= target timestamp (binary search)."""
        if target_ts_ms is None:
            return 0
        target = int(target_ts_ms)

        moved = 0
        for symbol, rows in self.symbol_rows.items():
            if symbol in self.finished_symbols:
                continue
            ts_list = self.symbol_timestamps_ms.get(symbol)
            if not ts_list:
                continue
            idx = int(self.symbol_index.get(symbol, 0))
            if isinstance(ts_list, np.ndarray):
                # Whole-array searchsorted is equivalent to bisect_left(lo=idx):
                # the array is sorted ascending, so any global position <= idx
                # is filtered by the `next_idx <= idx` guard below.
                next_idx = int(np.searchsorted(ts_list, target, side="left"))
            else:
                next_idx = bisect_left(ts_list, target, lo=idx)
            if next_idx <= idx:
                continue

            moved += next_idx - idx
            if next_idx >= len(rows):
                self.symbol_index[symbol] = len(rows)
                self.next_bar.pop(symbol, None)
                self.finished_symbols.add(symbol)
            else:
                self.symbol_index[symbol] = next_idx
                self.next_bar[symbol] = rows[next_idx]

        if moved > 0:
            self._rebuild_heap()
            if not self.next_bar:
                self.continue_backtest = False
        return moved

    def get_next_timestamp_ms(self) -> int | None:
        if not self.next_bar:
            return None
        candidate = min((self._bar_time_ms(bar[0]) for bar in self.next_bar.values()), default=None)
        if candidate is None:
            return None
        return int(candidate)

    def update_bars(self):
        """Push bars in global timestamp order across symbols.

        Multi-symbol mode emits one `MarketBatchEvent` per timestamp to reduce
        event fan-out and allow `portfolio.update_timeindex` to run once.
        """
        if not self.next_bar:
            self.continue_backtest = False
            return

        if self._single_symbol:
            symbol = self.symbol_list[0]
            bar = self.next_bar.get(symbol)
            if bar is None:
                self.continue_backtest = False
                return

            self.latest_symbol_data[symbol].append(bar)
            self.last_emitted_timestamp_ms = self._bar_time_ms(bar[0])
            self.events.put(
                MarketEvent(
                    bar[0],
                    symbol,
                    bar[1],
                    bar[2],
                    bar[3],
                    bar[4],
                    bar[5],
                )
            )
            self._advance_symbol(symbol)
            if not self.next_bar:
                self.continue_backtest = False
            return

        selected_time = None
        emit_symbols = []

        while self._bar_heap:
            bar_time, _, symbol = heapq.heappop(self._bar_heap)
            current = self.next_bar.get(symbol)
            if current is None or current[0] != bar_time:
                continue
            selected_time = bar_time
            emit_symbols.append(symbol)
            break

        if selected_time is None:
            if self.next_bar:
                self._rebuild_heap()
            else:
                self.continue_backtest = False
            return

        while self._bar_heap and self._bar_heap[0][0] == selected_time:
            bar_time, _, symbol = heapq.heappop(self._bar_heap)
            current = self.next_bar.get(symbol)
            if current is None or current[0] != bar_time:
                continue
            emit_symbols.append(symbol)

        batch_events: list[MarketEvent] = []
        for s in emit_symbols:
            bar = self.next_bar[s]
            self.latest_symbol_data[s].append(bar)
            batch_events.append(
                MarketEvent(
                    bar[0],
                    s,
                    bar[1],
                    bar[2],
                    bar[3],
                    bar[4],
                    bar[5],
                )
            )
            self._advance_symbol(s)

        self.last_emitted_timestamp_ms = self._bar_time_ms(selected_time)

        if len(batch_events) == 1:
            self.events.put(batch_events[0])
        else:
            self.events.put(MarketBatchEvent(time=selected_time, bars=tuple(batch_events)))

        if not self.next_bar:
            self.continue_backtest = False

    def get_latest_bar(self, symbol):
        # Returns Tuple
        if not self.latest_symbol_data.get(symbol):
            return None
        return self.latest_symbol_data[symbol][-1]

    def get_latest_bars(self, symbol, N=1):
        # Returns List of Tuples
        history = self.latest_symbol_data.get(symbol)
        if not history:
            return []
        if N <= 0:
            return []
        size = len(history)
        if size <= N:
            return list(history)
        return list(islice(history, size - N, None))

    def get_latest_bar_datetime(self, symbol):
        if not self.latest_symbol_data.get(symbol):
            return None
        return self.latest_symbol_data[symbol][-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        idx = self.col_idx.get(val_type)
        if idx is not None and self.latest_symbol_data.get(symbol):
            return self.latest_symbol_data[symbol][-1][idx]
        feature_value = self.get_latest_feature_value(symbol, val_type)
        if feature_value is not None:
            return float(feature_value)
        return 0.0

    def get_latest_feature_value(self, symbol, field):
        bar_time = self.get_latest_bar_datetime(symbol)
        timestamp_ms = self._bar_time_ms(bar_time)
        return self._feature_lookup.get_latest(
            str(symbol),
            str(field),
            timestamp_ms=timestamp_ms,
        )

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """Returns last N values for a specific column."""
        history = self.latest_symbol_data.get(symbol)
        idx = self.col_idx.get(val_type)
        if idx is None or not history or N <= 0:
            return []
        size = len(history)
        if size <= N:
            return [bar[idx] for bar in history]
        return [bar[idx] for bar in islice(history, size - N, None)]

    def get_market_spec(self, symbol):
        _ = symbol
        return {}


__all__ = ["DataHandler", "HistoricCSVDataHandler"]
