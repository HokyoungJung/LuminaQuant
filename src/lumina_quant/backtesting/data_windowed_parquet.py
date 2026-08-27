"""Windowed parquet/WAL-oriented backtest data handler."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from lumina_quant.backtesting.data import _ColumnarBarRows, HistoricCSVDataHandler
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.core.market_window_contract import build_market_window_event


@dataclass(frozen=True, slots=True)
class RawPoint:
    """Already-emitted raw 1s close value with row/close timestamps."""

    value: float
    row_timestamp_ms: int
    close_timestamp_ms: int


class _EpochMsWindowRows:
    """Lazy MARKET_WINDOW row view over a columnar OHLCV store.

    Serves ``(epoch_ms, open, high, low, close, volume)`` tuples on demand from
    the packed float64 numeric array plus the precomputed epoch-ms timestamps,
    so the windowed handler keeps the columnar memory layout instead of
    re-boxing the entire per-symbol history into Python tuples (audit X2). The
    emitted tuples are byte-identical to the previously eager-frozen tuples;
    only rows actually consumed into the sliding window are ever materialized.
    """

    __slots__ = ("_epoch_ms", "_numeric")

    def __init__(self, epoch_ms: Any, numeric: Any) -> None:
        self._epoch_ms = epoch_ms
        self._numeric = numeric

    def __len__(self) -> int:
        return len(self._numeric)

    def __bool__(self) -> bool:
        return len(self._numeric) > 0

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self._numeric))))
        n = len(self._numeric)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("row index out of range")
        row = self._numeric[idx]
        return (
            int(self._epoch_ms[idx]),
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
        )

    def __iter__(self):
        for i in range(len(self._numeric)):
            yield self[i]


class HistoricParquetWindowedDataHandler(HistoricCSVDataHandler):
    """Emit one MARKET_WINDOW tick per poll cadence with rolling 1s windows.

    The handler reuses the existing parquet/csv-preloaded tuple ingestion path,
    then streams timestamp-ordered 1-second rows while emitting window snapshots.
    """

    def __init__(
        self,
        events,
        csv_dir,
        symbol_list,
        start_date=None,
        end_date=None,
        data_dict=None,
        *,
        backtest_poll_seconds: int = 20,
        backtest_window_seconds: int = 20,
        market_window_parity_v2_enabled: bool | None = None,
        feature_db_path: str | None = None,
        feature_exchange: str | None = None,
        feature_lookup: Any = None,
    ) -> None:
        if feature_lookup is not None and str(feature_db_path or "").strip():
            raise ValueError("feature_lookup cannot be combined with feature_db_path")

        parent_feature_db_path = "" if feature_lookup is not None else feature_db_path
        parent_feature_exchange = "binance" if feature_lookup is not None else feature_exchange
        super().__init__(
            events,
            csv_dir,
            symbol_list,
            start_date=start_date,
            end_date=end_date,
            data_dict=data_dict,
            feature_db_path=parent_feature_db_path,
            feature_exchange=parent_feature_exchange,
        )
        if feature_lookup is not None:
            self._feature_lookup = feature_lookup
        self._freeze_rows_as_epoch_ms()
        self.backtest_poll_seconds = max(1, int(backtest_poll_seconds))
        self.backtest_window_seconds = max(self.backtest_poll_seconds, int(backtest_window_seconds))
        self.backtest_poll_ms = int(self.backtest_poll_seconds * 1000)
        self.skip_ahead_step_ms = int(self.backtest_poll_ms)

        max_rows = max(64, int(self.backtest_window_seconds + self.backtest_poll_seconds + 4))
        self._window_rows: dict[str, deque[tuple[Any, ...]]] = {
            symbol: deque(maxlen=max_rows) for symbol in self.symbol_list
        }
        self._window_row_timestamps_ms: dict[str, deque[int | None]] = {
            symbol: deque(maxlen=max_rows) for symbol in self.symbol_list
        }
        self._next_emit_ts_ms: int | None = None
        self._last_window_event_ms: int | None = None
        if market_window_parity_v2_enabled is None:
            try:
                _mw = get_default_runtime_config().market_window
                self._parity_v2_enabled = bool(_mw.parity_v2_enabled)
                self._metrics_log_path = str(_mw.metrics_log_path)
            except Exception:
                self._parity_v2_enabled = False
                self._metrics_log_path = "logs/live/market_window_metrics.ndjson"
        else:
            self._parity_v2_enabled = bool(market_window_parity_v2_enabled)
            self._metrics_log_path = "logs/live/market_window_metrics.ndjson"

    def _freeze_rows_as_epoch_ms(self) -> None:
        """Convert loaded 1s rows once into the canonical MARKET_WINDOW shape.

        The generic CSV/parquet handler keeps row[0] as whatever Polars yields
        (usually `datetime`). MARKET_WINDOW construction normalizes those rows
        into `(epoch_ms, float, float, float, float, float)` on every poll. This
        windowed handler only emits MARKET_WINDOW events, so freezing rows once
        keeps the public event payload identical while removing repeated
        datetime conversion and float casting from the hot loop.
        """
        for symbol, rows in list(self.symbol_rows.items()):
            timestamps = self.symbol_timestamps_ms.get(symbol)
            if (
                not rows
                or timestamps is None
                or len(timestamps) == 0
                or len(rows) != len(timestamps)
            ):
                continue
            if symbol in getattr(self, "_epoch_ms_prefrozen_symbols", set()):
                idx = int(self.symbol_index.get(symbol, 0))
                if 0 <= idx < len(rows):
                    self.next_bar[symbol] = rows[idx]
                continue
            if isinstance(rows, _ColumnarBarRows):
                # Keep the columnar float64 buffer; serve epoch-ms tuples lazily
                # (audit X2) instead of re-materializing the full history.
                frozen: Any = _EpochMsWindowRows(timestamps, rows._numeric)
            else:
                # Legacy object rows (e.g. a null/wide-int fallback frame):
                # materialize eagerly, which also fails loudly on a null bar.
                frozen = tuple(
                    (
                        int(ts_ms),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    )
                    for ts_ms, row in zip(timestamps, rows, strict=False)
                )
                if not frozen:
                    continue
            self.symbol_rows[symbol] = frozen
            idx = int(self.symbol_index.get(symbol, 0))
            if 0 <= idx < len(frozen):
                self.next_bar[symbol] = frozen[idx]

        self._rebuild_heap()

    def _align_emit_timestamp(self, ts_ms: int) -> int:
        step = max(1, int(self.backtest_poll_ms))
        base = (int(ts_ms) // step) * step
        if base < int(ts_ms):
            base += step
        return int(base)

    def _consume_next_timestamp(self) -> tuple[Any, dict[str, tuple[Any, ...]]] | None:
        selected_time = None
        emit_symbols: list[str] = []

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
            return None

        while self._bar_heap and self._bar_heap[0][0] == selected_time:
            bar_time, _, symbol = heapq.heappop(self._bar_heap)
            current = self.next_bar.get(symbol)
            if current is None or current[0] != bar_time:
                continue
            emit_symbols.append(symbol)

        emitted_rows: dict[str, tuple[Any, ...]] = {}
        selected_ts_ms = self._bar_time_ms(selected_time)
        for symbol in emit_symbols:
            bar = self.next_bar[symbol]
            self.latest_symbol_data[symbol].append(bar)
            self._window_rows[symbol].append(bar)
            self._window_row_timestamps_ms[symbol].append(selected_ts_ms)
            emitted_rows[symbol] = bar
            self._advance_symbol(symbol)

        self.last_emitted_timestamp_ms = selected_ts_ms
        if not self.next_bar:
            self.continue_backtest = False
        return selected_time, emitted_rows

    def _window_snapshot(self) -> dict[str, tuple[Any, ...]]:
        snapshot: dict[str, tuple[Any, ...]] = {}
        current_ms = self.last_emitted_timestamp_ms
        if current_ms is None:
            return {symbol: tuple() for symbol in self.symbol_list}

        cutoff_ms = int(current_ms) - (int(self.backtest_window_seconds) * 1000) + 1000
        for symbol in self.symbol_list:
            rows = self._window_rows.get(symbol)
            if not rows:
                snapshot[symbol] = tuple()
                continue
            scoped = []
            timestamp_rows = self._window_row_timestamps_ms.get(symbol)
            if timestamp_rows is None or len(timestamp_rows) != len(rows):
                timestamp_rows = deque(
                    (self._bar_time_ms(row[0]) for row in rows), maxlen=rows.maxlen
                )
                self._window_row_timestamps_ms[symbol] = timestamp_rows
            for ts_ms, row in zip(timestamp_rows, rows, strict=False):
                if ts_ms is None or int(ts_ms) < cutoff_ms:
                    continue
                scoped.append(row)
            snapshot[symbol] = tuple(scoped)
        return snapshot

    def _emit_window_event(self, event_time: Any) -> None:
        self.events.put(
            build_market_window_event(
                time=event_time,
                window_seconds=int(self.backtest_window_seconds),
                bars_1s=self._window_snapshot(),
                event_time_watermark_ms=self.last_emitted_timestamp_ms,
                commit_id=None,
                lag_ms=0,
                is_stale=False,
                parity_v2_enabled=self._parity_v2_enabled,
                metrics_log_path=self._metrics_log_path,
                emit_metrics=False,
            )
        )
        if self.last_emitted_timestamp_ms is not None:
            self._last_window_event_ms = int(self.last_emitted_timestamp_ms)

    def update_bars(self) -> None:
        if not self.next_bar:
            self.continue_backtest = False
            return

        if self._next_emit_ts_ms is None:
            next_ts = self.get_next_timestamp_ms()
            if next_ts is None:
                self.continue_backtest = False
                return
            self._next_emit_ts_ms = self._align_emit_timestamp(int(next_ts))

        last_time: Any = None
        while True:
            consumed = self._consume_next_timestamp()
            if consumed is None:
                break
            event_time, _ = consumed
            last_time = event_time

            current_ms = self.last_emitted_timestamp_ms
            if (
                current_ms is not None
                and self._next_emit_ts_ms is not None
                and int(current_ms) >= int(self._next_emit_ts_ms)
            ):
                self._emit_window_event(event_time)
                while int(current_ms) >= int(self._next_emit_ts_ms):
                    self._next_emit_ts_ms += int(self.backtest_poll_ms)
                return

            if not self.next_bar:
                break

        # Flush final partial window at end-of-data.
        if last_time is not None and not self.next_bar:
            current_ms = self.last_emitted_timestamp_ms
            if current_ms is not None and int(current_ms) != int(self._last_window_event_ms or -1):
                self._emit_window_event(last_time)

        if not self.next_bar:
            self.continue_backtest = False

    def alpha_max_exact_columnar_view(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Expose the frozen one-day columns to the exact Alpha-Max tick reducer."""
        if (
            self.backtest_poll_seconds != 1
            or self.backtest_window_seconds != 1
            or not self._parity_v2_enabled
            or any(int(self.symbol_index.get(symbol, -1)) != 0 for symbol in self.symbol_list)
        ):
            raise ValueError("alpha_max_columnar_view_state_invalid")
        output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        expected_timestamps: np.ndarray | None = None
        for symbol in self.symbol_list:
            rows = self.symbol_rows.get(symbol)
            if type(rows) is not _EpochMsWindowRows:
                raise TypeError("alpha_max_columnar_rows_identity_invalid")
            timestamps = rows._epoch_ms
            numeric = rows._numeric
            if (
                not isinstance(timestamps, np.ndarray)
                or timestamps.dtype != np.dtype(np.int64)
                or timestamps.ndim != 1
                or type(numeric) is not np.ndarray
                or numeric.dtype != np.dtype(np.float64)
                or numeric.ndim != 2
                or numeric.shape != (timestamps.size, 5)
                or timestamps.size == 0
                or (timestamps.size > 1 and not bool(np.all(np.diff(timestamps) == 1000)))
                or not bool(np.all(np.isfinite(numeric)))
            ):
                raise ValueError("alpha_max_columnar_rows_invalid")
            if expected_timestamps is None:
                expected_timestamps = timestamps
            elif not np.array_equal(timestamps, expected_timestamps):
                raise ValueError("alpha_max_columnar_timeline_mismatch")
            output[symbol] = (timestamps, numeric)
        return output

    def alpha_max_advance_without_event(self, start_index: int, end_index: int) -> None:
        """Advance exact aligned rows while retaining the ordinary handler tails."""
        if (
            type(start_index) is not int
            or type(end_index) is not int
            or start_index < 0
            or end_index < start_index
            or any(
                int(self.symbol_index.get(symbol, -1)) != start_index for symbol in self.symbol_list
            )
        ):
            raise ValueError("alpha_max_handler_advance_invalid")
        for symbol in self.symbol_list:
            rows = self.symbol_rows.get(symbol)
            if type(rows) is not _EpochMsWindowRows or end_index >= len(rows):
                raise ValueError("alpha_max_handler_advance_invalid")
            window_rows = self._window_rows[symbol]
            window_timestamps = self._window_row_timestamps_ms[symbol]
            retained_start = max(start_index, end_index - int(window_rows.maxlen or 1) + 1)
            for index in range(retained_start, end_index + 1):
                row = rows[index]
                window_rows.append(row)
                window_timestamps.append(int(row[0]))
            latest = self.latest_symbol_data[symbol]
            latest_start = max(start_index, end_index - int(latest.maxlen or 1) + 1)
            for index in range(latest_start, end_index + 1):
                latest.append(rows[index])
            next_index = end_index + 1
            self.symbol_index[symbol] = next_index
            if next_index < len(rows):
                self.next_bar[symbol] = rows[next_index]
            else:
                self.next_bar.pop(symbol, None)
        self.last_emitted_timestamp_ms = int(self.symbol_rows[self.symbol_list[0]][end_index][0])
        self._last_window_event_ms = self.last_emitted_timestamp_ms
        self._next_emit_ts_ms = self.last_emitted_timestamp_ms + self.backtest_poll_ms
        self.continue_backtest = bool(self.next_bar)
        self._rebuild_heap()

    def skip_to_timestamp_ms(self, target_ts_ms: int | None) -> int:
        moved = super().skip_to_timestamp_ms(target_ts_ms)
        if moved <= 0 or target_ts_ms is None:
            return moved

        target = int(target_ts_ms)
        cutoff = target - (int(self.backtest_window_seconds) * 1000)
        for symbol, rows in self._window_rows.items():
            kept_rows = []
            kept_timestamps = []
            timestamp_rows = self._window_row_timestamps_ms.get(symbol)
            if timestamp_rows is None or len(timestamp_rows) != len(rows):
                timestamp_rows = deque(
                    (self._bar_time_ms(row[0]) for row in rows), maxlen=rows.maxlen
                )
                self._window_row_timestamps_ms[symbol] = timestamp_rows
            for ts_ms, row in zip(timestamp_rows, rows, strict=False):
                if ts_ms is None:
                    continue
                if int(ts_ms) >= cutoff:
                    kept_rows.append(row)
                    kept_timestamps.append(ts_ms)
            rows.clear()
            rows.extend(kept_rows)
            timestamp_rows.clear()
            timestamp_rows.extend(kept_timestamps)

        self._next_emit_ts_ms = self._align_emit_timestamp(target)
        return moved

    def get_latest_raw_point(
        self,
        symbol: str,
        field: str,
        *,
        timestamp_ms: int | None,
    ) -> RawPoint | None:
        """Return the latest already-emitted finite positive raw close point."""
        token = str(field or "").strip().lower()
        if token != "close" or int(timestamp_ms or 0) <= 0:
            return None
        query_ms = int(timestamp_ms)
        rows = self._window_rows.get(symbol)
        if not rows:
            return None
        timestamp_rows = self._window_row_timestamps_ms.get(symbol)
        if timestamp_rows is None or len(timestamp_rows) != len(rows):
            row_timestamps = tuple(self._bar_time_ms(row[0]) for row in rows)
        else:
            row_timestamps = tuple(timestamp_rows)

        latest: RawPoint | None = None
        for row_ts, row in zip(row_timestamps, rows, strict=False):
            if row_ts is None:
                continue
            row_timestamp_ms = int(row_ts)
            close_timestamp_ms = row_timestamp_ms + 1000
            if close_timestamp_ms > query_ms:
                continue
            try:
                value = float(row[4])
            except Exception:
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            candidate = RawPoint(
                value=value,
                row_timestamp_ms=row_timestamp_ms,
                close_timestamp_ms=close_timestamp_ms,
            )
            if latest is None or candidate.close_timestamp_ms > latest.close_timestamp_ms:
                latest = candidate
        return latest
