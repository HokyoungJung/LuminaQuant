"""Windowed parquet/WAL-oriented backtest data handler."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any

from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.core.market_window_contract import build_market_window_event


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
    ) -> None:
        super().__init__(
            events,
            csv_dir,
            symbol_list,
            start_date=start_date,
            end_date=end_date,
            data_dict=data_dict,
        )
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
            if not rows or not timestamps or len(rows) != len(timestamps):
                continue
            if symbol in getattr(self, "_epoch_ms_prefrozen_symbols", set()):
                idx = int(self.symbol_index.get(symbol, 0))
                if 0 <= idx < len(rows):
                    self.next_bar[symbol] = rows[idx]
                continue
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
