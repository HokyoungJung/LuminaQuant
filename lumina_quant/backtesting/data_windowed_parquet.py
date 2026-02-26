"""Windowed parquet/WAL-oriented backtest data handler."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any

from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.core.events import MarketWindowEvent


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
    ) -> None:
        super().__init__(
            events,
            csv_dir,
            symbol_list,
            start_date=start_date,
            end_date=end_date,
            data_dict=data_dict,
        )
        self.backtest_poll_seconds = max(1, int(backtest_poll_seconds))
        self.backtest_window_seconds = max(self.backtest_poll_seconds, int(backtest_window_seconds))
        self.backtest_poll_ms = int(self.backtest_poll_seconds * 1000)
        self.skip_ahead_step_ms = int(self.backtest_poll_ms)

        max_rows = max(64, int(self.backtest_window_seconds + self.backtest_poll_seconds + 4))
        self._window_rows: dict[str, deque[tuple[Any, ...]]] = {
            symbol: deque(maxlen=max_rows) for symbol in self.symbol_list
        }
        self._next_emit_ts_ms: int | None = None
        self._last_window_event_ms: int | None = None

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
        for symbol in emit_symbols:
            bar = self.next_bar[symbol]
            self.latest_symbol_data[symbol].append(bar)
            self._window_rows[symbol].append(bar)
            emitted_rows[symbol] = bar
            self._advance_symbol(symbol)

        self.last_emitted_timestamp_ms = self._bar_time_ms(selected_time)
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
            for row in rows:
                ts_ms = self._bar_time_ms(row[0])
                if ts_ms is None or int(ts_ms) < cutoff_ms:
                    continue
                scoped.append(row)
            snapshot[symbol] = tuple(scoped)
        return snapshot

    def _emit_window_event(self, event_time: Any) -> None:
        self.events.put(
            MarketWindowEvent(
                time=event_time,
                window_seconds=int(self.backtest_window_seconds),
                bars_1s=self._window_snapshot(),
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
        for rows in self._window_rows.values():
            kept = []
            for row in rows:
                ts_ms = self._bar_time_ms(row[0])
                if ts_ms is None:
                    continue
                if int(ts_ms) >= cutoff:
                    kept.append(row)
            rows.clear()
            rows.extend(kept)

        self._next_emit_ts_ms = self._align_emit_timestamp(target)
        return moved
