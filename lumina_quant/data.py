import heapq
import os
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import deque
from datetime import UTC, datetime
from itertools import islice
from typing import Any

from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader
from lumina_quant.events import MarketBatchEvent, MarketEvent
from lumina_quant.market_data import resolve_symbol_csv_path


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

        self.symbol_rows: dict[str, tuple[tuple[Any, ...], ...]] = {}
        self.symbol_timestamps_ms: dict[str, list[int]] = {}
        self.symbol_index: dict[str, int] = {}
        self.next_bar: dict[str, tuple[Any, ...]] = {}
        self.finished_symbols = set()
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
                    else:
                        df = self._frame_loader.normalize(preloaded)
                        if df is None:
                            print(
                                "Warning: Missing or invalid OHLCV columns in "
                                f"{s}. Required: {self._frame_loader.columns}"
                            )
                            self.finished_symbols.add(s)
                            continue
                        rows = tuple(df.iter_rows(named=False))
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
                    rows = tuple(df.iter_rows(named=False))

                if not rows:
                    self.finished_symbols.add(s)
                    continue

                self.symbol_rows[s] = rows
                self.symbol_index[s] = 0
                self.next_bar[s] = rows[0]

                timestamps_ms = [self._bar_time_ms(row[0]) or 0 for row in rows]
                self.symbol_timestamps_ms[s] = timestamps_ms
                self._push_heap(s, rows[0])
            except Exception as e:
                print(f"Dataset Load Error for {s}: {e}")
                self.finished_symbols.add(s)

        if not self.next_bar:
            self.continue_backtest = False

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
        return 0.0

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
