import heapq
import os
from abc import ABC, abstractmethod
from collections import deque
from itertools import islice
from typing import Any

from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader
from lumina_quant.events import MarketEvent
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
    """HistoricCSVDataHandler using Polars for high performance.
    Optimized to use Tuple iteration (named=False) instead of Dictionaries.
    """

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
        self._single_symbol = len(symbol_list) == 1
        self._frame_loader = OHLCVFrameLoader(start_date=self.start_date, end_date=self.end_date)

        self.symbol_data = {}
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

        # Generators for iterating over data
        self.data_generators = {}
        self.next_bar = {}
        self._bar_heap = []
        self._heap_seq = 0
        self.finished_symbols = set()

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """Opens the CSV files using Polars and creates iterators.
        Filters by start_date and end_date if provided.
        """
        combined_data = {}
        if self.data_dict:
            combined_data = self.data_dict

        for s in self.symbol_list:
            try:
                # Load from Memory or Disk
                if s in combined_data:
                    preloaded = combined_data[s]
                    if self._is_prefrozen_rows(preloaded):
                        generator = iter(preloaded)
                        self.data_generators[s] = generator
                        first_bar = next(generator, None)
                        if first_bar is None:
                            self.finished_symbols.add(s)
                            continue
                        self.next_bar[s] = first_bar
                        self._push_heap(s, first_bar)
                        continue
                    df = self._frame_loader.normalize(preloaded)
                else:
                    # Load CSV with Polars
                    csv_path = self._resolve_symbol_csv_path(s)
                    if not os.path.exists(csv_path):
                        print(f"Warning: Data file not found for {s} at {csv_path}")
                        continue
                    df = self._frame_loader.load_csv(csv_path)

                if df is None:
                    print(
                        "Warning: Missing or invalid OHLCV columns in "
                        f"{s}. Required: {self._frame_loader.columns}"
                    )
                    continue

                # Convert to iterator of Tuples (much faster than Dicts)
                generator = df.iter_rows(named=False)
                self.data_generators[s] = generator

                # Prime first bar to support global timestamp-ordered merge
                first_bar = next(generator, None)
                if first_bar is None:
                    self.finished_symbols.add(s)
                    continue
                self.next_bar[s] = first_bar
                self._push_heap(s, first_bar)
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

    def _get_new_bar(self, symbol):
        """Returns the latest bar from the data feed."""
        try:
            return next(self.data_generators[symbol])
        except StopIteration:
            self.finished_symbols.add(symbol)
            return None

    def update_bars(self):
        """Pushes bars in global timestamp order across symbols.
        If one symbol ends earlier, others continue until all data is exhausted.
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

            nxt = self._get_new_bar(symbol)
            if nxt is None:
                self.next_bar.pop(symbol, None)
                self.continue_backtest = False
            else:
                self.next_bar[symbol] = nxt
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

        for s in emit_symbols:
            bar = self.next_bar[s]
            # bar is a Tuple: (datetime, open, high, low, close, volume)
            self.latest_symbol_data[s].append(bar)

            # Publish MarketEvent
            self.events.put(
                MarketEvent(
                    bar[0],  # datetime
                    s,
                    bar[1],  # open
                    bar[2],  # high
                    bar[3],  # low
                    bar[4],  # close
                    bar[5],  # volume
                )
            )

            # Advance only symbol that was emitted
            nxt = self._get_new_bar(s)
            if nxt is None:
                self.next_bar.pop(s, None)
            else:
                self.next_bar[s] = nxt
                self._push_heap(s, nxt)

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
