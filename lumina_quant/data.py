import polars as pl
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from lumina_quant.events import MarketEvent


class DataHandler(ABC):
    """
    DataHandler abstract base class.
    """

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Tuple]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_datetime(self, symbol: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bars_values(
        self, symbol: str, val_type: str, N: int = 1
    ) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def update_bars(self) -> None:
        raise NotImplementedError


class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler using Polars for high performance.
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

        self.symbol_data = {}
        self.latest_symbol_data = {s: [] for s in symbol_list}
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
        self.finished_symbols = set()

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files using Polars and creates iterators.
        Filters by start_date and end_date if provided.
        """
        combined_data = {}
        if self.data_dict:
            combined_data = self.data_dict

        for s in self.symbol_list:
            try:
                # Load from Memory or Disk
                if s in combined_data:
                    df = combined_data[s]
                else:
                    # Load CSV with Polars
                    csv_path = self._resolve_symbol_csv_path(s)
                    if not os.path.exists(csv_path):
                        print(f"Warning: Data file not found for {s} at {csv_path}")
                        continue
                    df = pl.read_csv(csv_path, try_parse_dates=True)

                # Ensure correct column order for tuple unpacking
                # datetime, open, high, low, close, volume
                # Add missing cols if needed or reorder
                required_cols = ["datetime", "open", "high", "low", "close", "volume"]

                # Basic validation logic (omitted for speed, assumming standard format)
                # Check if columns exist
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Missing columns in {s}. Required: {required_cols}")
                    continue

                df = df.select(required_cols).sort("datetime")

                # Date Filtering
                if self.start_date:
                    df = df.filter(pl.col("datetime") >= self.start_date)
                if self.end_date:
                    df = df.filter(pl.col("datetime") <= self.end_date)

                # Convert to iterator of Tuples (much faster than Dicts)
                generator = df.iter_rows(named=False)
                self.data_generators[s] = generator

                # Prime first bar to support global timestamp-ordered merge
                first_bar = next(generator, None)
                if first_bar is None:
                    self.finished_symbols.add(s)
                    continue
                self.next_bar[s] = first_bar
            except Exception as e:
                print(f"Dataset Load Error for {s}: {e}")
                self.finished_symbols.add(s)

        if not self.next_bar:
            self.continue_backtest = False

    def _resolve_symbol_csv_path(self, symbol):
        candidates = [
            os.path.join(self.csv_dir, f"{symbol}.csv"),
            os.path.join(self.csv_dir, f"{symbol.replace('/', '')}.csv"),
            os.path.join(self.csv_dir, f"{symbol.replace('/', '_')}.csv"),
            os.path.join(self.csv_dir, f"{symbol.replace('/', '-')}.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        try:
            return next(self.data_generators[symbol])
        except StopIteration:
            self.finished_symbols.add(symbol)
            return None

    def update_bars(self):
        """
        Pushes bars in global timestamp order across symbols.
        If one symbol ends earlier, others continue until all data is exhausted.
        """
        if not self.next_bar:
            self.continue_backtest = False
            return

        # Find earliest timestamp among currently available bars
        min_time = min(bar[0] for bar in self.next_bar.values())
        emit_symbols = [s for s, bar in self.next_bar.items() if bar[0] == min_time]

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

            # MEMORY OPTIMIZATION: Rolling Window
            if len(self.latest_symbol_data[s]) > self.max_lookback:
                self.latest_symbol_data[s].pop(0)

            # Advance only symbol that was emitted
            nxt = self._get_new_bar(s)
            if nxt is None:
                self.next_bar.pop(s, None)
            else:
                self.next_bar[s] = nxt

        if not self.next_bar:
            self.continue_backtest = False

    def get_latest_bar(self, symbol):
        # Returns Tuple
        if not self.latest_symbol_data.get(symbol):
            return None
        return self.latest_symbol_data[symbol][-1]

    def get_latest_bars(self, symbol, N=1):
        # Returns List of Tuples
        return self.latest_symbol_data.get(symbol, [])[-N:]

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
        """
        Returns last N values for a specific column.
        """
        bars = self.get_latest_bars(symbol, N)
        idx = self.col_idx.get(val_type)
        if idx is not None:
            return [b[idx] for b in bars]
        return []

    def get_market_spec(self, symbol):
        _ = symbol
        return {}
