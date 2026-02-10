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
                    csv_path = os.path.join(self.csv_dir, f"{s}.csv")
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
                self.data_generators[s] = df.iter_rows(named=False)
            except Exception as e:
                print(f"Dataset Load Error for {s}: {e}")
                self.continue_backtest = False

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        try:
            return next(self.data_generators[symbol])
        except StopIteration:
            self.continue_backtest = False
            return None

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure.
        """
        for s in self.symbol_list:
            bar = self._get_new_bar(s)

            if bar is not None:
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
                # Prevent unbounded growth. Strategy needs usually < 1000 bars.
                if len(self.latest_symbol_data[s]) > self.max_lookback:
                    self.latest_symbol_data[s].pop(0)

            else:
                self.continue_backtest = False

    def get_latest_bar(self, symbol):
        # Returns Tuple
        return self.latest_symbol_data[symbol][-1]

    def get_latest_bars(self, symbol, N=1):
        # Returns List of Tuples
        return self.latest_symbol_data[symbol][-N:]

    def get_latest_bar_datetime(self, symbol):
        return self.latest_symbol_data[symbol][-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        idx = self.col_idx.get(val_type)
        if idx is not None:
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
