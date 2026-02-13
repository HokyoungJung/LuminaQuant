import time
import threading
from lumina_quant.events import MarketEvent
from lumina_quant.data import DataHandler
from lumina_quant.interfaces import ExchangeInterface


class LiveDataHandler(DataHandler):
    """
    LiveDataHandler is designed to fetch live market data
    from an ExchangeInterface and push MarketEvents to the queue.
    It uses a separate thread to poll data so the main loop isn't blocked.
    """

    def __init__(self, events, symbol_list, config, exchange: ExchangeInterface):
        self.events = events
        self.symbol_list = symbol_list
        self.config = config
        self.exchange = exchange

        # Column Index Mapping
        self.col_idx = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }

        self.continue_backtest = True  # Kept for compatibility, serves as "is_running"

        self.latest_symbol_data = {s: [] for s in symbol_list}
        self.lock = threading.Lock()

        # Warmup Data
        self._warmup_data()

        # Start the polling thread
        self.polling_thread = threading.Thread(target=self._poll_market_data)
        self.polling_thread.daemon = True
        self.polling_thread.start()

    def _warmup_data(self):
        """
        Fetches historical data to warm up indicators.
        """
        print("Warming up data buffers...")
        timeframe = getattr(self.config, "TIMEFRAME", "1m")
        for s in self.symbol_list:
            try:
                # Fetch N candles (e.g. 100)
                ohlcv = self.exchange.fetch_ohlcv(s, timeframe, limit=100)
                if ohlcv:
                    with self.lock:
                        for candle in ohlcv:
                            # Standardize format: timestamp, open, high, low, close, volume
                            # Store as Tuple for performance/consistency
                            self.latest_symbol_data[s].append(tuple(candle[:6]))

                    print(f"Loaded {len(ohlcv)} historical bars for {s}")
            except Exception as e:
                print(f"Warmup failed for {s}: {e}")

    def _poll_market_data(self):
        """
        Polls market data in a loop.
        """
        print("Starting Live Data Polling...")
        while self.continue_backtest:
            try:
                for s in self.symbol_list:
                    timeframe = getattr(self.config, "TIMEFRAME", "1m")
                    ohlcv = self.exchange.fetch_ohlcv(s, timeframe, limit=2)

                    if ohlcv:
                        # Strategy logic usually waits for close.
                        current_bar = ohlcv[-1]
                        # current_bar is [ts, o, h, l, c, v] tuple.
                        bar_tuple = tuple(current_bar[:6])

                        timestamp = bar_tuple[0]

                        # Ensure we don't process the same timestamp multiple times
                        last_ts = (
                            self.latest_symbol_data[s][-1][0]
                            if self.latest_symbol_data[s]
                            else 0
                        )

                        if timestamp != last_ts:
                            with self.lock:
                                self.latest_symbol_data[s].append(bar_tuple)
                                # Keep list short
                                if len(self.latest_symbol_data[s]) > 100:
                                    self.latest_symbol_data[s].pop(0)

                            print(f"New Bar: {s} @ {timestamp}")
                            self.events.put(
                                MarketEvent(
                                    timestamp,
                                    s,
                                    bar_tuple[1],  # open
                                    bar_tuple[2],  # high
                                    bar_tuple[3],  # low
                                    bar_tuple[4],  # close
                                    bar_tuple[5],  # volume
                                )
                            )

                time.sleep(getattr(self.config, "POLL_INTERVAL", 2))

            except Exception as e:
                print(f"Error polling data: {e}")
                time.sleep(5)

    def update_bars(self):
        """
        In live mode, the thread handles updates.
        """
        pass

    def get_latest_bar(self, symbol):
        with self.lock:
            return self.latest_symbol_data[symbol][-1]

    def get_latest_bars(self, symbol, N=1):
        with self.lock:
            return self.latest_symbol_data[symbol][-N:]

    def get_latest_bar_datetime(self, symbol):
        with self.lock:
            return self.latest_symbol_data[symbol][-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        with self.lock:
            idx = self.col_idx.get(val_type)
            if idx is not None:
                return self.latest_symbol_data[symbol][-1][idx]
            return 0.0

    def get_latest_bars_values(self, symbol, val_type, N=1):
        with self.lock:
            data = self.latest_symbol_data[symbol][-N:]
            idx = self.col_idx.get(val_type)
            if idx is not None:
                return [d[idx] for d in data]
            return []

    def get_market_spec(self, symbol):
        if self.exchange and hasattr(self.exchange, "get_market_spec"):
            return self.exchange.get_market_spec(symbol)
        return {}
