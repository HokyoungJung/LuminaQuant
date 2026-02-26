import threading
import time
from collections import deque

from lumina_quant.backtesting.data import DataHandler
from lumina_quant.config import BaseConfig
from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.core.protocols import ExchangeInterface
from lumina_quant.market_data import MarketDataRepository


class LiveDataHandler(DataHandler):
    """LiveDataHandler is designed to fetch live market data
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

        self.latest_symbol_data = {s: deque(maxlen=100) for s in symbol_list}
        self.lock = threading.Lock()
        self._last_persisted_1s_ts = dict.fromkeys(symbol_list, 0)
        self._market_repo = MarketDataRepository(str(BaseConfig.MARKET_DATA_PARQUET_PATH))
        self._market_exchange = str(getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance"))
        self._persist_interval_sec = max(
            10,
            int(float(getattr(self.config, "LIVE_INGEST_INTERVAL_SEC", 60) or 60)),
        )
        self._last_persist_monotonic = 0.0
        self._poll_seconds = max(
            1.0,
            float(
                getattr(
                    self.config,
                    "LIVE_POLL_SECONDS",
                    getattr(
                        self.config,
                        "POLL_SECONDS",
                        getattr(self.config, "POLL_INTERVAL", 20),
                    ),
                )
                or 20
            ),
        )
        self._poll_error_backoff_seconds = max(
            1.0,
            float(
                getattr(
                    self.config,
                    "POLL_ERROR_BACKOFF_SECONDS",
                    self._poll_seconds,
                )
                or self._poll_seconds
            ),
        )
        self._ingest_window_seconds = max(
            1,
            int(
                float(
                    getattr(
                        self.config,
                        "INGEST_WINDOW_SECONDS",
                        getattr(
                            self.config,
                            "LIVE_WINDOW_SECONDS",
                            getattr(
                                self.config,
                                "WINDOW_SECONDS",
                                self._poll_seconds,
                            ),
                        ),
                    )
                    or self._poll_seconds
                )
            ),
        )

        # Warmup Data
        self._warmup_data()

        # Start the polling thread
        self.polling_thread = threading.Thread(target=self._poll_market_data)
        self.polling_thread.daemon = True
        self.polling_thread.start()

    def _warmup_data(self):
        """Fetches historical data to warm up indicators."""
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
        """Polls market data in a loop."""
        print("Starting Live Data Polling...")
        while self.continue_backtest:
            try:
                now_mono = time.monotonic()
                should_persist_1s = (
                    now_mono - self._last_persist_monotonic >= self._persist_interval_sec
                )
                event_time = None
                bars_1s_batch = {}
                for s in self.symbol_list:
                    rows_1s = self.exchange.fetch_ohlcv(
                        s,
                        "1s",
                        limit=self._ingest_window_seconds,
                    ) or []
                    normalized = []
                    for row in rows_1s:
                        if not row or len(row) < 6:
                            continue
                        normalized.append(tuple(row[:6]))

                    bars_1s_batch[s] = tuple(normalized)
                    if normalized:
                        latest_bar = normalized[-1]
                        with self.lock:
                            last_ts = (
                                self.latest_symbol_data[s][-1][0]
                                if self.latest_symbol_data[s]
                                else 0
                            )
                            if int(latest_bar[0]) != int(last_ts):
                                self.latest_symbol_data[s].append(latest_bar)
                        event_time = latest_bar[0]

                    if should_persist_1s and normalized:
                        self._persist_1s_rows(s, normalized)

                if event_time is None:
                    event_time = int(time.time() * 1000)
                self.events.put(
                    MarketWindowEvent(
                        time=event_time,
                        window_seconds=int(self._ingest_window_seconds),
                        bars_1s=bars_1s_batch,
                    )
                )

                if should_persist_1s:
                    self._last_persist_monotonic = now_mono
                time.sleep(self._poll_seconds)

            except Exception as e:
                print(f"Error polling data: {e}")
                time.sleep(self._poll_error_backoff_seconds)

    def _persist_1s_rows(self, symbol, rows_1s):
        if not rows_1s:
            return

        last_seen = int(self._last_persisted_1s_ts.get(symbol, 0) or 0)
        new_rows = []
        newest = last_seen
        for row in rows_1s:
            if not row or len(row) < 6:
                continue
            ts = int(row[0])
            if ts <= last_seen:
                continue
            newest = max(newest, ts)
            new_rows.append(
                {
                    "datetime": ts,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )

        if not new_rows:
            return

        self._market_repo.upsert_ohlcv(
            exchange=self._market_exchange,
            symbol=symbol,
            timeframe="1s",
            rows=new_rows,
        )
        self._last_persisted_1s_ts[symbol] = newest

    def update_bars(self):
        """In live mode, the thread handles updates."""
        pass

    def get_latest_bar(self, symbol):
        with self.lock:
            return self.latest_symbol_data[symbol][-1]

    def get_latest_bars(self, symbol, N=1):
        with self.lock:
            return list(self.latest_symbol_data[symbol])[-N:]

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
            data = list(self.latest_symbol_data[symbol])[-N:]
            idx = self.col_idx.get(val_type)
            if idx is not None:
                return [d[idx] for d in data]
            return []

    def get_market_spec(self, symbol):
        if self.exchange and hasattr(self.exchange, "get_market_spec"):
            return self.exchange.get_market_spec(symbol)
        return {}

__all__ = ["LiveDataHandler"]
