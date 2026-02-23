import asyncio
import json
import threading
from collections import deque

import websockets
from lumina_quant.data import DataHandler
from lumina_quant.events import MarketEvent


class BinanceWebSocketDataHandler(DataHandler):
    """Connects to Binance WebSocket for real-time trade/kline updates.
    significantly faster than polling.
    """

    def __init__(self, events, symbol_list, config, exchange=None):
        self.events = events
        self.symbol_list = symbol_list
        self.config = config
        self.exchange = exchange  # still needed for warmup

        # Internal State
        self.latest_symbol_data = {s: deque(maxlen=100) for s in symbol_list}
        self.lock = threading.Lock()
        self.ws_running = True
        self.continue_backtest = True  # Compatibility with engine/live trader stop flow

        # Column Map
        self.col_idx = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }

        # Warmup (IMPORTANT: Indicators need history)
        self._warmup_data()

        # Start Async Loop in a separate thread
        self.thread = threading.Thread(target=self._start_async_loop, daemon=True)
        self.thread.start()

    def _warmup_data(self):
        """Fetch historical data via REST API before connecting WS."""
        print("Warming up data buffers via REST...")
        timeframe = getattr(self.config, "TIMEFRAME", "1m")
        if self.exchange:
            for s in self.symbol_list:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(s, timeframe, limit=100)
                    if ohlcv:
                        with self.lock:
                            for candle in ohlcv:
                                self.latest_symbol_data[s].append(tuple(candle[:6]))
                    print(f"Loaded {len(ohlcv)} bars for {s}")
                except Exception as e:
                    print(f"Warmup failed for {s}: {e}")

    def _start_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._listen_socket())

    async def _listen_socket(self):
        """Connects to Binance Stream.
        URL format: wss://stream.binance.com:9443/stream?streams=<symbol>@kline_<interval>/...
        """
        base_url = "wss://stream.binance.com:9443/stream?streams="
        streams = []
        timeframe = getattr(self.config, "TIMEFRAME", "1m")

        for s in self.symbol_list:
            # Binance format: btcusdt@kline_1m
            symbol_lower = s.replace("/", "").lower()
            streams.append(f"{symbol_lower}@kline_{timeframe}")

        stream_url = base_url + "/".join(streams)
        print(f"Connecting to WebSocket: {stream_url[:50]}...")

        while self.ws_running:
            try:
                async with websockets.connect(stream_url) as ws:
                    print("✅ WebSocket Connected.")
                    while self.ws_running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        self._handle_message(data)
            except Exception as e:
                print(f"❌ WebSocket Connection Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def _handle_message(self, data):
        """Parses WebSocket message and pushes MarketEvent.
        Data format:
        {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "E": 123456789,
                "s": "BTCUSDT",
                "k": {
                    "t": 123400000, "T": 123460000,
                    "o": "0.0010", "c": "0.0020",
                    "h": "0.0025", "l": "0.0015",
                    "v": "1000", "x": false
                }
            }
        }
        """
        try:
            payload = data.get("data", {})
            kline = payload.get("k", {})
            symbol_raw = kline.get("s")  # BTCUSDT

            # Map raw symbol back to standard format (BTC/USDT)
            # Simple heuristic or map lookup
            # For now, let's assume we can match it against our symbol list normalized
            symbol = None
            for s in self.symbol_list:
                if s.replace("/", "") == symbol_raw:
                    symbol = s
                    break

            if not symbol:
                return

            # Check if Candle is Closed (x=True)
            # OR decide if we want to trade on partial candles (Riskier but faster)
            # For backtest consistency, usually we trade on CLOSE.
            # But "Speed" implies we might want to track price live.
            # Let's emit MarketEvent on every update, but Strategy decides.
            # Actually, standard logic waits for close.
            # Let's stick to Close for safety, OR emit update.
            # If we emit update every 200ms, strategy runs too often?
            # Let's emit ONLY when 'x': True (Bar Closed) for standard strategies.

            is_closed = kline.get("x", False)

            if is_closed:
                # Parse Data
                ts = kline.get("t")  # ms
                o = float(kline.get("o"))
                h = float(kline.get("h"))
                low_price = float(kline.get("l"))
                c = float(kline.get("c"))
                v = float(kline.get("v"))

                bar_tuple = (ts, o, h, low_price, c, v)

                with self.lock:
                    self.latest_symbol_data[symbol].append(bar_tuple)

                # Push Event
                print(f"⚡ WS Bar Closed: {symbol} @ {c}")
                self.events.put(MarketEvent(ts, symbol, o, h, low_price, c, v))

        except Exception as e:
            print(f"WS Parse Error: {e}")

    def update_bars(self):
        pass

    # GETTERS (Wrapped with Lock)
    def get_latest_bar(self, symbol):
        with self.lock:
            return self.latest_symbol_data[symbol][-1] if self.latest_symbol_data[symbol] else None

    def get_latest_bars(self, symbol, N=1):
        with self.lock:
            return list(self.latest_symbol_data[symbol])[-N:]

    def get_latest_bar_datetime(self, symbol):
        with self.lock:
            return (
                self.latest_symbol_data[symbol][-1][0] if self.latest_symbol_data[symbol] else None
            )

    def get_latest_bar_value(self, symbol, val_type):
        with self.lock:
            if not self.latest_symbol_data[symbol]:
                return 0.0
            idx = self.col_idx.get(val_type)
            return self.latest_symbol_data[symbol][-1][idx] if idx is not None else 0.0

    def get_latest_bars_values(self, symbol, val_type, N=1):
        with self.lock:
            data = list(self.latest_symbol_data[symbol])[-N:]
            idx = self.col_idx.get(val_type)
            return [d[idx] for d in data] if idx is not None else []

    def get_market_spec(self, symbol):
        if self.exchange and hasattr(self.exchange, "get_market_spec"):
            return self.exchange.get_market_spec(symbol)
        return {}
