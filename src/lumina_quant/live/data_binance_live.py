"""Real Binance live market-data handler with in-memory rolling MARKET_WINDOW aggregation."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.data.feature_points import FeaturePointLookup
from lumina_quant.live.binance_market_stream import (
    BinanceMarketStreamClient,
    BinanceMarketStreamConfig,
    build_trade_event_id,
    normalize_stream_symbol,
)
from lumina_quant.live.market_window_rolling import NormalizedTradeTick, RollingWindowAggregator


class BinanceLiveDataHandler:
    """Live data handler backed by Binance trade/aggTrade streams."""

    def __init__(
        self,
        events,
        symbol_list,
        config,
        exchange=None,
        *,
        transport: str = "ws",
    ) -> None:
        self.events = events
        self.symbol_list = [str(symbol) for symbol in list(symbol_list or [])]
        self.config = config
        self.exchange = exchange
        self.transport = str(transport or "ws").strip().lower()
        if self.transport not in {"ws", "poll"}:
            self.transport = "ws"
        self._feature_lookup = FeaturePointLookup(
            db_path=str(getattr(self.config, "MARKET_DATA_PARQUET_PATH", "data/market_parquet")),
            exchange=str(getattr(self.config, "MARKET_DATA_EXCHANGE", "binance")),
        )

        self.continue_backtest = True
        self._shutdown = threading.Event()
        self.lock = threading.Lock()
        self.latest_symbol_data = {symbol: deque(maxlen=500) for symbol in self.symbol_list}
        self.col_idx = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }

        self._poll_seconds = max(
            1.0,
            float(
                getattr(self.config, "LIVE_POLL_SECONDS", getattr(self.config, "POLL_SECONDS", 2))
                or 2
            ),
        )
        self._window_seconds = max(
            1,
            int(
                getattr(
                    self.config,
                    "INGEST_WINDOW_SECONDS",
                    getattr(self.config, "WINDOW_SECONDS", 20),
                )
                or 20
            ),
        )
        self._book_ticker_enabled = bool(getattr(self.config, "BOOK_TICKER_ENABLED", False))
        self._max_lateness_ms = max(
            0, int(getattr(self.config, "LIVE_MAX_LATENESS_MS", 1500) or 1500)
        )

        self.aggregator = RollingWindowAggregator(
            symbol_list=list(self.symbol_list),
            window_seconds=int(self._window_seconds),
            max_lateness_ms=int(self._max_lateness_ms),
        )

        self._cursor_ms: dict[str, int | None] = dict.fromkeys(self.symbol_list)
        self._fatal_channel: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._fatal_error: BaseException | None = None
        self._ws_consecutive_errors = 0
        self._ws_max_consecutive_errors = max(
            3,
            int(getattr(self.config, "LIVE_WS_MAX_CONSECUTIVE_ERRORS", 12) or 12),
        )

        self._thread = threading.Thread(target=self._run_loop, daemon=False)
        self._thread.start()

    def _publish_fatal(self, exc: BaseException) -> None:
        if self._fatal_error is not None:
            return
        self._fatal_error = exc
        self.continue_backtest = False
        self._shutdown.set()
        try:
            self._fatal_channel.put_nowait(exc)
        except queue.Full:
            pass

    def consume_fatal_error(self) -> BaseException | None:
        try:
            return self._fatal_channel.get_nowait()
        except queue.Empty:
            return None

    def poll_fatal_error(self) -> BaseException | None:
        return self.consume_fatal_error()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self.continue_backtest = False
        self._shutdown.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and getattr(thread, "is_alive", lambda: False)():
            thread.join(timeout=max(0.1, float(join_timeout)))

    def shutdown(self, join_timeout: float = 5.0) -> None:
        self.stop(join_timeout=join_timeout)

    def _run_loop(self) -> None:
        try:
            if self.transport == "poll":
                self._run_poll_loop()
            else:
                self._run_ws_loop()
        except Exception as exc:  # pragma: no cover - defensive live fatal
            self._publish_fatal(exc)

    def _run_ws_loop(self) -> None:
        config = BinanceMarketStreamConfig(
            symbols=list(self.symbol_list),
            include_book_ticker=bool(self._book_ticker_enabled),
            use_agg_trade=True,
        )
        client = BinanceMarketStreamClient(config)

        def _on_trade(tick: NormalizedTradeTick) -> None:
            self._ws_consecutive_errors = 0
            self._emit_from_tick(tick)

        def _on_error(exc: Exception) -> None:
            if self._shutdown.is_set() or not self.continue_backtest:
                return
            self._ws_consecutive_errors += 1
            if isinstance(exc, (ImportError, ModuleNotFoundError)):
                self._publish_fatal(exc)
                return
            if self._ws_consecutive_errors >= int(self._ws_max_consecutive_errors):
                self._publish_fatal(
                    RuntimeError(
                        "Binance websocket exceeded max consecutive errors: "
                        f"{self._ws_consecutive_errors} ({exc})"
                    )
                )

        client.run_ws_loop(
            stop_event=self._shutdown,
            on_trade=_on_trade,
            on_error=_on_error,
        )

    def _run_poll_loop(self) -> None:
        while self.continue_backtest and not self._shutdown.is_set():
            now_ms = int(time.time() * 1000)
            for symbol in list(self.symbol_list):
                for tick in self._fetch_symbol_ticks(symbol=symbol):
                    self._emit_from_tick(tick)
            for event in self.aggregator.flush_until(now_ms=now_ms):
                self._push_market_window(event)
            if self.continue_backtest and not self._shutdown.is_set():
                time.sleep(self._poll_seconds)

    def _push_market_window(self, event: MarketWindowEvent) -> None:
        with self.lock:
            for symbol, rows in dict(getattr(event, "bars_1s", {}) or {}).items():
                key = str(symbol)
                if key not in self.latest_symbol_data:
                    self.latest_symbol_data[key] = deque(maxlen=500)
                self.latest_symbol_data[key].clear()
                for row in rows:
                    if isinstance(row, (tuple, list)) and len(row) >= 6:
                        self.latest_symbol_data[key].append(
                            (
                                int(row[0]),
                                float(row[1]),
                                float(row[2]),
                                float(row[3]),
                                float(row[4]),
                                float(row[5]),
                            )
                        )
        self.events.put(event)

    def _emit_from_tick(self, tick: NormalizedTradeTick) -> None:
        for event in self.aggregator.ingest(tick):
            self._push_market_window(event)

    def _fetch_symbol_ticks(self, *, symbol: str) -> list[NormalizedTradeTick]:
        fetch_fn = getattr(self.exchange, "fetch_trades", None)
        if not callable(fetch_fn):
            inner = getattr(self.exchange, "exchange", None)
            if inner is None:
                return []
            fetch_fn = getattr(inner, "fetch_trades", None)
            if not callable(fetch_fn):
                return []

        since_ms = self._cursor_ms.get(symbol)
        try:
            rows = list(fetch_fn(symbol, since=since_ms, limit=500) or [])
        except TypeError:
            try:
                rows = list(fetch_fn(symbol, since_ms, 500) or [])
            except Exception:
                return []
        except Exception:
            return []

        ticks: list[NormalizedTradeTick] = []
        max_ts: int | None = since_ms
        for row in rows:
            tick = self._normalize_trade_row(symbol=symbol, row=dict(row or {}))
            if tick is None:
                continue
            ticks.append(tick)
            if max_ts is None or int(tick.exchange_ts_ms) > int(max_ts):
                max_ts = int(tick.exchange_ts_ms)
        if max_ts is not None:
            self._cursor_ms[symbol] = int(max_ts) + 1
        return ticks

    @staticmethod
    def _normalize_trade_row(*, symbol: str, row: dict[str, Any]) -> NormalizedTradeTick | None:
        ts_raw = row.get("timestamp")
        if ts_raw is None:
            return None
        try:
            exchange_ts_ms = int(ts_raw)
        except Exception:
            return None

        price_raw = row.get("price")
        qty_raw = row.get("amount")
        try:
            price = float(price_raw)
            qty = float(qty_raw or 0.0)
        except Exception:
            return None
        if price <= 0.0:
            return None

        resolved_symbol = normalize_stream_symbol(str(row.get("symbol") or symbol))
        raw_trade_id = row.get("id")
        payload = dict(row)
        info = payload.get("info")
        source = "trade"
        if isinstance(info, dict) and info.get("a") is not None:
            raw_trade_id = info.get("a")
            source = "aggTrade"

        event_id = build_trade_event_id(
            source=source,
            symbol=resolved_symbol,
            trade_id=raw_trade_id,
            payload=payload,
        )
        return NormalizedTradeTick(
            symbol=resolved_symbol,
            exchange_ts_ms=int(exchange_ts_ms),
            price=float(price),
            quantity=max(0.0, float(qty)),
            event_id=str(event_id),
            receive_ts_ms=int(time.time() * 1000),
        )

    def update_bars(self) -> None:
        return

    def get_latest_bar(self, symbol):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            return data[-1] if data else None

    def get_latest_bars(self, symbol, N=1):
        with self.lock:
            return list(self.latest_symbol_data.get(symbol, deque()))[-N:]

    def get_latest_bar_datetime(self, symbol):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            if not data:
                return None
            return data[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            if data:
                idx = self.col_idx.get(val_type)
                if idx is not None:
                    return float(data[-1][idx])
                timestamp_ms = int(data[-1][0])
            else:
                timestamp_ms = None
        feature_value = self._feature_lookup.get_latest(
            str(symbol),
            str(val_type),
            timestamp_ms=timestamp_ms,
        )
        if feature_value is not None:
            return float(feature_value)
        return 0.0

    def get_latest_bars_values(self, symbol, val_type, N=1):
        with self.lock:
            data = list(self.latest_symbol_data.get(symbol, deque()))[-N:]
            idx = self.col_idx.get(val_type)
            if idx is None:
                return []
            return [float(row[idx]) for row in data]

    def get_latest_feature_value(self, symbol, field):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            timestamp_ms = int(data[-1][0]) if data else None
        return self._feature_lookup.get_latest(
            str(symbol),
            str(field),
            timestamp_ms=timestamp_ms,
        )

    def get_market_spec(self, symbol):
        if self.exchange and hasattr(self.exchange, "get_market_spec"):
            return self.exchange.get_market_spec(symbol)
        return {}


__all__ = ["BinanceLiveDataHandler"]
