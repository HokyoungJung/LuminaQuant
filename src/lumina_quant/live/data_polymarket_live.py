"""Polymarket live market-data handler using public market websocket ticks."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from typing import Any

from lumina_quant.live.market_window_rolling import NormalizedTradeTick, RollingWindowAggregator


def _now_ms() -> int:
    return int(time.time() * 1000)


class PolymarketLiveDataHandler:
    """Aggregate Polymarket market-channel ticks into MARKET_WINDOW events."""

    def __init__(self, events, symbol_list, config, exchange=None, *, transport: str = "ws"):
        self.events = events
        asset_ids = list(getattr(config, "POLYMARKET_ASSET_IDS", []) or [])
        self.symbol_list = [str(item) for item in (asset_ids or list(symbol_list or []))]
        self.config = config
        self.exchange = exchange
        self.transport = str(transport or "ws").strip().lower()
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
                    self.config, "INGEST_WINDOW_SECONDS", getattr(self.config, "WINDOW_SECONDS", 20)
                )
                or 20
            ),
        )
        self._fatal_channel: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._fatal_error: BaseException | None = None
        self.aggregator = RollingWindowAggregator(
            symbol_list=list(self.symbol_list),
            window_seconds=int(self._window_seconds),
            max_lateness_ms=1500,
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
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(join_timeout)))

    def shutdown(self, join_timeout: float = 5.0) -> None:
        self.stop(join_timeout=join_timeout)

    def _run_loop(self) -> None:
        try:
            self._run_ws_loop()
        except Exception as exc:  # pragma: no cover - defensive
            self._publish_fatal(exc)

    def _push_market_window(self, event) -> None:
        with self.lock:
            for symbol, rows in dict(getattr(event, "bars_1s", {}) or {}).items():
                key = str(symbol)
                if key not in self.latest_symbol_data:
                    self.latest_symbol_data[key] = deque(maxlen=500)
                self.latest_symbol_data[key].clear()
                self.latest_symbol_data[key].extend(rows)
        self.events.put(event)

    def _emit_tick(self, tick: NormalizedTradeTick) -> None:
        for event in self.aggregator.ingest(tick):
            self._push_market_window(event)

    def _subscribe_messages(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "market",
                "assets_ids": list(self.symbol_list),
            },
        ]

    def _normalize_message(self, payload: dict[str, Any]) -> NormalizedTradeTick | None:
        event_type = str(payload.get("event_type") or "").strip().lower()
        if event_type != "last_trade_price":
            return None
        symbol = str(payload.get("asset_id") or "").strip()
        if not symbol:
            return None
        price = payload.get("price")
        size = payload.get("size", 0.0)
        timestamp = payload.get("timestamp")
        try:
            exchange_ts_ms = int(float(timestamp))
        except Exception:
            exchange_ts_ms = _now_ms()
        try:
            price_value = float(price)
            size_value = float(size or 0.0)
        except Exception:
            return None
        if price_value <= 0.0:
            return None
        event_id = str(
            payload.get("event_id")
            or payload.get("id")
            or f"{symbol}:{exchange_ts_ms}:{price_value}:{size_value}"
        )
        return NormalizedTradeTick(
            symbol=symbol,
            exchange_ts_ms=exchange_ts_ms,
            price=price_value,
            quantity=max(0.0, size_value),
            event_id=event_id,
            receive_ts_ms=_now_ms(),
        )

    def _run_ws_loop(self) -> None:
        try:
            import asyncio
            import websockets
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Polymarket live market data requires the websocket stack. Install the live-polymarket extra."
            ) from exc

        ws_url = str(getattr(self.config, "POLYMARKET_MARKET_WS_URL", "") or "").strip()
        if not ws_url:
            raise RuntimeError("POLYMARKET_MARKET_WS_URL is required for polymarket_live.")

        async def _consume() -> None:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as websocket:
                for message in self._subscribe_messages():
                    await websocket.send(json.dumps(message))
                while self.continue_backtest and not self._shutdown.is_set():
                    raw = await websocket.recv()
                    payload = json.loads(raw)
                    if isinstance(payload, list):
                        items = payload
                    else:
                        items = [payload]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        tick = self._normalize_message(item)
                        if tick is not None:
                            self._emit_tick(tick)

        asyncio.run(_consume())


__all__ = ["PolymarketLiveDataHandler"]
