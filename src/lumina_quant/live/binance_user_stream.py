"""Binance USDⓈ-M Futures user/account stream client."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

_PROD_WS_BASE_URL = "wss://fstream.binance.com"
_TESTNET_WS_BASE_URL = "wss://stream.binancefuture.com"


class BinanceUserStreamClient:
    """Native Binance USDⓈ-M Futures user stream client."""

    def __init__(
        self,
        *,
        exchange,
        market_type: str = "future",
        reconnect_delay_sec: float = 1.5,
        keepalive_interval_sec: float = 25 * 60,
    ) -> None:
        self.exchange = exchange
        self.market_type = str(market_type or "future").strip().lower()
        self.reconnect_delay_sec = max(0.25, float(reconnect_delay_sec))
        self.keepalive_interval_sec = max(60.0, float(keepalive_interval_sec))
        self.testnet = self._resolve_testnet(exchange)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _resolve_testnet(exchange) -> bool:
        """Best-effort testnet detection mirroring binance_futures_client config."""
        for value in (
            getattr(exchange, "testnet", None),
            getattr(getattr(exchange, "config", None), "IS_TESTNET", None),
            getattr(getattr(exchange, "rest_client", None), "config", None),
        ):
            if value is None:
                continue
            candidate = getattr(value, "testnet", value)
            if isinstance(candidate, bool):
                return candidate
        return False

    def start(
        self,
        *,
        on_event: Callable[[dict], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            kwargs={"on_event": on_event, "on_error": on_error},
            daemon=False,
        )
        self._thread.start()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, float(join_timeout)))

    def _run(
        self, *, on_event: Callable[[dict], None], on_error: Callable[[Exception], None] | None
    ) -> None:
        try:
            import websockets
        except Exception as exc:  # pragma: no cover - optional dependency path
            if on_error is not None:
                on_error(exc)
            return

        while not self._stop.is_set():
            listen_key: str | None = None
            try:
                listen_key = self._create_listen_key()
                if not listen_key:
                    raise RuntimeError("Failed to obtain Binance Futures listenKey.")
                ws_url = self._build_ws_url(listen_key)
                next_keepalive_monotonic = time.monotonic() + float(self.keepalive_interval_sec)
                connected_at = time.monotonic()
                with websockets.sync.client.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
                    while not self._stop.is_set():
                        if time.monotonic() - connected_at >= 23 * 60 * 60:
                            break
                        try:
                            raw = ws.recv(timeout=1)
                        except TimeoutError:
                            raw = None
                        if time.monotonic() >= next_keepalive_monotonic:
                            next_keepalive_monotonic = time.monotonic() + float(
                                self.keepalive_interval_sec
                            )
                            self._keepalive_listen_key(listen_key)
                        if raw is None:
                            continue
                        payload = json.loads(raw)
                        normalized = self.parse_message(payload)
                        if normalized is not None:
                            on_event(normalized)
            except TimeoutError:
                continue
            except Exception as exc:  # pragma: no cover - live reconnect path
                if on_error is not None:
                    on_error(exc)
                if self._stop.is_set():
                    break
                time.sleep(self.reconnect_delay_sec)
            finally:
                if listen_key:
                    try:
                        self._close_listen_key(listen_key)
                    except Exception:
                        pass

    def _ws_base_url(self) -> str:
        for value in (
            getattr(self.exchange, "websocket_base_url", None),
            getattr(getattr(self.exchange, "rest_client", None), "ws_base_url", None),
            getattr(getattr(self.exchange, "http", None), "config", None),
        ):
            if value is None:
                continue
            candidate = getattr(value, "websocket_base_url", value)
            token = str(candidate or "").strip()
            if token:
                return token.rstrip("/")
        return _TESTNET_WS_BASE_URL if self.testnet else _PROD_WS_BASE_URL

    def _build_ws_url(self, listen_key: str) -> str:
        return f"{self._ws_base_url()}/ws/{listen_key}"

    def _resolve_client(self):
        client = getattr(self.exchange, "_client", None)
        if callable(client):
            client = client()
        if client is None:
            client = getattr(self.exchange, "rest_client", None)
        if client is None:
            client = getattr(self.exchange, "http", None)
        if client is None:
            client = getattr(self.exchange, "exchange", None)
        return client

    def _create_listen_key(self) -> str | None:
        for target in (self.exchange, self._resolve_client()):
            if target is None:
                continue
            for name in ("create_listen_key", "start_user_stream", "start_user_data_stream"):
                fn = getattr(target, name, None)
                if not callable(fn):
                    continue
                try:
                    token = str(fn() or "").strip()
                except Exception:
                    continue
                if token:
                    return token
        return None

    def _keepalive_listen_key(self, listen_key: str) -> bool:
        for target in (self.exchange, self._resolve_client()):
            if target is None:
                continue
            for name in (
                "keepalive_listen_key",
                "keepalive_user_stream",
                "keepalive_user_data_stream",
            ):
                fn = getattr(target, name, None)
                if not callable(fn):
                    continue
                try:
                    result = fn(str(listen_key))
                except Exception:
                    continue
                return result is not False
        return False

    def _close_listen_key(self, listen_key: str) -> bool:
        for target in (self.exchange, self._resolve_client()):
            if target is None:
                continue
            for name in ("close_listen_key", "close_user_stream", "close_user_data_stream"):
                fn = getattr(target, name, None)
                if not callable(fn):
                    continue
                try:
                    result = fn(str(listen_key))
                except Exception:
                    continue
                return result is not False
        return False

    @staticmethod
    def parse_message(payload: dict) -> dict | None:
        if not isinstance(payload, dict):
            return None
        event_type = str(payload.get("e") or payload.get("event_type") or "")
        if event_type == "ORDER_TRADE_UPDATE":
            order = dict(payload.get("o") or {})
            return {
                "event_type": "executionReport",
                "exchange_ts_ms": int(payload.get("E") or order.get("T") or 0),
                "symbol": str(order.get("s") or ""),
                "order_id": str(order.get("i") or ""),
                "client_order_id": str(order.get("c") or ""),
                "exec_type": str(order.get("x") or ""),
                "order_status": str(order.get("X") or ""),
                "last_fill_qty": float(order.get("l") or 0.0),
                "cum_fill_qty": float(order.get("z") or 0.0),
                "last_fill_price": float(order.get("L") or 0.0),
                "trade_id": order.get("t"),
                "side": str(order.get("S") or ""),
                "position_side": str(order.get("ps") or ""),
                "reduce_only": bool(order.get("R") or False),
            }
        if event_type == "ACCOUNT_UPDATE":
            account = dict(payload.get("a") or {})
            return {
                "event_type": "accountUpdate",
                "exchange_ts_ms": int(payload.get("E") or 0),
                "balances": [dict(item or {}) for item in list(account.get("B") or [])],
                "positions": [dict(item or {}) for item in list(account.get("P") or [])],
                "reason": str(account.get("m") or ""),
            }
        return None


__all__ = ["BinanceUserStreamClient"]
