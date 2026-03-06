"""Binance user/account stream client for stream-driven order state projection."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable


class BinanceUserStreamClient:
    """Best-effort Binance user stream client.

    The client intentionally keeps request-plane logic separate from projection logic.
    """

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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
            try:
                listen_key = self._create_listen_key()
                if not listen_key:
                    raise RuntimeError("Failed to obtain Binance listenKey.")
                ws_url = self._build_ws_url(listen_key)
                next_keepalive_monotonic = time.monotonic() + float(self.keepalive_interval_sec)
                with websockets.sync.client.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=1)
                        except TimeoutError:
                            raw = None
                        if time.monotonic() >= next_keepalive_monotonic:
                            next_keepalive_monotonic = time.monotonic() + float(
                                self.keepalive_interval_sec
                            )
                            if not self._keepalive_listen_key(listen_key):
                                raise RuntimeError("Binance listenKey keepalive failed.")
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

    def _build_ws_url(self, listen_key: str) -> str:
        if self.market_type == "future":
            return f"wss://fstream.binance.com/ws/{listen_key}"
        return f"wss://stream.binance.com:9443/ws/{listen_key}"

    def _resolve_client(self):
        client = getattr(self.exchange, "_client", None)
        if callable(client):
            client = client()
        if client is None:
            client = getattr(self.exchange, "exchange", None)
        return client

    def _create_listen_key(self) -> str | None:
        # Best-effort across ccxt binance variants.
        client = self._resolve_client()
        if client is None:
            return None

        call_candidates = []
        if self.market_type == "future":
            call_candidates.extend(["fapiPrivatePostListenKey", "fapiprivate_post_listenkey"])
        call_candidates.extend(
            [
                "privatePostUserDataStream",
                "private_post_userdatastream",
            ]
        )

        for name in call_candidates:
            fn = getattr(client, name, None)
            if not callable(fn):
                continue
            try:
                payload = fn()
            except Exception:
                continue
            if isinstance(payload, dict):
                key = payload.get("listenKey")
                if isinstance(key, str) and key:
                    return key
        return None

    def _keepalive_listen_key(self, listen_key: str) -> bool:
        client = self._resolve_client()
        if client is None:
            return False
        call_candidates = []
        if self.market_type == "future":
            call_candidates.extend(["fapiPrivatePutListenKey", "fapiprivate_put_listenkey"])
        call_candidates.extend(
            [
                "privatePutUserDataStream",
                "private_put_userdatastream",
            ]
        )
        payload = {"listenKey": str(listen_key)}
        for name in call_candidates:
            fn = getattr(client, name, None)
            if not callable(fn):
                continue
            try:
                fn(payload)
                return True
            except TypeError:
                try:
                    fn(str(listen_key))
                    return True
                except Exception:
                    continue
            except Exception:
                continue
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
        if event_type == "executionReport":
            return {
                "event_type": "executionReport",
                "exchange_ts_ms": int(payload.get("E") or 0),
                "symbol": str(payload.get("s") or ""),
                "order_id": str(payload.get("i") or ""),
                "client_order_id": str(payload.get("c") or ""),
                "exec_type": str(payload.get("x") or ""),
                "order_status": str(payload.get("X") or ""),
                "last_fill_qty": float(payload.get("l") or 0.0),
                "cum_fill_qty": float(payload.get("z") or 0.0),
                "last_fill_price": float(payload.get("L") or 0.0),
                "trade_id": payload.get("t"),
                "side": str(payload.get("S") or ""),
            }
        if event_type == "outboundAccountPosition":
            return {
                "event_type": "outboundAccountPosition",
                "exchange_ts_ms": int(payload.get("E") or 0),
                "balances": list(payload.get("B") or []),
            }
        if event_type == "balanceUpdate":
            return {
                "event_type": "balanceUpdate",
                "exchange_ts_ms": int(payload.get("E") or 0),
                "asset": str(payload.get("a") or ""),
                "balance_delta": str(payload.get("d") or "0"),
                "clear_time_ms": int(payload.get("T") or 0),
            }
        return None


__all__ = ["BinanceUserStreamClient"]
