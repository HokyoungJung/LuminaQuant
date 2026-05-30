"""Binance USDⓈ-M Futures live market stream client utilities."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

from lumina_quant.live.market_window_rolling import NormalizedTradeTick

_FUTURES_WS_BASE_URL = "wss://fstream.binance.com"


def normalize_stream_symbol(symbol: str) -> str:
    """Normalize Binance compact stream symbols (e.g. BTCUSDT) to BTC/USDT."""
    token = str(symbol or "").strip().upper()
    if "/" in token:
        return token
    for quote in ("USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"):
        if token.endswith(quote) and len(token) > len(quote):
            return f"{token[: -len(quote)]}/{quote}"
    return token


def build_trade_event_id(
    *, source: str, symbol: str, trade_id: str | int | None, payload: dict
) -> str:
    """Build canonical trade dedupe key."""
    source_token = str(source or "trade").strip().lower()
    normalized_symbol = normalize_stream_symbol(symbol)
    if trade_id is not None and str(trade_id).strip() != "":
        if source_token == "aggtrade":
            return f"mkt:agg:{normalized_symbol}:{trade_id}"
        return f"mkt:trade:{normalized_symbol}:{trade_id}"

    ts = int(payload.get("E") or payload.get("T") or payload.get("timestamp") or 0)
    price = float(payload.get("p") or payload.get("price") or 0.0)
    qty = float(payload.get("q") or payload.get("quantity") or payload.get("amount") or 0.0)
    buyer_maker = bool(payload.get("m") or payload.get("isBuyerMaker") or False)
    return f"mkt:fallback:{normalized_symbol}:{ts}:{price}:{qty}:{int(buyer_maker)}"


@dataclass(slots=True)
class NormalizedBookTicker:
    """Normalized Binance best-bid/best-ask snapshot for slippage guards."""

    symbol: str
    exchange_ts_ms: int
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    receive_ts_ms: int

    @property
    def mid_price(self) -> float:
        return (float(self.bid_price) + float(self.ask_price)) / 2.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid <= 0.0:
            return float("inf")
        return ((float(self.ask_price) - float(self.bid_price)) / mid) * 10_000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange_ts_ms": int(self.exchange_ts_ms),
            "bid_price": float(self.bid_price),
            "bid_quantity": float(self.bid_quantity),
            "ask_price": float(self.ask_price),
            "ask_quantity": float(self.ask_quantity),
            "receive_ts_ms": int(self.receive_ts_ms),
            "mid_price": float(self.mid_price),
            "spread_bps": float(self.spread_bps),
        }


@dataclass(slots=True)
class BinanceMarketStreamConfig:
    symbols: list[str]
    include_book_ticker: bool = False
    use_agg_trade: bool = True
    reconnect_delay_sec: float = 1.5
    websocket_base_url: str = _FUTURES_WS_BASE_URL
    max_connection_age_sec: float = 23 * 60 * 60


class BinanceMarketStreamClient:
    """Simple combined-stream websocket client for Binance USDⓈ-M Futures events."""

    def __init__(self, config: BinanceMarketStreamConfig) -> None:
        self.config = config

    def build_streams(self) -> list[str]:
        """Return combined-stream endpoint tokens."""
        streams: list[str] = []
        for symbol in list(self.config.symbols or []):
            compact = normalize_stream_symbol(symbol).replace("/", "").lower()
            streams.append(f"{compact}@aggTrade")
            if bool(self.config.include_book_ticker):
                streams.append(f"{compact}@bookTicker")
        return streams

    def stream_url(self) -> str:
        """Combined websocket URL for configured symbols/channels."""
        tokens = self.build_streams()
        if not tokens:
            raise ValueError("No symbols configured for Binance Futures market stream.")
        base_url = str(self.config.websocket_base_url or _FUTURES_WS_BASE_URL).strip().rstrip("/")
        return f"{base_url}/stream?streams={'/'.join(tokens)}"

    @staticmethod
    def _payload_data(payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        data = payload.get("data", payload)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def parse_message(
        payload: dict, *, receive_ts_ms: int | None = None
    ) -> list[NormalizedTradeTick]:
        """Parse one websocket frame into normalized trade ticks."""
        receive_ms = int(receive_ts_ms if receive_ts_ms is not None else int(time.time() * 1000))
        data = BinanceMarketStreamClient._payload_data(payload)
        if not data:
            return []

        event_type = str(data.get("e") or "").strip()
        if event_type != "aggTrade":
            return []

        symbol = normalize_stream_symbol(str(data.get("s") or ""))
        exchange_ts_ms = int(data.get("E") or data.get("T") or receive_ms)
        price = float(data.get("p") or 0.0)
        qty = float(data.get("q") or 0.0)
        if not symbol or price <= 0.0:
            return []

        trade_id = data.get("a")
        event_id = build_trade_event_id(
            source=event_type,
            symbol=symbol,
            trade_id=trade_id,
            payload=data,
        )
        return [
            NormalizedTradeTick(
                symbol=symbol,
                exchange_ts_ms=int(exchange_ts_ms),
                price=float(price),
                quantity=max(0.0, float(qty)),
                event_id=str(event_id),
                receive_ts_ms=int(receive_ms),
            )
        ]

    @staticmethod
    def parse_book_ticker_message(
        payload: dict, *, receive_ts_ms: int | None = None
    ) -> NormalizedBookTicker | None:
        """Parse one Binance ``bookTicker`` websocket frame into a BBO snapshot."""
        receive_ms = int(receive_ts_ms if receive_ts_ms is not None else int(time.time() * 1000))
        data = BinanceMarketStreamClient._payload_data(payload)
        if not data:
            return None

        event_type = str(data.get("e") or "").strip()
        stream_name = str(payload.get("stream") or "").lower() if isinstance(payload, dict) else ""
        if event_type not in {"bookTicker", ""} and "@bookticker" not in stream_name:
            return None
        if (
            event_type == ""
            and "@bookticker" not in stream_name
            and not {"b", "a", "s"}.issubset(data)
        ):
            return None

        symbol = normalize_stream_symbol(str(data.get("s") or ""))
        if not symbol:
            return None
        bid = float(data.get("b") or data.get("bidPrice") or 0.0)
        ask = float(data.get("a") or data.get("askPrice") or 0.0)
        bid_qty = float(data.get("B") or data.get("bidQty") or 0.0)
        ask_qty = float(data.get("A") or data.get("askQty") or 0.0)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return None
        exchange_ts_ms = int(data.get("E") or data.get("T") or receive_ms)
        return NormalizedBookTicker(
            symbol=symbol,
            exchange_ts_ms=int(exchange_ts_ms),
            bid_price=float(bid),
            bid_quantity=max(0.0, float(bid_qty)),
            ask_price=float(ask),
            ask_quantity=max(0.0, float(ask_qty)),
            receive_ts_ms=int(receive_ms),
        )

    def run_ws_loop(
        self,
        *,
        stop_event: Event,
        on_trade: Callable[[NormalizedTradeTick], None],
        on_book_ticker: Callable[[NormalizedBookTicker], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Run websocket loop until stopped. Best-effort reconnect on transient errors."""
        try:
            import websockets
        except Exception as exc:  # pragma: no cover - optional dependency path
            if on_error is not None:
                on_error(exc)
            return

        while not stop_event.is_set():
            try:
                url = self.stream_url()
                connected_at = time.monotonic()
                with websockets.sync.client.connect(url, open_timeout=10, close_timeout=5) as ws:
                    while not stop_event.is_set():
                        if time.monotonic() - connected_at >= max(
                            60.0, float(self.config.max_connection_age_sec)
                        ):
                            break
                        raw = ws.recv(timeout=1)
                        if raw is None:
                            continue
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            continue
                        now_ms = int(time.time() * 1000)
                        if on_book_ticker is not None:
                            book = self.parse_book_ticker_message(payload, receive_ts_ms=now_ms)
                            if book is not None:
                                on_book_ticker(book)
                                continue
                        for tick in self.parse_message(payload, receive_ts_ms=now_ms):
                            on_trade(tick)
            except TimeoutError:
                continue
            except Exception as exc:  # pragma: no cover - network/reconnect path
                if on_error is not None:
                    on_error(exc)
                if stop_event.is_set():
                    break
                time.sleep(max(0.25, float(self.config.reconnect_delay_sec)))


__all__ = [
    "BinanceMarketStreamClient",
    "BinanceMarketStreamConfig",
    "NormalizedBookTicker",
    "build_trade_event_id",
    "normalize_stream_symbol",
]
