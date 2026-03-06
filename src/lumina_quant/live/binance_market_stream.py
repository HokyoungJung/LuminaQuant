"""Binance live market stream client utilities (trade/aggTrade/bookTicker)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from threading import Event
from collections.abc import Callable

from lumina_quant.live.market_window_rolling import NormalizedTradeTick


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
class BinanceMarketStreamConfig:
    symbols: list[str]
    include_book_ticker: bool = False
    use_agg_trade: bool = True
    reconnect_delay_sec: float = 1.5


class BinanceMarketStreamClient:
    """Simple combined-stream websocket client for Binance market events."""

    def __init__(self, config: BinanceMarketStreamConfig) -> None:
        self.config = config

    def build_streams(self) -> list[str]:
        """Return combined-stream endpoint tokens."""
        streams: list[str] = []
        trade_channel = "aggTrade" if bool(self.config.use_agg_trade) else "trade"
        for symbol in list(self.config.symbols or []):
            compact = normalize_stream_symbol(symbol).replace("/", "").lower()
            streams.append(f"{compact}@{trade_channel}")
            if bool(self.config.include_book_ticker):
                streams.append(f"{compact}@bookTicker")
        return streams

    def stream_url(self) -> str:
        """Combined websocket URL for configured symbols/channels."""
        tokens = self.build_streams()
        if not tokens:
            raise ValueError("No symbols configured for Binance market stream.")
        return f"wss://stream.binance.com:9443/stream?streams={'/'.join(tokens)}"

    @staticmethod
    def parse_message(
        payload: dict, *, receive_ts_ms: int | None = None
    ) -> list[NormalizedTradeTick]:
        """Parse one websocket frame into normalized trade ticks."""
        if not isinstance(payload, dict):
            return []
        receive_ms = int(receive_ts_ms if receive_ts_ms is not None else int(time.time() * 1000))

        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return []

        event_type = str(data.get("e") or "").strip()
        if event_type not in {"aggTrade", "trade"}:
            return []

        symbol = normalize_stream_symbol(str(data.get("s") or ""))
        exchange_ts_ms = int(data.get("E") or data.get("T") or receive_ms)
        price = float(data.get("p") or 0.0)
        qty = float(data.get("q") or 0.0)
        if not symbol or price <= 0.0:
            return []

        trade_id = data.get("a") if event_type == "aggTrade" else data.get("t")
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

    def run_ws_loop(
        self,
        *,
        stop_event: Event,
        on_trade: Callable[[NormalizedTradeTick], None],
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
                with websockets.sync.client.connect(url, open_timeout=10, close_timeout=5) as ws:
                    while not stop_event.is_set():
                        raw = ws.recv(timeout=1)
                        if raw is None:
                            continue
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            continue
                        now_ms = int(time.time() * 1000)
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
    "build_trade_event_id",
    "normalize_stream_symbol",
]
