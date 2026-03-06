"""Request-plane gateway for submit/cancel/query operations."""

from __future__ import annotations


class OrderGateway:
    """Thin adapter over ExchangeInterface request APIs."""

    def __init__(self, exchange) -> None:
        self.exchange = exchange

    def submit(
        self,
        *,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        return dict(
            self.exchange.execute_order(
                symbol=symbol,
                type=type,
                side=side,
                quantity=quantity,
                price=price,
                params=params,
            )
            or {}
        )

    def cancel(self, order_id: str, symbol: str | None = None) -> bool:
        return bool(self.exchange.cancel_order(order_id, symbol))

    def query_order(self, order_id: str, symbol: str | None = None) -> dict:
        return dict(self.exchange.fetch_order(order_id, symbol) or {})

    def query_open_orders(self, symbol: str | None = None) -> list[dict]:
        return [dict(item or {}) for item in list(self.exchange.fetch_open_orders(symbol) or [])]


__all__ = ["OrderGateway"]
