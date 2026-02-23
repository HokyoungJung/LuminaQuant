"""Central in-memory runtime cache for market and execution state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeCache:
    latest_market: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    positions: dict[str, float] = field(default_factory=dict)
    position_legs: dict[str, dict[str, float]] = field(default_factory=dict)
    account: dict[str, Any] = field(default_factory=dict)

    def update_market(self, symbol: str, payload: dict[str, Any]) -> None:
        self.latest_market[str(symbol)] = dict(payload)

    def update_order_state(self, order_id: str | None, payload: dict[str, Any]) -> None:
        if order_id is None:
            return
        state = str(payload.get("state", "")).upper()
        if state in {"FILLED", "CANCELED", "REJECTED", "TIMEOUT"}:
            self.open_orders.pop(str(order_id), None)
            return
        self.open_orders[str(order_id)] = dict(payload)

    def update_positions(self, positions: dict[str, Any]) -> None:
        self.positions = {str(symbol): float(qty) for symbol, qty in positions.items()}

    def update_position_legs(self, legs: dict[str, dict[str, Any]]) -> None:
        normalized: dict[str, dict[str, float]] = {}
        for symbol, payload in dict(legs or {}).items():
            if not isinstance(payload, dict):
                continue
            long_qty = max(0.0, float(payload.get("LONG", 0.0) or 0.0))
            short_qty = max(0.0, float(payload.get("SHORT", 0.0) or 0.0))
            if long_qty <= 0.0 and short_qty <= 0.0:
                continue
            normalized[str(symbol)] = {"LONG": long_qty, "SHORT": short_qty}
        self.position_legs = normalized

    def update_account(self, account: dict[str, Any]) -> None:
        self.account = dict(account)

    def snapshot(self) -> dict[str, Any]:
        return {
            "latest_market": dict(self.latest_market),
            "open_orders": dict(self.open_orders),
            "positions": dict(self.positions),
            "position_legs": dict(self.position_legs),
            "account": dict(self.account),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.latest_market = dict(payload.get("latest_market") or {})
        self.open_orders = dict(payload.get("open_orders") or {})
        self.positions = {
            str(symbol): float(qty) for symbol, qty in dict(payload.get("positions") or {}).items()
        }
        self.update_position_legs(dict(payload.get("position_legs") or {}))
        self.account = dict(payload.get("account") or {})
