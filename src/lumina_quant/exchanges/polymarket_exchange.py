"""Polymarket exchange adapter."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from lumina_quant.core.protocols import ExchangeInterface


def _load_sdk_symbols() -> dict[str, Any]:
    from py_clob_client.client import ClobClient  # type: ignore
    from py_clob_client.clob_types import ApiCreds, OpenOrderParams  # type: ignore
    from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore
    from py_clob_client.order_builder.constants import OrderType  # type: ignore

    try:
        from py_clob_client.clob_types import OrderArgs  # type: ignore
    except Exception:  # pragma: no cover - sdk version fallback
        from py_clob_client.order_builder.order_builder import OrderArgs  # type: ignore
    try:
        from py_clob_client.clob_types import MarketOrderArgs  # type: ignore
    except Exception:  # pragma: no cover - sdk version fallback
        from py_clob_client.order_builder.market_order import MarketOrderArgs  # type: ignore

    return {
        "ClobClient": ClobClient,
        "ApiCreds": ApiCreds,
        "OpenOrderParams": OpenOrderParams,
        "BUY": BUY,
        "SELL": SELL,
        "OrderArgs": OrderArgs,
        "MarketOrderArgs": MarketOrderArgs,
        "OrderType": OrderType,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class PolymarketExchange(ExchangeInterface):
    """Polymarket adapter with paper-first fallback and optional real execution."""

    def __init__(self, config):
        self.config = config
        self.connected = False
        self.client: Any = None
        self.sdk: dict[str, Any] | None = None
        self._creds: Any = None
        self.connect()

    def _host(self) -> str:
        return str(
            getattr(self.config, "POLYMARKET_HOST", "https://clob.polymarket.com")
            or "https://clob.polymarket.com"
        )

    def _data_host(self) -> str:
        return str(
            getattr(self.config, "POLYMARKET_DATA_HOST", "https://data-api.polymarket.com")
            or "https://data-api.polymarket.com"
        )

    def _private_key(self) -> str:
        env_name = str(
            getattr(self.config, "POLYMARKET_PRIVATE_KEY_ENV", "POLYMARKET_PRIVATE_KEY")
            or "POLYMARKET_PRIVATE_KEY"
        )
        return str(os.getenv(env_name, "") or "").strip()

    def _account_address(self) -> str:
        funder = str(getattr(self.config, "POLYMARKET_FUNDER", "") or "").strip()
        return funder

    def connect(self):
        self.connected = True
        try:
            self.sdk = _load_sdk_symbols()
        except Exception:
            self.sdk = None
            self.client = None
            return

        private_key = self._private_key()
        api_key = str(
            os.getenv(getattr(self.config, "POLYMARKET_API_KEY_ENV", "POLYMARKET_API_KEY"), "")
            or ""
        ).strip()
        api_secret = str(
            os.getenv(
                getattr(self.config, "POLYMARKET_API_SECRET_ENV", "POLYMARKET_API_SECRET"), ""
            )
            or ""
        ).strip()
        api_passphrase = str(
            os.getenv(
                getattr(self.config, "POLYMARKET_API_PASSPHRASE_ENV", "POLYMARKET_API_PASSPHRASE"),
                "",
            )
            or ""
        ).strip()
        chain_id = int(getattr(self.config, "POLYMARKET_CHAIN_ID", 137) or 137)
        signature_type = int(getattr(self.config, "POLYMARKET_SIGNATURE_TYPE", 0) or 0)
        funder = self._account_address() or None

        kwargs: dict[str, Any] = {"host": self._host(), "chain_id": chain_id}
        if private_key:
            kwargs["key"] = private_key
        if funder:
            kwargs["funder"] = funder
        if signature_type:
            kwargs["signature_type"] = signature_type

        client_cls = self.sdk["ClobClient"]
        self.client = client_cls(**kwargs)

        if api_key and api_secret and api_passphrase and hasattr(self.client, "set_api_creds"):
            self._creds = self.sdk["ApiCreds"](
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
            try:
                self.client.set_api_creds(self._creds)
            except Exception:
                pass
        elif (
            private_key
            and hasattr(self.client, "create_or_derive_api_creds")
            and hasattr(self.client, "set_api_creds")
        ):
            try:
                self._creds = self.client.create_or_derive_api_creds()
                self.client.set_api_creds(self._creds)
            except Exception:
                self._creds = None

    def _require_client(self) -> Any:
        if self.client is None or self.sdk is None:
            raise RuntimeError(
                "Polymarket support requires the live-polymarket extra and py-clob-client."
            )
        return self.client

    def _require_real_execution(self) -> Any:
        allow_real = bool(getattr(self.config, "POLYMARKET_ALLOW_REAL_EXECUTION", False))
        mode = str(getattr(self.config, "MODE", "paper") or "paper").strip().lower()
        if not allow_real or mode != "real":
            return None
        if not self._private_key():
            raise RuntimeError("Polymarket real execution requires a configured private key env.")
        return self._require_client()

    def _normalize_order(
        self, payload: dict[str, Any], *, symbol: str | None = None
    ) -> dict[str, Any]:
        item = dict(payload or {})
        order_id = item.get("orderID") or item.get("orderId") or item.get("id")
        price = item.get("price") or item.get("average_price") or item.get("avgPrice")
        side = str(item.get("side") or "").upper()
        size = item.get("original_size") or item.get("size") or item.get("amount")
        filled = item.get("filled") or item.get("matched_size") or item.get("executed_size") or 0.0
        status = item.get("status") or ("open" if item.get("success") else "rejected")
        return {
            "id": str(order_id or ""),
            "status": str(status).lower(),
            "filled": _safe_float(filled, 0.0),
            "amount": _safe_float(size, 0.0),
            "average": _safe_float(price, 0.0),
            "price": _safe_float(price, 0.0),
            "symbol": str(symbol or item.get("asset_id") or item.get("assetId") or ""),
            "side": side or None,
            "clientOrderId": item.get("clientOrderId") or item.get("client_order_id"),
        }

    def _data_api_get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = self._data_host().rstrip("/") + "/" + path.lstrip("/")
        if params:
            url += "?" + urllib.parse.urlencode(
                {key: value for key, value in params.items() if value is not None}
            )
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_balance(self, currency: str = "USDC") -> float:
        _ = currency
        return 0.0

    def get_all_positions(self) -> dict[str, float]:
        address = self._account_address()
        if not address:
            return {}
        try:
            payload = self._data_api_get("/positions", params={"user": address})
        except Exception:
            return {}
        positions: dict[str, float] = {}
        for row in list(payload or []):
            if not isinstance(row, dict):
                continue
            asset_id = str(
                row.get("asset") or row.get("asset_id") or row.get("assetId") or ""
            ).strip()
            if not asset_id:
                continue
            positions[asset_id] = positions.get(asset_id, 0.0) + _safe_float(
                row.get("size") or row.get("amount") or row.get("position"),
                0.0,
            )
        return positions

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[tuple]:
        _ = (symbol, timeframe, limit)
        return []

    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        client = self._require_real_execution()
        if client is None:
            return {
                "id": f"poly-paper-{symbol}-{side}-{quantity}",
                "status": "paper",
                "filled": 0.0,
                "amount": float(quantity),
                "average": _safe_float(price, 0.0),
                "price": _safe_float(price, 0.0),
                "symbol": symbol,
                "side": str(side).upper(),
                "params": dict(params or {}),
            }

        sdk = self.sdk or {}
        side_token = sdk["BUY"] if str(side).lower() == "buy" else sdk["SELL"]
        order_type = str(type or "").strip().lower()

        if order_type == "market" or price is None:
            args = sdk["MarketOrderArgs"](
                token_id=str(symbol),
                amount=float(quantity),
                side=side_token,
            )
            if hasattr(client, "create_market_order"):
                signed_order = client.create_market_order(args)
            else:
                signed_order = client.sign_market_order(args)
            response = client.post_order(signed_order, sdk["OrderType"].FOK)
        else:
            args = sdk["OrderArgs"](
                token_id=str(symbol),
                price=float(price),
                size=float(quantity),
                side=side_token,
            )
            if hasattr(client, "create_order"):
                signed_order = client.create_order(args)
            else:
                signed_order = client.sign_order(args)
            response = client.post_order(signed_order, sdk["OrderType"].GTC)
        return self._normalize_order(dict(response or {}), symbol=str(symbol))

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        client = self._require_client()
        sdk = self.sdk or {}
        params = sdk["OpenOrderParams"](**({"asset_id": symbol} if symbol else {}))
        if hasattr(client, "get_orders"):
            rows = list(client.get_orders(params) or [])
        else:
            rows = list(client.get_open_orders(params) or [])
        return [self._normalize_order(dict(row or {}), symbol=symbol) for row in rows]

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        _ = symbol
        client = self._require_client()
        if hasattr(client, "cancel"):
            response = client.cancel(str(order_id))
        elif hasattr(client, "cancel_order"):
            response = client.cancel_order(str(order_id))
        else:
            raise RuntimeError("Polymarket client does not expose cancel support.")
        if isinstance(response, dict):
            return bool(response.get("success", True))
        return True

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict:
        client = self._require_client()
        if hasattr(client, "get_order"):
            response = dict(client.get_order(str(order_id)) or {})
        else:
            response = dict(client.fetch_order(str(order_id)) or {})
        return self._normalize_order(response, symbol=symbol)


__all__ = ["PolymarketExchange"]
