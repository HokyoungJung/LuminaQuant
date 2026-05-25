"""Native Binance USDⓈ-M Futures REST client helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from lumina_quant.symbols import canonical_symbol

_PROD_REST_BASE_URL = "https://fapi.binance.com"
_TESTNET_REST_BASE_URL = "https://demo-fapi.binance.com"
_PROD_WS_BASE_URL = "wss://fstream.binance.com"
_TESTNET_WS_BASE_URL = "wss://stream.binancefuture.com"


class BinanceFuturesAPIError(RuntimeError):
    """Structured REST error from Binance USDⓈ-M Futures APIs."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(str(message))
        self.status_code = status_code
        self.error_code = error_code
        self.payload = payload


@dataclass(slots=True)
class BinanceFuturesClientConfig:
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = False
    recv_window: int = 5_000
    timeout_sec: float = 30.0



def normalize_futures_symbol(symbol: str) -> str:
    """Normalize user symbol into Binance compact futures symbol token."""
    return canonical_symbol(str(symbol or "")).replace("/", "")


class BinanceFuturesRESTClient:
    """Lightweight HMAC client for official Binance USDⓈ-M Futures REST APIs."""

    def __init__(self, config: BinanceFuturesClientConfig | None = None) -> None:
        self.config = config or BinanceFuturesClientConfig()
        self.base_url = _TESTNET_REST_BASE_URL if self.config.testnet else _PROD_REST_BASE_URL
        self.ws_base_url = _TESTNET_WS_BASE_URL if self.config.testnet else _PROD_WS_BASE_URL
        self.rateLimit = 0

    @property
    def api_key(self) -> str:
        return str(self.config.api_key or "")

    @property
    def secret_key(self) -> str:
        return str(self.config.secret_key or "")

    def _sign(self, payload: str) -> str:
        secret = self.secret_key
        if not secret:
            raise BinanceFuturesAPIError("Binance Futures secret key is required for signed requests.")
        digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256)
        return digest.hexdigest()

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            text = f"{value:.16f}".rstrip("0").rstrip(".")
            return text or "0"
        return str(value)

    def _encode_params(self, params: dict[str, Any] | None) -> str:
        if not params:
            return ""
        items: list[tuple[str, str]] = []
        for key, value in params.items():
            if value is None:
                continue
            items.append((str(key), self._stringify_value(value)))
        return urllib.parse.urlencode(items)

    @staticmethod
    def _decode_json(raw: bytes) -> Any:
        if not raw:
            return {}
        text = raw.decode("utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        api_key_required: bool = False,
    ) -> Any:
        payload = dict(params or {})
        headers: dict[str, str] = {"User-Agent": "LuminaQuant/BinanceFutures"}
        if api_key_required or signed:
            if not self.api_key:
                raise BinanceFuturesAPIError("Binance Futures API key is required.")
            headers["X-MBX-APIKEY"] = self.api_key
        if signed:
            payload.setdefault("recvWindow", int(self.config.recv_window))
            payload["timestamp"] = int(time.time() * 1000)
            query = self._encode_params(payload)
            query = f"{query}&signature={self._sign(query)}" if query else f"signature={self._sign('')}"
        else:
            query = self._encode_params(payload)

        target = f"{self.base_url}{path}"
        data: bytes | None = None
        if method.upper() == "GET":
            if query:
                target = f"{target}?{query}"
        else:
            data = query.encode("utf-8") if query else b""
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        request = urllib.request.Request(url=target, method=method.upper(), headers=headers, data=data)
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.timeout_sec)) as response:
                return self._decode_json(response.read())
        except urllib.error.HTTPError as exc:
            payload_obj = self._decode_json(exc.read())
            message = None
            error_code = None
            if isinstance(payload_obj, dict):
                message = payload_obj.get("msg") or payload_obj.get("message")
                try:
                    error_code = int(payload_obj.get("code")) if payload_obj.get("code") is not None else None
                except Exception:
                    error_code = None
            raise BinanceFuturesAPIError(
                message or f"HTTP {exc.code} for {path}",
                status_code=int(exc.code),
                error_code=error_code,
                payload=payload_obj,
            ) from exc
        except urllib.error.URLError as exc:
            raise BinanceFuturesAPIError(str(exc.reason or exc)) from exc

    def public_get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def signed_get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params, signed=True)

    def signed_post(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, params=params, signed=True)

    def signed_put(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("PUT", path, params=params, signed=True)

    def signed_delete(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("DELETE", path, params=params, signed=True)

    def api_key_post(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, params=params, api_key_required=True)

    def api_key_put(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("PUT", path, params=params, api_key_required=True)

    def api_key_delete(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("DELETE", path, params=params, api_key_required=True)

    def exchange_info(self) -> dict[str, Any]:
        payload = self.public_get("/fapi/v1/exchangeInfo")
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        payload = self.public_get(
            "/fapi/v1/klines",
            params={
                "symbol": normalize_futures_symbol(symbol),
                "interval": str(interval),
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit,
            },
        )
        return list(payload or []) if isinstance(payload, list) else []

    def agg_trades(
        self,
        *,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        from_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self.public_get(
            "/fapi/v1/aggTrades",
            params={
                "symbol": normalize_futures_symbol(symbol),
                "startTime": start_time,
                "endTime": end_time,
                "fromId": from_id,
                "limit": limit,
            },
        )
        return [dict(item or {}) for item in list(payload or [])] if isinstance(payload, list) else []

    def create_user_data_stream(self) -> str:
        payload = self.api_key_post("/fapi/v1/listenKey")
        token = str((payload or {}).get("listenKey") or "").strip()
        if not token:
            raise BinanceFuturesAPIError("Binance Futures user stream did not return listenKey.")
        return token

    def keepalive_user_data_stream(self, listen_key: str) -> dict[str, Any]:
        payload = self.api_key_put("/fapi/v1/listenKey", params={"listenKey": str(listen_key)})
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def close_user_data_stream(self, listen_key: str) -> dict[str, Any]:
        payload = self.api_key_delete("/fapi/v1/listenKey", params={"listenKey": str(listen_key)})
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def account_balance_v3(self) -> list[dict[str, Any]]:
        payload = self.signed_get("/fapi/v3/balance")
        return [dict(item or {}) for item in list(payload or [])] if isinstance(payload, list) else []

    def account_info_v3(self) -> dict[str, Any]:
        payload = self.signed_get("/fapi/v3/account")
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def position_risk_v3(self, *, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self.signed_get(
            "/fapi/v3/positionRisk",
            params={"symbol": normalize_futures_symbol(symbol) if symbol else None},
        )
        return [dict(item or {}) for item in list(payload or [])] if isinstance(payload, list) else []

    def new_order(self, **params: Any) -> dict[str, Any]:
        payload = self.signed_post("/fapi/v1/order", params=params)
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def new_algo_order(self, **params: Any) -> dict[str, Any]:
        payload = self.signed_post("/fapi/v1/algoOrder", params=params)
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str | int | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.signed_delete(
            "/fapi/v1/order",
            params={
                "symbol": normalize_futures_symbol(symbol),
                "orderId": order_id,
                "origClientOrderId": orig_client_order_id,
            },
        )
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def query_order(
        self,
        *,
        symbol: str,
        order_id: str | int | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.signed_get(
            "/fapi/v1/order",
            params={
                "symbol": normalize_futures_symbol(symbol),
                "orderId": order_id,
                "origClientOrderId": orig_client_order_id,
            },
        )
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def query_open_orders(self, *, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self.signed_get(
            "/fapi/v1/openOrders",
            params={"symbol": normalize_futures_symbol(symbol) if symbol else None},
        )
        return [dict(item or {}) for item in list(payload or [])] if isinstance(payload, list) else []

    def change_position_mode(self, *, hedge_mode: bool) -> dict[str, Any]:
        payload = self.signed_post(
            "/fapi/v1/positionSide/dual",
            params={"dualSidePosition": hedge_mode},
        )
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def change_margin_type(self, *, symbol: str, margin_type: str) -> dict[str, Any]:
        payload = self.signed_post(
            "/fapi/v1/marginType",
            params={
                "symbol": normalize_futures_symbol(symbol),
                "marginType": str(margin_type).upper(),
            },
        )
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def change_initial_leverage(self, *, symbol: str, leverage: int) -> dict[str, Any]:
        payload = self.signed_post(
            "/fapi/v1/leverage",
            params={"symbol": normalize_futures_symbol(symbol), "leverage": int(leverage)},
        )
        return dict(payload or {}) if isinstance(payload, dict) else {}


__all__ = [
    "BinanceFuturesAPIError",
    "BinanceFuturesClientConfig",
    "BinanceFuturesRESTClient",
    "normalize_futures_symbol",
]
