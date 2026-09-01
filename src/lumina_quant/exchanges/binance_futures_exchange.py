"""Native Binance USDⓈ-M Futures ExchangeInterface adapter."""

from __future__ import annotations

import time
from typing import Any

from lumina_quant.core.protocols import ExchangeInterface
from lumina_quant.exchanges.binance_futures_client import (
    BinanceFuturesAPIError,
    BinanceFuturesClientConfig,
    BinanceFuturesRESTClient,
    normalize_futures_symbol,
)
from lumina_quant.storage.parquet import normalize_symbol


_BINANCE_ALGO_ORDER_PARAM_KEYS = (
    "clientAlgoId",
    "positionSide",
    "timeInForce",
    "quantity",
    "price",
    "triggerPrice",
    "workingType",
    "priceProtect",
    "priceMatch",
    "closePosition",
    "reduceOnly",
    "activatePrice",
    "callbackRate",
    "newOrderRespType",
    "selfTradePreventionMode",
    "goodTillDate",
    "recvWindow",
)


class BinanceFuturesExchange(ExchangeInterface):
    """ExchangeInterface backed by official Binance USDⓈ-M Futures APIs only."""

    id = "binance"

    def __init__(self, config: object) -> None:
        self.config = config
        self.market_type = "future"
        self.options = {"defaultType": "future"}
        self.rest_client: BinanceFuturesRESTClient | None = None
        self.exchange: BinanceFuturesRESTClient | None = None
        self._markets: dict[str, dict[str, Any]] = {}
        self.connect()

    def _client(self) -> BinanceFuturesRESTClient:
        client = self.rest_client
        if client is None:
            raise RuntimeError("Binance Futures REST client is not initialized.")
        return client

    @property
    def websocket_base_url(self) -> str:
        return self._client().ws_base_url

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        token = str(value).strip().lower()
        if token in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        return bool(value)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return normalize_symbol(symbol)

    def _annotate_signed_bootstrap_error(
        self,
        exc: BinanceFuturesAPIError,
    ) -> BinanceFuturesAPIError:
        message = str(exc)
        invalid_credential = (
            getattr(exc, "error_code", None) in {-2015, -2014}
            or "Invalid API-key" in message
            or "API-key format invalid" in message
        )
        if not invalid_credential:
            return exc

        mode_hint = "paper/testnet" if bool(getattr(self.config, "IS_TESTNET", False)) else "real"
        return BinanceFuturesAPIError(
            (
                f"{message} "
                f"(Binance Futures {mode_hint} credentials were rejected during signed exchange bootstrap; "
                "check BINANCE_API_KEY/BINANCE_SECRET_KEY, futures permissions, testnet-vs-prod key pairing, "
                "and any IP allowlist restrictions.)"
            ),
            status_code=getattr(exc, "status_code", None),
            error_code=getattr(exc, "error_code", None),
            payload=getattr(exc, "payload", None),
        )

    def connect(self) -> None:
        exchange_config = getattr(self.config, "EXCHANGE", None)
        if not isinstance(exchange_config, dict):
            raise ValueError("EXCHANGE config must be a dictionary with driver/name fields.")
        driver = str(exchange_config.get("driver", "") or "").strip().lower()
        if driver not in {"binance_futures", "binance_native", "binance"}:
            raise ValueError(f"Unsupported Binance futures driver: {driver}")
        market_type = str(exchange_config.get("market_type", "future") or "future").strip().lower()
        if market_type != "future":
            raise ValueError(
                "BinanceFuturesExchange only supports USDⓈ-M Futures market_type='future'."
            )
        name = str(exchange_config.get("name", "binance") or "binance").strip().lower()
        if name != "binance":
            raise ValueError("BinanceFuturesExchange requires EXCHANGE.name='binance'.")

        client = BinanceFuturesRESTClient(
            BinanceFuturesClientConfig(
                api_key=str(getattr(self.config, "BINANCE_API_KEY", "") or ""),
                secret_key=str(getattr(self.config, "BINANCE_SECRET_KEY", "") or ""),
                testnet=bool(getattr(self.config, "IS_TESTNET", False)),
            )
        )
        self.rest_client = client
        self.exchange = client
        self.load_markets()
        try:
            position_mode = str(getattr(self.config, "POSITION_MODE", "HEDGE") or "HEDGE").upper()
            self.set_position_mode(position_mode)

            margin_mode = str(getattr(self.config, "MARGIN_MODE", "isolated") or "isolated")
            leverage = int(getattr(self.config, "LEVERAGE", 1) or 1)
            for symbol in list(getattr(self.config, "SYMBOLS", []) or []):
                self.set_margin_mode(str(symbol), margin_mode)
                self.set_leverage(str(symbol), leverage)
        except BinanceFuturesAPIError as exc:
            raise self._annotate_signed_bootstrap_error(exc) from exc

    def close(self) -> None:
        self.rest_client = None
        self.exchange = None

    def _symbol_filters(self, symbol_info: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in list(symbol_info.get("filters") or []):
            if not isinstance(item, dict):
                continue
            filter_type = str(item.get("filterType") or "").strip().upper()
            if filter_type:
                out[filter_type] = dict(item)
        return out

    def load_markets(self) -> dict:
        payload = self._client().exchange_info()
        markets: dict[str, dict[str, Any]] = {}
        for item in list(payload.get("symbols") or []):
            if not isinstance(item, dict):
                continue
            compact = str(item.get("symbol") or "").strip().upper()
            if not compact:
                continue
            symbol = self._normalize_symbol(compact)
            filters = self._symbol_filters(item)
            lot = filters.get("LOT_SIZE") or {}
            market_lot = filters.get("MARKET_LOT_SIZE") or lot
            notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
            markets[symbol] = {
                "symbol": symbol,
                "id": compact,
                "status": str(item.get("status") or ""),
                "contractType": str(item.get("contractType") or ""),
                "precision": {
                    "amount": self._as_float(lot.get("stepSize"), 0.0),
                    "price": self._as_float(
                        (filters.get("PRICE_FILTER") or {}).get("tickSize"), 0.0
                    ),
                },
                "limits": {
                    "amount": {
                        "min": self._as_float(market_lot.get("minQty"), 0.0),
                        "max": self._as_float(market_lot.get("maxQty"), 0.0),
                    },
                    "cost": {
                        "min": self._as_float(
                            notional.get("notional"),
                            self._as_float(notional.get("minNotional"), 0.0),
                        ),
                    },
                },
                "info": dict(item),
            }
        self._markets = markets
        return dict(self._markets)

    def get_market_spec(self, symbol: str) -> dict[str, Any]:
        if not self._markets:
            self.load_markets()
        market = dict(self._markets.get(self._normalize_symbol(symbol), {}) or {})
        limits = dict(market.get("limits") or {})
        precision = dict(market.get("precision") or {})
        return {
            "min_qty": self._as_float((limits.get("amount") or {}).get("min"), 0.0),
            "qty_step": self._as_float(precision.get("amount"), 0.0),
            "min_notional": self._as_float((limits.get("cost") or {}).get("min"), 0.0),
            "price_tick_size": self._as_float(precision.get("price"), 0.0),
        }

    def get_balance(self, currency: str = "USDT") -> float:
        asset = str(currency or "USDT").upper()
        for row in self._client().account_balance_v3():
            if str(row.get("asset") or "").upper() != asset:
                continue
            return self._as_float(
                row.get("availableBalance"), self._as_float(row.get("balance"), 0.0)
            )
        return 0.0

    def get_margin_balance(self, currency: str = "USDT") -> float:
        """Return the true perp account equity (wallet balance + unrealized PnL).

        C4 live equity reconciliation (trader.py `_fetch_exchange_equity`) needs the
        authoritative marginBalance figure, distinct from ``get_balance``'s
        availableBalance (which excludes locked position margin). Reads
        GET /fapi/v3/account via ``account_info_v3`` and prefers the per-asset
        ``marginBalance`` field (walletBalance + unrealizedProfit) for ``currency``;
        falls back to the account-wide ``totalMarginBalance`` when no matching
        per-asset row is present (e.g. multi-assets margin mode).

        NOTE: like ``get_balance``, a missing/malformed field silently coerces to
        0.0 via ``_as_float`` (it cannot distinguish "absent" from a genuine zero
        balance) — callers must treat a 0.0 result as unavailable.
        """
        asset = str(currency or "USDT").upper()
        payload = self._client().account_info_v3()
        for row in list(payload.get("assets") or []):
            if not isinstance(row, dict) or str(row.get("asset") or "").upper() != asset:
                continue
            wallet_balance = self._as_float(row.get("walletBalance"), 0.0)
            unrealized_pnl = self._as_float(row.get("unrealizedProfit"), 0.0)
            return self._as_float(row.get("marginBalance"), wallet_balance + unrealized_pnl)
        return self._as_float(payload.get("totalMarginBalance"), 0.0)

    def _position_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        symbol = self._normalize_symbol(str(row.get("symbol") or ""))
        position_side = str(row.get("positionSide") or "BOTH").upper()
        qty = self._as_float(row.get("positionAmt"), 0.0)
        return {
            "symbol": symbol,
            "contracts": qty,
            "positionAmt": qty,
            "side": position_side,
            "positionSide": position_side,
            "entryPrice": self._as_float(row.get("entryPrice"), 0.0),
            "unrealizedPnl": self._as_float(row.get("unRealizedProfit"), 0.0),
            "leverage": self._as_float(row.get("leverage"), 0.0),
            "marginType": str(row.get("marginType") or ""),
            "info": dict(row),
        }

    def fetch_positions(self, symbol: str | None = None) -> list[dict]:
        rows = self._client().position_risk_v3(symbol=symbol)
        return [
            self._position_payload(dict(row or {})) for row in rows if dict(row or {}).get("symbol")
        ]

    def get_all_positions(self) -> dict[str, float]:
        positions: dict[str, float] = {}
        for row in self.fetch_positions():
            symbol = str(row.get("symbol") or "")
            qty = self._as_float(row.get("positionAmt"), 0.0)
            side = str(row.get("positionSide") or row.get("side") or "BOTH").upper()
            if not symbol or abs(qty) <= 0.0:
                continue
            if side == "SHORT" and qty > 0.0:
                qty = -qty
            positions[symbol] = positions.get(symbol, 0.0) + qty
        return positions

    def get_all_position_legs(self) -> dict[str, dict[str, float]]:
        legs: dict[str, dict[str, float]] = {}
        for row in self.fetch_positions():
            symbol = str(row.get("symbol") or "")
            qty = abs(self._as_float(row.get("positionAmt"), 0.0))
            side = str(row.get("positionSide") or row.get("side") or "BOTH").upper()
            if not symbol or qty <= 0.0:
                continue
            bucket = legs.setdefault(symbol, {"LONG": 0.0, "SHORT": 0.0})
            if side == "SHORT":
                bucket["SHORT"] += qty
            else:
                bucket["LONG"] += qty
        return {
            symbol: payload
            for symbol, payload in legs.items()
            if payload["LONG"] > 0.0 or payload["SHORT"] > 0.0
        }

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[tuple]:
        rows = self._client().klines(
            symbol=symbol, interval=str(timeframe), limit=max(1, int(limit))
        )
        out: list[tuple] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            out.append(
                (
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
            )
        return out

    def fetch_trades(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict]:
        rows = self._client().agg_trades(
            symbol=symbol,
            start_time=int(since) if since is not None else None,
            end_time=(
                min(int(since) + 3_599_999, int(time.time() * 1000)) if since is not None else None
            ),
            limit=max(1, min(int(limit or 1000), 1000)),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            price = self._as_float(row.get("p"), 0.0)
            quantity = self._as_float(row.get("q"), 0.0)
            timestamp_ms = int(row.get("T") or 0)
            if price <= 0.0 or quantity < 0.0 or timestamp_ms <= 0:
                continue
            is_buyer_maker = bool(row.get("m"))
            out.append(
                {
                    "id": int(row.get("a") or 0),
                    "symbol": self._normalize_symbol(symbol),
                    "timestamp": timestamp_ms,
                    "price": price,
                    "amount": quantity,
                    "side": "sell" if is_buyer_maker else "buy",
                    "isBuyerMaker": is_buyer_maker,
                    "info": dict(row),
                }
            )
        return out

    @staticmethod
    def _normalize_order_status(value: Any) -> str:
        token = str(value or "").strip().upper()
        mapping = {
            "NEW": "open",
            "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "closed",
            "CANCELED": "canceled",
            "EXPIRED": "canceled",
            "REJECTED": "rejected",
        }
        return mapping.get(token, token.lower() or "unknown")

    @staticmethod
    def _normalize_algo_order_status(value: Any) -> str:
        token = str(value or "").strip().upper()
        mapping = {
            "NEW": "open",
            "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "closed",
            "CANCELED": "canceled",
            "EXPIRED": "canceled",
            "REJECTED": "rejected",
        }
        return mapping.get(token, token.lower() or "unknown")

    def _normalize_order(self, row: dict[str, Any]) -> dict[str, Any]:
        amount = self._as_float(row.get("origQty"), 0.0)
        filled = self._as_float(row.get("executedQty"), 0.0)
        price = self._as_float(row.get("price"), 0.0)
        average = self._as_float(row.get("avgPrice"), price)
        if average <= 0.0:
            average = price
        return {
            "id": str(row.get("orderId") or ""),
            "status": self._normalize_order_status(row.get("status")),
            "filled": filled,
            "average": average,
            "price": price,
            "amount": amount,
            "remaining": max(0.0, amount - filled),
            "timestamp": int(row.get("time") or row.get("updateTime") or 0),
            "fee": row.get("fee"),
            "info": dict(row),
            "symbol": self._normalize_symbol(str(row.get("symbol") or "")),
            "side": str(row.get("side") or "").lower(),
            "type": str(row.get("type") or row.get("origType") or "").lower(),
            "clientOrderId": row.get("clientOrderId"),
            "client_order_id": row.get("clientOrderId"),
            "positionSide": row.get("positionSide"),
            "reduceOnly": self._as_bool(row.get("reduceOnly")),
            "timeInForce": row.get("timeInForce"),
        }

    def _normalize_algo_order(self, row: dict[str, Any]) -> dict[str, Any]:
        amount = self._as_float(row.get("quantity"), 0.0)
        return {
            "id": str(row.get("algoId") or row.get("orderId") or ""),
            "status": self._normalize_algo_order_status(row.get("algoStatus") or row.get("status")),
            "filled": 0.0,
            "average": self._as_float(row.get("price"), 0.0),
            "price": self._as_float(row.get("price"), 0.0),
            "amount": amount,
            "remaining": amount,
            "timestamp": int(row.get("createTime") or row.get("updateTime") or 0),
            "fee": row.get("fee"),
            "info": dict(row),
            "symbol": self._normalize_symbol(str(row.get("symbol") or "")),
            "side": str(row.get("side") or "").lower(),
            "type": str(row.get("orderType") or row.get("type") or "").lower(),
            "clientOrderId": row.get("clientAlgoId") or row.get("clientOrderId"),
            "client_order_id": row.get("clientAlgoId") or row.get("clientOrderId"),
            "positionSide": row.get("positionSide"),
            "reduceOnly": self._as_bool(row.get("reduceOnly")),
            "timeInForce": row.get("timeInForce"),
        }

    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        request_params = dict(params or {})
        payload: dict[str, Any] = {
            "symbol": normalize_futures_symbol(symbol),
            "side": str(side).strip().upper(),
            "type": str(type).strip().upper(),
            "quantity": quantity,
        }
        if payload["type"] == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require price.")
            payload["price"] = price
            payload["timeInForce"] = str(
                request_params.pop("timeInForce", request_params.pop("time_in_force", "GTC"))
            )
        elif price is not None and request_params.get("price") is None:
            payload["price"] = price
        if payload["type"] == "MARKET":
            payload.setdefault("newOrderRespType", "RESULT")
        for key in (
            "newClientOrderId",
            "positionSide",
            "reduceOnly",
            "timeInForce",
            "stopPrice",
            "activationPrice",
            "callbackRate",
            "workingType",
            "closePosition",
            "priceProtect",
            "priceMatch",
            "selfTradePreventionMode",
            "goodTillDate",
        ):
            if key in request_params and request_params[key] is not None:
                payload[key] = request_params.pop(key)
        if "clientOrderId" in request_params and payload.get("newClientOrderId") is None:
            payload["newClientOrderId"] = request_params.pop("clientOrderId")
        if "client_order_id" in request_params and payload.get("newClientOrderId") is None:
            payload["newClientOrderId"] = request_params.pop("client_order_id")
        if str(
            getattr(self.config, "POSITION_MODE", "HEDGE") or "HEDGE"
        ).upper() == "HEDGE" and str(payload.get("positionSide") or "").upper() in {
            "LONG",
            "SHORT",
        }:
            payload.pop("reduceOnly", None)
        payload.update({key: value for key, value in request_params.items() if value is not None})
        return self._normalize_order(self._client().new_order(**payload))

    def execute_algo_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        params: dict | None = None,
    ) -> dict:
        request_params = dict(params or {})
        payload: dict[str, Any] = {
            "algoType": request_params.pop("algoType", "CONDITIONAL"),
            "symbol": normalize_futures_symbol(symbol),
            "side": str(side).strip().upper(),
            "type": str(type).strip().upper(),
            "quantity": quantity,
        }
        if "clientOrderId" in request_params and request_params.get("clientAlgoId") is None:
            request_params["clientAlgoId"] = request_params.pop("clientOrderId")
        if "client_order_id" in request_params and request_params.get("clientAlgoId") is None:
            request_params["clientAlgoId"] = request_params.pop("client_order_id")
        for key in _BINANCE_ALGO_ORDER_PARAM_KEYS:
            if key == "quantity":
                continue
            if key in request_params and request_params[key] is not None:
                payload[key] = request_params.pop(key)
        if str(
            getattr(self.config, "POSITION_MODE", "HEDGE") or "HEDGE"
        ).upper() == "HEDGE" and str(payload.get("positionSide") or "").upper() in {
            "LONG",
            "SHORT",
        }:
            payload.pop("reduceOnly", None)
        return self._normalize_algo_order(self._client().new_algo_order(**payload))

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        return [
            self._normalize_order(item) for item in self._client().query_open_orders(symbol=symbol)
        ]

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        if not symbol:
            return False
        try:
            self._client().cancel_order(symbol=str(symbol), order_id=order_id)
        except BinanceFuturesAPIError:
            return False
        return True

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict:
        if symbol is None:
            return {}
        return self._normalize_order(self._client().query_order(symbol=symbol, order_id=order_id))

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        self._client().change_initial_leverage(symbol=symbol, leverage=int(leverage))
        return True

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        try:
            self._client().change_margin_type(symbol=symbol, margin_type=str(margin_mode))
        except BinanceFuturesAPIError as exc:
            if getattr(exc, "error_code", None) in {-4046, -4047, -4048}:
                return True
            raise
        return True

    def set_position_mode(self, position_mode: str) -> bool:
        try:
            self._client().change_position_mode(hedge_mode=str(position_mode).upper() == "HEDGE")
        except BinanceFuturesAPIError as exc:
            if getattr(exc, "error_code", None) in {-4059}:
                return True
            raise
        return True

    def create_listen_key(self) -> str:
        return self._client().create_user_data_stream()

    def keepalive_listen_key(self, listen_key: str) -> bool:
        self._client().keepalive_user_data_stream(str(listen_key))
        return True

    def close_listen_key(self, listen_key: str) -> bool:
        self._client().close_user_data_stream(str(listen_key))
        return True


__all__ = ["BinanceFuturesExchange"]
