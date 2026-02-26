from typing import Any, cast

import ccxt
from lumina_quant.core.protocols import ExchangeInterface


class CCXTExchange(ExchangeInterface):
    """Implementation of ExchangeInterface using CCXT (e.g. for Binance)."""

    def __init__(self, config):
        self.config = config
        self.exchange: Any = None
        self.market_type = "spot"
        self._markets = {}
        self.connect()

    def _client(self) -> Any:
        if self.exchange is None:
            raise RuntimeError("Exchange client is not initialized")
        return cast(Any, self.exchange)

    def connect(self):
        exchange_config = getattr(self.config, "EXCHANGE", None)
        if not isinstance(exchange_config, dict):
            raise ValueError("EXCHANGE config must be a dictionary with name/market_type fields.")

        exchange_id = str(exchange_config.get("name", "")).strip().lower()
        self.market_type = str(exchange_config.get("market_type", "spot")).strip().lower()
        if not exchange_id:
            raise ValueError("EXCHANGE.name must be configured.")

        exchange_class = getattr(ccxt, exchange_id)

        exchange_kwargs = {
            "apiKey": getattr(self.config, "BINANCE_API_KEY", ""),
            "secret": getattr(self.config, "BINANCE_SECRET_KEY", ""),
            "enableRateLimit": True,
        }
        if str(self.market_type).lower() == "future":
            exchange_kwargs["options"] = {"defaultType": "future"}

        self.exchange = exchange_class(exchange_kwargs)
        ex = self._client()

        if getattr(self.config, "IS_TESTNET", False):
            ex.set_sandbox_mode(True)
            print(f"CCXTExchange ({exchange_id}): Running in Sandbox/Testnet Mode")

        # Futures setup (best-effort; some exchanges don't expose all endpoints)
        if str(self.market_type).lower() == "future":
            try:
                self.load_markets()
            except Exception:
                pass

            position_mode = getattr(self.config, "POSITION_MODE", "HEDGE").upper()
            try:
                if hasattr(ex, "set_position_mode"):
                    ex.set_position_mode(position_mode == "HEDGE")
            except Exception as e:
                print(f"Warning: failed to set position mode: {e}")

            margin_mode = getattr(self.config, "MARGIN_MODE", "isolated")
            leverage = int(getattr(self.config, "LEVERAGE", 1))
            for symbol in getattr(self.config, "SYMBOLS", []):
                self.set_margin_mode(symbol, margin_mode)
                self.set_leverage(symbol, leverage)

    def get_balance(self, currency: str = "USDT") -> float:
        try:
            ex = self._client()
            params = {}
            if str(self.market_type).lower() == "future":
                params = {"type": "future"}
            balance = ex.fetch_balance(params)
            entry = balance.get(currency, {})
            if isinstance(entry, dict):
                return float(entry.get("free", 0.0))
            if "free" in balance and isinstance(balance["free"], dict):
                return float(balance["free"].get(currency, 0.0))
            return 0.0
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    @staticmethod
    def _as_float(value, default=0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _extract_signed_position_qty(self, position: dict) -> float:
        info = position.get("info") if isinstance(position, dict) else {}
        if not isinstance(info, dict):
            info = {}

        candidates = [
            position.get("contracts"),
            position.get("positionAmt"),
            position.get("size"),
            position.get("amount"),
            info.get("positionAmt"),
            info.get("contracts"),
            info.get("size"),
        ]
        qty = 0.0
        for item in candidates:
            qty = self._as_float(item, 0.0)
            if abs(qty) > 0.0:
                break

        side_token = str(
            position.get("side") or position.get("positionSide") or info.get("positionSide") or ""
        ).upper()
        if side_token == "SHORT" and qty > 0.0:
            return -qty
        if side_token == "LONG" and qty < 0.0:
            return abs(qty)
        return qty

    def get_all_positions(self) -> dict[str, float]:
        positions = {}
        try:
            if str(self.market_type).lower() == "future":
                for p in self.fetch_positions():
                    symbol = p.get("symbol")
                    qty = self._extract_signed_position_qty(p)
                    if symbol and abs(qty) > 0.0:
                        positions[symbol] = positions.get(symbol, 0.0) + qty
                return positions

            ex = self._client()
            bal = ex.fetch_balance()
            total = bal.get("total", {})
            for coin, qty in total.items():
                if qty > 0:
                    symbol = f"{coin}/USDT"
                    positions[symbol] = qty
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {}

    def get_all_position_legs(self) -> dict[str, dict[str, float]]:
        legs: dict[str, dict[str, float]] = {}
        try:
            if str(self.market_type).lower() != "future":
                for symbol, qty in self.get_all_positions().items():
                    if qty > 0:
                        legs[str(symbol)] = {"LONG": float(qty), "SHORT": 0.0}
                    elif qty < 0:
                        legs[str(symbol)] = {"LONG": 0.0, "SHORT": abs(float(qty))}
                return legs

            for position in self.fetch_positions():
                symbol = position.get("symbol")
                if not symbol:
                    continue
                signed_qty = self._extract_signed_position_qty(position)
                if abs(signed_qty) <= 0.0:
                    continue
                bucket = legs.setdefault(str(symbol), {"LONG": 0.0, "SHORT": 0.0})
                if signed_qty > 0:
                    bucket["LONG"] += abs(float(signed_qty))
                else:
                    bucket["SHORT"] += abs(float(signed_qty))

            return {
                symbol: payload
                for symbol, payload in legs.items()
                if payload.get("LONG", 0.0) > 0.0 or payload.get("SHORT", 0.0) > 0.0
            }
        except Exception as e:
            print(f"Error fetching position legs: {e}")
            return {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[tuple]:
        try:
            ex = self._client()
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            # ccxt returns [timestamp, open, high, low, close, volume]
            # convert to list of tuples
            return [tuple(candle[:6]) for candle in ohlcv]
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
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
        try:
            ex = self._client()
            order_params = dict(params or {})
            # Type: market or limit
            # Side: buy or sell
            order = ex.create_order(
                symbol=symbol,
                type=type,
                side=side,
                amount=quantity,
                price=price,
                params=order_params,
            )

            # Standardize return
            return {
                "id": order["id"],
                "status": order["status"],
                "filled": order.get("filled", 0.0),
                "average": order.get("average", order.get("price")),
                "price": order.get("price"),
                "amount": order.get("amount"),
                "remaining": order.get("remaining"),
                "timestamp": order.get("timestamp"),
                "fee": order.get("fee"),
                "info": order.get("info"),
            }
        except Exception as e:
            print(f"Error executing order: {e}")
            raise e

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        try:
            ex = self._client()
            orders = ex.fetch_open_orders(symbol)
            result = []
            for order in orders:
                result.append(
                    {
                        "id": order["id"],
                        "symbol": order["symbol"],
                        "type": order["type"],
                        "side": order["side"],
                        "price": order["price"],
                        "amount": order["amount"],
                        "filled": order["filled"],
                        "status": order["status"],
                        "info": order["info"],
                    }
                )
            return result
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            ex = self._client()
            ex.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False

    def load_markets(self) -> dict:
        ex = self._client()
        self._markets = ex.load_markets()
        return self._markets

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            ex = self._client()
            if hasattr(ex, "set_leverage"):
                ex.set_leverage(int(leverage), symbol)
            return True
        except Exception as e:
            print(f"Warning: failed to set leverage for {symbol}: {e}")
            return False

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        try:
            ex = self._client()
            if hasattr(ex, "set_margin_mode"):
                ex.set_margin_mode(margin_mode, symbol)
            return True
        except Exception as e:
            print(f"Warning: failed to set margin mode for {symbol}: {e}")
            return False

    def fetch_positions(self, symbol: str | None = None) -> list[dict]:
        try:
            ex = self._client()
            if hasattr(ex, "fetch_positions"):
                return ex.fetch_positions([symbol] if symbol else None)
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict:
        try:
            ex = self._client()
            order = ex.fetch_order(order_id, symbol)
            return {
                "id": order.get("id"),
                "status": order.get("status"),
                "filled": order.get("filled", 0.0),
                "average": order.get("average", order.get("price")),
                "price": order.get("price"),
                "amount": order.get("amount"),
                "remaining": order.get("remaining"),
                "timestamp": order.get("timestamp"),
                "fee": order.get("fee"),
                "info": order.get("info"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "type": order.get("type"),
            }
        except Exception as e:
            print(f"Error fetching order {order_id}: {e}")
            return {}

    def get_market_spec(self, symbol: str) -> dict:
        if not self._markets:
            self.load_markets()
        market = self._markets.get(symbol, {})
        limits = market.get("limits", {})
        precision = market.get("precision", {})
        qty_step = precision.get("amount")
        # Some exchanges expose precision as decimal places (e.g., 3 -> 0.001)
        if isinstance(qty_step, int):
            qty_step = 10 ** (-qty_step)
        return {
            "min_qty": (limits.get("amount", {}) or {}).get("min"),
            "qty_step": qty_step,
            "min_notional": (limits.get("cost", {}) or {}).get("min"),
        }
