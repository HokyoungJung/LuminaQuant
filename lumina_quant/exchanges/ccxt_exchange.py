import ccxt
from typing import Dict, List, Optional
from lumina_quant.interfaces import ExchangeInterface


class CCXTExchange(ExchangeInterface):
    """
    Implementation of ExchangeInterface using CCXT (e.g. for Binance).
    """

    def __init__(self, config):
        self.config = config
        self.exchange = None
        self.market_type = "spot"
        self._markets = {}
        self.connect()

    def connect(self):
        # Determine exchange ID from config or default to binance
        exchange_id = "binance"
        exchange_config = getattr(self.config, "EXCHANGE", None)
        if exchange_config and isinstance(exchange_config, dict):
            exchange_id = exchange_config.get("name", "binance")
            self.market_type = exchange_config.get("market_type", "spot")
        else:
            exchange_id = getattr(self.config, "EXCHANGE_ID", "binance")
            self.market_type = getattr(self.config, "MARKET_TYPE", "spot")

        if not exchange_id:
            exchange_id = "binance"

        exchange_class = getattr(ccxt, exchange_id)

        exchange_kwargs = {
            "apiKey": getattr(self.config, "BINANCE_API_KEY", ""),
            "secret": getattr(self.config, "BINANCE_SECRET_KEY", ""),
            "enableRateLimit": True,
        }
        if str(self.market_type).lower() == "future":
            exchange_kwargs["options"] = {"defaultType": "future"}

        self.exchange = exchange_class(exchange_kwargs)

        if getattr(self.config, "IS_TESTNET", False):
            self.exchange.set_sandbox_mode(True)
            print(f"CCXTExchange ({exchange_id}): Running in Sandbox/Testnet Mode")

        # Futures setup (best-effort; some exchanges don't expose all endpoints)
        if str(self.market_type).lower() == "future":
            try:
                self.load_markets()
            except Exception:
                pass

            position_mode = getattr(self.config, "POSITION_MODE", "HEDGE").upper()
            try:
                if hasattr(self.exchange, "set_position_mode"):
                    self.exchange.set_position_mode(position_mode == "HEDGE")
            except Exception as e:
                print(f"Warning: failed to set position mode: {e}")

            margin_mode = getattr(self.config, "MARGIN_MODE", "isolated")
            leverage = int(getattr(self.config, "LEVERAGE", 1))
            for symbol in getattr(self.config, "SYMBOLS", []):
                self.set_margin_mode(symbol, margin_mode)
                self.set_leverage(symbol, leverage)

    def get_balance(self, currency: str = "USDT") -> float:
        try:
            params = {}
            if str(self.market_type).lower() == "future":
                params = {"type": "future"}
            balance = self.exchange.fetch_balance(params)
            entry = balance.get(currency, {})
            if isinstance(entry, dict):
                return float(entry.get("free", 0.0))
            if "free" in balance and isinstance(balance["free"], dict):
                return float(balance["free"].get(currency, 0.0))
            return 0.0
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def get_all_positions(self) -> Dict[str, float]:
        positions = {}
        try:
            if str(self.market_type).lower() == "future":
                for p in self.fetch_positions():
                    symbol = p.get("symbol")
                    qty = float(p.get("contracts") or p.get("positionAmt") or p.get("size") or 0.0)
                    if symbol and qty != 0:
                        positions[symbol] = positions.get(symbol, 0.0) + qty
                return positions

            bal = self.exchange.fetch_balance()
            total = bal.get("total", {})
            for coin, qty in total.items():
                if qty > 0:
                    symbol = f"{coin}/USDT"
                    positions[symbol] = qty
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[tuple]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        try:
            order_params = dict(params or {})
            # Type: market or limit
            # Side: buy or sell
            order = self.exchange.create_order(
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

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
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

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False

    def load_markets(self) -> Dict:
        self._markets = self.exchange.load_markets()
        return self._markets

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            if hasattr(self.exchange, "set_leverage"):
                self.exchange.set_leverage(int(leverage), symbol)
            return True
        except Exception as e:
            print(f"Warning: failed to set leverage for {symbol}: {e}")
            return False

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        try:
            if hasattr(self.exchange, "set_margin_mode"):
                self.exchange.set_margin_mode(margin_mode, symbol)
            return True
        except Exception as e:
            print(f"Warning: failed to set margin mode for {symbol}: {e}")
            return False

    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            if hasattr(self.exchange, "fetch_positions"):
                return self.exchange.fetch_positions([symbol] if symbol else None)
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        try:
            order = self.exchange.fetch_order(order_id, symbol)
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

    def get_market_spec(self, symbol: str) -> Dict:
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
