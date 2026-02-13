from typing import Dict, List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from lumina_quant.interfaces import ExchangeInterface


class MT5Exchange(ExchangeInterface):
    """
    Implementation of ExchangeInterface using MetaTrader5.
    Requires MetaTrader5 python package and a running MT5 terminal.
    """

    def __init__(self, config):
        self.config = config
        self.connected = False
        self.connect()

    def connect(self):
        if mt5 is None:
            print("MetaTrader5 package not installed. Cannot connect.")
            return

        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            self.connected = False
        else:
            print("MetaTrader5 package version:", mt5.__version__)
            self.connected = True

            # Optional: Login if credentials provided in config
            login = getattr(self.config, "MT5_LOGIN", None)
            password = getattr(self.config, "MT5_PASSWORD", None)
            server = getattr(self.config, "MT5_SERVER", None)

            if login and password and server:
                authorized = mt5.login(login, password=password, server=server)
                if authorized:
                    print(f"Connected to account #{login}")
                else:
                    print(
                        "failed to connect at account #{}, error code: {}".format(
                            login, mt5.last_error()
                        )
                    )

    def get_balance(self, currency: str = "USDT") -> float:
        if not self.connected:
            return 0.0
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0
        # MT5 usually has one balance for the account. Currency check might be needed.
        # account_info.balance gives the account balance in deposit currency
        return account_info.balance

    def get_all_positions(self) -> Dict[str, float]:
        if not self.connected:
            return {}
        positions = mt5.positions_get()
        if positions is None:
            return {}

        result = {}
        for pos in positions:
            result[pos.symbol] = pos.volume
        return result

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[tuple]:
        if not self.connected:
            return []

        # Map timeframe string to MT5 constant
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }

        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)

        # Get rates
        # from pos 0 to limit
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)

        if rates is None:
            print(f"Failed to get rates for {symbol}")
            return []

        data = []
        for rate in rates:
            # rate is a numpy void object or similar struct
            # format: (time, open, high, low, close, tick_volume, spread, real_volume)
            # We need (timestamp_ms, open, high, low, close, volume)
            # MT5 time is seconds, CCXT usually uses ms. Let's stick to seconds for now or convert to ms?
            # Existing system uses unix timestamp (float or int).
            # CCXT returns ms usually. Strategy might expect seconds.
            # Let's check live_data.py usage. It uses timestamp directly.
            timestamp = rate["time"] * 1000  # Convert to ms to match CCXT
            data.append(
                (
                    timestamp,
                    rate["open"],
                    rate["high"],
                    rate["low"],
                    rate["close"],
                    float(rate["tick_volume"]),
                )
            )

        return data

    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        params = params or {}

        # Prepare request
        action = mt5.TRADE_ACTION_DEAL

        # side: buy or sell
        mt5_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

        # type: market or limit.
        # If limit, we need TRADE_ACTION_PENDING and proper ORDER_TYPE
        if type == "limit":
            action = mt5.TRADE_ACTION_PENDING
            if side == "buy":
                mt5_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                mt5_type = mt5.ORDER_TYPE_SELL_LIMIT

        # For Market Buy, price should be Ask. For Market Sell, price should be Bid.
        if type == "market":
            symbol_info = mt5.symbol_info_tick(symbol)
            if symbol_info is None:
                raise RuntimeError(f"Symbol {symbol} not found")

            if side == "buy":
                price = symbol_info.ask
            else:
                price = symbol_info.bid

        # Get defaults from config
        magic = params.get("magic", getattr(self.config, "MT5_MAGIC", 234000))
        deviation = params.get("deviation", getattr(self.config, "MT5_DEVIATION", 20))
        comment = params.get(
            "comment", getattr(self.config, "MT5_COMMENT", "LuminaQuant")
        )
        filler_type = params.get("type_filling", mt5.ORDER_FILLING_IOC)
        sl = params.get("sl", 0.0)
        tp = params.get("tp", 0.0)

        request = {
            "action": action,
            "symbol": symbol,
            "volume": quantity,
            "type": mt5_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filler_type,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("Order send failed, retcode={}".format(result.retcode))
            raise RuntimeError(f"Order failed: {result.comment}")

        return {
            "id": str(result.order),
            "status": "closed" if type == "market" else "open",  # Simplified
            "filled": result.volume,
            "average": result.price,
            "price": result.price,
            "amount": result.volume,
        }

    def load_markets(self) -> Dict:
        return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        _ = (symbol, leverage)
        return True

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        _ = (symbol, margin_mode)
        return True

    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        return [
            {"symbol": sym, "contracts": qty}
            for sym, qty in self.get_all_positions().items()
            if symbol is None or sym == symbol
        ]

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        _ = symbol
        if not self.connected:
            return {}
        try:
            ticket = int(order_id)
        except Exception:
            return {}

        orders = mt5.orders_get(ticket=ticket)
        if not orders:
            return {}
        order = orders[0]
        return {
            "id": str(order.ticket),
            "status": "open",
            "filled": order.volume_initial - order.volume_current,
            "average": order.price_open,
            "price": order.price_open,
            "amount": order.volume_initial,
            "symbol": order.symbol,
        }

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        if not self.connected:
            return []

        # mt5.orders_get(symbol=...) or group=...
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        result = []
        for order in orders:
            # order is a named tuple
            result.append(
                {
                    "id": str(order.ticket),
                    "symbol": order.symbol,
                    "type": "buy"
                    if order.type == mt5.ORDER_TYPE_BUY
                    else "sell",  # Simplified
                    "side": "buy"
                    if order.type
                    in [
                        mt5.ORDER_TYPE_BUY,
                        mt5.ORDER_TYPE_BUY_LIMIT,
                        mt5.ORDER_TYPE_BUY_STOP,
                    ]
                    else "sell",
                    "price": order.price_open,
                    "amount": order.volume_initial,
                    "filled": order.volume_current,  # In MT5 volume_current is remaining volume? Check docs.
                    # volume_current: VOLUME_CURRENT: Volume current
                    # volume_initial: VOLUME_INITIAL: Volume initial
                    # So filled = initial - current
                    "filled_qty": order.volume_initial - order.volume_current,
                    "status": "open",
                    "info": order._asdict(),
                }
            )
        return result

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        if not self.connected:
            return False

        # To cancel, we send a TRADE_ACTION_REMOVE
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
            "comment": "LuminaQuant Cancel",
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Cancel failed: {result.comment}")
            return False
        return True

    def __del__(self):
        if mt5:
            mt5.shutdown()
