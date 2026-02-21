from abc import ABC, abstractmethod


class ExchangeInterface(ABC):
    """Abstract Interface for interacting with different exchanges."""

    @abstractmethod
    def connect(self):
        """Establishes connection to the exchange API or terminal."""
        pass

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> float:
        """Returns the free balance of the specified currency."""
        pass

    @abstractmethod
    def get_all_positions(self) -> dict[str, float]:
        """Returns a dictionary of open positions {symbol: quantity}."""
        pass

    def get_all_position_legs(self) -> dict[str, dict[str, float]]:
        """Optional side-aware positions {symbol: {LONG: qty, SHORT: qty}}."""
        return {}

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[tuple]:
        """Fetches OHLCV data.
        Returns a list of tuples: (timestamp, open, high, low, close, volume)
        """
        pass

    @abstractmethod
    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        """Executes an order on the exchange.
        Returns the order details (id, status, filled_qty, average_price).
        """
        pass

    @abstractmethod
    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetches open orders. Returns a list of order dicts."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancels an order. Returns True if successful."""
        pass

    # Optional extension points for futures/live robustness.
    def load_markets(self) -> dict:
        return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        _ = (symbol, leverage)
        return True

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        _ = (symbol, margin_mode)
        return True

    def fetch_positions(self, symbol: str | None = None) -> list[dict]:
        _ = symbol
        return []

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict:
        _ = (order_id, symbol)
        return {}
