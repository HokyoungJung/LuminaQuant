from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ExchangeInterface(ABC):
    """
    Abstract Interface for interacting with different exchanges.
    """

    @abstractmethod
    def connect(self):
        """
        Establishes connection to the exchange API or terminal.
        """
        pass

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> float:
        """
        Returns the free balance of the specified currency.
        """
        pass

    @abstractmethod
    def get_all_positions(self) -> Dict[str, float]:
        """
        Returns a dictionary of open positions {symbol: quantity}.
        """
        pass

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[tuple]:
        """
        Fetches OHLCV data.
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
        price: Optional[float] = None,
        params: Dict = {},
    ) -> Dict:
        """
        Executes an order on the exchange.
        Returns the order details (id, status, filled_qty, average_price).
        """
        pass

    @abstractmethod
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetches open orders. Returns a list of order dicts."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancels an order. Returns True if successful."""
        pass

    # Optional extension points for futures/live robustness.
    def load_markets(self) -> Dict:
        return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        _ = (symbol, leverage)
        return True

    def set_margin_mode(self, symbol: str, margin_mode: str) -> bool:
        _ = (symbol, margin_mode)
        return True

    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        _ = symbol
        return []

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        _ = (order_id, symbol)
        return {}
