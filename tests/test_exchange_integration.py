import os
import sys
import unittest
from unittest.mock import MagicMock

# Add Parent Dir to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumina_quant.interfaces import ExchangeInterface
from lumina_quant.live_data import LiveDataHandler
from lumina_quant.live_execution import LiveExecutionHandler
from lumina_quant.live_trader import LiveTrader
from lumina_quant.portfolio import Portfolio


class MockConfig:
    EXCHANGE = {"driver": "mock_driver", "name": "mock_exchange"}
    BINANCE_API_KEY = "test"
    BINANCE_SECRET_KEY = "test"
    IS_TESTNET = True
    TIMEFRAME = "1m"
    POLL_INTERVAL = 1
    SYMBOLS = ["BTC/USDT"]
    INITIAL_CAPITAL = 10000
    TARGET_ALLOCATION = 0.1
    COMMISSION_RATE = 0.001
    SLIPPAGE_RATE = 0.0005
    MIN_TRADE_QTY = 0.001


class MockExchange(ExchangeInterface):
    def connect(self):
        pass

    def get_balance(self, currency="USDT"):
        return 10000.0

    def get_all_positions(self):
        return {}

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        return []

    def execute_order(self, symbol, type, side, quantity, price=None, params={}):
        return {
            "id": "123",
            "status": "closed",
            "filled": quantity,
            "average": price or 100.0,
            "price": price or 100.0,
            "amount": quantity,
        }

    def fetch_open_orders(self, symbol=None):
        return []

    def cancel_order(self, order_id, symbol=None):
        return True


# Patch get_exchange in live_trader namespace
import lumina_quant.live_trader


def mock_get_exchange(config):
    return MockExchange()


lumina_quant.live_trader.get_exchange = mock_get_exchange


class TestExchangeIntegration(unittest.TestCase):
    def test_instantiation(self):
        print("\nTesting LiveTrader Instantiation with MockExchange...")

        # We need to mock strategy to avoid complex logic
        mock_strategy_cls = MagicMock()

        trader = LiveTrader(
            symbol_list=["BTC/USDT"],
            data_handler_cls=LiveDataHandler,
            execution_handler_cls=LiveExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=mock_strategy_cls,
        )

        self.assertIsInstance(trader.exchange, MockExchange)
        self.assertIsInstance(trader.data_handler, LiveDataHandler)
        self.assertIsInstance(trader.execution_handler, LiveExecutionHandler)

        print("LiveTrader instantiated successfully with Exchange.")

        # Test basic method delegation
        balance = trader.execution_handler.get_balance()
        self.assertEqual(balance, 10000.0)
        print("get_balance delegated correctly.")

        # Test new interface methods existence
        try:
            trader.exchange.fetch_open_orders()
            trader.exchange.cancel_order("123")
            print("New interface methods called successfully.")
        except AttributeError:
            self.fail("ExchangeInterface missing new methods!")


if __name__ == "__main__":
    unittest.main()
