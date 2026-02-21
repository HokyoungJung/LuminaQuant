import os
import sys
import unittest
from unittest.mock import MagicMock

# Add Parent Dir to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import module under test
import lumina_quant.exchanges.ccxt_exchange as ccxt_module
from lumina_quant.exchanges.ccxt_exchange import CCXTExchange

# Prepare Mock
mock_ccxt = MagicMock()
mock_exchange_inst = MagicMock()
mock_ccxt.binance.return_value = mock_exchange_inst


class MockConfig:
    EXCHANGE = {
        "driver": "ccxt",
        "name": "binance",
        "market_type": "spot",
    }
    BINANCE_API_KEY = "test_key"
    BINANCE_SECRET_KEY = "test_secret"
    IS_TESTNET = True


class FutureConfig(MockConfig):
    EXCHANGE = {
        "driver": "ccxt",
        "name": "binance",
        "market_type": "future",
    }
    IS_TESTNET = False


class TestCCXTExchange(unittest.TestCase):
    def setUp(self):
        # Manually patch ccxt in the module
        self.original_ccxt = ccxt_module.ccxt
        ccxt_module.ccxt = mock_ccxt

        self.config = MockConfig()
        mock_exchange_inst.reset_mock()

        self.exchange = CCXTExchange(self.config)

    def tearDown(self):
        ccxt_module.ccxt = self.original_ccxt

    def test_connect(self):
        # Verify binance was instantiated with correct keys
        mock_ccxt.binance.assert_called_with(
            {
                "apiKey": "test_key",
                "secret": "test_secret",
                "enableRateLimit": True,
            }
        )
        # Verify sandbox mode was set
        mock_exchange_inst.set_sandbox_mode.assert_called_with(True)

    def test_get_balance(self):
        mock_exchange_inst.fetch_balance.return_value = {
            "USDT": {"free": 1000.0},
            "BTC": {"free": 0.5},
        }
        balance = self.exchange.get_balance("USDT")
        self.assertEqual(balance, 1000.0)

    def test_execute_order(self):
        mock_exchange_inst.create_order.return_value = {
            "id": "12345",
            "status": "closed",
            "filled": 1.0,
            "price": 100.0,
            "amount": 1.0,
        }

        result = self.exchange.execute_order(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            quantity=1.0,
            price=100.0,
            params={"timeInForce": "GTC"},
        )

        mock_exchange_inst.create_order.assert_called_with(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            amount=1.0,
            price=100.0,
            params={"timeInForce": "GTC"},
        )
        self.assertEqual(result["id"], "12345")

    def test_fetch_open_orders(self):
        mock_exchange_inst.fetch_open_orders.return_value = [
            {
                "id": "111",
                "symbol": "BTC/USDT",
                "type": "limit",
                "side": "buy",
                "price": 90.0,
                "amount": 1.0,
                "filled": 0.0,
                "status": "open",
                "info": {},
            }
        ]

        orders = self.exchange.fetch_open_orders("BTC/USDT")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["id"], "111")
        mock_exchange_inst.fetch_open_orders.assert_called_with("BTC/USDT")

    def test_cancel_order(self):
        self.exchange.cancel_order("111", "BTC/USDT")
        mock_exchange_inst.cancel_order.assert_called_with("111", "BTC/USDT")

    def test_get_all_position_legs_future_mode(self):
        mock_exchange_inst.reset_mock()
        exchange = CCXTExchange(FutureConfig())
        mock_exchange_inst.fetch_positions.return_value = [
            {"symbol": "BTC/USDT", "contracts": 0.4, "side": "long"},
            {"symbol": "BTC/USDT", "contracts": 0.1, "side": "short"},
            {"symbol": "ETH/USDT", "positionAmt": "-0.2", "info": {"positionSide": "SHORT"}},
        ]

        legs = exchange.get_all_position_legs()
        net = exchange.get_all_positions()

        self.assertEqual(legs["BTC/USDT"]["LONG"], 0.4)
        self.assertEqual(legs["BTC/USDT"]["SHORT"], 0.1)
        self.assertEqual(legs["ETH/USDT"]["SHORT"], 0.2)
        self.assertAlmostEqual(net["BTC/USDT"], 0.3)
        self.assertAlmostEqual(net["ETH/USDT"], -0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
