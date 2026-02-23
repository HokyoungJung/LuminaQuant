import os
import sys
import unittest
from unittest.mock import MagicMock

# Add Parent Dir to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import module under test
# It handles ImportError of MetaTrader5, setting mt5=None
import lumina_quant.exchanges.mt5_exchange as mt5_module
from lumina_quant.exchanges.mt5_exchange import MT5Exchange

# Prepare Mock
mock_mt5 = MagicMock()
# Set constants
mock_mt5.TIMEFRAME_M1 = 1
mock_mt5.TRADE_ACTION_DEAL = 1
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
mock_mt5.TRADE_ACTION_PENDING = 5
mock_mt5.ORDER_TIME_GTC = 0
mock_mt5.ORDER_FILLING_IOC = 1
mock_mt5.TRADE_RETCODE_DONE = 10009
mock_mt5.TRADE_ACTION_REMOVE = 8
mock_mt5.__version__ = "5.0.0"
mock_mt5.initialize.return_value = True
mock_mt5.login.return_value = True


class MockConfig:
    MT5_LOGIN = 123
    MT5_PASSWORD = "pass"
    MT5_SERVER = "server"
    MT5_MAGIC = 111111
    MT5_DEVIATION = 10


class TestMT5Exchange(unittest.TestCase):
    def setUp(self):
        # Manually patch the module global
        self.original_mt5 = mt5_module.mt5
        mt5_module.mt5 = mock_mt5

        self.config = MockConfig()
        # Ensure mock is reset or re-configured if needed
        mock_mt5.reset_mock()
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True

        # Instantiate with the patched mt5
        self.exchange = MT5Exchange(self.config)
        print(f"DEBUG: Connected state after init: {self.exchange.connected}")

    def tearDown(self):
        mt5_module.mt5 = self.original_mt5

    def test_connect(self):
        mock_mt5.initialize.assert_called()
        mock_mt5.login.assert_called_with(123, password="pass", server="server")
        self.assertTrue(self.exchange.connected)

    def test_execute_order_defaults(self):
        # Setup mock symbol info
        mock_info = MagicMock()
        mock_info.ask = 1.05
        mock_mt5.symbol_info_tick.return_value = mock_info

        # Setup order_send result
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 999
        mock_result.volume = 1.0
        mock_result.price = 1.05
        mock_mt5.order_send.return_value = mock_result

        # Execute Market Buy
        self.exchange.execute_order(symbol="EURUSD", type="market", side="buy", quantity=1.0)

        # Verify arguments
        args, _ = mock_mt5.order_send.call_args
        request = args[0]

        self.assertEqual(request["magic"], 111111)  # From Config
        self.assertEqual(request["deviation"], 10)  # From Config
        self.assertEqual(request["comment"], "LuminaQuant")  # Default

    def test_execute_order_params_override(self):
        # Setup mock
        mock_info = MagicMock()
        mock_info.bid = 1.04
        mock_mt5.symbol_info_tick.return_value = mock_info

        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 1000
        mock_result.volume = 0.5
        mock_result.price = 1.04
        mock_mt5.order_send.return_value = mock_result

        # Execute Market Sell with overrides
        self.exchange.execute_order(
            symbol="EURUSD",
            type="market",
            side="sell",
            quantity=0.5,
            params={"magic": 999999, "deviation": 50, "comment": "Override"},
        )

        args, _ = mock_mt5.order_send.call_args
        request = args[0]

        self.assertEqual(request["magic"], 999999)  # Overridden
        self.assertEqual(request["deviation"], 50)  # Overridden
        self.assertEqual(request["comment"], "Override")  # Overridden


if __name__ == "__main__":
    unittest.main(verbosity=2)
