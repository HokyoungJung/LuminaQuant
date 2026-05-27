import json
import os
import unittest
from unittest.mock import patch

import lumina_quant.exchanges.mt5_exchange as mt5_module
from lumina_quant.exchanges.mt5_exchange import MT5Exchange


class _BridgeConfig:
    MT5_LOGIN = 123456
    MT5_PASSWORD = "pass"
    MT5_SERVER = "demo-server"
    MT5_MAGIC = 234000
    MT5_DEVIATION = 20


def _proc(payload: dict) -> object:
    class _Result:
        returncode = 0
        stdout = json.dumps(payload, ensure_ascii=True)
        stderr = ""

    return _Result()


class TestMT5ExchangeBridge(unittest.TestCase):
    def setUp(self):
        self.original_mt5 = mt5_module.mt5
        mt5_module.mt5 = None
        self.original_bridge_python = os.environ.get("LQ_MT5_BRIDGE_PYTHON")
        self.original_bridge_script = os.environ.get("LQ_MT5_BRIDGE_SCRIPT")
        self.original_bridge_wslpath = os.environ.get("LQ_MT5_BRIDGE_USE_WSLPATH")
        os.environ["LQ_MT5_BRIDGE_PYTHON"] = "python.exe"
        os.environ.pop("LQ_MT5_BRIDGE_SCRIPT", None)
        os.environ["LQ_MT5_BRIDGE_USE_WSLPATH"] = "0"

    def tearDown(self):
        mt5_module.mt5 = self.original_mt5
        if self.original_bridge_python is None:
            os.environ.pop("LQ_MT5_BRIDGE_PYTHON", None)
        else:
            os.environ["LQ_MT5_BRIDGE_PYTHON"] = self.original_bridge_python

        if self.original_bridge_script is None:
            os.environ.pop("LQ_MT5_BRIDGE_SCRIPT", None)
        else:
            os.environ["LQ_MT5_BRIDGE_SCRIPT"] = self.original_bridge_script

        if self.original_bridge_wslpath is None:
            os.environ.pop("LQ_MT5_BRIDGE_USE_WSLPATH", None)
        else:
            os.environ["LQ_MT5_BRIDGE_USE_WSLPATH"] = self.original_bridge_wslpath

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_bridge_connect_and_fetch_ohlcv(self, run_mock):
        run_mock.side_effect = [
            _proc({"ok": True, "result": {"connected": True}, "error": ""}),
            _proc(
                {
                    "ok": True,
                    "result": [[1700000000000, 1.0, 2.0, 0.5, 1.5, 10.0]],
                    "error": "",
                }
            ),
        ]

        exchange = MT5Exchange(_BridgeConfig())
        self.assertTrue(exchange.connected)

        candles = exchange.fetch_ohlcv("EURUSD", "1m", 1)
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0][0], 1700000000000)
        self.assertEqual(candles[0][4], 1.5)

        first_cmd = run_mock.call_args_list[0].args[0]
        self.assertIn("--action", first_cmd)
        self.assertEqual(first_cmd[first_cmd.index("--action") + 1], "connect")

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_bridge_execute_order(self, run_mock):
        run_mock.side_effect = [
            _proc({"ok": True, "result": {"connected": True}, "error": ""}),
            _proc(
                {
                    "ok": True,
                    "result": {
                        "id": "123",
                        "status": "closed",
                        "filled": 0.1,
                        "average": 1.2345,
                        "price": 1.2345,
                        "amount": 0.1,
                    },
                    "error": "",
                }
            ),
        ]

        exchange = MT5Exchange(_BridgeConfig())
        payload = exchange.execute_order(
            symbol="EURUSD",
            type="market",
            side="buy",
            quantity=0.1,
        )
        self.assertEqual(payload.get("id"), "123")
        self.assertEqual(payload.get("status"), "closed")
