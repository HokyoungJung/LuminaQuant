"""Tests for the MT5Exchange real-money execution fence (supports_real_money=False)."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lumina_quant.exchanges.mt5_exchange as mt5_module
from lumina_quant.exchanges.mt5_exchange import MT5Exchange


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _proc(payload: dict) -> object:
    class _Result:
        returncode = 0
        stdout = json.dumps(payload, ensure_ascii=True)
        stderr = ""

    return _Result()


def _paper_config(**overrides):
    ns = SimpleNamespace(
        MT5_LOGIN=123,
        MT5_PASSWORD="pass",
        MT5_SERVER="demo-server",
        MT5_MAGIC=234000,
        MT5_DEVIATION=20,
        MT5_COMMENT="LuminaQuant",
        MODE="paper",
        MT5_ALLOW_REAL_EXECUTION=False,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _real_config(**overrides):
    ns = _paper_config(MODE="real", MT5_ALLOW_REAL_EXECUTION=True)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


# ---------------------------------------------------------------------------
# Helpers to build a mock_mt5
# ---------------------------------------------------------------------------


def _make_mock_mt5():
    mock_mt5 = MagicMock()
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
    return mock_mt5


# ---------------------------------------------------------------------------
# Class-level capability flag
# ---------------------------------------------------------------------------


class TestMT5FenceClassAttr(unittest.TestCase):
    def test_supports_real_money_is_false(self):
        self.assertFalse(MT5Exchange.supports_real_money)


# ---------------------------------------------------------------------------
# In-process MT5 path: fence blocks real-money, paper is unaffected
# ---------------------------------------------------------------------------


class TestMT5FenceInProcess(unittest.TestCase):
    def setUp(self):
        self.original_mt5 = mt5_module.mt5
        self.mock_mt5 = _make_mock_mt5()
        mt5_module.mt5 = self.mock_mt5

    def tearDown(self):
        mt5_module.mt5 = self.original_mt5

    def test_real_money_execute_order_raises(self):
        exchange = MT5Exchange(_real_config())
        with self.assertRaises(RuntimeError) as ctx:
            exchange.execute_order("EURUSD", "market", "buy", 1.0)
        self.assertIn("not approved for real-money execution", str(ctx.exception))

    def test_paper_mode_execute_order_proceeds(self):
        """Paper mode must NOT be blocked by the fence."""
        exchange = MT5Exchange(_paper_config())
        self.assertTrue(exchange.connected)

        # Set up mock so order_send succeeds
        mock_info = MagicMock()
        mock_info.ask = 1.05
        self.mock_mt5.symbol_info_tick.return_value = mock_info

        mock_result = MagicMock()
        mock_result.retcode = self.mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 42
        mock_result.volume = 1.0
        mock_result.price = 1.05
        self.mock_mt5.order_send.return_value = mock_result

        result = exchange.execute_order("EURUSD", "market", "buy", 1.0)
        self.assertEqual(result["id"], "42")

    def test_real_flag_false_is_not_fenced(self):
        """MODE=real but MT5_ALLOW_REAL_EXECUTION=False must NOT raise the fence."""
        exchange = MT5Exchange(_real_config(MT5_ALLOW_REAL_EXECUTION=False))
        self.assertTrue(exchange.connected)

        mock_info = MagicMock()
        mock_info.ask = 1.05
        self.mock_mt5.symbol_info_tick.return_value = mock_info

        mock_result = MagicMock()
        mock_result.retcode = self.mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 99
        mock_result.volume = 0.5
        mock_result.price = 1.05
        self.mock_mt5.order_send.return_value = mock_result

        result = exchange.execute_order("EURUSD", "market", "buy", 0.5)
        self.assertEqual(result["id"], "99")


# ---------------------------------------------------------------------------
# Bridge mode: fence blocks real-money, paper is unaffected
# ---------------------------------------------------------------------------


class TestMT5FenceBridge(unittest.TestCase):
    def setUp(self):
        self.original_mt5 = mt5_module.mt5
        mt5_module.mt5 = None
        self._orig_bridge_python = os.environ.get("LQ_MT5_BRIDGE_PYTHON")
        self._orig_bridge_wslpath = os.environ.get("LQ_MT5_BRIDGE_USE_WSLPATH")
        os.environ["LQ_MT5_BRIDGE_PYTHON"] = "python.exe"
        os.environ["LQ_MT5_BRIDGE_USE_WSLPATH"] = "0"

    def tearDown(self):
        mt5_module.mt5 = self.original_mt5
        if self._orig_bridge_python is None:
            os.environ.pop("LQ_MT5_BRIDGE_PYTHON", None)
        else:
            os.environ["LQ_MT5_BRIDGE_PYTHON"] = self._orig_bridge_python
        if self._orig_bridge_wslpath is None:
            os.environ.pop("LQ_MT5_BRIDGE_USE_WSLPATH", None)
        else:
            os.environ["LQ_MT5_BRIDGE_USE_WSLPATH"] = self._orig_bridge_wslpath

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_bridge_real_money_execute_order_raises(self, run_mock):
        run_mock.return_value = _proc({"ok": True, "result": {"connected": True}, "error": ""})
        exchange = MT5Exchange(_real_config())
        with self.assertRaises(RuntimeError) as ctx:
            exchange.execute_order("EURUSD", "market", "buy", 0.1)
        self.assertIn("not approved for real-money execution", str(ctx.exception))
        # Bridge order_send should NOT have been called (fence fires first)
        self.assertEqual(run_mock.call_count, 1)  # only the connect call

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_bridge_paper_mode_execute_order_proceeds(self, run_mock):
        run_mock.side_effect = [
            _proc({"ok": True, "result": {"connected": True}, "error": ""}),
            _proc(
                {
                    "ok": True,
                    "result": {
                        "id": "55",
                        "status": "closed",
                        "filled": 0.1,
                        "average": 1.25,
                        "price": 1.25,
                        "amount": 0.1,
                    },
                    "error": "",
                }
            ),
        ]
        exchange = MT5Exchange(_paper_config())
        result = exchange.execute_order("EURUSD", "market", "buy", 0.1)
        self.assertEqual(result["id"], "55")
        self.assertEqual(result["status"], "closed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
