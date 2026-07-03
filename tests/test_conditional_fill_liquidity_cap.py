"""Conditional-fill liquidity cap gate (2026-07-03 audit finding B).

Triggered STOP/TAKE_PROFIT/TRAIL_STOP fills historically bypassed the
max_bar_volume_ratio liquidity cap (stops fire exactly on the bars where
liquidity is worst). execution.apply_liquidity_cap_to_conditional_fills
(default OFF = legacy) caps the fill and chases the excess as a MKT
remainder, mirroring the MKT partial-fill path.
"""

import os
import queue
import unittest
from datetime import datetime

from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler


class MockBars:
    symbol_list = ["BTC/USDT"]

    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)


class BaseConfig:
    TAKER_FEE_RATE = 0.0004
    COMMISSION_RATE = 0.0004
    MAKER_FEE_RATE = 0.0002
    SPREAD_RATE = 0.0
    SLIPPAGE_RATE = 0.0
    MAX_BAR_VOLUME_RATIO = 0.1
    LEVERAGE = 1
    MAINTENANCE_MARGIN_RATE = 0.005
    RANDOM_SEED = 42


class CappedConfig(BaseConfig):
    APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS = True


class Market:
    type = "MARKET"

    def __init__(self):
        self.time = datetime(2026, 1, 1, 1, 0)
        self.symbol = "BTC/USDT"
        self.open = 100.0
        self.high = 101.0
        self.low = 99.0
        self.close = 100.0
        self.volume = 500.0


class TestConditionalFillLiquidityCap(unittest.TestCase):
    def setUp(self):
        os.environ["LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS"] = "1"

    def _run(self, config):
        events = queue.Queue()
        handler = SimulatedExecutionHandler(events, MockBars(), config)
        handler.active_orders.append(
            {
                "order_id": "T-1",
                "symbol": "BTC/USDT",
                "type": "STOP",
                "quantity": 1000.0,
                "direction": "SELL",
                "status": "PENDING",
                "stop_price": 100.0,
                "position_side": "LONG",
                "reduce_only": True,
                "client_order_id": "cid-1",
            }
        )
        handler.check_open_orders(Market())
        fills = []
        while not events.empty():
            fills.append(events.get())
        return handler, fills

    def test_default_off_fills_full_quantity(self):
        handler, fills = self._run(BaseConfig)
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].quantity, 1000.0)
        self.assertEqual(handler.active_orders, [])

    def test_flag_on_caps_fill_and_chases_remainder(self):
        handler, fills = self._run(CappedConfig)
        self.assertEqual(len(fills), 1)
        # Cap = max_bar_volume_ratio * bar_volume = 0.1 * 500 = 50.
        self.assertAlmostEqual(fills[0].quantity, 50.0)
        remainders = [o for o in handler.active_orders if o.get("order_id") == "T-1-R"]
        self.assertEqual(len(remainders), 1)
        self.assertEqual(remainders[0]["type"], "MKT")
        self.assertAlmostEqual(remainders[0]["quantity"], 950.0)
        self.assertTrue(remainders[0]["reduce_only"])

    def test_runtime_dotpath_resolution(self):
        class Rt:
            class execution:
                apply_liquidity_cap_to_conditional_fills = True

        class DotpathConfig(BaseConfig):
            _rt = Rt()

        flag = SimulatedExecutionHandler._execution_flag(
            DotpathConfig,
            "APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS",
            "apply_liquidity_cap_to_conditional_fills",
        )
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
