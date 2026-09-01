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

    def test_flag_on_zero_volume_bar_emits_no_fill_and_chases_full_remainder(self):
        # N5: a triggered STOP on a zero-volume bar (executed_qty == 0 under the
        # conditional liquidity cap) must NOT emit a quantity-0 FILLED fill nor
        # dismantle protection — only the remainder MKT chase survives.
        events = queue.Queue()
        handler = SimulatedExecutionHandler(events, MockBars(), CappedConfig)
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
                "is_protective": True,
                "oco_group": "G1",
            }
        )
        # Sibling protective take-profit in the same OCO group; it must survive
        # because nothing actually filled on this bar.
        handler.active_orders.append(
            {
                "order_id": "T-2",
                "symbol": "BTC/USDT",
                "type": "TAKE_PROFIT",
                "quantity": 1000.0,
                "direction": "SELL",
                "status": "PENDING",
                "stop_price": 200.0,  # bar_high=101 → does not trigger
                "position_side": "LONG",
                "reduce_only": True,
                "client_order_id": "cid-2",
                "is_protective": True,
                "oco_group": "G1",
            }
        )
        mkt = Market()
        mkt.volume = 0.0
        handler.check_open_orders(mkt)
        fills = []
        while not events.empty():
            fills.append(events.get())

        # No zero-qty fill emitted.
        self.assertEqual(fills, [])
        ids = {o.get("order_id") for o in handler.active_orders}
        # Spent STOP dropped; full remainder chase queued as reduce-only MKT.
        self.assertNotIn("T-1", ids)
        remainders = [o for o in handler.active_orders if o.get("order_id") == "T-1-R"]
        self.assertEqual(len(remainders), 1)
        self.assertEqual(remainders[0]["type"], "MKT")
        self.assertAlmostEqual(remainders[0]["quantity"], 1000.0)
        self.assertTrue(remainders[0]["reduce_only"])
        # Sibling protective TP was NOT torn down (OCO group not closed).
        self.assertIn("T-2", ids)

    def test_mkt_zero_volume_bar_emits_no_fill_and_keeps_remainder(self):
        # N5: the MKT liquidity cap is always on. A zero-volume bar yields
        # executed_qty == 0 and must not emit a quantity-0 fill even on the
        # default profile — only the remainder chase is kept.
        events = queue.Queue()
        handler = SimulatedExecutionHandler(events, MockBars(), BaseConfig)
        handler.active_orders.append(
            {
                "order_id": "M-1",
                "symbol": "BTC/USDT",
                "type": "MKT",
                "quantity": 1000.0,
                "direction": "BUY",
                "status": "PENDING",
                "position_side": "LONG",
                "reduce_only": False,
                "client_order_id": "cid-m1",
                "stop_loss": None,
                "take_profit": None,
                "trailing_percent": None,
            }
        )
        mkt = Market()
        mkt.volume = 0.0
        handler.check_open_orders(mkt)
        fills = []
        while not events.empty():
            fills.append(events.get())

        self.assertEqual(fills, [])
        ids = {o.get("order_id") for o in handler.active_orders}
        self.assertNotIn("M-1", ids)
        remainders = [o for o in handler.active_orders if o.get("order_id") == "M-1-R"]
        self.assertEqual(len(remainders), 1)
        self.assertEqual(remainders[0]["type"], "MKT")
        self.assertAlmostEqual(remainders[0]["quantity"], 1000.0)

    def test_mkt_positive_volume_bar_still_fills(self):
        # Guard must not over-fire: with volume present the MKT order fills.
        events = queue.Queue()
        handler = SimulatedExecutionHandler(events, MockBars(), BaseConfig)
        handler.active_orders.append(
            {
                "order_id": "M-2",
                "symbol": "BTC/USDT",
                "type": "MKT",
                "quantity": 10.0,  # < cap (0.1 * 500 = 50) → fills fully
                "direction": "BUY",
                "status": "PENDING",
                "position_side": "LONG",
                "reduce_only": False,
                "client_order_id": "cid-m2",
                "stop_loss": None,
                "take_profit": None,
                "trailing_percent": None,
            }
        )
        handler.check_open_orders(Market())  # volume=500.0
        fills = []
        while not events.empty():
            fills.append(events.get())
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].quantity, 10.0)

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
