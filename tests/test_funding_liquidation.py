import queue
import unittest
from datetime import datetime, timedelta

from lumina_quant.events import FillEvent
from lumina_quant.portfolio import Portfolio


class MockBars:
    symbol_list = ["BTC/USDT"]

    def __init__(self, start_dt, price):
        self.current_dt = start_dt
        self.open = price
        self.high = price
        self.low = price
        self.close = price

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return self.current_dt

    def get_latest_bar_value(self, symbol, val_type):
        _ = symbol
        mapping = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }
        return mapping.get(val_type, self.close)

    def get_market_spec(self, symbol):
        _ = symbol
        return {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 5.0}


class FundingConfig:
    INITIAL_CAPITAL = 10000.0
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.1
    MAX_DAILY_LOSS_PCT = 0.99
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    DEFAULT_STOP_LOSS_PCT = 0.01
    FUNDING_INTERVAL_HOURS = 8
    FUNDING_RATE_PER_8H = 0.001
    LEVERAGE = 3
    MAINTENANCE_MARGIN_RATE = 0.005
    TAKER_FEE_RATE = 0.0004
    COMMISSION_RATE = 0.0004


class LiquidationConfig(FundingConfig):
    FUNDING_RATE_PER_8H = 0.0


class TestFundingAndLiquidation(unittest.TestCase):
    def test_funding_is_applied_on_interval(self):
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, FundingConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        # First update sets baseline funding timestamp, no payment yet.
        p.update_timeindex(None)
        cash_before = p.current_holdings["cash"]

        # Move forward one full funding interval.
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)
        self.assertNotEqual(p.current_holdings["cash"], cash_before)
        self.assertGreater(p.total_funding_paid, 0.0)

    def test_liquidation_event_emitted(self):
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, LiquidationConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        # Force severe adverse move below liquidation threshold.
        # Close still above expected liq range, but low breaches intrabar.
        bars.close = 80.0
        bars.low = 60.0
        bars.high = 102.0
        bars.current_dt += timedelta(hours=1)
        p.update_timeindex(None)

        self.assertFalse(events.empty())
        evt = events.get()
        self.assertIsInstance(evt, FillEvent)
        self.assertEqual(evt.status, "LIQUIDATED")
        self.assertEqual(evt.symbol, "BTC/USDT")


if __name__ == "__main__":
    unittest.main()
