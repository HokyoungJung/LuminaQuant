import queue
import unittest
from datetime import datetime, timedelta

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import FillEvent, MarketWindowEvent


class MockBars:
    symbol_list = ["BTC/USDT"]

    def __init__(self, start_dt, price, *, funding_rate=None):
        self.current_dt = start_dt
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.funding_rate = funding_rate

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

    def get_latest_feature_value(self, symbol, field):
        _ = symbol
        if field == "funding_rate":
            return self.funding_rate
        return None


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


class DynamicFundingConfig(FundingConfig):
    FUNDING_RATE_PER_8H = 0.0


class BoundaryFundingConfig(FundingConfig):
    # Report defect #8: charge funding on crossed 00/08/16 UTC boundaries.
    FUNDING_ON_UTC_BOUNDARY = True
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

    def test_dynamic_funding_feature_rate_is_used_when_config_default_is_zero(self):
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=0.001)
        p = Portfolio(bars, events, bars.current_dt, DynamicFundingConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        p.update_timeindex(None)
        cash_before = p.current_holdings["cash"]

        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)
        self.assertLess(p.current_holdings["cash"], cash_before)
        self.assertGreater(p.total_funding_paid, 0.0)

    def test_no_funding_charged_across_flat_gap_after_reopen(self):
        """Regression: closing a position resets the funding anchor.

        open -> close -> flat gap (10 days) -> reopen must NOT back-charge
        funding for the flat interval. Before the fix, _last_funding_ts kept
        its pre-close value, so the first _apply_funding after the reopen
        charged ~30 intervals of phantom funding.
        """
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, FundingConfig)

        # Open a long; first update anchors the funding timestamp.
        p.update_positions_from_fill(
            FillEvent(
                timeindex=bars.current_dt,
                symbol="BTC/USDT",
                exchange="TEST",
                quantity=1.0,
                direction="BUY",
                fill_cost=100.0,
                commission=0.0,
            )
        )
        p.update_timeindex(None)
        self.assertIsNotNone(p._last_funding_ts["BTC/USDT"])

        # Close the position flat -> anchor must be cleared.
        p.update_positions_from_fill(
            FillEvent(
                timeindex=bars.current_dt,
                symbol="BTC/USDT",
                exchange="TEST",
                quantity=1.0,
                direction="SELL",
                fill_cost=100.0,
                commission=0.0,
            )
        )
        self.assertEqual(p.current_positions["BTC/USDT"], 0.0)
        self.assertIsNone(p._last_funding_ts["BTC/USDT"])

        # Stay flat for a long gap (10 days = 30 funding intervals).
        bars.current_dt += timedelta(days=10)
        p.update_timeindex(None)
        funding_before_reopen = p.total_funding_paid

        # Reopen the long; the first bar re-anchors (no payment yet).
        p.update_positions_from_fill(
            FillEvent(
                timeindex=bars.current_dt,
                symbol="BTC/USDT",
                exchange="TEST",
                quantity=1.0,
                direction="BUY",
                fill_cost=100.0,
                commission=0.0,
            )
        )
        p.update_timeindex(None)

        # No funding may have been booked for the flat gap.
        self.assertEqual(p.total_funding_paid, funding_before_reopen)

        # One genuine interval after the reopen DOES charge funding.
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)
        self.assertGreater(p.total_funding_paid, funding_before_reopen)

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

    def test_windowed_liquidation_uses_full_window_extremes(self):
        """Regression: a liquidation breach in any 1s bar of a window triggers.

        The windowed handler advances get_latest_bar_value to only the LAST 1s
        bar. Before the fix, an intra-window maintenance-margin breach (a dip in
        an earlier second) was silently skipped. _check_liquidations now reads
        the window's full low/high extremes from the event's bars_1s rows.
        """
        events = queue.Queue()
        # Latest 1s bar stays benign (low 99.5) so single-bar checks would miss.
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        bars.low = 99.5
        bars.high = 101.0
        bars.close = 100.0
        p = Portfolio(bars, events, bars.current_dt, LiquidationConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        ts = int(bars.current_dt.timestamp() * 1000)
        # Middle 1s bar dips to 60.0 — breaches the long liquidation level.
        bars_1s = {
            "BTC/USDT": [
                (ts, 100.0, 101.0, 99.5, 100.0, 10.0),
                (ts + 1000, 100.0, 102.0, 60.0, 80.0, 10.0),
                (ts + 2000, 100.0, 101.0, 99.5, 100.0, 10.0),
            ]
        }
        evt = MarketWindowEvent(time=bars.current_dt, window_seconds=20, bars_1s=bars_1s)
        p.update_timeindex(evt)

        self.assertFalse(events.empty())
        fill = events.get()
        self.assertEqual(fill.status, "LIQUIDATED")
        self.assertEqual(fill.symbol, "BTC/USDT")


class TestFundingUtcBoundary(unittest.TestCase):
    """Report defect #8 — execution.funding_on_utc_boundary.

    OFF (default) charges funding on the entry-anchored 8h clock (byte-identical
    to legacy). ON charges funding on crossed wall-clock 00/08/16 UTC boundaries,
    so a sub-8h hold that straddles a boundary now pays one funding event.
    """

    def test_flag_defaults_off_byte_identical_path(self):
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 7, 0), 100.0, funding_rate=0.001)
        p = Portfolio(bars, events, bars.current_dt, DynamicFundingConfig)
        self.assertFalse(p.funding_on_utc_boundary)

    def test_flag_off_sub_interval_straddle_charges_nothing(self):
        """Legacy entry-anchored clock: a 2h hold across 08:00 UTC pays no funding."""
        events = queue.Queue()
        # 07:00 UTC entry, advance 2h to 09:00 UTC -> straddles the 08:00 boundary.
        bars = MockBars(datetime(2026, 1, 1, 7, 0), 100.0, funding_rate=0.001)
        p = Portfolio(bars, events, bars.current_dt, DynamicFundingConfig)
        self.assertFalse(p.funding_on_utc_boundary)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        p.update_timeindex(None)  # anchors funding ts at 07:00
        bars.current_dt += timedelta(hours=2)  # 09:00 -> only 2h elapsed (< 8h)
        p.update_timeindex(None)

        self.assertEqual(p.total_funding_paid, 0.0)

    def test_flag_on_sub_interval_straddle_charges_one_event(self):
        """Flag ON: the same 2h hold crossing the 08:00 UTC boundary pays one event."""
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 7, 0), 100.0, funding_rate=0.001)
        p = Portfolio(bars, events, bars.current_dt, BoundaryFundingConfig)
        self.assertTrue(p.funding_on_utc_boundary)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        p.update_timeindex(None)  # anchors at 07:00
        bars.current_dt += timedelta(hours=2)  # 09:00 -> crosses the 08:00 boundary
        p.update_timeindex(None)

        # notional 100 * rate 0.001 * 1 period = 0.1 charged to a long.
        self.assertAlmostEqual(p.total_funding_paid, 0.1, places=9)

    def test_flag_on_within_single_bucket_charges_nothing(self):
        """Flag ON but no boundary crossed (01:00 -> 05:00 inside [00:00,08:00))."""
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 1, 0), 100.0, funding_rate=0.001)
        p = Portfolio(bars, events, bars.current_dt, BoundaryFundingConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        p.update_timeindex(None)
        bars.current_dt += timedelta(hours=4)  # 05:00, same funding bucket
        p.update_timeindex(None)

        self.assertEqual(p.total_funding_paid, 0.0)

    def test_flag_on_multiple_boundaries_charge_each(self):
        """Flag ON: 07:00 -> 17:00 crosses 08:00 and 16:00 -> two funding events."""
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 7, 0), 100.0, funding_rate=0.001)
        p = Portfolio(bars, events, bars.current_dt, BoundaryFundingConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0

        p.update_timeindex(None)
        bars.current_dt += timedelta(hours=10)  # 17:00 -> two boundaries crossed
        p.update_timeindex(None)

        self.assertAlmostEqual(p.total_funding_paid, 0.2, places=9)

    def test_flag_on_matches_legacy_when_boundary_aligned(self):
        """With a boundary-aligned entry advancing in whole 8h intervals, the
        boundary clock and the entry-anchored clock agree exactly."""

        def run(cfg):
            events = queue.Queue()
            bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=0.001)
            p = Portfolio(bars, events, bars.current_dt, cfg)
            p.current_positions["BTC/USDT"] = 1.0
            p.entry_prices["BTC/USDT"] = 100.0
            p.update_timeindex(None)  # anchors at the 00:00 boundary
            for _ in range(3):
                bars.current_dt += timedelta(hours=8)
                p.update_timeindex(None)
            return p.total_funding_paid

        self.assertAlmostEqual(run(DynamicFundingConfig), run(BoundaryFundingConfig), places=12)


if __name__ == "__main__":
    unittest.main()
