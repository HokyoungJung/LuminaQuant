"""Audit-hardening backtest-path gates (fix/audit-hardening).

Covers the three config-gated controls added to portfolio_backtest.Portfolio:

  1. risk.enforce_order_risk_gate_in_backtest -> RiskManager.check_order backstop
  2. risk.attach_default_protective_stop      -> synthetic stop when signal has none
  3. execution.require_funding_coverage       -> raise instead of silent 0.0 funding

Each control has an OFF case (default — behavior unchanged) and an ON case.
Fixtures are tiny in-memory synthetics; no real backtest is run.
"""

import os
import queue
import unittest
from datetime import datetime, timedelta

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import OrderEvent, SignalEvent


class MockBars:
    symbol_list = ["BTC/USDT"]

    def __init__(self, start_dt, price, *, funding_rate=None, has_feature=True):
        self.current_dt = start_dt
        self.close = price
        self.funding_rate = funding_rate
        self.has_feature = has_feature

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return self.current_dt

    def get_latest_bar_value(self, symbol, val_type):
        _ = symbol
        if val_type == "funding_rate":
            return None
        return self.close

    def get_market_spec(self, symbol):
        _ = symbol
        return {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 5.0}

    def get_latest_feature_value(self, symbol, field):
        _ = symbol
        if not self.has_feature:
            return None
        if field == "funding_rate":
            return self.funding_rate
        return None


class BaseConfig:
    INITIAL_CAPITAL = 10000.0
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.1
    TARGET_ALLOCATION_MODE = "legacy_notional_cap"
    MAX_DAILY_LOSS_PCT = 0.99
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    MAX_ORDER_NOTIONAL_PCT = 0.0
    DEFAULT_STOP_LOSS_PCT = 0.01
    FUNDING_INTERVAL_HOURS = 8
    FUNDING_RATE_PER_8H = 0.0
    LEVERAGE = 3
    MAINTENANCE_MARGIN_RATE = 0.005
    TAKER_FEE_RATE = 0.0004
    COMMISSION_RATE = 0.0004


def _signal(signal_type, *, stop_loss=None):
    return SignalEvent(
        strategy_id="t",
        symbol="BTC/USDT",
        datetime=datetime(2026, 1, 1, 0, 0),
        signal_type=signal_type,
        stop_loss=stop_loss,
    )


# --------------------------------------------------------------------------- #
# Fix 1: order-time risk gate
# --------------------------------------------------------------------------- #
class TestOrderRiskGate(unittest.TestCase):
    @staticmethod
    def _over_cap_order():
        # Notional 100 * 100 = 10_000 > MAX_ORDER_VALUE default 5_000.
        return OrderEvent(
            symbol="BTC/USDT",
            order_type="MKT",
            quantity=100.0,
            direction="BUY",
            position_side="LONG",
        )

    def test_default_off_over_cap_order_is_emitted(self):
        # With the gate OFF an over-cap order is NOT screened: update_signal emits
        # whatever the sizer produced (legacy behavior — no backstop in backtest).
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, BaseConfig)
        self.assertFalse(p.enforce_order_risk_gate)
        p.update_signal(_signal("LONG"))
        self.assertEqual(events.qsize(), 1)
        # Direct gate is never consulted when the flag is OFF (no RiskManager built).
        self.assertIsNone(p._risk_manager)

    def test_flag_on_over_cap_order_is_rejected(self):
        class Cfg(BaseConfig):
            ENFORCE_ORDER_RISK_GATE_IN_BACKTEST = True

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        self.assertTrue(p.enforce_order_risk_gate)
        # The gate rejects an over-cap order (notional 10_000 > MAX_ORDER_VALUE 5_000).
        self.assertFalse(p._passes_order_risk_gate(self._over_cap_order()))

    def test_flag_on_within_cap_order_passes_and_is_emitted(self):
        class Cfg(BaseConfig):
            ENFORCE_ORDER_RISK_GATE_IN_BACKTEST = True

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        p.update_signal(_signal("LONG"))  # sized within caps -> passes the gate
        self.assertEqual(events.qsize(), 1)


# --------------------------------------------------------------------------- #
# Fix 2: default protective stop
# --------------------------------------------------------------------------- #
class TestDefaultProtectiveStop(unittest.TestCase):
    def test_default_off_naked_long_has_no_stop(self):
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, BaseConfig)
        self.assertFalse(p.attach_default_protective_stop)
        order = p.generate_order_from_signal(_signal("LONG"))
        self.assertIsNotNone(order)
        self.assertIsNone(order.stop_loss)

    def test_flag_on_long_gets_default_stop_below_entry(self):
        class Cfg(BaseConfig):
            ATTACH_DEFAULT_PROTECTIVE_STOP = True

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        order = p.generate_order_from_signal(_signal("LONG"))
        self.assertIsNotNone(order)
        # entry 100 * (1 - 0.01) = 99.0
        self.assertAlmostEqual(order.stop_loss, 99.0, places=6)

    def test_flag_on_short_gets_default_stop_above_entry(self):
        class Cfg(BaseConfig):
            ATTACH_DEFAULT_PROTECTIVE_STOP = True

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        order = p.generate_order_from_signal(_signal("SHORT"))
        self.assertIsNotNone(order)
        # entry 100 * (1 + 0.01) = 101.0
        self.assertAlmostEqual(order.stop_loss, 101.0, places=6)

    def test_flag_on_respects_explicit_signal_stop(self):
        class Cfg(BaseConfig):
            ATTACH_DEFAULT_PROTECTIVE_STOP = True

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        order = p.generate_order_from_signal(_signal("LONG", stop_loss=95.0))
        self.assertIsNotNone(order)
        self.assertEqual(order.stop_loss, 95.0)  # explicit stop wins over synthetic


# --------------------------------------------------------------------------- #
# Fix 3: funding coverage enforcement
# --------------------------------------------------------------------------- #
class TestFundingCoverage(unittest.TestCase):
    def _run_funding(self, cfg):
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=None, has_feature=False)
        p = Portfolio(bars, events, bars.current_dt, cfg)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0
        p.update_timeindex(None)  # anchor funding ts
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)  # would charge funding here
        return p

    def test_default_off_silently_charges_zero(self):
        # Default OFF: no funding data + zero default => silent 0.0 (legacy).
        p = self._run_funding(BaseConfig)
        self.assertFalse(p.require_funding_coverage)
        self.assertEqual(p.total_funding_paid, 0.0)

    def test_flag_on_leveraged_no_data_raises(self):
        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True

        with self.assertRaises(ValueError) as ctx:
            self._run_funding(Cfg)
        self.assertIn("BTC/USDT", str(ctx.exception))
        self.assertIn("require_funding_coverage", str(ctx.exception))

    def test_flag_on_unleveraged_does_not_raise(self):
        # Coverage required but leverage <= 1 => no funding exposure, no raise.
        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True
            LEVERAGE = 1

        p = self._run_funding(Cfg)
        self.assertEqual(p.total_funding_paid, 0.0)

    def test_flag_on_with_feature_data_does_not_raise(self):
        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=0.001, has_feature=True)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0
        p.update_timeindex(None)
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)  # feature funding present -> charged, no raise
        self.assertGreater(p.total_funding_paid, 0.0)

    def test_flag_on_honors_configured_nonzero_static_default(self):
        # Regression: a legitimately configured non-zero funding_rate_per_8h is
        # usable coverage. With no per-bar data the gate must honor the static
        # default and charge funding, NOT raise (round-1 raised before the static
        # fallback). leverage > 1 so funding exposure exists.
        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True
            FUNDING_RATE_PER_8H = 0.0001

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=None, has_feature=False)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        # Direct call: configured static default honored, no raise.
        rate = p._resolve_funding_rate("BTC/USDT", default=0.0001)
        self.assertAlmostEqual(rate, 0.0001, places=9)
        # End-to-end: funding is actually charged off the static default.
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0
        p.update_timeindex(None)
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)
        self.assertGreater(p.total_funding_paid, 0.0)

    def test_flag_on_real_zero_funding_bar_does_not_raise(self):
        # Regression: a genuine 0.0 per-bar funding rate is real data, NOT
        # "absent". round-1 treated bar 0.0 as missing (fallback not in (None, 0.0))
        # and raised on a leveraged run. A real 0.0 bar must skip silently (no raise,
        # no charge) even with no static default.
        class ZeroBars(MockBars):
            def get_latest_bar_value(self, symbol, val_type):
                _ = symbol
                if val_type == "funding_rate":
                    return 0.0  # genuine zero funding for this bar
                return self.close

        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True  # FUNDING_RATE_PER_8H stays 0.0

        events = queue.Queue()
        bars = ZeroBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=None, has_feature=False)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        self.assertEqual(p._resolve_funding_rate("BTC/USDT", default=0.0), 0.0)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0
        p.update_timeindex(None)
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)  # real 0.0 funding -> no raise, no charge
        self.assertEqual(p.total_funding_paid, 0.0)

    def test_flag_on_truly_no_data_no_default_raises(self):
        # The raise survives only for the genuine failure: leveraged, require
        # coverage, NO per-bar data (feature + bar both None) AND no usable
        # non-zero static default.
        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True  # FUNDING_RATE_PER_8H == 0.0

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=None, has_feature=False)
        p = Portfolio(bars, events, bars.current_dt, Cfg)
        with self.assertRaises(ValueError) as ctx:
            p._resolve_funding_rate("BTC/USDT", default=0.0)
        self.assertIn("BTC/USDT", str(ctx.exception))
        self.assertIn("require_funding_coverage", str(ctx.exception))


# --------------------------------------------------------------------------- #
# Fix: tz-correct epoch helpers (naive datetimes are UTC, not host-local)
# --------------------------------------------------------------------------- #
class TestTimestampHelpersUTC(unittest.TestCase):
    """_to_timestamp_ms / _to_unix_seconds must treat naive datetimes as UTC so
    equity-sampling cadence (_should_sample) is host-tz independent. Regression
    for the host-local datetime.timestamp() skew.
    """

    @staticmethod
    def _portfolio():
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        return Portfolio(bars, events, bars.current_dt, BaseConfig)

    def test_to_timestamp_ms_naive_is_utc(self):
        p = self._portfolio()
        # 2026-01-01T00:00:00Z == 1767225600000 ms, regardless of host tz.
        self.assertEqual(p._to_timestamp_ms(datetime(2026, 1, 1, 0, 0)), 1767225600000)

    def test_to_unix_seconds_naive_is_utc(self):
        p = self._portfolio()
        self.assertEqual(p._to_unix_seconds(datetime(2026, 1, 1, 0, 0)), 1767225600.0)

    def test_naive_equals_explicit_utc(self):
        # A naive datetime and the same instant tagged UTC must map identically.
        from datetime import UTC as _UTC

        p = self._portfolio()
        naive = datetime(2026, 6, 18, 12, 30, 15)
        aware = naive.replace(tzinfo=_UTC)
        self.assertEqual(p._to_timestamp_ms(naive), p._to_timestamp_ms(aware))
        self.assertEqual(p._to_unix_seconds(naive), p._to_unix_seconds(aware))

    def test_iso_string_naive_is_utc(self):
        p = self._portfolio()
        self.assertEqual(p._to_timestamp_ms("2026-01-01T00:00:00"), 1767225600000)
        self.assertEqual(p._to_timestamp_ms("2026-01-01T00:00:00Z"), 1767225600000)

    def test_to_timestamp_ms_independent_of_host_tz(self):
        # Monkeypatch the process tz to a non-UTC offset and confirm the naive
        # datetime still maps to the UTC-based epoch (host-local .timestamp()
        # would have shifted it by the host offset).
        import time

        p = self._portfolio()
        expected = 1767225600000
        original_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"  # UTC-5 / UTC-4
            if hasattr(time, "tzset"):
                time.tzset()
            self.assertEqual(p._to_timestamp_ms(datetime(2026, 1, 1, 0, 0)), expected)
            self.assertEqual(p._to_unix_seconds(datetime(2026, 1, 1, 0, 0)), 1767225600.0)
        finally:
            if original_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_tz
            if hasattr(time, "tzset"):
                time.tzset()

    def test_should_sample_cadence_host_tz_independent(self):
        # _should_sample uses _to_timestamp_ms; with a sampling interval the
        # alignment (ts_ms % interval) must be computed against UTC epochs so the
        # sampled set is identical regardless of host tz.
        import time

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, BaseConfig, sampling_timeframe="1h")
        sample_at = datetime(2026, 1, 1, 1, 0)  # exactly on a 1h boundary in UTC
        original_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "Asia/Kolkata"  # UTC+5:30 (off the hour grid)
            if hasattr(time, "tzset"):
                time.tzset()
            p._last_sample_timestamp_ms = None
            self.assertTrue(p._should_sample(sample_at))
        finally:
            if original_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_tz
            if hasattr(time, "tzset"):
                time.tzset()


if __name__ == "__main__":
    unittest.main()
