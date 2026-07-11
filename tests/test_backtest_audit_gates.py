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


# --------------------------------------------------------------------------- #
# Fix (2026-07-03 audit): 0.0 sentinel from get_latest_bar_value must not be
# conflated with genuine funding data. Mimics HistoricCSVDataHandler exactly:
# col_idx WITHOUT funding_rate + get_latest_bar_value returning the 0.0
# sentinel for unknown fields.
# --------------------------------------------------------------------------- #
class SentinelBars(MockBars):
    """Real-handler-shaped mock: OHLCV col_idx, 0.0 sentinel for absent fields."""

    def __init__(self, start_dt, price, *, with_funding_column=False, column_rate=0.0):
        super().__init__(start_dt, price, funding_rate=None, has_feature=False)
        self.col_idx = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}
        if with_funding_column:
            self.col_idx["funding_rate"] = 6
        self._column_rate = column_rate

    def get_latest_bar_value(self, symbol, val_type):
        _ = symbol
        if val_type == "funding_rate":
            if "funding_rate" in self.col_idx:
                return self._column_rate
            return 0.0  # the sentinel HistoricCSVDataHandler returns when absent
        return self.close


class TestFundingSentinelFix(unittest.TestCase):
    def _run(self, cfg, bars):
        events = queue.Queue()
        p = Portfolio(bars, events, bars.current_dt, cfg)
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0
        p.update_timeindex(None)
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)
        return p

    def test_static_default_charged_despite_sentinel(self):
        # THE bug: the 0.0 sentinel used to shadow the configured static
        # default, so funding was never charged on the real CSV/parquet path.
        class Cfg(BaseConfig):
            FUNDING_RATE_PER_8H = 1e-4

        bars = SentinelBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = self._run(Cfg, bars)
        self.assertAlmostEqual(p.total_funding_paid, 1e-4 * 100.0, places=10)

    def test_coverage_gate_fires_despite_sentinel(self):
        # THE bug, gate variant: require_funding_coverage could never raise on
        # the real handler because the sentinel masqueraded as coverage.
        class Cfg(BaseConfig):
            REQUIRE_FUNDING_COVERAGE = True

        bars = SentinelBars(datetime(2026, 1, 1, 0, 0), 100.0)
        with self.assertRaises(ValueError) as ctx:
            self._run(Cfg, bars)
        self.assertIn("require_funding_coverage", str(ctx.exception))

    def test_genuine_funding_column_still_trusted(self):
        # A handler that actually carries a funding_rate column keeps working,
        # including a GENUINE 0.0 rate (which must not fall through to the
        # static default).
        class Cfg(BaseConfig):
            FUNDING_RATE_PER_8H = 1e-4

        bars = SentinelBars(
            datetime(2026, 1, 1, 0, 0), 100.0, with_funding_column=True, column_rate=0.0005
        )
        p = self._run(Cfg, bars)
        self.assertAlmostEqual(p.total_funding_paid, 0.0005 * 100.0, places=10)

        bars_zero = SentinelBars(
            datetime(2026, 1, 1, 0, 0), 100.0, with_funding_column=True, column_rate=0.0
        )
        p_zero = self._run(Cfg, bars_zero)
        self.assertEqual(p_zero.total_funding_paid, 0.0)

    def test_default_config_unchanged(self):
        # Golden safety: default config (rate 0.0, gate off) stays at 0.0
        # funding on the sentinel path — byte-identical behavior.
        bars = SentinelBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = self._run(BaseConfig, bars)
        self.assertEqual(p.total_funding_paid, 0.0)


# --------------------------------------------------------------------------- #
# Fix (2026-07-03 audit): execution.enforce_reduce_only clamps reduce-only
# fills to live-exchange semantics (reduce toward zero, never through it).
# --------------------------------------------------------------------------- #
class TestEnforceReduceOnly(unittest.TestCase):
    def _portfolio(self, cfg):
        from lumina_quant.core.events import FillEvent

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        p = Portfolio(bars, events, bars.current_dt, cfg)
        return p, FillEvent

    def _fill(self, FillEvent, *, qty, direction, reduce_only=True):
        return FillEvent(
            timeindex=datetime(2026, 1, 1, 0, 0),
            symbol="BTC/USDT",
            exchange="SIM",
            quantity=qty,
            direction=direction,
            fill_cost=qty * 100.0,
            commission=qty * 0.04,
            metadata={"reduce_only": reduce_only},
        )

    def test_default_off_preserves_legacy_phantom_flip(self):
        # Golden safety: default OFF keeps the legacy behavior byte-identical —
        # an oversized reduce-only exit still flips the book (the bug).
        p, FillEvent = self._portfolio(BaseConfig)
        p.current_positions["BTC/USDT"] = 1.0
        p.update_fill(self._fill(FillEvent, qty=2.0, direction="SELL"))
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], -1.0)

    def test_flag_on_clamps_oversized_exit_to_flat(self):
        class Cfg(BaseConfig):
            ENFORCE_REDUCE_ONLY = True

        p, FillEvent = self._portfolio(Cfg)
        p.current_positions["BTC/USDT"] = 1.0
        before_cash = p.current_holdings["cash"]
        p.update_fill(self._fill(FillEvent, qty=2.0, direction="SELL"))
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], 0.0)
        # Only the reducible 1.0 traded: cash moved by ~1.0*100 minus scaled fee.
        self.assertAlmostEqual(p.current_holdings["cash"] - before_cash, 100.0 - 0.04, places=8)

    def test_flag_on_skips_reduce_only_from_flat(self):
        class Cfg(BaseConfig):
            ENFORCE_REDUCE_ONLY = True

        p, FillEvent = self._portfolio(Cfg)
        trades_before = p.trade_count
        p.update_fill(self._fill(FillEvent, qty=1.0, direction="SELL"))
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], 0.0)
        self.assertEqual(p.trade_count, trades_before)  # true no-op

    def test_flag_on_skips_same_side_reduce_only(self):
        class Cfg(BaseConfig):
            ENFORCE_REDUCE_ONLY = True

        p, FillEvent = self._portfolio(Cfg)
        p.current_positions["BTC/USDT"] = 1.0
        p.update_fill(self._fill(FillEvent, qty=1.0, direction="BUY"))
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], 1.0)

    def test_flag_on_partial_reduction_passes_through(self):
        class Cfg(BaseConfig):
            ENFORCE_REDUCE_ONLY = True

        p, FillEvent = self._portfolio(Cfg)
        p.current_positions["BTC/USDT"] = 1.0
        p.update_fill(self._fill(FillEvent, qty=0.4, direction="SELL"))
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], 0.6)

    def test_flag_on_non_reduce_only_fills_untouched(self):
        class Cfg(BaseConfig):
            ENFORCE_REDUCE_ONLY = True

        p, FillEvent = self._portfolio(Cfg)
        p.update_fill(self._fill(FillEvent, qty=2.0, direction="SELL", reduce_only=False))
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], -2.0)


# --------------------------------------------------------------------------- #
# Fix M5: the simulated liquidation engine + simulated funding charge must be
# gated off the live path (self._live_liquidation_disabled). On the live path a
# modeled maintenance-margin breach is downgraded to a WARNING / audit record
# and never fabricates a synthetic LIQUIDATED fill; real liquidations/funding
# arrive as exchange events. Default (backtest) is byte-identical.
# --------------------------------------------------------------------------- #
class LiqBars(MockBars):
    """Bars mock with independently settable high/low/close for a liq breach."""

    def __init__(self, start_dt, close, *, high=None, low=None):
        super().__init__(start_dt, close)
        self._high = high if high is not None else close
        self._low = low if low is not None else close

    def get_latest_bar_value(self, symbol, val_type):
        _ = symbol
        if val_type == "funding_rate":
            return None
        if val_type == "high":
            return self._high
        if val_type == "low":
            return self._low
        return self.close


class _LiqConfig(BaseConfig):
    # Long entry 100, leverage 3 => liq_price ~ 67.2; a bar dipping to 60 breaches.
    MARGIN_MODE = "isolated"


class TestLiveLiquidationGate(unittest.TestCase):
    @staticmethod
    def _long_position(portfolio, *, entry=100.0):
        portfolio.current_positions["BTC/USDT"] = 1.0
        portfolio.entry_prices["BTC/USDT"] = entry

    def test_backtest_path_still_enqueues_synthetic_liquidation_fill(self):
        # Golden safety: default (backtest) path is unchanged — a modeled breach
        # still enqueues a LIQUIDATED fill on the SIM_LIQUIDATION exchange.
        events = queue.Queue()
        bars = LiqBars(datetime(2026, 1, 1, 0, 0), 60.0, high=60.0, low=60.0)
        p = Portfolio(bars, events, bars.current_dt, _LiqConfig)
        self.assertFalse(p._live_liquidation_disabled)
        self._long_position(p)
        p._check_liquidations(bars.current_dt, None)

        self.assertEqual(events.qsize(), 1)
        fill = events.get_nowait()
        self.assertEqual(fill.status, "LIQUIDATED")
        self.assertEqual(fill.exchange, "SIM_LIQUIDATION")
        self.assertEqual(len(p.liquidation_events), 1)
        self.assertAlmostEqual(p.liquidation_events[0]["fill_cost"], fill.fill_cost)
        self.assertAlmostEqual(p.liquidation_events[0]["commission"], fill.commission)
        self.assertEqual(p._modeled_liquidation_warnings, [])
        self.assertIn("BTC/USDT", p._pending_liquidation)

    def test_live_path_downgrades_breach_to_audit_record_no_fill(self):
        events = queue.Queue()
        bars = LiqBars(datetime(2026, 1, 1, 0, 0), 60.0, high=60.0, low=60.0)
        p = Portfolio(bars, events, bars.current_dt, _LiqConfig)
        p._live_liquidation_disabled = True  # emulate the live binding
        self._long_position(p)
        with self.assertLogs("lumina_quant.backtesting.portfolio_backtest", level="WARNING"):
            p._check_liquidations(bars.current_dt, None)

        # No fabricated fill, no real liquidation record, position untouched.
        self.assertEqual(events.qsize(), 0)
        self.assertEqual(p.liquidation_events, [])
        self.assertAlmostEqual(p.current_positions["BTC/USDT"], 1.0)
        # Modeled breach captured for audit/alerting, flagged modeled_only.
        self.assertEqual(len(p._modeled_liquidation_warnings), 1)
        rec = p._modeled_liquidation_warnings[0]
        self.assertTrue(rec["modeled_only"])
        self.assertEqual(rec["symbol"], "BTC/USDT")
        # De-spam: symbol marked pending so it is not re-warned every bar.
        self.assertIn("BTC/USDT", p._pending_liquidation)

    def test_live_path_does_not_re_warn_while_pending(self):
        events = queue.Queue()
        bars = LiqBars(datetime(2026, 1, 1, 0, 0), 60.0, high=60.0, low=60.0)
        p = Portfolio(bars, events, bars.current_dt, _LiqConfig)
        p._live_liquidation_disabled = True
        self._long_position(p)
        p._check_liquidations(bars.current_dt, None)
        p._check_liquidations(bars.current_dt, None)  # still pending -> skipped
        self.assertEqual(len(p._modeled_liquidation_warnings), 1)

    def test_live_portfolio_subclass_sets_flag(self):
        from lumina_quant.live.portfolio import get_live_portfolio_cls

        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        live_cls = get_live_portfolio_cls()
        live = live_cls(bars, events, bars.current_dt, BaseConfig)
        self.assertTrue(live._live_liquidation_disabled)
        # A plain backtest Portfolio keeps the flag OFF.
        base = Portfolio(bars, events, bars.current_dt, BaseConfig)
        self.assertFalse(base._live_liquidation_disabled)

    def test_live_path_skips_simulated_funding_charge(self):
        # With the live gate on, _apply_funding must not simulate a charge even
        # when per-bar funding data is present (real funding is reconciled from
        # the exchange balance by the trader).
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=0.001, has_feature=True)
        p = Portfolio(bars, events, bars.current_dt, BaseConfig)
        p._live_liquidation_disabled = True
        p.current_positions["BTC/USDT"] = 1.0
        p.entry_prices["BTC/USDT"] = 100.0
        p.update_timeindex(None)  # anchor
        bars.current_dt += timedelta(hours=8)
        p.update_timeindex(None)  # would charge in backtest, gated off on live
        self.assertEqual(p.total_funding_paid, 0.0)

        # Sanity: the same setup on the backtest path DOES charge funding.
        events2 = queue.Queue()
        bars2 = MockBars(datetime(2026, 1, 1, 0, 0), 100.0, funding_rate=0.001, has_feature=True)
        p2 = Portfolio(bars2, events2, bars2.current_dt, BaseConfig)
        p2.current_positions["BTC/USDT"] = 1.0
        p2.entry_prices["BTC/USDT"] = 100.0
        p2.update_timeindex(None)
        bars2.current_dt += timedelta(hours=8)
        p2.update_timeindex(None)
        self.assertGreater(p2.total_funding_paid, 0.0)


# --------------------------------------------------------------------------- #
# Fix X5: the position-invariant liquidation price is cached per symbol keyed by
# (qty, entry_price) and invalidated on fill — bit-identical to recomputing.
# --------------------------------------------------------------------------- #
class TestLiquidationPriceCache(unittest.TestCase):
    @staticmethod
    def _portfolio():
        events = queue.Queue()
        bars = MockBars(datetime(2026, 1, 1, 0, 0), 100.0)
        return Portfolio(bars, events, bars.current_dt, BaseConfig)

    def test_cache_is_bit_identical_and_populated(self):
        p = self._portfolio()
        expected = p.execution_model.liquidation_price(qty=1.0, entry_price=100.0)
        got = p._cached_liquidation_price("BTC/USDT", 1.0, 100.0)
        self.assertEqual(got, expected)
        self.assertEqual(p._liq_price_cache["BTC/USDT"], (1.0, 100.0, expected))
        # Second call reuses the cached value (identical result, no divergence).
        self.assertEqual(p._cached_liquidation_price("BTC/USDT", 1.0, 100.0), expected)

    def test_cache_miss_on_changed_inputs(self):
        p = self._portfolio()
        long_liq = p._cached_liquidation_price("BTC/USDT", 1.0, 100.0)
        # Flipping to a short at a new entry recomputes (different key).
        short_liq = p._cached_liquidation_price("BTC/USDT", -1.0, 120.0)
        self.assertNotEqual(long_liq, short_liq)
        self.assertEqual(p._liq_price_cache["BTC/USDT"], (-1.0, 120.0, short_liq))

    def test_fill_invalidates_cache(self):
        from lumina_quant.core.events import FillEvent

        p = self._portfolio()
        p._cached_liquidation_price("BTC/USDT", 1.0, 100.0)
        self.assertIn("BTC/USDT", p._liq_price_cache)
        p.update_positions_from_fill(
            FillEvent(
                timeindex=datetime(2026, 1, 1, 0, 0),
                symbol="BTC/USDT",
                exchange="SIM",
                quantity=1.0,
                direction="BUY",
                fill_cost=100.0,
                commission=0.04,
            )
        )
        self.assertNotIn("BTC/USDT", p._liq_price_cache)
