"""Regression test: fast-stats value ranges for synthetic equity curves.

Locks the sign/direction contract of sharpe, cagr, and max_drawdown
returned by Portfolio.output_summary_stats_fast() — extends
test_portfolio_fast_stats.py which only checks key presence.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from lumina_quant.backtesting.portfolio_backtest import Portfolio


class MockBars:
    """Minimal bars adapter reused from test_portfolio_fast_stats.py."""

    symbol_list = ["BTC/USDT"]

    def __init__(self, closes):
        self._closes = list(closes)
        self._idx = 0
        self._start = datetime(2024, 1, 1)

    def advance(self):
        if self._idx < len(self._closes) - 1:
            self._idx += 1

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return self._start + timedelta(days=self._idx)

    def get_latest_bar_value(self, symbol, val_type):
        _ = symbol
        close = float(self._closes[self._idx])
        return close


class MockConfig:
    INITIAL_CAPITAL = 10_000.0
    ANNUAL_PERIODS = 252
    LEVERAGE = 1


def _build_portfolio_from_closes(closes):
    """Drive a no-trade Portfolio through every bar of *closes*.

    The portfolio holds no positions so equity == cash == INITIAL_CAPITAL
    throughout; what matters is the shape of the equity-curve that drives
    the fast-stats calculation.  We inject the equity directly by patching
    _metric_totals instead of wiring up actual positions/fills, which keeps
    the fixture minimal and deterministic.
    """
    bars = MockBars(closes)
    portfolio = Portfolio(
        bars,
        events=None,
        start_date=datetime(2024, 1, 1),
        config=MockConfig,
        record_history=False,
    )
    return portfolio


def _build_portfolio_with_totals(totals):
    """Construct a Portfolio and replace its _metric_totals with *totals*.

    This bypasses the bar-driven loop so we can shape the equity curve
    precisely for the assertion scenarios.
    """
    # Use a minimal bars fixture — the actual prices don't matter here.
    bars = MockBars([100.0] * len(totals))
    portfolio = Portfolio(
        bars,
        events=None,
        start_date=datetime(2024, 1, 1),
        config=MockConfig,
        record_history=False,
    )
    # Replace the accumulated metric totals with the synthetic series.
    portfolio._metric_totals = list(totals)
    return portfolio


class TestPortfolioFastStatsRanges(unittest.TestCase):
    """Value-range assertions for output_summary_stats_fast()."""

    # ------------------------------------------------------------------
    # Monotonically rising equity curve
    # ------------------------------------------------------------------

    def _rising_totals(self, n=60, start=10_000.0, step=100.0):
        """Generate a strictly increasing equity series."""
        return [start + i * step for i in range(n)]

    def test_rising_curve_sharpe_positive(self):
        """A purely rising equity curve must produce sharpe > 0."""
        portfolio = _build_portfolio_with_totals(self._rising_totals())
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        self.assertGreater(
            stats["sharpe"],
            0.0,
            msg=f"Expected sharpe > 0 for rising curve, got {stats['sharpe']}",
        )

    def test_rising_curve_cagr_positive(self):
        """A purely rising equity curve must produce cagr > 0."""
        portfolio = _build_portfolio_with_totals(self._rising_totals())
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        self.assertGreater(
            stats["cagr"],
            0.0,
            msg=f"Expected cagr > 0 for rising curve, got {stats['cagr']}",
        )

    def test_rising_curve_max_drawdown_near_zero(self):
        """A purely rising equity curve has no drawdown — max_drawdown magnitude must be ~0.

        The backend encodes max_drawdown as a positive magnitude (fraction of peak lost).
        For a strictly monotone rising series there is no peak-to-trough decline, so the
        magnitude must be approximately zero.
        """
        portfolio = _build_portfolio_with_totals(self._rising_totals())
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        self.assertAlmostEqual(
            stats["max_drawdown"],
            0.0,
            places=6,
            msg=f"Expected max_drawdown ~0 for rising curve, got {stats['max_drawdown']}",
        )

    # ------------------------------------------------------------------
    # Monotonically declining equity curve
    # ------------------------------------------------------------------

    def _declining_totals(self, n=60, start=10_000.0, step=100.0):
        """Generate a strictly decreasing equity series."""
        return [start - i * step for i in range(n)]

    def test_declining_curve_sharpe_negative(self):
        """A purely declining equity curve must produce sharpe < 0."""
        portfolio = _build_portfolio_with_totals(self._declining_totals())
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        self.assertLess(
            stats["sharpe"],
            0.0,
            msg=f"Expected sharpe < 0 for declining curve, got {stats['sharpe']}",
        )

    def test_declining_curve_cagr_negative(self):
        """A purely declining equity curve must produce cagr < 0."""
        portfolio = _build_portfolio_with_totals(self._declining_totals())
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        self.assertLess(
            stats["cagr"],
            0.0,
            msg=f"Expected cagr < 0 for declining curve, got {stats['cagr']}",
        )

    def test_declining_curve_max_drawdown_significant(self):
        """A purely declining equity curve must have a substantial positive drawdown magnitude.

        Note: the backend encodes max_drawdown as a positive magnitude (fraction of peak
        that was lost), consistent with the rising-curve test where it is ~0 (no loss).
        A strictly falling curve should produce a large positive value (close to 1.0).
        """
        portfolio = _build_portfolio_with_totals(self._declining_totals())
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        # max_drawdown is a positive magnitude; for a large decline it must be substantial.
        self.assertGreater(
            stats["max_drawdown"],
            0.1,
            msg=f"Expected max_drawdown > 0.1 (significant) for declining curve, got {stats['max_drawdown']}",
        )

    # ------------------------------------------------------------------
    # Sanity: status field
    # ------------------------------------------------------------------

    def test_ok_status_for_sufficient_data(self):
        """Any curve with >= 2 points must return status=='ok'."""
        portfolio = _build_portfolio_with_totals([10_000.0, 11_000.0])
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")

    def test_not_enough_data_for_single_point(self):
        """A single-point series must return status=='not_enough_data' and sentinel sharpe."""
        portfolio = _build_portfolio_with_totals([10_000.0])
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "not_enough_data")
        self.assertEqual(stats["sharpe"], -999.0)


if __name__ == "__main__":
    unittest.main()
