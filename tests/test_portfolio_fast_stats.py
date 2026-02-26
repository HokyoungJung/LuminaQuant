"""Fast portfolio stats path tests."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from lumina_quant.backtesting.portfolio_backtest import Portfolio


class MockBars:
    """Minimal bars adapter for fast-stats tests."""

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
        if val_type == "close":
            return close
        if val_type == "high":
            return close
        if val_type == "low":
            return close
        return close


class MockConfig:
    """Minimal config for Portfolio."""

    INITIAL_CAPITAL = 10000.0
    ANNUAL_PERIODS = 252
    LEVERAGE = 1


class TestPortfolioFastStats(unittest.TestCase):
    """Validate optimization-focused fast stats method."""

    def test_not_enough_data(self):
        bars = MockBars([100.0])
        portfolio = Portfolio(bars, events=None, start_date=datetime(2024, 1, 1), config=MockConfig)
        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "not_enough_data")
        self.assertEqual(stats["sharpe"], -999.0)

    def test_fast_stats_after_updates(self):
        bars = MockBars([100.0, 101.0, 99.0, 103.0])
        portfolio = Portfolio(
            bars,
            events=None,
            start_date=datetime(2024, 1, 1),
            config=MockConfig,
            record_history=False,
        )
        for _ in range(3):
            portfolio.update_timeindex(event=None)
            bars.advance()
        portfolio.update_timeindex(event=None)

        stats = portfolio.output_summary_stats_fast()
        self.assertEqual(stats["status"], "ok")
        self.assertIn("sharpe", stats)
        self.assertIn("cagr", stats)
        self.assertIn("max_drawdown", stats)


if __name__ == "__main__":
    unittest.main()
