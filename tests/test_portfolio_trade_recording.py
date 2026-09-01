"""Portfolio trade recording mode tests."""

from __future__ import annotations

import queue
import unittest
from datetime import datetime

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import FillEvent


class MockBars:
    """Minimal bars adapter for portfolio fill tests."""

    symbol_list = ["BTC/USDT"]

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_market_spec(symbol):
        _ = symbol
        return {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 5.0}


class MockConfig:
    """Minimal config for portfolio tests."""

    INITIAL_CAPITAL = 10000.0
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.1
    MAX_DAILY_LOSS_PCT = 0.99
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    DEFAULT_STOP_LOSS_PCT = 0.01


class TestPortfolioTradeRecording(unittest.TestCase):
    """Validate memory-light trade counting mode."""

    def _fill(self):
        return FillEvent(
            timeindex=datetime(2026, 1, 1),
            symbol="BTC/USDT",
            exchange="SIM",
            quantity=1.0,
            direction="BUY",
            fill_cost=100.0,
            commission=0.1,
            status="FILLED",
        )

    def test_record_trades_false_keeps_count_without_log_growth(self):
        portfolio = Portfolio(
            MockBars(),
            queue.Queue(),
            datetime(2026, 1, 1),
            MockConfig,
            record_trades=False,
        )
        portfolio.update_fill(self._fill())

        self.assertEqual(portfolio.trade_count, 1)
        self.assertEqual(len(portfolio.trades), 0)

    def test_record_trades_true_keeps_trade_log(self):
        portfolio = Portfolio(
            MockBars(),
            queue.Queue(),
            datetime(2026, 1, 1),
            MockConfig,
            record_trades=True,
        )
        portfolio.update_fill(self._fill())

        self.assertEqual(portfolio.trade_count, 1)
        self.assertEqual(len(portfolio.trades), 1)


if __name__ == "__main__":
    unittest.main()
