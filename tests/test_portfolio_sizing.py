import queue
import unittest
from datetime import datetime

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import SignalEvent
from lumina_quant.services.portfolio import PortfolioSizingService


class MockBars:
    symbol_list = ["BTC/USDT"]

    def get_latest_bar_value(self, symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    def get_market_spec(self, symbol):
        _ = symbol
        return {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 5.0}


class MockConfig:
    INITIAL_CAPITAL = 10000.0
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.1
    MAX_DAILY_LOSS_PCT = 0.03
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    DEFAULT_STOP_LOSS_PCT = 0.01


class ReplayParityConfig(MockConfig):
    TARGET_ALLOCATION = 0.15
    TARGET_ALLOCATION_MODE = "isolated_margin_fraction"
    LEVERAGE = 7
    MAX_SYMBOL_EXPOSURE_PCT = 1.10
    MAX_ORDER_VALUE = 0.0
    MAX_ORDER_NOTIONAL_PCT = 1.10
    DEFAULT_STOP_LOSS_PCT = 0.025


class TestPortfolioSizing(unittest.TestCase):
    def test_risk_based_order_generation(self):
        p = Portfolio(MockBars(), queue.Queue(), datetime(2026, 1, 1), MockConfig)
        signal = SignalEvent(
            strategy_id=1,
            symbol="BTC/USDT",
            datetime=datetime(2026, 1, 1),
            signal_type="LONG",
            stop_loss=99.0,
            trailing_percent=0.02,
        )
        order = p.generate_order_from_signal(signal)
        self.assertIsNotNone(order)
        self.assertGreater(order.quantity, 0.0)
        self.assertEqual(order.direction, "BUY")
        self.assertEqual(order.position_side, "LONG")
        self.assertAlmostEqual(float(order.trailing_percent), 0.02)


def _sizing_signal(metadata=None):
    return SignalEvent(
        strategy_id=1,
        symbol="BTC/USDT",
        datetime=datetime(2026, 1, 1),
        signal_type="LONG",
        stop_loss=97.5,
        metadata=metadata,
    )


def test_explicit_isolated_margin_fraction_targets_replay_notional():
    qty = PortfolioSizingService.risk_based_quantity(
        signal=_sizing_signal(),
        current_price=100.0,
        equity=10_000.0,
        risk_per_trade=0.001,
        default_stop_loss_pct=0.025,
        max_symbol_exposure_pct=1.10,
        target_allocation=0.15,
        max_order_value=0.0,
        target_allocation_mode="isolated_margin_fraction",
        leverage=7.0,
        max_order_notional_pct=1.10,
    )

    assert qty * 100.0 == 10_500.0


def test_explicit_notional_fraction_targets_notional_fraction():
    qty = PortfolioSizingService.risk_based_quantity(
        signal=_sizing_signal(),
        current_price=100.0,
        equity=10_000.0,
        risk_per_trade=0.001,
        default_stop_loss_pct=0.025,
        max_symbol_exposure_pct=0.25,
        target_allocation=0.15,
        max_order_value=0.0,
        target_allocation_mode="notional_fraction",
        leverage=7.0,
    )

    assert qty * 100.0 == 1_500.0


def test_omitted_target_allocation_mode_keeps_legacy_risk_cap_semantics():
    qty = PortfolioSizingService.risk_based_quantity(
        signal=_sizing_signal(),
        current_price=100.0,
        equity=10_000.0,
        risk_per_trade=0.001,
        default_stop_loss_pct=0.025,
        max_symbol_exposure_pct=0.25,
        target_allocation=0.15,
        max_order_value=0.0,
    )

    assert qty * 100.0 == 400.0


def test_portfolio_order_generation_matches_alpha_zoo_replay_notional_contract():
    portfolio = Portfolio(MockBars(), queue.Queue(), datetime(2026, 1, 1), ReplayParityConfig)
    signal = SignalEvent(
        strategy_id=1,
        symbol="BTC/USDT",
        datetime=datetime(2026, 1, 1),
        signal_type="LONG",
        stop_loss=97.5,
        metadata={
            "target_allocation_mode": "isolated_margin_fraction",
            "target_allocation": 0.15,
            "leverage": 7,
            "max_order_notional_pct": 1.10,
        },
    )

    order = portfolio.generate_order_from_signal(signal)

    assert order is not None
    expected_replay_notional = 10_000.0 * 0.15 * 7.0
    assert order.quantity * 100.0 == expected_replay_notional


if __name__ == "__main__":
    unittest.main()
