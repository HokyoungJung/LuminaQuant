from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.risk_manager import RiskManager


class _Config:
    MAX_ORDER_VALUE = 5000.0
    MAX_DAILY_LOSS_PCT = 0.05
    MAX_INTRADAY_DRAWDOWN_PCT = 0.03
    MAX_ROLLING_LOSS_PCT_1H = 0.05
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_TOTAL_MARGIN_PCT = 0.5
    FREEZE_NEW_ENTRIES_ON_BREACH = True
    AUTO_FLATTEN_ON_BREACH = False
    TARGET_ALLOCATION_MODE = "legacy_notional_cap"
    LEVERAGE = 1


def _portfolio(*, equity: float, day_start: float, rolling_loss: float, frozen: bool = False):
    class _P:
        symbol_list = ["BTC/USDT"]

        def __init__(self):
            self.current_holdings = {"total": equity, "BTC/USDT": 1000.0}
            self.current_positions = {"BTC/USDT": 1.0}
            self.current_position_legs = {}
            self.day_start_equity = day_start
            self.circuit_breaker_tripped = False
            self.trading_frozen = frozen

        @staticmethod
        def get_rolling_loss_pct(window_seconds=3600):
            _ = window_seconds
            return rolling_loss

    return _P()


def test_risk_manager_freeze_on_intraday_drawdown_breach():
    manager = RiskManager(_Config)
    portfolio = _portfolio(equity=9600.0, day_start=10000.0, rolling_loss=0.0)

    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)

    assert passed is False
    assert reason == "Intraday drawdown breach"
    assert action == "FREEZE"
    assert float(details["intraday_loss_pct"]) >= 0.03


def test_risk_manager_reduce_only_allowed_during_trade_freeze():
    manager = RiskManager(_Config)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0, frozen=True)
    reduce_only_order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=0.1,
        direction="SELL",
        reduce_only=True,
    )

    passed, reason = manager.check_order(
        reduce_only_order, current_price=100.0, portfolio=portfolio
    )

    assert passed is True
    assert "reduce-only" in reason.lower()


def test_risk_manager_blocks_new_entries_during_trade_freeze():
    manager = RiskManager(_Config)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0, frozen=True)
    order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=0.1,
        direction="BUY",
        reduce_only=False,
    )

    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)

    assert passed is False
    assert "freeze" in reason.lower()


def test_risk_manager_blocks_hedge_leg_when_side_aware_exposure_exceeds_cap():
    manager = RiskManager(_Config)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0)
    portfolio.current_positions = {"BTC/USDT": 0.0}
    portfolio.current_position_legs = {"BTC/USDT": {"LONG": 10.0, "SHORT": 10.0}}
    order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=10.0,
        direction="BUY",
        position_side="LONG",
        reduce_only=False,
    )

    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)

    assert passed is False
    assert "exposure" in reason.lower()


def test_risk_manager_allows_side_aware_reduce_of_short_leg_within_cap():
    manager = RiskManager(_Config)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0)
    portfolio.current_positions = {"BTC/USDT": 0.0}
    portfolio.current_position_legs = {"BTC/USDT": {"LONG": 15.0, "SHORT": 5.0}}
    order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=2.0,
        direction="BUY",
        position_side="SHORT",
        reduce_only=False,
    )

    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)

    assert passed is True
    assert reason == "Passed"


def test_risk_manager_allows_equity_scaled_isolated_lane_without_fixed_cap():
    class _IsolatedConfig(_Config):
        MAX_ORDER_VALUE = 0.0
        MAX_ORDER_NOTIONAL_PCT = 1.10
        MAX_SYMBOL_EXPOSURE_PCT = 1.10
        MAX_TOTAL_NOTIONAL_PCT = 1.20

    manager = RiskManager(_IsolatedConfig)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0)
    portfolio.current_holdings["BTC/USDT"] = 0.0
    portfolio.current_positions = {"BTC/USDT": 0.0}
    order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=105.0,
        direction="BUY",
        reduce_only=False,
    )

    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)

    assert passed is True
    assert reason == "Passed"


def test_risk_manager_explicit_absolute_emergency_cap_still_blocks():
    class _EmergencyCapConfig(_Config):
        MAX_ORDER_VALUE = 5000.0
        MAX_ORDER_NOTIONAL_PCT = 1.10
        MAX_SYMBOL_EXPOSURE_PCT = 1.10
        MAX_TOTAL_NOTIONAL_PCT = 1.20

    manager = RiskManager(_EmergencyCapConfig)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0)
    portfolio.current_holdings["BTC/USDT"] = 0.0
    portfolio.current_positions = {"BTC/USDT": 0.0}
    order = SimpleNamespace(
        symbol="BTC/USDT",
        quantity=105.0,
        direction="BUY",
        reduce_only=False,
    )

    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)

    assert passed is False
    assert "exceeds limit" in reason


def test_evaluate_portfolio_risk_uses_isolated_margin_not_notional_for_margin_utilization():
    class _IsolatedConfig(_Config):
        TARGET_ALLOCATION_MODE = "isolated_margin_fraction"
        LEVERAGE = 7
        MAX_TOTAL_MARGIN_PCT = 0.20

    manager = RiskManager(_IsolatedConfig)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0)
    portfolio.current_holdings["BTC/USDT"] = 10500.0

    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)

    assert passed is True
    assert reason == "Passed"
    assert action == "NONE"
    assert details["margin_utilization"] == 0.15
    assert details["target_allocation_mode"] == "isolated_margin_fraction"


def test_evaluate_portfolio_risk_blocks_isolated_margin_when_margin_cap_breached():
    class _IsolatedConfig(_Config):
        TARGET_ALLOCATION_MODE = "isolated_margin_fraction"
        LEVERAGE = 7
        MAX_TOTAL_MARGIN_PCT = 0.10

    manager = RiskManager(_IsolatedConfig)
    portfolio = _portfolio(equity=10000.0, day_start=10000.0, rolling_loss=0.0)
    portfolio.current_holdings["BTC/USDT"] = 10500.0

    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)

    assert passed is False
    assert reason == "Margin utilization breach"
    assert action == "FREEZE"
    assert details["margin_utilization"] == 0.15
