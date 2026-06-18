"""Audit-hardening tests (fix/audit-hardening).

Covers:
  * signal-metadata risk overrides are clamped to config ceilings by default,
    but kept unclamped when allow_metadata_risk_override=True;
  * a downward override still lowers below the ceiling;
  * the daily-loss cap is wired as a defense-in-depth FREEZE/FLATTEN check;
  * hard_drawdown_flatten_pct forces FLATTEN while the default 0.0 does not.
"""

from __future__ import annotations

from datetime import datetime

from lumina_quant.core.events import SignalEvent
from lumina_quant.risk_manager import RiskManager
from lumina_quant.services.portfolio import PortfolioSizingService


def _signal(metadata=None):
    return SignalEvent(
        strategy_id=1,
        symbol="BTC/USDT",
        datetime=datetime(2026, 1, 1),
        signal_type="LONG",
        stop_loss=97.5,
        metadata=metadata,
    )


def _base_kwargs(**overrides):
    kwargs = dict(
        current_price=100.0,
        equity=10_000.0,
        risk_per_trade=0.001,
        default_stop_loss_pct=0.025,
        max_symbol_exposure_pct=0.25,
        target_allocation=0.15,
        max_order_value=2_500.0,
        target_allocation_mode="isolated_margin_fraction",
        leverage=2.0,
        max_order_notional_pct=0.50,
    )
    kwargs.update(overrides)
    return kwargs


# --- Metadata override clamping (fix #1) -----------------------------------


# Relax the per-order caps so leverage is the binding constraint for these tests.
def _leverage_kwargs(**overrides):
    return _base_kwargs(
        max_order_value=0.0,
        max_order_notional_pct=10.0,
        max_symbol_exposure_pct=10.0,
        **overrides,
    )


def test_metadata_leverage_override_clamped_to_config_ceiling_by_default():
    # Override tries to RAISE leverage 2 -> 50; clamped to the config leverage (2.0).
    clamped = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"leverage": 50.0}), **_leverage_kwargs()
    )
    legacy = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"leverage": 50.0}),
        allow_metadata_risk_override=True,
        **_leverage_kwargs(),
    )
    no_override = PortfolioSizingService.risk_based_quantity(signal=_signal(), **_leverage_kwargs())
    # Clamped result matches the no-override config-leverage run.
    assert clamped == no_override
    # Legacy unclamped applies the raised leverage, producing a larger notional.
    assert legacy > clamped


def test_metadata_leverage_override_respects_explicit_max_leverage_arg():
    clamped = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"leverage": 50.0}),
        **_leverage_kwargs(max_leverage=3.0),
    )
    ceiling_run = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"leverage": 3.0}),
        **_leverage_kwargs(max_leverage=3.0),
    )
    assert clamped == ceiling_run


def test_metadata_leverage_downward_override_still_lowers():
    lowered = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"leverage": 1.0}), **_leverage_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(signal=_signal(), **_leverage_kwargs())
    assert lowered < no_override


# Relax the per-order caps and raise target_allocation so the symbol-exposure cap
# is the binding constraint for these tests.
def _exposure_kwargs(**overrides):
    return _base_kwargs(
        max_order_value=0.0,
        max_order_notional_pct=10.0,
        target_allocation=10.0,
        **overrides,
    )


def test_metadata_max_symbol_exposure_override_clamped_by_default():
    clamped = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_symbol_exposure_pct": 5.0}), **_exposure_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(signal=_signal(), **_exposure_kwargs())
    legacy = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_symbol_exposure_pct": 5.0}),
        allow_metadata_risk_override=True,
        **_exposure_kwargs(),
    )
    assert clamped == no_override
    assert legacy > clamped


def test_metadata_max_symbol_exposure_downward_override_still_lowers():
    lowered = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_symbol_exposure_pct": 0.05}), **_exposure_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(signal=_signal(), **_exposure_kwargs())
    assert lowered < no_override


# Relax exposure / notional_pct so the fixed max_order_value cap is binding.
def _order_value_kwargs(**overrides):
    return _base_kwargs(
        max_symbol_exposure_pct=10.0,
        max_order_notional_pct=10.0,
        target_allocation=10.0,
        **overrides,
    )


def test_metadata_max_order_value_override_clamped_by_default():
    # Try to RAISE the per-order cap 2_500 -> 100_000; clamped back to 2_500.
    clamped = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_value": 100_000.0}), **_order_value_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(
        signal=_signal(), **_order_value_kwargs()
    )
    legacy = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_value": 100_000.0}),
        allow_metadata_risk_override=True,
        **_order_value_kwargs(),
    )
    assert clamped == no_override
    assert legacy > clamped
    # Defence-in-depth: clamped notional never exceeds the configured $2.5k cap.
    assert clamped * 100.0 <= 2_500.0 + 1e-9


def test_metadata_max_order_value_scale_path_clamped_by_default():
    # The *_scale path also routes through the config ceiling clamp.
    clamped = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_value_scale": 40.0}), **_order_value_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(
        signal=_signal(), **_order_value_kwargs()
    )
    assert clamped == no_override
    assert clamped * 100.0 <= 2_500.0 + 1e-9


def test_metadata_max_order_value_downward_override_still_lowers():
    lowered = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_value": 500.0}), **_order_value_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(
        signal=_signal(), **_order_value_kwargs()
    )
    assert lowered < no_override
    assert lowered * 100.0 <= 500.0 + 1e-9


# Relax order_value / exposure so the equity-scaled notional_pct cap is binding.
def _notional_pct_kwargs(**overrides):
    return _base_kwargs(
        max_order_value=0.0,
        max_symbol_exposure_pct=10.0,
        target_allocation=10.0,
        **overrides,
    )


def test_metadata_max_order_notional_pct_override_clamped_when_configured():
    clamped = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_notional_pct": 5.0}), **_notional_pct_kwargs()
    )
    no_override = PortfolioSizingService.risk_based_quantity(
        signal=_signal(), **_notional_pct_kwargs()
    )
    legacy = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_notional_pct": 5.0}),
        allow_metadata_risk_override=True,
        **_notional_pct_kwargs(),
    )
    assert clamped == no_override
    assert legacy > clamped


def test_metadata_max_order_notional_pct_override_kept_when_no_cap_configured():
    # config max_order_notional_pct == 0.0 ("no cap"): override is left intact, but
    # max_order_value + exposure caps still bind.
    kwargs = _base_kwargs(max_order_notional_pct=0.0)
    override = PortfolioSizingService.risk_based_quantity(
        signal=_signal({"max_order_notional_pct": 5.0}), **kwargs
    )
    no_override = PortfolioSizingService.risk_based_quantity(signal=_signal(), **kwargs)
    # Both bounded by the $2.5k max_order_value cap, so identical here.
    assert override == no_override
    assert override * 100.0 <= 2_500.0 + 1e-9


# --- RiskManager daily-loss + hard-flatten tiers (fix #2 / #3) --------------


class _Config:
    MAX_ORDER_VALUE = 5000.0
    MAX_DAILY_LOSS_PCT = 0.05
    MAX_INTRADAY_DRAWDOWN_PCT = 0.10  # raised above daily-loss so daily-loss can fire
    MAX_ROLLING_LOSS_PCT_1H = 0.20
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_TOTAL_MARGIN_PCT = 0.5
    FREEZE_NEW_ENTRIES_ON_BREACH = True
    AUTO_FLATTEN_ON_BREACH = False
    TARGET_ALLOCATION_MODE = "legacy_notional_cap"
    LEVERAGE = 1


def _portfolio(*, equity: float, day_start: float, rolling_loss: float = 0.0):
    class _P:
        symbol_list = ["BTC/USDT"]

        def __init__(self):
            self.current_holdings = {"total": equity, "BTC/USDT": 0.0}
            self.current_positions = {"BTC/USDT": 0.0}
            self.current_position_legs = {}
            self.day_start_equity = day_start
            self.circuit_breaker_tripped = False
            self.trading_frozen = False

        @staticmethod
        def get_rolling_loss_pct(window_seconds=3600):
            _ = window_seconds
            return rolling_loss

    return _P()


def test_daily_loss_breach_freezes_when_no_auto_flatten():
    # 6% loss > 5% daily cap, < 10% intraday cap -> daily-loss check fires.
    manager = RiskManager(_Config)
    portfolio = _portfolio(equity=9400.0, day_start=10000.0)

    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)

    assert passed is False
    assert reason == "Daily loss breach"
    assert action == "FREEZE"
    assert float(details["threshold"]) == 0.05


def test_daily_loss_breach_flattens_when_auto_flatten_enabled():
    class _AutoFlatten(_Config):
        AUTO_FLATTEN_ON_BREACH = True

    manager = RiskManager(_AutoFlatten)
    portfolio = _portfolio(equity=9400.0, day_start=10000.0)

    passed, reason, action, _ = manager.evaluate_portfolio_risk(portfolio)

    assert passed is False
    assert reason == "Daily loss breach"
    assert action == "FLATTEN"


def test_hard_drawdown_flatten_pct_forces_flatten_without_auto_flatten():
    class _HardTier(_Config):
        AUTO_FLATTEN_ON_BREACH = False
        HARD_DRAWDOWN_FLATTEN_PCT = 0.08

    manager = RiskManager(_HardTier)
    portfolio = _portfolio(equity=9100.0, day_start=10000.0)  # 9% loss

    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)

    assert passed is False
    assert reason == "Hard drawdown flatten breach"
    assert action == "FLATTEN"
    assert float(details["threshold"]) == 0.08


def test_hard_drawdown_flatten_pct_default_zero_does_not_force_flatten():
    # Default config (no HARD_DRAWDOWN_FLATTEN_PCT) -> tier disabled, legacy behavior.
    manager = RiskManager(_Config)
    assert manager.hard_drawdown_flatten_pct == 0.0
    portfolio = _portfolio(equity=9100.0, day_start=10000.0)  # 9% loss

    passed, reason, action, _ = manager.evaluate_portfolio_risk(portfolio)

    # 9% > 5% daily cap but < 10% intraday cap; daily-loss tier fires as FREEZE,
    # never the hard FLATTEN tier.
    assert passed is False
    assert reason == "Daily loss breach"
    assert action == "FREEZE"
