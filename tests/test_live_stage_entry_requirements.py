"""Phase 5 go-live stage entry requirements and auto-halt tests.

Tests cover:
- IS_TESTNET routing: testnet/shadow → True, canary/full → False (free composition,
  no sequential traversal required — spec R7)
- EFFECTIVE_POSITION_FRACTION: canary → canary_position_fraction, others → 1.0
- Consecutive-loss auto-halt: N losses → new entries blocked; reduce-only allowed
- Daily-loss auto-halt: intraday drawdown >= max_intraday_drawdown_pct → FREEZE
- record_loss / reset_consecutive_losses lifecycle
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from lumina_quant.configuration.loader import build_runtime_config
from lumina_quant.live.trader import _build_live_config_namespace
from lumina_quant.risk_manager import RiskManager


# ── helpers ──────────────────────────────────────────────────────────────────


def _base_raw(*, stage: str = "testnet") -> dict:
    return {
        "trading": {"symbols": ["BTC/USDT"], "timeframe": "1m", "timeframes": ["1s", "1m"]},
        "storage": {
            "materializer_required_timeframes": ["1s", "1m"],
            "materializer_base_timeframe": "1s",
        },
        "live": {
            "mode": "paper",
            "go_live_stage": stage,
            "market_data_source": "committed",
            "order_state_source": "polling",
            "exchange": {
                "driver": "binance_futures",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 2,
            },
        },
    }


def _view(*, stage: str = "testnet", canary_fraction: float = 0.10):
    raw = _base_raw(stage=stage)
    raw["live"]["canary_position_fraction"] = canary_fraction
    return _build_live_config_namespace(build_runtime_config(raw, env={}), symbols=["BTC/USDT"])


def _portfolio(*, equity: float, day_start: float, rolling_loss: float = 0.0, frozen: bool = False):
    class _P:
        symbol_list = ["BTC/USDT"]

        def __init__(self):
            self.current_holdings = {"total": equity, "BTC/USDT": 0.0}
            self.current_positions = {"BTC/USDT": 0.0}
            self.current_position_legs = {}
            self.day_start_equity = day_start
            self.circuit_breaker_tripped = False
            self.trading_frozen = frozen

        @staticmethod
        def get_rolling_loss_pct(window_seconds=3600):
            _ = window_seconds
            return rolling_loss

    return _P()


def _risk_cfg(
    *,
    consecutive_loss_halt_count: int = 3,
    max_daily_loss: float = 0.05,
    freeze: bool = True,
):
    class _Cfg:
        MAX_ORDER_VALUE = 0.0
        MAX_ORDER_NOTIONAL_PCT = 0.0
        MAX_DAILY_LOSS_PCT = max_daily_loss
        MAX_INTRADAY_DRAWDOWN_PCT = max_daily_loss
        MAX_ROLLING_LOSS_PCT_1H = 0.99
        MAX_SYMBOL_EXPOSURE_PCT = 1.0
        MAX_TOTAL_MARGIN_PCT = 1.0
        MAX_TOTAL_NOTIONAL_PCT = 0.0
        FREEZE_NEW_ENTRIES_ON_BREACH = freeze
        AUTO_FLATTEN_ON_BREACH = False
        TARGET_ALLOCATION_MODE = "legacy_notional_cap"
        LEVERAGE = 1
        CONSECUTIVE_LOSS_HALT_COUNT = consecutive_loss_halt_count

    return _Cfg


# ── IS_TESTNET routing ────────────────────────────────────────────────────────


def test_testnet_stage_is_testnet_true():
    assert _view(stage="testnet").IS_TESTNET is True


def test_shadow_stage_is_testnet_true():
    assert _view(stage="shadow").IS_TESTNET is True


def test_canary_stage_is_testnet_false():
    assert _view(stage="canary").IS_TESTNET is False


def test_full_stage_is_testnet_false():
    assert _view(stage="full").IS_TESTNET is False


def test_go_live_stage_attribute_exposed():
    for stage in ("testnet", "shadow", "canary", "full"):
        assert stage == _view(stage=stage).GO_LIVE_STAGE


# ── Spec R7: free composition (no sequential traversal required) ─────────────


def test_full_stage_accessible_without_prior_stages():
    """Operator may select 'full' directly — no sequential traversal required."""
    view = _view(stage="full")
    # IS_TESTNET=False means prod endpoint is selected — gate logic is at startup
    assert view.IS_TESTNET is False
    assert view.GO_LIVE_STAGE == "full"


# ── Canary sizing ─────────────────────────────────────────────────────────────


def test_canary_stage_effective_fraction_uses_canary_value():
    view = _view(stage="canary", canary_fraction=0.10)
    assert pytest.approx(0.10) == view.EFFECTIVE_POSITION_FRACTION


def test_testnet_stage_effective_fraction_is_one():
    assert pytest.approx(1.0) == _view(stage="testnet").EFFECTIVE_POSITION_FRACTION


def test_shadow_stage_effective_fraction_is_one():
    assert pytest.approx(1.0) == _view(stage="shadow").EFFECTIVE_POSITION_FRACTION


def test_full_stage_effective_fraction_is_one():
    assert pytest.approx(1.0) == _view(stage="full").EFFECTIVE_POSITION_FRACTION


def test_canary_position_fraction_exposed():
    view = _view(stage="canary", canary_fraction=0.05)
    assert pytest.approx(0.05) == view.CANARY_POSITION_FRACTION


# ── Consecutive-loss auto-halt ────────────────────────────────────────────────


def test_consecutive_loss_halt_blocks_new_entry_after_n_losses():
    """N consecutive losses → new entries blocked."""
    manager = RiskManager(_risk_cfg(consecutive_loss_halt_count=3))
    portfolio = _portfolio(equity=10000.0, day_start=10000.0)
    order = SimpleNamespace(symbol="BTC/USDT", quantity=0.1, direction="BUY", reduce_only=False)

    for i in range(3):
        manager.record_loss(realized_pnl=-100.0)

    passed, reason = manager.check_order(order, current_price=100.0, portfolio=portfolio)
    assert passed is False
    assert "consecutive" in reason.lower()


def test_consecutive_loss_halt_allows_reduce_only():
    """Reduce-only orders are always allowed even after consecutive-loss halt."""
    manager = RiskManager(_risk_cfg(consecutive_loss_halt_count=2))
    portfolio = _portfolio(equity=10000.0, day_start=10000.0)
    reduce_order = SimpleNamespace(
        symbol="BTC/USDT", quantity=0.1, direction="SELL", reduce_only=True
    )

    for _ in range(5):
        manager.record_loss(realized_pnl=-50.0)

    passed, _reason = manager.check_order(reduce_order, current_price=100.0, portfolio=portfolio)
    assert passed is True


def test_consecutive_loss_resets_on_profit():
    """A profitable fill resets the consecutive-loss counter."""
    manager = RiskManager(_risk_cfg(consecutive_loss_halt_count=3))
    portfolio = _portfolio(equity=10000.0, day_start=10000.0)
    order = SimpleNamespace(symbol="BTC/USDT", quantity=0.1, direction="BUY", reduce_only=False)

    for _ in range(3):
        manager.record_loss(realized_pnl=-100.0)

    # Profit resets counter
    manager.record_loss(realized_pnl=50.0)

    passed, _ = manager.check_order(order, current_price=100.0, portfolio=portfolio)
    assert passed is True


def test_reset_consecutive_losses_clears_counter():
    manager = RiskManager(_risk_cfg(consecutive_loss_halt_count=2))
    portfolio = _portfolio(equity=10000.0, day_start=10000.0)
    order = SimpleNamespace(symbol="BTC/USDT", quantity=0.1, direction="BUY", reduce_only=False)

    for _ in range(5):
        manager.record_loss(realized_pnl=-100.0)

    manager.reset_consecutive_losses()

    passed, _ = manager.check_order(order, current_price=100.0, portfolio=portfolio)
    assert passed is True


def test_record_loss_increments_count():
    manager = RiskManager(_risk_cfg())
    assert manager._consecutive_loss_count == 0
    count = manager.record_loss(realized_pnl=-1.0)
    assert count == 1
    manager.record_loss(realized_pnl=-1.0)
    assert manager._consecutive_loss_count == 2


def test_record_profit_resets_to_zero():
    manager = RiskManager(_risk_cfg())
    manager.record_loss(realized_pnl=-100.0)
    manager.record_loss(realized_pnl=-100.0)
    count = manager.record_loss(realized_pnl=50.0)
    assert count == 0


# ── Daily-loss / intraday-drawdown auto-halt ──────────────────────────────────


def test_daily_loss_breach_triggers_freeze():
    """Simulated intraday drawdown breach → evaluate_portfolio_risk returns FREEZE."""
    manager = RiskManager(_risk_cfg(max_daily_loss=0.05))
    # 6% drawdown exceeds 5% threshold
    portfolio = _portfolio(equity=9400.0, day_start=10000.0)
    passed, reason, action, details = manager.evaluate_portfolio_risk(portfolio)
    assert passed is False
    assert action == "FREEZE"
    assert "drawdown" in reason.lower() or "breach" in reason.lower()
    assert float(details["intraday_loss_pct"]) >= 0.05


def test_daily_loss_below_threshold_passes():
    manager = RiskManager(_risk_cfg(max_daily_loss=0.10))
    portfolio = _portfolio(equity=9500.0, day_start=10000.0)  # 5% loss < 10% threshold
    passed, _reason, action, _ = manager.evaluate_portfolio_risk(portfolio)
    assert passed is True
    assert action == "NONE"


def test_consecutive_loss_halt_without_portfolio_has_no_check():
    """Without portfolio, consecutive-loss check is not applied (no position context)."""
    manager = RiskManager(_risk_cfg(consecutive_loss_halt_count=1))
    for _ in range(5):
        manager.record_loss(realized_pnl=-100.0)
    order = SimpleNamespace(symbol="BTC/USDT", quantity=0.01, direction="BUY", reduce_only=False)
    # check_order without portfolio skips the portfolio-level block
    passed, _ = manager.check_order(order, current_price=100.0, portfolio=None)
    assert passed is True  # portfolio-level checks are skipped when portfolio=None


# ── Phase 5 schema field defaults ────────────────────────────────────────────


def test_kill_switch_enabled_default_true():
    view = _view(stage="testnet")
    assert view.KILL_SWITCH_ENABLED is True


def test_shadow_parity_min_ratio_exposed():
    raw = _base_raw()
    raw["live"]["shadow_parity_min_ratio"] = 0.95
    view = _build_live_config_namespace(build_runtime_config(raw, env={}), symbols=["BTC/USDT"])
    assert pytest.approx(0.95) == view.SHADOW_PARITY_MIN_RATIO


def test_shadow_parity_window_bars_exposed():
    raw = _base_raw()
    raw["live"]["shadow_parity_window_bars"] = 500
    view = _build_live_config_namespace(build_runtime_config(raw, env={}), symbols=["BTC/USDT"])
    assert view.SHADOW_PARITY_WINDOW_BARS == 500
