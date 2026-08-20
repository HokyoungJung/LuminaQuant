"""L-C / L-D engine cost gates (docs/research_note/performance_lever_measurement.md).

Covers the three config-gated levers wired into the REAL engine seam
(StrategyQualityOverlay + Portfolio.generate_order_from_signal):

- min_hold_bars: bare EXITs / reversals blocked inside the hold window;
  protective-marker exits always pass; OFF default is byte-identical.
- no_trade_band_bps: sub-band entry / partial-exit orders dropped; full
  exits exempt; OFF default is byte-identical.
- funding_entry_guard: declared sub-funding-interval holds that straddle a
  settlement boundary are never opened; undeclared signals never blocked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import FillEvent, SignalEvent
from lumina_quant.portfolio.strategy_quality import StrategyQualityOverlay

SYMBOL = "BTC/USDT"


class _OverlayConfig:
    """Plain-class config stub (uppercase-attr surface)."""

    STRATEGY_QUALITY_ENABLED = False
    STRATEGY_QUALITY_MIN_HOLD_BARS = 0


class _BarsStub:
    symbol_list = [SYMBOL]

    def __init__(self, now=None, price=100.0):
        self.now = now or datetime(2026, 1, 1, tzinfo=UTC)
        self.price = price

    def get_latest_bar_value(self, symbol, field):
        _ = symbol, field
        return self.price

    def get_latest_bar_datetime(self, symbol):
        _ = symbol
        return self.now


class _PortfolioConfig:
    INITIAL_CAPITAL = 10_000.0
    MAX_DAILY_LOSS_PCT = 0.05
    RISK_PER_TRADE = 0.005
    MAX_SYMBOL_EXPOSURE_PCT = 0.25
    MAX_ORDER_VALUE = 5000.0
    DEFAULT_STOP_LOSS_PCT = 0.01
    MIN_TRADE_QTY = 0.001
    TARGET_ALLOCATION = 0.10
    SYMBOL_LIMITS = {}
    TIMEFRAME = "1h"


def _overlay(min_hold_bars=0, enabled=False):
    config = _OverlayConfig()
    config.STRATEGY_QUALITY_ENABLED = bool(enabled)
    config.STRATEGY_QUALITY_MIN_HOLD_BARS = int(min_hold_bars)
    return StrategyQualityOverlay(config)


def _signal(signal_type="EXIT", metadata=None):
    return SignalEvent(
        strategy_id="probe",
        symbol=SYMBOL,
        datetime=datetime(2026, 1, 1, tzinfo=UTC),
        signal_type=signal_type,
        strength=1.0,
        metadata=dict(metadata or {}),
    )


def _fill(quantity=1.0, direction="BUY"):
    return FillEvent(
        timeindex=datetime(2026, 1, 1, tzinfo=UTC),
        symbol=SYMBOL,
        exchange="SIM",
        quantity=quantity,
        direction=direction,
        fill_cost=100.0,
        commission=0.0,
        metadata={"signal_metadata": {"strategy_quality_strategy": "probe"}},
    )


def _apply(overlay, signal):
    return overlay.apply(
        signal,
        bars=SimpleNamespace(),
        current_price=100.0,
        current_equity=10_000.0,
    )


def _enter_long(overlay):
    overlay.note_fill(_fill(), old_qty=0.0, new_qty=1.0, fill_price=100.0)


# --------------------------------------------------------------------------- #
# min_hold_bars (overlay seam)
# --------------------------------------------------------------------------- #
def test_min_hold_defaults_are_byte_identical() -> None:
    overlay = _overlay(min_hold_bars=0, enabled=False)
    baseline_state = _overlay(min_hold_bars=0, enabled=False).get_state()

    # note_fill early-outs exactly as before: no entry record accumulates.
    overlay.note_fill(_fill(), old_qty=0.0, new_qty=1.0, fill_price=100.0)
    assert overlay.get_state() == baseline_state

    # apply passes the signal through untouched.
    signal = _signal("EXIT")
    decision = _apply(overlay, signal)
    assert decision.signal is signal
    assert decision.blocked_reason == ""


def test_min_hold_blocks_bare_exit_then_releases_at_hold() -> None:
    overlay = _overlay(min_hold_bars=3)
    _enter_long(overlay)

    for _ in range(2):
        overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
        decision = _apply(overlay, _signal("EXIT"))
        assert decision.signal is None
        assert decision.blocked_reason == "min_hold_active"

    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    decision = _apply(overlay, _signal("EXIT"))
    assert decision.signal is not None


def test_min_hold_lets_protective_exits_pass() -> None:
    for marker in (
        {"risk_exit": True},
        {"exit_reason": "stop_loss"},
        {"overlay_reason": "drawdown_ladder"},
    ):
        overlay = _overlay(min_hold_bars=10)
        _enter_long(overlay)
        overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
        decision = _apply(overlay, _signal("EXIT", metadata=marker))
        assert decision.signal is not None, marker


def test_min_hold_blocks_reversal_but_not_same_direction_add() -> None:
    overlay = _overlay(min_hold_bars=5)
    _enter_long(overlay)
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))

    blocked = _apply(overlay, _signal("SHORT"))
    assert blocked.signal is None
    assert blocked.blocked_reason == "min_hold_reversal_block"

    added = _apply(overlay, _signal("LONG"))
    assert added.signal is not None


def test_min_hold_state_roundtrip_preserves_entry_bar() -> None:
    overlay = _overlay(min_hold_bars=5)
    _enter_long(overlay)
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))

    restored = _overlay(min_hold_bars=5)
    restored.set_state(overlay.get_state())

    decision = _apply(restored, _signal("EXIT"))
    assert decision.signal is None
    assert decision.blocked_reason == "min_hold_active"


def test_min_hold_untracked_symbol_passes() -> None:
    overlay = _overlay(min_hold_bars=5)
    decision = _apply(overlay, _signal("EXIT"))
    assert decision.signal is not None


# --------------------------------------------------------------------------- #
# no_trade_band_bps (Portfolio seam)
# --------------------------------------------------------------------------- #
def _portfolio(**config_overrides):
    config = _PortfolioConfig()
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return Portfolio(
        _BarsStub(),
        SimpleNamespace(put=lambda event: None),
        datetime(2026, 1, 1, tzinfo=UTC),
        config,
    )


def test_no_trade_band_off_default_passes_small_entry() -> None:
    portfolio = _portfolio()
    order = portfolio.generate_order_from_signal(_signal("LONG"))
    assert order is not None


def test_no_trade_band_blocks_sub_band_entry() -> None:
    # Default sizing on this stub: TARGET_ALLOCATION 0.10 * 10k equity =
    # $1000 notional => 1000bps of equity. 1500bps blocks; 500bps passes.
    blocked = _portfolio(STRATEGY_QUALITY_NO_TRADE_BAND_BPS=1500.0)
    assert blocked.generate_order_from_signal(_signal("LONG")) is None

    passing = _portfolio(STRATEGY_QUALITY_NO_TRADE_BAND_BPS=500.0)
    assert passing.generate_order_from_signal(_signal("LONG")) is not None


def test_no_trade_band_full_exit_exempt_partial_exit_blocked() -> None:
    portfolio = _portfolio(STRATEGY_QUALITY_NO_TRADE_BAND_BPS=200.0)
    portfolio.update_fill(
        FillEvent(
            timeindex=datetime(2026, 1, 1, tzinfo=UTC),
            symbol=SYMBOL,
            exchange="SIM",
            quantity=1.0,
            direction="BUY",
            fill_cost=100.0,
            commission=0.0,
        )
    )

    # Position notional = $100 = 100bps of equity, below the 200bps band.
    partial = portfolio.generate_order_from_signal(_signal("EXIT", metadata={"exit_fraction": 0.5}))
    assert partial is None

    full = portfolio.generate_order_from_signal(_signal("EXIT"))
    assert full is not None
    assert full.quantity == 1.0


# --------------------------------------------------------------------------- #
# funding_entry_guard (Portfolio seam)
# --------------------------------------------------------------------------- #
def _guard_portfolio(now, **config_overrides):
    config = _PortfolioConfig()
    config.FUNDING_ENTRY_GUARD = True
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return Portfolio(
        _BarsStub(now=now),
        SimpleNamespace(put=lambda event: None),
        datetime(2026, 1, 1, tzinfo=UTC),
        config,
    )


def test_funding_guard_blocks_straddling_short_hold() -> None:
    # 07:30 UTC: 1800s to the 08:00 boundary < 3600s hold < 28800s interval.
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 7, 30, tzinfo=UTC))
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_seconds": 3600})
    )
    assert order is None


def test_funding_guard_passes_non_straddling_short_hold() -> None:
    # 00:10 UTC: 28200s to the 08:00 boundary > 3600s hold.
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 0, 10, tzinfo=UTC))
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_seconds": 3600})
    )
    assert order is not None


def test_funding_guard_passes_holds_at_or_above_interval() -> None:
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 7, 30, tzinfo=UTC))
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_seconds": 28800})
    )
    assert order is not None


def test_funding_guard_never_blocks_undeclared_or_exit_signals() -> None:
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    assert portfolio.generate_order_from_signal(_signal("LONG")) is not None

    portfolio.update_fill(
        FillEvent(
            timeindex=datetime(2026, 1, 1, tzinfo=UTC),
            symbol=SYMBOL,
            exchange="SIM",
            quantity=1.0,
            direction="BUY",
            fill_cost=100.0,
            commission=0.0,
        )
    )
    exit_order = portfolio.generate_order_from_signal(
        _signal("EXIT", metadata={"intended_hold_seconds": 60})
    )
    assert exit_order is not None


def test_funding_guard_intended_hold_bars_uses_config_timeframe() -> None:
    # 1h timeframe, 1 bar hold = 3600s; entered at 07:30 it straddles 08:00.
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 7, 30, tzinfo=UTC))
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_bars": 1})
    )
    assert order is None


def test_funding_guard_off_default_ignores_declared_holds() -> None:
    config = _PortfolioConfig()
    portfolio = Portfolio(
        _BarsStub(now=datetime(2026, 1, 1, 7, 59, tzinfo=UTC)),
        SimpleNamespace(put=lambda event: None),
        datetime(2026, 1, 1, tzinfo=UTC),
        config,
    )
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_seconds": 120})
    )
    assert order is not None
