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


def test_min_hold_ledger_records_fill_without_price_diagnostics() -> None:
    overlay = _overlay(min_hold_bars=3)
    overlay.note_fill(_fill(), old_qty=0.0, new_qty=1.0, fill_price=0.0)
    assert overlay.get_state()["min_hold_entries"]["__net__|BTC/USDT"]["entry_bar"] == 0


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


def test_min_hold_blocks_descriptive_exit_reason_labels() -> None:
    # F4: any-truthy exit_reason must NOT neutralize the gate — only the
    # pre-registered protective whitelist passes.
    overlay = _overlay(min_hold_bars=10)
    _enter_long(overlay)
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    decision = _apply(overlay, _signal("EXIT", metadata={"exit_reason": "rebalance"}))
    assert decision.signal is None
    assert decision.blocked_reason == "min_hold_active"


def test_min_hold_defers_blocked_exit_and_releases_at_maturity() -> None:
    # F1: a blocked bare EXIT is deferred, not dropped — the overlay releases
    # it at hold maturity so one-shot transition-emit strategies stay in sync.
    overlay = _overlay(min_hold_bars=3)
    _enter_long(overlay)
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))

    blocked = _apply(overlay, _signal("EXIT", metadata={"exit_fraction": 1.0}))
    assert blocked.signal is None
    assert overlay.pop_matured_pending_exits() == []

    # Same-direction re-entry while the exit is pending must NOT stack.
    stacked = _apply(overlay, _signal("LONG"))
    assert stacked.signal is None
    assert stacked.blocked_reason == "min_hold_exit_pending"

    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    released = overlay.pop_matured_pending_exits()
    assert len(released) == 1
    assert released[0]["symbol"] == SYMBOL
    assert released[0]["component_id"] == ""
    assert released[0]["metadata"].get("exit_fraction") == 1.0
    # Unfilled/rejected dispatches retry with the same idempotency key until an
    # authoritative fill proves the book flat.
    retry = overlay.pop_matured_pending_exits()
    assert len(retry) == 1
    assert retry[0]["client_order_id"] == released[0]["client_order_id"]
    assert overlay.get_state()["min_hold_entries"]["__net__|BTC/USDT"]["exit_pending"] is True


def test_min_hold_pending_lifecycle_survives_restart_until_book_is_flat() -> None:
    overlay = _overlay(min_hold_bars=1)
    _enter_long(overlay)
    assert _apply(overlay, _signal("EXIT")).signal is None
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    pending = overlay.pop_matured_pending_exits()[0]
    overlay.mark_pending_exit_state(pending["key"], "REJECTED")

    restored = _overlay(min_hold_bars=1)
    restored.set_state(overlay.get_state())
    retry = restored.pop_matured_pending_exits()[0]
    assert retry["client_order_id"] == pending["client_order_id"]
    restored.mark_pending_exit_state(pending["key"], "CANCELLED")
    assert _apply(restored, _signal("LONG")).blocked_reason == "min_hold_exit_pending"
    restored.reconcile_min_hold_positions({SYMBOL: 0.0}, {})
    assert restored.pop_matured_pending_exits() == []


def test_min_hold_release_survives_state_roundtrip() -> None:
    overlay = _overlay(min_hold_bars=3)
    _enter_long(overlay)
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    assert _apply(overlay, _signal("EXIT")).signal is None

    restored = _overlay(min_hold_bars=3)
    restored.set_state(overlay.get_state())
    restored.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    restored.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    released = restored.pop_matured_pending_exits()
    assert len(released) == 1


def test_min_hold_is_component_scoped() -> None:
    # F6: one component's fresh entry must not gate another component's exit.
    overlay = _overlay(min_hold_bars=10)
    fill_a = _fill()
    fill_a.metadata = {"component_id": "comp-a"}
    overlay.note_fill(
        fill_a,
        1.0,
        2.0,
        100.0,
        component_id="comp-a",
        component_old_qty=0.0,
        component_new_qty=1.0,
    )
    overlay.next_bar(datetime(2026, 1, 1, tzinfo=UTC))

    # comp-b holds nothing tracked: its EXIT passes.
    exit_b = _apply(overlay, _signal("EXIT", metadata={"component_id": "comp-b"}))
    assert exit_b.signal is not None
    # comp-a's own bare EXIT is gated.
    exit_a = _apply(overlay, _signal("EXIT", metadata={"component_id": "comp-a"}))
    assert exit_a.signal is None
    # Un-scoped net-book signals are not gated by comp-a's record.
    exit_net = _apply(overlay, _signal("EXIT"))
    assert exit_net.signal is not None


def test_portfolio_update_timeindex_emits_released_min_hold_exit() -> None:
    events: list = []
    portfolio = _portfolio(STRATEGY_QUALITY_MIN_HOLD_BARS=2)
    portfolio.events = SimpleNamespace(put=events.append)
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
    portfolio.strategy_quality.next_bar(datetime(2026, 1, 1, tzinfo=UTC))
    blocked = portfolio.generate_order_from_signal(_signal("EXIT"))
    assert blocked is None

    portfolio.update_timeindex(SimpleNamespace(type="MARKET"))
    released_exits = [e for e in events if getattr(e, "signal_type", "") == "EXIT"]
    assert len(released_exits) == 1
    assert released_exits[0].metadata.get("overlay_reason") == "min_hold_released"


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
    config.FUNDING_ON_UTC_BOUNDARY = True
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return Portfolio(
        _BarsStub(now=now),
        SimpleNamespace(put=lambda event: None),
        datetime(2026, 1, 1, tzinfo=UTC),
        config,
    )


def test_funding_guard_blocks_straddling_short_hold() -> None:
    # Decision bar stamped 06:30 UTC on 1h bars: the MKT order fills at the
    # NEXT bar open (~07:30), so the 3600s hold [07:30, 08:30] straddles the
    # 08:00 settlement -> blocked (fill-time anchoring, F2/W1).
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 6, 30, tzinfo=UTC))
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_seconds": 3600})
    )
    assert order is None


def test_funding_guard_passes_non_straddling_short_hold() -> None:
    # 00:10 UTC decision -> fill ~01:10; hold [01:10, 02:10] touches no boundary.
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 0, 10, tzinfo=UTC))
    order = portfolio.generate_order_from_signal(
        _signal("LONG", metadata={"intended_hold_seconds": 3600})
    )
    assert order is not None


def test_funding_guard_anchors_at_fill_time_not_signal_bar() -> None:
    # Decision 07:30 on 1h bars fills at ~08:30 — AFTER the 08:00 settlement —
    # so the hold [08:30, 09:30] is clean and must NOT be blocked (the old
    # signal-bar anchor falsely blocked exactly this case).
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 7, 30, tzinfo=UTC))
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


def test_funding_guard_fails_closed_for_missing_hold_metadata_but_not_exits() -> None:
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    assert portfolio.generate_order_from_signal(_signal("LONG")) is None

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


def test_funding_guard_requires_utc_boundary_mode_and_finite_metadata() -> None:
    now = datetime(2026, 1, 1, 0, 10, tzinfo=UTC)
    assert (
        _guard_portfolio(now, FUNDING_ON_UTC_BOUNDARY=False).generate_order_from_signal(
            _signal("LONG", metadata={"intended_hold_seconds": 60})
        )
        is None
    )
    assert (
        _guard_portfolio(now).generate_order_from_signal(
            _signal("LONG", metadata={"intended_hold_seconds": float("nan")})
        )
        is None
    )


def test_funding_guard_intended_hold_bars_uses_config_timeframe() -> None:
    # 1h timeframe, 1-bar hold = 3600s; decision at 06:30 fills ~07:30 and the
    # hold [07:30, 08:30] straddles the 08:00 settlement.
    portfolio = _guard_portfolio(datetime(2026, 1, 1, 6, 30, tzinfo=UTC))
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
