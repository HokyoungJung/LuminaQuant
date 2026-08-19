"""Tests for the equity-curve kill-switch governance overlay.

The child strategy used here is a leading-underscore probe class that is NEVER
registered: ``resolve_strategy_class`` is monkeypatched inside the overlay
module (the same isolation the incumbent vol-managed overlay test uses), so the
operational strategy registry stays untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.core.events import SignalEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategies import equity_curve_kill_switch_overlay as MODULE
from lumina_quant.strategies.equity_curve_kill_switch_overlay import (
    DEFAULT_LADDER,
    EquityCurveKillSwitchOverlayStrategy,
    drawdown_ladder_scale,
    kill_switch_scale,
    parse_drawdown_ladder,
)

_SYMBOL = "AAA"
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _stamp(idx: int) -> str:
    return (_EPOCH + timedelta(days=idx)).isoformat().replace("+00:00", "Z")


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _OverlayProbeChild:
    """Deterministic probe child: emits EXIT/LONG on scripted bar indices."""

    required_features = ()
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    decision_cadence_seconds = 86400

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        self.long_at = set(params.get("long_at") or ())
        self.short_at = set(params.get("short_at") or ())
        self.exit_at = set(params.get("exit_at") or ())
        # Zero-allocation LONGs: they leave the shadow book untouched but still
        # carry the overlay's metadata, so the scale can be observed every bar.
        self.probe_at = set(params.get("probe_at") or ())
        self.target_allocation = float(params.get("target_allocation", 1.0))
        self.max_order_value = float(params.get("max_order_value", 500.0))
        self.raises = bool(params.get("raises", False))
        # Bars on which the child blows up AFTER queueing its scripted signals.
        self.raise_after = set(params.get("raise_after") or ())
        self.bar_index = -1
        self._state: dict[str, Any] = {}

    def calculate_signals(self, event: Any) -> None:
        if self.raises:
            raise RuntimeError("probe child blew up")
        self.bar_index += 1
        for symbol in self.symbol_list:
            if self.bar_index in self.exit_at:
                self.events.put(self._signal(symbol, event, "EXIT"))
            if self.bar_index in self.long_at:
                self.events.put(self._signal(symbol, event, "LONG"))
            if self.bar_index in self.short_at:
                self.events.put(self._signal(symbol, event, "SHORT"))
            if self.bar_index in self.probe_at:
                self.events.put(self._signal(symbol, event, "LONG", allocation=0.0))
        if self.bar_index in self.raise_after:
            raise RuntimeError("probe child blew up after queueing")

    def _signal(
        self, symbol: str, event: Any, signal_type: str, *, allocation: float | None = None
    ) -> SignalEvent:
        metadata: dict[str, Any] = {"strategy": "probe_child"}
        if signal_type != "EXIT":
            alloc = self.target_allocation if allocation is None else float(allocation)
            metadata["target_allocation"] = alloc
            metadata["max_symbol_exposure_pct"] = alloc
            metadata["max_order_value"] = self.max_order_value
        return SignalEvent(
            strategy_id="probe_child",
            symbol=symbol,
            datetime=getattr(event, "time", None),
            signal_type=signal_type,
            strength=1.0,
            metadata=metadata,
        )

    def get_state(self) -> dict[str, Any]:
        return {"bar_index": self.bar_index, "marker": dict(self._state)}

    def set_state(self, state: dict[str, Any]) -> None:
        self.bar_index = int(state.get("bar_index", -1))
        self._state = dict(state.get("marker") or {})


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every child class name to the unregistered probe child."""
    monkeypatch.setattr(
        MODULE, "resolve_strategy_class", lambda name, default_name=None: _OverlayProbeChild
    )


def _market_event(idx: int, close: float, symbol: str = _SYMBOL) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=_stamp(idx),
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def _make(
    events: _Queue, *, child_params: dict[str, Any] | None = None, **overrides: Any
) -> EquityCurveKillSwitchOverlayStrategy:
    bars = SimpleNamespace(
        symbol_list=[_SYMBOL],
        get_latest_feature_value=lambda *a, **k: None,
        get_latest_bar_value=lambda *a, **k: None,
    )
    params: dict[str, Any] = {
        "child_strategy_class": "ProbeChild",
        "child_params": dict(child_params or {"long_at": {0}}),
        "child_symbols": [_SYMBOL],
    }
    params.update(overrides)
    return EquityCurveKillSwitchOverlayStrategy(bars, events, **params)


def _run(
    strategy: EquityCurveKillSwitchOverlayStrategy, prices: list[float], *, start: int = 0
) -> list[tuple[int, SignalEvent]]:
    """Feed a price path, returning ``(bar_index, signal)`` for every emission."""
    events = strategy.events
    tagged: list[tuple[int, SignalEvent]] = []
    for offset, close in enumerate(prices):
        idx = start + offset
        before = len(events.items)
        strategy.calculate_signals(_market_event(idx, close))
        tagged.extend((idx, item) for item in events.items[before:])
    return tagged


def _decline(count: int, *, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start - step * i for i in range(count)]


# --------------------------------------------------------------------------- #
# (a) pure functions
# --------------------------------------------------------------------------- #


def test_drawdown_ladder_scale_returns_rung_of_deepest_breach() -> None:
    assert drawdown_ladder_scale(0.0, DEFAULT_LADDER) == pytest.approx(1.0)
    assert drawdown_ladder_scale(0.049, DEFAULT_LADDER) == pytest.approx(1.0)
    assert drawdown_ladder_scale(0.05, DEFAULT_LADDER) == pytest.approx(0.75)
    assert drawdown_ladder_scale(0.09, DEFAULT_LADDER) == pytest.approx(0.75)
    assert drawdown_ladder_scale(0.10, DEFAULT_LADDER) == pytest.approx(0.50)
    assert drawdown_ladder_scale(0.14, DEFAULT_LADDER) == pytest.approx(0.50)
    assert drawdown_ladder_scale(0.15, DEFAULT_LADDER) == pytest.approx(0.25)
    assert drawdown_ladder_scale(0.20, DEFAULT_LADDER) == pytest.approx(0.0)
    assert drawdown_ladder_scale(0.95, DEFAULT_LADDER) == pytest.approx(0.0)
    # Sign-agnostic: a negative depth means the same drawdown.
    assert drawdown_ladder_scale(-0.15, DEFAULT_LADDER) == pytest.approx(0.25)


def test_drawdown_ladder_scale_degenerate_inputs_are_inert() -> None:
    assert drawdown_ladder_scale(0.9, "") == pytest.approx(1.0)
    assert drawdown_ladder_scale(0.9, None) == pytest.approx(1.0)
    assert drawdown_ladder_scale(0.9, "garbage,0.1") == pytest.approx(1.0)
    assert drawdown_ladder_scale("not-a-number", DEFAULT_LADDER) == pytest.approx(1.0)
    assert drawdown_ladder_scale(0.12, ((0.10, 0.4), (0.05, 0.8))) == pytest.approx(0.4)


def test_parse_drawdown_ladder_sorts_and_drops_bad_rungs() -> None:
    rungs = parse_drawdown_ladder("0.20:0.0,0.05:0.75,bad,0.10:0.5,-0.3:0.1")
    assert rungs == ((0.05, 0.75), (0.10, 0.5), (0.20, 0.0))


def test_kill_switch_scale_empty_and_flat_curves() -> None:
    scale, diagnostics = kill_switch_scale([])
    assert scale == pytest.approx(1.0)
    assert diagnostics["drawdown"] == pytest.approx(0.0)

    scale, diagnostics = kill_switch_scale([1.0] * 10, month_loss_limit=0.0)
    assert scale == pytest.approx(1.0)
    assert diagnostics["reasons"] == []
    assert diagnostics["reason"] == "none"


def test_kill_switch_scale_ladder_component_on_synthetic_curve() -> None:
    scale, diagnostics = kill_switch_scale([1.0, 0.97, 0.94], month_loss_limit=0.0)
    assert diagnostics["drawdown"] == pytest.approx(0.06)
    assert diagnostics["ladder_scale"] == pytest.approx(0.75)
    assert diagnostics["loss_streak"] == 2  # curve-derived: two down-bars
    assert diagnostics["streak_scale"] == pytest.approx(1.0)
    assert scale == pytest.approx(0.75)
    assert diagnostics["reason"] == "drawdown_ladder"


def test_kill_switch_scale_consecutive_loss_halving() -> None:
    curve = [1.0, 0.99, 0.98, 0.97]  # three consecutive down-bars
    scale, diagnostics = kill_switch_scale(curve, ladder="", month_loss_limit=0.0)
    assert diagnostics["loss_streak"] == 3
    assert scale == pytest.approx(0.5)
    assert diagnostics["reason"] == "loss_streak"

    # Explicit closed-trade streak overrides the curve-derived one; each further
    # loss halves again, floored at min_scale.
    scale, _ = kill_switch_scale(curve, ladder="", month_loss_limit=0.0, loss_streak=5)
    assert scale == pytest.approx(0.125)
    scale, _ = kill_switch_scale(curve, ladder="", month_loss_limit=0.0, loss_streak=9)
    assert scale == pytest.approx(0.10)
    # Disabled when the trigger is 0.
    scale, _ = kill_switch_scale(
        curve, ladder="", month_loss_limit=0.0, consecutive_loss_halving_from=0
    )
    assert scale == pytest.approx(1.0)


def test_kill_switch_scale_month_limit_kills_and_uses_pair_anchor() -> None:
    scale, diagnostics = kill_switch_scale(
        [1.0, 0.95, 0.88], ladder="", month_loss_limit=0.10, month_start_equity=1.0
    )
    assert diagnostics["month_loss"] == pytest.approx(0.12)
    assert diagnostics["month_scale"] == pytest.approx(0.0)
    assert scale == pytest.approx(0.0)
    assert "month_limit" in diagnostics["reasons"]

    # Pair form: the anchor is the first sample of the LAST sample's UTC month.
    pairs = [
        ("2026-01-20T00:00:00Z", 1.30),
        ("2026-02-01T00:00:00Z", 1.20),
        ("2026-02-10T00:00:00Z", 1.02),
    ]
    scale, diagnostics = kill_switch_scale(pairs, ladder="", month_loss_limit=0.10)
    assert diagnostics["month_loss"] == pytest.approx(0.15)
    assert scale == pytest.approx(0.0)


def test_kill_switch_scale_equity_ma_filter_and_peak_override() -> None:
    curve = [1.06, 1.05, 1.04, 1.03, 1.02]
    scale, diagnostics = kill_switch_scale(
        curve,
        ladder="",
        month_loss_limit=0.0,
        consecutive_loss_halving_from=0,
        equity_ma_window=5,
        below_ma_scale=0.5,
    )
    assert diagnostics["equity_ma"] == pytest.approx(1.04)
    assert diagnostics["ma_scale"] == pytest.approx(0.5)
    assert scale == pytest.approx(0.5)
    assert diagnostics["reason"] == "below_equity_ma"

    # A running peak outside the retained window still drives the ladder.
    scale, diagnostics = kill_switch_scale(
        curve, month_loss_limit=0.0, consecutive_loss_halving_from=0, peak=1.25
    )
    assert diagnostics["peak"] == pytest.approx(1.25)
    assert diagnostics["drawdown"] == pytest.approx(1.0 - 1.02 / 1.25)
    assert scale == pytest.approx(0.25)


def test_kill_switch_scale_components_multiply_and_kill_beats_min_scale() -> None:
    scale, diagnostics = kill_switch_scale(
        [1.0, 0.90, 0.88],
        ladder=DEFAULT_LADDER,
        month_loss_limit=0.0,
        loss_streak=3,
        equity_ma_window=0,
    )
    # ladder(0.12) == 0.5, streak(3) == 0.5 -> 0.25
    assert diagnostics["ladder_scale"] == pytest.approx(0.5)
    assert diagnostics["streak_scale"] == pytest.approx(0.5)
    assert scale == pytest.approx(0.25)
    assert diagnostics["reasons"] == ["drawdown_ladder", "loss_streak"]

    # A ladder 0-rung is a KILL: min_scale must not lift it back off zero.
    scale, _ = kill_switch_scale([1.0, 0.70], month_loss_limit=0.0, min_scale=0.10)
    assert scale == pytest.approx(0.0)


def test_kill_switch_scale_never_levers_above_one() -> None:
    scale, _ = kill_switch_scale([1.0, 1.5, 2.0], month_loss_limit=0.0)
    assert scale == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# (b) overlay: ladder trim, kill, suppression, recovery
# --------------------------------------------------------------------------- #


def test_ladder_trim_scales_the_next_child_entry() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0, 12}, "target_allocation": 1.0, "max_order_value": 400.0},
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _decline(13))

    first = next(sig for idx, sig in emitted if idx == 0)
    assert first.signal_type == "LONG"
    assert first.metadata["overlay_scale"] == pytest.approx(1.0)
    assert first.metadata["target_allocation"] == pytest.approx(1.0)

    # bar 12 close is 94.0 -> proxy equity 0.94 -> 6% drawdown -> the 0.05 rung.
    trimmed = next(sig for idx, sig in emitted if idx == 12)
    assert trimmed.signal_type == "LONG"
    assert trimmed.metadata["overlay_scale"] == pytest.approx(0.75)
    assert trimmed.metadata["target_allocation"] == pytest.approx(0.75)
    assert trimmed.metadata["max_order_value"] == pytest.approx(300.0)
    assert trimmed.strength == pytest.approx(0.75)
    assert trimmed.metadata["proxy_drawdown"] == pytest.approx(0.06)
    assert trimmed.metadata["overlay_reason"] == "drawdown_ladder"
    # max_symbol_exposure_pct is a min-clamp cap and must NOT be rescaled.
    assert trimmed.metadata["max_symbol_exposure_pct"] == pytest.approx(1.0)


def test_kill_emits_one_exit_and_suppresses_further_entries() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0, 42, 44}, "target_allocation": 1.0},
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _decline(46))

    kill_exits = [(idx, sig) for idx, sig in emitted if sig.metadata.get("kill_switch")]
    assert len(kill_exits) == 1, "the kill EXIT must be emitted exactly once per episode"
    kill_idx, kill_exit = kill_exits[0]
    # close 80.0 at bar 40 -> proxy equity 0.80 -> the 0.20 rung is the kill rung.
    assert kill_idx == 40
    assert kill_exit.signal_type == "EXIT"
    assert kill_exit.symbol == _SYMBOL
    assert kill_exit.metadata["overlay_scale"] == pytest.approx(0.0)
    assert kill_exit.metadata["proxy_drawdown"] == pytest.approx(0.20)

    assert not [idx for idx, sig in emitted if sig.signal_type == "LONG" and idx >= 40]
    assert strategy._overlay.killed is True
    assert strategy._overlay.kill_source == "ladder"


def test_kill_recovers_and_re_enables_entries() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0, 58}, "target_allocation": 1.0},
        month_loss_limit=0.0,
    )
    prices = _decline(45)  # bars 0..44, closing at 78.0 (22% proxy drawdown)
    prices += [78.0 + 1.5 * (i + 1) for i in range(15)]  # bars 45..59 recover
    emitted = _run(strategy, prices)

    assert [idx for idx, sig in emitted if sig.metadata.get("kill_switch")] == [40]
    reopened = [(idx, sig) for idx, sig in emitted if sig.signal_type == "LONG" and idx > 40]
    assert len(reopened) == 1, "the child entry must be forwarded again after recovery"
    idx, signal = reopened[0]
    assert idx == 58
    # bar 58 close is 99.0 -> 1% drawdown -> no rung breached.
    assert signal.metadata["overlay_scale"] == pytest.approx(1.0)
    assert strategy._overlay.killed is False


def test_rerisk_hysteresis_delays_the_scale_increase() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={
            "long_at": {0},
            "probe_at": set(range(1, 20)),
            "target_allocation": 1.0,
        },
        month_loss_limit=0.0,
        rerisk_min_bars=5,
    )
    # Flat, one 6% dip at bar 11, then straight back: the de-risk lands on bar 11
    # and the re-risk must wait ``rerisk_min_bars`` bars past it.
    prices = [100.0] * 11 + [94.0] + [100.0] * 8
    emitted = _run(strategy, prices)
    scale_by_bar = {
        idx: sig.metadata["overlay_scale"] for idx, sig in emitted if sig.signal_type == "LONG"
    }

    assert scale_by_bar[10] == pytest.approx(1.0)
    assert scale_by_bar[11] == pytest.approx(0.75), "de-risk must apply on the breach bar"
    for bar in (12, 13, 14, 15):
        assert scale_by_bar[bar] == pytest.approx(0.75), "re-risk must be held back"
    assert scale_by_bar[16] == pytest.approx(1.0), "re-risk after rerisk_min_bars bars"


# --------------------------------------------------------------------------- #
# (c) EXIT passthrough
# --------------------------------------------------------------------------- #


def test_child_exit_is_forwarded_unscaled() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0}, "exit_at": {12, 42}, "target_allocation": 1.0},
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _decline(46))

    trimmed_exit = next(
        sig for idx, sig in emitted if idx == 12 and not sig.metadata.get("kill_switch")
    )
    assert trimmed_exit.signal_type == "EXIT"
    assert trimmed_exit.strength == pytest.approx(1.0)
    assert "overlay_scale" not in trimmed_exit.metadata

    # The book is flat after bar 12, so the ladder never reaches its kill rung;
    # the second child EXIT is still forwarded verbatim.
    late_exit = next(sig for idx, sig in emitted if idx == 42)
    assert late_exit.signal_type == "EXIT"
    assert late_exit.strength == pytest.approx(1.0)
    assert "overlay_scale" not in late_exit.metadata


def test_child_exit_is_forwarded_while_the_overlay_is_killed() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0}, "exit_at": {42}, "target_allocation": 1.0},
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _decline(46))
    assert strategy._overlay.killed is True
    child_exit = next(
        sig for idx, sig in emitted if idx == 42 and not sig.metadata.get("kill_switch")
    )
    assert child_exit.signal_type == "EXIT"
    assert child_exit.strength == pytest.approx(1.0)


def test_child_exit_queued_before_a_child_exception_is_still_forwarded() -> None:
    # Regression: draining the capture queue on a child exception threw away the
    # signals the child had ALREADY queued -- including de-risking EXITs, which
    # would strand an open position.  A crash may cancel new risk, never an exit.
    events = _Queue()
    strategy = _make(
        events,
        child_params={
            "long_at": {0, 3},
            "exit_at": {3},
            "raise_after": {3},
            "target_allocation": 1.0,
        },
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _decline(5))

    crash_bar = [sig for idx, sig in emitted if idx == 3]
    assert len(crash_bar) == 1, "only the EXIT survives the crash; the entry is discarded"
    salvaged = crash_bar[0]
    assert salvaged.signal_type == "EXIT"
    assert salvaged.symbol == _SYMBOL
    assert salvaged.strength == pytest.approx(1.0), "EXITs are forwarded unscaled"
    assert "overlay_scale" not in salvaged.metadata
    # The shadow book must reflect the flattening the EXIT just performed.
    assert strategy._overlay.weights[_SYMBOL] == pytest.approx(0.0)


def test_child_exception_salvages_exits_from_every_dispatcher() -> None:
    child_params = {"long_at": {0}, "exit_at": {1}, "raise_after": {1}, "target_allocation": 1.0}

    events_market = _Queue()
    market = _make(events_market, child_params=dict(child_params), month_loss_limit=0.0)
    market.calculate_signals(_market_event(0, 100.0))
    market.calculate_signals(_market_event(1, 99.0))

    events_window = _Queue()
    window = _make(events_window, child_params=dict(child_params), month_loss_limit=0.0)
    window.calculate_signals_window(_market_event(0, 100.0), None)
    window.calculate_signals_window(_market_event(1, 99.0), None)

    events_context = _Queue()
    context = _make(events_context, child_params=dict(child_params), month_loss_limit=0.0)
    context.calculate_signals_context(SimpleNamespace(event=_market_event(0, 100.0)))
    context.calculate_signals_context(SimpleNamespace(event=_market_event(1, 99.0)))

    for queue in (events_market, events_window, events_context):
        assert [sig.signal_type for sig in queue.items] == ["LONG", "EXIT"]
        assert queue.items[-1].strength == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# overlay: SHORT sign convention in the shadow book
# --------------------------------------------------------------------------- #


def _short_prices(count: int, *, step: float) -> list[float]:
    return [100.0 + step * i for i in range(count)]


def test_short_intent_gains_on_a_falling_path_and_never_de_risks() -> None:
    # Sign regression: a SHORT must enter the shadow book with a NEGATIVE weight.
    # With the sign flipped the falling path below would read as a loss and the
    # ladder would trim (and eventually kill) a position that is actually up.
    events = _Queue()
    strategy = _make(
        events,
        child_params={
            "short_at": {0},
            "probe_at": set(range(1, 26)),
            "target_allocation": 1.0,
        },
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _short_prices(26, step=-1.0))

    assert strategy._overlay.weights[_SYMBOL] == pytest.approx(-1.0)
    assert strategy._overlay.equity > 1.0
    assert strategy._overlay.killed is False
    assert strategy._overlay.effective_scale == pytest.approx(1.0)
    assert not [sig for _, sig in emitted if sig.metadata.get("kill_switch")]
    scales = [sig.metadata["overlay_scale"] for _, sig in emitted if sig.signal_type == "LONG"]
    assert scales and all(scale == pytest.approx(1.0) for scale in scales)
    assert strategy._last_diagnostics["drawdown"] == pytest.approx(0.0)


def test_short_intent_loses_on_a_rising_path_and_walks_down_the_ladder() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={
            "short_at": {0},
            "probe_at": set(range(1, 26)),
            "target_allocation": 1.0,
        },
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _short_prices(26, step=1.0))

    assert strategy._overlay.equity < 1.0
    scale_by_bar = {
        idx: sig.metadata["overlay_scale"] for idx, sig in emitted if sig.signal_type == "LONG"
    }
    # proxy equity 0.951923 at bar 5 (4.8% down) -> nothing breached yet.
    assert scale_by_bar[5] == pytest.approx(1.0)
    # 0.908257 at bar 10 (9.2% down) -> the 0.05 rung.
    assert scale_by_bar[10] == pytest.approx(0.75)
    # 0.846154 at bar 18 (15.4% down) -> the 0.15 rung.
    assert scale_by_bar[18] == pytest.approx(0.25)
    # 0.798387 at bar 25 (20.2% down) -> the terminal 0-rung is a KILL.
    kill_exits = [(idx, sig) for idx, sig in emitted if sig.metadata.get("kill_switch")]
    assert len(kill_exits) == 1
    assert kill_exits[0][0] == 25
    assert kill_exits[0][1].signal_type == "EXIT"
    assert strategy._overlay.killed is True
    assert strategy._overlay.kill_source == "ladder"


# --------------------------------------------------------------------------- #
# overlay: streak / month / MA components end-to-end
# --------------------------------------------------------------------------- #


def test_consecutive_losing_trades_halve_the_next_entry() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={
            "long_at": {0, 4, 8, 12},
            "exit_at": {2, 6, 10},
            "target_allocation": 1.0,
        },
        ladder="",
        month_loss_limit=0.0,
    )
    emitted = _run(strategy, _decline(13, step=1.0))

    assert strategy._overlay.loss_streak == 3
    third = next(sig for idx, sig in emitted if idx == 8 and sig.signal_type == "LONG")
    assert third.metadata["overlay_scale"] == pytest.approx(1.0), "streak 2 is below the trigger"
    fourth = next(sig for idx, sig in emitted if idx == 12 and sig.signal_type == "LONG")
    assert fourth.metadata["overlay_scale"] == pytest.approx(0.5)
    assert fourth.metadata["overlay_reason"] == "loss_streak"


def test_winning_trade_resets_the_losing_streak() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0, 4, 8}, "exit_at": {2, 6, 10}, "target_allocation": 1.0},
        ladder="",
        month_loss_limit=0.0,
    )
    prices = [100.0, 99.0, 98.0, 98.0, 98.0, 99.0, 100.0, 100.0, 100.0, 99.0, 98.0]
    _run(strategy, prices)
    # trades: bar0->bar2 loss, bar4->bar6 win (resets), bar8->bar10 loss.
    assert strategy._overlay.loss_streak == 1


def test_month_loss_limit_kills_until_the_next_calendar_month() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0, 20, 40}, "target_allocation": 1.0},
        ladder="",
        month_loss_limit=0.10,
    )
    # The month anchor is bar 0 (equity 1.0); bar 13 closes at 89.6, a 10.4%
    # month loss, which is the first bar at or beyond the 10% limit.
    prices = [100.0 - 0.8 * i for i in range(16)]
    prices += [88.0] * 30  # flat through the month roll at bar 31
    emitted = _run(strategy, prices)

    kill_exits = [(idx, sig) for idx, sig in emitted if sig.metadata.get("kill_switch")]
    assert len(kill_exits) == 1
    assert kill_exits[0][0] == 13
    assert not [idx for idx, sig in emitted if sig.signal_type == "LONG" and idx == 20]
    # 2026-01-01 + 31 days == 2026-02-01: the month roll lifts the kill.
    reopened = next(sig for idx, sig in emitted if idx == 40 and sig.signal_type == "LONG")
    assert reopened.metadata["overlay_scale"] == pytest.approx(1.0)
    assert strategy._overlay.month_key == "2026-02"


def test_month_kill_source_is_labelled_month_even_with_an_active_ladder() -> None:
    # Regression: a 0.0 component is falsy.  Reading it through ``or`` labelled a
    # month kill as a LADDER kill, which then waited for a drawdown recovery the
    # month rule never promises -- the overlay stayed dead past the month roll.
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0}, "probe_at": {20, 40}, "target_allocation": 1.0},
        month_loss_limit=0.10,
    )
    # -10.4% by bar 13: past the month limit but only on the ladder's 0.10 rung,
    # so the ONLY zero component is the month limit.
    prices = [100.0 - 0.8 * i for i in range(14)] + [89.6] * 32
    emitted = _run(strategy, prices)

    kill_exit = next(sig for _, sig in emitted if sig.metadata.get("kill_switch"))
    assert strategy._overlay.month_key == "2026-02"
    assert "kill_switch_month" in kill_exit.metadata["overlay_reason"]
    assert not [idx for idx, sig in emitted if sig.signal_type == "LONG" and idx == 20]
    # After the month roll the ladder alone governs: 10.4% drawdown -> 0.10 rung.
    reopened = next(sig for idx, sig in emitted if idx == 40 and sig.signal_type == "LONG")
    assert reopened.metadata["overlay_scale"] == pytest.approx(0.5)
    assert strategy._overlay.killed is False


def test_equity_ma_filter_halves_the_entry_below_the_curve_average() -> None:
    events = _Queue()
    strategy = _make(
        events,
        child_params={"long_at": {0, 12}, "target_allocation": 1.0},
        ladder="",
        month_loss_limit=0.0,
        equity_ma_window=5,
        below_ma_scale=0.5,
    )
    prices = [100.0 + i for i in range(8)] + [106.0, 105.0, 104.0, 103.0, 102.0]
    emitted = _run(strategy, prices)

    signal = next(sig for idx, sig in emitted if idx == 12 and sig.signal_type == "LONG")
    assert signal.metadata["overlay_scale"] == pytest.approx(0.5)
    assert signal.metadata["overlay_reason"] == "below_equity_ma"


# --------------------------------------------------------------------------- #
# (d) state round-trip, degenerate input, wiring
# --------------------------------------------------------------------------- #


def test_get_set_state_round_trip_preserves_behaviour() -> None:
    child_params = {"long_at": {0, 12, 25}, "exit_at": {6}, "target_allocation": 1.0}
    prices = _decline(30)

    events_a = _Queue()
    strategy_a = _make(events_a, child_params=dict(child_params), month_loss_limit=0.0)
    _run(strategy_a, prices[:20])
    snapshot = strategy_a.get_state()
    assert snapshot["overlay"]["equity_curve"]
    assert snapshot["child"]["bar_index"] == 19

    events_b = _Queue()
    strategy_b = _make(events_b, child_params=dict(child_params), month_loss_limit=0.0)
    strategy_b.set_state(snapshot)
    assert list(strategy_b._overlay.equity_curve) == list(strategy_a._overlay.equity_curve)
    assert strategy_b._overlay.weights == strategy_a._overlay.weights
    assert strategy_b._overlay.open_trades == strategy_a._overlay.open_trades
    assert strategy_b._overlay.loss_streak == strategy_a._overlay.loss_streak
    assert strategy_b._overlay.effective_scale == pytest.approx(strategy_a._overlay.effective_scale)

    # Both instances now see the same remaining bars and must agree exactly.
    tail_a = _run(strategy_a, prices[20:], start=20)
    tail_b = _run(strategy_b, prices[20:], start=20)
    assert len(tail_a) == len(tail_b) and tail_a
    for (_, sig_a), (_, sig_b) in zip(tail_a, tail_b, strict=True):
        assert sig_a.signal_type == sig_b.signal_type
        assert sig_a.symbol == sig_b.symbol
        assert sig_a.metadata == sig_b.metadata
        assert sig_a.strength == pytest.approx(sig_b.strength)


def test_set_state_ignores_malformed_payloads() -> None:
    events = _Queue()
    strategy = _make(events, month_loss_limit=0.0)
    _run(strategy, _decline(5))
    before = list(strategy._overlay.equity_curve)
    strategy.set_state({})
    strategy.set_state({"overlay": "not-a-dict"})
    strategy.set_state([])  # type: ignore[arg-type]
    assert list(strategy._overlay.equity_curve) == before


def test_overlay_never_raises_and_drops_signals_on_child_failure() -> None:
    events = _Queue()
    strategy = _make(events, child_params={"long_at": {0}, "raises": True})
    strategy.calculate_signals(_market_event(0, 100.0))
    strategy.calculate_signals_window(_market_event(1, 99.0), None)
    strategy.calculate_signals_context(SimpleNamespace(event=_market_event(2, 98.0)))
    assert events.items == []


def test_no_scaling_before_any_bar_history() -> None:
    events = _Queue()
    strategy = _make(events, child_params={"long_at": {0}, "target_allocation": 0.25})
    # A bar with no usable close: no equity sample, no crash, child still runs.
    strategy.calculate_signals(SimpleNamespace(type="MARKET", time=_stamp(0), symbol=_SYMBOL))
    assert list(strategy._overlay.equity_curve) == []
    assert len(events.items) == 1
    assert events.items[0].metadata["target_allocation"] == pytest.approx(0.25)


def test_decision_cadence_and_contract_are_inherited_from_the_child() -> None:
    events = _Queue()
    strategy = _make(events)
    assert strategy.decision_cadence_seconds == 86400
    assert strategy.preferred_contract == "market_window"
    assert strategy.symbol_list == [_SYMBOL]


def test_child_strategy_alias_param_resolves_the_child() -> None:
    events = _Queue()
    bars = SimpleNamespace(symbol_list=[_SYMBOL])
    strategy = EquityCurveKillSwitchOverlayStrategy(
        bars, events, child_strategy="ProbeChild", child_params={"long_at": {0}}
    )
    assert strategy.child_strategy_class == "ProbeChild"


def test_param_schema_defaults() -> None:
    schema = EquityCurveKillSwitchOverlayStrategy.get_param_schema()
    assert schema["ladder"].default == DEFAULT_LADDER
    assert schema["consecutive_loss_halving_from"].default == 3
    assert schema["month_loss_limit"].default == pytest.approx(0.10)
    assert schema["equity_ma_window"].default == 0
    assert schema["below_ma_scale"].default == pytest.approx(0.5)
    assert schema["rerisk_min_bars"].default == 5
    assert schema["recover_drawdown"].default == pytest.approx(0.10)
    assert schema["min_scale"].default == pytest.approx(0.10)
    assert schema["child_strategy_class"].tunable is False
    assert schema["ladder"].tunable is False


# --------------------------------------------------------------------------- #
# (e) registry
# --------------------------------------------------------------------------- #


def test_strategy_is_registered() -> None:
    assert (
        GLOBAL_REGISTRY.get("strategy", "EquityCurveKillSwitchOverlayStrategy")
        is EquityCurveKillSwitchOverlayStrategy
    )
    assert "_OverlayProbeChild" not in GLOBAL_REGISTRY.list_names("strategy")
    assert "ProbeChild" not in GLOBAL_REGISTRY.list_names("strategy")
