from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.core.events import SignalEvent
from lumina_quant.strategies import vol_managed_risk_overlay as MODULE
from lumina_quant.strategies.registry import (
    get_live_strategy_map,
    get_strategy_map,
    get_strategy_tier,
)
from lumina_quant.strategies.vol_managed_risk_overlay import VolManagedRiskOverlayStrategy

_BENCH = "BTC/USDT"


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every child class to the scripted stub (registry has no stub)."""
    monkeypatch.setattr(
        MODULE, "resolve_strategy_class", lambda name, default_name=None: _ScriptedChild
    )


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _ScriptedChild:
    """Child stub that emits a fixed batch of signals on each event.

    ``required_features``/``preferred_contract`` mirror a real sleeve so the
    overlay's requirement propagation can be exercised; ``calculate_signals``
    re-puts a deep copy of the scripted signals so the overlay can rescale them.
    """

    required_features = ("funding_rate",)
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    decision_cadence_seconds = 60

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        self._scripted: list[SignalEvent] = list(params.get("scripted") or [])
        self._state: dict[str, Any] = {}

    def calculate_signals(self, event: Any) -> None:
        for signal in self._scripted:
            self.events.put(_clone_signal(signal))

    def get_state(self) -> dict[str, Any]:
        return {"marker": dict(self._state)}

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state.get("marker") or {})


def _clone_signal(signal: SignalEvent) -> SignalEvent:
    return SignalEvent(
        strategy_id=signal.strategy_id,
        symbol=signal.symbol,
        datetime=signal.datetime,
        signal_type=signal.signal_type,
        strength=signal.strength,
        price=signal.price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        position_side=signal.position_side,
        client_order_id=signal.client_order_id,
        time_in_force=signal.time_in_force,
        metadata=dict(signal.metadata or {}),
        trailing_percent=signal.trailing_percent,
    )


def _signal(
    symbol: str,
    idx: int,
    signal_type: str = "LONG",
    *,
    target_allocation: float = 0.20,
    max_order_value: float = 500.0,
    strength: float = 1.0,
) -> SignalEvent:
    metadata: dict[str, Any] = {"strategy": "child"}
    if target_allocation > 0.0:
        metadata["target_allocation"] = target_allocation
        metadata["max_symbol_exposure_pct"] = target_allocation
    if max_order_value > 0.0:
        metadata["max_order_value"] = max_order_value
    return SignalEvent(
        strategy_id="child::scripted",
        symbol=symbol,
        datetime=f"2026-05-02T00:{idx % 60:02d}:00Z",
        signal_type=signal_type,
        strength=strength,
        metadata=metadata,
    )


def _market_event(idx: int, close: float, *, symbol: str = _BENCH) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T00:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def _make(
    scripted: list[SignalEvent], events: _Queue, **overrides: Any
) -> VolManagedRiskOverlayStrategy:
    bars = SimpleNamespace(
        symbol_list=[_BENCH],
        get_latest_feature_value=lambda *a, **k: None,
        get_latest_bar_value=lambda *a, **k: None,
    )
    params: dict[str, Any] = {
        "child_strategy_class": "ScriptedChild",
        "child_params": {"scripted": scripted},
        "child_symbols": [_BENCH],
    }
    params.update(overrides)
    return VolManagedRiskOverlayStrategy(bars, events, **params)


def _calm_series(n: int, level: float = 100.0) -> list[float]:
    """Constant series -> realized vol == 0 -> vol clamp pins at max_scale."""
    return [level for _ in range(n)]


# ---------------------------------------------------------------- (1) calm


def test_calm_regime_scale_one_byte_identical() -> None:
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make(
        [base],
        events,
        target_vol_per_bar=0.01,
        min_scale=0.25,
        max_scale=1.0,
        rebalance_band=0.0,
    )
    for i, close in enumerate(_calm_series(80)):
        events.items.clear()
        strategy.calculate_signals(_market_event(i, close))

    assert len(events.items) == 1
    out = events.items[0]
    # Calm (zero realized vol) -> vol clamp == max_scale == 1.0; crash gate == 1.0.
    assert out.metadata["overlay_scale"] == pytest.approx(1.0)
    assert out.metadata["target_allocation"] == pytest.approx(base.metadata["target_allocation"])
    assert out.metadata["max_order_value"] == pytest.approx(base.metadata["max_order_value"])
    assert out.metadata["max_symbol_exposure_pct"] == pytest.approx(
        base.metadata["max_symbol_exposure_pct"]
    )
    assert out.strength == pytest.approx(base.strength)


# ------------------------------------------------------------- (2) vol spike


def _volatile_series(n: int, level: float = 100.0, amp: float = 0.15) -> list[float]:
    return [level * (1.0 + amp * (1.0 if i % 2 == 0 else -1.0)) for i in range(n)]


def test_vol_spike_scales_toward_min_and_exit_unscaled() -> None:
    events = _Queue()
    longs = _signal(_BENCH, 0, "LONG")
    exit_sig = _signal("ETH/USDT", 0, "EXIT", target_allocation=0.0, max_order_value=0.0)
    strategy = _make(
        [longs, exit_sig],
        events,
        target_vol_per_bar=0.001,
        vol_window=20,
        min_scale=0.25,
        max_scale=1.0,
        rebalance_band=0.0,
        crash_scale=0.0,
    )
    for i, close in enumerate(_volatile_series(80)):
        events.items.clear()
        strategy.calculate_signals(_market_event(i, close))

    long_out = next(s for s in events.items if s.signal_type == "LONG")
    exit_out = next(s for s in events.items if s.signal_type == "EXIT")
    # High realized vol with a tiny target -> clamp collapses to min_scale.
    assert long_out.metadata["overlay_scale"] == pytest.approx(0.25)
    assert long_out.metadata["target_allocation"] == pytest.approx(0.20 * 0.25)
    assert long_out.strength == pytest.approx(0.25)
    # EXIT carries no notional metadata and is NOT rescaled.
    assert "overlay_scale" not in exit_out.metadata
    assert exit_out.strength == pytest.approx(1.0)


# ----------------------------------------------------------- (3) crash gate


def _crash_series(n: int) -> list[float]:
    """Calm ramp to a peak, then a SHORT sharp drawdown with churning vol.

    The calm prefix prints near-zero realized vol so the trailing vol-history is
    low; the short, large-swing crash tail spikes recent realized vol well above
    that baseline (vol-z elevated) while also pushing the price > 20% below the
    peak (deep drawdown) -- the two conditions of the conjunction together.
    """
    out: list[float] = []
    crash_len = 12
    calm_len = n - crash_len
    for i in range(calm_len):
        out.append(100.0 + 0.1 * i)  # gentle ramp -> peak ~ 110, tiny vol
    peak = out[-1]
    for j in range(crash_len):
        base = peak * 0.6  # ~ -40% from peak -> deep drawdown
        out.append(base * (1.0 + 0.15 * (1.0 if j % 2 == 0 else -1.0)))
    return out


def _drawdown_only_series(n: int) -> list[float]:
    """Deep drawdown but SMOOTH (no vol-z elevation) -> gate must NOT fire.

    A gentle monotone decline reaches a deep drawdown without any return spike,
    so trailing realized vol stays flat and the vol-z never elevates.
    """
    out: list[float] = []
    calm_len = n // 2
    for i in range(calm_len):
        out.append(100.0 + 0.1 * i)
    peak = out[-1]
    # Smooth geometric glide down to ~ -45% over the remaining bars.
    steps = n - calm_len
    ratio = (peak * 0.55 / peak) ** (1.0 / steps)
    price = peak
    for _ in range(steps):
        price *= ratio
        out.append(price)
    return out


def _run(series: list[float], **overrides: Any) -> SignalEvent:
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make([base], events, rebalance_band=0.0, **overrides)
    for i, close in enumerate(series):
        events.items.clear()
        strategy.calculate_signals(_market_event(i, close))
    return events.items[0]


def test_crash_gate_fires_only_under_conjunction() -> None:
    crash_kwargs = dict(
        target_vol_per_bar=1.0,  # keep the vol clamp pinned at max so we isolate the gate
        max_scale=1.0,
        min_scale=0.25,
        crash_vol_window=10,
        crash_vol_z_window=20,
        crash_drawdown_window=80,
        crash_drawdown_threshold=-0.20,
        crash_vol_z_threshold=0.5,
        crash_scale=0.0,
    )
    # Conjunction: deep drawdown AND elevated vol-z together -> gate to crash_scale.
    crashed = _run(_crash_series(80), **crash_kwargs)
    assert crashed.metadata["overlay_scale"] == pytest.approx(0.0)

    # Drawdown alone (smooth bottom, no vol-z) -> gate stays at 1.0.
    dd_only = _run(_drawdown_only_series(80), **crash_kwargs)
    assert dd_only.metadata["overlay_scale"] == pytest.approx(1.0)

    # Vol-z alone (volatile but shallow -- never > 20% below peak) -> gate at 1.0.
    vol_only = _run(_volatile_series(80, amp=0.05), **crash_kwargs)
    assert vol_only.metadata["overlay_scale"] == pytest.approx(1.0)


# -------------------------------------------------------- (4) rebalance_band


def test_rebalance_band_suppresses_sub_band_reemit() -> None:
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make(
        [base],
        events,
        target_vol_per_bar=0.01,
        max_scale=1.0,
        min_scale=0.25,
        crash_scale=0.0,
        rebalance_band=0.50,
    )
    # Calm throughout -> scale constant at 1.0; only the first emit should pass,
    # subsequent identical multipliers are suppressed by the band.
    emitted = 0
    for i, close in enumerate(_calm_series(60)):
        events.items.clear()
        strategy.calculate_signals(_market_event(i, close))
        emitted += len(events.items)
    assert emitted == 1


def test_rebalance_band_allows_supra_band_move() -> None:
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make(
        [base],
        events,
        target_vol_per_bar=0.001,
        vol_window=10,
        max_scale=1.0,
        min_scale=0.25,
        crash_scale=0.0,
        rebalance_band=0.10,
    )
    # Calm prefix (scale ~1.0), then a volatile tail (scale collapses to 0.25):
    # the >0.10 move must re-emit.
    series = _calm_series(20) + _volatile_series(40)
    scales: list[float] = []
    for i, close in enumerate(series):
        events.items.clear()
        strategy.calculate_signals(_market_event(i, close))
        scales.extend(s.metadata["overlay_scale"] for s in events.items)
    assert any(s == pytest.approx(1.0) for s in scales)
    assert any(s == pytest.approx(0.25) for s in scales)


def test_band_does_not_suppress_changed_child_intent() -> None:
    # Regression: the band must only suppress a REDUNDANT re-size of the SAME
    # child intent — a changed child trade (different target_allocation or a side
    # flip) must always be forwarded even when the overlay scale is unchanged.
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG", target_allocation=0.20)
    strategy = _make(
        [base],
        events,
        target_vol_per_bar=0.01,
        max_scale=1.0,
        min_scale=0.25,
        crash_scale=0.0,
        rebalance_band=0.50,
    )
    series = _calm_series(60)  # calm => overlay scale constant; isolates intent logic
    for i in range(40):
        strategy.calculate_signals(_market_event(i, series[i]))
    # (a) same side, changed target_allocation -> must forward (not suppressed).
    strategy._child._scripted = [_signal(_BENCH, 41, "LONG", target_allocation=0.40)]
    events.items.clear()
    strategy.calculate_signals(_market_event(41, series[41]))
    assert len(events.items) == 1, "changed child target_allocation must bypass the band"
    # (b) side flip -> must forward.
    strategy._child._scripted = [_signal(_BENCH, 42, "SHORT", target_allocation=0.40)]
    events.items.clear()
    strategy.calculate_signals(_market_event(42, series[42]))
    assert any(s.signal_type == "SHORT" for s in events.items), "side flip must bypass the band"


# ----------------------------------------------------------- (5) state I/O


def test_get_set_state_round_trip_includes_child() -> None:
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make([base], events, target_vol_per_bar=0.01, rebalance_band=0.0)
    for i, close in enumerate(_calm_series(40)):
        strategy.calculate_signals(_market_event(i, close))
    strategy._child._state = {"hello": 7}

    snapshot = strategy.get_state()
    assert snapshot["child"]["marker"] == {"hello": 7}
    assert snapshot["overlay"]["closes"]
    assert "last_time_key" in snapshot["overlay"]

    restored_events = _Queue()
    restored = _make([base], restored_events, target_vol_per_bar=0.01, rebalance_band=0.0)
    restored.set_state(snapshot)
    assert restored._child._state == {"hello": 7}
    assert list(restored._overlay.closes) == list(strategy._overlay.closes)
    assert list(restored._overlay.vol_history) == list(strategy._overlay.vol_history)
    assert restored._overlay.last_time_key == strategy._overlay.last_time_key
    assert restored._last_forward == strategy._last_forward


# --------------------------------------------------------- (6) no-lookahead


def test_no_lookahead_scale_uses_no_future_bar() -> None:
    base = _signal(_BENCH, 0, "LONG")
    series = _volatile_series(80)

    # Run A: feed the full series.
    events_a = _Queue()
    strat_a = _make([base], events_a, target_vol_per_bar=0.001, vol_window=20, rebalance_band=0.0)
    scale_at_t: float | None = None
    cutoff = 50
    for i, close in enumerate(series):
        events_a.items.clear()
        strat_a.calculate_signals(_market_event(i, close))
        if i == cutoff:
            scale_at_t = events_a.items[0].metadata["overlay_scale"]

    # Run B: feed ONLY up to and including bar ``cutoff`` (no future bars exist).
    events_b = _Queue()
    strat_b = _make([base], events_b, target_vol_per_bar=0.001, vol_window=20, rebalance_band=0.0)
    scale_truncated: float | None = None
    for i, close in enumerate(series[: cutoff + 1]):
        events_b.items.clear()
        strat_b.calculate_signals(_market_event(i, close))
        if i == cutoff:
            scale_truncated = events_b.items[0].metadata["overlay_scale"]

    assert scale_at_t is not None and scale_truncated is not None
    assert scale_at_t == pytest.approx(scale_truncated)


# --------------------------------------------------- (7) never raises


def test_never_raises_on_empty_or_none() -> None:
    events = _Queue()
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make([base], events)
    # None event, event with missing/None close, empty market window.
    strategy.calculate_signals(SimpleNamespace(type="MARKET", time=None, symbol=None, close=None))
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", time="2026-05-02T00:01:00Z", bars_1s={})
    )
    strategy.calculate_signals(_market_event(2, float("nan")))
    # The dispatcher swallows child explosions without propagating.

    class _Boom(_ScriptedChild):
        def calculate_signals(self, event: Any) -> None:
            raise RuntimeError("child blew up")

    strategy._child = _Boom(strategy.bars, strategy._child_queue)
    strategy.calculate_signals(_market_event(3, 100.0))  # must not raise


# ------------------------------------------------ (8) tier / live exclusion


def test_tier_is_research_only_and_excluded_from_live_maps() -> None:
    name = "VolManagedRiskOverlayStrategy"
    assert name in get_strategy_map()
    assert get_strategy_tier(name) == "research_only"
    assert name not in get_live_strategy_map(include_opt_in=True)
    assert name not in get_live_strategy_map(include_opt_in=False)


def test_module_marker_constants() -> None:
    assert MODULE._STRATEGY_NAME == "VolManagedRiskOverlayStrategy"
    assert math.isfinite(MODULE._EPS)
