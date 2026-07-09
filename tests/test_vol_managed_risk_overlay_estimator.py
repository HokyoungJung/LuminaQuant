"""X1: config-gated ``vol_estimator`` param on ``VolManagedRiskOverlayStrategy``.

The default ``close_to_close`` estimator must reproduce the historical overlay
byte-for-byte (proven here + by the unmodified
``tests/unit/test_vol_managed_risk_overlay.py`` staying green). The opt-in
range estimators (Parkinson / Garman-Klass / Yang-Zhang) feed the vol-target
clamp from benchmark OHLC in their literature-endorsed SIZING role only -- no
directional logic changes. All estimators are pure Python/numpy (no
scipy/sklearn) and never raise on degenerate bars.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.core.events import SignalEvent
from lumina_quant.strategies import vol_managed_risk_overlay as MODULE
from lumina_quant.strategies.vol_managed_risk_overlay import VolManagedRiskOverlayStrategy

_BENCH = "BTC/USDT"


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every child class to the scripted stub (registry has no stub)."""
    monkeypatch.setattr(
        MODULE, "resolve_strategy_class", lambda name, default_name=None: _ScriptedChild
    )


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    """Deterministic uniform stream in [0, 1) (no ``random`` -> run-twice stable)."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _ScriptedChild:
    """Child stub that re-emits a fixed batch of signals on each event."""

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


def _ohlc_event(
    idx: int,
    close: float,
    high: float,
    low: float,
    *,
    open_px: float | None = None,
    symbol: str = _BENCH,
) -> SimpleNamespace:
    """A benchmark MARKET bar with an explicit intrabar range (distinct H/L)."""
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close if open_px is None else open_px,
        high=high,
        low=low,
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


def _wide_range_feed(n: int, *, seed: int, level: float = 100.0) -> list[SimpleNamespace]:
    """FLAT closes but a WIDE intrabar range: close-to-close vol == 0 while the
    range-based estimators see large per-bar volatility."""
    gen = _lcg_stream(seed)
    feed: list[SimpleNamespace] = []
    for idx in range(n):
        spread = level * (0.03 + 0.05 * next(gen))  # 3%-8% half-range
        feed.append(_ohlc_event(idx, level, high=level + spread, low=level - spread))
    return feed


def _drive_scales(strategy: VolManagedRiskOverlayStrategy, events: _Queue, feed) -> list[float]:
    """Feed the strategy bar-by-bar; return the LONG overlay_scale seen per bar."""
    scales: list[float] = []
    for event in feed:
        events.items.clear()
        strategy.calculate_signals(event)
        for out in events.items:
            if out.signal_type == "LONG":
                scales.append(float(out.metadata["overlay_scale"]))
    return scales


# --------------------------------------------------------------------------- #
# (a) DEFAULT byte-identity
# --------------------------------------------------------------------------- #


def test_default_matches_explicit_close_to_close_on_ohlc_feed() -> None:
    """Omitted ``vol_estimator`` == explicit ``close_to_close`` on an identical
    OHLC feed: every emitted signal AND the serialized state agree exactly."""
    base = _signal(_BENCH, 0, "LONG")
    feed = _wide_range_feed(70, seed=101)

    ev_default = _Queue()
    strat_default = _make(
        [base], ev_default, target_vol_per_bar=0.005, vol_window=20, rebalance_band=0.0
    )
    scales_default = _drive_scales(strat_default, ev_default, feed)

    ev_explicit = _Queue()
    strat_explicit = _make(
        [base],
        ev_explicit,
        vol_estimator="close_to_close",
        target_vol_per_bar=0.005,
        vol_window=20,
        rebalance_band=0.0,
    )
    scales_explicit = _drive_scales(strat_explicit, ev_explicit, feed)

    assert scales_default == scales_explicit
    assert strat_default.get_state() == strat_explicit.get_state()


def test_default_ignores_intrabar_range() -> None:
    """A wide-range / flat-close feed keeps the DEFAULT overlay pinned at
    max_scale -- proving OHL never leaks into the close_to_close path (a range
    estimator on the SAME feed would drop below max_scale, see test below)."""
    base = _signal(_BENCH, 0, "LONG")
    feed = _wide_range_feed(70, seed=7)
    events = _Queue()
    strategy = _make(
        [base],
        events,
        target_vol_per_bar=0.005,
        vol_window=20,
        min_scale=0.0,
        max_scale=1.0,
        rebalance_band=0.0,
    )
    scales = _drive_scales(strategy, events, feed)
    assert scales[-1] == pytest.approx(1.0)


def test_default_state_omits_ohl_keys() -> None:
    """The close_to_close snapshot carries NO OHL keys (byte-identical to the
    historical overlay state); a range estimator adds them."""
    base = _signal(_BENCH, 0, "LONG")
    events = _Queue()
    strat = _make([base], events, vol_window=20, rebalance_band=0.0)
    for event in _wide_range_feed(30, seed=3):
        strat.calculate_signals(event)
    overlay = strat.get_state()["overlay"]
    assert "opens" not in overlay and "highs" not in overlay and "lows" not in overlay

    events_r = _Queue()
    strat_r = _make([base], events_r, vol_estimator="parkinson", vol_window=20, rebalance_band=0.0)
    for event in _wide_range_feed(30, seed=3):
        strat_r.calculate_signals(event)
    overlay_r = strat_r.get_state()["overlay"]
    assert overlay_r["opens"] and overlay_r["highs"] and overlay_r["lows"]


def test_unknown_estimator_falls_back_to_close_to_close() -> None:
    """An unrecognized estimator name resolves to the byte-identical default."""
    base = _signal(_BENCH, 0, "LONG")
    events = _Queue()
    strat = _make([base], events, vol_estimator="not_a_real_estimator", vol_window=20)
    assert strat.vol_estimator == "close_to_close"
    assert strat._uses_range_estimator is False


# --------------------------------------------------------------------------- #
# (b) estimator-live
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("estimator", ["parkinson", "garman_klass", "yang_zhang"])
def test_range_estimator_scales_below_close_to_close(estimator: str) -> None:
    """Wide intrabar range with flat closes: range-vol > close-to-close vol, so
    the range estimator's scaled notional is LOWER (range-vol higher -> throttle
    harder). Direction is the load-bearing assertion."""
    base = _signal(_BENCH, 0, "LONG")
    feed = _wide_range_feed(70, seed=202)
    kwargs = dict(
        target_vol_per_bar=0.005,
        vol_window=20,
        min_scale=0.0,
        max_scale=1.0,
        rebalance_band=0.0,
    )

    ev_ctc = _Queue()
    ctc_scales = _drive_scales(_make([base], ev_ctc, **kwargs), ev_ctc, feed)

    ev_rng = _Queue()
    rng_scales = _drive_scales(
        _make([base], ev_rng, vol_estimator=estimator, **kwargs), ev_rng, feed
    )

    # Flat closes -> close_to_close vol is ~0 -> clamp pinned at max_scale.
    assert ctc_scales[-1] == pytest.approx(1.0)
    # Range estimator sees the wide bars -> measurably throttled below the clamp.
    assert rng_scales[-1] < ctc_scales[-1]
    assert rng_scales[-1] < 0.5


def test_range_estimator_notional_lower_than_close_to_close() -> None:
    """The throttle propagates to the child's scaled target_allocation."""
    base = _signal(_BENCH, 0, "LONG", target_allocation=0.20)
    feed = _wide_range_feed(70, seed=303)
    kwargs = dict(
        target_vol_per_bar=0.005,
        vol_window=20,
        min_scale=0.0,
        max_scale=1.0,
        rebalance_band=0.0,
    )

    ev_ctc = _Queue()
    strat_ctc = _make([base], ev_ctc, **kwargs)
    _drive_scales(strat_ctc, ev_ctc, feed)
    ctc_alloc = next(s for s in ev_ctc.items if s.signal_type == "LONG").metadata[
        "target_allocation"
    ]

    ev_rng = _Queue()
    strat_rng = _make([base], ev_rng, vol_estimator="parkinson", **kwargs)
    _drive_scales(strat_rng, ev_rng, feed)
    rng_alloc = next(s for s in ev_rng.items if s.signal_type == "LONG").metadata[
        "target_allocation"
    ]

    assert ctc_alloc == pytest.approx(0.20)  # flat closes -> full size
    assert rng_alloc < ctc_alloc  # range-vol throttles the notional down


# --------------------------------------------------------------------------- #
# (c) determinism / never-raise / state roundtrip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("estimator", ["parkinson", "garman_klass", "yang_zhang"])
def test_determinism_run_twice(estimator: str) -> None:
    base = _signal(_BENCH, 0, "LONG")
    feed = _wide_range_feed(70, seed=404)
    kwargs = dict(
        vol_estimator=estimator, target_vol_per_bar=0.005, vol_window=20, rebalance_band=0.0
    )

    ev_a = _Queue()
    scales_a = _drive_scales(_make([base], ev_a, **kwargs), ev_a, feed)
    ev_b = _Queue()
    scales_b = _drive_scales(_make([base], ev_b, **kwargs), ev_b, feed)
    assert scales_a == scales_b


@pytest.mark.parametrize("estimator", ["parkinson", "garman_klass", "yang_zhang"])
def test_never_raises_on_degenerate_bars(estimator: str) -> None:
    base = _signal(_BENCH, 0, "LONG")
    strategy = _make([base], _Queue(), vol_estimator=estimator, vol_window=10)
    # Missing high/low attributes entirely.
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET", time="2026-05-02T00:00:00Z", symbol=_BENCH, close=100.0)
    )
    # Empty market window (no bars).
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", time="2026-05-02T00:01:00Z", bars_1s={})
    )
    # NaN close (skipped at ingest), zero close (skipped), None close.
    strategy.calculate_signals(_ohlc_event(2, float("nan"), high=float("nan"), low=float("nan")))
    strategy.calculate_signals(_ohlc_event(3, 0.0, high=0.0, low=0.0))
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET", time="2026-05-02T00:04:00Z", symbol=_BENCH, close=None)
    )
    # High/low inverted (low > high) and a normal bar -> still no raise.
    strategy.calculate_signals(_ohlc_event(5, 100.0, high=95.0, low=105.0))
    strategy.calculate_signals(_ohlc_event(6, 100.0, high=110.0, low=90.0))


@pytest.mark.parametrize("estimator", ["parkinson", "garman_klass", "yang_zhang"])
def test_state_roundtrip_preserves_ohl(estimator: str) -> None:
    base = _signal(_BENCH, 0, "LONG")
    feed = _wide_range_feed(50, seed=505)

    events = _Queue()
    strat = _make(
        [base],
        events,
        vol_estimator=estimator,
        target_vol_per_bar=0.005,
        vol_window=20,
        rebalance_band=0.0,
    )
    _drive_scales(strat, events, feed)
    strat._child._state = {"hello": 9}

    snapshot = strat.get_state()
    assert snapshot["overlay"]["opens"] and snapshot["overlay"]["highs"]

    restored_events = _Queue()
    restored = _make(
        [base],
        restored_events,
        vol_estimator=estimator,
        target_vol_per_bar=0.005,
        vol_window=20,
        rebalance_band=0.0,
    )
    restored.set_state(snapshot)
    assert restored._child._state == {"hello": 9}
    assert list(restored._overlay.opens) == list(strat._overlay.opens)
    assert list(restored._overlay.highs) == list(strat._overlay.highs)
    assert list(restored._overlay.lows) == list(strat._overlay.lows)
    assert list(restored._overlay.closes) == list(strat._overlay.closes)

    # A continued bar produces the SAME scale from either instance (state parity).
    next_event = _ohlc_event(50, 100.0, high=104.0, low=96.0)
    ev_a = _Queue()
    strat.events = ev_a
    strat.calculate_signals(next_event)
    ev_b = _Queue()
    restored.events = ev_b
    restored.calculate_signals(next_event)
    scale_a = next(s.metadata["overlay_scale"] for s in ev_a.items if s.signal_type == "LONG")
    scale_b = next(s.metadata["overlay_scale"] for s in ev_b.items if s.signal_type == "LONG")
    assert scale_a == pytest.approx(scale_b)


def test_adversarial_set_state_never_raises() -> None:
    base = _signal(_BENCH, 0, "LONG")
    strat = _make([base], _Queue(), vol_estimator="parkinson", vol_window=20)
    # Garbage / partial payloads must be swallowed, not raised.
    strat.set_state({})
    strat.set_state({"overlay": {"opens": [None, "x", float("nan")], "highs": None}})
    strat.set_state(
        {"overlay": {"closes": [100.0], "opens": [100.0], "highs": [101.0], "lows": [99.0]}}
    )
    strat.set_state("not-a-dict")  # type: ignore[arg-type]


def test_schema_exposes_snake_case_categorical() -> None:
    schema = VolManagedRiskOverlayStrategy.get_param_schema()
    assert "vol_estimator" in schema
    param = schema["vol_estimator"]
    assert param.default == "close_to_close"
    assert set(param.choices or ()) == {"close_to_close", "parkinson", "garman_klass", "yang_zhang"}
