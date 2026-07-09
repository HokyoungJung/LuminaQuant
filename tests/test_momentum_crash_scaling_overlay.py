"""Author-time BUILD GATE + lane invariants for the momentum-crash scaling overlay.

The load-bearing divergence (spec W2-6): on a deep-drawdown bear REBOUND (P1) the
incumbent ``VolManagedRiskOverlayStrategy`` -- run WITH its Daniel-Moskowitz binary
crash gate active -- forwards the SAME scale it forwards on a bear CONTINUATION
(P2), because its gate is blind to the rebound state; this continuous overlay
throttles P1 strictly below P2 and strictly below the incumbent.  Every incumbent
is instantiated as the real class on the same synthetic bars.
"""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.core.events import SignalEvent
from lumina_quant.strategies import momentum_crash_scaling_overlay as MODULE
from lumina_quant.strategies.momentum_crash_scaling_overlay import (
    MomentumCrashDynamicScalingOverlayStrategy,
    _LADDER,
    _quantize_to_ladder,
    bear_rebound_state,
)
from lumina_quant.strategies import vol_managed_risk_overlay as VMR_MODULE
from lumina_quant.strategies.vol_managed_risk_overlay import VolManagedRiskOverlayStrategy

_BENCH = "BTC/USDT"
_BASE = datetime.datetime(2025, 1, 1)


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every child class to the scripted stub (registry has no stub)."""
    for module in (MODULE, VMR_MODULE):
        monkeypatch.setattr(
            module, "resolve_strategy_class", lambda name, default_name=None: _ScriptedChild
        )


# --------------------------------------------------------------------------- #
# Harness: scripted child + queue + MARKET_WINDOW builder.
# --------------------------------------------------------------------------- #
class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _ScriptedChild:
    required_features = ()
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    decision_cadence_seconds = 86400

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        self._scripted: list[SignalEvent] = list(params.get("scripted") or [])
        self._state: dict[str, Any] = {}

    def calculate_signals(self, event: Any) -> None:
        for signal in self._scripted:
            self.events.put(_clone(signal))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        self.calculate_signals(event)

    def get_state(self) -> dict[str, Any]:
        return {"marker": dict(self._state)}

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state.get("marker") or {})


def _clone(signal: SignalEvent) -> SignalEvent:
    return SignalEvent(
        strategy_id=signal.strategy_id,
        symbol=signal.symbol,
        datetime=signal.datetime,
        signal_type=signal.signal_type,
        strength=signal.strength,
        price=signal.price,
        position_side=signal.position_side,
        metadata=dict(signal.metadata or {}),
    )


def _signal(
    symbol: str,
    *,
    signal_type: str = "LONG",
    target_allocation: float = 0.20,
    max_order_value: float = 500.0,
    position_side: str | None = None,
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
        datetime="2025-01-01T00:00:00Z",
        signal_type=signal_type,
        strength=1.0,
        position_side=position_side,
        metadata=metadata,
    )


def _window_event(
    idx: int, closes: dict[str, float], *, half_range: float = 0.0
) -> SimpleNamespace:
    t = (_BASE + datetime.timedelta(days=idx)).isoformat()
    bars_1s: dict[str, list[dict[str, Any]]] = {}
    for symbol, close in closes.items():
        high = close * math.exp(half_range)
        low = close * math.exp(-half_range)
        bars_1s[symbol] = [
            {"time": t, "open": close, "high": high, "low": low, "close": close, "volume": 1000.0}
        ]
    return SimpleNamespace(type="MARKET_WINDOW", time=t, bars_1s=bars_1s)


def _make(
    scripted: list[SignalEvent],
    events: _Queue,
    *,
    symbols: list[str] | None = None,
    **overrides: Any,
) -> MomentumCrashDynamicScalingOverlayStrategy:
    symbols = symbols or [_BENCH]
    bars = SimpleNamespace(
        symbol_list=list(symbols),
        get_latest_feature_value=lambda *a, **k: None,
        get_latest_bar_value=lambda *a, **k: None,
    )
    params: dict[str, Any] = {
        "child_strategy_class": "ScriptedChild",
        "child_params": {"scripted": scripted},
        "child_symbols": list(symbols),
    }
    params.update(overrides)
    strat = MomentumCrashDynamicScalingOverlayStrategy(bars, events, **params)
    strat.symbol_list = list(symbols)
    return strat


def _lcg(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0x100000000


# --------------------------------------------------------------------------- #
# Deterministic P1/P2 bear-rebound vs bear-continuation paths.
# --------------------------------------------------------------------------- #
def _calm_prefix(n: int = 120, level: float = 100.0, step: float = -0.001) -> list[float]:
    out = [level]
    for _ in range(1, n):
        out.append(out[-1] * (1.0 + step))
    return out


def _p1_series() -> list[float]:  # bear then REBOUND
    calm = _calm_prefix()
    tail: list[float] = []
    price = calm[-1]
    for _ in range(10):
        price *= 1.006
        tail.append(price)
    return calm + tail


def _p2_series() -> list[float]:  # bear then CONTINUATION (mirror-signed tail)
    calm = _calm_prefix()
    tail: list[float] = []
    price = calm[-1]
    for _ in range(10):
        price *= 0.994
        tail.append(price)
    return calm + tail


# =========================================================================== #
# (0) helper units
# =========================================================================== #
def test_bear_rebound_state_bear_rebound_and_continuation() -> None:
    assert bear_rebound_state(
        _p1_series(),
        bear_lookback=90,
        rebound_lookback=10,
        rebound_threshold=0.05,
        dd_window=120,
        dd_threshold=-0.05,
    ) == (1, 1)
    assert bear_rebound_state(
        _p2_series(),
        bear_lookback=90,
        rebound_lookback=10,
        rebound_threshold=0.05,
        dd_window=120,
        dd_threshold=-0.05,
    ) == (1, 0)


def test_bear_rebound_state_warmup_and_degenerate_never_raise() -> None:
    assert (
        bear_rebound_state(
            [100.0, 101.0, 102.0],
            bear_lookback=90,
            rebound_lookback=10,
            rebound_threshold=0.05,
            dd_window=120,
            dd_threshold=-0.05,
        )
        is None
    )
    # NaN / non-positive / empty must be filtered, never raise.
    junk = [float("nan"), 0.0, -5.0, float("inf")] + [100.0] * 5
    assert (
        bear_rebound_state(
            junk,
            bear_lookback=90,
            rebound_lookback=10,
            rebound_threshold=0.05,
            dd_window=120,
            dd_threshold=-0.05,
        )
        is None
    )


def test_quantize_to_ladder() -> None:
    assert _quantize_to_ladder(0.0) == 0.0
    assert _quantize_to_ladder(0.12) == 0.0
    assert _quantize_to_ladder(0.125) == 0.25  # round-half-up
    assert _quantize_to_ladder(0.5) == 0.5
    assert _quantize_to_ladder(0.9) == 1.0
    assert _quantize_to_ladder(1.7) == 1.0
    assert _quantize_to_ladder(-0.5) == 0.0
    assert set(_LADDER) == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_raw_scale_graded_product_clamp() -> None:
    events = _Queue()
    lam = 4.0e-4
    strat = _make([_signal(_BENCH)], events, lambda_scale=lam)
    # At the calibration point sigma^2 == lambda: raw == mu_hat (calm -> 1.0).
    assert strat._raw_scale(1.0, lam) == pytest.approx(1.0)
    assert strat._raw_scale(0.5, lam) == pytest.approx(0.5)  # bear-only, calm vol
    assert strat._raw_scale(0.0, lam) == pytest.approx(0.0)  # bear-rebound
    # 4x variance halves nothing linearly: lambda/(4 lambda) = 0.25 own-vol throttle.
    assert strat._raw_scale(1.0, 4.0 * lam) == pytest.approx(0.25)
    # No vol info -> vol throttle inert, mu_hat governs; never amplifies.
    assert strat._raw_scale(0.5, None) == pytest.approx(0.5)
    assert strat._raw_scale(2.0, lam) == pytest.approx(1.0)  # clamped at 1.0


# =========================================================================== #
# (1) CORE build gate: P1/P2 divergence vs the REAL vol-managed incumbent
# =========================================================================== #
def _run_my_overlay(series: list[float], **overrides: Any) -> SignalEvent:
    events = _Queue()
    strat = _make([_signal(_BENCH)], events, rebalance_band=0.0, **overrides)
    last: SignalEvent | None = None
    for i, close in enumerate(series):
        events.items.clear()
        strat.calculate_signals(_window_event(i, {_BENCH: close}))
        if events.items:
            last = events.items[-1]
    assert last is not None
    return last


def _run_incumbent(series: list[float]) -> SignalEvent:
    events = _Queue()
    inc = VolManagedRiskOverlayStrategy(
        SimpleNamespace(
            symbol_list=[_BENCH],
            get_latest_feature_value=lambda *a, **k: None,
            get_latest_bar_value=lambda *a, **k: None,
        ),
        events,
        child_strategy_class="ScriptedChild",
        child_params={"scripted": [_signal(_BENCH)]},
        child_symbols=[_BENCH],
        crash_benchmark_symbol=_BENCH,
        rebalance_band=0.0,
    )
    last: SignalEvent | None = None
    for i, close in enumerate(series):
        events.items.clear()
        inc.calculate_signals(_window_event(i, {_BENCH: close}))
        if events.items:
            last = events.items[-1]
    assert last is not None
    return last


def test_p1_p2_divergence_vs_vol_managed_incumbent() -> None:
    overlay_kwargs = dict(
        bear_lookback_bars=90,
        dd_window_bars=120,
        dd_threshold=-0.05,
        rebound_lookback_bars=10,
        rebound_threshold=0.05,
        lambda_scale=1.0,  # vol throttle inert -> isolate the bear x rebound term
        bucket_min_hold_bars=0,
    )
    my_p1 = _run_my_overlay(_p1_series(), **overlay_kwargs)
    my_p2 = _run_my_overlay(_p2_series(), **overlay_kwargs)
    inc_p1 = _run_incumbent(_p1_series())
    inc_p2 = _run_incumbent(_p2_series())

    # (i) The incumbent -- crash gate active -- forwards the SAME scale on P1 and
    #     P2: its binary conjunction is blind to the panic-rebound state.
    assert inc_p1.metadata["overlay_scale"] == pytest.approx(inc_p2.metadata["overlay_scale"])
    assert inc_p1.metadata["overlay_scale"] > 0.0

    # (ii) This overlay's bucket on P1 is STRICTLY LOWER than on P2 (only P1 fires
    #      I_bear * I_rebound), and its forwarded notional on P1 is strictly below
    #      the incumbent's on P1.
    assert my_p1.metadata["overlay_scale"] < my_p2.metadata["overlay_scale"]
    assert my_p1.metadata["overlay_scale"] == pytest.approx(0.0)
    assert my_p1.metadata.get("target_allocation", 0.0) < inc_p1.metadata["target_allocation"]


# =========================================================================== #
# (2) never-boost + basket preservation (graveyard-#3 dodge)
# =========================================================================== #
def test_never_boost_and_preserves_basket_and_relative_weights() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    scripted = [
        _signal("BTC/USDT", target_allocation=0.40, max_order_value=800.0),
        _signal("ETH/USDT", target_allocation=0.20, max_order_value=400.0),
        _signal("SOL/USDT", signal_type="SHORT", target_allocation=0.10, position_side="SHORT"),
    ]
    events = _Queue()
    # A volatile benchmark so the own-vol throttle engages (< 1.0 common multiplier).
    strat = _make(
        scripted,
        events,
        symbols=symbols,
        benchmark_symbol="BTC/USDT",
        lambda_scale=1.0e-5,
        sigma_half_life_bars=8,
        rebalance_band=0.0,
        bucket_min_hold_bars=0,
    )
    price = dict.fromkeys(symbols, 100.0)
    last_batch: list[SignalEvent] = []
    for i in range(80):
        events.items.clear()
        sign = 1.0 if i % 2 == 0 else -1.0
        price = {s: price[s] * (1.0 + 0.06 * sign) for s in symbols}
        strat.calculate_signals(_window_event(i, price))
        if events.items:
            last_batch = list(events.items)

    assert {s.symbol for s in last_batch} == set(symbols)  # symbol set preserved
    multipliers = {s.symbol: s.metadata["overlay_scale"] for s in last_batch}
    assert len(set(multipliers.values())) == 1  # ONE common multiplier
    mult = next(iter(multipliers.values()))
    assert mult <= 1.0  # never boosts above child intent
    # Relative weights preserved: forwarded alloc == child alloc * common multiplier.
    child_alloc = {"BTC/USDT": 0.40, "ETH/USDT": 0.20, "SOL/USDT": 0.10}
    for sig in last_batch:
        assert sig.metadata["target_allocation"] == pytest.approx(child_alloc[sig.symbol] * mult)
        assert sig.metadata["target_allocation"] <= child_alloc[sig.symbol] + 1e-12
        # A min-clamp cap must NOT be inflated by the overlay.
        assert sig.metadata["max_symbol_exposure_pct"] == pytest.approx(child_alloc[sig.symbol])


def test_bullbear_rotation_changes_set_while_overlay_preserves_set() -> None:
    from lumina_quant.strategies.bull_bear_regime_rotation import BullBearRegimeRotationStrategy

    symbols = [f"S{i}/USDT" for i in range(6)]
    bars = SimpleNamespace(
        symbol_list=list(symbols),
        get_latest_feature_value=lambda *a, **k: None,
        get_latest_bar_value=lambda *a, **k: None,
    )
    bb_events = _Queue()
    bb = BullBearRegimeRotationStrategy(bars, bb_events)

    # Phase up then Phase down -> BullBear must flip regime (rotates the SET/side).
    price = dict.fromkeys(symbols, 100.0)
    regimes: list[str] = []
    for i in range(60):
        drift = 0.02 if i < 30 else -0.02
        price = {s: price[s] * (1.0 + drift) for s in symbols}
        bb.calculate_signals(_window_event(i, price))
        regimes.append(bb._regime)
    assert "BULL" in regimes and "BEAR" in regimes  # incumbent rotates the regime

    # This overlay wraps a fixed 6-symbol LONG basket: the forwarded symbol SET is
    # invariant to regime; only a common <=1 multiplier changes.
    scripted = [_signal(s, target_allocation=0.10) for s in symbols]
    ov_events = _Queue()
    ov = _make(scripted, ov_events, symbols=symbols, lambda_scale=1.0, benchmark_symbol=symbols[0])
    price = dict.fromkeys(symbols, 100.0)
    seen_sets: set[frozenset[str]] = set()
    for i in range(60):
        ov_events.items.clear()
        drift = 0.02 if i < 30 else -0.02
        price = {s: price[s] * (1.0 + drift) for s in symbols}
        ov.calculate_signals(_window_event(i, price))
        if ov_events.items:
            seen_sets.add(frozenset(s.symbol for s in ov_events.items))
    assert seen_sets == {frozenset(symbols)}  # SET preserved across the regime flip


# =========================================================================== #
# (3) ladder hysteresis + turnover guard
# =========================================================================== #
def test_min_hold_delays_re_risk_but_derisk_is_immediate() -> None:
    events = _Queue()
    strat = _make([_signal(_BENCH)], events, lambda_scale=1.0, bucket_min_hold_bars=5)
    st = strat._overlay
    st.current_bucket = 1.0
    st.bars_since_bucket_change = 0

    # De-risk target below current -> immediate.
    strat._raw_scale = lambda mu, sig: 0.0  # type: ignore[method-assign]
    strat._mu_hat = lambda: 0.0  # type: ignore[method-assign]
    strat._child_variance = lambda: 1.0e-6  # type: ignore[method-assign]
    strat._advance_bucket()
    assert st.current_bucket == 0.0

    # Re-risk target above current -> blocked until min-hold elapses (5 bars held).
    strat._raw_scale = lambda mu, sig: 1.0  # type: ignore[method-assign]
    for _ in range(5):
        strat._advance_bucket()
        assert st.current_bucket == 0.0  # still held (bars_since < 5)
    strat._advance_bucket()  # 6th call -> min-hold satisfied, re-risk allowed
    assert st.current_bucket == 1.0


def test_within_bucket_oscillation_produces_zero_reemit() -> None:
    events = _Queue()
    # rebalance_band > 0 and a raw scale that oscillates but stays in one bucket:
    # the quantizer pins the bucket so the band suppresses redundant re-emits.
    strat = _make([_signal(_BENCH)], events, lambda_scale=1.0, rebalance_band=0.30)
    raws = iter([0.92, 0.88, 0.95, 0.90] * 30)
    strat._mu_hat = lambda: 1.0  # type: ignore[method-assign]
    strat._child_variance = lambda: 1.0e-6  # type: ignore[method-assign]
    strat._raw_scale = lambda mu, sig: next(raws)  # type: ignore[method-assign]
    emitted = 0
    for i in range(40):
        events.items.clear()
        strat.calculate_signals(_window_event(i, {_BENCH: 100.0 + i}))
        emitted += len(events.items)
    assert emitted == 1  # all raws quantize to 1.0 -> single emit


# =========================================================================== #
# (4) EXIT unscaled + de-risk-only property
# =========================================================================== #
def test_exit_forwarded_unscaled_and_scale_never_exceeds_one() -> None:
    longs = _signal(_BENCH, target_allocation=0.20)
    exit_sig = _signal("ETH/USDT", signal_type="EXIT", target_allocation=0.0, max_order_value=0.0)
    events = _Queue()
    strat = _make(
        [longs, exit_sig],
        events,
        symbols=[_BENCH, "ETH/USDT"],
        benchmark_symbol=_BENCH,
        lambda_scale=1.0e-5,
        sigma_half_life_bars=8,
        rebalance_band=0.0,
    )
    price = {_BENCH: 100.0, "ETH/USDT": 100.0}
    scales: list[float] = []
    for i in range(60):
        events.items.clear()
        sign = 1.0 if i % 2 == 0 else -1.0
        price = {s: price[s] * (1.0 + 0.08 * sign) for s in price}
        strat.calculate_signals(_window_event(i, price))
        for s in events.items:
            if s.signal_type == "LONG":
                scales.append(s.metadata["overlay_scale"])
    exit_out = next(s for s in events.items if s.signal_type == "EXIT")
    assert "overlay_scale" not in exit_out.metadata  # EXIT never rescaled
    assert exit_out.strength == pytest.approx(1.0)
    assert scales and all(sc <= 1.0 for sc in scales)  # never amplifies


# =========================================================================== #
# (5) determinism, state, no-lookahead, never-raise
# =========================================================================== #
def _lcg_series(seed: int, n: int = 90) -> list[float]:
    gen = _lcg(seed)
    price = 100.0
    out = [price]
    for _ in range(1, n):
        price *= 1.0 + (next(gen) - 0.5) * 0.05
        out.append(price)
    return out


def test_run_twice_bit_identical() -> None:
    series = _lcg_series(20260709)

    def run() -> list[tuple[str, float]]:
        events = _Queue()
        strat = _make([_signal(_BENCH)], events, rebalance_band=0.0)
        rows: list[tuple[str, float]] = []
        for i, close in enumerate(series):
            events.items.clear()
            strat.calculate_signals(_window_event(i, {_BENCH: close}))
            rows.extend((s.signal_type, s.metadata.get("overlay_scale", 1.0)) for s in events.items)
        return rows

    assert run() == run()


def test_state_roundtrip_and_adversarial_set_state() -> None:
    series = _lcg_series(7)
    events = _Queue()
    strat = _make([_signal(_BENCH)], events, rebalance_band=0.0)
    for i, close in enumerate(series):
        strat.calculate_signals(_window_event(i, {_BENCH: close}))
    strat._child._state = {"x": 3}
    snap = strat.get_state()

    restored = _make([_signal(_BENCH)], _Queue(), rebalance_band=0.0)
    restored.set_state(snap)
    assert restored._child._state == {"x": 3}
    assert list(restored._overlay.closes) == list(strat._overlay.closes)
    assert list(restored._overlay.proxy_returns) == list(strat._overlay.proxy_returns)
    assert restored._overlay.current_bucket == strat._overlay.current_bucket
    assert restored._overlay.last_signed_alloc == strat._overlay.last_signed_alloc

    # Adversarial garbage must never raise.
    for junk in (None, {}, {"overlay": 5}, {"overlay": {"closes": ["x", None]}}, {"guard": 1}):
        restored.set_state(junk)  # type: ignore[arg-type]


def test_no_lookahead_scale_uses_no_future_bar() -> None:
    series = _p1_series()
    cutoff = 125
    kwargs = dict(bear_lookback_bars=90, dd_window_bars=120, lambda_scale=1.0, rebalance_band=0.0)

    events_a = _Queue()
    strat_a = _make([_signal(_BENCH)], events_a, **kwargs)
    scale_full = None
    for i, close in enumerate(series):
        events_a.items.clear()
        strat_a.calculate_signals(_window_event(i, {_BENCH: close}))
        if i == cutoff and events_a.items:
            scale_full = events_a.items[-1].metadata["overlay_scale"]

    events_b = _Queue()
    strat_b = _make([_signal(_BENCH)], events_b, **kwargs)
    scale_trunc = None
    for i, close in enumerate(series[: cutoff + 1]):
        events_b.items.clear()
        strat_b.calculate_signals(_window_event(i, {_BENCH: close}))
        if i == cutoff and events_b.items:
            scale_trunc = events_b.items[-1].metadata["overlay_scale"]

    assert scale_full is not None and scale_full == pytest.approx(scale_trunc)


def test_never_raises_on_degenerate_input_or_child_explosion() -> None:
    events = _Queue()
    strat = _make([_signal(_BENCH)], events)
    strat.calculate_signals(SimpleNamespace(type="MARKET", time=None, symbol=None, close=None))
    strat.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", time="2025-02-02T00:00:00Z", bars_1s={})
    )
    strat.calculate_signals(_window_event(3, {_BENCH: float("nan")}))

    class _Boom(_ScriptedChild):
        def calculate_signals(self, event: Any) -> None:
            raise RuntimeError("boom")

    strat._child = _Boom(strat.bars, strat._child_queue)
    strat.calculate_signals(_window_event(4, {_BENCH: 100.0}))  # must not raise


def test_module_marker_constants() -> None:
    assert MODULE._STRATEGY_NAME == "MomentumCrashDynamicScalingOverlayStrategy"
    assert math.isfinite(MODULE._EPS)
    assert MODULE._SUGGESTED_FAMILY == "overlay"
