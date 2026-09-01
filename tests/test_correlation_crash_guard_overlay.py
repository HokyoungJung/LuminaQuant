"""Author-time BUILD GATE + lane invariants for the avg-correlation crash guard.

The load-bearing divergence (spec wave-2b Lane 3): on a basket whose average
pairwise correlation spikes from 0 to 1 WITHOUT any change to per-symbol vol,
drawdown, breadth or vol-of-vol, this guard de-risks while the incumbent
gauges -- ``VolManagedRiskOverlayStrategy`` (vol clamp + crash gate),
``BreadthRegimeTrendTimerStrategy`` (breadth census) and
``VolOfVolRegimeTrendGateStrategy`` (GK/rv governor) -- are all UNCHANGED.  The
converse (a vol spike WITHOUT a correlation change) leaves this guard neutral
while the vol overlay de-risks.  Every incumbent is the real class on the same
synthetic bars.
"""

from __future__ import annotations

import datetime
import math
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lumina_quant.core.events import SignalEvent
from lumina_quant.strategies import correlation_crash_guard_overlay as MODULE
from lumina_quant.strategies import vol_managed_risk_overlay as VMR_MODULE
from lumina_quant.strategies.correlation_crash_guard_overlay import (
    AvgCorrelationCrashGuardOverlayStrategy,
    _average_pairwise_correlation,
)
from lumina_quant.strategies.micro_signal_alpha_sleeves import VolOfVolRegimeTrendGateStrategy
from lumina_quant.strategies.vol_managed_risk_overlay import VolManagedRiskOverlayStrategy
from lumina_quant.strategies.vol_term_breadth_alpha_sleeves import BreadthRegimeTrendTimerStrategy

_SYMS = [f"S{i}/USDT" for i in range(6)]
_BASE = datetime.datetime(2025, 1, 1)


# --------------------------------------------------------------------------- #
# Harness.
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
        metadata=dict(signal.metadata or {}),
    )


def _signal(
    symbol: str, *, signal_type: str = "LONG", target_allocation: float = 0.10
) -> SignalEvent:
    metadata: dict[str, Any] = {"strategy": "child"}
    if target_allocation > 0.0:
        metadata["target_allocation"] = target_allocation
        metadata["max_symbol_exposure_pct"] = target_allocation
        metadata["max_order_value"] = 500.0
    return SignalEvent(
        strategy_id="child::scripted",
        symbol=symbol,
        datetime="2025-01-01T00:00:00Z",
        signal_type=signal_type,
        strength=1.0,
        metadata=metadata,
    )


@pytest.fixture(autouse=True)
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (MODULE, VMR_MODULE):
        monkeypatch.setattr(
            module, "resolve_strategy_class", lambda name, default_name=None: _ScriptedChild
        )


def _hadamard8() -> np.ndarray:
    h2 = np.array([[1, 1], [1, -1]], dtype=float)
    h4 = np.kron(h2, h2)
    return np.kron(h2, h4)


def _rows(n: int = 6) -> list[np.ndarray]:
    """``n`` non-constant, mutually orthogonal Hadamard rows (rows 1..n)."""
    h8 = _hadamard8()
    return [h8[i] for i in range(1, n + 1)]


def _window_event(idx: int, closes: dict[str, float], ranges: dict[str, float]) -> SimpleNamespace:
    t = (_BASE + datetime.timedelta(days=idx)).isoformat()
    bars_1s: dict[str, list[dict[str, Any]]] = {}
    for symbol, close in closes.items():
        half = ranges[symbol]
        bars_1s[symbol] = [
            {
                "time": t,
                "open": close,
                "high": close * math.exp(half),
                "low": close * math.exp(-half),
                "close": close,
                "volume": 1000.0,
            }
        ]
    return SimpleNamespace(type="MARKET_WINDOW", time=t, bars_1s=bars_1s)


def _phase_returns(
    rows: list[np.ndarray], length: int, *, drift: float, mag: float
) -> list[np.ndarray]:
    """Per-symbol return vectors: ``drift + mag*sign`` tiled over ``length`` bars."""
    reps = length // rows[0].size + 1
    return [(drift + mag * np.tile(row, reps)[:length]) for row in rows]


def _build_timeline(
    phases: list[tuple[str, int, float]], *, drift: float = 0.006, r: float = 0.01
) -> list[SimpleNamespace]:
    """Assemble MARKET_WINDOW events; each phase = (kind, length, mag).

    ``kind`` is ``"orthogonal"`` (Hadamard rows -> pairwise corr 0) or ``"lockstep"``
    (all symbols share one row -> corr 1).  Per-bar OHLC range is fixed by ``mag`` so
    per-symbol vol is a function of magnitude only.
    """
    rows = _rows(len(_SYMS))
    price = dict.fromkeys(_SYMS, 100.0)
    events: list[SimpleNamespace] = []
    idx = 0
    for kind, length, mag in phases:
        if kind == "lockstep":
            shared = drift + mag * np.tile(rows[0], length // rows[0].size + 1)[:length]
            rets = dict.fromkeys(_SYMS, shared)
        else:
            per = _phase_returns(rows, length, drift=drift, mag=mag)
            rets = {s: per[i] for i, s in enumerate(_SYMS)}
        for t in range(length):
            closes = {s: price[s] * math.exp(float(rets[s][t])) for s in _SYMS}
            price = closes
            ranges = dict.fromkeys(_SYMS, mag / 2.0)
            events.append(_window_event(idx, closes, ranges))
            idx += 1
    return events


def _make(events: _Queue, **overrides: Any) -> AvgCorrelationCrashGuardOverlayStrategy:
    bars = SimpleNamespace(
        symbol_list=list(_SYMS),
        get_latest_feature_value=lambda *a, **k: None,
        get_latest_bar_value=lambda *a, **k: None,
    )
    params: dict[str, Any] = {
        "child_strategy_class": "ScriptedChild",
        "child_params": {"scripted": [_signal(s) for s in _SYMS]},
        "child_symbols": list(_SYMS),
        "corr_window": 48,
        "corr_z_window": 240,
        "min_symbols": 4,
        "rebalance_band": 0.0,
    }
    params.update(overrides)
    return AvgCorrelationCrashGuardOverlayStrategy(bars, events, **params)


def _returns_from_closes(closes: dict[str, list[float]]) -> list[list[float]]:
    return [[math.log(b / a) for a, b in pairwise(v)] for v in closes.values()]


# =========================================================================== #
# (0) equicorrelation helper + Stage-1 premises
# =========================================================================== #
def test_equicorrelation_identity() -> None:
    rows = _rows(6)
    win = 48
    ortho = [(0.006 + 0.01 * np.tile(row, win // 8)).tolist() for row in rows]
    assert _average_pairwise_correlation(ortho) == pytest.approx(0.0, abs=1e-9)
    shared = (0.006 + 0.01 * np.tile(rows[0], win // 8)).tolist()
    assert _average_pairwise_correlation([shared] * 6) == pytest.approx(1.0, abs=1e-9)
    # magnitude 3x, still orthogonal -> corr ~0 (vol changed, correlation did not)
    ortho3 = [(0.006 + 0.03 * np.tile(row, win // 8)).tolist() for row in rows]
    assert _average_pairwise_correlation(ortho3) == pytest.approx(0.0, abs=1e-9)
    # degenerate
    assert _average_pairwise_correlation([[0.0] * 48] * 6) is None
    assert _average_pairwise_correlation([ortho[0]]) is None


def test_stage1_premises_on_built_closes() -> None:
    # 49 closes -> 48 returns == 6 full Hadamard periods -> exact orthogonality.
    events_a = _build_timeline([("orthogonal", 49, 0.01)])
    closes_a = {s: [ev.bars_1s[s][0]["close"] for ev in events_a] for s in _SYMS}
    assert _average_pairwise_correlation(_returns_from_closes(closes_a)) == pytest.approx(
        0.0, abs=1e-9
    )
    events_b = _build_timeline([("lockstep", 49, 0.01)])
    closes_b = {s: [ev.bars_1s[s][0]["close"] for ev in events_b] for s in _SYMS}
    assert _average_pairwise_correlation(_returns_from_closes(closes_b)) == pytest.approx(
        1.0, abs=1e-9
    )


# =========================================================================== #
# (1) guard engages on the correlation spike (Phase A released, Phase B engaged)
# =========================================================================== #
def test_guard_released_in_phase_a_and_engaged_in_phase_b() -> None:
    phase_a = 200
    phase_b = 140
    events = _build_timeline([("orthogonal", phase_a, 0.01), ("lockstep", phase_b, 0.01)])
    q = _Queue()
    strat = _make(q, min_dwell_bars=12)
    scale_end_a: float | None = None
    for i, ev in enumerate(events):
        strat.calculate_signals(ev)
        if i == phase_a - 1:
            scale_end_a = strat._overlay_scale()
    scale_end_b = strat._overlay_scale()
    assert scale_end_a == pytest.approx(1.0)  # decorrelated regime -> full exposure
    assert scale_end_b == pytest.approx(strat.derisk_scale)  # correlation spike -> de-risk
    assert strat._guard.engaged is True
    # And the child's book was actually scaled by derisk_scale at the end.
    q.items.clear()
    strat.calculate_signals(events[-1])  # same key -> no new bar, but re-drain child
    # Re-feed a genuinely new bar to observe forwarding under engagement.
    new_ev = _build_timeline([("lockstep", phase_b + 1, 0.01)])[-1]
    q.items.clear()
    strat.calculate_signals(new_ev)
    longs = [s for s in q.items if s.signal_type == "LONG"]
    assert longs and all(
        s.metadata["overlay_scale"] == pytest.approx(strat.derisk_scale) for s in longs
    )
    assert all(
        s.metadata["target_allocation"] == pytest.approx(0.10 * strat.derisk_scale) for s in longs
    )


# =========================================================================== #
# (2) vol-managed overlay is BLIND to the pure-correlation spike
# =========================================================================== #
def _run_incumbent_capture(inc: Any, events: list[SimpleNamespace], phase_a_len: int) -> tuple:
    """Feed the timeline; return the incumbent's forwarded scale at end A / end B."""
    scale_a = None
    for i, ev in enumerate(events):
        inc.events.items.clear()
        inc.calculate_signals(ev)
        if i == phase_a_len - 1 and inc.events.items:
            scale_a = inc.events.items[-1].metadata.get("overlay_scale")
    scale_b = inc.events.items[-1].metadata.get("overlay_scale") if inc.events.items else None
    return scale_a, scale_b


def test_vol_managed_overlay_blind_to_correlation() -> None:
    phase_a, phase_b = 160, 120
    events = _build_timeline([("orthogonal", phase_a, 0.01), ("lockstep", phase_b, 0.01)])
    q = _Queue()
    inc = VolManagedRiskOverlayStrategy(
        SimpleNamespace(
            symbol_list=list(_SYMS),
            get_latest_feature_value=lambda *a, **k: None,
            get_latest_bar_value=lambda *a, **k: None,
        ),
        q,
        child_strategy_class="ScriptedChild",
        child_params={"scripted": [_signal(s) for s in _SYMS]},
        child_symbols=list(_SYMS),
        crash_benchmark_symbol=_SYMS[0],
        vol_window=20,
        rebalance_band=0.0,
    )
    scale_a, scale_b = _run_incumbent_capture(inc, events, phase_a)
    assert scale_a is not None and scale_b is not None
    # Benchmark vol + drawdown unchanged across the spike -> identical scale.
    assert scale_b == pytest.approx(scale_a)

    # This guard DIVERGES on the same input: full in Phase A, de-risked in Phase B.
    q2 = _Queue()
    guard = _make(q2, min_dwell_bars=12)
    for i, ev in enumerate(events):
        guard.calculate_signals(ev)
        if i == phase_a - 1:
            assert guard._overlay_scale() == pytest.approx(1.0)
    assert guard._overlay_scale() == pytest.approx(guard.derisk_scale)


# =========================================================================== #
# (3) breadth-count timer is BLIND to the pure-correlation spike
# =========================================================================== #
def test_breadth_regime_timer_blind_to_correlation() -> None:
    phase_a, phase_b = 160, 120
    events = _build_timeline([("orthogonal", phase_a, 0.01), ("lockstep", phase_b, 0.01)])
    q = _Queue()
    br = BreadthRegimeTrendTimerStrategy(
        SimpleNamespace(
            symbol_list=list(_SYMS),
            get_latest_feature_value=lambda *a, **k: None,
            get_latest_bar_value=lambda *a, **k: None,
        ),
        q,
        min_symbols=4,
        trend_window=20,
        rebalance_bars=1,
    )
    breadth_a = risk_on_a = None
    for i, ev in enumerate(events):
        br.calculate_signals(ev)
        if i == phase_a - 1:
            breadth_a, risk_on_a = br._last_breadth, br._risk_on
    # Strong common drift -> every name above its own trend in BOTH phases; the
    # breadth census and risk-on flag are identical across the correlation spike.
    assert breadth_a == pytest.approx(1.0) and risk_on_a is True
    assert br._last_breadth == pytest.approx(breadth_a) and br._risk_on is risk_on_a


# =========================================================================== #
# (4) vol-of-vol governor is BLIND to the pure-correlation spike
# =========================================================================== #
def _vol_of_vol_s0_signals(events: list[SimpleNamespace]) -> tuple[list[tuple], float]:
    q = _Queue()
    vv = VolOfVolRegimeTrendGateStrategy(
        SimpleNamespace(
            symbol_list=list(_SYMS),
            get_latest_feature_value=lambda *a, **k: None,
            get_latest_bar_value=lambda *a, **k: None,
        ),
        q,
        min_price=0.0,
    )
    for ev in events:
        vv.calculate_signals(ev)
    stream = [
        (
            s.signal_type,
            round(float(s.metadata.get("target_allocation", 0.0)), 9),
            round(float(s.strength), 9),
        )
        for s in q.items
        if s.symbol == _SYMS[0]
    ]
    return stream, vv._extra[_SYMS[0]].last_size_multiplier


def test_vol_of_vol_gate_blind_to_correlation() -> None:
    # ISOLATION via two runs where S0's OWN OHLCV is byte-identical but the basket
    # correlation differs: in the orthogonal run S0 == rows[0] and the others take
    # their own orthogonal rows (rho 0); in the lockstep run EVERY symbol takes
    # rows[0], so S0's own path is unchanged while the basket comoves (rho 1).  A
    # per-symbol GK/rv governor reads only S0's own bars, so its S0 decisions and
    # size multiplier are identical -- blind to the cross-symbol correlation.
    ortho = _build_timeline([("orthogonal", 200, 0.01)])
    lock = _build_timeline([("lockstep", 200, 0.01)])
    stream_ortho, mult_ortho = _vol_of_vol_s0_signals(ortho)
    stream_lock, mult_lock = _vol_of_vol_s0_signals(lock)
    assert stream_ortho  # non-vacuous: the incumbent actually acts on S0
    assert stream_ortho == stream_lock
    assert mult_lock == pytest.approx(mult_ortho)

    # Candidate-ACTS: on a decorrelated->correlated TRANSITION (the spike the guard
    # keys off) this guard engages, while VolOfVol's per-symbol governor above did not
    # move on the pure correlation change.
    transition = _build_timeline([("orthogonal", 160, 0.01), ("lockstep", 120, 0.01)])
    q_g = _Queue()
    guard = _make(q_g, min_dwell_bars=12)
    for i, ev in enumerate(transition):
        guard.calculate_signals(ev)
        if i == 159:
            assert guard._overlay_scale() == pytest.approx(1.0)
    assert guard._overlay_scale() == pytest.approx(guard.derisk_scale)


# =========================================================================== #
# (5) CONVERSE: a vol spike WITHOUT correlation leaves this guard neutral
#     while the vol overlay de-risks (divergence in the other direction).
# =========================================================================== #
def test_converse_vol_spike_without_correlation() -> None:
    phase_a, phase_c = 160, 80
    # Phase C: magnitude tripled but signs stay orthogonal -> rho ~ 0, vol up 3x.
    events = _build_timeline([("orthogonal", phase_a, 0.01), ("orthogonal", phase_c, 0.03)])

    q = _Queue()
    guard = _make(q, min_dwell_bars=12)
    for ev in events:
        guard.calculate_signals(ev)
    assert guard._overlay_scale() == pytest.approx(1.0)  # no correlation change -> neutral
    assert guard._guard.engaged is False

    q2 = _Queue()
    inc = VolManagedRiskOverlayStrategy(
        SimpleNamespace(
            symbol_list=list(_SYMS),
            get_latest_feature_value=lambda *a, **k: None,
            get_latest_bar_value=lambda *a, **k: None,
        ),
        q2,
        child_strategy_class="ScriptedChild",
        child_params={"scripted": [_signal(s) for s in _SYMS]},
        child_symbols=list(_SYMS),
        crash_benchmark_symbol=_SYMS[0],
        target_vol_per_bar=0.01,
        vol_window=20,
        rebalance_band=0.0,
    )
    scale_a, scale_c = _run_incumbent_capture(inc, events, phase_a)
    assert scale_a is not None and scale_c is not None
    assert scale_c < scale_a  # the vol clamp DROPS on the vol spike


# =========================================================================== #
# (6) de-risk-only property, release hysteresis, EXIT unscaled, determinism, ...
# =========================================================================== #
def test_scale_never_exceeds_one_and_derisks_at_least_once() -> None:
    events = _build_timeline([("orthogonal", 140, 0.01), ("lockstep", 120, 0.01)])
    q = _Queue()
    strat = _make(q, min_dwell_bars=12)
    scales: list[float] = []
    for ev in events:
        q.items.clear()
        strat.calculate_signals(ev)
        scales.extend(s.metadata["overlay_scale"] for s in q.items if s.signal_type != "EXIT")
    assert scales and all(sc <= 1.0 for sc in scales)  # never amplifies the child
    assert any(sc == pytest.approx(strat.derisk_scale) for sc in scales)  # did de-risk


def test_release_requires_z_exit_and_min_dwell() -> None:
    # Engage in a lockstep phase, then decorrelate: release only after z<=z_exit AND
    # min_dwell bars have elapsed.
    events = _build_timeline(
        [("orthogonal", 120, 0.01), ("lockstep", 120, 0.01), ("orthogonal", 120, 0.01)]
    )
    q = _Queue()
    strat = _make(q, min_dwell_bars=30)
    engaged_seen = False
    released_after = None
    for i, ev in enumerate(events):
        strat.calculate_signals(ev)
        if i == 239 and strat._guard.engaged:  # end of lockstep phase
            engaged_seen = True
        if engaged_seen and released_after is None and not strat._guard.engaged:
            released_after = i
    assert engaged_seen  # it engaged during lockstep
    assert released_after is not None and released_after > 240  # released back in Phase 3


def test_exit_forwarded_unscaled_while_engaged() -> None:
    events = _build_timeline([("orthogonal", 120, 0.01), ("lockstep", 100, 0.01)])
    q = _Queue()
    scripted = [_signal(s) for s in _SYMS] + [
        _signal("Z/USDT", signal_type="EXIT", target_allocation=0.0)
    ]
    strat = _make(q, child_params={"scripted": scripted})
    for ev in events:
        q.items.clear()
        strat.calculate_signals(ev)
    assert strat._guard.engaged is True
    exit_out = next(s for s in q.items if s.signal_type == "EXIT")
    assert "overlay_scale" not in exit_out.metadata
    assert exit_out.strength == pytest.approx(1.0)
    longs = [s for s in q.items if s.signal_type == "LONG"]
    assert longs and all(
        s.metadata["overlay_scale"] == pytest.approx(strat.derisk_scale) for s in longs
    )


def test_run_twice_bit_identical() -> None:
    events = _build_timeline([("orthogonal", 120, 0.01), ("lockstep", 80, 0.01)])

    def run() -> list[float]:
        q = _Queue()
        strat = _make(q, min_dwell_bars=12)
        out: list[float] = []
        for ev in events:
            strat.calculate_signals(ev)
            out.append(strat._overlay_scale())
        return out

    assert run() == run()


def test_state_roundtrip_and_adversarial_set_state() -> None:
    events = _build_timeline([("orthogonal", 120, 0.01), ("lockstep", 60, 0.01)])
    q = _Queue()
    strat = _make(q, min_dwell_bars=12)
    for ev in events:
        strat.calculate_signals(ev)
    strat._child._state = {"k": 9}
    snap = strat.get_state()

    restored = _make(_Queue(), min_dwell_bars=12)
    restored.set_state(snap)
    assert restored._child._state == {"k": 9}
    assert restored._guard.engaged == strat._guard.engaged
    assert restored._guard.dwell == strat._guard.dwell
    assert list(restored._guard.rho_history) == list(strat._guard.rho_history)
    for s in _SYMS:
        assert list(restored._guard.closes[s]) == list(strat._guard.closes[s])

    for junk in (
        None,
        {},
        {"guard": 5},
        {"guard": {"closes": 3}},
        {"guard": {"rho_history": ["x"]}},
    ):
        restored.set_state(junk)  # type: ignore[arg-type]


def test_neutral_and_never_raise_on_degenerate_input() -> None:
    q = _Queue()
    strat = _make(q)
    # Warmup / sub-min_symbols / empty / NaN must stay neutral 1.0 and never raise.
    strat.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", time="2025-03-01T00:00:00Z", bars_1s={})
    )
    assert strat._overlay_scale() == pytest.approx(1.0)
    only_two = _window_event(
        1, {"S0/USDT": 100.0, "S1/USDT": 100.0}, {"S0/USDT": 0.005, "S1/USDT": 0.005}
    )
    strat.calculate_signals(only_two)
    assert strat._overlay_scale() == pytest.approx(1.0)  # sub-min_symbols -> neutral
    nan_ev = _window_event(2, {s: float("nan") for s in _SYMS}, dict.fromkeys(_SYMS, 0.005))
    strat.calculate_signals(nan_ev)

    class _Boom(_ScriptedChild):
        def calculate_signals(self, event: Any) -> None:
            raise RuntimeError("boom")

    strat._child = _Boom(strat.bars, strat._child_queue)
    strat.calculate_signals(
        _window_event(3, dict.fromkeys(_SYMS, 100.0), dict.fromkeys(_SYMS, 0.005))
    )


def test_no_lookahead_scale_uses_no_future_bar() -> None:
    events = _build_timeline([("orthogonal", 120, 0.01), ("lockstep", 80, 0.01)])
    cutoff = 150

    q_a = _Queue()
    strat_a = _make(q_a, min_dwell_bars=12)
    scale_full = None
    for i, ev in enumerate(events):
        strat_a.calculate_signals(ev)
        if i == cutoff:
            scale_full = strat_a._overlay_scale()

    q_b = _Queue()
    strat_b = _make(q_b, min_dwell_bars=12)
    scale_trunc = None
    for i, ev in enumerate(events[: cutoff + 1]):
        strat_b.calculate_signals(ev)
        if i == cutoff:
            scale_trunc = strat_b._overlay_scale()

    assert scale_full is not None and scale_full == pytest.approx(scale_trunc)


def test_module_marker_constants() -> None:
    assert MODULE._STRATEGY_NAME == "AvgCorrelationCrashGuardOverlayStrategy"
    assert math.isfinite(MODULE._EPS)
    assert MODULE._SUGGESTED_FAMILY == "overlay"


def test_slice_multi_timeframe_keys_and_bounds() -> None:
    """Pin the 1d/4h/1h slice: mirrored variants + keys, scaled cells in-bounds.

    Guards against a silent schema clamp so the written 1h values (``corr_window``
    / ``corr_z_window`` capped at the schema maxima) equal the effective ones, and
    pins the guard thresholds/scale as tf-invariant.
    """
    slice_dict = MODULE._CORRELATION_CRASH_GUARD_OVERLAY_SLICE
    assert set(slice_dict) == {"1d", "4h", "1h"}
    counts = {tf: len(cells) for tf, cells in slice_dict.items()}
    assert len(set(counts.values())) == 1, counts
    base = {cell["variant"]: cell for cell in slice_dict["1d"]}
    schema = AvgCorrelationCrashGuardOverlayStrategy.get_param_schema()
    for tf, cells in slice_dict.items():
        assert tuple(c["variant"] for c in cells) == tuple(base), (tf, counts)
        for cell in cells:
            assert set(cell) == set(base[cell["variant"]]), (tf, cell["variant"])
            for key, value in cell.items():
                if key == "variant" or isinstance(value, bool):
                    continue
                hp = schema[key]
                if hp.low is not None:
                    assert value >= hp.low, (tf, key, value)
                if hp.high is not None:
                    assert value <= hp.high, (tf, key, value)
    # De-risk thresholds/scale are tf-invariant (only the windows/dwell scale).
    for tf in ("4h", "1h"):
        for cell in slice_dict[tf]:
            b = base[cell["variant"]]
            for key in ("z_enter", "z_exit", "corr_abs_floor", "derisk_scale"):
                assert cell[key] == b[key], (tf, key)
