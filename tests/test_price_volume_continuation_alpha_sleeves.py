"""Deterministic tests for PriceVolumeCorrContinuationStrategy (wave-2b Lane 9).

Direct class import only (no ``@register`` on this lane).  The BUILD GATE runs
the REAL incumbents on ONE shared price path where cases A and B have IDENTICAL
closes but different volume, so every pure-price incumbent is provably blind to
the A-vs-B distinction this sleeve trades:

- Case A (healthy): volume-change is sign-aligned with the return innovations ->
  return/volume-change correlation ~= +1.
- Case B (exhaustion): volume shrinks and anti-aligns -> correlation ~= -1.

The gate asserts materially different EMITTED actions (anti-strawman /
anti-vacuous rules) versus the pure-price trend sleeves
(``LowTurnoverTrendPersistenceStrategy`` acts identically on A and B),
``VolumeClockMomentumRiderStrategy`` (live AND blind -- LONG on both), and
``CrossSectionalFlowShareRotationStrategy`` (empty book when every symbol carries
an identical volume share).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.flow_share import cross_sectional_share
from lumina_quant.strategies.flow_share_rotation_alpha_sleeves import (
    CrossSectionalFlowShareRotationStrategy,
    _rolling_z,
)
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.strategies.price_volume_continuation_alpha_sleeves import (
    _PRICE_VOLUME_CONTINUATION_SLICE,
    PriceVolumeCorrContinuationStrategy,
    _return_volume_change_correlation,
)
from lumina_quant.strategies.volume_clock_alpha_sleeves import VolumeClockMomentumRiderStrategy
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# shared synthetic bars
# --------------------------------------------------------------------------- #

_N = 210
_BASE = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday -> the weekly clock advances
_TS = [(_BASE + timedelta(days=t)).isoformat() for t in range(_N)]
_SYM = "X/USDT"

# SHARED PRICE PATH: monotone rising, alternating per-bar log returns +0.8%/+0.2%
# (mean +0.5%/bar), IDENTICAL in cases A and B.
_R = [0.008 if t % 2 == 0 else 0.002 for t in range(_N - 1)]
_CLOSES = [100.0]
for _x in _R:
    _CLOSES.append(_CLOSES[-1] * math.exp(_x))
# Case A: volume-change sign-aligned with the return innovation.
_VOL_A = [1000.0] + [1000.0 * (1 + 8 * (_R[t] - 0.005)) for t in range(_N - 1)]
# Case B: shrinking AND anti-aligned (volume never near zero).
_VOL_B = [1000.0] + [1000.0 * (0.995**t) * (1 - 8 * (_R[t] - 0.005)) for t in range(_N - 1)]


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _market_event(symbol: str, t: int, close: float, volume: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        symbol=symbol,
        time=_TS[t],
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _feed_one(
    strategy: Any, symbol: str, closes: list[float], volumes: list[float | None], n: int = _N
) -> None:
    for t in range(n):
        strategy.calculate_signals(_market_event(symbol, t, closes[t], volumes[t]))


def _stream(items: list[Any], symbol: str | None = None) -> list[str]:
    return [str(s.signal_type).upper() for s in items if symbol is None or s.symbol == symbol]


def _new_sleeve(**overrides: Any) -> PriceVolumeCorrContinuationStrategy:
    params: dict[str, Any] = dict(
        corr_window=70,
        mom_window_weeks=8,
        bars_per_week=7,
        corr_entry=0.25,
        corr_exit=0.0,
        min_hold_decisions=4,
        cooldown_decisions=2,
        vol_window=30,
        target_vol=0.20,
        allow_short=True,
        min_price=0.0,
    )
    params.update(overrides)
    return PriceVolumeCorrContinuationStrategy(_Bars([_SYM]), _Queue(), **params)


# --------------------------------------------------------------------------- #
# Stage-1 premises
# --------------------------------------------------------------------------- #


def test_stage1_case_a_confirms_case_b_exhausts() -> None:
    corr_a = _return_volume_change_correlation(_CLOSES, _VOL_A, window=70)
    corr_b = _return_volume_change_correlation(_CLOSES, _VOL_B, window=70)
    assert corr_a is not None and corr_a >= 0.8, corr_a
    assert corr_b is not None and corr_b <= -0.8, corr_b
    assert min(_VOL_B) > 100.0, min(_VOL_B)  # volume never near zero in case B


# --------------------------------------------------------------------------- #
# BUILD GATE leg (1): divergent action on an IDENTICAL price path
# --------------------------------------------------------------------------- #


def test_gate_candidate_diverges_on_identical_price_path() -> None:
    healthy = _new_sleeve()
    _feed_one(healthy, _SYM, _CLOSES, _VOL_A)
    assert _stream(healthy.events.items) == ["LONG"], _stream(healthy.events.items)

    exhausted = _new_sleeve()
    _feed_one(exhausted, _SYM, _CLOSES, _VOL_B)
    # Same price path, exhaustion volume -> no entry at all.
    assert _stream(exhausted.events.items) == [], _stream(exhausted.events.items)

    # Case-A prefix seeds a LONG; switching to case-B volume forces an EXIT once
    # the correlation window fills (past the hard min-hold).
    vol_switch: list[float] = _VOL_A[:100] + [
        1000.0 * (0.995**t) * (1 - 8 * (_R[t] - 0.005)) for t in range(99, _N - 1)
    ]
    switched = _new_sleeve()
    _feed_one(switched, _SYM, _CLOSES, vol_switch)
    assert _stream(switched.events.items) == ["LONG", "EXIT"], _stream(switched.events.items)


def test_gate_pure_price_trend_sleeve_is_blind_to_a_vs_b() -> None:
    """LowTurnoverTrendPersistence (OHLC only) emits IDENTICAL actions on A and B,
    while it is LIVE (enters LONG) -- so the equality is not vacuous."""

    def _run(volumes: list[float]) -> list[tuple[str, str]]:
        sleeve = LowTurnoverTrendPersistenceStrategy(
            _Bars([_SYM]),
            _Queue(),
            tsmom_short=28,
            tsmom_mid=56,
            tsmom_long=84,
            efficiency_period=20,
            min_efficiency=0.30,
            adx_period=14,
            adx_min=20.0,
            vol_persist_fast=16,
            vol_persist_slow=64,
            vol_persist_max=1.5,
            min_hold_bars=36,
            cooldown_bars=4,
            vol_window=56,
            allow_short=True,
            min_price=0.0,
        )
        _feed_one(sleeve, _SYM, _CLOSES, volumes)
        return [(s.symbol, str(s.signal_type)) for s in sleeve.events.items]

    on_a = _run(_VOL_A)
    on_b = _run(_VOL_B)
    assert on_a, "the pure-price incumbent must be live (emit an action) on the path"
    assert on_a == on_b, (on_a, on_b)


# --------------------------------------------------------------------------- #
# BUILD GATE leg (2): vs VolumeClockMomentumRider (live + structurally blind)
# --------------------------------------------------------------------------- #


def test_gate_vs_volume_clock_is_live_and_blind() -> None:
    for volumes in (_VOL_A, _VOL_B):
        clock = VolumeClockMomentumRiderStrategy(
            _Bars([_SYM]),
            _Queue(),
            vbar_ref_window=50,
            vbar_mult=1.0,
            vmom_bars=10,
            vmom_entry=0.02,
            max_stale_bars=50,
            allow_short=True,
            min_price=0.0,
        )
        _feed_one(clock, _SYM, _CLOSES, volumes)
        # Incumbent-LIVE control: the volume clock is ticking and directional in
        # BOTH cases -- it trades a healthy and an exhausted advance identically.
        assert clock._clock_is_live(_SYM) is True
        assert clock._vclock_direction(_SYM) == "LONG"
        assert _stream(clock.events.items), "VolumeClock must emit a non-empty book"

    # And the candidate diverges: it trades case A but abstains on case B.
    candidate_a = _new_sleeve()
    _feed_one(candidate_a, _SYM, _CLOSES, _VOL_A)
    candidate_b = _new_sleeve()
    _feed_one(candidate_b, _SYM, _CLOSES, _VOL_B)
    assert _stream(candidate_a.events.items) == ["LONG"]
    assert _stream(candidate_b.events.items) == []


# --------------------------------------------------------------------------- #
# BUILD GATE leg (3): vs CrossSectionalFlowShareRotation (pinned-branch silence)
# --------------------------------------------------------------------------- #


def test_gate_vs_flow_share_constant_share_silence() -> None:
    # A universe where every symbol carries the IDENTICAL price+volume path makes
    # each turnover share exactly 1/5 forever.
    assert cross_sectional_share(100.0, 500.0) == 0.2
    # A constant share series hits the sigma<=EPS branch of _rolling_z -> 0.0.
    assert _rolling_z([0.2] * 20) == 0.0

    syms = [f"S{i}/USDT" for i in range(5)]
    flow = CrossSectionalFlowShareRotationStrategy(
        _Bars(syms),
        _Queue(),
        share_z_window=20,
        confirm_window=3,
        share_z_entry=1.0,
        top_n=3,
        vol_window=8,
        allow_short=True,
        min_symbols=5,
        min_price=0.0,
    )
    for t in range(_N):
        for sym in syms:
            flow.calculate_signals(_market_event(sym, t, _CLOSES[t], _VOL_A[t]))
    # Flow-share emits nothing (its share_z collapses to 0 for every symbol).
    assert [s for s in flow.events.items if str(s.signal_type).upper() != "EXIT"] == []

    # The candidate, on the same case-A path, is LONG.
    candidate = _new_sleeve()
    _feed_one(candidate, _SYM, _CLOSES, _VOL_A)
    assert _stream(candidate.events.items) == ["LONG"]


# --------------------------------------------------------------------------- #
# indicator unit tests (lane-local numeric)
# --------------------------------------------------------------------------- #


def test_correlation_guards_and_bounds() -> None:
    # Insufficient history -> None.
    assert _return_volume_change_correlation([1.0, 2.0], [1.0, 2.0], window=70) is None
    # Constant close (zero return variance) -> None.
    assert _return_volume_change_correlation([100.0] * 80, list(range(80)), window=70) is None
    # Constant volume (zero volume-change variance) -> None.
    assert _return_volume_change_correlation(_CLOSES, [1000.0] * _N, window=70) is None
    # Zero / non-positive volume is floored, never raises, stays bounded.
    corr = _return_volume_change_correlation(_CLOSES, [0.0] + [v for v in _VOL_A[1:]], window=70)
    assert corr is None or -1.0 <= corr <= 1.0


# --------------------------------------------------------------------------- #
# lane invariants
# --------------------------------------------------------------------------- #


def test_run_twice_bit_identical() -> None:
    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        sleeve = _new_sleeve()
        _feed_one(sleeve, _SYM, _CLOSES, _VOL_A)
        return [
            (s.symbol, s.signal_type, s.strength, dict(s.metadata or {}))
            for s in sleeve.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal on case A"


def test_state_roundtrip_lossless() -> None:
    sleeve = _new_sleeve()
    _feed_one(sleeve, _SYM, _CLOSES, _VOL_A)
    snapshot = sleeve.get_state()
    restored = _new_sleeve()
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    assert list(restored._state[_SYM].closes) == list(sleeve._state[_SYM].closes)
    assert list(restored._state[_SYM].volumes) == list(sleeve._state[_SYM].volumes)
    assert restored._state[_SYM].mode == sleeve._state[_SYM].mode
    assert restored._state[_SYM].bars_held == sleeve._state[_SYM].bars_held


def test_adversarial_set_state_never_raises() -> None:
    sleeve = _new_sleeve()
    sleeve.set_state(None)  # type: ignore[arg-type]
    sleeve.set_state("not a dict")  # type: ignore[arg-type]
    sleeve.set_state(12345)  # type: ignore[arg-type]
    sleeve.set_state([])  # type: ignore[arg-type]
    sleeve.set_state({"symbol_state": "not a dict"})
    sleeve.set_state({"symbol_state": {_SYM: "not a dict either"}})
    sleeve.set_state({"symbol_state": {_SYM: {"closes": 12345}}})
    sleeve.set_state({"symbol_state": {_SYM: {"volumes": {"nested": "dict"}}}})
    sleeve.set_state(
        {
            "symbol_state": {
                _SYM: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
                    "volumes": ["bad", 5.0, None, float("nan")],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "bars_since_exit": [1, 2],
                    "last_bar_key": 123,
                    "last_decision_week": None,
                    "score": [1, 2, 3],
                }
            }
        }
    )
    assert sleeve._state[_SYM].mode in {"OUT", "LONG", "SHORT"}
    # Still functions afterward.
    _feed_one(sleeve, _SYM, _CLOSES, _VOL_A)


def test_never_raise_on_degenerate_input() -> None:
    sleeve = _new_sleeve()
    sleeve.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    sleeve.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    sleeve.calculate_signals(_market_event(_SYM, 0, 0.0, 1000.0))  # non-positive close
    sleeve.calculate_signals(_market_event(_SYM, 1, float("nan"), 1000.0))
    sleeve.calculate_signals(_market_event(_SYM, 2, float("inf"), None))  # None volume
    sleeve.calculate_signals(_market_event(_SYM, 3, 100.0, 0.0))  # zero volume
    sleeve.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_TS[4], bars_1s={}))
    sleeve.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_TS[5], bars_1s={_SYM: []}))
    assert [s for s in sleeve.events.items if str(s.signal_type).upper() != "EXIT"] == []


def test_zero_and_none_volume_never_raise_across_full_feed() -> None:
    sleeve = _new_sleeve()
    volumes: list[float | None] = [0.0 if t % 3 == 0 else None for t in range(_N)]
    _feed_one(sleeve, _SYM, _CLOSES, volumes)  # must not raise


def test_short_only_disallows_short_entry() -> None:
    # A monotone FALLING path with confirming volume, allow_short=False -> no entry.
    falling = [100.0]
    for _x in _R:
        falling.append(falling[-1] / math.exp(_x))
    # confirming volume for a downtrend: down-bars carry the volume (corr < 0),
    # so dir(-1) * corr(<0) > 0 -> the score qualifies, but shorts are disabled.
    vols = [1000.0] + [1000.0 * (1 - 8 * (_R[t] - 0.005)) for t in range(_N - 1)]
    sleeve = _new_sleeve(allow_short=False)
    _feed_one(sleeve, _SYM, falling, vols)
    assert _stream(sleeve.events.items) == [], _stream(sleeve.events.items)


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = PriceVolumeCorrContinuationStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "corr_window",
        "mom_window_weeks",
        "corr_entry",
        "corr_exit",
        "min_hold_decisions",
        "cooldown_decisions",
        "vol_window",
        "allow_short",
        "target_allocation",
    ):
        assert required in schema


def test_decision_cadence_at_least_30m() -> None:
    assert PriceVolumeCorrContinuationStrategy.decision_cadence_seconds >= 1800


def test_slice_multi_timeframe_keys_and_bounds() -> None:
    """Pin the 1d/4h/1h slice: mirrored variants + keys, scaled cells in-bounds.

    Guards against a silent schema clamp (out-of-bounds slice value would be
    coerced by ``resolve_params_from_schema`` so the written value would NOT be
    the effective one) and pins the weekly DECISION-denominated params as
    tf-invariant while the bar-denominated windows scale wall-clock.
    """
    slice_dict = _PRICE_VOLUME_CONTINUATION_SLICE
    assert set(slice_dict) == {"1d", "4h", "1h"}
    counts = {tf: len(cells) for tf, cells in slice_dict.items()}
    assert len(set(counts.values())) == 1, counts
    base = {cell["variant"]: cell for cell in slice_dict["1d"]}
    schema = PriceVolumeCorrContinuationStrategy.get_param_schema()
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
    # Weekly DECISION clock + week horizon are timestamp-based -> tf-invariant.
    for tf in ("4h", "1h"):
        for cell in slice_dict[tf]:
            b = base[cell["variant"]]
            assert cell["min_hold_decisions"] == b["min_hold_decisions"]
            assert cell["cooldown_decisions"] == b["cooldown_decisions"]
            assert cell["mom_window_weeks"] == b["mom_window_weeks"]
    # ``bars_per_week`` scales x6 (4h) / x24 (1h) so the momentum horizon holds.
    for cell in slice_dict["4h"]:
        assert cell["bars_per_week"] == 6 * base[cell["variant"]]["bars_per_week"]
    for cell in slice_dict["1h"]:
        assert cell["bars_per_week"] == 24 * base[cell["variant"]]["bars_per_week"]
