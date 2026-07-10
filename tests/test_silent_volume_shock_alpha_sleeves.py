"""Deterministic build-gate + hygiene tests for the silent-volume-shock lane.

Direct class import only (no ``@register`` on this lane).  The build gate drives
the REAL incumbents through their full decision paths on identical synthetic
bars and asserts a materially DIVERGENT emitted action (never a different
internal score alone).  All randomness is a small seeded LCG (no ``random``
module), so every run is bit-for-bit reproducible.

The load-bearing fixture is F1: a BASELINE tape (close alternating +/-0.10%
around 100, volume 1000, close at the exact mid of a tight symmetric band so the
close-location value is 0 every bar) with a SILENT VOLUME SHOCK at bar 101
(volume 12000, log return -0.05%, same tight range) followed by two flat bars
and a +2.0% RESOLUTION bar at bar 104.

- LEG A  lead structure: zero events at the shock bar, exactly one LONG at the
         resolution bar, ARMED with a counting-down window in between.
- LEG B  vs AbnormalReturnContinuationStrategy (KEEP/DROP): removing only the
         bar-101 abnormal volume (F2) silences the candidate while the return
         follower fires LONG on BOTH F1 and F2 -- the volume antecedent is
         load-bearing.
- LEG C  vs LiquidityShockReversionStrategy: the shock incumbent's complementary
         volume-AND-range-AND-return conjunction stays silent on F1's quiet
         shock, while it fires on a violent shock (F3) where the candidate's
         quiet veto refuses to arm.
- LEG D  vs PriceVolumeCorrContinuationStrategy: silent on F1 (its confirmation
         score is below entry -- pinned), fires on a volume-confirmed trend
         where the candidate never arms.
- LEG E  vs VolumeClockMomentumRiderStrategy: silent on F1 (flat volume-time
         momentum), fires on a monotone tape where the candidate never arms.
- LEG F  vs CrossSectionalFlowShareRotationStrategy + CLV accumulation
         (KEEP/DROP): flow-share SHORTS the shock bar (contemporaneous blow-off)
         while the candidate LONGS the LATER resolution; the candidate is
         invariant to close-location where the CLV geometry sleeve is not.
- LEG G  hygiene/expiry: a window that never resolves trades nothing; an adverse
         cross inside the min-hold does not exit; determinism / state roundtrip /
         adversarial / degenerate / schema.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.abnormal_return_continuation import (
    AbnormalReturnContinuationStrategy,
)
from lumina_quant.strategies.clv_accumulation_alpha_sleeves import (
    CrossSectionalCloseLocationAccumulationStrategy,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    LiquidityShockReversionStrategy,
    _bars_per_year_from_spacing,
)
from lumina_quant.strategies.flow_share_rotation_alpha_sleeves import (
    CrossSectionalFlowShareRotationStrategy,
)
from lumina_quant.strategies.price_volume_continuation_alpha_sleeves import (
    PriceVolumeCorrContinuationStrategy,
)
from lumina_quant.strategies.silent_volume_shock_alpha_sleeves import (
    SilentVolumeShockResolutionStrategy,
)
from lumina_quant.strategies.volume_clock_alpha_sleeves import (
    VolumeClockMomentumRiderStrategy,
)
from lumina_quant.tuning import HyperParam

_START = datetime(2026, 1, 1, tzinfo=UTC)

# OHLCV bar tuple: (open, high, low, close, volume).
_Bar = tuple[float, float, float, float, float]


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _LatestBars:
    """A ``bars`` shim exposing ``get_latest_bar_*`` for the ARC incumbent."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)
        self._latest: dict[str, tuple[datetime, _Bar]] = {}

    def update(self, symbol: str, moment: datetime, bar: _Bar) -> None:
        self._latest[symbol] = (moment, bar)

    def get_latest_bar_datetime(self, symbol: str) -> datetime | None:
        pair = self._latest.get(symbol)
        return pair[0] if pair else None

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        pair = self._latest.get(symbol)
        if pair is None:
            return None
        o, h, low, c, v = pair[1]
        return {"open": o, "high": h, "low": low, "close": c, "volume": v}.get(field)


def _market_event(symbol: str, i: int, bar: _Bar) -> SimpleNamespace:
    o, h, low, c, v = bar
    return SimpleNamespace(
        type="MARKET",
        time=_START + timedelta(days=i),
        symbol=symbol,
        open=o,
        high=h,
        low=low,
        close=c,
        volume=v,
    )


def _window_event(i: int, rows: dict[str, _Bar]) -> SimpleNamespace:
    t = _START + timedelta(days=i)
    bars_1s = {
        sym: [{"time": t, "open": o, "high": h, "low": low, "close": c, "volume": v}]
        for sym, (o, h, low, c, v) in rows.items()
    }
    return SimpleNamespace(type="MARKET_WINDOW", time=t, bars_1s=bars_1s)


def _feed_single(strategy: Any, symbol: str, bars: list[_Bar], *, start: int = 0) -> None:
    for offset, bar in enumerate(bars):
        strategy.calculate_signals(_market_event(symbol, start + offset, bar))


def _feed_window(strategy: Any, series: dict[str, list[_Bar]], symbols: list[str]) -> None:
    n = len(series[symbols[0]])
    for i in range(n):
        strategy.calculate_signals_window(
            _window_event(i, {sym: series[sym][i] for sym in symbols}), None
        )


def _entries(strategy: Any) -> list[Any]:
    return [s for s in strategy.events.items if s.signal_type in {"LONG", "SHORT"}]


def _entries_for(strategy: Any, symbol: str) -> list[Any]:
    return [s for s in _entries(strategy) if s.symbol == symbol]


# --------------------------------------------------------------------------- #
# deterministic fixtures
# --------------------------------------------------------------------------- #


def _baseline(n: int = 101, *, base: float = 100.0, amp: float = 0.001) -> list[_Bar]:
    """Alternating +/-``amp`` closes; close at the exact mid of a symmetric band."""
    bars: list[_Bar] = []
    for i in range(n):
        c = base * (1.0 + amp if i % 2 == 0 else 1.0 - amp)
        bars.append((c, c * (1.0 + amp), c * (1.0 - amp), c, 1000.0))
    return bars


def _f1(*, shock_volume: float = 12000.0, resolution: bool = True) -> list[_Bar]:
    """BASELINE + silent shock at bar 101 + flat 102/103 + +2% resolution at 104."""
    bars = _baseline(101)
    c100 = bars[100][3]
    c101 = c100 * math.exp(-0.0005)  # -0.05% log return, quiet
    amp = 0.001
    bars.append((c101, c101 * (1.0 + amp), c101 * (1.0 - amp), c101, shock_volume))
    bars.append((c101, c101 * (1.0 + amp), c101 * (1.0 - amp), c101, 1000.0))  # flat 102
    bars.append((c101, c101 * (1.0 + amp), c101 * (1.0 - amp), c101, 1000.0))  # flat 103
    c104 = c101 * 1.02 if resolution else c101
    bars.append((c104, c104 * (1.0 + amp), c104 * (1.0 - amp), c104, 1000.0))  # resolution 104
    for i in range(10):
        c = c104 * (1.0 + amp if i % 2 == 0 else 1.0 - amp)
        bars.append((c, c * (1.0 + amp), c * (1.0 - amp), c, 1000.0))
    return bars


def _f3_violent() -> list[_Bar]:
    """BASELINE + a VIOLENT shock at bar 101 (volume 12000, -3%, wide range)."""
    bars = _baseline(101)
    c100 = bars[100][3]
    c101 = c100 * 0.97
    bars.append((c101, c101 * 1.03, c101 * 0.97, c101, 12000.0))
    for i in range(12):
        c = c101 * (1.001 if i % 2 == 0 else 0.999)
        bars.append((c, c * 1.001, c * 0.999, c, 1000.0))
    return bars


def _filler(n: int, *, base: float = 100.0, amp: float = 0.001) -> list[_Bar]:
    return _baseline(n, base=base, amp=amp)


def _to_close_at_high(bars: list[_Bar]) -> list[_Bar]:
    """Reposition each bar's H/L band so the (unchanged) close sits at the HIGH.

    The close value and the log range width ``log(H/L)`` are preserved exactly,
    so the candidate -- which reads only close, volume, and range width -- is
    invariant, while the bar's close-location value flips from 0 to +1.
    """
    out: list[_Bar] = []
    for o, h, low, c, v in bars:
        width = math.log(h / low)
        out.append((o, c, c * math.exp(-width), c, v))
    return out


_CAND_KW: dict[str, Any] = dict(
    shock_window=60,
    v_shock_z=2.0,
    quiet_ret_z=0.5,
    quiet_range_z=1.0,
    resolution_max_bars=10,
    resolution_ret_mult=1.0,
    min_hold_bars=7,
    max_hold_bars=21,
    cooldown_bars=5,
    min_history_bars=60,
    base_allocation=0.02,
    min_price=0.01,
)


def _candidate(symbols: list[str], **overrides: Any) -> SilentVolumeShockResolutionStrategy:
    return SilentVolumeShockResolutionStrategy(
        _Bars(symbols), _Queue(), **dict(_CAND_KW, **overrides)
    )


# --------------------------------------------------------------------------- #
# Stage-1 premise assertions on the F1 fixture arithmetic
# --------------------------------------------------------------------------- #


def test_stage1_f1_shock_bar_is_a_quiet_volume_shock() -> None:
    bars = _f1()
    strategy = _candidate(["A/USDT"])
    # Feed up to and including the shock bar (index 101).
    _feed_single(strategy, "A/USDT", bars[:102])
    item = strategy._state["A/USDT"]
    vz, rz, range_z = strategy._shock_features(item)
    assert vz is not None and vz >= 3.0  # abnormal turnover
    assert rz is not None and rz <= 0.5  # quiet price
    assert range_z <= 1.0  # quiet range
    # The close sits at the exact mid of the band on every bar (CLV == 0).
    for _o, h, low, c, _v in bars:
        assert math.isclose(2.0 * c, h + low, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
# LEG A -- lead structure (candidate ACTS on the lagged resolution only)
# --------------------------------------------------------------------------- #


def test_leg_a_lead_structure_zero_at_shock_one_long_at_resolution() -> None:
    bars = _f1()
    strategy = _candidate(["A/USDT"])
    # Through the shock bar and the two flat bars: ARMED, no entry, counting down.
    _feed_single(strategy, "A/USDT", bars[:104])
    assert _entries(strategy) == []
    item = strategy._state["A/USDT"]
    assert item.armed and item.mode == "OUT"
    assert 0 < item.arm_age < strategy.resolution_max_bars
    snap = strategy.get_state()["symbol_state"]["A/USDT"]
    assert snap["armed"] is True and snap["arm_close"] is not None

    # The resolution bar (index 104) triggers exactly one LONG with the +sign.
    _feed_single(strategy, "A/USDT", bars[104:], start=104)
    entries = _entries(strategy)
    assert [s.signal_type for s in entries] == ["LONG"]
    meta = entries[0].metadata or {}
    assert meta.get("reason") == "silent_shock_resolution"
    assert meta.get("resolution_cum_return") is not None and meta["resolution_cum_return"] > 0.0
    assert int(meta.get("resolution_bars", 0)) == 3


# --------------------------------------------------------------------------- #
# LEG B -- vs AbnormalReturnContinuationStrategy (KEEP/DROP: volume-blindness)
# --------------------------------------------------------------------------- #

_ARC_KW: dict[str, Any] = dict(return_z_window=20, entry_z=1.5, hold_bars=2, allow_short=True)


def _run_arc(bars: list[_Bar]) -> list[str]:
    latest = _LatestBars(["A/USDT"])
    arc = AbnormalReturnContinuationStrategy(latest, _Queue(), **_ARC_KW)
    for i, bar in enumerate(bars):
        latest.update("A/USDT", _START + timedelta(days=i), bar)
        arc.calculate_signals(_market_event("A/USDT", i, bar))
    return [s.signal_type for s in _entries(arc)]


def test_leg_b_volume_antecedent_is_load_bearing_vs_return_continuation() -> None:
    f1 = _f1(shock_volume=12000.0)
    f2 = _f1(shock_volume=1000.0)  # identical except the shock bar's volume

    # Candidate: arms and resolves on F1, but goes SILENT on F2 (no volume shock).
    cand_f1 = _candidate(["A/USDT"])
    _feed_single(cand_f1, "A/USDT", f1)
    assert [s.signal_type for s in _entries(cand_f1)] == ["LONG"]

    cand_f2 = _candidate(["A/USDT"])
    _feed_single(cand_f2, "A/USDT", f2)
    assert _entries(cand_f2) == []

    # Incumbent-LIVE control + volume-blindness: the return follower sees only the
    # +2% resolution bar (z >> 1.5) and fires LONG on BOTH feeds identically.
    assert _run_arc(f1) == ["LONG"]
    assert _run_arc(f2) == ["LONG"]


# --------------------------------------------------------------------------- #
# LEG C -- vs LiquidityShockReversionStrategy (complementary conjunction)
# --------------------------------------------------------------------------- #

_LSR_KW: dict[str, Any] = dict(
    volume_window=20,
    range_window=20,
    volume_shock_z=2.0,
    range_shock_z=1.5,
    return_shock_pct=0.012,
    stop_loss_pct=0.0,
    min_price=0.01,
)


def _run_lsr(bars: list[_Bar]) -> list[str]:
    lsr = LiquidityShockReversionStrategy(_Bars(["A/USDT"]), _Queue(), **_LSR_KW)
    _feed_single(lsr, "A/USDT", bars)
    return [s.signal_type for s in _entries(lsr)]


def test_leg_c_liquidity_shock_conjunction_is_complementary() -> None:
    f1 = _f1()
    # Quiet shock: the incumbent's volume-AND-range-AND-return conjunction fails
    # (|ret|=0.0005 < 0.012 and range z ~ 0 < 1.5) -> silent, while the candidate
    # arms and LONGs the resolution.
    assert _run_lsr(f1) == []
    cand = _candidate(["A/USDT"])
    _feed_single(cand, "A/USDT", f1)
    assert [s.signal_type for s in _entries(cand)] == ["LONG"]

    # Positive control (violent shock F3): the incumbent fires its fade entry;
    # the candidate's quiet veto (|ret| and range z both large) refuses to arm.
    f3 = _f3_violent()
    assert _run_lsr(f3)  # incumbent-LIVE
    cand_f3 = _candidate(["A/USDT"])
    _feed_single(cand_f3, "A/USDT", f3)
    assert _entries(cand_f3) == []
    assert cand_f3._state["A/USDT"].armed is False


# --------------------------------------------------------------------------- #
# LEG D -- vs PriceVolumeCorrContinuationStrategy (not a contemporaneous corr)
# --------------------------------------------------------------------------- #

_PVCC_KW: dict[str, Any] = dict(
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
    min_price=0.01,
)


def _pvcc_bull(n: int = 150) -> list[_Bar]:
    """Rising closes with volume EXPANDING in the up direction (corr > 0)."""
    bars: list[_Bar] = []
    c = 100.0
    v = 1000.0
    for i in range(n):
        if i % 2 == 0:
            c *= 1.02
            v *= 1.5
        else:
            c *= 1.002
            v *= 0.9
        bars.append((c, c * 1.001, c * 0.999, c, v))
    return bars


def test_leg_d_price_volume_corr_continuation_is_not_the_lead_structure() -> None:
    f1 = _f1()
    pvcc = PriceVolumeCorrContinuationStrategy(_Bars(["A/USDT"]), _Queue(), **_PVCC_KW)
    _feed_single(pvcc, "A/USDT", f1)
    assert _entries(pvcc) == []
    # Pin the WHY: its confirmation score never reaches corr_entry.
    _dir, _corr, score = pvcc._confirmation(pvcc._state["A/USDT"])
    assert score is None or abs(score) < _PVCC_KW["corr_entry"]

    # Positive control: a volume-confirmed uptrend makes the incumbent LONG,
    # while the candidate never arms (no bar is a price-flat volume shock).
    bull = _pvcc_bull()
    pvcc_bull = PriceVolumeCorrContinuationStrategy(_Bars(["A/USDT"]), _Queue(), **_PVCC_KW)
    _feed_single(pvcc_bull, "A/USDT", bull)
    assert [s.signal_type for s in _entries(pvcc_bull)][:1] == ["LONG"]
    cand_bull = _candidate(["A/USDT"])
    _feed_single(cand_bull, "A/USDT", bull)
    assert _entries(cand_bull) == []


# --------------------------------------------------------------------------- #
# LEG E -- vs VolumeClockMomentumRiderStrategy (not volume-time momentum)
# --------------------------------------------------------------------------- #

_RIDER_KW: dict[str, Any] = dict(
    vbar_ref_window=20,
    vbar_mult=1.0,
    vmom_bars=10,
    vmom_entry=0.05,
    max_stale_bars=50,
    vol_window=24,
    target_vol=0.04,
    max_hold_bars=60,
    allow_short=True,
    atr_period=10,
)


def _monotone(n: int = 140, *, step: float = 0.008) -> list[_Bar]:
    bars: list[_Bar] = []
    c = 100.0
    for _ in range(n):
        c *= 1.0 + step
        bars.append((c, c * 1.001, c * 0.999, c, 1000.0))
    return bars


def test_leg_e_volume_clock_rider_is_not_the_lead_structure() -> None:
    f1 = _f1()
    rider = VolumeClockMomentumRiderStrategy(_Bars(["A/USDT"]), _Queue(), **_RIDER_KW)
    _feed_single(rider, "A/USDT", f1)
    assert _entries(rider) == []
    # Pin the WHY: the volume-clock momentum is flat and the entry decision empty.
    assert rider._vclock_direction("A/USDT") == ""
    momentum = rider._vclock_momentum("A/USDT")
    assert momentum is None or abs(momentum) < _RIDER_KW["vmom_entry"]

    # Positive control: a monotone tape makes the rider LONG while the candidate
    # never arms (every bar carries a large return -> quiet veto).
    mono = _monotone()
    rider_mono = VolumeClockMomentumRiderStrategy(_Bars(["A/USDT"]), _Queue(), **_RIDER_KW)
    _feed_single(rider_mono, "A/USDT", mono)
    assert [s.signal_type for s in _entries(rider_mono)][:1] == ["LONG"]
    cand_mono = _candidate(["A/USDT"])
    _feed_single(cand_mono, "A/USDT", mono)
    assert _entries(cand_mono) == []


# --------------------------------------------------------------------------- #
# LEG F -- vs flow-share rotation + CLV accumulation (KEEP/DROP)
# --------------------------------------------------------------------------- #

_FLOW_KW: dict[str, Any] = dict(
    share_z_window=20,
    confirm_window=1,
    share_z_entry=1.0,
    blowoff_extremeness=0.85,
    top_n=8,
    vol_window=8,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)

_CLV_KW: dict[str, Any] = dict(
    accumulation_window_bars=20,
    momentum_window_bars=20,
    nearness_window_bars=60,
    min_history_bars=25,
    vol_window=10,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_decisions=0,
    allow_short=True,
    min_symbols=5,
    min_price=0.01,
)

_LEG_F_SYMBOLS = ["SHOCK/USDT", "C0/USDT", "C1/USDT", "C2/USDT", "C3/USDT"]


def _leg_f_series() -> dict[str, list[_Bar]]:
    shock = _f1(shock_volume=12000.0)
    n = len(shock)
    series: dict[str, list[_Bar]] = {"SHOCK/USDT": shock}
    for symbol in _LEG_F_SYMBOLS[1:]:
        series[symbol] = _filler(n)
    return series


def test_leg_f_flowshare_shorts_shock_bar_where_candidate_longs_resolution() -> None:
    series = _leg_f_series()

    flow = CrossSectionalFlowShareRotationStrategy(_Bars(_LEG_F_SYMBOLS), _Queue(), **_FLOW_KW)
    _feed_window(flow, series, _LEG_F_SYMBOLS)
    shock_entries = _entries_for(flow, "SHOCK/USDT")
    # A single, contemporaneous BLOW-OFF short on the shock bar (long_ok False:
    # confirm_return < 0; blow-off branch fires).
    assert [s.signal_type for s in shock_entries] == ["SHORT"]
    meta = shock_entries[0].metadata or {}
    assert (
        meta.get("extremeness") is not None
        and meta["extremeness"] >= _FLOW_KW["blowoff_extremeness"]
    )
    assert meta.get("confirm_return") is not None and meta["confirm_return"] < 0.0

    # Candidate: a single LAGGED LONG on the resolution -- opposite side AND
    # different bar from the incumbent's contemporaneous short.
    candidate = SilentVolumeShockResolutionStrategy(_Bars(_LEG_F_SYMBOLS), _Queue(), **_CAND_KW)
    _feed_window(candidate, series, _LEG_F_SYMBOLS)
    assert [s.signal_type for s in _entries_for(candidate, "SHOCK/USDT")] == ["LONG"]


def test_leg_f_candidate_is_close_location_invariant_where_clv_separates() -> None:
    # Candidate geometry-invariance: relocating the close to the top of the band
    # (same close value + range width) leaves every emitted signal bit-identical.
    mid = _f1()
    hi = _to_close_at_high(mid)

    def _sigs(bars: list[_Bar]) -> list[tuple[str, str, Any]]:
        strategy = _candidate(["A/USDT"])
        _feed_single(strategy, "A/USDT", bars)
        return [
            (s.symbol, s.signal_type, (s.metadata or {}).get("resolution_cum_return"))
            for s in strategy.events.items
        ]

    assert _sigs(mid) == _sigs(hi)

    # Incumbent-LIVE: the CLV geometry sleeve rank-separates an accumulation
    # (close-at-high) vs distribution (close-at-low) book the candidate cannot see.
    gen = _lcg_stream(5)
    n = 120
    pinned: dict[str, list[_Bar]] = {}
    for k, symbol in enumerate(_LEG_F_SYMBOLS):
        bars: list[_Bar] = []
        c = 100.0
        for _ in range(n):
            c *= 1.0 + (next(gen) - 0.5) * 0.01
            if k % 2 == 0:  # accumulation: close pinned at the high
                h, low = c, c * 0.97
            else:  # distribution: close pinned at the low
                h, low = c * 1.03, c
            bars.append(((h + low) / 2.0, h, low, c, 1000.0))
        pinned[symbol] = bars
    clv = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(_LEG_F_SYMBOLS), _Queue(), **_CLV_KW
    )
    _feed_window(clv, pinned, _LEG_F_SYMBOLS)
    assert _entries(clv), "CLV must trade the accumulation/distribution book"


# --------------------------------------------------------------------------- #
# LEG G -- expiry + hygiene (section 0.1)
# --------------------------------------------------------------------------- #


def test_leg_g_window_expiry_trades_nothing() -> None:
    # F4: F1 without the resolution bar -> the arming window expires with no trade.
    f4 = _f1(resolution=False)
    strategy = _candidate(["A/USDT"])
    _feed_single(strategy, "A/USDT", f4)
    assert _entries(strategy) == []
    assert strategy._state["A/USDT"].armed is False


def test_leg_g_adverse_cross_inside_min_hold_does_not_exit() -> None:
    # Enter LONG on the resolution, then cross back below the arming close while
    # still inside the min-hold -> the position is HELD (no EXIT emitted yet).
    bars = _f1()
    below = bars[104][3] * 0.999  # just under the resolution close, above arm ref
    arm_ref = bars[101][3]
    adverse = min(below, arm_ref * 0.999)  # force a cross back through the arm ref
    bars = [*bars[:105], (adverse, adverse * 1.001, adverse * 0.999, adverse, 1000.0)]
    strategy = _candidate(["A/USDT"], min_hold_bars=7)
    _feed_single(strategy, "A/USDT", bars)
    exits = [s for s in strategy.events.items if s.signal_type == "EXIT"]
    assert exits == []
    assert strategy._state["A/USDT"].mode == "LONG"
    assert strategy._state["A/USDT"].bars_held < strategy.min_hold_bars


def test_determinism_two_runs_identical_signals() -> None:
    bars = _f1()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _candidate(["A/USDT"])
        _feed_single(strategy, "A/USDT", bars)
        return [
            (s.symbol, s.signal_type, s.strength, dict(s.metadata or {}))
            for s in strategy.events.items
        ]

    first = _run()
    assert first, "expected at least one signal"
    assert first == _run()


def test_state_roundtrip_lossless_mid_armed_and_mid_position() -> None:
    bars = _f1()
    # Mid-ARMED snapshot (between the shock and the resolution).
    armed = _candidate(["A/USDT"])
    _feed_single(armed, "A/USDT", bars[:103])
    snap_armed = armed.get_state()
    restored = _candidate(["A/USDT"])
    restored.set_state(snap_armed)
    assert restored.get_state() == snap_armed
    assert restored._state["A/USDT"].armed is True

    # Mid-position snapshot (after the resolution LONG).
    live = _candidate(["A/USDT"])
    _feed_single(live, "A/USDT", bars[:106])
    snap_live = live.get_state()
    assert snap_live["symbol_state"]["A/USDT"]["mode"] == "LONG"
    restored_live = _candidate(["A/USDT"])
    restored_live.set_state(snap_live)
    assert restored_live.get_state() == snap_live


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT"]
    strategy = _candidate(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "nope"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 12345}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "highs": ["x", float("nan"), 1.0],
                    "lows": {"bad": "type"},
                    "closes": [float("inf"), 2.0, None],
                    "volumes": "nope",
                    "mode": 999,
                    "entry_price": "abc",
                    "ref_price": "abc",
                    "bars_held": "oops",
                    "cooldown_remaining": -4,
                    "armed": "yes",
                    "arm_close": "abc",
                    "arm_sigma": "abc",
                    "arm_age": -3,
                    "last_time_key": 123,
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
    # Still functional afterward.
    _feed_single(strategy, "A/USDT", _f1())


def test_degenerate_inputs_never_raise() -> None:
    strategy = _candidate(["Z/USDT"])
    strategy.calculate_signals(_market_event("Z/USDT", 0, (100.0, 90.0, 110.0, 0.0, None)))
    strategy.calculate_signals(_market_event("Z/USDT", 1, (-5.0, -5.0, -5.0, -5.0, 1.0)))
    strategy.calculate_signals(
        _market_event("Z/USDT", 2, (float("nan"), float("nan"), float("nan"), float("nan"), 1.0))
    )
    strategy.calculate_signals(
        _market_event("Z/USDT", 3, (float("inf"), float("inf"), float("inf"), float("inf"), 1.0))
    )
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert _entries(strategy) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = SilentVolumeShockResolutionStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "shock_window",
        "v_shock_z",
        "quiet_ret_z",
        "quiet_range_z",
        "resolution_max_bars",
        "resolution_ret_mult",
        "min_hold_bars",
        "max_hold_bars",
        "cooldown_bars",
        "allow_short",
        "min_history_bars",
    ):
        assert required in schema


def test_slice_multi_timeframe_cells_pinned() -> None:
    """4h/1h scale the episodic bar clocks; 1h tightens the v_shock_z trigger."""
    from lumina_quant.strategies.silent_volume_shock_alpha_sleeves import (
        _SILENT_VOLUME_SHOCK_SLICE as sl,
    )

    assert {"1d", "4h", "1h"} <= set(sl)
    base = tuple(cell["variant"] for cell in sl["1d"])
    for tf in ("4h", "1h"):
        assert tuple(cell["variant"] for cell in sl[tf]) == base
    assert sl["4h"][0]["shock_window"] == 540
    assert sl["4h"][0]["min_hold_bars"] == 42
    assert sl["1h"][0]["shock_window"] == 2160
    assert sl["1h"][0]["min_hold_bars"] == 168
    # 4h keeps the native trigger; 1h is stricter (cost-safety at 24x arming freq).
    assert sl["1d"][0]["v_shock_z"] == 2.0
    assert sl["4h"][0]["v_shock_z"] == 2.0
    assert sl["1h"][0]["v_shock_z"] == 2.5
    assert sl["1h"][1]["v_shock_z"] == 3.0


# --------------------------------------------------------------------------- #
# vol-target horizon fix (Class-B throttle): regression.
#
# ``target_vol`` DEFAULTS to 0.0 here, so the throttle is off by default and this
# fix leaves default sizing byte-identical.  When ENABLED it must annualize the
# per-bar realized vol by ``sqrt(bars_per_year)`` (cadence from the median bar
# spacing) before comparing it to ``target_vol``.
# --------------------------------------------------------------------------- #


def test_vol_target_throttle_annualizes_realized_vol_on_resolution() -> None:
    bars = _f1()
    # Enabled throttle: emitted scalar equals the ANNUALIZED-vol target ratio.
    strat = _candidate(["A/USDT"], target_vol=0.05)
    _feed_single(strat, "A/USDT", bars)
    entries = _entries(strat)
    assert [s.signal_type for s in entries] == ["LONG"]
    meta = entries[0].metadata or {}
    realized_vol = meta["realized_vol"]
    scalar = meta["inverse_vol_scalar"]
    assert realized_vol is not None and realized_vol > 0.0
    bpy = _bars_per_year_from_spacing(list(strat._recent_times))
    assert abs(bpy - 365.25) < 1e-9  # daily fixture cadence
    expected = min(1.0, 0.05 / (realized_vol * math.sqrt(bpy)))
    assert abs(scalar - expected) < 1e-12  # annualized, not per-bar, comparison
    # Guaranteed engagement: a tiny target de-risks (annualized vol >> target).
    tiny = _candidate(["A/USDT"], target_vol=1e-6)
    _feed_single(tiny, "A/USDT", bars)
    assert (_entries(tiny)[0].metadata or {})["inverse_vol_scalar"] < 1.0
    # Default target_vol=0.0 -> throttle OFF -> scalar exactly 1.0 (byte-identical).
    off = _candidate(["A/USDT"])
    _feed_single(off, "A/USDT", bars)
    assert (_entries(off)[0].metadata or {})["inverse_vol_scalar"] == 1.0
