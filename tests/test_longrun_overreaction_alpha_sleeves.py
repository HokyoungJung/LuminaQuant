"""Deterministic build-gate + hygiene tests for the long-run overreaction lane.

Direct class import only (no ``@register`` on this lane).  The build gates drive
the REAL cross-sectional incumbents through ``calculate_signals_window`` with
``rebalance_bars=1`` (every tick evaluates) over 6 symbols and >=280 daily bars,
asserting materially DIVERGENT emitted actions:

- (A) HORIZON divergence vs ``CrossSectionalEquityMomentumStrategy``: a symbol
      that is mid-rank on 12-1 momentum (excluded by the momentum quintile) but
      the most extreme multi-month LOSER on the skip-adjusted 6-1 formation
      return (LONG here).  The incumbent emits a non-empty long/short pair
      (LIVE control) that excludes it.
- (B) EXTREMENESS ABSTENTION / anti-sign-mirror: a dislocation-free
      cross-section (balanced 3-3 formation split, |z| < z_min for every name)
      where 12-1 momenta still differ -> the momentum incumbent trades both
      sides while this sleeve emits ZERO new targets.
- (C) SKIP-MONTH divergence vs ``DispersionConditionedReversionStrategy``: a
      symbol that is up +80% over the formation window then crashes -15% in the
      last 5 bars -> the 5-bar dispersion-reversion incumbent LONGs it while the
      skip-month excludes the crash and this sleeve SHORTs the extreme
      formation WINNER (opposite side, same symbol, same bar).

All randomness is a small seeded LCG (no ``random`` module), so every run is
bit-for-bit reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    DispersionConditionedReversionStrategy,
)
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    CrossSectionalEquityMomentumStrategy,
)
from lumina_quant.strategies.longrun_overreaction_alpha_sleeves import (
    LongRunOverreactionReversalStrategy,
    _LONGRUN_OVERREACTION_SLICE,
)
from lumina_quant.tuning import HyperParam

_START = datetime(2026, 1, 1, tzinfo=UTC)
_N = 285
# Index anchors: recent = closes[-22] (idx 263); 6-1 formation base = closes[-148]
# (idx 137); 12-1 momentum base = closes[-274] (idx 11).
_IDX_RECENT, _IDX_FORM = 263, 137


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


def _interp(i: int, anchors: list[tuple[int, float]]) -> float:
    if i <= anchors[0][0]:
        return anchors[0][1]
    if i >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, p0), (x1, p1) in pairwise(anchors):
        if x0 <= i <= x1:
            return p0 + (p1 - p0) * (i - x0) / (x1 - x0)
    return anchors[-1][1]


def _path(
    anchors: list[tuple[int, float]], seed: int, *, n: int = _N, noise: float = 0.0006
) -> list[float]:
    gen = _lcg_stream(seed)
    ordered = sorted(anchors)
    return [_interp(i, ordered) * (1.0 + (next(gen) - 0.5) * noise) for i in range(n)]


def _window_event(t: datetime, closes: dict[str, float]) -> SimpleNamespace:
    bars_1s = {
        sym: [{"time": t, "open": c, "high": c, "low": c, "close": c, "volume": 1000.0}]
        for sym, c in closes.items()
    }
    return SimpleNamespace(type="MARKET_WINDOW", time=t, bars_1s=bars_1s)


def _feed(
    strategy: Any, series: dict[str, list[float]], symbols: list[str], *, n: int = _N
) -> None:
    for i in range(n):
        rows = {sym: series[sym][i] for sym in symbols}
        strategy.calculate_signals_window(_window_event(_START + timedelta(days=i), rows), None)


def _entries(strategy: Any) -> list[Any]:
    return [s for s in strategy.events.items if s.signal_type in {"LONG", "SHORT"}]


def _running_targets(strategy: Any) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in strategy.events.items:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


_CAND_KW: dict[str, Any] = dict(
    formation_bars=126,
    skip_bars=21,
    z_min=1.0,
    max_universe=12,
    rebalance_bars=1,
    min_hold_bars=0,
    quantile_pct=0.25,
    min_symbols=5,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)
_MOM_KW: dict[str, Any] = dict(
    lookback_bars=252,
    skip_bars=21,
    vol_window=63,
    regime_sma_bars=200,
    rebalance_bars=1,
    signal_threshold=0.0,
    quintile_pct=0.20,
    min_symbols=5,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)
_DISP_KW: dict[str, Any] = dict(
    reversion_lookback_bars=5,
    dispersion_threshold=0.02,
    rebalance_bars=1,
    quantile_pct=0.25,
    min_symbols=5,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)


def _candidate(symbols: list[str], **overrides: Any) -> LongRunOverreactionReversalStrategy:
    return LongRunOverreactionReversalStrategy(
        _Bars(symbols), _Queue(), **dict(_CAND_KW, **overrides)
    )


# --------------------------------------------------------------------------- #
# (A) HORIZON divergence vs cross-sectional momentum
# --------------------------------------------------------------------------- #


def _gate_a_series() -> tuple[list[str], dict[str, list[float]]]:
    symbols = ["MIDMOM", "WINNER", "LOSER", "F1", "F2", "F3"]
    series = {
        # up to 150 by the formation base then down to 95 by t-skip: mid 12-1
        # momentum but the most-negative 6-1 formation return.
        "MIDMOM": _path([(0, 100), (_IDX_FORM, 150), (_IDX_RECENT, 95), (_N - 1, 95)], 1),
        "WINNER": _path([(0, 100), (_IDX_RECENT, 140), (_N - 1, 140)], 2),
        "LOSER": _path([(0, 100), (_IDX_RECENT, 60), (_N - 1, 60)], 3),
        "F1": _path([(0, 100), (_N - 1, 98)], 4),
        "F2": _path([(0, 100), (_N - 1, 97)], 5),
        "F3": _path([(0, 100), (_N - 1, 99)], 6),
    }
    return symbols, series


def test_gate_a_horizon_divergence_vs_cross_sectional_momentum() -> None:
    symbols, series = _gate_a_series()

    candidate = _candidate(symbols)
    _feed(candidate, series, symbols)
    cand_targets = _running_targets(candidate)

    incumbent = CrossSectionalEquityMomentumStrategy(_Bars(symbols), _Queue(), **_MOM_KW)
    _feed(incumbent, series, symbols)
    mom_targets = _running_targets(incumbent)

    # Candidate LONGs the extreme multi-month loser MIDMOM.
    assert cand_targets.get("MIDMOM") == "LONG"
    # Incumbent-LIVE control: a non-empty long/short pair that EXCLUDES MIDMOM.
    assert "LONG" in mom_targets.values() and "SHORT" in mom_targets.values()
    assert "MIDMOM" not in mom_targets
    # Divergence by HORIZON, not sign: same bar, MIDMOM traded here / excluded there.
    assert candidate is not incumbent


# --------------------------------------------------------------------------- #
# (B) EXTREMENESS ABSTENTION -- not a negated-momentum sleeve
# --------------------------------------------------------------------------- #


def _gate_b_series() -> tuple[list[str], dict[str, list[float]]]:
    # Two distinct paths, three copies each -> the cross-section is ALWAYS a
    # balanced 3-3 split (|z| = 0.913 < z_min), and a shared recent tail makes
    # the final-state dispersion 0.  The paths differ only in the far-early
    # region, so 12-1 momenta differ; A rises from a low start (positive 12-1
    # momentum) and B falls from a high start, both ending below the 200-bar SMA.
    symbols = ["A1", "A2", "A3", "B1", "B2", "B3"]
    tail = _path([(_IDX_FORM, 92.0), (_N - 1, 85.0)], 999, noise=0.0008)
    early_a = _path([(0, 80.0), (60, 88.0), (_IDX_FORM, 92.0)], 101)
    early_b = _path([(0, 100.0), (60, 110.0), (_IDX_FORM, 92.0)], 202)
    path_a = early_a[:_IDX_FORM] + tail[_IDX_FORM:]
    path_b = early_b[:_IDX_FORM] + tail[_IDX_FORM:]
    series = {"A1": path_a, "A2": path_a, "A3": path_a, "B1": path_b, "B2": path_b, "B3": path_b}
    return symbols, series


def test_gate_b_extremeness_gate_abstains_where_momentum_trades() -> None:
    symbols, series = _gate_b_series()

    candidate = _candidate(symbols)
    _feed(candidate, series, symbols)
    # No genuine multi-month dislocation -> the sleeve emits ZERO new targets.
    assert _entries(candidate) == []

    incumbent = CrossSectionalEquityMomentumStrategy(_Bars(symbols), _Queue(), **_MOM_KW)
    _feed(incumbent, series, symbols)
    mom_targets = _running_targets(incumbent)
    # Incumbent-LIVE control: threshold-0 momentum still trades BOTH sides.
    assert "LONG" in mom_targets.values() and "SHORT" in mom_targets.values()


# --------------------------------------------------------------------------- #
# (C) SKIP-MONTH divergence vs dispersion-conditioned reversion
# --------------------------------------------------------------------------- #


def _gate_c_series() -> tuple[list[str], dict[str, list[float]]]:
    symbols = ["RECENTCRASH", "LOSER", "F1", "F2", "F3", "F4"]
    peak = 180.0
    crash = peak * 0.85  # -15% over the last 5 bars (inside the skip window)
    series = {
        "RECENTCRASH": _path(
            [(0, 100), (_IDX_FORM, 100), (_IDX_RECENT, peak), (279, peak), (_N - 1, crash)], 1
        ),
        "LOSER": _path([(0, 100), (_IDX_FORM, 100), (_IDX_RECENT, 60), (_N - 1, 60)], 2),
        "F1": _path([(0, 100), (_N - 1, 100.5)], 3),
        "F2": _path([(0, 100), (_N - 1, 99.5)], 4),
        "F3": _path([(0, 100), (_N - 1, 100.3)], 5),
        "F4": _path([(0, 100), (_N - 1, 99.7)], 6),
    }
    return symbols, series


def test_gate_c_skip_month_opposes_dispersion_reversion() -> None:
    symbols, series = _gate_c_series()

    candidate = _candidate(symbols)
    _feed(candidate, series, symbols)
    cand_targets = _running_targets(candidate)

    incumbent = DispersionConditionedReversionStrategy(_Bars(symbols), _Queue(), **_DISP_KW)
    _feed(incumbent, series, symbols)
    disp_targets = _running_targets(incumbent)

    # Skip-month excludes the crash -> RECENTCRASH is the extreme formation
    # WINNER -> candidate SHORTs it, while the 5-bar reversion incumbent (LIVE)
    # LONGs it (fades the crash): opposite sides, same symbol.
    assert cand_targets.get("RECENTCRASH") == "SHORT"
    assert disp_targets.get("RECENTCRASH") == "LONG"


# --------------------------------------------------------------------------- #
# hygiene: determinism / state roundtrip / adversarial / degenerate / schema
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    symbols, series = _gate_a_series()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _candidate(symbols)
        _feed(strategy, series, symbols)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    assert first, "expected at least one signal in gate-A scenario"
    assert first == _run()


def test_state_roundtrip_lossless() -> None:
    symbols, series = _gate_a_series()
    strategy = _candidate(symbols)
    _feed(strategy, series, symbols)
    snapshot = strategy.get_state()

    restored = _candidate(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    assert restored._tick == strategy._tick
    for symbol in symbols:
        assert restored._state[symbol].mode == strategy._state[symbol].mode
        assert list(restored._state[symbol].closes) == list(strategy._state[symbol].closes)


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT"]
    strategy = _candidate(symbols, min_symbols=2)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("nope")  # type: ignore[arg-type]
    strategy.set_state(42)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "nope"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 123}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), float("inf"), 1.0, None],
                    "volumes": {"bad": "type"},
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 123,
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_degenerate_inputs_never_raise() -> None:
    strategy = _candidate(["A/USDT", "B/USDT"], min_symbols=2)
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="A/USDT", close=None))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="A/USDT", close=float("nan")))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="A/USDT", close=float("inf")))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=5.0))
    assert _entries(strategy) == []


def test_self_skip_below_min_symbols_and_min_history() -> None:
    # Too few symbols in the universe -> no cross-sectional book.
    symbols = ["A/USDT", "B/USDT", "C/USDT"]
    gen = _lcg_stream(7)
    short_series = {
        s: [100.0 * (1.0 + (next(gen) - 0.5) * 0.01) for _ in range(_N)] for s in symbols
    }
    strategy = _candidate(symbols, min_symbols=5)
    _feed(strategy, short_series, symbols)
    assert _entries(strategy) == []

    # Enough symbols but history below the formation+skip floor -> never-raise skip.
    symbols6 = ["S1", "S2", "S3", "S4", "S5", "S6"]
    gen2 = _lcg_stream(8)
    n_short = 60  # < formation_bars + skip_bars + 1 = 148
    tiny = {
        s: [100.0 * (1.0 + (next(gen2) - 0.5) * 0.01) for _ in range(n_short)] for s in symbols6
    }
    strategy2 = _candidate(symbols6)
    _feed(strategy2, tiny, symbols6, n=n_short)
    assert _entries(strategy2) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = LongRunOverreactionReversalStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "formation_bars",
        "skip_bars",
        "z_min",
        "max_universe",
        "rebalance_bars",
        "min_hold_bars",
        "quantile_pct",
        "min_symbols",
        "allow_short",
    ):
        assert required in schema


def test_slice_multi_timeframe_scaling() -> None:
    """1h/4h cells mirror the 1d variants with x24/x6-scaled bar-denominated params."""
    slice_map = _LONGRUN_OVERREACTION_SLICE
    assert set(slice_map.keys()) == {"1h", "4h", "1d"}
    base = slice_map["1d"]
    names = tuple(spec["variant"] for spec in base)
    assert names == ("formation_126", "formation_91")
    for timeframe, variants in slice_map.items():
        assert len(variants) == len(base), timeframe
        assert tuple(spec["variant"] for spec in variants) == names, timeframe

    # The multi-month formation window, skip-month excision, and monthly clocks
    # are wall-clock horizons that scale x6 (4h) / x24 (1h); the extremeness gate,
    # universe/symbol counts, and quantile ratio stay identical.
    scaled_keys = ("formation_bars", "skip_bars", "rebalance_bars", "min_hold_bars")
    unchanged_keys = ("z_min", "max_universe", "quantile_pct", "min_symbols", "allow_short")
    for factor, timeframe in ((6, "4h"), (24, "1h")):
        for base_spec, tf_spec in zip(base, slice_map[timeframe], strict=True):
            for key in scaled_keys:
                assert tf_spec[key] == base_spec[key] * factor, (timeframe, key)
            for key in unchanged_keys:
                assert tf_spec[key] == base_spec[key], (timeframe, key)
    for variants in slice_map.values():
        for spec in variants:
            assert spec["skip_bars"] < spec["formation_bars"], spec["variant"]
            assert spec["formation_bars"] <= 9000, spec["variant"]
