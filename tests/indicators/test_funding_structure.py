"""Deterministic tests for the funding-structure indicator primitives.

Covers the PARITY LOCK of :func:`funding_momentum` against a verbatim inline
copy of the pre-extraction strategy-private ``_funding_momentum`` (exact float
equality on seeded LCG funding paths); the import alias in the consuming
carry sleeve; direction/None guards for all four primitives; and the
``require_term_structure_agreement`` consumer wiring on
``CrossSectionalFundingMomentumCarryStrategy`` (flag OFF byte-identical
default, flag ON skips ONLY disagreeing NEW entries and never adds trades).
No backtest is run; no data files are read.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.indicators.funding_structure import (
    basis_funding_gap_bps,
    funding_implied_basis_bps,
    funding_momentum,
    funding_term_structure_spread,
)
from lumina_quant.strategies.cross_sectional_funding_momentum_carry import (
    CrossSectionalFundingMomentumCarryStrategy,
    _funding_momentum,
)

_EPS = 1e-12


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield state / 2**31


# --------------------------------------------------------------------------- #
# Verbatim PRE-EXTRACTION reference (copied from the previously private
# ``_funding_momentum`` / ``_ols_slope`` in
# strategies/cross_sectional_funding_momentum_carry.py) -- the parity oracle.
# --------------------------------------------------------------------------- #
def _reference_ols_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / float(n)
    num = 0.0
    den = 0.0
    for i, value in enumerate(values):
        dx = i - mean_x
        num += dx * (value - mean_y)
        den += dx * dx
    if den <= _EPS:
        return 0.0
    return num / den


def _reference_funding_momentum(
    funding: list[float], *, diff_window: int, slope_window: int, ewma_span: int
) -> float | None:
    if len(funding) < 2:
        return None
    diffs = [funding[i] - funding[i - 1] for i in range(1, len(funding))]
    diff_tail = diffs[-max(1, diff_window) :]
    span = max(1, int(ewma_span))
    alpha = 2.0 / (span + 1.0)
    ewma = diff_tail[0]
    for value in diff_tail[1:]:
        ewma = alpha * value + (1.0 - alpha) * ewma
    level_tail = funding[-max(2, slope_window) :]
    slope = _reference_ols_slope(level_tail)
    return float(ewma) + float(slope)


# --------------------------------------------------------------------------- #
# funding_momentum: parity lock + direction
# --------------------------------------------------------------------------- #
def test_funding_momentum_parity_with_pre_extraction_reference() -> None:
    gen = _lcg(20260820)
    for length in (2, 3, 8, 21, 64):
        path = [(next(gen) - 0.5) * 0.002 for _ in range(length)]
        for kw in (
            {"diff_window": 6, "slope_window": 8, "ewma_span": 4},
            {"diff_window": 1, "slope_window": 2, "ewma_span": 1},
            {"diff_window": 12, "slope_window": 21, "ewma_span": 8},
        ):
            got = funding_momentum(path, **kw)
            want = _reference_funding_momentum(path, **kw)
            assert got == want, (length, kw)


def test_sleeve_import_alias_is_the_indicator_function() -> None:
    # The carry sleeve's ``_funding_momentum`` IS the extracted indicator: the
    # refactor moved the function, it did not fork it.
    assert _funding_momentum is funding_momentum


def test_funding_momentum_direction_and_short_history() -> None:
    kw = {"diff_window": 6, "slope_window": 8, "ewma_span": 4}
    rising = [0.0001 * i for i in range(12)]
    falling = [-0.0001 * i for i in range(12)]
    flat = [0.0005 for _ in range(12)]
    assert funding_momentum(rising, **kw) > 0.0
    assert funding_momentum(falling, **kw) < 0.0
    assert funding_momentum(flat, **kw) == pytest.approx(0.0, abs=1e-12)
    assert funding_momentum([0.001], **kw) is None
    assert funding_momentum([], **kw) is None
    assert funding_momentum(None, **kw) is None


def test_funding_momentum_never_raises_on_garbage() -> None:
    kw = {"diff_window": 6, "slope_window": 8, "ewma_span": 4}
    assert funding_momentum(["x", None, float("nan")], **kw) is None
    # Garbage entries are dropped, finite ones survive.
    got = funding_momentum(["x", 0.001, None, 0.002, float("inf"), 0.003], **kw)
    want = _reference_funding_momentum([0.001, 0.002, 0.003], **kw)
    assert got == want


# --------------------------------------------------------------------------- #
# funding_term_structure_spread
# --------------------------------------------------------------------------- #
def test_term_structure_spread_sign_and_guards() -> None:
    rising = [0.0001 * i for i in range(25)]
    falling = [-0.0001 * i for i in range(25)]
    flat = [0.0004 for _ in range(25)]
    assert funding_term_structure_spread(rising) > 0.0
    assert funding_term_structure_spread(falling) < 0.0
    assert funding_term_structure_spread(flat) == pytest.approx(0.0, abs=1e-15)
    # Fewer than long_window prints => None (no partial-window guessing).
    assert funding_term_structure_spread(rising[:20]) is None
    assert funding_term_structure_spread([]) is None
    assert funding_term_structure_spread(None) is None
    # Degenerate window configs => None, never raise.
    assert funding_term_structure_spread(rising, short_window=21, long_window=3) is None
    assert funding_term_structure_spread(rising, short_window=0, long_window=21) is None
    assert funding_term_structure_spread(rising, short_window="x", long_window=21) is None  # type: ignore[arg-type]


def test_term_structure_spread_hand_computed() -> None:
    # 13 prints at 0.01 then 12 prints rising -0.016 .. -0.005: the recent mean
    # sits far below the trailing 21-print mean => negative spread, while the
    # short-window momentum of the same path is positive (the disagreement
    # fixture reused by the consumer-flag tests below).
    path = [0.01] * 13 + [-0.016 + 0.001 * i for i in range(12)]
    spread = funding_term_structure_spread(path, short_window=3, long_window=21)
    short_mean = sum(path[-3:]) / 3.0
    long_mean = sum(path[-21:]) / 21.0
    assert spread == pytest.approx(short_mean - long_mean, abs=1e-15)
    assert spread < 0.0
    assert funding_momentum(path, diff_window=6, slope_window=8, ewma_span=4) > 0.0


# --------------------------------------------------------------------------- #
# funding_implied_basis_bps / basis_funding_gap_bps
# --------------------------------------------------------------------------- #
def test_funding_implied_basis_bps() -> None:
    assert funding_implied_basis_bps(0.0001) == pytest.approx(1.0)
    assert funding_implied_basis_bps(-0.0002) == pytest.approx(-2.0)
    assert funding_implied_basis_bps(0.0) == pytest.approx(0.0)
    assert funding_implied_basis_bps(None) is None
    assert funding_implied_basis_bps(float("nan")) is None
    assert funding_implied_basis_bps("garbage") is None


def test_basis_funding_gap_bps_hand_computed() -> None:
    # basis = (101/100 - 1) * 1e4 = 100bps; implied = 0.0001 * 1e4 = 1bp.
    gap = basis_funding_gap_bps(101.0, 100.0, 0.0001)
    assert gap == pytest.approx(99.0)
    # Discount: mark below index with positive funding => negative gap.
    assert basis_funding_gap_bps(99.0, 100.0, 0.0001) == pytest.approx(-101.0)


def test_basis_funding_gap_bps_guards_never_raise() -> None:
    assert basis_funding_gap_bps(None, 100.0, 0.0001) is None
    assert basis_funding_gap_bps(101.0, None, 0.0001) is None
    assert basis_funding_gap_bps(101.0, 100.0, None) is None
    assert basis_funding_gap_bps(101.0, 0.0, 0.0001) is None  # index <= 0
    assert basis_funding_gap_bps(float("nan"), 100.0, 0.0001) is None
    assert basis_funding_gap_bps("x", "y", "z") is None


def test_indicators_deterministic_two_calls() -> None:
    gen = _lcg(7)
    path = [(next(gen) - 0.5) * 0.001 for _ in range(30)]
    kw = {"diff_window": 6, "slope_window": 8, "ewma_span": 4}
    assert funding_momentum(path, **kw) == funding_momentum(list(path), **kw)
    assert funding_term_structure_spread(path) == funding_term_structure_spread(list(path))
    assert basis_funding_gap_bps(101.0, 100.0, 0.0001) == basis_funding_gap_bps(
        101.0, 100.0, 0.0001
    )


# --------------------------------------------------------------------------- #
# Consumer wiring: require_term_structure_agreement on the carry sleeve
# (entry-SKIP only, default OFF byte-identical).
# --------------------------------------------------------------------------- #
_SYMBOLS = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "XXX/USDT"]
_N = 25


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    def __init__(self, symbols: list[str]):
        self.symbol_list = list(symbols)

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        return None

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        return None


def _series(n: int, *, start: float, drift: float, noise: float) -> list[float]:
    out: list[float] = []
    for i in range(n):
        zig = 1.0 if i % 2 == 0 else -1.0
        out.append(start * (1.0 + drift) ** i * (1.0 + noise * zig))
    return out


def _market_event(symbol: str, idx: int, close: float, funding: float) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        funding_rate=funding,
    )


def _panel() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """4 agree-symbols (rising funding: momentum>0, spread>0) + 1 disagree.

    XXX/USDT: 13 prints at 0.01 then 12 rising from -0.016 -- once 21 prints
    exist its short-window momentum is strongly POSITIVE (largest in the
    cross-section => LONG target) while its 3-vs-21 spread is NEGATIVE
    (disagreement => the flag must skip its NEW entry).
    """
    series = {
        s: _series(_N, start=100.0 + i, drift=0.003, noise=0.01) for i, s in enumerate(_SYMBOLS)
    }
    funding: dict[str, list[float]] = {
        s: [0.0001 * (i + 1) * j for j in range(_N)] for i, s in enumerate(_SYMBOLS[:-1])
    }
    funding["XXX/USDT"] = [0.01] * 13 + [-0.016 + 0.001 * i for i in range(12)]
    return series, funding


def _make(queue: _Queue, **extra: Any) -> CrossSectionalFundingMomentumCarryStrategy:
    return CrossSectionalFundingMomentumCarryStrategy(
        _Bars(_SYMBOLS),
        queue,
        min_symbols=4,
        threshold=0.0,
        max_longs=5,
        max_shorts=0,
        allow_short=False,
        rebalance_band=10.0,
        target_vol=0.0,
        vol_window=10,
        **extra,
    )


def _feed(strategy: CrossSectionalFundingMomentumCarryStrategy) -> None:
    series, funding = _panel()
    for idx in range(_N):
        for symbol in _SYMBOLS:
            strategy.calculate_signals(
                _market_event(symbol, idx, series[symbol][idx], funding[symbol][idx])
            )


def _stream(queue: _Queue) -> list[tuple[str, str, Any, tuple[tuple[str, Any], ...]]]:
    return [
        (s.symbol, s.signal_type, s.datetime, tuple(sorted((s.metadata or {}).items())))
        for s in queue.items
    ]


def test_flag_defaults_off_and_not_tunable() -> None:
    schema = CrossSectionalFundingMomentumCarryStrategy.get_param_schema()
    assert "require_term_structure_agreement" in schema
    assert schema["require_term_structure_agreement"].tunable is False
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_SYMBOLS), _Queue())
    assert strat.require_term_structure_agreement is False


def test_flag_off_is_byte_identical_to_default() -> None:
    q_default, q_off = _Queue(), _Queue()
    _feed(_make(q_default))
    _feed(_make(q_off, require_term_structure_agreement=False))
    assert _stream(q_default) == _stream(q_off)
    assert _stream(q_default), "fixture must actually emit signals"


def test_flag_on_skips_only_disagreeing_new_entries_and_adds_nothing() -> None:
    q_off, q_on = _Queue(), _Queue()
    _feed(_make(q_off))
    _feed(_make(q_on, require_term_structure_agreement=True))

    entries_off = {s.symbol for s in q_off.items if s.signal_type == "LONG"}
    entries_on = {s.symbol for s in q_on.items if s.signal_type == "LONG"}
    # The disagreeing symbol entered with the flag OFF and is skipped ON.
    assert "XXX/USDT" in entries_off
    assert "XXX/USDT" not in entries_on
    # Entry-SKIP only: no entry is ever ADDED by the flag, agreeing entries stay.
    assert entries_on == entries_off - {"XXX/USDT"}
    assert entries_on, "agreeing symbols must still enter"


def test_flag_on_never_raises_with_short_funding_history() -> None:
    # Fewer than 21 prints => spread None => the filter never skips and never
    # raises (entries proceed exactly as with the flag OFF).
    q_on, q_off = _Queue(), _Queue()
    on = _make(q_on, require_term_structure_agreement=True)
    off = _make(q_off)
    series, funding = _panel()
    for idx in range(15):  # < 21 prints everywhere
        for symbol in _SYMBOLS:
            event = _market_event(symbol, idx, series[symbol][idx], funding[symbol][idx])
            on.calculate_signals(event)
            off.calculate_signals(event)
    assert _stream(q_on) == _stream(q_off)
