"""Deterministic build-gate + hygiene tests for the capital-gains-overhang lane.

Direct class import only (no ``@register`` on this lane, so no registry / tier /
candidate-wiring assertions here -- those land with the integration wave).  The
build gate instantiates the REAL incumbents (``CrossSectionalNearHighAnchoring
Strategy`` and ``VwapReversionStrategy``) on the same hand-built synthetic feed
and asserts materially OPPOSITE emitted actions:

- FIXTURE A (vs near-high anchoring): signed divergence in BOTH directions --
  GAINHELD is candidate-LONG / incumbent-SHORT; ATHCHURN is incumbent-LONG while
  the candidate does NOT long it (heavy volume traded AT the high => low
  overhang).
- FIXTURE B (vs VWAP reversion): an above-VWAP dislocation with positive
  overhang -> candidate LONG while the incumbent SHORTs (revert-to-VWAP).

FIXTURE C (the indicator-level non-VWAP-alias proof) lives in
``tests/test_reference_price.py``.  All randomness is drawn from a small seeded
LCG (no ``random`` module) so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.cgo_disposition_alpha_sleeves import (
    CrossSectionalCapitalGainsOverhangStrategy,
    _CGO_DISPOSITION_SLICE,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.vwap_reversion import VwapReversionStrategy
from lumina_quant.tuning import HyperParam

_N = 71


# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _market_event(
    symbol: str, idx: int, close: float, volume: float | None, high: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-01-01T{idx // 60:02d}:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close,
        high=close if high is None else max(high, close),
        low=close,
        close=close,
        volume=volume,
    )


def _feed(
    strategy: Any,
    symbols: list[str],
    series: dict[str, tuple[list[float], list[float], list[float]]],
    n: int,
) -> None:
    for idx in range(n):
        for symbol in symbols:
            prices, volumes, highs = series[symbol]
            strategy.calculate_signals(
                _market_event(symbol, idx, prices[idx], volumes[idx], highs[idx])
            )


def _entries(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() in {"LONG", "SHORT"}]


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


# --------------------------------------------------------------------------- #
# deterministic price/volume path generators
# --------------------------------------------------------------------------- #


def _gainheld() -> tuple[list[float], list[float], list[float]]:
    """Thin early spike-high, heavy accumulation near 100, thin drift up to 130.

    Nearness = 130/200 = 0.65 (low) but the volume-weighted cost basis ~ 100, so
    the overhang is large and positive.
    """
    prices = [100.0]
    volumes = [1.0]
    highs = [200.0]
    gen = _lcg_stream(11)
    for _ in range(60):
        close = 100.0 * (1.0 + (next(gen) - 0.5) * 0.004)
        prices.append(close)
        volumes.append(1000.0)
        highs.append(close)
    for i in range(10):
        close = 100.0 + 30.0 * (i + 1) / 10.0
        prices.append(close)
        volumes.append(1.0)
        highs.append(close)
    return prices, volumes, highs


def _athchurn() -> tuple[list[float], list[float], list[float]]:
    """Climb to 150 where the last 20 bars carry ~95% of window volume at 148-150.

    Nearness = 1.0 (close == rolling max) but the cost basis ~ 149, so the
    overhang is near zero (all holders just bought at the high).
    """
    prices: list[float] = []
    volumes: list[float] = []
    gen = _lcg_stream(21)
    for i in range(51):
        close = 100.0 + 48.0 * (i + 1) / 51.0 * (1.0 + (next(gen) - 0.5) * 0.002)
        prices.append(close)
        volumes.append(5.0)
    for i in range(20):
        prices.append(148.0 + 2.0 * (i + 1) / 20.0)
        volumes.append(1000.0)
    highs = list(prices)  # high == close so nearness == 1.0 at the rolling max
    return prices, volumes, highs


def _filler(seed: int, n: int = _N) -> tuple[list[float], list[float], list[float]]:
    """Flat symmetric-jitter path: nearness < 1 and overhang ~ 0."""
    gen = _lcg_stream(seed)
    prices: list[float] = []
    volumes: list[float] = []
    for _ in range(n):
        close = 100.0 * (1.0 + (next(gen) - 0.5) * 0.01)
        prices.append(close)
        volumes.append(1000.0)
    highs = list(prices)
    return prices, volumes, highs


def _overhang() -> tuple[list[float], list[float], list[float]]:
    """Heavy accumulation at 100, then a thin drift to 103 (~3% above VWAP)."""
    prices = [100.0] * 40 + [103.0] * 8
    volumes = [1000.0] * 40 + [30.0] * 8
    highs = list(prices)
    return prices, volumes, highs


def _cgo(symbols: list[str], **overrides: Any) -> CrossSectionalCapitalGainsOverhangStrategy:
    params: dict[str, Any] = dict(
        window_bars=80,
        skip_recent=1,
        vol_window=8,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=0,
        min_history_bars=30,
        min_symbols=5,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    params.update(overrides)
    return CrossSectionalCapitalGainsOverhangStrategy(_Bars(symbols), _Queue(), **params)


# --------------------------------------------------------------------------- #
# FIXTURE A -- signed divergence vs CrossSectionalNearHighAnchoringStrategy
# --------------------------------------------------------------------------- #


def _fixture_a_universe() -> tuple[
    list[str], dict[str, tuple[list[float], list[float], list[float]]]
]:
    symbols = ["GAINHELD/USDT", "ATHCHURN/USDT", *[f"F{i}/USDT" for i in range(5)]]
    series = {
        "GAINHELD/USDT": _gainheld(),
        "ATHCHURN/USDT": _athchurn(),
    }
    for i in range(5):
        series[f"F{i}/USDT"] = _filler(500 + i)
    for symbol in symbols:
        assert len(series[symbol][0]) == _N
    return symbols, series


def test_fixture_a_gainheld_opposite_signed_vs_near_high() -> None:
    symbols, series = _fixture_a_universe()

    candidate = _cgo(symbols)
    _feed(candidate, symbols, series, _N)
    cand_side = _final_side(candidate.events.items)

    incumbent = CrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols),
        _Queue(),
        high_lookback_bars=500,
        min_history_bars=30,
        vol_window=8,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=0,
        min_symbols=5,
        allow_short=True,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(incumbent, symbols, series, _N)
    incumbent_side = _final_side(incumbent.events.items)

    # Signed divergence direction 1: cost-basis overhang LONGs the name the
    # nearness incumbent SHORTs.
    assert cand_side.get("GAINHELD/USDT") == "LONG", cand_side
    assert incumbent_side.get("GAINHELD/USDT") == "SHORT", incumbent_side

    # Signed divergence direction 2: heavy volume traded AT the high => low
    # overhang, so the candidate does NOT long the name the incumbent longs.
    assert incumbent_side.get("ATHCHURN/USDT") == "LONG", incumbent_side
    assert cand_side.get("ATHCHURN/USDT") != "LONG", cand_side

    # Candidate-ACTS: both book sides are non-empty on the divergence leg.
    entry_sides = {sig.signal_type for sig in _entries(candidate.events.items)}
    assert "LONG" in entry_sides and "SHORT" in entry_sides


# --------------------------------------------------------------------------- #
# FIXTURE B -- signed divergence vs VwapReversionStrategy
# --------------------------------------------------------------------------- #


def test_fixture_b_overhang_long_vs_vwap_short() -> None:
    op, ov, oh = _overhang()
    length = len(op)

    # The per-symbol VWAP-reversion incumbent fades the above-VWAP dislocation.
    vwap = VwapReversionStrategy(
        _Bars(["OVERHANG/USDT"]),
        _Queue(),
        window=24,
        entry_dev=0.02,
        exit_dev=0.005,
        allow_short=True,
    )
    for idx in range(length):
        vwap.calculate_signals(_market_event("OVERHANG/USDT", idx, op[idx], ov[idx], oh[idx]))
    vwap_entries = [str(sig.signal_type).upper() for sig in _entries(vwap.events.items)]
    assert "SHORT" in vwap_entries, vwap_entries  # incumbent-LIVE: it fades SHORT

    # The candidate, on the identical bars, sees positive top-quantile overhang
    # and LONGs the same name.
    symbols = ["OVERHANG/USDT", *[f"F{i}/USDT" for i in range(5)]]
    series = {"OVERHANG/USDT": (op, ov, oh)}
    for i in range(5):
        series[f"F{i}/USDT"] = _filler(700 + i, n=length)
    candidate = _cgo(symbols, window_bars=40, min_history_bars=30)
    _feed(candidate, symbols, series, length)
    cand_side = _final_side(candidate.events.items)
    assert cand_side.get("OVERHANG/USDT") == "LONG", cand_side


# --------------------------------------------------------------------------- #
# min-hold suppression (the proven turnover rescue, encoded as a test)
# --------------------------------------------------------------------------- #


def _min_hold_universe() -> tuple[list[str], dict[str, list[float]]]:
    symbols = ["T/USDT", "F0/USDT", "F1/USDT", "F2/USDT", "F3/USDT"]
    prices: dict[str, list[float]] = {s: [] for s in symbols}
    for i in range(45):
        prices["F0/USDT"].append(100.0 * (1.0 + 0.002 * i))  # rising -> top CGO
        prices["F1/USDT"].append(100.0 * (1.0 - 0.002 * i))  # falling -> bottom CGO
        prices["F2/USDT"].append(100.0 + (0.01 if i % 2 else -0.01))
        prices["F3/USDT"].append(100.0 + (0.01 if i % 3 else -0.01))
        prices["T/USDT"].append(100.0 + (0.01 if i % 2 else -0.01))  # flat -> mid, unselected
    prices["T/USDT"][41] = 112.0  # jump up -> T becomes top CGO (enter LONG)
    prices["T/USDT"][42] = 113.0
    prices["T/USDT"][43] = 70.0  # crash -> T becomes bottom CGO (flip attempt)
    prices["T/USDT"][44] = 68.0
    return symbols, prices


def _run_min_hold(min_hold: int) -> str | None:
    symbols, prices = _min_hold_universe()
    strategy = _cgo(
        symbols,
        window_bars=20,
        vol_window=8,
        min_history_bars=15,
        min_hold_bars=min_hold,
    )
    for idx in range(45):
        for symbol in symbols:
            strategy.calculate_signals(_market_event(symbol, idx, prices[symbol][idx], 1000.0))
    return _final_side(strategy.events.items).get("T/USDT")


def test_min_hold_suppresses_side_flip() -> None:
    # Control: without a min-hold the position flips LONG -> SHORT when the rank
    # inverts one bar after entry.
    assert _run_min_hold(0) == "SHORT"
    # With the hard min-hold, the same one-bar rank inversion is suppressed and
    # the LONG is retained (the C1-analog turnover rescue).
    assert _run_min_hold(5) == "LONG"


# --------------------------------------------------------------------------- #
# self-skip below min_symbols
# --------------------------------------------------------------------------- #


def test_self_skip_below_min_symbols() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT"]
    series = {symbol: _filler(900 + idx) for idx, symbol in enumerate(symbols)}
    strategy = _cgo(symbols, min_symbols=5)
    _feed(strategy, symbols, series, _N)
    assert _non_exit(strategy.events.items) == []


# --------------------------------------------------------------------------- #
# never-raise on degenerate input
# --------------------------------------------------------------------------- #


def test_missing_volume_never_raises() -> None:
    symbols = ["RISER/USDT", *[f"F{i}/USDT" for i in range(5)]]
    series: dict[str, tuple[list[float], list[float], list[float]]] = {}
    prices, _volumes, highs = _filler(1)
    series["RISER/USDT"] = (prices, [0.0] * _N, highs)  # no volume anywhere
    for i in range(5):
        series[f"F{i}/USDT"] = _filler(1100 + i)
    strategy = _cgo(symbols)
    # every RISER bar arrives with volume=None -> must not raise
    for idx in range(_N):
        for symbol in symbols:
            prices, volumes, highs = series[symbol]
            vol = None if symbol == "RISER/USDT" else volumes[idx]
            strategy.calculate_signals(_market_event(symbol, idx, prices[idx], vol, highs[idx]))


def test_degenerate_closes_and_empty_window_never_raise() -> None:
    strategy = _cgo(["Z/USDT"], min_symbols=5)
    strategy.calculate_signals(_market_event("Z/USDT", 0, 0.0, 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", 1, -5.0, 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", 2, float("nan"), 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", 3, float("inf"), 1000.0))
    empty = SimpleNamespace(type="MARKET_WINDOW", symbol="Z/USDT", bars_1s={}, time="t0")
    strategy.calculate_signals(empty)
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert _non_exit(strategy.events.items) == []


# --------------------------------------------------------------------------- #
# state roundtrip / adversarial set_state / determinism / schema
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    symbols, series = _fixture_a_universe()
    strategy = _cgo(symbols)
    _feed(strategy, symbols, series, _N)
    snapshot = strategy.get_state()

    restored = _cgo(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    for symbol in symbols:
        assert list(restored._state[symbol].closes) == list(strategy._state[symbol].closes)
        assert list(restored._state[symbol].volumes) == list(strategy._state[symbol].volumes)
        assert restored._state[symbol].mode == strategy._state[symbol].mode


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strategy = _cgo(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("nope")  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 123}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), 12.5, None],
                    "volumes": {"unexpected": "type"},
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 123,
                    "score": [1, 2, 3],
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_determinism_two_runs_identical_signals() -> None:
    symbols, series = _fixture_a_universe()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _cgo(symbols)
        _feed(strategy, symbols, series, _N)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalCapitalGainsOverhangStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in ("window_bars", "skip_recent", "v_clip_max", "quantile_pct", "min_hold_bars"):
        assert required in schema


def test_slice_multi_timeframe_scaling() -> None:
    """1h/4h cells mirror the 1d variants with x24/x6-scaled bar-denominated params."""
    slice_map = _CGO_DISPOSITION_SLICE
    assert set(slice_map.keys()) == {"1h", "4h", "1d"}
    base = slice_map["1d"]
    names = tuple(spec["variant"] for spec in base)
    assert names == ("wk8", "wk13")
    for timeframe, variants in slice_map.items():
        assert len(variants) == len(base), timeframe
        assert tuple(spec["variant"] for spec in variants) == names, timeframe

    # Every bar-denominated window (reference/skip/vol/history) and the weekly
    # decision clocks scale x6 (4h) / x24 (1h); ratios, counts and price/vol
    # thresholds are timeframe-agnostic and stay identical.
    scaled_keys = (
        "window_bars",
        "skip_recent",
        "vol_window",
        "rebalance_bars",
        "min_hold_bars",
        "min_history_bars",
    )
    unchanged_keys = (
        "quantile_pct",
        "min_symbols",
        "allow_short",
        "target_gross_exposure",
        "target_vol",
        "stop_loss_pct",
    )
    for factor, timeframe in ((6, "4h"), (24, "1h")):
        for base_spec, tf_spec in zip(base, slice_map[timeframe], strict=True):
            for key in scaled_keys:
                assert tf_spec[key] == base_spec[key] * factor, (timeframe, key)
            for key in unchanged_keys:
                assert tf_spec[key] == base_spec[key], (timeframe, key)
    for variants in slice_map.values():
        for spec in variants:
            # Per-symbol history floor still exceeds the reference window, and no
            # window reaches the ~9000-bar cap.
            assert spec["min_history_bars"] > spec["window_bars"], spec["variant"]
            assert spec["window_bars"] <= 9000, spec["variant"]
