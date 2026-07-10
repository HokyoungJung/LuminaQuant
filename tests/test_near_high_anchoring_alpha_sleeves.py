"""Deterministic tests for CrossSectionalNearHighAnchoringStrategy.

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

The load-bearing test is the B2 HARD BUILD GATE: on a single synthetic
cross-section where nearness and momentum DISAGREE, the anchoring sleeve must
diverge from the incumbent ``NearHighMomentumStrategy`` on BOTH axes it was
built to separate -- (i) it LONGS a name pinned to its trailing high that has
weak/negative recent momentum (which the incumbent EXCLUDES on its
``min_momentum`` gate), and (ii) it SHORTS the low-nearness tail (which the
long-only incumbent never touches).  If that divergence cannot be produced with
an honest implementation the leaf is redundant and must be dropped -- the test
is not weakened to make it pass.

All pseudo-randomness (only ever a tiny jitter that makes realised volatility
non-degenerate) is drawn from a small seeded linear-congruential generator (no
``random`` module), so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import NearHighMomentumStrategy
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _jitter(prices: list[float], seed: int, amp: float = 0.0015) -> list[float]:
    """Apply a tiny multiplicative LCG jitter so realised vol is non-degenerate."""
    gen = _lcg_stream(seed)
    return [price * (1.0 + (next(gen) - 0.5) * 2.0 * amp) for price in prices]


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(stop)]
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


# --------------------------------------------------------------------------- #
# harness (MARKET_WINDOW feed keeps the cross-section coherent per bar)
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _ts(idx: int) -> str:
    return f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z"


# series[symbol] -> (closes, highs, volumes); ``first_bar`` shifts a symbol's
# first appearance later (to model a young / just-listed alt).
Series = dict[str, tuple[list[float], list[float], list[float]]]


def _feed_window(
    strategy: Any,
    symbols: list[str],
    series: Series,
    n: int,
    first_bar: dict[str, int] | None = None,
) -> None:
    first_bar = first_bar or {}
    for idx in range(n):
        bars_1s: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            start = first_bar.get(symbol, 0)
            if idx < start:
                continue
            local = idx - start
            closes, highs, volumes = series[symbol]
            if local >= len(closes):
                continue
            bars_1s[symbol] = [
                {
                    "time": _ts(idx),
                    "open": closes[local],
                    "high": highs[local],
                    "low": closes[local],
                    "close": closes[local],
                    "volume": volumes[local],
                }
            ]
        event = SimpleNamespace(type="MARKET_WINDOW", time=_ts(idx), bars_1s=bars_1s)
        strategy.calculate_signals(event)


def _with_high(closes: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Package a close path as (closes, highs=closes, volumes) for the window feed."""
    return closes, list(closes), [1000.0] * len(closes)


def _entries(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if sig.signal_type in {"LONG", "SHORT"}]


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
# B2 build-gate cross-section: nearness and momentum DISAGREE.
#
#   NEAR  : pinned to its trailing high (nearness ~0.99) but drifting DOWN
#           (negative 5-bar momentum)  -> anchoring LONGS it, incumbent EXCLUDES
#           it (min_momentum gate).
#   MOMUP : below its (older) high but recovering (positive momentum)
#           -> incumbent LONGS it (proves the incumbent pipeline is alive and it
#           is the momentum gate, not a dead engine, that drops NEAR).
#   CTRL* : mid-nearness (~0.90) fillers, selected by neither side.
#   FAR   : stranded far below its trailing high (nearness ~0.62)
#           -> anchoring SHORTS it, the long-only incumbent never shorts.
# --------------------------------------------------------------------------- #

_B2_N = 28


def _near_series() -> tuple[list[float], list[float], list[float]]:
    base = _linspace(100.0, 120.0, 24) + [120.0 * 0.997**k for k in range(1, 5)]
    return _with_high(_jitter(base, seed=101))


def _momup_series() -> tuple[list[float], list[float], list[float]]:
    base = (
        _linspace(100.0, 115.0, 13)
        + _linspace(115.0, 100.0, 11)[1:]
        + _linspace(100.0, 110.0, 6)[1:]
    )
    return _with_high(_jitter(base, seed=202))


def _far_series() -> tuple[list[float], list[float], list[float]]:
    base = _linspace(100.0, 120.0, 13) + [105.0, 90.0, 74.0] + [74.0] * 12
    return _with_high(_jitter(base, seed=303))


def _ctrl_series(
    n: int, peak: float, trough: float, seed: int
) -> tuple[list[float], list[float], list[float]]:
    half = n // 2
    base = _linspace(100.0, peak, half) + _linspace(peak, trough, n - half + 1)[1:]
    return _with_high(_jitter(base, seed=seed))


def _b2_universe() -> tuple[list[str], Series]:
    symbols = ["NEAR/USDT", "MOMUP/USDT", "CTRL1/USDT", "CTRL2/USDT", "FAR/USDT"]
    series: Series = {
        "NEAR/USDT": _near_series(),
        "MOMUP/USDT": _momup_series(),
        "CTRL1/USDT": _ctrl_series(_B2_N, 110.0, 99.0, seed=404),
        "CTRL2/USDT": _ctrl_series(_B2_N, 108.0, 96.0, seed=505),
        "FAR/USDT": _far_series(),
    }
    return symbols, series


_A1_B2_KWARGS: dict[str, Any] = dict(
    high_lookback_bars=20,
    min_history_bars=5,
    vol_window=5,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_bars=0,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)

_INCUMBENT_B2_KWARGS: dict[str, Any] = dict(
    high_lookback_bars=20,
    momentum_lookback_bars=5,
    rebalance_bars=1,
    near_high_pct=0.06,
    min_momentum=0.006,
    max_longs=4,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)


def test_b2_build_gate_diverges_from_incumbent_on_both_axes() -> None:
    """A1 longs a high-nearness/negative-momentum name the incumbent excludes,
    and shorts the low-nearness tail the long-only incumbent never touches."""
    symbols, series = _b2_universe()

    anchoring = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    _feed_window(anchoring, symbols, series, _B2_N)
    a1_side = _final_side(anchoring.events.items)

    incumbent = NearHighMomentumStrategy(_Bars(symbols), _Queue(), **_INCUMBENT_B2_KWARGS)
    _feed_window(incumbent, symbols, series, _B2_N)
    inc_signals = incumbent.events.items
    inc_side = _final_side(inc_signals)

    # --- Axis (i): PURE nearness vs momentum-gated ------------------------- #
    # A1 LONGS the near-high / negative-momentum name ...
    assert a1_side.get("NEAR/USDT") == "LONG", a1_side
    # ... which the incumbent EXCLUDES on its min_momentum gate.
    assert "NEAR/USDT" not in inc_side, inc_side
    # The incumbent is not a dead engine: it DOES admit the near-high name that
    # has positive momentum, so NEAR's exclusion is the momentum gate itself.
    assert inc_side.get("MOMUP/USDT") == "LONG", inc_side

    # --- Axis (ii): long-SHORT vs long-only -------------------------------- #
    # A1 SHORTS the stranded low-nearness tail ...
    assert a1_side.get("FAR/USDT") == "SHORT", a1_side
    # ... while the long-only incumbent takes NO short anywhere on this input.
    assert [sig for sig in inc_signals if str(sig.signal_type).upper() == "SHORT"] == []

    # Sanity: A1's own final book is exactly {NEAR long, FAR short}; the
    # mid-nearness fillers and MOMUP are selected by neither quantile.
    assert a1_side == {"NEAR/USDT": "LONG", "FAR/USDT": "SHORT"}, a1_side

    # The LONG on NEAR is genuinely a top-quantile nearness decision.
    near_longs = [
        sig
        for sig in _entries(anchoring.events.items)
        if sig.symbol == "NEAR/USDT" and sig.signal_type == "LONG"
    ]
    assert near_longs, "expected a LONG entry on the near-high name"
    meta = near_longs[-1].metadata or {}
    assert meta.get("nearness") is not None and meta["nearness"] > 0.95, meta
    assert float(meta.get("target_allocation", 0.0)) > 0.0


# --------------------------------------------------------------------------- #
# Item-2 breadth: a young alt (<52wk but >= min-history) is ADMITTED via the
# max_available fallback; a symbol below min-history is skipped without raising.
# --------------------------------------------------------------------------- #

_BREADTH_N = 30


def _young_series() -> tuple[list[float], list[float], list[float]]:
    # 15 bars only: ramps to a high and sits pinned to it (highest nearness).
    base = [*_linspace(100.0, 120.0, 11), 119.6, 119.5, 119.7, 119.5]
    assert len(base) == 15
    return _with_high(_jitter(base, seed=606))


def test_item2_breadth_young_alt_admitted_via_max_available_fallback() -> None:
    symbols = [
        "YOUNG/USDT",
        "TOOSHORT/USDT",
        "CTRLA/USDT",
        "CTRLB/USDT",
        "CTRLC/USDT",
        "CTRLD/USDT",
    ]
    series: Series = {
        "YOUNG/USDT": _young_series(),  # 15 bars, appears from bar 15
        "TOOSHORT/USDT": _with_high(_linspace(100.0, 105.0, 5)),  # 5 bars, from bar 25
        "CTRLA/USDT": _ctrl_series(_BREADTH_N, 110.0, 99.0, seed=707),
        "CTRLB/USDT": _ctrl_series(_BREADTH_N, 108.0, 96.0, seed=808),
        "CTRLC/USDT": _ctrl_series(_BREADTH_N, 112.0, 100.0, seed=909),
        "CTRLD/USDT": _ctrl_series(_BREADTH_N, 106.0, 95.0, seed=111),
    }
    first_bar = {"YOUNG/USDT": 15, "TOOSHORT/USDT": 25}

    strategy = CrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols),
        _Queue(),
        high_lookback_bars=40,  # "52wk-equivalent" > any symbol's available history
        min_history_bars=10,
        vol_window=5,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=0,
        allow_short=True,
        min_symbols=4,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    _feed_window(strategy, symbols, series, _BREADTH_N, first_bar=first_bar)
    signals = strategy.events.items
    side = _final_side(signals)

    # The young alt (15 bars < 40-bar window, >= 10-bar floor) is ADMITTED and,
    # being the highest-nearness name, is selected LONG -- proof it contributed
    # a score rather than being dropped.
    assert side.get("YOUNG/USDT") == "LONG", side
    young_longs = [
        sig for sig in _entries(signals) if sig.symbol == "YOUNG/USDT" and sig.signal_type == "LONG"
    ]
    assert young_longs, "young alt should have been admitted into the ranking"
    meta = young_longs[-1].metadata or {}
    # Admitted specifically through the max_available window, NOT the full 52wk:
    # the young alt is scored from the moment it clears the min-history floor
    # (10 bars) using its own available window, always shorter than the 40-bar
    # full lookback.
    assert meta.get("full_lookback") is False, meta
    assert 10 <= meta.get("lookback_used") < 40, meta

    # The below-min-history symbol is skipped entirely -- never scored, never a
    # signal -- and nothing raised getting here.
    assert all(sig.symbol != "TOOSHORT/USDT" for sig in signals)


# --------------------------------------------------------------------------- #
# Determinism: two identical runs -> bit-identical signal stream.
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    symbols, series = _b2_universe()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = CrossSectionalNearHighAnchoringStrategy(
            _Bars(symbols), _Queue(), **_A1_B2_KWARGS
        )
        _feed_window(strategy, symbols, series, _B2_N)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# State roundtrip + adversarial set_state.
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    symbols, series = _b2_universe()
    strategy = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    _feed_window(strategy, symbols, series, _B2_N)
    snapshot = strategy.get_state()

    restored = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in symbols:
        r = restored._state[symbol]
        o = strategy._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert list(r.highs) == list(o.highs)
        assert r.mode == o.mode
        assert r.bars_held == o.bars_held
        assert r.last_time_key == o.last_time_key
    assert restored._tick == strategy._tick
    assert restored._last_eval_time_key == strategy._last_eval_time_key


def test_restored_state_reproduces_behavior() -> None:
    """A strategy resumed mid-stream from a snapshot carries the same book and
    converges to the same final internal state as one fed the whole history."""
    symbols, series = _b2_universe()
    split = _B2_N - 4

    full = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    _feed_window(full, symbols, series, _B2_N)

    warm = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    _feed_window(warm, symbols, series, split)
    snapshot = warm.get_state()

    resumed = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    resumed.set_state(snapshot)
    # Feed only the remaining tail (bars split..N-1) through the resumed strategy;
    # it inherits the warmed-up deques/positions purely from the snapshot.
    for idx in range(split, _B2_N):
        bars_1s = {
            sym: [
                {
                    "time": _ts(idx),
                    "open": series[sym][0][idx],
                    "high": series[sym][1][idx],
                    "low": series[sym][0][idx],
                    "close": series[sym][0][idx],
                    "volume": series[sym][2][idx],
                }
            ]
            for sym in symbols
        }
        resumed.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_ts(idx), bars_1s=bars_1s)
        )

    # The full run resolves to {NEAR long, FAR short}; the resumed run holds the
    # identical book in its internal state (positions are carried by the
    # snapshot, so they are not necessarily re-emitted as fresh signals).
    assert _final_side(full.events.items) == {"NEAR/USDT": "LONG", "FAR/USDT": "SHORT"}
    for symbol in symbols:
        assert resumed._state[symbol].mode == full._state[symbol].mode, symbol
    assert resumed._state["NEAR/USDT"].mode == "LONG"
    assert resumed._state["FAR/USDT"].mode == "SHORT"


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strategy = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 12345}}})
    strategy.set_state({"symbol_state": {"A/USDT": {"highs": {"nested": "dict"}}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "highs": {"unexpected": "type"},
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

    # Still functions afterwards, using the SAME symbols it was constructed with.
    series: Series = {
        "A/USDT": _near_series(),
        "B/USDT": _momup_series(),
        "C/USDT": _ctrl_series(_B2_N, 110.0, 99.0, seed=404),
        "D/USDT": _ctrl_series(_B2_N, 108.0, 96.0, seed=505),
        "E/USDT": _far_series(),
    }
    _feed_window(strategy, symbols, series, _B2_N)  # must not raise


# --------------------------------------------------------------------------- #
# Never-raise on degenerate input.
# --------------------------------------------------------------------------- #


def _market_event(symbol: str, idx: int, close: Any, high: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=_ts(idx),
        symbol=symbol,
        open=close,
        high=close if high is None else high,
        low=close,
        close=close,
        volume=1000.0,
    )


def test_degenerate_market_events_never_raise() -> None:
    strategy = CrossSectionalNearHighAnchoringStrategy(_Bars(["Z/USDT"]), _Queue(), **_A1_B2_KWARGS)
    strategy.calculate_signals(_market_event("Z/USDT", 0, 0.0))
    strategy.calculate_signals(_market_event("Z/USDT", 1, -5.0))
    strategy.calculate_signals(_market_event("Z/USDT", 2, float("nan")))
    strategy.calculate_signals(_market_event("Z/USDT", 3, float("inf")))
    strategy.calculate_signals(_market_event("Z/USDT", 4, None))
    assert strategy.events.items == []


def test_empty_and_unknown_events_never_raise() -> None:
    strategy = CrossSectionalNearHighAnchoringStrategy(
        _Bars(["Z/USDT", "Y/USDT"]), _Queue(), **_A1_B2_KWARGS
    )
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", bars_1s={}, time="t0"))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="UNKNOWN/USDT", close=None))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", close=None))
    assert strategy.events.items == []


def test_self_skip_below_min_symbols() -> None:
    symbols = ["A/USDT", "B/USDT"]  # min_symbols is 5
    series: Series = {
        "A/USDT": _near_series(),
        "B/USDT": _far_series(),
    }
    strategy = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    _feed_window(strategy, symbols, series, _B2_N)
    assert strategy.events.items == []


def test_self_skip_when_history_too_short() -> None:
    symbols, series = _b2_universe()
    # Only 4 bars fed: far below the vol/min-history floors for every symbol.
    strategy = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_A1_B2_KWARGS)
    _feed_window(strategy, symbols, series, 4)
    assert strategy.events.items == []


# --------------------------------------------------------------------------- #
# Behaviour: long/short quantile membership on a hand-ordered nearness ladder.
# --------------------------------------------------------------------------- #


def test_quantile_membership_on_ordered_nearness() -> None:
    n = 24
    peak = 120.0
    targets = [0.99, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80, 0.70]
    symbols = [f"S{i}/USDT" for i in range(len(targets))]
    series: Series = {
        symbols[i]: _ctrl_series(n, peak, peak * targets[i], seed=1000 + i)
        for i in range(len(targets))
    }
    strategy = CrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols),
        _Queue(),
        high_lookback_bars=n,
        min_history_bars=5,
        vol_window=5,
        quantile_pct=0.25,  # 8 eligible -> 2 per side
        rebalance_bars=1,
        min_hold_bars=0,
        allow_short=True,
        min_symbols=4,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    _feed_window(strategy, symbols, series, n)
    side = _final_side(strategy.events.items)

    # Top-2 nearness -> LONG, bottom-2 -> SHORT, the middle four flat.
    assert side.get("S0/USDT") == "LONG", side
    assert side.get("S1/USDT") == "LONG", side
    assert side.get("S6/USDT") == "SHORT", side
    assert side.get("S7/USDT") == "SHORT", side
    for mid in ("S2/USDT", "S3/USDT", "S4/USDT", "S5/USDT"):
        assert mid not in side, (mid, side)


def test_long_only_when_short_disabled() -> None:
    n = 24
    peak = 120.0
    targets = [0.99, 0.95, 0.92, 0.90, 0.88, 0.70]
    symbols = [f"S{i}/USDT" for i in range(len(targets))]
    series: Series = {
        symbols[i]: _ctrl_series(n, peak, peak * targets[i], seed=2000 + i)
        for i in range(len(targets))
    }
    strategy = CrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols),
        _Queue(),
        high_lookback_bars=n,
        min_history_bars=5,
        vol_window=5,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=0,
        allow_short=False,  # long-only mode
        min_symbols=4,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    _feed_window(strategy, symbols, series, n)
    assert [sig for sig in strategy.events.items if str(sig.signal_type).upper() == "SHORT"] == []
    assert _final_side(strategy.events.items).get("S0/USDT") == "LONG"


def test_min_hold_suppresses_flip_inside_hold_window() -> None:
    """A would-be side flip inside the min-hold window is suppressed."""
    n = 24

    # FLIP/USDT is the UNAMBIGUOUS top-nearness name early (always making new
    # highs -> nearness ~1.0) and the UNAMBIGUOUS bottom-nearness name late (it
    # crashes far below its trailing high).  Every other name sets an early high
    # and stays a fixed fraction below it throughout, so it is below FLIP early
    # and above FLIP late.  With a min-hold longer than the run, the early LONG
    # must persist rather than flipping to SHORT.
    def _flip_series() -> tuple[list[float], list[float], list[float]]:
        base = _linspace(100.0, 130.0, 12) + _linspace(130.0, 70.0, 13)[1:]
        return _with_high(_jitter(base, seed=3001))

    def _mid_series(
        high: float, low: float, recover: float, seed: int
    ) -> tuple[list[float], list[float], list[float]]:
        # Early high at bar 0, decline, then partial recovery: nearness stays a
        # moderate fraction below the (fixed) trailing high the whole run.
        base = _linspace(high, low, 12) + _linspace(low, recover, 13)[1:]
        return _with_high(_jitter(base, seed=seed))

    symbols = ["FLIP/USDT", "REV/USDT", "M1/USDT", "M2/USDT", "M3/USDT"]
    series: Series = {
        "FLIP/USDT": _flip_series(),
        "REV/USDT": _mid_series(110.0, 100.0, 108.0, seed=3002),
        "M1/USDT": _mid_series(111.0, 101.0, 109.0, seed=3003),
        "M2/USDT": _mid_series(112.0, 102.0, 110.0, seed=3004),
        "M3/USDT": _mid_series(113.0, 103.0, 111.0, seed=3005),
    }
    strategy = CrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols),
        _Queue(),
        high_lookback_bars=n,
        min_history_bars=5,
        vol_window=5,
        quantile_pct=0.2,
        rebalance_bars=1,
        min_hold_bars=n,  # longer than the whole run: no flip may occur once held
        allow_short=True,
        min_symbols=4,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    _feed_window(strategy, symbols, series, n)
    flip_signals = [sig for sig in strategy.events.items if sig.symbol == "FLIP/USDT"]
    # FLIP entered LONG early and, held by the min-hold floor, never flipped to
    # SHORT despite becoming the lowest-nearness name later.
    kinds = [str(sig.signal_type).upper() for sig in flip_signals]
    assert "LONG" in kinds
    assert "SHORT" not in kinds


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalNearHighAnchoringStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "high_lookback_bars",
        "min_history_bars",
        "vol_window",
        "quantile_pct",
        "rebalance_bars",
        "min_hold_bars",
        "allow_short",
        "min_symbols",
        "target_gross_exposure",
        "target_vol",
        "stop_loss_pct",
        "max_hold_bars",
        "base_allocation",
        "max_symbol_exposure_pct",
        "max_order_value",
        "min_price",
    ):
        assert required in schema
    for cap in ("base_allocation", "max_symbol_exposure_pct", "max_order_value"):
        assert schema[cap].tunable is False


# --------------------------------------------------------------------------- #
# vol-target HORIZON fix: the ``min(1, target_vol / portfolio_vol)`` scalar must
# annualize the per-bar portfolio vol (via sqrt(bars_per_year) inferred from the
# observed bar spacing) before the Moreira-Muir clamp.  Nearness spread is fixed
# by the (elevated, constant) trailing highs; per-bar vol is controlled purely by
# the jitter amplitude, so calm vs stormy inputs isolate the throttle.  The OLD
# inert code compared 0.20 against a per-bar 0.0005..0.04 and always returned
# scalar == 1.0, so ``stormy_scalar < 1.0`` is itself proof the fix engaged.
# --------------------------------------------------------------------------- #

# (symbol, constant trailing high) -- close is flat ~100 so nearness = 100 / high;
# highs stay >= any jittered close (< ~104 at amp 0.03) so the ordering is stable.
_VT_LEVELS: list[tuple[str, float]] = [
    ("VL0/USDT", 104.0),  # highest nearness -> LONG
    ("VL1/USDT", 106.0),
    ("VM0/USDT", 118.0),
    ("VM1/USDT", 130.0),
    ("VS0/USDT", 150.0),
    ("VS1/USDT", 185.0),  # lowest nearness -> SHORT
]
_VT_N = 20
_VT_KWARGS: dict[str, Any] = dict(
    high_lookback_bars=30,
    min_history_bars=6,
    vol_window=5,
    quantile_pct=0.16,  # int(0.16 * 6) -> n_side clamps to 1 (extreme long/short)
    rebalance_bars=1,
    min_hold_bars=0,
    allow_short=True,
    min_symbols=4,
    target_gross_exposure=1.0,
    target_vol=0.20,
    base_allocation=0.20,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)


def _feed_daily(strategy: Any, symbols: list[str], series: Series, n: int) -> None:
    """Feed ``n`` MARKET_WINDOW bars spaced ONE DAY apart (epoch-second times)."""
    base_epoch = 1_700_000_000
    for idx in range(n):
        bars_1s: dict[str, list[dict[str, Any]]] = {}
        stamp = base_epoch + idx * 86_400
        for symbol in symbols:
            closes, highs, volumes = series[symbol]
            if idx >= len(closes):
                continue
            bars_1s[symbol] = [
                {
                    "time": stamp,
                    "open": closes[idx],
                    "high": highs[idx],
                    "low": closes[idx],
                    "close": closes[idx],
                    "volume": volumes[idx],
                }
            ]
        event = SimpleNamespace(type="MARKET_WINDOW", time=stamp, bars_1s=bars_1s)
        strategy.calculate_signals(event)


def _vt_universe(amp: float) -> tuple[list[str], Series]:
    symbols = [name for name, _ in _VT_LEVELS]
    series: Series = {}
    for offset, (name, high) in enumerate(_VT_LEVELS):
        closes = _jitter([100.0] * _VT_N, seed=971 + offset, amp=amp)
        series[name] = (closes, [high] * _VT_N, [1000.0] * _VT_N)
    return symbols, series


def _entry_scalars(signals: list[Any]) -> list[float]:
    return [
        float((sig.metadata or {}).get("vol_target_scalar"))
        for sig in _entries(signals)
        if (sig.metadata or {}).get("vol_target_scalar") is not None
    ]


def _entry_alloc_sum(signals: list[Any]) -> float:
    return sum(
        float((sig.metadata or {}).get("target_allocation", 0.0)) for sig in _entries(signals)
    )


def test_vol_target_scalar_throttles_high_vol_at_daily_spacing() -> None:
    """Stormy input trips the vol-target clamp (scalar < 1); calm input passes
    through at scalar == 1.  The stormy book carries a strictly smaller notional
    -- the fix the horizon-mismatch bug (inert scalar) leaves impossible."""
    calm_symbols, calm_series = _vt_universe(amp=0.0004)
    calm = CrossSectionalNearHighAnchoringStrategy(_Bars(calm_symbols), _Queue(), **_VT_KWARGS)
    _feed_daily(calm, calm_symbols, calm_series, _VT_N)
    calm_scalars = _entry_scalars(calm.events.items)

    storm_symbols, storm_series = _vt_universe(amp=0.03)
    storm = CrossSectionalNearHighAnchoringStrategy(_Bars(storm_symbols), _Queue(), **_VT_KWARGS)
    _feed_daily(storm, storm_symbols, storm_series, _VT_N)
    storm_scalars = _entry_scalars(storm.events.items)

    assert calm_scalars, "expected at least one calm entry carrying a vol_target_scalar"
    assert storm_scalars, "expected at least one stormy entry carrying a vol_target_scalar"

    # (ii) calm input -> throttle passes through (annualized vol << target_vol).
    assert max(calm_scalars) == 1.0 and min(calm_scalars) == 1.0, calm_scalars
    # (i) stormy input -> throttle ENGAGES (annualized vol > target_vol).  Under
    # the pre-fix per-bar comparison this scalar was pinned at 1.0.
    assert max(storm_scalars) < 1.0, storm_scalars
    # ... and the reduced scalar visibly shrinks the emitted notional.
    assert _entry_alloc_sum(storm.events.items) < _entry_alloc_sum(calm.events.items)


def test_vol_target_scalar_passthrough_when_spacing_unavailable() -> None:
    """A single-bar-spacing feed (all bars share one timestamp key is impossible
    here, so we feed identical epochs collapsed to one accepted bar) cannot infer
    spacing; the throttle must PASS THROUGH (scalar == 1.0) rather than guess."""
    symbols, series = _vt_universe(amp=0.03)
    strategy = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_VT_KWARGS)
    # Feed with a constant timestamp: only the first bar per symbol is ingested
    # (dedup on last_time_key), so at most one decision-bar epoch is recorded and
    # bar spacing is never inferable.
    for _ in range(_VT_N):
        bars_1s = {
            name: [
                {
                    "time": 1_700_000_000,
                    "open": series[name][0][0],
                    "high": series[name][1][0],
                    "low": series[name][0][0],
                    "close": series[name][0][0],
                    "volume": 1000.0,
                }
            ]
            for name in symbols
        }
        strategy.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=1_700_000_000, bars_1s=bars_1s)
        )
    # With < 2 distinct timestamps the annualizer returns None -> pass-through.
    for scalar in _entry_scalars(strategy.events.items):
        assert scalar == 1.0


def test_vol_target_fix_determinism_two_runs_identical() -> None:
    """Re-running the throttled path twice yields byte-identical signals."""
    symbols, series = _vt_universe(amp=0.03)

    def _run() -> list[tuple[Any, ...]]:
        strat = CrossSectionalNearHighAnchoringStrategy(_Bars(symbols), _Queue(), **_VT_KWARGS)
        _feed_daily(strat, symbols, series, _VT_N)
        return [
            (
                sig.symbol,
                str(sig.signal_type),
                round(float((sig.metadata or {}).get("vol_target_scalar", 0.0)), 12),
                round(float((sig.metadata or {}).get("target_allocation", 0.0)), 12),
            )
            for sig in strat.events.items
        ]

    assert _run() == _run()
