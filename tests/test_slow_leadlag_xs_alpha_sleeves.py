"""Deterministic tests for SlowCrossSectionalLeadLagStrategy (A2, CONDITIONAL).

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

This lane is a keep-or-drop experiment.  The two-part BUILD GATE below is the
keep/drop key:

  (i)  DIVERGENCE -- on one synthetic input, the broad-XS lead-lag loading book
       is materially different from ALL THREE verified lead-lag incumbents,
       imported directly and driven through their own interfaces:
         - ``CrossCryptoSlowDiffusionStrategy``  (single fixed BTC->ETH pair)
         - ``SemisLeadLagRotationStrategy``      (single equity-leader trigger)
         - ``IntermarketLeadLagContinuationStrategy`` (fixed commodity->equity
           pairs; already daily + hold-suppressed -- occupies the low-turnover
           claim, so it is NOT omittable from the gate).
  (ii) RPT-SURVIVABILITY DESIGN property -- with ``min_hold_bars >= N`` the
       synthetic turnover over the feed collapses to <= a stated, N-derived
       threshold (deterministic entry count on an alternating-loading feed whose
       ranking flips EVERY week).

If either gate part could not pass with an honest implementation, this lane
DROPS (delete both files) per the plan -- weakening the gate to pass is
forbidden.

All pseudo-randomness is drawn from a small seeded linear-congruential generator
(no ``random`` module), so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.cross_asset_trend_alpha_sleeves import (
    IntermarketLeadLagContinuationStrategy,
)
from lumina_quant.strategies.cross_crypto_slow_diffusion import (
    CrossCryptoSlowDiffusionStrategy,
)
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    SemisLeadLagRotationStrategy,
)
from lumina_quant.strategies.slow_leadlag_xs_alpha_sleeves import (
    SlowCrossSectionalLeadLagStrategy,
)
from lumina_quant.tuning import HyperParam

_LEADERS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")


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


class _Aggregator:
    """Minimal timeframe-aggregator stand-in for the pair-diffusion incumbent.

    ``get_bars`` returns ``(time, o, h, l, c, v)`` rows including a currently-
    forming last bar (the incumbent drops it), so the growing weekly history is
    replayed exactly as the aggregator contract expects.
    """

    def __init__(self, rows_by_symbol: dict[str, list[tuple[Any, ...]]]) -> None:
        self._rows = rows_by_symbol

    def get_bars(self, symbol: str, timeframe: str, n: int) -> list[tuple[Any, ...]]:
        _ = timeframe
        rows = self._rows.get(symbol, [])
        return list(rows[-n:]) if n else list(rows)


def _week_iso(index: int) -> str:
    base = datetime.datetime(2025, 1, 6, tzinfo=datetime.UTC)
    return (base + datetime.timedelta(weeks=index)).isoformat()


def _bar_row(time_str: str, price: float, volume: float = 1000.0) -> tuple[Any, ...]:
    return (time_str, price, price, price, price, volume)


def _window_event(
    index: int, symbols: list[str], prices: dict[str, list[float]]
) -> SimpleNamespace:
    bars_1s = {symbol: [_bar_row(_week_iso(index), prices[symbol][index])] for symbol in symbols}
    return SimpleNamespace(type="MARKET_WINDOW", time=_week_iso(index), bars_1s=bars_1s)


def _feed_window(strategy: Any, symbols: list[str], prices: dict[str, list[float]]) -> None:
    n = len(prices[symbols[0]])
    for index in range(n):
        strategy.calculate_signals(_window_event(index, symbols, prices))


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _entry_count(signals: list[Any]) -> int:
    return sum(1 for sig in signals if str(sig.signal_type).upper() in {"LONG", "SHORT"})


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# deterministic price-path builders
# --------------------------------------------------------------------------- #


def _compound(returns: list[float], p0: float = 100.0) -> list[float]:
    prices = [p0]
    for ret in returns:
        prices.append(prices[-1] * (1.0 + ret))
    return prices


def _jitter(index: int, seed: int, scale: float = 0.004) -> float:
    """A single deterministic LCG draw folded into a return (keeps vol non-zero)."""
    gen = _lcg_stream(seed=seed + index)
    return (next(gen) - 0.5) * scale


def _lag1(series: list[float], gain: float) -> list[float]:
    """A follower that loads ``gain`` on the lag-1 leader return (a laggard echo)."""
    return [0.0] + [gain * value for value in series[:-1]]


def _trending_panel(n: int = 52) -> tuple[list[str], dict[str, list[float]]]:
    """Four DISTINCT trending leaders + followers with distinct lead-lag loadings.

    The loading ranking is stable over time (a trending, non-oscillating panel),
    so the book converges and the behaviour / divergence claims are decidable.
    """
    l1 = [0.030 * math.sin(i * 0.55 + 0.1) + _jitter(i, 11) for i in range(n)]
    l2 = [0.025 * math.cos(i * 0.40 + 0.3) + _jitter(i, 22) for i in range(n)]
    l3 = [0.020 * math.sin(i * 0.70 + 0.9) + _jitter(i, 33) for i in range(n)]
    l4 = [0.028 * math.cos(i * 0.33 + 1.7) + _jitter(i, 44) for i in range(n)]
    strong = [v + _jitter(i, 55) for i, v in enumerate(_lag1(l1, 1.0))]
    mild = [v + _jitter(i, 66) for i, v in enumerate(_lag1(l2, 0.5))]
    neg = [v + _jitter(i, 77) for i, v in enumerate(_lag1(l3, -1.0))]
    noise = [0.001 * math.sin(i * 2.1) + _jitter(i, 88) for i in range(n)]
    prices = {
        "BTC/USDT": _compound(l1),
        "ETH/USDT": _compound(l2),
        "BNB/USDT": _compound(l3),
        "SOL/USDT": _compound(l4),
        "STRONG/USDT": _compound(strong),
        "MILD/USDT": _compound(mild),
        "NEG/USDT": _compound(neg),
        "NOISE/USDT": _compound(noise),
    }
    return list(prices), prices


def _alternating_panel(n: int = 64) -> tuple[list[str], dict[str, list[float]]]:
    """Leaders flip sign every week, so every follower's loading flips weekly.

    This is the churn stress-test: absent a hold brake the book would flip on
    every single weekly decision.  It is the input for the RPT design property.
    """
    alt = [(0.05 if (i % 2 == 0) else -0.05) for i in range(n)]
    l1 = [alt[i] * (1.0 + 0.10 * math.sin(i)) + _jitter(i, 101) for i in range(n)]
    l2 = [alt[i] * (1.0 + 0.10 * math.cos(i)) + _jitter(i, 202) for i in range(n)]
    l3 = [alt[i] * (1.0 + 0.10 * math.sin(i * 1.3)) + _jitter(i, 303) for i in range(n)]
    l4 = [alt[i] * (1.0 + 0.10 * math.cos(i * 0.7)) + _jitter(i, 404) for i in range(n)]
    prices = {
        "BTC/USDT": _compound(l1),
        "ETH/USDT": _compound(l2),
        "BNB/USDT": _compound(l3),
        "SOL/USDT": _compound(l4),
        "ECHO/USDT": _compound([v + _jitter(i, 505) for i, v in enumerate(_lag1(l1, 1.0))]),
        "ANTI/USDT": _compound([v + _jitter(i, 606) for i, v in enumerate(_lag1(l2, -1.0))]),
        "ECHO2/USDT": _compound([v + _jitter(i, 707) for i, v in enumerate(_lag1(l3, 1.0))]),
        "ANTI2/USDT": _compound([v + _jitter(i, 808) for i, v in enumerate(_lag1(l4, -1.0))]),
    }
    return list(prices), prices


_MIN_HISTORY = 40


def _make(symbols: list[str], **overrides: Any) -> SlowCrossSectionalLeadLagStrategy:
    kwargs: dict[str, Any] = dict(
        leader_symbols=",".join(_LEADERS),
        min_history=_MIN_HISTORY,
        min_symbols=4,
        entry_z=0.15,
        exit_z=0.0,
        max_longs=2,
        max_shorts=2,
        min_hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        target_vol=0.0,
    )
    kwargs.update(overrides)
    return SlowCrossSectionalLeadLagStrategy(_Bars(symbols), _Queue(), **kwargs)


# --------------------------------------------------------------------------- #
# (i) BUILD GATE part 1 -- divergence vs ALL THREE incumbents
# --------------------------------------------------------------------------- #


def test_build_gate_divergence_vs_all_three_incumbents() -> None:
    symbols, prices = _trending_panel()

    # --- our sleeve: a broad cross-sectional lead-lag long-short book ---------
    mine = _make(symbols, max_longs=2, max_shorts=2, min_hold_bars=4)
    _feed_window(mine, symbols, prices)
    my_book = _final_side(mine.events.items)
    assert my_book, "the broad-XS lead-lag sleeve must produce a non-empty book"
    # The book ranks the ALT followers by loading; it never trades a leader.
    assert set(my_book).isdisjoint(set(_LEADERS)), my_book
    assert len(my_book) >= 2, my_book

    # --- incumbent 1: single fixed BTC->ETH pair (driven via its aggregator) --
    cross1 = CrossCryptoSlowDiffusionStrategy(_Bars(symbols), _Queue())
    n = len(prices[symbols[0]])
    for index in range(n):
        rows_by_symbol = {
            symbol: [_bar_row(_week_iso(j), prices[symbol][j]) for j in range(index + 1)]
            for symbol in symbols
        }
        event = SimpleNamespace(type="MARKET_WINDOW", time=_week_iso(index))
        cross1.calculate_signals_window(event, _Aggregator(rows_by_symbol))
    book1 = _final_side(cross1.events.items)
    # Structural fact: the pair-diffusion incumbent can only ever touch its one
    # configured target symbol -- never a broad book.
    assert set(book1) <= {"ETH/USDT"}, book1
    assert my_book != book1
    assert set(my_book) != set(book1)

    # --- incumbent 2: single equity-leader trigger ---------------------------
    # Default config: its named equity leader (SOXL) is absent from a crypto
    # universe, so it can never rebalance -> architecturally flat.
    cross2 = SemisLeadLagRotationStrategy(_Bars(symbols), _Queue())
    _feed_window(cross2, symbols, prices)
    book2 = _final_side(cross2.events.items)
    assert book2 == {}, book2
    assert my_book != book2

    # Non-strawman strengthening: even RE-POINTED at a present crypto leader so
    # it CAN fire, its under-reaction-gap book still differs from our
    # spillover-loading book.
    cross2_btc = SemisLeadLagRotationStrategy(
        _Bars(symbols), _Queue(), leader_symbol="BTC/USDT", min_symbols=4
    )
    _feed_window(cross2_btc, symbols, prices)
    assert my_book != _final_side(cross2_btc.events.items)

    # --- incumbent 3: fixed commodity->equity pairs (occupies low-turnover) ---
    # Default config: none of its configured pairs (CL/BZ->XLE) exist here, so
    # it has zero legs -> architecturally flat.
    cross3 = IntermarketLeadLagContinuationStrategy(_Bars(symbols), _Queue())
    _feed_window(cross3, symbols, prices)
    book3 = _final_side(cross3.events.items)
    assert book3 == {}, book3
    assert my_book != book3

    # Non-strawman strengthening: even RE-POINTED at a present crypto pair so it
    # forms a leg and trades, its single-pair continuation book differs from our
    # broad book.
    cross3_pair = IntermarketLeadLagContinuationStrategy(
        _Bars(symbols), _Queue(), pair_spec="BTC/USDT>STRONG/USDT", entry_z=0.0, min_leader_move=0.0
    )
    _feed_window(cross3_pair, symbols, prices)
    assert my_book != _final_side(cross3_pair.events.items)

    # The decisive claim: our book contains at least one name that NONE of the
    # three incumbents (in any of the configurations above) produced.
    incumbent_names = (
        set(book1)
        | set(book2)
        | set(book3)
        | set(_final_side(cross2_btc.events.items))
        | set(_final_side(cross3_pair.events.items))
    )
    assert set(my_book) - incumbent_names, (my_book, incumbent_names)


# --------------------------------------------------------------------------- #
# (ii) BUILD GATE part 2 -- RPT-survivability design property (min-hold)
# --------------------------------------------------------------------------- #


def test_build_gate_min_hold_caps_turnover() -> None:
    # Low-turnover design point (stated): N = 8 weekly decision bars of minimum
    # hold.  On a feed whose loading ranking flips EVERY week, min-hold >= N caps
    # each of the S = max_longs + max_shorts slots to at most one re-entry per N
    # bars, so the entry count collapses from the churn baseline.
    symbols, prices = _alternating_panel(n=64)
    n = len(prices[symbols[0]])
    slots = 2 + 2  # max_longs + max_shorts
    m_rankable = n - _MIN_HISTORY  # weekly decisions that can rank (stated)
    design_n = 8

    def _entries(min_hold: int) -> int:
        strat = _make(
            symbols,
            entry_z=0.2,
            max_longs=2,
            max_shorts=2,
            min_hold_bars=min_hold,
            cooldown_bars=0,
            stop_loss_pct=0.0,
        )
        _feed_window(strat, symbols, prices)
        return _entry_count(strat.events.items)

    churn_baseline = _entries(1)  # no meaningful hold -> flips ~every week
    rescued = _entries(design_n)

    # (a) the input genuinely churns without the brake (property is non-vacuous)
    assert churn_baseline >= slots * 4, churn_baseline
    # (b) absolute low-turnover bound, DERIVED from N: <= slots * (M/N + margin)
    turnover_bound = slots * (m_rankable // design_n + 2)
    assert rescued <= turnover_bound, (rescued, turnover_bound)
    # (c) the min-hold brake delivers a large (>=3x) turnover reduction
    assert churn_baseline >= 3 * rescued, (churn_baseline, rescued)
    # monotonicity: more hold -> not more turnover
    assert _entries(12) <= rescued


# --------------------------------------------------------------------------- #
# behaviour: high-loading laggard longed, low-loading shorted
# --------------------------------------------------------------------------- #


def test_high_loading_longed_low_loading_shorted() -> None:
    symbols, prices = _trending_panel()
    strat = _make(symbols, max_longs=2, max_shorts=2, entry_z=0.15)
    _feed_window(strat, symbols, prices)

    # The sleeve's OWN loading ranking on the warmed weekly panel (its real
    # `cross_leadlag_spillover` call), mapped to the long/short book by its real
    # selection logic -- no mock, no re-derivation.
    scores = strat._spillover_scores()
    assert scores, "the warmed sleeve must produce lead-lag loadings"
    targets = strat._desired_targets(scores)
    longs = [symbol for symbol, mode in targets.items() if mode == "LONG"]
    shorts = [symbol for symbol, mode in targets.items() if mode == "SHORT"]
    assert longs and shorts, targets

    # The highest-loading eligible laggard is LONG; the lowest-loading is SHORT.
    top = max(scores, key=lambda key: scores[key])
    bottom = min(scores, key=lambda key: scores[key])
    assert targets.get(top) == "LONG", (top, scores)
    assert targets.get(bottom) == "SHORT", (bottom, scores)
    # Every longed loading strictly exceeds every shorted loading (clean split).
    assert min(scores[s] for s in longs) > max(scores[s] for s in shorts)

    # End-to-end sign invariant: over the full feed, every LONG entry recorded a
    # positive loading and every SHORT entry a negative loading, and the sleeve
    # genuinely exercised BOTH legs.
    long_loadings = [
        s.metadata["leadlag_loading"] for s in strat.events.items if s.signal_type == "LONG"
    ]
    short_loadings = [
        s.metadata["leadlag_loading"] for s in strat.events.items if s.signal_type == "SHORT"
    ]
    assert long_loadings and short_loadings, "expected both LONG and SHORT entries"
    assert all(value > 0.0 for value in long_loadings), long_loadings
    assert all(value < 0.0 for value in short_loadings), short_loadings


# --------------------------------------------------------------------------- #
# determinism: two identical runs -> identical signal streams
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    symbols, prices = _trending_panel()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strat = _make(symbols)
        _feed_window(strat, symbols, prices)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strat.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# state roundtrip + adversarial set_state
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    symbols, prices = _trending_panel()
    strat = _make(symbols)
    _feed_window(strat, symbols, prices)
    snapshot = strat.get_state()

    restored = _make(symbols)
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in symbols:
        r = restored._state[symbol]
        o = strat._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert r.mode == o.mode
        assert r.bars_held == o.bars_held
        assert r.bars_since_exit == o.bars_since_exit
        assert r.last_bar_key == o.last_bar_key
    assert restored._tick == strat._tick
    assert restored._last_decision_week == strat._last_decision_week


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "A/USDT"]
    strat = _make(symbols)

    strat.set_state(None)  # type: ignore[arg-type]
    strat.set_state("not a dict")  # type: ignore[arg-type]
    strat.set_state(12345)  # type: ignore[arg-type]
    strat.set_state([])  # type: ignore[arg-type]
    strat.set_state({"symbol_state": "not a dict"})
    strat.set_state({"symbol_state": {"BTC/USDT": "not a dict either"}})
    strat.set_state({"symbol_state": {"BTC/USDT": {"closes": 12345}}})
    strat.set_state({"symbol_state": {"BTC/USDT": {"closes": {"nested": "dict"}}}})
    strat.set_state(
        {
            "last_decision_week": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "last_close": "abc",
                    "pending_dirty": "sometimes",
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "bars_since_exit": None,
                    "last_bar_key": 123,
                    "score": [1, 2, 3],
                }
                for symbol in symbols
            },
        }
    )
    for item in strat._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
        assert item.bars_held >= 0
        assert item.bars_since_exit >= 0

    # Still functions normally afterward (feed the symbols that carry a path).
    _, prices = _trending_panel()
    feed_symbols = [s for s in symbols if s in prices]
    _feed_window(strat, feed_symbols, {s: prices[s] for s in feed_symbols})  # must not raise


# --------------------------------------------------------------------------- #
# never-raise on degenerate input
# --------------------------------------------------------------------------- #


def test_degenerate_input_never_raises() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "A/USDT"]
    strat = _make(symbols)
    # zero / negative / NaN / inf closes, missing bars, empty window, wrong types
    strat.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strat.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    for index, value in enumerate((0.0, -5.0, float("nan"), float("inf"), 100.0)):
        bars_1s = {"BTC/USDT": [_bar_row(_week_iso(index), value)]}
        strat.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_week_iso(index), bars_1s=bars_1s)
        )
    assert _non_exit(strat.events.items) == []


def test_self_skip_below_min_symbols() -> None:
    symbols = ["BTC/USDT", "ETH/USDT"]  # below min_symbols/min for spillover
    _, full = _trending_panel()
    prices = {s: full[s] for s in symbols}
    strat = _make(symbols, min_symbols=4)
    _feed_window(strat, symbols, prices)
    assert _non_exit(strat.events.items) == []


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = SlowCrossSectionalLeadLagStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "leader_symbols",
        "max_lag",
        "ridge_alpha",
        "spillover_window",
        "min_history",
        "entry_z",
        "exit_z",
        "max_longs",
        "max_shorts",
        "allow_short",
        "min_symbols",
        "min_hold_bars",
        "cooldown_bars",
        "max_hold_bars",
        "vol_window",
        "target_gross_exposure",
        "target_vol",
        "stop_loss_pct",
        "base_allocation",
        "max_symbol_exposure_pct",
        "max_order_value",
        "min_price",
    ):
        assert required in schema
    for cap in ("base_allocation", "max_symbol_exposure_pct", "max_order_value", "leader_symbols"):
        assert schema[cap].tunable is False
