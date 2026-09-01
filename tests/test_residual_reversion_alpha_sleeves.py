"""Deterministic tests for StationarityGatedResidualReversionStrategy (N4).

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

The load-bearing test is the BUILD GATE (ADF isolation): the strategy is kept
ONLY if the augmented Dickey-Fuller stationarity gate can be shown to be the
sole mechanism differentiator versus the registered incumbent
``EquityBenchmarkResidualReversalStrategy``.  The two fixtures share an
identical steep final ``lookback``-bar residual descent, so the incumbent's
recent-move (reversion) signal is identical across them; they differ ONLY in
the SHAPE of the residual's history (a mean-reverting AR(1) -> stationary vs a
random walk -> non-stationary).  On the stationary residual BOTH fade it; on the
non-stationary residual the incumbent trades and N4 abstains, and flipping ONLY
N4's ``adf_reject_threshold`` on that same fixture flips N4 from abstain to
trade -- isolating the ADF gate.

All pseudo-randomness is drawn from a small seeded linear-congruential generator
(no ``random`` module), so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.stationarity import adf_t_statistic
from lumina_quant.strategies.cross_asset_tradfi_alpha_sleeves import (
    EquityBenchmarkResidualReversalStrategy,
)
from lumina_quant.strategies.residual_reversion_alpha_sleeves import (
    StationarityGatedResidualReversionStrategy,
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


def _market_event(symbol: str, idx: int, close: float, volume: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-{1 + idx // 24:02d}T{idx % 24:02d}:00:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _feed(strategy: Any, prices: dict[str, list[float]], volumes: dict[str, float | None]) -> None:
    symbols = list(prices.keys())
    n = len(prices[symbols[0]])
    for idx in range(n):
        for symbol in symbols:
            strategy.calculate_signals(
                _market_event(symbol, idx, prices[symbol][idx], volumes[symbol])
            )


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _entries(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() in {"LONG", "SHORT"}]


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# deterministic residual/price fixtures
# --------------------------------------------------------------------------- #

_N = 160
_LOOKBACK = 5
# Identical steep final descent shared by BOTH build-gate fixtures: the
# incumbent's `lookback`-bar residual (and hence its reversion signal) is the
# same across them, so ONLY the ADF verdict can differ.
_TAIL = [-0.030, -0.042, -0.054, -0.066, -0.078, -0.090]


def _benchmark_levels(n: int) -> list[float]:
    """A bounded (stationary) benchmark log-price path with non-degenerate returns."""
    gen = _lcg_stream(7)
    base = math.log(100.0)
    return [
        base + 0.02 * math.sin(2.0 * math.pi * t / 17.0) + (next(gen) - 0.5) * 0.004
        for t in range(n)
    ]


def _stationary_residual(n: int, *, seed: int = 23, tail: list[float] | None = None) -> list[float]:
    """AR(1) (phi<1) -> mean-reverting/stationary residual level series."""
    gen = _lcg_stream(seed)
    x = 0.0
    out: list[float] = []
    for _ in range(n):
        x = 0.5 * x + (next(gen) - 0.5) * 0.10
        out.append(x)
    tail = _TAIL if tail is None else tail
    out[n - len(tail) :] = tail
    return out


def _nonstationary_residual(n: int, *, seed: int = 9) -> list[float]:
    """Random walk (unit root) -> non-stationary residual level series."""
    gen = _lcg_stream(seed)
    level = 0.0
    out: list[float] = []
    for _ in range(n):
        level += (next(gen) - 0.5) * 0.024
        out.append(level)
    out[n - len(_TAIL) :] = _TAIL
    return out


def _prices_from_residual(
    residual: list[float], benchmark: list[float] | None = None
) -> dict[str, list[float]]:
    benchmark = _benchmark_levels(_N) if benchmark is None else benchmark
    return {
        "BENCH/USDT": [math.exp(v) for v in benchmark],
        "TARGET/USDT": [math.exp(benchmark[t] + residual[t]) for t in range(_N)],
    }


_SHARED_KWARGS: dict[str, Any] = dict(
    lookback_bars=_LOOKBACK,
    beta_window=60,
    vol_window=20,
    entry_score=0.45,
    rebalance_bars=1,
    max_longs=4,
    max_shorts=4,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)


def _incumbent_kwargs(**extra: Any) -> dict[str, Any]:
    kwargs = dict(
        _SHARED_KWARGS,
        equity_symbols="TARGET/USDT",
        benchmark_symbol="BENCH/USDT",
    )
    kwargs.update(extra)
    return kwargs


def _n4_kwargs(**extra: Any) -> dict[str, Any]:
    kwargs = dict(
        _SHARED_KWARGS,
        equity_symbols="TARGET/USDT",
        benchmark_symbol="BENCH/USDT",
        adf_window=120,
        adf_lags=0,
        adf_reject_threshold=-2.86,
        min_hold_bars=0,
        min_dollar_volume=0.0,
        hedge_benchmark=True,
    )
    kwargs.update(extra)
    return kwargs


def _run_incumbent(
    prices: dict[str, list[float]], **extra: Any
) -> EquityBenchmarkResidualReversalStrategy:
    strategy = EquityBenchmarkResidualReversalStrategy(
        _Bars(list(prices)), _Queue(), **_incumbent_kwargs(**extra)
    )
    _feed(strategy, prices, dict.fromkeys(prices, 1000.0))
    return strategy


def _run_n4(
    prices: dict[str, list[float]],
    volumes: dict[str, float | None] | None = None,
    **extra: Any,
) -> StationarityGatedResidualReversionStrategy:
    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(list(prices)), _Queue(), **_n4_kwargs(**extra)
    )
    _feed(strategy, prices, volumes or dict.fromkeys(prices, 1000.0))
    return strategy


# --------------------------------------------------------------------------- #
# (0) fixture sanity: the two residuals straddle the ADF reject threshold
# --------------------------------------------------------------------------- #


def test_fixtures_straddle_adf_threshold() -> None:
    stationary = _stationary_residual(_N)
    nonstationary = _nonstationary_residual(_N)
    adf_stat = adf_t_statistic(stationary[-120:], lags=0)
    adf_non = adf_t_statistic(nonstationary[-120:], lags=0)
    assert adf_stat is not None and adf_non is not None
    # Stationary residual clears the reject threshold; the random walk does not.
    assert adf_stat <= -2.86, adf_stat
    assert adf_non > -2.86, adf_non
    # The shared tail is identical (the incumbent's recent-move signal is equal).
    assert stationary[-(_LOOKBACK + 1) :] == nonstationary[-(_LOOKBACK + 1) :] == _TAIL


# --------------------------------------------------------------------------- #
# (a) BUILD GATE -- ADF isolation, two cases
# --------------------------------------------------------------------------- #


def test_build_gate_stationary_both_trade() -> None:
    """Case (ii): a stationary residual -> BOTH the incumbent and N4 fade it.

    A single terminal rebalance (``rebalance_bars == _N``) gives exactly one
    decision on the fully-formed residual window (no sliding-window flicker).
    """
    prices = _prices_from_residual(_stationary_residual(_N))
    incumbent_side = _final_side(_run_incumbent(prices, rebalance_bars=_N).events.items)
    n4_side = _final_side(_run_n4(prices, rebalance_bars=_N).events.items)
    assert incumbent_side.get("TARGET/USDT") is not None, incumbent_side
    assert n4_side.get("TARGET/USDT") is not None, n4_side
    # Agreement when the gate passes: same side (both LONG a beaten-down residual).
    assert n4_side["TARGET/USDT"] == incumbent_side["TARGET/USDT"] == "LONG"


def test_build_gate_nonstationary_incumbent_trades_n4_abstains() -> None:
    """Case (i): a non-stationary residual -> incumbent TRADES, N4 ABSTAINS (flat)."""
    prices = _prices_from_residual(_nonstationary_residual(_N))
    incumbent_side = _final_side(_run_incumbent(prices, rebalance_bars=_N).events.items)
    n4 = _run_n4(prices, rebalance_bars=_N)
    n4_target_entries = [sig for sig in _entries(n4.events.items) if sig.symbol == "TARGET/USDT"]
    # The incumbent (no stationarity gate) trades the dislocation ...
    assert incumbent_side.get("TARGET/USDT") is not None, incumbent_side
    # ... while N4 abstains entirely on the non-stationary residual.
    assert n4_target_entries == [], [s.signal_type for s in n4_target_entries]
    assert _final_side(n4.events.items).get("TARGET/USDT") is None


def test_build_gate_adf_is_sole_differentiator() -> None:
    """On the IDENTICAL non-stationary fixture, flipping ONLY the ADF threshold
    flips N4 abstain -> trade, isolating the ADF gate as the sole mechanism
    differentiator (and matching the incumbent's action once the gate opens)."""
    prices = _prices_from_residual(_nonstationary_residual(_N))
    incumbent_side = _final_side(_run_incumbent(prices, rebalance_bars=_N).events.items).get(
        "TARGET/USDT"
    )

    strict = _run_n4(prices, rebalance_bars=_N, adf_reject_threshold=-2.86)
    relaxed = _run_n4(prices, rebalance_bars=_N, adf_reject_threshold=-1.0)

    strict_entries = [s for s in _entries(strict.events.items) if s.symbol == "TARGET/USDT"]
    relaxed_side = _final_side(relaxed.events.items).get("TARGET/USDT")

    assert strict_entries == []  # gate closed -> abstain
    assert relaxed_side is not None  # gate opened by the ONLY changed knob -> trade
    # Once the gate opens N4's core scoring reproduces the incumbent's action.
    assert relaxed_side == incumbent_side


# --------------------------------------------------------------------------- #
# (b) market-neutrality: emitted legs hedge benchmark exposure (net beta ~0)
# --------------------------------------------------------------------------- #


def test_market_neutral_net_beta_is_zero_by_construction() -> None:
    """A single terminal rebalance over a liquid multi-asset book: the emitted
    asset legs plus the benchmark hedge leg sum to a net beta exposure of ~0."""
    benchmark = _benchmark_levels(_N)
    residuals = {
        "A0/USDT": _stationary_residual(_N, seed=101),
        "A1/USDT": _stationary_residual(
            _N, seed=202, tail=[0.030, 0.042, 0.054, 0.066, 0.078, 0.090]
        ),
        "A2/USDT": _stationary_residual(_N, seed=303),
        "A3/USDT": _stationary_residual(
            _N, seed=404, tail=[0.020, 0.030, 0.045, 0.060, 0.075, 0.090]
        ),
    }
    prices = {"BENCH/USDT": [math.exp(v) for v in benchmark]}
    for symbol, residual in residuals.items():
        prices[symbol] = [math.exp(benchmark[t] + residual[t]) for t in range(_N)]

    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(list(prices)),
        _Queue(),
        **_n4_kwargs(
            equity_symbols=",".join(residuals),
            benchmark_symbol="BENCH/USDT",
            rebalance_bars=_N,  # a single terminal rebalance -> every leg is fresh
        ),
    )
    _feed(strategy, prices, dict.fromkeys(prices, 1000.0))

    entries = _entries(strategy.events.items)
    asset_legs = [sig for sig in entries if sig.symbol != "BENCH/USDT"]
    hedge_legs = [sig for sig in entries if sig.symbol == "BENCH/USDT"]
    assert len(asset_legs) >= 2, "expected a non-trivial long/short book"
    assert len(hedge_legs) == 1, "expected exactly one benchmark hedge leg"

    for leg in asset_legs:
        assert leg.metadata.get("beta") is not None
        assert leg.metadata.get("beta_exposure") is not None
    # The hedge exactly offsets the book's signed beta exposure.
    net = math.fsum(float(sig.metadata.get("beta_exposure", 0.0)) for sig in entries)
    assert abs(net) < 1e-9, net
    assert hedge_legs[0].metadata.get("reason") == "benchmark_hedge"


# --------------------------------------------------------------------------- #
# (c) liquid-book filter behaviour
# --------------------------------------------------------------------------- #


def test_liquid_book_filter_excludes_thin_names() -> None:
    """A stationary, dislocated (would-trade) name is skipped when its trailing
    median dollar volume is below the floor, and trades once the floor is low."""
    prices = _prices_from_residual(_stationary_residual(_N))
    volumes = dict.fromkeys(prices, 1000.0)  # dollar volume ~ close * 1000 ~ 1e5

    # Floor above the name's ~1e5 dollar volume -> excluded (no entry).
    thin = _run_n4(prices, volumes, min_dollar_volume=1e6, dollar_volume_window=20)
    assert [s for s in _entries(thin.events.items) if s.symbol == "TARGET/USDT"] == []

    # Floor comfortably below -> the same name trades.
    liquid = _run_n4(prices, volumes, min_dollar_volume=1e4, dollar_volume_window=20)
    assert _final_side(liquid.events.items).get("TARGET/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# (d) determinism: two identical runs -> identical signal streams
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    prices = _prices_from_residual(_stationary_residual(_N))

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _run_n4(prices)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# (e) get_state / set_state roundtrip + adversarial set_state
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    prices = _prices_from_residual(_stationary_residual(_N))
    strategy = _run_n4(prices)
    snapshot = strategy.get_state()

    restored = StationarityGatedResidualReversionStrategy(
        _Bars(list(prices)), _Queue(), **_n4_kwargs()
    )
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in prices:
        r = restored._state[symbol]
        o = strategy._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert list(r.volumes) == list(o.volumes)
        assert r.mode == o.mode
        assert r.bars_held == o.bars_held
        assert r.last_time_key == o.last_time_key
    assert restored._exposure == strategy._exposure
    assert restored._tick == strategy._tick
    assert restored._last_eval_time_key == strategy._last_eval_time_key


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["BENCH/USDT", "TARGET/USDT", "A0/USDT"]
    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(symbols), _Queue(), **_n4_kwargs(equity_symbols="TARGET/USDT,A0/USDT")
    )

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"TARGET/USDT": "not a dict either"}})
    strategy.set_state({"symbol_state": {"TARGET/USDT": {"closes": 12345}}})
    strategy.set_state({"exposure": "not-a-dict"})
    strategy.set_state({"exposure": {"UNKNOWN/USDT": 1.0, "TARGET/USDT": "bad"}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "exposure": {"TARGET/USDT": float("nan"), "A0/USDT": float("inf")},
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "volumes": {"unexpected": "type"},
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
    assert all(math.isfinite(v) for v in strategy._exposure.values())

    # Still functions afterward on the same symbols.
    benchmark = _benchmark_levels(_N)
    residual = _stationary_residual(_N)
    prices = {
        "BENCH/USDT": [math.exp(v) for v in benchmark],
        "TARGET/USDT": [math.exp(benchmark[t] + residual[t]) for t in range(_N)],
        "A0/USDT": [
            math.exp(benchmark[t] + _stationary_residual(_N, seed=55)[t]) for t in range(_N)
        ],
    }
    _feed(strategy, prices, dict.fromkeys(prices, 1000.0))  # must not raise


# --------------------------------------------------------------------------- #
# (f) never-raise on degenerate input
# --------------------------------------------------------------------------- #


def test_degenerate_closes_never_raise() -> None:
    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(["BENCH/USDT", "Z/USDT"]),
        _Queue(),
        **_n4_kwargs(equity_symbols="Z/USDT"),
    )
    for idx, close in enumerate([0.0, -5.0, float("nan"), float("inf")]):
        strategy.calculate_signals(_market_event("Z/USDT", idx, close, 1000.0))
        strategy.calculate_signals(_market_event("BENCH/USDT", idx, close, 1000.0))
    assert _non_exit(strategy.events.items) == []


def test_empty_and_heartbeat_events_never_raise() -> None:
    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(["BENCH/USDT", "Z/USDT"]), _Queue(), **_n4_kwargs(equity_symbols="Z/USDT")
    )
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", symbol="Z/USDT", bars_1s={}, time="t0")
    )
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert strategy.events.items == []


def test_short_history_abstains_never_raises() -> None:
    # Far fewer bars than adf_window -> no entries, no raise.
    benchmark = _benchmark_levels(8)
    residual = _stationary_residual(8, tail=[-0.03, -0.05, -0.07, -0.09])
    prices = {
        "BENCH/USDT": [math.exp(v) for v in benchmark],
        "TARGET/USDT": [math.exp(benchmark[t] + residual[t]) for t in range(8)],
    }
    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(list(prices)), _Queue(), **_n4_kwargs()
    )
    _feed(strategy, prices, dict.fromkeys(prices, 1000.0))
    assert _entries(strategy.events.items) == []


def test_missing_volume_never_raises() -> None:
    prices = _prices_from_residual(_stationary_residual(_N))
    volumes: dict[str, float | None] = dict.fromkeys(prices, 1000.0)
    volumes["TARGET/USDT"] = None  # every TARGET bar arrives with volume=None
    strategy = _run_n4(prices, volumes)  # must not raise
    assert all(v == 0.0 for v in strategy._state["TARGET/USDT"].volumes)


# --------------------------------------------------------------------------- #
# (g) schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (pm) POSTMORTEM of the measured 4h cell
# `stationarity_gated_residual_reversion_4h_strict_adf` (research_note 2026-07-09:
# OOS Sharpe -18.41, return -39.26%).  A Sharpe that negative implicates one of
# two mechanical faults -- a SIGN INVERSION or an every-bar TURNOVER churn -- so
# these tests pin BOTH as absent.  The residual loss is therefore a DIRECTIONAL
# (wrong-signed / absent) edge on the crypto-perp universe (the pre-registered
# EXPECTED-NULL), not a code defect.  They also lock the sign so a future "fix"
# that silently flips the reversion direction (curve-fitting one OOS cell) fails.
# --------------------------------------------------------------------------- #


def test_pm_sign_is_reversion_buy_the_loser_sell_the_winner() -> None:
    """THEORY: fade the beta-hedged residual -- LONG an under-performed
    (dislocated-DOWN) residual, SHORT an over-performed (dislocated-UP) one.

    Both fixtures are stationary (ADF gate open) and differ ONLY in the sign of
    the terminal dislocation, so the emitted side must track the THEORY
    direction: this is the culprit-1 (sign-orientation) lock.
    """
    down = _prices_from_residual(_stationary_residual(_N))  # shared _TAIL descends
    up = _prices_from_residual(
        _stationary_residual(_N, tail=[0.030, 0.042, 0.054, 0.066, 0.078, 0.090])
    )
    assert _final_side(_run_n4(down, rebalance_bars=_N).events.items).get("TARGET/USDT") == "LONG"
    assert _final_side(_run_n4(up, rebalance_bars=_N).events.items).get("TARGET/USDT") == "SHORT"


def test_pm_decision_clock_rebalances_weekly_not_every_bar() -> None:
    """The 4h cell scores on a ``rebalance_bars`` clock (42 bars ~= weekly), NOT
    every bar: ``_score_rows`` fires ``floor(n_bars / rebalance_bars)`` times.

    This is the culprit-2 (turnover-churn) lock: at a weekly scoring cadence the
    round-trip count -- and hence the 20bps cost bleed -- is far too small to
    explain the measured -39% OOS return, so the loss is directional.
    """
    n_bars, rebalance_bars = 210, 42
    benchmark = _benchmark_levels(n_bars)
    prices = {
        "BENCH/USDT": [math.exp(v) for v in benchmark],
        "TARGET/USDT": [math.exp(benchmark[t] + 0.01 * math.sin(t / 3.0)) for t in range(n_bars)],
    }
    strategy = StationarityGatedResidualReversionStrategy(
        _Bars(list(prices)),
        _Queue(),
        **_n4_kwargs(rebalance_bars=rebalance_bars, min_hold_bars=rebalance_bars),
    )
    calls = {"n": 0}
    original = strategy._score_rows

    def _counting() -> Any:
        calls["n"] += 1
        return original()

    strategy._score_rows = _counting  # type: ignore[method-assign]
    _feed(strategy, prices, dict.fromkeys(prices, 1000.0))
    assert strategy._tick == n_bars
    assert calls["n"] == n_bars // rebalance_bars  # 5 weekly scorings, not 210


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = StationarityGatedResidualReversionStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "equity_symbols",
        "benchmark_symbol",
        "lookback_bars",
        "beta_window",
        "vol_window",
        "adf_window",
        "adf_lags",
        "adf_reject_threshold",
        "entry_score",
        "rebalance_bars",
        "min_hold_bars",
        "max_hold_bars",
        "max_longs",
        "max_shorts",
        "allow_short",
        "hedge_benchmark",
        "min_dollar_volume",
        "dollar_volume_window",
        "stop_loss_pct",
        "target_gross_exposure",
        "max_order_value",
        "min_price",
    ):
        assert required in schema
    for cap in ("equity_symbols", "benchmark_symbol", "target_gross_exposure", "max_order_value"):
        assert schema[cap].tunable is False
