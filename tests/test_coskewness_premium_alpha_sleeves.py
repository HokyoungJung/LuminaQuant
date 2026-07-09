"""Deterministic build-gate tests for SystematicCoskewnessPremiumStrategy.

Direct class import only (no ``@register`` on this lane).

The gate builds a closed-form panel (period-4 benchmark alternating calm/turbulent
magnitudes) with:

- ``X`` = ``-K*(r_m**2 - mean(r_m**2))`` -> beta 0, own-skew 0, strongly NEGATIVE
  standardized coskewness (the sleeve's long);
- ``Y = -X`` -> beta 0, own-skew 0, strongly POSITIVE coskewness, and (by mirror)
  IDENTICAL residual volatility to ``X`` (the sleeve's short);
- ``Z`` -> a spiky own-skew / MAX lottery name with coskewness ~0;
- ``A``/``D`` -> linear beta loadings (-0.5 / 1.5) that own the BAB extremes;
- ``FLAT`` -> a near-flat low-lottery / low-idio-vol name.

It then proves the beta-residualized THIRD co-moment is behaviorally spanned by
NONE of ``BettingAgainstBetaStrategy`` (first co-moment), ``LotterySkewnessStrategy``
(own third moment), or ``IdiosyncraticVolatilityStrategy`` (second residual
moment), each run as a real class.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.comoment import standardized_coskewness
from lumina_quant.indicators.rolling_stats import rolling_beta, sample_std
from lumina_quant.strategies.coskewness_premium_alpha_sleeves import (
    SystematicCoskewnessPremiumStrategy,
    _simple_returns,
)
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    IdiosyncraticVolatilityStrategy,
    LotterySkewnessStrategy,
)
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import BettingAgainstBetaStrategy
from lumina_quant.tuning import HyperParam

_BENCH = "BTC/USDT"
_T = 160
_WINDOW = 120


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


def _window_event(symbols: list[str], closes: dict[str, float], idx: int) -> SimpleNamespace:
    bars_1s = {
        symbol: [
            {
                "open": closes[symbol],
                "high": closes[symbol],
                "low": closes[symbol],
                "close": closes[symbol],
                "volume": 1000.0,
            }
        ]
        for symbol in symbols
        if symbol in closes
    }
    return SimpleNamespace(
        type="MARKET_WINDOW", time=f"2026-02-01T00:00:00Z#{idx}", bars_1s=bars_1s
    )


def _feed(strategy: Any, symbols: list[str], closes_by_symbol: dict[str, list[float]]) -> None:
    n = len(next(iter(closes_by_symbol.values())))
    for idx in range(n):
        snapshot = {sym: closes_by_symbol[sym][idx] for sym in symbols if sym in closes_by_symbol}
        strategy.calculate_signals(_window_event(symbols, snapshot, idx))


def _closes_from_simple_returns(returns: list[float], start: float = 100.0) -> list[float]:
    closes = [start]
    for ret in returns:
        closes.append(closes[-1] * (1.0 + ret))
    return closes


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
# panel construction
# --------------------------------------------------------------------------- #

_PERIOD = [0.005, -0.005, 0.04, -0.04]  # calm +/-, turbulent +/- -> mean 0, skew 0


def _panel_returns() -> tuple[list[float], dict[str, list[float]]]:
    r_m = [_PERIOD[t % 4] for t in range(_T)]
    c = sum(x * x for x in r_m) / len(r_m)
    k = 8.0
    gen_a = _lcg_stream(201)
    gen_d = _lcg_stream(203)
    gen_f = _lcg_stream(205)
    x = [-k * (r_m[t] ** 2 - c) for t in range(_T)]
    returns = {
        "X": x,
        "Y": [-value for value in x],
        "Z": [(-0.001) + (0.06 if (t % 20 in (7, 17)) else 0.0) for t in range(_T)],
        "A": [-0.5 * r_m[t] + (next(gen_a) - 0.5) * 0.0006 for t in range(_T)],
        "D": [1.5 * r_m[t] + (next(gen_d) - 0.5) * 0.0006 for t in range(_T)],
        "FLAT": [
            (0.0004 if t % 2 == 0 else -0.0004) + (next(gen_f) - 0.5) * 2e-5 for t in range(_T)
        ],
    }
    return r_m, returns


def _panel_closes() -> tuple[list[str], dict[str, list[float]], list[float]]:
    r_m, returns = _panel_returns()
    closes = {_BENCH: _closes_from_simple_returns(r_m)}
    for symbol, rets in returns.items():
        closes[symbol] = _closes_from_simple_returns(rets)
    symbols = [_BENCH, *returns.keys()]
    return symbols, closes, r_m


def _sleeve(symbols: list[str], **overrides: Any) -> SystematicCoskewnessPremiumStrategy:
    params: dict[str, Any] = dict(
        benchmark_symbol=_BENCH,
        coskew_window=_WINDOW,
        beta_residualize=True,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=1,
        hysteresis_band=0.10,
        vol_window=20,
        min_symbols=4,
        allow_short=True,
        target_gross_exposure=1.0,
        min_price=0.01,
    )
    params.update(overrides)
    return SystematicCoskewnessPremiumStrategy(_Bars(symbols), _Queue(), **params)


def _skew(values: list[float]) -> float:
    mean_value = sum(values) / len(values)
    std = sample_std(values)
    assert std is not None
    return sum((v - mean_value) ** 3 for v in values) / len(values) / (std**3)


# --------------------------------------------------------------------------- #
# Stage-1 premises
# --------------------------------------------------------------------------- #


def test_stage1_premises_coskew_beta_skew() -> None:
    _symbols, closes, _r_m = _panel_closes()
    bench = _simple_returns(closes[_BENCH])[-_WINDOW:]
    ck = {
        sym: standardized_coskewness(_simple_returns(closes[sym])[-_WINDOW:], bench)
        for sym in ("X", "Y", "Z", "A", "D", "FLAT")
    }
    assert ck["X"] is not None and ck["X"] < -0.3
    assert ck["Y"] is not None and ck["Y"] > 0.3
    assert ck["Z"] is not None and abs(ck["Z"]) < 0.1
    for sym in ("A", "D", "FLAT"):
        assert ck[sym] is not None and abs(ck[sym]) < 0.3
    for sym in ("X", "Y"):
        rets = _simple_returns(closes[sym])[-_WINDOW:]
        beta = rolling_beta(rets, bench)
        assert beta is not None and abs(beta) < 0.05, (sym, beta)
        assert abs(_skew(rets)) < 0.2, sym
    # X and Y are mirror images -> identical own-return distribution shape.
    assert ck["X"] is not None and ck["Y"] is not None
    assert abs(ck["X"] + ck["Y"]) < 1e-9


# --------------------------------------------------------------------------- #
# (a) candidate acts: LONG X, SHORT Y, Z left out of both extremes
# --------------------------------------------------------------------------- #


def test_candidate_longs_negcoskew_shorts_poscoskew() -> None:
    symbols, closes, _r_m = _panel_closes()
    strategy = _sleeve(symbols)
    _feed(strategy, symbols, closes)
    side = _final_side(strategy.events.items)
    assert side.get("X") == "LONG", side
    assert side.get("Y") == "SHORT", side
    assert "Z" not in side, side


# --------------------------------------------------------------------------- #
# (b) vs BettingAgainstBeta: BAB owns the beta extremes, blind to X/Y
# --------------------------------------------------------------------------- #


def test_bab_owns_beta_extremes_candidate_owns_coskew() -> None:
    symbols, closes, _r_m = _panel_closes()
    bab = BettingAgainstBetaStrategy(
        _Bars(symbols),
        _Queue(),
        benchmark_symbol=_BENCH,
        beta_window=60,
        rebalance_bars=1,
        quintile_pct=0.20,
        min_symbols=4,
        allow_short=True,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(bab, symbols, closes)
    bab_side = _final_side(bab.events.items)
    assert bab_side.get("A") == "LONG", bab_side  # lowest beta (-0.5)
    assert bab_side.get("D") == "SHORT", bab_side  # highest beta (1.5)
    # X and Y sit at beta ~0 -> BAB cannot separate them (never opposite sides).
    assert bab_side.get("X") == bab_side.get("Y"), bab_side

    sleeve = _sleeve(symbols)
    _feed(sleeve, symbols, closes)
    sleeve_side = _final_side(sleeve.events.items)
    assert sleeve_side.get("X") == "LONG"
    assert sleeve_side.get("Y") == "SHORT"
    assert "A" not in sleeve_side and "D" not in sleeve_side, sleeve_side


# --------------------------------------------------------------------------- #
# (c) vs LotterySkewness: lottery owns Z, blind to the coskew pair
# --------------------------------------------------------------------------- #


def test_lottery_owns_z_candidate_ignores_it() -> None:
    symbols, closes, _r_m = _panel_closes()
    lottery = LotterySkewnessStrategy(
        _Bars(symbols),
        _Queue(),
        skew_window=60,
        max_window=20,
        rebalance_bars=1,
        quantile_pct=0.25,
        min_symbols=4,
        allow_short=True,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(lottery, symbols, closes)
    lottery_side = _final_side(lottery.events.items)
    # Lottery SHORTS the high own-skew / MAX name (Z) -> live and univariate.
    assert lottery_side.get("Z") == "SHORT", lottery_side
    # ... and leaves the coskew pair out of its extremes.
    assert "X" not in lottery_side and "Y" not in lottery_side, lottery_side

    sleeve = _sleeve(symbols)
    _feed(sleeve, symbols, closes)
    sleeve_side = _final_side(sleeve.events.items)
    # The candidate ignores Z (coskew ~0) while trading X/Y -- disjoint books.
    assert "Z" not in sleeve_side, sleeve_side
    assert sleeve_side.get("X") == "LONG"
    assert sleeve_side.get("Y") == "SHORT"


# --------------------------------------------------------------------------- #
# (d) vs IdiosyncraticVolatility: X/Y carry identical residual vol
# --------------------------------------------------------------------------- #


def test_idiovol_cannot_separate_mirror_pair() -> None:
    symbols, closes, _r_m = _panel_closes()
    bench = _simple_returns(closes[_BENCH])[-_WINDOW:]

    # Direct primitive: X and Y have IDENTICAL idiosyncratic (residual) vol.
    resid_vol = {}
    for sym in ("X", "Y"):
        rets = _simple_returns(closes[sym])[-_WINDOW:]
        beta = rolling_beta(rets, bench) or 0.0
        residuals = [r - beta * b for r, b in zip(rets, bench, strict=False)]
        resid_vol[sym] = sample_std(residuals)
    assert resid_vol["X"] is not None and resid_vol["Y"] is not None
    assert abs(resid_vol["X"] - resid_vol["Y"]) < 1e-9

    idio = IdiosyncraticVolatilityStrategy(
        _Bars(symbols),
        _Queue(),
        benchmark_symbol=_BENCH,
        beta_window=60,
        vol_window=30,
        rebalance_bars=1,
        quantile_pct=0.25,
        min_symbols=4,
        allow_short=True,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(idio, symbols, closes)
    idio_side = _final_side(idio.events.items)
    assert any(s == "SHORT" for s in idio_side.values()), idio_side  # live
    # Equal residual vol => IdioVol cannot put the pair on opposite sides.
    assert idio_side.get("X") == idio_side.get("Y"), idio_side

    sleeve = _sleeve(symbols)
    _feed(sleeve, symbols, closes)
    sleeve_side = _final_side(sleeve.events.items)
    assert sleeve_side.get("X") == "LONG"
    assert sleeve_side.get("Y") == "SHORT"


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    symbols, closes, _r_m = _panel_closes()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _sleeve(symbols)
        _feed(strategy, symbols, closes)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    assert _run() == _run()


def test_min_hold_suppresses_rank_flip() -> None:
    # Second half swaps X<->Y so the coskew ranking flips; a huge min-hold freezes
    # the book, min_hold=1 lets it rotate -- proving min-hold is the cause.
    r_m, returns = _panel_returns()
    swapped = dict(returns)
    swapped["X"], swapped["Y"] = returns["Y"], returns["X"]
    long_closes = {_BENCH: _closes_from_simple_returns(r_m + r_m)}
    for sym in returns:
        long_closes[sym] = _closes_from_simple_returns(returns[sym] + swapped[sym])
    full_symbols = [_BENCH, *returns.keys()]

    held = _sleeve(full_symbols, coskew_window=60, min_hold_bars=100_000)
    _feed(held, full_symbols, long_closes)
    assert _final_side(held.events.items).get("X") == "LONG"

    free = _sleeve(full_symbols, coskew_window=60, min_hold_bars=1)
    _feed(free, full_symbols, long_closes)
    assert _final_side(free.events.items).get("X") != "LONG"


def test_constant_benchmark_and_degenerate_never_raise() -> None:
    symbols = [_BENCH, "S1", "S2", "S3", "S4"]
    # constant benchmark closes -> zero variance -> no coskew, abstain
    closes = {sym: [100.0] * 130 for sym in symbols}
    strategy = _sleeve(symbols)
    _feed(strategy, symbols, closes)  # must not raise
    assert [s for s in strategy.events.items if str(s.signal_type).upper() != "EXIT"] == []

    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    for idx, bad in enumerate((0.0, -1.0, float("nan"), float("inf"))):
        strategy.calculate_signals(_window_event(symbols, dict.fromkeys(symbols, bad), idx))


def test_self_skip_below_min_symbols() -> None:
    symbols = [_BENCH, "A/USDT", "B/USDT"]
    r_m, _returns = _panel_returns()
    closes = {sym: _closes_from_simple_returns(r_m) for sym in symbols}
    strategy = _sleeve(symbols, min_symbols=4)
    _feed(strategy, symbols, closes)
    assert [s for s in strategy.events.items if str(s.signal_type).upper() != "EXIT"] == []


def test_state_roundtrip_lossless() -> None:
    symbols, closes, _r_m = _panel_closes()
    strategy = _sleeve(symbols)
    _feed(strategy, symbols, closes)
    snapshot = strategy.get_state()
    restored = _sleeve(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    symbols = [_BENCH, "A/USDT", "B/USDT", "C/USDT", "D/USDT"]
    strategy = _sleeve(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("nope")  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 123}}})
    strategy.set_state(
        {
            "tick": "bad",
            "symbol_state": {
                sym: {
                    "closes": ["x", float("nan"), 12.5, None],
                    "volumes": {"bad": "type"},
                    "mode": 999,
                    "bars_held": "oops",
                }
                for sym in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = SystematicCoskewnessPremiumStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in ("coskew_window", "beta_residualize", "min_hold_bars", "hysteresis_band"):
        assert required in schema
