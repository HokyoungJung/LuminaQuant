"""Deterministic build-gate tests for CrossSectionalDownsideBetaAsymmetryStrategy.

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions -- those land with the integration wave).

The load-bearing gate is a PERMUTATION-INVARIANCE kill shot: on a closed-form
panel where CRASHB (co-crashes, does not co-rally) and RALLYB (mirror) share an
identical unconditional beta and identical idiosyncratic volatility, this sleeve
emits OPPOSITE-signed targets while the real ``BettingAgainstBetaStrategy`` and
``IdiosyncraticVolatilityStrategy`` cannot separate them, and an IDIOX symbol
with an extreme own downside-semivariance / idio-vol is traded by the univariate
incumbents while this sleeve abstains -- isolating benchmark-sign-conditioned
CO-movement as the sole differentiator vs the occupied beta-level and univariate
moment axes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.comoment import conditional_semibeta
from lumina_quant.indicators.rolling_stats import rolling_beta
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    IdiosyncraticVolatilityStrategy,
    LotterySkewnessStrategy,
)
from lumina_quant.strategies.downside_beta_asymmetry_alpha_sleeves import (
    CrossSectionalDownsideBetaAsymmetryStrategy,
    _log_returns,
)
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import BettingAgainstBetaStrategy
from lumina_quant.strategies.kalman_semivar_alpha_sleeves import _signed_semivariance_asymmetry
from lumina_quant.tuning import HyperParam

_BENCH = "BTC/USDT"


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# harness (MARKET_WINDOW feed -> whole cross-section fresh at each rebalance)
# --------------------------------------------------------------------------- #


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
        type="MARKET_WINDOW", time=f"2026-01-01T00:00:00Z#{idx}", bars_1s=bars_1s
    )


def _feed(strategy: Any, symbols: list[str], closes_by_symbol: dict[str, list[float]]) -> None:
    n = len(next(iter(closes_by_symbol.values())))
    for idx in range(n):
        snapshot = {sym: closes_by_symbol[sym][idx] for sym in symbols if sym in closes_by_symbol}
        strategy.calculate_signals(_window_event(symbols, snapshot, idx))


def _closes_from_returns(returns: list[float], start: float = 100.0) -> list[float]:
    closes = [start]
    for ret in returns:
        closes.append(closes[-1] * math.exp(ret))
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
# fixture: benchmark cycle with within-side variance + the permutation panel
# --------------------------------------------------------------------------- #

_CYCLE = [-0.01, 0.03, -0.03, 0.01]  # 2 down bars, 2 up bars; within-side variance on each side


def _panel_returns(reps: int = 24) -> tuple[list[float], dict[str, list[float]]]:
    r_m = _CYCLE * reps
    gen1 = _lcg_stream(11)
    gen2 = _lcg_stream(29)
    symbols_returns = {
        # co-crasher: 2x on benchmark-DOWN bars, flat on UP -> beta_minus=2, beta_plus=0
        "CRASHB": [2.0 * x if x < 0 else 0.0 for x in r_m],
        # mirror co-rallier -> beta_minus=0, beta_plus=2
        "RALLYB": [0.0 if x < 0 else 2.0 * x for x in r_m],
        # symmetric high / low unconditional beta -> DBS 0
        "HIBETA": [1.6 * x for x in r_m],
        "LOBETA": [0.4 * x for x in r_m],
        # zero benchmark loading, large negative idiosyncratic spikes on UP bars ->
        # DBS 0 but extreme idio-vol / negative semivariance asymmetry
        "IDIOX": [0.0 if x < 0 else -0.10 for x in r_m],
        # ~unit beta fillers with tiny idiosyncratic noise (distinct idio-vol, DBS ~0)
        "MID1": [1.0 * x + (next(gen1) - 0.5) * 0.002 for x in r_m],
        "MID2": [1.0 * x + (next(gen2) - 0.5) * 0.002 for x in r_m],
    }
    return r_m, symbols_returns


def _panel_closes(reps: int = 24) -> tuple[list[str], dict[str, list[float]], list[float]]:
    r_m, symbols_returns = _panel_returns(reps)
    closes_by_symbol = {_BENCH: _closes_from_returns(r_m)}
    for symbol, rets in symbols_returns.items():
        closes_by_symbol[symbol] = _closes_from_returns(rets)
    symbols = [_BENCH, *symbols_returns.keys()]
    return symbols, closes_by_symbol, r_m


def _sleeve(symbols: list[str], **overrides: Any) -> CrossSectionalDownsideBetaAsymmetryStrategy:
    params: dict[str, Any] = dict(
        benchmark_symbol=_BENCH,
        benchmark_mode="btc",
        window_bars=200,
        min_side_obs=8,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=1,
        hysteresis_buffer_ranks=0,
        vol_window=20,
        min_symbols=5,
        allow_short=True,
        target_gross_exposure=1.0,
        min_price=0.01,
    )
    params.update(overrides)
    return CrossSectionalDownsideBetaAsymmetryStrategy(_Bars(symbols), _Queue(), **params)


# --------------------------------------------------------------------------- #
# Stage-1 premises: the conditioned betas match the closed-form design
# --------------------------------------------------------------------------- #


def test_stage1_conditioned_betas_match_closed_form() -> None:
    _symbols, closes, r_m = _panel_closes()
    bench_ret = _log_returns(closes[_BENCH])
    expected = {
        "CRASHB": (2.0, 0.0),
        "RALLYB": (0.0, 2.0),
        "HIBETA": (1.6, 1.6),
        "LOBETA": (0.4, 0.4),
        "IDIOX": (0.0, 0.0),
    }
    for symbol, (exp_minus, exp_plus) in expected.items():
        beta_minus, beta_plus = conditional_semibeta(
            _log_returns(closes[symbol]), bench_ret, threshold=0.0, min_side_obs=8
        )
        assert beta_minus is not None and beta_plus is not None, symbol
        assert abs(beta_minus - exp_minus) < 1e-9, (symbol, beta_minus)
        assert abs(beta_plus - exp_plus) < 1e-9, (symbol, beta_plus)
    _ = r_m


# --------------------------------------------------------------------------- #
# Leg 1: candidate LONGs the co-crasher, SHORTs the co-rallier
# --------------------------------------------------------------------------- #


def test_candidate_longs_cocrasher_shorts_corallier() -> None:
    symbols, closes, _r_m = _panel_closes()
    strategy = _sleeve(symbols)
    _feed(strategy, symbols, closes)
    side = _final_side(strategy.events.items)
    assert side.get("CRASHB") == "LONG", side
    assert side.get("RALLYB") == "SHORT", side
    # IDIOX (extreme univariate moments, DBS~0) is left flat by the sleeve.
    assert "IDIOX" not in side, side


# --------------------------------------------------------------------------- #
# Leg 2: BAB kill shot -- identical unconditional beta, cannot separate the pair
# --------------------------------------------------------------------------- #


def test_bab_cannot_separate_the_pair_but_candidate_can() -> None:
    symbols, closes, _r_m = _panel_closes()
    bench_ret = _log_returns(closes[_BENCH])

    # Direct primitive: the ranking statistic BAB uses (full-sample beta) is
    # IDENTICAL for the pair within 1e-9 -- BAB is structurally blind to them.
    beta_crashb = rolling_beta(_log_returns(closes["CRASHB"]), bench_ret)
    beta_rallyb = rolling_beta(_log_returns(closes["RALLYB"]), bench_ret)
    assert beta_crashb is not None and beta_rallyb is not None
    assert abs(beta_crashb - beta_rallyb) < 1e-9

    bab = BettingAgainstBetaStrategy(
        _Bars(symbols),
        _Queue(),
        benchmark_symbol=_BENCH,
        beta_window=40,
        rebalance_bars=1,
        quintile_pct=0.20,
        min_symbols=4,
        allow_short=True,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(bab, symbols, closes)
    bab_side = _final_side(bab.events.items)
    # Incumbent is LIVE (level-driven): it shorts the highest-beta name.
    assert bab_side.get("HIBETA") == "SHORT", bab_side
    assert any(s == "LONG" for s in bab_side.values()), bab_side
    # ... but assigns the pair identical (mid-rank) treatment -> neither traded,
    # and in particular NOT opposite sides.
    assert bab_side.get("CRASHB") == bab_side.get("RALLYB"), bab_side

    sleeve = _sleeve(symbols)
    _feed(sleeve, symbols, closes)
    sleeve_side = _final_side(sleeve.events.items)
    assert sleeve_side.get("CRASHB") == "LONG"
    assert sleeve_side.get("RALLYB") == "SHORT"


# --------------------------------------------------------------------------- #
# Leg 3: non-redundancy vs the univariate idio-vol / semivariance axes
# --------------------------------------------------------------------------- #


def test_univariate_incumbents_act_on_idiox_while_candidate_abstains() -> None:
    symbols, closes, _r_m = _panel_closes()

    # The realized-semivariance rider's own-return asymmetry fires hard on IDIOX.
    idiox_asym = _signed_semivariance_asymmetry(_log_returns(closes["IDIOX"]))
    assert idiox_asym is not None and idiox_asym < -0.5

    idio = IdiosyncraticVolatilityStrategy(
        _Bars(symbols),
        _Queue(),
        benchmark_symbol=_BENCH,
        beta_window=40,
        vol_window=20,
        rebalance_bars=1,
        quantile_pct=0.25,
        min_symbols=4,
        allow_short=True,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(idio, symbols, closes)
    idio_side = _final_side(idio.events.items)
    # IdioVol SHORTS the extreme idio-vol name (IDIOX) -> live and univariate.
    assert idio_side.get("IDIOX") == "SHORT", idio_side
    # ... but the co-crash/co-rally pair carries identical idiosyncratic vol, so
    # IdioVol cannot separate them (never opposite sides).
    assert idio_side.get("CRASHB") == idio_side.get("RALLYB"), idio_side

    sleeve = _sleeve(symbols)
    _feed(sleeve, symbols, closes)
    sleeve_side = _final_side(sleeve.events.items)
    # The sleeve ABSTAINS on IDIOX (benchmark-sign-conditioned DBS ~0) while the
    # univariate incumbents trade it -- the sole differentiator is co-movement.
    assert "IDIOX" not in sleeve_side, sleeve_side
    assert sleeve_side.get("CRASHB") == "LONG"
    assert sleeve_side.get("RALLYB") == "SHORT"


# --------------------------------------------------------------------------- #
# Leg 4: all-up-market -> down-side obs below the floor -> abstain, never-raise
# --------------------------------------------------------------------------- #


def test_all_up_market_abstains_without_raising() -> None:
    n = 120
    up_returns = [0.01 + 0.005 * (idx % 3) for idx in range(n)]  # strictly positive
    symbols = [_BENCH, "S1", "S2", "S3", "S4", "S5"]
    closes_by_symbol = {_BENCH: _closes_from_returns(up_returns)}
    gens = {sym: _lcg_stream(100 + i) for i, sym in enumerate(symbols[1:])}
    for sym in symbols[1:]:
        gen = gens[sym]
        closes_by_symbol[sym] = _closes_from_returns(
            [ret * (0.8 + 0.4 * next(gen)) for ret in up_returns]
        )
    strategy = _sleeve(symbols, min_symbols=4)
    _feed(strategy, symbols, closes_by_symbol)  # must not raise
    assert _final_side(strategy.events.items) == {}


# --------------------------------------------------------------------------- #
# Leg 5: hard min-hold suppresses a rank flip inside the hold window
# --------------------------------------------------------------------------- #


def _swapped_phase_closes(reps: int = 24) -> tuple[list[str], dict[str, list[float]]]:
    """Phase-1 standard panel then Phase-2 with CRASHB<->RALLYB roles swapped."""
    r_m1, rets1 = _panel_returns(reps)
    r_m2 = _CYCLE * reps
    gen1 = _lcg_stream(11)
    gen2 = _lcg_stream(29)
    rets2 = {
        "CRASHB": [0.0 if x < 0 else 2.0 * x for x in r_m2],  # now the co-rallier
        "RALLYB": [2.0 * x if x < 0 else 0.0 for x in r_m2],  # now the co-crasher
        "HIBETA": [1.6 * x for x in r_m2],
        "LOBETA": [0.4 * x for x in r_m2],
        "IDIOX": [0.0 if x < 0 else -0.10 for x in r_m2],
        "MID1": [1.0 * x + (next(gen1) - 0.5) * 0.002 for x in r_m2],
        "MID2": [1.0 * x + (next(gen2) - 0.5) * 0.002 for x in r_m2],
    }
    all_returns = {_BENCH: r_m1 + r_m2}
    for symbol in rets1:
        all_returns[symbol] = rets1[symbol] + rets2[symbol]
    closes = {symbol: _closes_from_returns(rets) for symbol, rets in all_returns.items()}
    symbols = [_BENCH, *rets1.keys()]
    return symbols, closes


def test_min_hold_suppresses_rank_flip() -> None:
    symbols, closes = _swapped_phase_closes()

    # With a huge min-hold, the flip in Phase-2 CANNOT rotate the book: CRASHB is
    # frozen LONG for the whole run.
    held = _sleeve(symbols, window_bars=30, min_side_obs=8, min_hold_bars=100_000)
    _feed(held, symbols, closes)
    assert _final_side(held.events.items).get("CRASHB") == "LONG"

    # With min-hold released, the SAME feed rotates CRASHB out of the long book
    # (it becomes the co-rallier) -- proving min-hold, not the ranking, held it.
    free = _sleeve(symbols, window_bars=30, min_side_obs=8, min_hold_bars=1)
    _feed(free, symbols, closes)
    assert _final_side(free.events.items).get("CRASHB") != "LONG"


# --------------------------------------------------------------------------- #
# Leg 6: determinism -- two identical runs produce identical signal streams
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


# --------------------------------------------------------------------------- #
# lane hygiene: state roundtrip / adversarial set_state / degenerate never-raise
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    symbols, closes, _r_m = _panel_closes()
    strategy = _sleeve(symbols)
    _feed(strategy, symbols, closes)
    snapshot = strategy.get_state()

    restored = _sleeve(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    symbols = [_BENCH, "A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strategy = _sleeve(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("nope")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "not a dict"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 123}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "bad",
            "symbol_state": {
                sym: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
                    "volumes": {"bad": "type"},
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 123,
                }
                for sym in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_degenerate_input_never_raises() -> None:
    symbols = [_BENCH, "Z1", "Z2", "Z3", "Z4", "Z5"]
    strategy = _sleeve(symbols)
    # empty window, missing symbols, zero / negative / non-finite closes
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    for idx, bad in enumerate((0.0, -5.0, float("nan"), float("inf"))):
        strategy.calculate_signals(_window_event(symbols, dict.fromkeys(symbols, bad), idx))
    assert [s for s in strategy.events.items if str(s.signal_type).upper() != "EXIT"] == []


def test_self_skip_below_min_symbols() -> None:
    symbols = [_BENCH, "A/USDT", "B/USDT"]
    _r_m, _rets = _panel_returns()
    r_m = _CYCLE * 24
    closes = {sym: _closes_from_returns([1.0 * x for x in r_m]) for sym in symbols}
    strategy = _sleeve(symbols, min_symbols=5)
    _feed(strategy, symbols, closes)
    assert [s for s in strategy.events.items if str(s.signal_type).upper() != "EXIT"] == []


# --------------------------------------------------------------------------- #
# schema sanity
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalDownsideBetaAsymmetryStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in ("window_bars", "min_side_obs", "min_hold_bars", "hysteresis_buffer_ranks"):
        assert required in schema

    lottery = LotterySkewnessStrategy  # imported for parity with the incumbent set
    assert lottery is not None
