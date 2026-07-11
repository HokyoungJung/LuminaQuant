"""Deterministic build-gate + hygiene tests for the price-delay premium sleeve.

Direct class import only (no ``@register`` on this lane, so no registry / tier /
candidate-wiring assertions here -- those land with the integration wave).

The BUILD GATE proves the UNCONDITIONAL Hou-Moskowitz D1 delay-CHARACTERISTIC
book is behaviorally distinct from the occupied CONDITIONAL lead-lag axis and
from the beta / momentum incumbents, by RUNNING the real incumbents on
hand-built fixtures:

- LEG 1 vs ``SlowCrossSectionalLeadLagStrategy``: when the last leader returns
  are strongly negative the lead-lag book SHORTS the delayed name (its forecast
  flips with the leader sign) while the candidate LONGS it (sign-free
  characteristic) -- opposite emitted action on the same symbol/bar.
- LEG 2: with the last leader returns ~0 the lead-lag book emits NOTHING (its
  spillover scores collapse below entry) while the candidate's delay ranks are
  unchanged -- trade-vs-abstain with the blocking statistic pinned.
- LEG 3 vs ``BettingAgainstBetaStrategy``: two names with EQUAL contemporaneous
  beta but different delay are TIED by BAB (neither in its extremes) while the
  candidate LONGS the high-delay one and SHORTS the low-delay one.
- LEG 4 vs ``ResidualEquityMomentumStrategy``: negating every return leaves the
  delay book bit-identical (R^2 is sign-free) while the momentum book FLIPS.
- LEG 5: a pure-idiosyncratic coin (no systematic loading) has an UNDEFINED
  delay and is never admitted to either tail.

All pseudo-randomness is a small seeded LCG (no ``random`` module), so every run
is bit-for-bit reproducible.
"""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.price_delay import price_delay_share
from lumina_quant.indicators.rolling_stats import rolling_beta
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    BettingAgainstBetaStrategy,
    ResidualEquityMomentumStrategy,
)
from lumina_quant.strategies.price_delay_premium_alpha_sleeves import (
    CrossSectionalPriceDelayPremiumStrategy,
)
from lumina_quant.strategies.slow_leadlag_xs_alpha_sleeves import (
    SlowCrossSectionalLeadLagStrategy,
)
from lumina_quant.tuning import HyperParam

_N = 130  # weekly-stamped bars
_W = 100  # delay window (weekly bars in this scaled fixture)
_LAGS = 5


# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg(seed: int):
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


def _week_iso(index: int) -> str:
    base = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    return (base + datetime.timedelta(weeks=index)).isoformat()


def _window_event(
    index: int,
    symbols: list[str],
    prices: dict[str, list[float]],
    volumes: dict[str, list[float]] | None = None,
) -> SimpleNamespace:
    stamp = _week_iso(index)
    bars = {
        symbol: [
            (
                stamp,
                prices[symbol][index],
                prices[symbol][index],
                prices[symbol][index],
                prices[symbol][index],
                (volumes[symbol][index] if volumes is not None else 1000.0),
            )
        ]
        for symbol in symbols
    }
    return SimpleNamespace(type="MARKET_WINDOW", time=stamp, bars_1s=bars)


def _feed(
    strategy: Any,
    symbols: list[str],
    prices: dict[str, list[float]],
    volumes: dict[str, list[float]] | None = None,
) -> None:
    for index in range(len(prices[symbols[0]])):
        strategy.calculate_signals(_window_event(index, symbols, prices, volumes))


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


def _cov(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    am = sum(a[-n:]) / n
    bm = sum(b[-n:]) / n
    return sum((a[-n:][i] - am) * (b[-n:][i] - bm) for i in range(n)) / n


# --------------------------------------------------------------------------- #
# fixture -- designed returns exp-compounded to prices so the strategy's log
# returns equal the designed returns exactly.
# --------------------------------------------------------------------------- #


def _build_panel(
    *, last2: float | None = None, negate: bool = False
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    bench_gen = _lcg(7)
    btc = [
        0.03 * (1 if next(bench_gen) > 0.5 else -1) + (next(bench_gen) - 0.5) * 2e-4
        for _ in range(_N)
    ]
    if last2 is not None:
        btc[-1] = last2
        btc[-2] = last2
    btc_lag1 = [0.0, *btc[:-1]]
    # Residualize the lag-1 factor against the contemporaneous benchmark over the
    # scoring window so DELAYED_HI and SYNC_LO carry an IDENTICAL contemporaneous
    # beta (the pair BAB cannot separate) by construction.
    beta_lw = _cov(btc_lag1[-_W:], btc[-_W:]) / _cov(btc[-_W:], btc[-_W:])
    lag_factor = [btc_lag1[i] - beta_lw * btc[i] for i in range(_N)]

    def _jit(seed: int) -> list[float]:
        gen = _lcg(seed)
        return [(next(gen) - 0.5) * 1e-6 for _ in range(_N)]

    returns = {
        "BTC/USDT": btc,
        "DELAYED/USDT": [0.8 * btc_lag1[i] + _jit(101)[i] for i in range(_N)],
        "SYNC/USDT": [0.8 * btc[i] + _jit(202)[i] for i in range(_N)],
        # No jitter on the tied pair: an EXACT contemporaneous-beta tie.
        "DELAYED_HI/USDT": [0.4 * btc[i] + 1.2 * lag_factor[i] for i in range(_N)],
        "SYNC_LO/USDT": [0.4 * btc[i] for i in range(_N)],
        "IDIO/USDT": [(next(_lcg(999 + i)) - 0.5) * 0.05 for i in range(_N)],
        # Fillers with distinct contemporaneous betas so BAB produces a live book.
        "FILL_A/USDT": [0.2 * btc[i] + _jit(11)[i] for i in range(_N)],
        "FILL_B/USDT": [0.65 * btc[i] + _jit(22)[i] for i in range(_N)],
        "FILL_C/USDT": [0.9 * btc[i] + _jit(33)[i] for i in range(_N)],
        "FILL_D/USDT": [1.1 * btc[i] + _jit(44)[i] for i in range(_N)],
    }
    if negate:
        returns = {symbol: [-value for value in series] for symbol, series in returns.items()}

    prices: dict[str, list[float]] = {}
    for symbol, series in returns.items():
        path = [100.0]
        for value in series:
            path.append(path[-1] * math.exp(value))
        prices[symbol] = path[1:]
    return list(returns), prices, returns


def _candidate(symbols: list[str], **overrides: Any) -> CrossSectionalPriceDelayPremiumStrategy:
    kwargs: dict[str, Any] = dict(
        benchmark_symbol="BTC/USDT",
        delay_window=_W,
        delay_lags=_LAGS,
        min_delay_obs=30,
        quantile_entry_pct=0.25,
        quantile_exit_pct=0.40,
        min_hold_decisions=1,
        cooldown_decisions=0,
        vol_window=20,
        min_symbols=5,
        allow_short=True,
    )
    kwargs.update(overrides)
    return CrossSectionalPriceDelayPremiumStrategy(_Bars(symbols), _Queue(), **kwargs)


def _slow_leadlag(symbols: list[str]) -> SlowCrossSectionalLeadLagStrategy:
    return SlowCrossSectionalLeadLagStrategy(
        _Bars(symbols),
        _Queue(),
        leader_symbols="BTC/USDT",
        max_lag=2,
        min_history=40,
        spillover_window=90,
        min_symbols=5,
        entry_z=0.5,
        exit_z=0.1,
        max_longs=3,
        max_shorts=3,
        min_hold_bars=1,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        target_vol=0.0,
    )


# --------------------------------------------------------------------------- #
# LEG 0 -- stage-1 premises (real primitives, before any strategy runs)
# --------------------------------------------------------------------------- #


def test_stage1_premises() -> None:
    _, _, returns = _build_panel()
    btc = returns["BTC/USDT"][1:]  # align to the strategy's log-return indexing

    d_delayed = price_delay_share(returns["DELAYED/USDT"][1:], btc, lags=_LAGS, min_obs=30)
    d_sync = price_delay_share(returns["SYNC/USDT"][1:], btc, lags=_LAGS, min_obs=30)
    d_hi = price_delay_share(returns["DELAYED_HI/USDT"][1:], btc, lags=_LAGS, min_obs=30)
    d_lo = price_delay_share(returns["SYNC_LO/USDT"][1:], btc, lags=_LAGS, min_obs=30)
    d_idio = price_delay_share(returns["IDIO/USDT"][1:], btc, lags=_LAGS, min_obs=30)

    assert d_delayed is not None and d_delayed > 0.8, d_delayed
    assert d_sync is not None and d_sync < 0.05, d_sync
    assert d_hi is not None and d_hi > 0.5, d_hi
    assert d_lo is not None and d_lo < 0.05, d_lo
    # Pure-idiosyncratic coin: delay is UNDEFINED (never admitted to either tail).
    assert d_idio is None, d_idio

    # The contemporaneous beta of the DELAYED_HI / SYNC_LO pair is identical -- the
    # pair BAB provably cannot separate -- while their delay differs materially.
    beta_hi = rolling_beta(returns["DELAYED_HI/USDT"][1:][-_W:], btc[-_W:])
    beta_lo = rolling_beta(returns["SYNC_LO/USDT"][1:][-_W:], btc[-_W:])
    assert beta_hi is not None and beta_lo is not None
    assert abs(beta_hi - beta_lo) < 1e-6, (beta_hi, beta_lo)
    assert d_hi - d_lo > 0.5, (d_hi, d_lo)

    # Closed-form: a pure lag-loader vs a pure contemporaneous-loader.
    pure_lag = price_delay_share(returns["DELAYED/USDT"][1:], btc, lags=_LAGS, min_obs=30)
    pure_contemp = price_delay_share(returns["SYNC/USDT"][1:], btc, lags=_LAGS, min_obs=30)
    assert pure_lag > 0.9 > 0.1 > pure_contemp


def test_stage1_candidate_book_and_scores() -> None:
    symbols, prices, _ = _build_panel()
    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)
    scores, _vols = candidate._delay_scores()

    assert scores["DELAYED/USDT"] > 0.8
    assert scores["SYNC/USDT"] < 0.05
    assert scores["DELAYED_HI/USDT"] > 0.5
    assert scores["SYNC_LO/USDT"] < 0.05
    assert "IDIO/USDT" not in scores  # pure-idio excluded
    assert "BTC/USDT" not in scores  # benchmark never ranked

    book = _final_side(candidate.events.items)
    # Both book sides non-empty (long high delay, short low delay).
    assert book.get("DELAYED/USDT") == "LONG", book
    assert any(mode == "SHORT" for mode in book.values()), book


# --------------------------------------------------------------------------- #
# LEG 1 -- opposite action vs SlowCrossSectionalLeadLag (conditional forecast)
# --------------------------------------------------------------------------- #


def test_gate_leg1_opposite_action_vs_slow_leadlag() -> None:
    symbols, prices, _ = _build_panel(last2=-0.04)

    incumbent = _slow_leadlag(symbols)
    _feed(incumbent, symbols, prices)
    inc_book = _final_side(incumbent.events.items)
    # Incumbent-LIVE: it SHORTS the delayed name (its lagged-leader forecast flips
    # negative when the last leader returns are strongly negative).
    assert inc_book.get("DELAYED/USDT") == "SHORT", inc_book
    scores = incumbent._spillover_scores()
    assert scores.get("DELAYED/USDT", 0.0) <= -incumbent.entry_z, scores

    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)
    cand_book = _final_side(candidate.events.items)
    # Candidate LONGS the same name on the same bars -- opposite emitted action.
    assert cand_book.get("DELAYED/USDT") == "LONG", cand_book


# --------------------------------------------------------------------------- #
# LEG 2 -- conditional-vs-unconditional (trade vs abstain)
# --------------------------------------------------------------------------- #


def test_gate_leg2_conditional_vs_unconditional() -> None:
    symbols, prices, _ = _build_panel(last2=1e-6)

    incumbent = _slow_leadlag(symbols)
    _feed(incumbent, symbols, prices)
    # WHY of silence: every spillover score collapses below the entry band when
    # the last leader returns are ~0, so the incumbent emits no book.
    scores = incumbent._spillover_scores()
    assert scores, "the incumbent must have warmed a scored panel"
    assert max(abs(value) for value in scores.values()) < incumbent.entry_z, scores
    assert _final_side(incumbent.events.items) == {}

    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)
    # The candidate's delay ranks are unchanged -> it still trades a full book.
    cand_book = _final_side(candidate.events.items)
    assert cand_book.get("DELAYED/USDT") == "LONG", cand_book
    assert any(mode == "SHORT" for mode in cand_book.values()), cand_book


# --------------------------------------------------------------------------- #
# LEG 3 -- tie-break vs BettingAgainstBeta (equal beta, different delay)
# --------------------------------------------------------------------------- #


def test_gate_leg3_tie_break_vs_betting_against_beta() -> None:
    symbols, prices, returns = _build_panel()

    bab = BettingAgainstBetaStrategy(
        _Bars(symbols),
        _Queue(),
        benchmark_symbol="BTC/USDT",
        beta_window=_W,
        rebalance_bars=1,
        quintile_pct=0.20,
        min_symbols=5,
    )
    _feed(bab, symbols, prices)
    bab_book = _final_side(bab.events.items)
    # Incumbent-LIVE: fillers carry distinct betas, so BAB emits a non-empty book.
    assert bab_book, bab_book
    # BAB CANNOT separate the equal-beta pair: their betas match within 1e-6 and
    # neither name lands in BAB's extreme quintiles (same treatment).
    btc = returns["BTC/USDT"][1:]
    beta_hi = rolling_beta(returns["DELAYED_HI/USDT"][1:][-_W:], btc[-_W:])
    beta_lo = rolling_beta(returns["SYNC_LO/USDT"][1:][-_W:], btc[-_W:])
    assert abs(beta_hi - beta_lo) < 1e-6, (beta_hi, beta_lo)
    assert "DELAYED_HI/USDT" not in bab_book, bab_book
    assert "SYNC_LO/USDT" not in bab_book, bab_book

    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)
    cand_book = _final_side(candidate.events.items)
    # The candidate DOES split the pair: LONG the high-delay one, and the
    # low-delay one is NOT in its long book (it is shorted / excluded).
    assert cand_book.get("DELAYED_HI/USDT") == "LONG", cand_book
    assert cand_book.get("SYNC_LO/USDT") != "LONG", cand_book


# --------------------------------------------------------------------------- #
# LEG 4 -- sign invariance vs momentum (R^2 is sign-free)
# --------------------------------------------------------------------------- #


def test_gate_leg4_sign_invariance_vs_momentum() -> None:
    symbols, prices, _ = _build_panel()
    symbols_neg, prices_neg, _ = _build_panel(negate=True)

    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)
    base_book = _final_side(candidate.events.items)

    candidate_neg = _candidate(symbols_neg)
    _feed(candidate_neg, symbols_neg, prices_neg)
    neg_book = _final_side(candidate_neg.events.items)
    # Negating every return leaves the delay book bit-identical (delay is a
    # sign-free R^2 ratio -- zero first-moment content).
    assert base_book == neg_book, (base_book, neg_book)

    rem = ResidualEquityMomentumStrategy(
        _Bars(symbols),
        _Queue(),
        benchmark_symbol="BTC/USDT",
        lookback_bars=40,
        skip_bars=2,
        beta_window=40,
        rebalance_bars=1,
        quintile_pct=0.25,
        min_symbols=5,
    )
    _feed(rem, symbols, prices)
    rem_book = _final_side(rem.events.items)
    rem_neg = ResidualEquityMomentumStrategy(
        _Bars(symbols_neg),
        _Queue(),
        benchmark_symbol="BTC/USDT",
        lookback_bars=40,
        skip_bars=2,
        beta_window=40,
        rebalance_bars=1,
        quintile_pct=0.25,
        min_symbols=5,
    )
    _feed(rem_neg, symbols_neg, prices_neg)
    rem_neg_book = _final_side(rem_neg.events.items)
    # Momentum FLIPS sides on the mirrored panel (non-vacuous: a real book).
    assert rem_book, rem_book
    flipped = {sym: mode for sym, mode in rem_book.items() if sym in rem_neg_book}
    assert flipped, (rem_book, rem_neg_book)
    for sym, mode in flipped.items():
        opposite = "SHORT" if mode == "LONG" else "LONG"
        assert rem_neg_book[sym] == opposite, (sym, mode, rem_neg_book[sym])


# --------------------------------------------------------------------------- #
# LEG 5 -- pure-idiosyncratic exclusion
# --------------------------------------------------------------------------- #


def test_gate_leg5_pure_idio_excluded() -> None:
    symbols, prices, returns = _build_panel()
    d_idio = price_delay_share(returns["IDIO/USDT"][1:], returns["BTC/USDT"][1:], lags=_LAGS)
    assert d_idio is None

    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)  # must not raise
    scores, _ = candidate._delay_scores()
    assert "IDIO/USDT" not in scores
    book = _final_side(candidate.events.items)
    assert "IDIO/USDT" not in book, book


# --------------------------------------------------------------------------- #
# LEG 6 -- min-hold suppresses a flip inside the window (C1 rescue as a test)
# --------------------------------------------------------------------------- #


def _min_hold_panel() -> tuple[list[str], dict[str, list[float]]]:
    switch = 30
    n = 80
    bench_gen = _lcg(7)
    btc = [
        0.03 * (1 if next(bench_gen) > 0.5 else -1) + (next(bench_gen) - 0.5) * 2e-4
        for _ in range(n)
    ]
    btc_lag1 = [0.0, *btc[:-1]]

    def _jit(seed: int) -> list[float]:
        gen = _lcg(seed)
        return [(next(gen) - 0.5) * 1e-6 for _ in range(n)]

    # P is a strong LAG loader (high delay -> long) until ``switch``, then a
    # contemporaneous loader (its trailing-window delay decays out of the book).
    returns = {
        "BTC/USDT": btc,
        "P/USDT": [
            (0.9 * btc_lag1[i] if i < switch else 0.9 * btc[i]) + _jit(101)[i] for i in range(n)
        ],
        "L1/USDT": [0.8 * btc[i] + _jit(11)[i] for i in range(n)],
        "L2/USDT": [0.8 * btc[i] + _jit(22)[i] for i in range(n)],
        "L3/USDT": [0.8 * btc[i] + _jit(33)[i] for i in range(n)],
        "C1/USDT": [0.8 * btc[i] + _jit(44)[i] for i in range(n)],
        "C2/USDT": [0.8 * btc[i] + _jit(55)[i] for i in range(n)],
    }
    prices: dict[str, list[float]] = {}
    for symbol, series in returns.items():
        path = [100.0]
        for value in series:
            path.append(path[-1] * math.exp(value))
        prices[symbol] = path[1:]
    return list(returns), prices


def _run_min_hold(min_hold: int, max_hold: int) -> tuple[str, list[str]]:
    symbols, prices = _min_hold_panel()
    strategy = _candidate(
        symbols,
        delay_window=20,
        delay_lags=3,
        min_delay_obs=14,
        quantile_entry_pct=0.34,
        quantile_exit_pct=0.50,
        vol_window=10,
        min_hold_decisions=min_hold,
        max_hold_decisions=max_hold,
    )
    _feed(strategy, symbols, prices)
    p_signals = [sig.signal_type for sig in strategy.events.items if sig.symbol == "P/USDT"]
    return strategy._state["P/USDT"].mode, p_signals


def test_min_hold_suppresses_flip_inside_window() -> None:
    # A hard hold covering the whole post-entry span suppresses the flip: P is
    # entered once and never exited.
    mode_hold, sig_hold = _run_min_hold(500, 2000)
    assert mode_hold == "LONG", (mode_hold, sig_hold)
    assert sig_hold == ["LONG"], sig_hold
    # Contrast (anti-vacuous): with min_hold=1 the guard is off and P churns/flips.
    mode_free, sig_free = _run_min_hold(1, 52)
    assert mode_free != "LONG", (mode_free, sig_free)
    assert "EXIT" in sig_free, sig_free


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical() -> None:
    symbols, prices, _ = _build_panel()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        candidate = _candidate(symbols)
        _feed(candidate, symbols, prices)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in candidate.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal"


def test_state_roundtrip_lossless() -> None:
    symbols, prices, _ = _build_panel()
    candidate = _candidate(symbols)
    _feed(candidate, symbols, prices)
    snapshot = candidate.get_state()

    restored = _candidate(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    for symbol in symbols:
        assert list(restored._state[symbol].closes) == list(candidate._state[symbol].closes)
        assert restored._state[symbol].mode == candidate._state[symbol].mode
        assert restored._state[symbol].bars_held == candidate._state[symbol].bars_held
        assert restored._state[symbol].bars_since_exit == candidate._state[symbol].bars_since_exit
    assert restored._tick == candidate._tick
    assert restored._last_decision_week == candidate._last_decision_week


def test_adversarial_set_state_never_raises() -> None:
    symbols = [f"S{i}/USDT" for i in range(6)]
    strategy = _candidate(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "nope"})
    strategy.set_state({"symbol_state": {"S0/USDT": "not a dict"}})
    strategy.set_state({"symbol_state": {"S0/USDT": {"closes": 999}}})
    strategy.set_state(
        {
            "last_decision_week": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
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
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
        assert item.bars_held >= 0
        assert item.bars_since_exit >= 0


def test_degenerate_inputs_never_raise() -> None:
    symbols = ["BTC/USDT", "A/USDT", "B/USDT", "C/USDT", "D/USDT"]
    strategy = _candidate(symbols)
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    for index, value in enumerate((0.0, -5.0, float("nan"), float("inf"), 100.0)):
        bars = {"BTC/USDT": [(_week_iso(index), value, value, value, value, 1000.0)]}
        strategy.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_week_iso(index), bars_1s=bars)
        )
    assert _non_exit(strategy.events.items) == []


def test_self_skip_below_min_symbols() -> None:
    symbols = ["BTC/USDT", "DELAYED/USDT"]
    _, full, _ = _build_panel()
    prices = {symbol: full[symbol] for symbol in symbols}
    strategy = _candidate(symbols, min_symbols=5)
    _feed(strategy, symbols, prices)
    assert _non_exit(strategy.events.items) == []


def test_self_skip_history_too_short() -> None:
    symbols = ["BTC/USDT", "DELAYED/USDT", "SYNC/USDT", "FILL_A/USDT", "FILL_B/USDT", "FILL_C/USDT"]
    _, full, _ = _build_panel()
    prices = {symbol: full[symbol][:40] for symbol in symbols}  # below delay_window+1
    strategy = _candidate(symbols)
    _feed(strategy, symbols, prices)
    assert _non_exit(strategy.events.items) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalPriceDelayPremiumStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "benchmark_symbol",
        "delay_window",
        "delay_lags",
        "min_delay_obs",
        "score_mode",
        "quantile_entry_pct",
        "quantile_exit_pct",
        "min_hold_decisions",
        "cooldown_decisions",
        "max_hold_decisions",
        "vol_window",
        "min_symbols",
        "allow_short",
        "target_gross_exposure",
    ):
        assert required in schema
    for cap in ("benchmark_symbol", "min_r2", "base_allocation", "max_order_value"):
        assert schema[cap].tunable is False


def test_lag_weighted_mode_trades() -> None:
    symbols, prices, _ = _build_panel()
    candidate = _candidate(symbols, score_mode="lag_weighted")
    _feed(candidate, symbols, prices)
    book = _final_side(candidate.events.items)
    # The lag-weighted coefficient share still ranks the delayed names long.
    assert book, book
    assert any(mode == "LONG" for mode in book.values())
    assert any(mode == "SHORT" for mode in book.values())


def test_volume_invariance_not_a_dollar_volume_alias() -> None:
    # Illiquidity-alias check (author-time): the delay characteristic is computed
    # from PRICE returns only.  Feeding the identical price panel with wildly
    # different (price/dollar-volume-correlated) volume profiles must leave the
    # book bit-identical -- the signal structurally CANNOT be a dollar-volume /
    # turnover alias.
    symbols, prices, _ = _build_panel()

    baseline = _candidate(symbols)
    _feed(baseline, symbols, prices)
    baseline_book = _final_side(baseline.events.items)

    # Adversarial volume: proportional to a per-symbol multiplier x price (a
    # dollar-volume signal that a turnover alias WOULD latch onto).
    volumes = {
        symbol: [
            (idx + 1) * (7.0 + 3.0 * symbols.index(symbol)) * prices[symbol][idx]
            for idx in range(_N)
        ]
        for symbol in symbols
    }
    with_volume = _candidate(symbols)
    _feed(with_volume, symbols, prices, volumes)
    volume_book = _final_side(with_volume.events.items)

    assert baseline_book == volume_book, (baseline_book, volume_book)
    assert baseline_book, baseline_book


def test_slice_multi_timeframe_cells_pinned() -> None:
    """4h/1h scale delay_window (span) but hold the Dimson lag order and decisions."""
    from lumina_quant.strategies.price_delay_premium_alpha_sleeves import (
        _PRICE_DELAY_PREMIUM_SLICE as sl,
    )

    assert {"1d", "4h", "1h"} <= set(sl)
    base = tuple(cell["variant"] for cell in sl["1d"])
    for tf in ("4h", "1h"):
        assert tuple(cell["variant"] for cell in sl[tf]) == base
    assert sl["4h"][0]["delay_window"] == 1080
    assert sl["1h"][0]["delay_window"] == 4320
    assert sl["4h"][0]["vol_window"] == 180
    assert sl["1h"][0]["vol_window"] == 720
    # The Dimson lag ORDER and the weekly ISO decisions are timeframe invariant.
    for tf in ("1d", "4h", "1h"):
        assert sl[tf][0]["delay_lags"] == 5
        assert sl[tf][0]["min_hold_decisions"] == 4


# --------------------------------------------------------------------------- #
# v5 zero-alloc gate: a computed alloc of 0 must NOT emit a LONG/SHORT --
# ``_target_metadata`` omits ``target_allocation`` at alloc 0 and the engine
# would resize the entry to its DEFAULT allocation (an unsized, un-vol-gated bet).
# The inverse-vol weights are computed INTERNALLY, so an empty ``vols`` map
# forces every weight (and thus every alloc) to 0.
# --------------------------------------------------------------------------- #


def test_zero_alloc_entry_skipped_not_default_sized() -> None:
    symbols = ["BTC/USDT", "FLIP/USDT", "FRESH/USDT", "N0/USDT", "N1/USDT", "N2/USDT"]
    strat = _candidate(symbols)
    flip = strat._state["FLIP/USDT"]
    flip.mode = "LONG"
    flip.entry_price = 100.0
    flip.bars_held = 10_000
    desired = {"FLIP/USDT": "SHORT", "FRESH/USDT": "LONG"}
    # empty vols -> internal inverse-vol weights are all 0 -> alloc == 0
    strat._emit_targets(desired, {}, {}, {}, "2026-01-01T00:00:00Z")
    kinds = [(sig.symbol, str(sig.signal_type).upper()) for sig in strat.events.items]
    assert not [sym for sym, kind in kinds if kind in {"LONG", "SHORT"}], kinds
    # the side-flip EXIT still fired and FLIP is now flat (state matches the exit)
    assert ("FLIP/USDT", "EXIT") in kinds
    assert strat._state["FLIP/USDT"].mode == "OUT"
    assert strat._state["FLIP/USDT"].entry_price is None

    # a positive inverse-vol weight (from a real vol) DOES emit a sized entry
    # carrying a strictly positive ``target_allocation``.
    sized = _candidate(symbols)
    sized._emit_targets(
        {"FRESH/USDT": "LONG"},
        {"FRESH/USDT": 0.5},
        {"FRESH/USDT": 1.0},
        {"FRESH/USDT": 0.1},
        "2026-01-01T00:00:00Z",
    )
    entries = [
        sig for sig in sized.events.items if str(sig.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert entries, "a positive inverse-vol weight must emit a sized entry"
    assert all(float((sig.metadata or {}).get("target_allocation", 0.0)) > 0.0 for sig in entries)
