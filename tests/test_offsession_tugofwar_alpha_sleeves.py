"""Deterministic BUILD-GATE tests for CrossSectionalOffSessionTugOfWarStrategy.

Direct class import only (this lane ships without ``@register``, so there are no
registry/tier/candidate-wiring assertions -- those land with the W3 integration
wave). The fixture is a hand-built 63-day stream of UTC-stamped 1h bars from
Monday 2025-01-06.

Two TradFi legs carry the load:

* NIGHTPUMP -- its daily drift accrues entirely in OFF-session hours (a +budget in
  six night hours, a -budget across the six cash hours 14..19), so its
  ``TOW = mean(N_d) - mean(D_d)`` is strongly POSITIVE.
* SESSIONGRIND -- the exact MIRROR allocation (positives in the cash hours,
  negatives in the night hours): its per-UTC-day TOTAL return is bit-equal to
  NIGHTPUMP's every day (so their daily closes match to machine precision and
  every whole-day-aligned incumbent window scores them identically), but its TOW
  is strongly NEGATIVE.

The candidate SHORTS the persistently off-pumped leg (NIGHTPUMP) and LONGS the
cash-pumped leg (SESSIONGRIND); the whole-day-blind cross-sectional equity
momentum incumbent provably ties them, while the per-symbol session/overnight
riders go LONG the pumped night session -- opposite-signed action on identical
input.

Fixtures are closed-form (no RNG). ASCII only.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    CrossSectionalEquityMomentumStrategy,
)
from lumina_quant.strategies.intraday_overnight_alpha_sleeves import (
    IntradaySeasonalMomentumRiderStrategy,
    OvernightSessionReturnRiderStrategy,
    _bucket_signal,
)
from lumina_quant.strategies.offsession_tugofwar_alpha_sleeves import (
    CrossSectionalOffSessionTugOfWarStrategy,
)

# --------------------------------------------------------------------------- #
# fixture parameters
# --------------------------------------------------------------------------- #

_START = datetime(2025, 1, 6, tzinfo=UTC)  # Monday
_N_DAYS = 63
_CASH_HOURS = frozenset(range(14, 20))  # [14, 20)
_OFF_BUDGET_HOURS = frozenset({0, 1, 2, 3, 4, 5})  # six night hours carry +/- budget

NIGHT = "TSLAUSDT"
GRIND = "NVDAUSDT"
FILLERS = ["AAPLUSDT", "MSFTUSDT", "AMZNUSDT", "METAUSDT"]
BTC = "BTC/USDT"
_TRADFI = [NIGHT, GRIND, *FILLERS]
_ALL_SYMBOLS = [*_TRADFI, BTC]


def _budgets(day: int) -> tuple[float, float]:
    """Per-day positive/negative hour budgets (mild day-to-day variation)."""
    positive = 0.0006667 * (1.0 + 0.05 * ((day % 5) - 2))
    negative = 0.0005 * (1.0 + 0.05 * ((day % 3) - 1))
    return positive, negative


def _hour_return(symbol: str, day: int, weekday: int, hour: int) -> float:
    if weekday >= 5:
        return 0.0  # weekends flat
    positive, negative = _budgets(day)
    if symbol == NIGHT:
        if hour in _CASH_HOURS:
            return -negative
        return positive if hour in _OFF_BUDGET_HOURS else 0.0
    if symbol == GRIND:
        if hour in _CASH_HOURS:
            return positive
        return -negative if hour in _OFF_BUDGET_HOURS else 0.0
    # Fillers: dispersed daily totals so the incumbent has a live top-of-book.
    index = FILLERS.index(symbol)
    drift = [0.0008, -0.0006, 0.0002, 0.0011][index]
    if hour in _CASH_HOURS:
        return drift / 6.0 * (1.5 if index % 2 == 0 else 0.5)
    if hour in _OFF_BUDGET_HOURS:
        return drift / 6.0 * (0.5 if index % 2 == 0 else 1.5)
    return 0.0


def _build_series() -> tuple[list[datetime], dict[str, list[float]]]:
    times: list[datetime] = []
    closes: dict[str, list[float]] = {symbol: [] for symbol in _ALL_SYMBOLS}
    prices = dict.fromkeys(_ALL_SYMBOLS, 100.0)
    for step in range(_N_DAYS * 24):
        moment = _START + timedelta(hours=step)
        times.append(moment)
        day, weekday, hour = step // 24, moment.weekday(), moment.hour
        for symbol in _ALL_SYMBOLS:
            if symbol == BTC:
                ret = 0.0009 * math.sin(step / 5.0)
            else:
                ret = _hour_return(symbol, day, weekday, hour)
            prices[symbol] *= math.exp(ret)
            closes[symbol].append(prices[symbol])
    return times, closes


_TIMES, _CLOSES = _build_series()


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _event(symbol: str, moment: datetime, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=moment.isoformat().replace("+00:00", "Z"),
        symbol=symbol,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1000.0,
    )


def _feed(strategy: Any, symbols: list[str]) -> None:
    for index, moment in enumerate(_TIMES):
        for symbol in symbols:
            strategy.calculate_signals(_event(symbol, moment, _CLOSES[symbol][index]))


def _final_sides(items: list[Any]) -> dict[str, str]:
    sides: dict[str, str] = {}
    for signal in items:
        kind = str(signal.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            sides[signal.symbol] = kind
        elif kind == "EXIT":
            sides.pop(signal.symbol, None)
    return sides


def _make_candidate(symbols: list[str] | None = None, **overrides: Any) -> Any:
    params: dict[str, Any] = dict(
        formation_days=42,
        min_history_days=42,
        min_symbols=5,
        vol_window=168,
        quantile_pct=0.25,
        min_hold_decisions=2,
        allow_short=True,
    )
    params.update(overrides)
    bars = SimpleNamespace(symbol_list=list(symbols if symbols is not None else _ALL_SYMBOLS))
    return CrossSectionalOffSessionTugOfWarStrategy(bars, _Queue(), **params)


# --------------------------------------------------------------------------- #
# Stage-1: fixture properties asserted with the real primitives.
# --------------------------------------------------------------------------- #


def test_stage1_daily_closes_equal_and_tow_signs_and_classifier() -> None:
    # NIGHTPUMP and SESSIONGRIND share bit-equal daily (day-end) closes.
    day_end = [24 * day + 23 for day in range(_N_DAYS)]
    max_close_gap = max(abs(_CLOSES[NIGHT][i] - _CLOSES[GRIND][i]) for i in day_end)
    assert max_close_gap < 1e-9

    # The whole-day-aligned incumbent momentum + vol tie them to machine grain.
    mom_night = _CLOSES[NIGHT][-1] / _CLOSES[NIGHT][-1009] - 1.0
    mom_grind = _CLOSES[GRIND][-1] / _CLOSES[GRIND][-1009] - 1.0
    assert abs(mom_night - mom_grind) < 1e-9
    assert (
        abs(
            realized_volatility(_CLOSES[NIGHT], window=168)
            - realized_volatility(_CLOSES[GRIND], window=168)
        )
        < 1e-9
    )

    # TOW(NIGHT) > 0 > TOW(GRIND) read straight off the candidate's own state.
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    night_days = list(candidate._state[NIGHT].day_returns)[-42:]
    grind_days = list(candidate._state[GRIND].day_returns)[-42:]
    tow_night = sum(n for _c, n in night_days) / 42 - sum(c for c, _n in night_days) / 42
    tow_grind = sum(n for _c, n in grind_days) / 42 - sum(c for c, _n in grind_days) / 42
    assert tow_night > 0.0 > tow_grind

    # Classifier: Saturday hour-15 is OFF; Wednesday hour-13 is AMBIGUOUS.
    assert candidate._classify_hour(datetime(2025, 1, 11, 15, tzinfo=UTC)) == "OFF"
    assert candidate._classify_hour(datetime(2025, 1, 8, 13, tzinfo=UTC)) == "AMBIGUOUS"
    assert candidate._classify_hour(datetime(2025, 1, 8, 15, tzinfo=UTC)) == "CASH"


# --------------------------------------------------------------------------- #
# LEG 1: vs CrossSectionalEquityMomentum (whole-day permutation kill shot).
# --------------------------------------------------------------------------- #


def test_leg1_candidate_fades_while_momentum_incumbent_ties() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    sides = _final_sides(candidate.events.items)
    # Candidate ACTS on both book sides (long-short), fading the off-pumped leg.
    assert sides.get(NIGHT) == "SHORT"
    assert sides.get(GRIND) == "LONG"

    incumbent = CrossSectionalEquityMomentumStrategy(
        SimpleNamespace(symbol_list=list(_TRADFI)),
        _Queue(),
        lookback_bars=24 * 42,
        skip_bars=0,
        vol_window=24 * 7,
        rebalance_bars=1,
        min_symbols=4,
        allow_short=True,
        quintile_pct=0.20,
    )
    _feed(incumbent, _TRADFI)
    incumbent_sides = _final_sides(incumbent.events.items)
    # Incumbent-LIVE: it emits a non-empty book (longs the top-total-return filler).
    assert incumbent_sides
    assert incumbent_sides.get("METAUSDT") == "LONG"  # drift 0.0011 is the max
    # ... and treats NIGHTPUMP/SESSIONGRIND identically (both excluded from the
    # extreme quantiles it can see -- whole-day windows make them a permutation).
    assert incumbent_sides.get(NIGHT) == incumbent_sides.get(GRIND)


def test_leg1_btc_is_never_traded() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    assert BTC not in candidate.symbol_list  # universe-filtered out entirely
    assert all(signal.symbol != BTC for signal in candidate.events.items)


# --------------------------------------------------------------------------- #
# LEG 2: vs OvernightSessionReturnRiderStrategy (opposite-signed action).
# --------------------------------------------------------------------------- #


def test_leg2_overnight_rider_longs_the_pumped_night_while_candidate_shorts() -> None:
    rider = OvernightSessionReturnRiderStrategy(
        SimpleNamespace(symbol_list=[NIGHT]),
        _Queue(),
        overnight_start_hour_utc=21,
        overnight_end_hour_utc=13,
        session_decay=0.05,
        min_session_observations=10,
        session_t_threshold=1.5,
        allow_short=True,
    )
    _feed(rider, [NIGHT])
    rider_longs = [s for s in rider.events.items if s.symbol == NIGHT and s.signal_type == "LONG"]
    # Incumbent-LIVE + acting: it goes LONG the night session it harvests.
    assert rider_longs
    # WHY: the overnight bucket's decay-weighted mean return is positive (the
    # sentiment pump the candidate fades), and the bucket signal is non-negative.
    overnight = rider._session_stats[NIGHT].get("overnight")
    assert overnight is not None and overnight[0] > 0.0
    assert (
        _bucket_signal(
            rider._session_stats[NIGHT],
            "overnight",
            min_obs=10,
            t_threshold=1.5,
            decay=0.05,
        )
        >= 0
    )

    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    assert _final_sides(candidate.events.items).get(NIGHT) == "SHORT"


# --------------------------------------------------------------------------- #
# LEG 3: vs IntradaySeasonalMomentumRiderStrategy (second opposite-sign leg).
# --------------------------------------------------------------------------- #


def test_leg3_intraday_seasonal_rider_longs_night_while_candidate_shorts() -> None:
    rider = IntradaySeasonalMomentumRiderStrategy(SimpleNamespace(symbol_list=[NIGHT]), _Queue())
    _feed(rider, [NIGHT])
    rider_longs = [s for s in rider.events.items if s.symbol == NIGHT and s.signal_type == "LONG"]
    # Aligned uptrend + significant positive off-session slots -> the slot rider
    # arms LONG (live + acting).
    assert rider_longs

    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    assert _final_sides(candidate.events.items).get(NIGHT) == "SHORT"


# --------------------------------------------------------------------------- #
# LEG 4: not-secretly-momentum -- collinear TOW/momentum -> residual ~0 -> abstain.
# --------------------------------------------------------------------------- #


def test_leg4_collinear_tow_and_momentum_makes_the_sleeve_abstain() -> None:
    # Cash hours flat for every symbol -> D_d == 0 for all days, so
    # momentum == formation * TOW exactly (per symbol): TOW is a pure linear
    # transform of momentum, the residual collapses, and the sleeve abstains.
    times: list[datetime] = []
    closes: dict[str, list[float]] = {symbol: [] for symbol in _TRADFI}
    prices = dict.fromkeys(_TRADFI, 100.0)
    for step in range(_N_DAYS * 24):
        moment = _START + timedelta(hours=step)
        times.append(moment)
        day, weekday, hour = step // 24, moment.weekday(), moment.hour
        for index, symbol in enumerate(_TRADFI):
            ret = 0.0
            if weekday < 5 and hour in _OFF_BUDGET_HOURS:
                # Distinct per-symbol off drift with day-to-day variation (defeats
                # the zero-off-variance guard) -- but zero cash return throughout.
                base = 0.0004 * (index + 1)
                ret = base * (1.0 + 0.1 * ((day % 4) - 1.5))
            prices[symbol] *= math.exp(ret)
            closes[symbol].append(prices[symbol])

    candidate = _make_candidate(symbols=_TRADFI)
    for index, moment in enumerate(times):
        for symbol in _TRADFI:
            candidate.calculate_signals(_event(symbol, moment, closes[symbol][index]))
    # Sanity: cash-session component is identically zero, so TOW and momentum are
    # perfectly collinear and no long/short target may be emitted.
    night_days = list(candidate._state[_TRADFI[0]].day_returns)[-42:]
    assert all(abs(cash) < 1e-15 for cash, _off in night_days)
    assert not [s for s in candidate.events.items if s.signal_type in {"LONG", "SHORT"}]


# --------------------------------------------------------------------------- #
# LEG 5: hygiene -- min-hold, zero-off-variance abstain, state, adversarial, degenerate.
# --------------------------------------------------------------------------- #


def test_leg5_min_hold_suppresses_flip_until_the_hold_clears() -> None:
    def _seed(min_hold: int) -> Any:
        strategy = _make_candidate(min_hold_decisions=min_hold)
        item = strategy._state[NIGHT]
        item.mode = "LONG"
        item.entry_price = 100.0
        item.decisions_held = 0
        item.closes.append(100.0)
        return strategy, item

    targets = {NIGHT: ("SHORT", -1.5, {})}
    weights = {NIGHT: 1.0}

    held, held_item = _seed(min_hold=2)
    held._emit_targets(targets, weights, 1.0, _TIMES[-1])
    assert held_item.mode == "LONG"  # flip suppressed inside the hold window

    flips, flip_item = _seed(min_hold=0)
    flips._emit_targets(targets, weights, 1.0, _TIMES[-1])
    assert flip_item.mode == "SHORT"  # min_hold=0 reference flips immediately


def test_leg5_zero_off_session_variance_symbol_abstains() -> None:
    # A symbol pinned off-hours (all-zero off-session tape) has zero N_d
    # dispersion -> the data-quality probe skips it even with live cash moves.
    candidate = _make_candidate()
    item = candidate._state[NIGHT]
    for _ in range(50):
        item.day_returns.append((0.001, 0.0))  # non-trivial cash, dead off tape
    residual, _vols, _metas = candidate._residual_scores()
    assert NIGHT not in residual


def test_leg5_state_roundtrip_is_stable_mid_position() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    snapshot = candidate.get_state()
    revived = _make_candidate()
    revived.set_state(snapshot)
    assert revived.get_state() == snapshot


def test_leg5_adversarial_set_state_never_raises() -> None:
    candidate = _make_candidate()
    for junk in (
        None,
        {},
        {"last_eval_week": "bad", "symbol_state": 5},
        {"symbol_state": {NIGHT: {"closes": ["x", None], "day_returns": [[1], "y", 3]}}},
        {"symbol_state": {NIGHT: {"mode": 123, "decisions_held": -9, "prev_close": "z"}}},
        {"last_eval_week": [1], "symbol_state": {"UNKNOWN": {"closes": [1.0]}}},
    ):
        candidate.set_state(junk)  # must never raise


def test_leg5_degenerate_input_never_raises() -> None:
    candidate = _make_candidate(symbols=[NIGHT, GRIND, *FILLERS])
    for event in (
        SimpleNamespace(type="MARKET", time=None, symbol=NIGHT, close=None),
        SimpleNamespace(type="MARKET", time="not-a-date", symbol=NIGHT, close=100.0),
        SimpleNamespace(
            type="MARKET", time="2025-01-06T00:00:00Z", symbol=NIGHT, close=float("nan")
        ),
        SimpleNamespace(type="MARKET_WINDOW", time="2025-01-06T00:00:00Z", bars_1s={}),
        SimpleNamespace(type="OTHER"),
    ):
        candidate.calculate_signals(event)  # must never raise


def test_leg5_self_skips_below_min_symbols() -> None:
    candidate = _make_candidate(symbols=[NIGHT, GRIND], min_symbols=5)
    _feed(candidate, [NIGHT, GRIND])
    assert not [s for s in candidate.events.items if s.signal_type in {"LONG", "SHORT"}]


# --------------------------------------------------------------------------- #
# (pm) POSTMORTEM of the measured 1h cell `offsession_tugofwar_1h_tow_42d_fade`
# (research_note 2026-07-09: OOS Sharpe -33.95, return -63.19%, MDD 63.85%).  A
# Sharpe that negative implicates a SIGN INVERSION or an every-bar TURNOVER
# churn; these pin both as absent, so the loss is a DIRECTIONAL (wrong-signed /
# absent) fade edge on this universe -- the docstring's declared wrong-sign
# alternative (off-session moves are genuine Asia/Europe information ->
# continuation), i.e. the pre-registered EXPECTED-NULL, not a code defect.
# --------------------------------------------------------------------------- #


def test_pm_fade_sign_shorts_high_tow_longs_low_tow() -> None:
    """THEORY (Akbas-Boehmer-Jiang-Koch): persistent off-session strength
    predicts NEGATIVELY -> FADE.  The high-TOW leg (NIGHTPUMP) must be SHORT and
    the low-TOW leg (SESSIONGRIND) LONG.  Culprit-1 (sign) regression lock:
    a silent flip masquerading as a fix for the OOS loss breaks here.
    """
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)
    sides = _final_sides(candidate.events.items)
    assert sides.get(NIGHT) == "SHORT"
    assert sides.get(GRIND) == "LONG"


def test_pm_decision_clock_is_weekly_iso_not_every_bar() -> None:
    """The 1h cell decides on an ISO-week clock: ``_evaluate`` fires once per NEW
    ISO week (the first week only seeds ``_last_eval_week``), NOT once per 1h bar.

    Culprit-2 (turnover-churn) lock: at a weekly decision cadence over 1512 fed
    bars the round-trip count -- and the 20bps cost bleed -- is far too small to
    explain the measured -63% OOS return, so the loss is directional.
    """
    candidate = _make_candidate()
    calls = {"n": 0}
    original = candidate._evaluate

    def _counting(event_time: Any) -> None:
        calls["n"] += 1
        original(event_time)

    candidate._evaluate = _counting  # type: ignore[method-assign]
    _feed(candidate, _ALL_SYMBOLS)
    weeks = len({moment.isocalendar()[:2] for moment in _TIMES})
    assert calls["n"] == weeks - 1  # one decision per new ISO week, seed week excluded
    assert calls["n"] <= 12  # far below the 1512 one-hour bars fed


def test_pm_vol_target_scalar_engages_at_hourly_vol_scale() -> None:
    """FIXED (rewrite of the offsession postmortem's Culprit-3 diagnostic).

    Culprit-3 (sizing) WAS: the sleeve sizes off
    ``realized_volatility(closes, window=vol_window)``, a PER-1h-BAR stdev
    (``annualization=1.0``), while ``target_vol`` defaults to 0.20 (an
    annualized-scale figure), so ``scalar = min(1.0, target_vol / portfolio_vol)``
    stayed pinned at 1.0 for every realistic hourly vol -- the throttle NEVER
    de-risked.  The fix annualizes the per-bar portfolio vol by
    ``sqrt(bars_per_year)`` (cadence inferred from the median observed bar
    spacing) BEFORE the comparison.  On the 1h fixture a hurricane per-1h vol now
    DE-RISKS the book, where the old horizon-mismatched code left the knob inert.

    The postmortem's OTHER revival precondition -- a benchmark hedge leg for the
    directional exposure -- is OUT of this fix's scope and REMAINS OPEN.
    """
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)  # seeds distinct 1h decision-bar epochs
    assert len(candidate._recent_times) >= 3  # cadence is now inferable (~1h spacing)
    targets = {NIGHT: ("SHORT", -1.5, {}), GRIND: ("LONG", 1.5, {})}
    # High per-1h vol annualizes far above the 0.20 target -> throttle ENGAGES.
    _weights, hot = candidate._inverse_vol_weights(targets, {NIGHT: 0.02, GRIND: 0.02 * 1.3})
    assert hot < 1.0, hot
    # Control (calm): a genuinely tiny per-1h vol annualizes below target -> the
    # throttle stays at 1.0 and does NOT needlessly de-risk.
    _weights, calm = candidate._inverse_vol_weights(targets, {NIGHT: 0.0002, GRIND: 0.0002 * 1.3})
    assert calm == 1.0, calm
    # Control (unknown cadence): a fresh candidate with no observed bars keeps the
    # conservative unity scalar -- it never inflates leverage on an unknown horizon.
    fresh = _make_candidate()
    _weights, unknown = fresh._inverse_vol_weights(targets, {NIGHT: 0.02, GRIND: 0.02 * 1.3})
    assert unknown == 1.0, unknown


def test_run_twice_bit_identical() -> None:
    first = _make_candidate()
    _feed(first, _ALL_SYMBOLS)
    second = _make_candidate()
    _feed(second, _ALL_SYMBOLS)
    a = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in first.events.items]
    b = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in second.events.items]
    assert a == b


def test_ambiguous_hours_cover_both_dst_open_straddles() -> None:
    """The NYSE open sits at :30, so the open-straddling hour is 13 under EDT
    but 14 under EST -- both must be dropped from BOTH components (v5 fix:
    the EST winter pre-open half-hour used to be classified as CASH)."""
    strat = _make_candidate()
    assert strat._ambiguous_hours == frozenset({13, 14, 20})
    est_winter_weekday = datetime(2026, 1, 7, 14, 0, tzinfo=UTC)  # Wednesday
    assert strat._classify_hour(est_winter_weekday) == "AMBIGUOUS"
    fully_cash = datetime(2026, 1, 7, 15, 0, tzinfo=UTC)
    assert strat._classify_hour(fully_cash) == "CASH"


# --------------------------------------------------------------------------- #
# v5 zero-alloc gate: a computed alloc of 0 must NOT emit a LONG/SHORT --
# ``_target_metadata`` omits ``target_allocation`` at alloc 0 and the engine
# would resize the entry to its DEFAULT allocation (an unsized, un-vol-gated bet).
# --------------------------------------------------------------------------- #


def test_zero_alloc_entry_skipped_not_default_sized() -> None:
    # This lane admits only its fixed TradFi-ETF universe, so FLIP/FRESH must be
    # real members of ``_TRADFI_EQUITY_ETF_UNIVERSE``.
    symbols = list(_ALL_SYMBOLS)[:6]
    flip_sym, fresh_sym = symbols[0], symbols[1]
    strat = _make_candidate(symbols)
    flip = strat._state[flip_sym]
    flip.mode = "LONG"
    flip.entry_price = 100.0
    flip.decisions_held = 10_000  # clear the min-hold so the side flip is live
    targets = {
        flip_sym: ("SHORT", -1.0, {}),
        fresh_sym: ("LONG", 1.0, {}),
    }
    # empty weights -> alloc == 0 for every targeted symbol
    strat._emit_targets(targets, {}, 1.0, "2026-01-01T00:00:00Z")
    kinds = [(sig.symbol, str(sig.signal_type).upper()) for sig in strat.events.items]
    assert not [sym for sym, kind in kinds if kind in {"LONG", "SHORT"}], kinds
    # the side-flip EXIT still fired and FLIP is now flat (state matches the exit)
    assert (flip_sym, "EXIT") in kinds
    assert strat._state[flip_sym].mode == "OUT"
    assert strat._state[flip_sym].entry_price is None

    # a positive inverse-vol weight DOES emit a sized entry carrying a strictly
    # positive ``target_allocation``.
    sized = _make_candidate(symbols)
    sized._emit_targets(
        {fresh_sym: ("LONG", 1.0, {})}, {fresh_sym: 0.5}, 1.0, "2026-01-01T00:00:00Z"
    )
    entries = [
        sig for sig in sized.events.items if str(sig.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert entries, "a positive inverse-vol weight must emit a sized entry"
    assert all(float((sig.metadata or {}).get("target_allocation", 0.0)) > 0.0 for sig in entries)
