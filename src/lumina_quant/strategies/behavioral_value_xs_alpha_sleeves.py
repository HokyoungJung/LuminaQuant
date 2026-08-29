"""Behavioral-value weekly XS fade sleeves: salience-theory + prospect-theory.

Two cross-sectional basket rebalancers built on OHLCV closes ONLY (no feature
store), sharing one weekly ISO-week rebalancer skeleton:

- ``SalienceTheoryValueStrategy`` (id ``salience_theory_value_xs``): per symbol,
  over a trailing 21-UTC-day formation of daily perp returns ``r_d`` vs the
  equal-weight universe return ``m_d``, salience
  ``sigma_d = |r_d - m_d| / (|r_d| + |m_d| + theta)`` with ``theta = 0.1``;
  days ranked by ``sigma`` descending get weights ``w_d ~ delta**rank``
  (``delta = 0.7``, BGS canonical); ``ST = sum(w_d * r_d) - mean(r_d)`` is the
  salience-induced distortion of the perceived vs objective mean.  High-ST
  names are overbought by salient-upside co-timing -> negative forward
  returns: SHORT the top quintile, LONG the bottom quintile.
- ``ProspectTheoryValueStrategy`` (id ``prospect_theory_value_xs``): per
  symbol, the TK-1992 prospect-theory value of the sorted histogram of the
  trailing 90 UTC days of daily returns (``alpha = 0.88``, ``lam = 2.25``,
  ``gamma = 0.61`` gains / ``0.69`` losses, all canonical and frozen).
  Barberis-Jin-Wang: distributions attractive to prospect-theory investors get
  overbought -> SHORT the top-TK quintile, LONG the bottom quintile.

SHARED SKELETON (both classes).  UTC-day sampling: bars of any feed timeframe
are collapsed to one completed close per UTC day, so the daily-return
formation is timeframe-invariant.  Weekly ISO-week rebalance; enter-q20 /
exit-q35 rank hysteresis; hard min-hold of 5 DAILY decisions (the proven RPT
rescue); inverse-realized-vol sizing; the characteristic is residualized via
``lumina_quant.indicators.cross_sectional_residualize`` against the 21d
momentum z AND the 21d MAX-daily-return z.  RESIDUALIZATION NOTE: the full
lottery score (skew + MAX blend) lives in the PRIVATE
``LotterySkewnessStrategy._lottery_score``; per the pre-registered fallback,
this module residualizes against momentum plus the MAX(21d max daily return)
component computed INLINE from OHLCV (the Bali-Cakici-Whitelaw MAX leg of the
lottery blend) -- the remaining skew leg is stripped on the data-PC by the
falsifying measurement itself (Gram-Schmidt vs the full lottery score).

EXTERNAL PRIOR (verbatim from the pre-registered proposals)
-----------------------------------------------------------
ST: Bordalo, Gennaioli & Shleifer (QJE 2012, salience theory of choice under
risk; canonical theta=0.1, delta=0.7); Cosemans & Frehen (JFE 2021) negative
ST premium in equities surviving MAX/skew/IVOL controls; post-2022 crypto
replication: "Salience theory and cryptocurrency returns" (Finance Research
Letters, ~2022) reports the same negative ST premium in coin cross-sections
(the equity anchor is certain).
PT: Tversky & Kahneman (1992) parameter set; Barberis, Jin & Wang (JF 2021)
"Prospect theory and stock market anomalies" -- TK value negatively predicts
XS returns beyond MAX/skew/IVOL; post-2022 crypto replications of TK-value
pricing in coin cross-sections (IRFA/FRL 2023-2024 strand; the equity anchor
and the mechanism are the load-bearing prior).

EXPECTED NULL / FALSIFYING MEASUREMENT (verbatim, pre-registered)
-----------------------------------------------------------------
ST null: "In this 110-name liquid universe the ST distortion is subsumed by
the lottery blend (skew+MAX) already traded by LotterySkewnessStrategy:
residualized ST carries zero incremental IC, or the quintile spread dies
inside the 20-30bps cost grid."  ST falsifier: "On data-PC: weekly rank IC of
ST residualized (Gram-Schmidt) against lottery score, 21d momentum, and IVOL
over the full sample; admission requires incremental orthogonal factor_ic > 0
vs LotterySkewnessStrategy plus the standard gates (DSR>=0.90, SPA<=0.05,
net-20bps edge>0).  Residual IC <= 0 on validation-forward folds kills the
lane in one measurement."
PT null: "TK value on a 110-name liquid universe collapses onto the lottery
blend and/or onto ST value: incremental orthogonal IC vs
LotterySkewnessStrategy is zero, or it loses the pre-registered tournament
against salience-theory-value-xs."  PT falsifier: "Same single measurement as
ST lane: validation-forward weekly rank IC of TK residualized against lottery
score, momentum, IVOL, AND ST value; residual IC <= 0 or tournament loss vs ST
(when |rho|>0.6) kills it.  Standard admission gates apply on top."

REDUNDANCY TOURNAMENT (pre-registered, binding on data-PC admission)
--------------------------------------------------------------------
If ``|XS rank corr(TK, ST)| > 0.6`` on the data-PC, only the lane with the
higher orthogonal IC vs LotterySkewnessStrategy admits -- the pair is a
tournament, not two slots.

NEAREST INCUMBENT / ORTHOGONALITY.  ``LotterySkewnessStrategy`` (skew + MAX
blend) and ``CGO``.  Skew/MAX/CGO are invariant to WHEN a symbol's big days
occur relative to the market; ST is a co-timing functional -- permute which
market day each own-return aligns with and ST changes while the lottery score
is bit-identical (mechanical data-free proof, test (a)).  TK is a
full-distribution functional: two symbols with identical skew and identical
MAX but different mid-distribution loss mass get different TK (mechanical
data-free proof, test (b)).  Both sleeves additionally residualize against
momentum + MAX ex-ante.

NEAREST GRAVEYARD.  #5 OHLCV single leaves / #17 single-leaf robust selectors
and killer (i) val->OOS collapse: pre-empted because these are slow XS basket
rebalancers (weekly clock, quintile hysteresis, hard min-hold), not per-symbol
leaf triggers, and EVERY constant is a literature value pre-registered ex-ante
(theta=0.1, delta=0.7, TK-1992 set, 21d/90d formations) with no tunable grid.

TURNOVER.  Overlapping formations (21d ST / 90d PT) give week-over-week rank
autocorrelation near 1; the weekly clock + enter-q20/exit-q35 hysteresis +
min-hold 5 daily decisions target ~2-4 week holds for ST and 3-6 weeks for PT
(~13-26 round trips/yr/slot), keeping RPT >= 10bps plausible against a ~20bps
round-trip cost.  Entry metadata stamps ``intended_hold_seconds`` (~2 weeks ST
/ ~3 weeks PT).

This module is data-local (no I/O), completed-bar (a UTC day's close is used
only after a later bar proves the day complete), and never raises from
``calculate_signals``.  Zero/negative computed allocation emits NOTHING.
Registration is class-level only; tier hints, BATCH wiring, and candidate
builders land in the orchestrator's integration commit.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import pairwise
from datetime import date
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.behavioral_value import prospect_theory_value
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.cross_sectional_residualize import (
    cross_sectional_residualize,
)
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _state_size
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _Snapshot,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_ST_STRATEGY_ID = "salience_theory_value_xs"
_ST_STRATEGY_NAME = "SalienceTheoryValueStrategy"
_PT_STRATEGY_ID = "prospect_theory_value_xs"
_PT_STRATEGY_NAME = "ProspectTheoryValueStrategy"

# Pre-registered intended holds stamped into entry metadata (~2 weeks ST /
# ~3 weeks PT), consumed by the engine's funding-entry / hold-accounting seams.
_ST_INTENDED_HOLD_SECONDS = 14 * 86400
_PT_INTENDED_HOLD_SECONDS = 21 * 86400


@dataclass(slots=True)
class _State:
    """Per-symbol UTC-day close series + position / min-hold bookkeeping."""

    daily_closes: deque[float]  # one close per COMPLETED UTC day
    daily_days: deque[str]  # committed UTC day key per daily_closes entry
    pending_day: str = ""  # UTC day of the not-yet-committed close
    pending_close: float | None = None
    last_committed_day: str = ""  # staleness guard for panel eligibility
    last_bar_key: str = ""  # dedup identical ingested bars
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    days_held: int = 0  # DAILY decisions in the current position
    score: float | None = None


def _utc_datetime(raw_time: Any) -> Any | None:
    """Return a parseable UTC timestamp; never substitute a lexical clock."""
    if isinstance(raw_time, bool) or (
        isinstance(raw_time, (int, float)) and not math.isfinite(raw_time)
    ):
        return None
    return _event_datetime_utc(raw_time)


def _utc_day(raw_day: Any) -> date | None:
    """Parse an exact serialized UTC calendar day."""
    if not isinstance(raw_day, str) or len(raw_day) != 10:
        return None
    try:
        parsed = date.fromisoformat(raw_day)
    except TypeError, ValueError:
        return None
    return parsed if parsed.isoformat() == raw_day else None


def _utc_week(raw_week: Any) -> bool:
    """Validate an exact ISO week clock."""
    if not isinstance(raw_week, str) or len(raw_week) != 8 or raw_week[4:6] != "-W":
        return False
    try:
        year, week = int(raw_week[:4]), int(raw_week[6:])
        return (
            f"{year:04d}-W{week:02d}" == raw_week
            and date.fromisocalendar(year, week, 1) is not None
        )
    except TypeError, ValueError:
        return False


def _consecutive_days(days: list[str], *, ending: str | None = None) -> bool:
    """Require a complete consecutive UTC-day calendar, optionally anchored."""
    parsed = [_utc_day(day) for day in days]
    if any(day is None for day in parsed):
        return False
    if ending is not None and (not parsed or parsed[-1] != _utc_day(ending)):
        return False
    return all(right == left.fromordinal(left.toordinal() + 1) for left, right in pairwise(parsed))


def _strictly_newer_time(raw_time: Any, previous_key: str) -> bool:
    """Compare only parseable timestamps chronologically."""
    candidate = _utc_datetime(raw_time)
    if candidate is None:
        return False
    if not previous_key:
        return True
    previous = _utc_datetime(previous_key)
    return previous is not None and candidate > previous


def _xs_zscores(values: dict[str, float]) -> dict[str, float]:
    """Return the cross-sectional z-score of each value (degenerate -> all 0)."""
    if not values:
        return {}
    items = list(values.values())
    count = len(items)
    mean_value = sum(items) / float(count)
    sigma = sample_std(items)
    if sigma is None or sigma <= _EPS:
        return dict.fromkeys(values, 0.0)
    return {symbol: (value - mean_value) / sigma for symbol, value in values.items()}


def _daily_simple_returns(closes: list[float]) -> list[float]:
    """Day-over-day simple returns of a positive daily close path."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and prev > _EPS and value == value:
            out.append(value / prev - 1.0)
        prev = value
    return out


def _momentum_return(returns: list[float], days: int) -> float | None:
    """Compounded simple return over the trailing ``days`` daily returns."""
    tail = returns[-max(1, int(days)) :]
    if len(tail) < max(1, int(days)):
        return None
    growth = 1.0
    for value in tail:
        growth *= 1.0 + value
    result = growth - 1.0
    return float(result) if math.isfinite(result) else None


def _max_daily_return(returns: list[float], days: int) -> float | None:
    """Bali-Cakici-Whitelaw MAX: the largest single daily return in the window."""
    tail = returns[-max(1, int(days)) :]
    if len(tail) < max(1, int(days)):
        return None
    result = max(tail)
    return float(result) if math.isfinite(result) else None


def _salience_value(
    returns: list[float],
    market_returns: list[float],
    *,
    theta: float,
    delta: float,
    probability_mass: list[float] | None = None,
) -> float | None:
    """BGS distortion with one consistent probability measure for both values."""
    if len(returns) != len(market_returns) or len(returns) < 2:
        return None
    masses = probability_mass or [1.0 / len(returns)] * len(returns)
    if len(masses) != len(returns) or any(mass < 0.0 or not math.isfinite(mass) for mass in masses):
        return None
    total_mass = sum(masses)
    if total_mass <= _EPS:
        return None
    masses = [mass / total_mass for mass in masses]
    if any(not math.isfinite(value) for value in (*returns, *market_returns)):
        return None
    salience = [
        abs(own - market) / (abs(own) + abs(market) + theta)
        for own, market in zip(returns, market_returns, strict=True)
    ]
    order = sorted(range(len(returns)), key=lambda idx: (-salience[idx], idx))
    attention = [masses[idx] * delta**rank for rank, idx in enumerate(order)]
    attention_total = sum(attention)
    if attention_total <= _EPS:
        return None
    distorted = sum(
        (weight / attention_total) * returns[idx]
        for weight, idx in zip(attention, order, strict=True)
    )
    objective = sum(mass * value for mass, value in zip(masses, returns, strict=True))
    result = distorted - objective
    return float(result) if math.isfinite(result) else None


class _WeeklyBehavioralValueXS(Strategy):
    """Shared weekly XS fade rebalancer over UTC-day sampled closes.

    Subclasses provide the behavioral characteristic via
    ``_characteristic_value`` plus their registry identity and intended hold.
    The skeleton handles UTC-day sampling, the ISO-week decision clock, the
    enter-q20/exit-q35 hysteresis, the hard daily min-hold, inverse-vol
    sizing, momentum+MAX residualization, and emission.  Never raises from
    ``calculate_signals``; zero/negative computed allocation emits nothing.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly internal decision clock
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_id = ""
    strategy_name = ""
    intended_hold_seconds = 0

    # ------------------------------------------------------------------ #
    # shared schema / init
    # ------------------------------------------------------------------ #
    @classmethod
    def _shared_param_schema(cls, *, formation_days_default: int) -> dict[str, HyperParam]:
        return {
            "formation_days": HyperParam.integer(
                "formation_days", default=formation_days_default, low=5, high=400, tunable=False
            ),
            "momentum_days": HyperParam.integer(
                "momentum_days", default=21, low=2, high=400, tunable=False
            ),
            "max_days": HyperParam.integer("max_days", default=21, low=2, high=400, tunable=False),
            "vol_window": HyperParam.integer(
                "vol_window", default=21, low=2, high=400, tunable=False
            ),
            "quantile_entry_pct": HyperParam.floating(
                "quantile_entry_pct", default=0.20, low=0.02, high=0.50, tunable=False
            ),
            "quantile_exit_pct": HyperParam.floating(
                "quantile_exit_pct", default=0.35, low=0.02, high=1.0, tunable=False
            ),
            "min_hold_daily_decisions": HyperParam.integer(
                "min_hold_daily_decisions", default=5, low=0, high=100000, tunable=False
            ),
            "max_hold_daily_decisions": HyperParam.integer(
                "max_hold_daily_decisions", default=28, low=1, high=200000, tunable=False
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def _init_shared(self, bars: Any, events: Any, resolved: dict[str, Any]) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        self.formation_days = max(5, int(resolved["formation_days"]))
        self.momentum_days = max(2, int(resolved["momentum_days"]))
        self.max_days = max(2, int(resolved["max_days"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        entry = min(0.5, max(0.02, float(resolved["quantile_entry_pct"])))
        exit_pct = min(1.0, max(entry, float(resolved["quantile_exit_pct"])))
        self.quantile_entry_pct = entry
        self.quantile_exit_pct = exit_pct
        self.min_hold_daily_decisions = max(0, int(resolved["min_hold_daily_decisions"]))
        self.max_hold_daily_decisions = max(1, int(resolved["max_hold_daily_decisions"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self._required_days = (
            max(self.formation_days, self.momentum_days, self.max_days, self.vol_window) + 1
        )
        size = _state_size(self._required_days, self.max_hold_daily_decisions)
        self._state: dict[str, _State] = {
            symbol: _State(daily_closes=deque(maxlen=size), daily_days=deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_day_key = ""
        self._last_committed_day = ""
        self._last_decision_week = ""
        self._last_window_key = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # subclass hook
    # ------------------------------------------------------------------ #
    def _characteristic_value(
        self, returns: list[float], market_returns: list[float]
    ) -> float | None:  # pragma: no cover - overridden
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_day_key": self._last_day_key,
            "last_committed_day": self._last_committed_day,
            "last_decision_week": self._last_decision_week,
            "last_window_key": self._last_window_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "daily_closes": list(item.daily_closes),
                    "daily_days": list(item.daily_days),
                    "pending_day": item.pending_day,
                    "pending_close": item.pending_close,
                    "last_committed_day": item.last_committed_day,
                    "last_bar_key": item.last_bar_key,
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "days_held": int(item.days_held),
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Install only a fully coherent checkpoint; leave state untouched otherwise."""
        top_level_fields = {
            "last_day_key",
            "last_committed_day",
            "last_decision_week",
            "last_window_key",
            "tick",
            "symbol_state",
        }
        symbol_fields = {
            "daily_closes",
            "daily_days",
            "pending_day",
            "pending_close",
            "last_committed_day",
            "last_bar_key",
            "mode",
            "entry_price",
            "days_held",
            "score",
        }
        if type(state) is not dict or set(state) != top_level_fields:
            return
        raw = state["symbol_state"]
        if type(raw) is not dict or set(raw) != set(self._state):
            return
        last_day = state["last_day_key"]
        last_committed = state["last_committed_day"]
        last_week = state["last_decision_week"]
        last_window = state["last_window_key"]
        tick = state["tick"]
        if (
            type(last_day) is not str
            or type(last_committed) is not str
            or type(last_week) is not str
            or type(last_window) is not str
            or type(tick) is not int
            or tick < 1
            or _utc_day(last_day) is None
            or _utc_day(last_committed) is None
            or not _utc_week(last_week)
            or _utc_datetime(last_window) is None
            or _utc_day(last_committed) >= _utc_day(last_day)
            or self._day_key(last_window) != last_day
            or self._week_key(last_window) != last_week
        ):
            return
        restored: dict[str, _State] = {}
        common_days: list[str] | None = None
        common_pending: str | None = None
        for symbol in self.symbol_list:
            payload = raw[symbol]
            if type(payload) is not dict or set(payload) != symbol_fields:
                return
            closes, days = payload["daily_closes"], payload["daily_days"]
            if (
                type(closes) is not list
                or type(days) is not list
                or not closes
                or len(closes) != len(days)
                or len(closes) > int(self._state[symbol].daily_closes.maxlen or 0)
                or not all(
                    type(value) is float and math.isfinite(value) and value > self.min_price
                    for value in closes
                )
                or not all(type(value) is str for value in days)
                or not _consecutive_days(days, ending=last_committed)
            ):
                return
            pending_day, pending_close = (
                payload["pending_day"],
                payload["pending_close"],
            )
            item_last_committed, last_bar = (
                payload["last_committed_day"],
                payload["last_bar_key"],
            )
            mode, entry_price, days_held, score = (
                payload["mode"],
                payload["entry_price"],
                payload["days_held"],
                payload["score"],
            )
            if (
                type(pending_day) is not str
                or type(item_last_committed) is not str
                or type(last_bar) is not str
                or mode not in {"OUT", "LONG", "SHORT"}
                or type(days_held) is not int
                or days_held < 0
                or (item_last_committed != last_committed)
                or _utc_day(pending_day) is None
                or pending_day != last_day
                or _utc_datetime(last_bar) is None
                or _utc_datetime(last_bar) != _utc_datetime(last_window)
                or type(pending_close) is not float
                or not math.isfinite(pending_close)
                or pending_close <= self.min_price
                or (
                    entry_price is not None
                    and (
                        type(entry_price) is not float
                        or not math.isfinite(entry_price)
                        or entry_price <= self.min_price
                    )
                )
                or (score is not None and (type(score) is not float or not math.isfinite(score)))
                or (
                    mode == "OUT"
                    and (entry_price is not None or days_held != 0 or score is not None)
                )
                or (mode != "OUT" and (entry_price is None or score is None))
            ):
                return
            if common_days is None:
                common_days = days
                common_pending = pending_day
            elif days != common_days or pending_day != common_pending:
                return
            restored[symbol] = _State(
                daily_closes=deque(
                    closes,
                    maxlen=self._state[symbol].daily_closes.maxlen,
                ),
                daily_days=deque(days, maxlen=self._state[symbol].daily_days.maxlen),
                pending_day=pending_day,
                pending_close=pending_close,
                last_committed_day=item_last_committed,
                last_bar_key=last_bar,
                mode=mode,
                entry_price=entry_price,
                days_held=days_held,
                score=score,
            )
        self._last_day_key = last_day
        self._last_committed_day = last_committed
        self._last_decision_week = last_week
        self._last_window_key = last_window
        self._tick = tick
        self._state = restored

    # ------------------------------------------------------------------ #
    # ingestion / UTC-day + ISO-week clocks
    # ------------------------------------------------------------------ #
    def _day_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into a UTC ``YYYY-MM-DD`` day key."""
        dt = _utc_datetime(raw_time)
        return dt.date().isoformat() if dt is not None else ""

    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key."""
        dt = _utc_datetime(raw_time)
        if dt is None:
            return ""
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if not _strictly_newer_time(snapshot.time, item.last_bar_key):
            return False
        day = self._day_key(snapshot.time)
        if not day:
            return False
        item.last_bar_key = key
        # The latest close seen within a UTC day wins; the day commits only
        # when a bar from a LATER day proves it complete (completed-close only).
        item.pending_day = day
        item.pending_close = float(close)
        return True

    def _window_updates(self, event: Any) -> list[tuple[str, _Snapshot]] | None:
        """Validate a complete, common-time window before mutating any state."""
        event_time = getattr(event, "time", None)
        event_dt = _utc_datetime(event_time)
        event_key = time_key(event_time)
        if (
            event_dt is None
            or not event_key
            or not _strictly_newer_time(event_time, self._last_window_key)
        ):
            return None
        if len(_event_symbols(event, self.symbol_list)) != len(self.symbol_list):
            return None
        updates: list[tuple[str, _Snapshot]] = []
        for symbol in self.symbol_list:
            raw_rows = list((getattr(event, "bars_1s", {}) or {}).get(symbol) or [])
            row_times = [
                row.get("time")
                if isinstance(row, dict)
                else row[0]
                if isinstance(row, (tuple, list)) and row
                else None
                for row in raw_rows
            ]
            row_datetimes = [_utc_datetime(raw_time) for raw_time in row_times]
            snapshot = _window_snapshot(event, symbol)
            if (
                snapshot is None
                or not raw_rows
                or any(raw_time is None for raw_time in row_datetimes)
                or len(set(row_datetimes)) != len(raw_rows)
                or _utc_datetime(snapshot.time) != event_dt
                or safe_float(snapshot.close) is None
                or safe_float(snapshot.close) <= self.min_price
                or not _strictly_newer_time(snapshot.time, self._state[symbol].last_bar_key)
            ):
                return None
            updates.append((symbol, snapshot))
        return updates

    def _roll_day(self, new_day: str, event_time: Any) -> None:
        """Commit completed UTC days, age daily counters, sweep max-hold."""
        if not new_day or (self._last_day_key and new_day <= self._last_day_key):
            return
        committed = self._last_committed_day
        for item in self._state.values():
            if item.pending_close is not None and item.pending_day and item.pending_day < new_day:
                item.daily_closes.append(float(item.pending_close))
                item.daily_days.append(item.pending_day)
                item.last_committed_day = item.pending_day
                if item.pending_day > committed:
                    committed = item.pending_day
                item.pending_day = ""
                item.pending_close = None
        self._last_committed_day = committed
        for item in self._state.values():
            if item.mode in {"LONG", "SHORT"}:
                item.days_held += 1
        self._age_max_hold(event_time)

    def _handle_day(self, event_time: Any) -> None:
        day = self._day_key(event_time)
        if day and (not self._last_day_key or day > self._last_day_key):
            if self._last_day_key:
                self._roll_day(day, event_time)
            self._last_day_key = day

    def _maybe_decide(self, event_time: Any) -> None:
        week = self._week_key(event_time)
        if not week or week == self._last_decision_week:
            return
        self._last_decision_week = week
        self._tick += 1
        self._rebalance(event_time)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updates = self._window_updates(event)
        if updates is None:
            return
        self._last_window_key = time_key(getattr(event, "time", None))
        self._handle_day(getattr(event, "time", None))
        for symbol, snapshot in updates:
            self._update_symbol(symbol, snapshot)
        self._maybe_decide(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        # A single-symbol callback cannot prove a common-time panel.
        return

    # ------------------------------------------------------------------ #
    # scoring (characteristic -> momentum+MAX residualization -> fade)
    # ------------------------------------------------------------------ #
    def _formation_panel(self) -> dict[str, list[float]]:
        """Formation daily returns per eligible (fresh, long-enough) symbol.

        CALENDAR alignment: each symbol must contribute the SAME trailing
        ``formation_days + 1`` committed UTC day keys as the panel consensus
        (the most common trailing day tuple).  A symbol missing an interior
        UTC day would otherwise shift its whole formation window vs the
        market, so it is skipped; gap-free data passes trivially.
        """
        if not self._last_committed_day:
            return {}
        window = self.formation_days + 1
        if _utc_day(self._last_committed_day) is None:
            return {}
        candidates: dict[str, list[float]] = {}
        expected_days: tuple[str, ...] | None = None
        for symbol, item in self._state.items():
            if item.last_committed_day != self._last_committed_day:
                continue
            if len(item.daily_closes) < self._required_days:
                continue
            if len(item.daily_days) != len(item.daily_closes):
                continue
            returns = _daily_simple_returns(list(item.daily_closes))
            if len(returns) < self.formation_days:
                continue
            days = list(item.daily_days)[-window:]
            if len(days) != window or not _consecutive_days(days, ending=self._last_committed_day):
                continue
            day_tuple = tuple(days)
            if expected_days is None:
                expected_days = day_tuple
            elif day_tuple != expected_days:
                continue
            candidates[symbol] = returns[-self.formation_days :]
        return candidates

    def _score_symbols(self) -> dict[str, tuple[float, dict[str, Any]]]:
        panel = self._formation_panel()
        if len(panel) < self.min_symbols:
            return {}
        symbols = sorted(panel)
        # Equal-weight universe return per aligned formation day (self-inclusive).
        market = [
            sum(panel[symbol][day] for symbol in symbols) / float(len(symbols))
            for day in range(self.formation_days)
        ]
        raw_char: dict[str, float] = {}
        raw_mom: dict[str, float] = {}
        raw_max: dict[str, float] = {}
        for symbol in symbols:
            returns = panel[symbol]
            char = self._characteristic_value(returns, market)
            mom = _momentum_return(returns, self.momentum_days)
            max_ret = _max_daily_return(returns, self.max_days)
            if char is None or mom is None or max_ret is None:
                continue
            raw_char[symbol] = char
            raw_mom[symbol] = mom
            raw_max[symbol] = max_ret
        if len(raw_char) < self.min_symbols:
            return {}
        ordered = sorted(raw_char)
        z_char = _xs_zscores(raw_char)
        z_mom = _xs_zscores(raw_mom)
        z_max = _xs_zscores(raw_max)
        residual = cross_sectional_residualize(
            [z_char[symbol] for symbol in ordered],
            [
                [z_mom[symbol] for symbol in ordered],
                [z_max[symbol] for symbol in ordered],
            ],
        )
        if residual is None:
            return {}
        if len(residual) != len(ordered) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in residual
        ):
            return {}
        dispersion = sample_std([float(value) for value in residual])
        if dispersion is None or not math.isfinite(dispersion) or dispersion <= _EPS:
            return {}
        scored: dict[str, tuple[float, dict[str, Any]]] = {}
        for idx, symbol in enumerate(ordered):
            # FADE: high behavioral value -> overbought -> SHORT; the ranking
            # score is the NEGATED residual so the top-of-book is LONG.
            resid = float(residual[idx])
            scored[symbol] = (
                -resid,
                {
                    "behavioral_value": float(raw_char[symbol]),
                    "residual_value_z": resid,
                    "momentum_return": float(raw_mom[symbol]),
                    "max_daily_return": float(raw_max[symbol]),
                },
            )
        return scored

    # ------------------------------------------------------------------ #
    # selection (rank-band hysteresis + hard daily min-hold)
    # ------------------------------------------------------------------ #
    def _desired_book(self, scored: dict[str, tuple[float, dict[str, Any]]]) -> dict[str, str]:
        ordered = sorted(scored.items(), key=lambda kv: (kv[1][0], kv[0]), reverse=True)
        rank = {symbol: idx for idx, (symbol, _payload) in enumerate(ordered)}
        n = len(ordered)
        k_enter = max(1, int(n * self.quantile_entry_pct))
        k_exit = max(k_enter, int(n * self.quantile_exit_pct))
        desired: dict[str, str] = {}
        for symbol, item in self._state.items():
            idx = rank.get(symbol)
            cur = item.mode
            # Hard min-hold (daily decisions): freeze the side until it matures.
            if cur != "OUT" and item.days_held < self.min_hold_daily_decisions:
                desired[symbol] = cur
                continue
            if idx is None:
                desired[symbol] = "OUT"
                continue
            long_enter = idx < k_enter
            long_hold = idx < k_exit
            short_enter = idx >= n - k_enter
            short_hold = idx >= n - k_exit
            if cur == "LONG" and long_hold:
                desired[symbol] = "LONG"
            elif cur == "SHORT" and short_hold and self.allow_short:
                desired[symbol] = "SHORT"
            elif long_enter:
                desired[symbol] = "LONG"
            elif short_enter and self.allow_short:
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = "OUT"
        return desired

    def _inverse_vol_weights(self, book: list[str]) -> dict[str, float]:
        if not book:
            return {}
        vols: dict[str, float | None] = {}
        for symbol in book:
            vols[symbol] = realized_volatility(
                list(self._state[symbol].daily_closes), window=self.vol_window
            )
        valid = [1.0 / vol for vol in vols.values() if vol is not None and vol > _EPS]
        avg_inv = (sum(valid) / len(valid)) if valid else 1.0
        inv = {
            symbol: (1.0 / vol if (vol is not None and vol > _EPS) else avg_inv)
            for symbol, vol in vols.items()
        }
        total = sum(inv.values())
        if total <= _EPS:
            share = self.target_gross_exposure / float(len(book))
            return dict.fromkeys(book, share)
        return {symbol: self.target_gross_exposure * inv[symbol] / total for symbol in book}

    # ------------------------------------------------------------------ #
    # rebalance / emission
    # ------------------------------------------------------------------ #
    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        scored = self._score_symbols()
        if len(scored) < self.min_symbols:
            return
        desired = self._desired_book(scored)
        book = [symbol for symbol, side in desired.items() if side in {"LONG", "SHORT"}]
        weights = self._inverse_vol_weights(book)
        self._emit_book(desired, weights, scored, event_time)

    def _age_max_hold(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            if item.days_held >= self.max_hold_daily_decisions:
                price = self._reference_price(item)
                _emit(
                    self.events,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": self.strategy_name, "reason": "max_hold"},
                )
                self._flatten(item)

    def _reference_price(self, item: _State) -> float | None:
        if item.pending_close is not None:
            return item.pending_close
        return item.daily_closes[-1] if item.daily_closes else None

    def _emit_book(
        self,
        desired: dict[str, str],
        weights: dict[str, float],
        scored: dict[str, tuple[float, dict[str, Any]]],
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            side = desired.get(symbol, "OUT")
            price = self._reference_price(item)
            if side == "OUT":
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": self.strategy_name, "reason": "rank_lapsed"},
                    )
                    self._flatten(item)
                continue
            if item.mode == side:
                score, _meta = scored.get(symbol, (item.score or 0.0, {}))
                item.score = float(score)
                continue
            weight = max(0.0, float(weights.get(symbol, 0.0)))
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": self.strategy_name, "reason": "side_flip"},
                )
                self._flatten(item)
            # v5 zero-alloc trap: a zero/negative computed allocation emits
            # NOTHING (no zero-size entry, no phantom resize).
            if weight <= _EPS:
                continue
            score, meta = scored.get(symbol, (0.0, {}))
            metadata = _target_metadata(
                strategy=self.strategy_name,
                target_allocation=weight,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=side,
                intended_hold_seconds=int(self.intended_hold_seconds),
                **meta,
            )
            _emit(
                self.events,
                strategy_id=self.strategy_id,
                symbol=symbol,
                event_time=event_time,
                signal_type=side,
                strength=max(0.25, min(3.0, abs(float(score)))),
                price=price,
                metadata=metadata,
            )
            item.mode = side
            item.entry_price = price
            item.days_held = 0
            item.score = float(score)

    def _flatten(self, item: _State) -> None:
        item.mode = "OUT"
        item.entry_price = None
        item.days_held = 0
        item.score = None


@register("strategy", _ST_STRATEGY_NAME, interface="event_driven")
class SalienceTheoryValueStrategy(_WeeklyBehavioralValueXS):
    """Weekly XS fade of the BGS salience-theory value (co-timing distortion).

    SHORT the top-quintile ST names (overbought by salient-upside co-timing),
    LONG the bottom quintile, after residualizing ST against 21d momentum z
    and inline MAX z (see the module docstring).  All constants are the BGS
    canonical values (theta=0.1, delta=0.7, 21-UTC-day formation), frozen.

    REDUNDANCY TOURNAMENT (pre-registered): if ``|XS rank corr(TK, ST)| > 0.6``
    on the data-PC, only the lane with the higher orthogonal IC vs
    LotterySkewnessStrategy admits -- this sleeve and
    ``ProspectTheoryValueStrategy`` are a tournament, not two slots.
    """

    strategy_id = _ST_STRATEGY_ID
    strategy_name = _ST_STRATEGY_NAME
    intended_hold_seconds = _ST_INTENDED_HOLD_SECONDS

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = cls._shared_param_schema(formation_days_default=21)
        schema["theta"] = HyperParam.floating(
            "theta", default=0.1, low=0.001, high=10.0, tunable=False
        )
        schema["delta"] = HyperParam.floating(
            "delta", default=0.7, low=0.01, high=1.0, tunable=False
        )
        return schema

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.theta = float(resolved["theta"])
        self.delta = float(resolved["delta"])
        self._init_shared(bars, events, resolved)

    def _characteristic_value(
        self, returns: list[float], market_returns: list[float]
    ) -> float | None:
        return _salience_value(returns, market_returns, theta=self.theta, delta=self.delta)


@register("strategy", _PT_STRATEGY_NAME, interface="event_driven")
class ProspectTheoryValueStrategy(_WeeklyBehavioralValueXS):
    """Weekly XS fade of the TK-1992 prospect-theory value of the 90d histogram.

    SHORT the top-quintile TK names (distributions attractive to
    prospect-theory investors get overbought, Barberis-Jin-Wang), LONG the
    bottom quintile, after residualizing TK against 21d momentum z and inline
    MAX z (see the module docstring).  All constants are the TK-1992 canonical
    values (alpha=0.88, lam=2.25, gamma 0.61/0.69, 90-UTC-day formation),
    frozen.

    REDUNDANCY TOURNAMENT (pre-registered): if ``|XS rank corr(TK, ST)| > 0.6``
    on the data-PC, only the lane with the higher orthogonal IC vs
    LotterySkewnessStrategy admits -- this sleeve and
    ``SalienceTheoryValueStrategy`` are a tournament, not two slots.
    """

    strategy_id = _PT_STRATEGY_ID
    strategy_name = _PT_STRATEGY_NAME
    intended_hold_seconds = _PT_INTENDED_HOLD_SECONDS

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        schema = cls._shared_param_schema(formation_days_default=90)
        schema["max_hold_daily_decisions"] = HyperParam.integer(
            "max_hold_daily_decisions", default=42, low=1, high=200000, tunable=False
        )
        schema["alpha"] = HyperParam.floating(
            "alpha", default=0.88, low=0.05, high=1.0, tunable=False
        )
        schema["lam"] = HyperParam.floating("lam", default=2.25, low=0.0, high=20.0, tunable=False)
        schema["gamma_gain"] = HyperParam.floating(
            "gamma_gain", default=0.61, low=0.05, high=1.0, tunable=False
        )
        schema["gamma_loss"] = HyperParam.floating(
            "gamma_loss", default=0.69, low=0.05, high=1.0, tunable=False
        )
        return schema

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.alpha = float(resolved["alpha"])
        self.lam = float(resolved["lam"])
        self.gamma_gain = float(resolved["gamma_gain"])
        self.gamma_loss = float(resolved["gamma_loss"])
        self._init_shared(bars, events, resolved)

    def _characteristic_value(
        self, returns: list[float], market_returns: list[float]
    ) -> float | None:
        _ = market_returns
        return prospect_theory_value(
            returns,
            alpha=self.alpha,
            lam=self.lam,
            gamma_gain=self.gamma_gain,
            gamma_loss=self.gamma_loss,
        )


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- the orchestrator lands tier hints, BATCH wiring, and
# candidate builders).  Honest family: cross-sectional behavioral value,
# admitted via `allow_multi_asset=True`; NOT carry/momentum, so no fake tags
# are added to game the allocator superset route.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "behavioral_value",
    "salience_fade",
    "prospect_theory_fade",
    "lottery_adjacent",
    "low_turnover",
    "crypto",
)

# Candidate slices.  The sleeves sample ONE close per UTC day internally, so
# every formation window is calendar-defined and timeframe INVARIANT: the same
# pre-registered cells apply verbatim at 1d/4h/1h feeds (sub-daily bars only
# refine the day-completion detection, never the formation math).  Two cells
# per sleeve: the literature-canonical cell and its declared lower-churn
# min-hold sibling (the proven RPT rescue axis) -- no tunable grid.
_ST_CELLS: tuple[dict[str, Any], ...] = (
    {
        "variant": "bgs_canonical_21d_minhold5",
        "formation_days": 21,
        "theta": 0.1,
        "delta": 0.7,
        "quantile_entry_pct": 0.20,
        "quantile_exit_pct": 0.35,
        "min_hold_daily_decisions": 5,
        "min_symbols": 5,
        "allow_short": True,
        "target_gross_exposure": 1.0,
    },
    {
        "variant": "bgs_canonical_21d_minhold10",
        "formation_days": 21,
        "theta": 0.1,
        "delta": 0.7,
        "quantile_entry_pct": 0.20,
        "quantile_exit_pct": 0.35,
        "min_hold_daily_decisions": 10,
        "min_symbols": 5,
        "allow_short": True,
        "target_gross_exposure": 1.0,
    },
)
_SALIENCE_VALUE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": _ST_CELLS,
    "4h": _ST_CELLS,
    "1h": _ST_CELLS,
}

_PT_CELLS: tuple[dict[str, Any], ...] = (
    {
        "variant": "tk1992_canonical_90d_minhold5",
        "formation_days": 90,
        "alpha": 0.88,
        "lam": 2.25,
        "gamma_gain": 0.61,
        "gamma_loss": 0.69,
        "quantile_entry_pct": 0.20,
        "quantile_exit_pct": 0.35,
        "min_hold_daily_decisions": 5,
        "min_symbols": 5,
        "allow_short": True,
        "target_gross_exposure": 1.0,
    },
    {
        "variant": "tk1992_canonical_90d_minhold10",
        "formation_days": 90,
        "alpha": 0.88,
        "lam": 2.25,
        "gamma_gain": 0.61,
        "gamma_loss": 0.69,
        "quantile_entry_pct": 0.20,
        "quantile_exit_pct": 0.35,
        "min_hold_daily_decisions": 10,
        "min_symbols": 5,
        "allow_short": True,
        "target_gross_exposure": 1.0,
    },
)
_PROSPECT_VALUE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": _PT_CELLS,
    "4h": _PT_CELLS,
    "1h": _PT_CELLS,
}

__all__ = ["ProspectTheoryValueStrategy", "SalienceTheoryValueStrategy"]
