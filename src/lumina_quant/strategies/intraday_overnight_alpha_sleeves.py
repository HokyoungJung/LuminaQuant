"""Intraday time-of-day and overnight-session return-structure riders.

Two per-symbol single-asset, return-first DIRECTIONAL sleeves that subclass the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and add a fresh TIME-STRUCTURE entry
edge that the existing registry does not cover (the only seasonal incumbent,
:class:`CalendarSeasonalityOverlayStrategy`, is DAY-of-week/month level).  Both
are wired ONLY at >=30m timeframes and stay ``candidate_mix_type=="single"`` so
they bypass the multi-asset admission gate.

- :class:`IntradaySeasonalMomentumRiderStrategy`: OHLCV-only.  THEORY (intraday
  periodicity / time-of-day drift: Aleti-Bollerslev intraday periodicity;
  documented crypto hour-of-day effects).  Each decision bar is bucketed into a
  UTC time-of-day SLOT (derived from the bar timestamp).  An ONLINE,
  decay-weighted estimate of each slot's mean return and dispersion is maintained
  and is updated with a ONE-BAR DEFERRAL so the bar being decided is never in its
  own slot statistic (no look-ahead).  A directional ride is ARMED only when the
  current slot has enough history AND a statistically meaningful drift AND that
  drift ALIGNS with the prevailing price trend — i.e. a trend ride CONDITIONED on
  a favorable intraday slot, NEVER a seasonal bet alone.  DISTINCT from
  :class:`CalendarSeasonalityOverlayStrategy` (day/month level) and from the plain
  trend riders: an unfavorable / insufficient-history slot SUPPRESSES an entry an
  aligned trend would otherwise take.

- :class:`OvernightSessionReturnRiderStrategy`: OHLCV-only.  THEORY (the
  overnight-vs-active-session return anomaly: Cooper-Cliff-Gulen overnight
  returns; Lou-Polk-Skouras "A Tug of War" — a large, robust share of total
  return historically accrues in specific low-activity "overnight" windows while
  active windows are flatter or negative).  The UTC day is partitioned into an
  "overnight" session (a configurable UTC-hour window, midnight-wrap aware) and
  the complementary "active" session; an ONLINE decay-weighted mean return is
  tracked per session (same one-bar-deferred, look-ahead-free update).  The sleeve
  STRUCTURALLY tilts LONG while the current session's historical mean return is
  significantly POSITIVE (and SHORT when significantly negative + ``allow_short``),
  and SUPPRESSES entries otherwise — a session-return harvester that is NOT
  trend-conditioned.  This is the key separation from the intraday-seasonal sleeve
  (which requires trend alignment + a strong slot) and from
  :class:`SessionFilteredPairCarryStrategy` (a pairs carry filtered by session, not
  a single-asset session-return tilt).  Exits are managed by the inherited
  trailing stop / ``max_hold_bars`` (there is no forced flat hook); the negative
  session is expressed by suppressed entries.

Both: an ATR trailing stop is the primary return lever; a TopCap-scale
``target_allocation`` (``tunable=False``); vol-targeted sizing; a
``max_hold_bars`` cap; per-symbol bounded state with pyramiding bookkeeping;
``None``-safe graceful guards (never raise).  ``decision_cadence_seconds``
defaults to 1800 (>=30m floor).  They are data-local (no fetching, no artifact
writes) and must still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _event_datetime_utc,
    _Snapshot,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam

# An online estimate of a bucket's return is held as a ``(mean, var, n)`` tuple.
_EWStat = tuple[float, float, int]


def _ew_update(stats: dict[Any, _EWStat], key: Any, ret: float, decay: float) -> None:
    """Fold one realized return into a bucket's decay-weighted ``(mean, var, n)``.

    Uses the standard incremental exponentially-weighted mean/variance recursion
    (``mean += decay*delta``; ``var = (1-decay)*(var + decay*delta^2)``).  The
    count ``n`` is the raw number of observations; significance gating caps the
    effective sample size at the EW half-life so a long history does not make
    every slot spuriously "significant".
    """
    rec = stats.get(key)
    if rec is None:
        stats[key] = (ret, 0.0, 1)
        return
    mean, var, n = rec
    delta = ret - mean
    mean += decay * delta
    var = (1.0 - decay) * (var + decay * delta * delta)
    stats[key] = (mean, var, n + 1)


def _bucket_signal(
    stats: dict[Any, _EWStat], key: Any, *, min_obs: int, t_threshold: float, decay: float
) -> int:
    """Return ``+1``/``-1``/``0`` for a bucket's significant decay-weighted drift.

    ``0`` (no signal) when the bucket has fewer than ``min_obs`` observations or
    the pseudo t-statistic ``|mean| / std * sqrt(eff_n)`` is below ``t_threshold``.
    ``eff_n`` is capped at the EW effective sample size ``1/decay`` so an
    arbitrarily long history cannot manufacture significance.  A near-zero
    dispersion with a non-zero mean is treated as significant in the mean's sign.
    """
    rec = stats.get(key)
    if rec is None:
        return 0
    mean, var, n = rec
    if n < min_obs:
        return 0
    eff_n = min(float(n), 1.0 / decay if decay > _EPS else float(n))
    std = math.sqrt(var) if var > 0.0 else 0.0
    if std <= _EPS:
        return 1 if mean > 0.0 else (-1 if mean < 0.0 else 0)
    t_stat = abs(mean) / std * math.sqrt(max(eff_n, 1.0))
    if t_stat < t_threshold:
        return 0
    return 1 if mean > 0.0 else (-1 if mean < 0.0 else 0)


def _trend_sign(
    closes: list[float], *, trend_lookback: int, trend_ma_window: int, min_trend_roc: float
) -> int:
    """Return ``+1``/``-1``/``0`` for the prevailing long-horizon trend.

    Requires the long-horizon ROC and the close-vs-trend-MA relationship to agree,
    so a flat/whipsaw regime yields ``0`` and blocks entry.
    """
    roc = rate_of_change(closes, period=trend_lookback)
    if roc is None or abs(roc) < min_trend_roc:
        return 0
    ma = simple_moving_average(closes, trend_ma_window)
    if ma is None or not closes:
        return 0
    close = closes[-1]
    if roc > 0.0 and close >= ma:
        return 1
    if roc < 0.0 and close <= ma:
        return -1
    return 0


@register("strategy", "IntradaySeasonalMomentumRiderStrategy", interface="event_driven")
class IntradaySeasonalMomentumRiderStrategy(_ReturnRiderBase):
    """Ride a trend ONLY in a UTC time-of-day slot with significant aligned drift.

    Each decision bar is bucketed into a UTC time-of-day slot of ``slot_minutes``
    width.  A decay-weighted ``(mean, var, n)`` estimate per slot is maintained
    with a one-bar deferral (the bar being decided is never folded into its own
    slot before the decision, so there is no look-ahead).  A LONG/SHORT ride is
    armed only when the current slot has ``>= min_slot_observations`` history, a
    pseudo t-statistic ``>= slot_t_threshold``, AND the slot's drift sign ALIGNS
    with the prevailing price trend; otherwise the entry is SUPPRESSED.  The winner
    is ridden with the inherited ATR trailing stop + pyramiding.

    DISTINCT from :class:`CalendarSeasonalityOverlayStrategy` (DAY-of-week/month):
    this is a sub-day INTRADAY slot used as a CONDITIONING gate on a trend ride, so
    a strong trend in an unfavorable / insufficient-history slot does NOT trade.
    OHLCV-only, per-symbol single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "IntradaySeasonalMomentumRiderStrategy"
    strategy_id = "intraday_seasonal_momentum_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "slot_minutes": HyperParam.integer(
                "slot_minutes", default=60, low=1, high=1440, tunable=False
            ),
            "seasonal_decay": HyperParam.floating(
                "seasonal_decay", default=0.05, low=0.001, high=1.0
            ),
            "min_slot_observations": HyperParam.integer(
                "min_slot_observations", default=10, low=1, high=100000
            ),
            "slot_t_threshold": HyperParam.floating(
                "slot_t_threshold", default=1.5, low=0.0, high=20.0
            ),
            "trend_lookback": HyperParam.integer("trend_lookback", default=48, low=3, high=20000),
            "trend_ma_window": HyperParam.integer("trend_ma_window", default=48, low=2, high=8192),
            "min_trend_roc": HyperParam.floating("min_trend_roc", default=0.0, low=0.0, high=1.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        # Per-symbol online slot statistics and the slot of the bar that is being
        # decided this tick (also the slot whose return is folded next tick).
        self._slot_stats: dict[str, dict[int, _EWStat]] = {
            symbol: {} for symbol in self.symbol_list
        }
        self._decision_slot: dict[str, int | None] = dict.fromkeys(self.symbol_list)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.slot_minutes = max(1, min(1440, int(resolved["slot_minutes"])))
        self.seasonal_decay = max(0.001, min(1.0, float(resolved["seasonal_decay"])))
        self.min_slot_observations = max(1, int(resolved["min_slot_observations"]))
        self.slot_t_threshold = max(0.0, float(resolved["slot_t_threshold"]))
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(int(resolved["trend_lookback"]), int(resolved["trend_ma_window"])) + 2,
        )

    def _slot_of(self, snapshot: _Snapshot) -> int | None:
        dt = _event_datetime_utc(snapshot.time)
        if dt is None:
            return None
        return (dt.hour * 60 + dt.minute) // self.slot_minutes

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Refresh the slot statistics BEFORE the base advances the OHLC ring /
        # runs ride+entry, mirroring the base time-key dedup.  At this point
        # ``item.closes`` holds bars up to the PREVIOUS one (the base appends the
        # live close afterwards), so folding closes[-1]/closes[-2] attributes the
        # PREVIOUS bar's return to the PREVIOUS bar's slot — the bar being decided
        # is therefore never in its own slot statistic (no look-ahead).
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._refresh_slot(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _refresh_slot(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        closes = list(item.closes)
        prev_slot = self._decision_slot.get(symbol)
        if prev_slot is not None and len(closes) >= 2 and closes[-2] > _EPS:
            ret = closes[-1] / closes[-2] - 1.0
            if math.isfinite(ret):
                _ew_update(
                    self._slot_stats.setdefault(symbol, {}), prev_slot, ret, self.seasonal_decay
                )
        self._decision_slot[symbol] = self._slot_of(snapshot)

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _aligned_direction(self, symbol: str, item: _RiderState) -> str:
        slot = self._decision_slot.get(symbol)
        if slot is None:
            return ""
        sig = _bucket_signal(
            self._slot_stats.get(symbol, {}),
            slot,
            min_obs=self.min_slot_observations,
            t_threshold=self.slot_t_threshold,
            decay=self.seasonal_decay,
        )
        if sig == 0:
            # Insufficient history or an insignificant slot drift -> no ride.
            return ""
        trend = _trend_sign(
            list(item.closes),
            trend_lookback=self.trend_lookback,
            trend_ma_window=self.trend_ma_window,
            min_trend_roc=self.min_trend_roc,
        )
        if trend == 0 or trend != sig:
            # The favorable slot must AGREE with the prevailing trend.
            return ""
        return "LONG" if sig > 0 else "SHORT"

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._aligned_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._aligned_direction(symbol, item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        slot = self._decision_slot.get(symbol) if symbol is not None else None
        rec = (
            self._slot_stats.get(symbol, {}).get(slot)
            if (symbol is not None and slot is not None)
            else None
        )
        return {
            "slot": int(slot) if slot is not None else None,
            "slot_mean_return": float(rec[0]) if rec is not None else None,
            "slot_observations": int(rec[2]) if rec is not None else 0,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["seasonal_state"] = {
            symbol: {
                "slot_stats": {
                    int(k): [float(v[0]), float(v[1]), int(v[2])] for k, v in stats.items()
                },
                "decision_slot": self._decision_slot.get(symbol),
            }
            for symbol, stats in self._slot_stats.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("seasonal_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._slot_stats or not isinstance(payload, dict):
                continue
            stats: dict[int, _EWStat] = {}
            for k, v in (payload.get("slot_stats") or {}).items():
                try:
                    seq = list(v)
                    stats[int(k)] = (float(seq[0]), float(seq[1]), int(seq[2]))
                except Exception:
                    continue
            self._slot_stats[symbol] = stats
            slot = payload.get("decision_slot")
            try:
                self._decision_slot[symbol] = int(slot) if slot is not None else None
            except Exception:
                self._decision_slot[symbol] = None


@register("strategy", "OvernightSessionReturnRiderStrategy", interface="event_driven")
class OvernightSessionReturnRiderStrategy(_ReturnRiderBase):
    """Tilt LONG during the UTC session whose historical mean return is positive.

    The UTC day is split into an "overnight" session — the half-open UTC-hour
    window ``[overnight_start_hour_utc, overnight_end_hour_utc)`` (midnight-wrap
    aware) — and the complementary "active" session.  A decay-weighted mean return
    per session is maintained with a one-bar deferral (no look-ahead).  The sleeve
    enters LONG while the CURRENT session's mean return is significantly POSITIVE,
    SHORT (when ``allow_short``) when significantly negative, and SUPPRESSES entry
    otherwise.  This is a STRUCTURAL session-return harvester, NOT trend-conditioned
    — the separation from :class:`IntradaySeasonalMomentumRiderStrategy` (which
    needs a strong intraday slot AND trend alignment).  Exits are managed by the
    inherited ATR trailing stop / ``max_hold_bars``; the negative session is
    expressed by suppressed entries.

    DISTINCT from :class:`CalendarSeasonalityOverlayStrategy` (day/month level) and
    :class:`SessionFilteredPairCarryStrategy` (a pairs carry filtered by session).
    OHLCV-only, per-symbol single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "OvernightSessionReturnRiderStrategy"
    strategy_id = "overnight_session_return_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "overnight_start_hour_utc": HyperParam.integer(
                "overnight_start_hour_utc", default=0, low=0, high=23, tunable=False
            ),
            "overnight_end_hour_utc": HyperParam.integer(
                "overnight_end_hour_utc", default=8, low=1, high=24, tunable=False
            ),
            "session_decay": HyperParam.floating(
                "session_decay", default=0.02, low=0.001, high=1.0
            ),
            "min_session_observations": HyperParam.integer(
                "min_session_observations", default=10, low=1, high=100000
            ),
            "session_t_threshold": HyperParam.floating(
                "session_t_threshold", default=1.5, low=0.0, high=20.0
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=48, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        self._session_stats: dict[str, dict[str, _EWStat]] = {
            symbol: {} for symbol in self.symbol_list
        }
        self._decision_session: dict[str, str | None] = dict.fromkeys(self.symbol_list)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.overnight_start_hour_utc = max(0, min(23, int(resolved["overnight_start_hour_utc"])))
        self.overnight_end_hour_utc = max(1, min(24, int(resolved["overnight_end_hour_utc"])))
        self.session_decay = max(0.001, min(1.0, float(resolved["session_decay"])))
        self.min_session_observations = max(1, int(resolved["min_session_observations"]))
        self.session_t_threshold = max(0.0, float(resolved["session_t_threshold"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=int(resolved["vol_window"]) + 2)

    def _session_of(self, snapshot: _Snapshot) -> str | None:
        dt = _event_datetime_utc(snapshot.time)
        if dt is None:
            return None
        hour = dt.hour
        start = self.overnight_start_hour_utc
        end = self.overnight_end_hour_utc
        if start < end:
            overnight = start <= hour < end
        else:
            # Window wraps past midnight (e.g. 22:00 -> 04:00).
            overnight = hour >= start or hour < end
        return "overnight" if overnight else "active"

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Refresh the session statistics BEFORE the base entry, with the same
        # one-bar-deferred, look-ahead-free fold as the intraday-seasonal sleeve.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._refresh_session(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _refresh_session(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        closes = list(item.closes)
        prev_session = self._decision_session.get(symbol)
        if prev_session is not None and len(closes) >= 2 and closes[-2] > _EPS:
            ret = closes[-1] / closes[-2] - 1.0
            if math.isfinite(ret):
                _ew_update(
                    self._session_stats.setdefault(symbol, {}),
                    prev_session,
                    ret,
                    self.session_decay,
                )
        self._decision_session[symbol] = self._session_of(snapshot)

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _session_direction(self, symbol: str) -> str:
        session = self._decision_session.get(symbol)
        if session is None:
            return ""
        sig = _bucket_signal(
            self._session_stats.get(symbol, {}),
            session,
            min_obs=self.min_session_observations,
            t_threshold=self.session_t_threshold,
            decay=self.session_decay,
        )
        if sig > 0:
            return "LONG"
        if sig < 0:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._session_direction(symbol)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._session_direction(symbol) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        session = self._decision_session.get(symbol) if symbol is not None else None
        rec = (
            self._session_stats.get(symbol, {}).get(session)
            if (symbol is not None and session is not None)
            else None
        )
        return {
            "session": session,
            "session_mean_return": float(rec[0]) if rec is not None else None,
            "session_observations": int(rec[2]) if rec is not None else 0,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["session_return_state"] = {
            symbol: {
                "session_stats": {
                    str(k): [float(v[0]), float(v[1]), int(v[2])] for k, v in stats.items()
                },
                "decision_session": self._decision_session.get(symbol),
            }
            for symbol, stats in self._session_stats.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("session_return_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._session_stats or not isinstance(payload, dict):
                continue
            stats: dict[str, _EWStat] = {}
            for k, v in (payload.get("session_stats") or {}).items():
                try:
                    seq = list(v)
                    stats[str(k)] = (float(seq[0]), float(seq[1]), int(seq[2]))
                except Exception:
                    continue
            self._session_stats[symbol] = stats
            session = payload.get("decision_session")
            self._decision_session[symbol] = str(session) if session is not None else None


__all__ = [
    "IntradaySeasonalMomentumRiderStrategy",
    "OvernightSessionReturnRiderStrategy",
]
