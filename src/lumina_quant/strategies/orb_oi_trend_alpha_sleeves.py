"""Session opening-range-breakout and open-interest-confirmed trend riders.

Two per-symbol single-asset, return-first DIRECTIONAL sleeves that subclass the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and add a fresh ENTRY edge.  Both are
wired ONLY at >=30m timeframes and stay ``candidate_mix_type=="single"`` so they
bypass the multi-asset admission gate.

- :class:`OpeningRangeBreakoutRiderStrategy`: OHLCV-only.  THEORY (Toby Crabel's
  Opening-Range Breakout; the 2023 Zarattini-Aziz "A Profitable Day Trading
  Strategy For The U.S. Equity Market" reports large risk-adjusted returns from a
  session ORB).  The novelty is a SESSION-ANCHORED opening range: the session is
  bucketed by the UTC calendar day (derived from the bar timestamp via the shared
  :func:`_session_key`/:func:`_event_datetime_utc` helpers); the opening-range
  HIGH/LOW are accumulated over the first ``opening_range_bars`` decision bars of
  each new UTC day (no trading during accumulation), then a breakout is ARMED —
  LONG when the close breaks above the opening-range high (plus an optional ATR
  buffer), SHORT (when ``allow_short``) when it breaks below the opening-range low,
  one-shot per session.  The winner is ridden with the inherited ATR trailing stop
  + pyramiding; the opening range RESETS each UTC day.  DISTINCT from
  :class:`VolatilityBreakoutRiderStrategy` (a ROLLING N-bar Donchian channel, no
  session anchor) and from :class:`VolatilitySqueezeBreakoutRiderStrategy` (which
  requires a prior volatility-CONTRACTION regime): here the breakout level is the
  day's first-k-bars extreme, reset daily, not a rolling window or a vol-state.

- :class:`OpenInterestTrendConfirmationRiderStrategy`: crypto-perp only
  (open-interest is a crypto-perp field).  THEORY (futures microstructure: the
  open-interest / price relationship).  A price trend backed by RISING open
  interest is fresh money committing (a healthy, persistent trend); a price rise
  on FALLING OI is short-covering (weak, fades).  The novelty is the OPEN-INTEREST
  CHANGE as the trend-confirmation gate (the OI/price quadrant): enter a
  directional ride ONLY when the trend sign and the OI change CONFIRM each other —
  LONG when the trend is up AND OI is rising over ``oi_lookback`` (new longs),
  SHORT (when ``allow_short``) when the trend is down AND OI is rising (new
  shorts); entries are SUPPRESSED when OI is falling (unwind / short-covering,
  low conviction).  The winner is ridden with the inherited ATR trailing stop +
  pyramiding.  OI is read from the perp feature points exactly like
  :class:`FundingHarvestCarryStrategy` reads funding, with ``None``-safety, and
  the sleeve degrades to NO entries when OI data is absent.  DISTINCT from
  :class:`CarryTrendConfluenceRiderStrategy` (which gates the trend on the FUNDING
  sign, i.e. carry, not OI), from :class:`FundingHarvestCarryStrategy` (which
  trades funding), and from the pure trend riders (which use no positioning data):
  the gate here is the OI/price quadrant.

Both: an ATR trailing stop is the primary return lever; a TopCap-scale
``target_allocation`` (``tunable=False``); vol-targeted sizing; a
``max_hold_bars`` cap; per-symbol bounded-deque state with pyramiding
bookkeeping; ``None``-safe graceful guards (never raise).
``decision_cadence_seconds`` defaults to 1800 (>=30m floor).  They are
data-local (no fetching, no artifact writes) and must still be validated on the
data-bearing machine.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _session_key,
    _Snapshot,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


@register("strategy", "OpeningRangeBreakoutRiderStrategy", interface="event_driven")
class OpeningRangeBreakoutRiderStrategy(_ReturnRiderBase):
    """Ride a SESSION-anchored opening-range breakout; the range resets each UTC day.

    Per symbol the sleeve buckets a SESSION by the UTC calendar day (derived from
    the bar timestamp).  For the first ``opening_range_bars`` decision bars of each
    new UTC day it accumulates the opening-range HIGH/LOW and does NOT trade.  Once
    the range is established it ARMS a one-shot breakout: a close above the
    opening-range high (plus ``buffer_atr_mult`` ATRs) opens a LONG; a close below
    the opening-range low (minus the buffer) opens a SHORT when ``allow_short``.
    The winner is ridden with the inherited ATR trailing stop + pyramiding (on
    continued favorable extension beyond the breakout level).  The opening range
    RESETS at each new UTC day.

    DISTINCT from :class:`VolatilityBreakoutRiderStrategy` (a ROLLING N-bar
    Donchian channel with no session anchor) and from
    :class:`VolatilitySqueezeBreakoutRiderStrategy` (which requires a prior
    volatility-CONTRACTION regime): the breakout level here is the SESSION opening
    range (the day's first-k-bars extremes), reset daily.  OHLCV-only, per-symbol
    single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "OpeningRangeBreakoutRiderStrategy"
    strategy_id = "opening_range_breakout_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "opening_range_bars": HyperParam.integer(
                "opening_range_bars", default=2, low=1, high=512
            ),
            "session_start_minute_utc": HyperParam.integer(
                "session_start_minute_utc", default=0, low=0, high=1439, tunable=False
            ),
            "buffer_atr_mult": HyperParam.floating(
                "buffer_atr_mult", default=0.0, low=0.0, high=20.0
            ),
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
        # Per-symbol SESSION state for the opening-range anchor.  The OHLC ring
        # buffers live on the base ``_RiderState``; this parallel record holds the
        # current UTC-day session key, the accumulated opening-range high/low, the
        # number of bars seen so far in the session, and whether a one-shot
        # breakout has already armed-and-fired this session.
        self._session_key: dict[str, str] = dict.fromkeys(self.symbol_list, "")
        self._or_high: dict[str, float | None] = dict.fromkeys(self.symbol_list)
        self._or_low: dict[str, float | None] = dict.fromkeys(self.symbol_list)
        self._session_bars: dict[str, int] = dict.fromkeys(self.symbol_list, 0)
        self._session_fired: dict[str, bool] = dict.fromkeys(self.symbol_list, False)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.opening_range_bars = max(1, int(resolved["opening_range_bars"]))
        self.session_start_minute_utc = max(0, min(1439, int(resolved["session_start_minute_utc"])))
        self.buffer_atr_mult = max(0.0, float(resolved["buffer_atr_mult"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=int(resolved["opening_range_bars"]) + 2,
        )

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Refresh the SESSION opening-range record BEFORE the base advances the
        # OHLC ring buffers / runs ride+entry, so _entry_decision sees the freshest
        # opening-range high/low for this bar.  Mirror the base time-key dedup so a
        # repeated boundary firing never double-counts a session bar.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._refresh_session(symbol, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _refresh_session(self, symbol: str, snapshot: _Snapshot) -> None:
        """Bucket the bar into its UTC-day session and accumulate the opening range.

        A new UTC-day session RESETS the opening-range high/low and the one-shot
        ``fired`` latch.  During the first ``opening_range_bars`` bars of a session
        the high/low extremes are accumulated (no trading).  ``None``-safe: a bar
        with no parseable timestamp or a degenerate close leaves the record intact.
        """
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        key = _session_key(snapshot.time, start_minute_utc=self.session_start_minute_utc)
        if not key:
            return
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if high is None:
            high = close
        if low is None:
            low = close
        if key != self._session_key.get(symbol, ""):
            # New UTC-day session: reset the opening range and the one-shot latch.
            self._session_key[symbol] = key
            self._or_high[symbol] = high
            self._or_low[symbol] = low
            self._session_bars[symbol] = 1
            self._session_fired[symbol] = False
            return
        if self._session_bars.get(symbol, 0) < self.opening_range_bars:
            cur_high = self._or_high.get(symbol)
            cur_low = self._or_low.get(symbol)
            self._or_high[symbol] = max(float(cur_high if cur_high is not None else high), high)
            self._or_low[symbol] = min(float(cur_low if cur_low is not None else low), low)
        self._session_bars[symbol] = self._session_bars.get(symbol, 0) + 1

    def _range_established(self, symbol: str) -> bool:
        """Return whether the opening range has finished accumulating this session."""
        return (
            self._session_bars.get(symbol, 0) > self.opening_range_bars
            and self._or_high.get(symbol) is not None
            and self._or_low.get(symbol) is not None
        )

    def _breakout_direction(self, symbol: str, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` for a one-shot opening-range break."""
        if self._session_fired.get(symbol, False) or not self._range_established(symbol):
            return ""
        or_high = self._or_high.get(symbol)
        or_low = self._or_low.get(symbol)
        if or_high is None or or_low is None:
            return ""
        closes = list(item.closes)
        if not closes:
            return ""
        close = closes[-1]
        atr = self._current_atr(item)
        buffer = (atr * self.buffer_atr_mult) if (atr is not None and atr > _EPS) else 0.0
        if close >= or_high + buffer:
            return "LONG"
        if close <= or_low - buffer:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        direction = self._breakout_direction(symbol, item)
        if direction in {"LONG", "SHORT"}:
            # One-shot per session: latch so a fresh break is required next session
            # (and re-entry within the same session does not churn).
            self._session_fired[symbol] = True
        return direction

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Keep compounding only while price holds beyond the opening-range level in
        # our direction; the base ATR add-step gate already enforces favorable
        # extension.  Re-arm is suppressed because the position is non-flat.
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        or_high = self._or_high.get(symbol)
        or_low = self._or_low.get(symbol)
        closes = list(item.closes)
        if not closes:
            return False
        close = closes[-1]
        if item.mode == "LONG":
            return or_high is not None and close >= or_high
        return or_low is not None and close <= or_low

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        or_high = self._or_high.get(symbol) if symbol is not None else None
        or_low = self._or_low.get(symbol) if symbol is not None else None
        return {
            "session_key": str(self._session_key.get(symbol, "")) if symbol is not None else "",
            "opening_range_high": float(or_high) if or_high is not None else None,
            "opening_range_low": float(or_low) if or_low is not None else None,
            "opening_range_bars": int(self.opening_range_bars),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["session_state"] = {
            symbol: {
                "session_key": str(self._session_key.get(symbol, "")),
                "or_high": self._or_high.get(symbol),
                "or_low": self._or_low.get(symbol),
                "session_bars": int(self._session_bars.get(symbol, 0)),
                "session_fired": bool(self._session_fired.get(symbol, False)),
            }
            for symbol in self._state
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("session_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            self._session_key[symbol] = str(payload.get("session_key", ""))
            self._or_high[symbol] = safe_float(payload.get("or_high"))
            self._or_low[symbol] = safe_float(payload.get("or_low"))
            bars = payload.get("session_bars")
            try:
                self._session_bars[symbol] = max(0, int(bars))
            except Exception:
                self._session_bars[symbol] = 0
            self._session_fired[symbol] = bool(payload.get("session_fired", False))


@register("strategy", "OpenInterestTrendConfirmationRiderStrategy", interface="event_driven")
class OpenInterestTrendConfirmationRiderStrategy(_ReturnRiderBase):
    """Ride a trend ONLY when rising open interest CONFIRMS the trend (fresh money).

    Per symbol the sleeve reads the perp ``open_interest`` feature (own 3-arg
    :meth:`_extract_feature`, identical cascade and ``None``-safety to
    :class:`FundingHarvestCarryStrategy`) into a trailing record, and measures the
    price trend from a long-horizon ROC plus a trend-MA confirmation.  The
    OI/price-quadrant gate is the edge:

    - LONG only when the trend is UP **and** open interest is RISING over
      ``oi_lookback`` (new longs committing — a healthy, persistent up-trend).
    - SHORT (when ``allow_short``) only when the trend is DOWN **and** open
      interest is RISING (new shorts committing).
    - Entries are SUPPRESSED whenever OI is FALLING (position unwind /
      short-covering — a low-conviction move that tends to fade).

    The winner is ridden with the inherited ATR trailing stop and pyramided while
    the trend-and-OI confirmation persists.  When OI data is absent the sleeve
    degrades gracefully (no entries), mirroring the carry-harvest fallback.

    DISTINCT from :class:`CarryTrendConfluenceRiderStrategy` (which gates the trend
    on the FUNDING/carry sign, not OI), from :class:`FundingHarvestCarryStrategy`
    (which trades funding), and from the pure trend riders (which use no
    positioning data): the novelty is the open-interest CHANGE — the OI/price
    quadrant — as the trend-confirmation gate.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "OpenInterestTrendConfirmationRiderStrategy"
    strategy_id = "open_interest_trend_confirmation_rider"
    required_features = ("open_interest",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_lookback": HyperParam.integer("trend_lookback", default=48, low=3, high=20000),
            "trend_ma_window": HyperParam.integer("trend_ma_window", default=48, low=2, high=8192),
            "min_trend_roc": HyperParam.floating("min_trend_roc", default=0.0, low=0.0, high=1.0),
            "oi_lookback": HyperParam.integer("oi_lookback", default=8, low=1, high=4096),
            "oi_rise_threshold": HyperParam.floating(
                "oi_rise_threshold", default=0.0, low=0.0, high=1.0
            ),
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
        size = max(self.oi_lookback, 1) + 8
        self._open_interest: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))
        self.oi_lookback = max(1, int(resolved["oi_lookback"]))
        self.oi_rise_threshold = max(0.0, float(resolved["oi_rise_threshold"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.trend_lookback, self.trend_ma_window) + 2,
        )

    def _extract_feature(self, symbol: str, field: str) -> float | None:
        """Resolve a per-symbol feature value (here: ``open_interest``).

        Tries the bars feature/value getters in turn; returns ``None`` (skip, never
        raise) when no source provides a finite value.  Own 3-arg signature
        mirroring :class:`FundingHarvestCarryStrategy`.
        """
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                parsed = safe_float(getter(symbol, field))
            except Exception:
                parsed = None
            if parsed is not None:
                return parsed
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                return safe_float(getter(symbol, field))
            except Exception:
                return None
        return None

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Capture the latest open-interest observation for this accepted bar BEFORE
        # the base advances the OHLC ring buffers / runs ride+entry, so the OI gate
        # sees current open interest.  Dedup matches the base time-key guard.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                oi = self._extract_feature(symbol, "open_interest")
                if oi is not None:
                    self._open_interest.setdefault(
                        symbol, deque(maxlen=max(self.oi_lookback, 1) + 8)
                    ).append(float(oi))
        super()._process_symbol(symbol, snapshot)

    def _oi_rising(self, symbol: str) -> bool:
        """Return whether open interest is RISING over the trailing ``oi_lookback``.

        Requires a full ``oi_lookback + 1`` window so the change is measured over
        the intended horizon; returns ``False`` (suppress entries) when the OI
        record is too short or the prior reference is non-positive.
        """
        history = self._open_interest.get(symbol)
        if not history or len(history) < self.oi_lookback + 1:
            return False
        window = list(history)
        prev = window[-(self.oi_lookback + 1)]
        latest = window[-1]
        if prev <= _EPS:
            return False
        return (latest - prev) / prev > self.oi_rise_threshold

    def _trend_sign(self, closes: list[float]) -> int:
        """Return ``+1``/``-1``/``0`` for the prevailing long-horizon trend.

        Requires the long-horizon ROC and the close-vs-trend-MA relationship to
        agree, so a flat/whipsaw regime yields ``0`` and blocks entry.
        """
        roc = rate_of_change(closes, period=self.trend_lookback)
        if roc is None or abs(roc) < self.min_trend_roc:
            return 0
        ma = simple_moving_average(closes, self.trend_ma_window)
        if ma is None or not closes:
            return 0
        close = closes[-1]
        if roc > 0.0 and close >= ma:
            return 1
        if roc < 0.0 and close <= ma:
            return -1
        return 0

    def _confirmed_direction(self, symbol: str, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` from the trend + rising-OI confirm."""
        if not self._oi_rising(symbol):
            # OI falling/unknown -> unwind/short-covering -> low conviction -> skip.
            return ""
        trend = self._trend_sign(list(item.closes))
        if trend > 0:
            return "LONG"
        if trend < 0:
            return "SHORT"
        return ""

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._confirmed_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Keep compounding only while the trend-and-rising-OI confirmation still
        # points our way; the base ATR add-step gate already enforces favorable
        # extension.
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        direction = self._confirmed_direction(symbol, item)
        return direction == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        history = self._open_interest.get(symbol) if symbol is not None else None
        latest_oi = history[-1] if history else None
        trend = self._trend_sign(list(item.closes))
        return {
            "open_interest": float(latest_oi) if latest_oi is not None else None,
            "oi_rising": bool(self._oi_rising(symbol)) if symbol is not None else False,
            "trend_sign": int(trend),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["open_interest"] = {
            symbol: list(history) for symbol, history in self._open_interest.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("open_interest") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            target = self._open_interest.get(symbol)
            if target is None:
                continue
            target.clear()
            for value in list(payload or [])[-int(target.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)


__all__ = [
    "OpenInterestTrendConfirmationRiderStrategy",
    "OpeningRangeBreakoutRiderStrategy",
]
