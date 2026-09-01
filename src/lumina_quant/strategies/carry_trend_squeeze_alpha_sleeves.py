"""Carry-trend-confluence and volatility-squeeze-breakout directional riders.

Two per-symbol single-asset, return-first DIRECTIONAL sleeves that subclass the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and add a fresh ENTRY edge.  Both are
wired ONLY at >=30m timeframes and stay ``candidate_mix_type=="single"`` so they
bypass the multi-asset admission gate.

- :class:`CarryTrendConfluenceRiderStrategy`: crypto-perp only (funding is a
  crypto-perp field).  THEORY (Koijen-Moskowitz-Pedersen-Vrij "Carry"; Asness-
  Moskowitz-Pedersen "Value and Momentum Everywhere" — carry and momentum are
  complementary, partially-independent return sources).  The novelty is a
  BOTH-AGREE confluence gate: only take a directional ride when the TREND sign
  and the CARRY sign AGREE.  A long perp RECEIVES funding when funding < 0, so a
  LONG is only opened when the trend is up (price above a trend MA / positive
  momentum) AND funding sits at/below ``long_carry_funding`` (carry not bleeding
  the long — ideally paying it).  A SHORT (when ``allow_short``) is only opened
  when the trend is down AND funding is high-positive (a short receives funding).
  The winner is then ridden with the inherited ATR trailing stop + pyramiding.
  DISTINCT from :class:`FundingHarvestCarryStrategy` (harvests the persistent
  funding sign REGARDLESS of trend — it can fight the trend) and from the pure
  trend riders (which ignore funding cost and get bled by negative carry in
  crowded longs): here the carry tailwind FILTERS the trend ride.  Degrades
  gracefully (no entries) when funding data is absent, mirroring
  ``FundingHarvestCarry``'s fallback.

- :class:`VolatilitySqueezeBreakoutRiderStrategy`: OHLCV-only.  THEORY
  (volatility clustering / the contraction-expansion cycle; Bollinger squeeze;
  TTM squeeze by John Carter).  The novelty is a volatility-CONTRACTION
  precondition gating a directional breakout continuation ride: it first detects
  a low-vol SQUEEZE regime (Bollinger Bandwidth at a multi-bar percentile low,
  optionally Bollinger-inside-Keltner), then on the volatility EXPANSION + range
  break enters in the BREAKOUT direction and rides with the inherited ATR
  trailing stop + pyramiding.  DISTINCT from
  :class:`VWAPCompressionReversionStrategy` (vol compression -> MEAN-REVERSION,
  opposite polarity — fades the move) and from
  :class:`VolatilityBreakoutRiderStrategy` (a Donchian N-bar channel breakout NOT
  conditioned on a prior volatility-contraction state): here the squeeze regime
  is a required precondition for the breakout continuation.

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
from lumina_quant.indicators.bands import (
    bollinger_bands,
    donchian_channel,
    keltner_channel,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.indicators.oscillators import percentile_rank, rate_of_change
from lumina_quant.indicators.volatility import bollinger_bandwidth
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


@register("strategy", "CarryTrendConfluenceRiderStrategy", interface="event_driven")
class CarryTrendConfluenceRiderStrategy(_ReturnRiderBase):
    """Ride a trend ONLY when the carry (funding) sign agrees with the trend sign.

    Per symbol the sleeve reads the perp ``funding_rate`` (own
    :meth:`_extract_feature`, identical cascade and ``None``-safety to
    :class:`FundingHarvestCarryStrategy`) into a trailing-average carry sign, and
    measures the price trend from a long-horizon ROC plus a trend-MA confirmation.
    The BOTH-AGREE confluence gate is the edge:

    - LONG only when the trend is UP **and** the trailing funding is at/below
      ``long_carry_funding`` (a long perp receives funding when funding < 0, so a
      low/negative funding means the carry is a tailwind, not a bleed).
    - SHORT (when ``allow_short``) only when the trend is DOWN **and** the trailing
      funding is at/above ``short_carry_funding`` (a short receives funding when
      funding is high-positive).

    The winner is ridden with the inherited ATR trailing stop and pyramided while
    the trend-and-carry confluence persists.  When funding data is absent the
    sleeve degrades gracefully (no entries), mirroring the carry-harvest fallback.

    DISTINCT from :class:`FundingHarvestCarryStrategy` (which harvests the funding
    sign regardless of trend and can fight the trend) and from the pure trend
    riders (which ignore funding cost): the novelty is the confluence filter that
    only rides trends with a carry tailwind.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "CarryTrendConfluenceRiderStrategy"
    strategy_id = "carry_trend_confluence_rider"
    required_features = ("funding_rate",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_lookback": HyperParam.integer("trend_lookback", default=48, low=3, high=20000),
            "trend_ma_window": HyperParam.integer("trend_ma_window", default=48, low=2, high=8192),
            "min_trend_roc": HyperParam.floating("min_trend_roc", default=0.0, low=0.0, high=1.0),
            "funding_window": HyperParam.integer("funding_window", default=8, low=1, high=4096),
            "long_carry_funding": HyperParam.floating(
                "long_carry_funding", default=0.00005, low=-0.05, high=0.05
            ),
            "short_carry_funding": HyperParam.floating(
                "short_carry_funding", default=0.00005, low=-0.05, high=0.05
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
        size = max(self.funding_window, 1) + 8
        self._funding: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))
        self.funding_window = max(1, int(resolved["funding_window"]))
        self.long_carry_funding = float(resolved["long_carry_funding"])
        self.short_carry_funding = float(resolved["short_carry_funding"])

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.trend_lookback, self.trend_ma_window) + 2,
        )

    # -- feature read (perp funding with graceful None fallback) -------------
    def _extract_feature(self, symbol: str, field: str) -> float | None:
        """Resolve a per-symbol feature value (here ``funding_rate``).

        Tries the bars feature/value getters in turn; returns ``None`` (skip,
        never raise) when no source provides a finite value.  Own 3-arg signature
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

    def _avg_funding(self, symbol: str) -> float | None:
        """Return the trailing-average funding rate, or ``None`` when unknown.

        Requires a full ``funding_window`` of observations so the carry sign is
        persistent rather than reacting to a single first print.
        """
        history = self._funding.get(symbol)
        if not history or len(history) < self.funding_window:
            return None
        window = list(history)[-self.funding_window :]
        if not window:
            return None
        return sum(window) / float(len(window))

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Capture the latest funding observation for this accepted bar BEFORE the
        # base advances the OHLC ring buffers / runs ride+entry, so the carry
        # decision sees current funding.  Dedup matches the base time-key guard.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                funding = self._extract_feature(symbol, "funding_rate")
                if funding is not None:
                    self._funding.setdefault(
                        symbol, deque(maxlen=max(self.funding_window, 1) + 8)
                    ).append(float(funding))
        super()._process_symbol(symbol, snapshot)

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

    def _confluence_direction(self, symbol: str, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` from the trend-carry confluence."""
        avg = self._avg_funding(symbol)
        if avg is None:
            return ""
        trend = self._trend_sign(list(item.closes))
        if trend == 0:
            return ""
        # LONG: trend up AND funding low/negative (long perp receives carry).
        if trend > 0 and avg <= self.long_carry_funding:
            return "LONG"
        # SHORT: trend down AND funding high-positive (short perp receives carry).
        if trend < 0 and avg >= self.short_carry_funding:
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
        return self._confluence_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Keep compounding only while the trend-and-carry confluence still points
        # our way; the base ATR add-step gate already enforces favorable extension.
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        direction = self._confluence_direction(symbol, item)
        return direction == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        avg = self._avg_funding(symbol) if symbol is not None else None
        trend = self._trend_sign(list(item.closes))
        return {
            "avg_funding_rate": float(avg) if avg is not None else None,
            "trend_sign": int(trend),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["funding"] = {symbol: list(history) for symbol, history in self._funding.items()}
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("funding") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            target = self._funding.get(symbol)
            if target is None:
                continue
            target.clear()
            for value in list(payload or [])[-int(target.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)


@register("strategy", "VolatilitySqueezeBreakoutRiderStrategy", interface="event_driven")
class VolatilitySqueezeBreakoutRiderStrategy(_ReturnRiderBase):
    """Ride a directional range-break that follows a low-vol SQUEEZE regime.

    The volatility-CONTRACTION precondition is the edge.  Per symbol the sleeve
    keeps a trailing record of Bollinger Bandwidth and flags a SQUEEZE when the
    latest bandwidth sits at a multi-bar percentile low
    (``percentile_rank(bandwidths) <= squeeze_percentile``), optionally also
    requiring the Bollinger band to sit INSIDE the Keltner channel (the classic
    TTM-squeeze "bands inside keltner" condition) when ``require_bb_in_kc`` is set.

    A squeeze on the PRIOR bar plus a volatility EXPANSION on the current bar
    (bandwidth rising back above ``expansion_mult`` times the squeeze-bar
    bandwidth) AND a range break (close beyond the prior ``breakout_window``-bar
    high/low) triggers a directional entry in the BREAKOUT direction.  The move is
    ridden with the inherited ATR trailing stop and pyramided on follow-through.
    Long and short.

    DISTINCT from :class:`VWAPCompressionReversionStrategy` (vol compression ->
    mean reversion, opposite polarity) and from
    :class:`VolatilityBreakoutRiderStrategy` (a Donchian channel breakout NOT
    conditioned on a prior contraction state).  OHLCV-only, per-symbol
    single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "VolatilitySqueezeBreakoutRiderStrategy"
    strategy_id = "volatility_squeeze_breakout_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "bandwidth_window": HyperParam.integer(
                "bandwidth_window", default=20, low=2, high=4096
            ),
            "bandwidth_num_std": HyperParam.floating(
                "bandwidth_num_std", default=2.0, low=0.1, high=10.0
            ),
            "squeeze_percentile_window": HyperParam.integer(
                "squeeze_percentile_window", default=48, low=4, high=8192
            ),
            "squeeze_percentile": HyperParam.floating(
                "squeeze_percentile", default=0.25, low=0.0, high=1.0
            ),
            "expansion_mult": HyperParam.floating(
                "expansion_mult", default=1.5, low=1.0, high=20.0
            ),
            "breakout_window": HyperParam.integer("breakout_window", default=20, low=2, high=4096),
            "require_bb_in_kc": HyperParam.boolean(
                "require_bb_in_kc", default=False, grid=[True, False]
            ),
            "keltner_window": HyperParam.integer("keltner_window", default=20, low=2, high=4096),
            "keltner_atr_window": HyperParam.integer(
                "keltner_atr_window", default=10, low=2, high=1024
            ),
            "keltner_atr_mult": HyperParam.floating(
                "keltner_atr_mult", default=1.5, low=0.1, high=20.0
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.5, low=0.1, high=20.0
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
        # Trailing Bollinger Bandwidth record per symbol (one value per decision
        # bar) for the percentile-low squeeze detection.
        size = max(self.squeeze_percentile_window, self.bandwidth_window) + 8
        self._bandwidths: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }
        # Whether the PRIOR accepted bar was in a squeeze regime, plus that bar's
        # bandwidth (the baseline the current bar's expansion is measured against).
        self._prev_squeeze: dict[str, bool] = dict.fromkeys(self.symbol_list, False)
        self._squeeze_bandwidth: dict[str, float | None] = dict.fromkeys(self.symbol_list)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.bandwidth_window = max(2, int(resolved["bandwidth_window"]))
        self.bandwidth_num_std = max(0.1, float(resolved["bandwidth_num_std"]))
        self.squeeze_percentile_window = max(4, int(resolved["squeeze_percentile_window"]))
        self.squeeze_percentile = max(0.0, min(1.0, float(resolved["squeeze_percentile"])))
        self.expansion_mult = max(1.0, float(resolved["expansion_mult"]))
        self.breakout_window = max(2, int(resolved["breakout_window"]))
        self.require_bb_in_kc = bool(resolved["require_bb_in_kc"])
        self.keltner_window = max(2, int(resolved["keltner_window"]))
        self.keltner_atr_window = max(2, int(resolved["keltner_atr_window"]))
        self.keltner_atr_mult = max(0.1, float(resolved["keltner_atr_mult"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(
                int(resolved["bandwidth_window"]),
                int(resolved["squeeze_percentile_window"]),
                int(resolved["breakout_window"]),
                int(resolved["keltner_window"]),
            )
            + 2,
        )

    def _bb_inside_kc(self, item: _RiderState) -> bool:
        """Return whether the Bollinger band sits inside the Keltner channel.

        ``True`` when the band data is unavailable so the optional gate never
        silences the sleeve during warmup; the real squeeze gate is the
        percentile-low bandwidth.
        """
        closes = list(item.closes)
        _mid_bb, upper_bb, lower_bb = bollinger_bands(
            closes, window=self.bandwidth_window, num_std=self.bandwidth_num_std
        )
        if upper_bb is None or lower_bb is None:
            return True
        highs = list(item.highs)
        lows = list(item.lows)
        n = min(len(highs), len(lows), len(closes))
        if n < self.keltner_window:
            return True
        _mid_kc, upper_kc, lower_kc = keltner_channel(
            highs[-n:],
            lows[-n:],
            closes[-n:],
            window=self.keltner_window,
            atr_window=self.keltner_atr_window,
            atr_multiplier=self.keltner_atr_mult,
        )
        if upper_kc is None or lower_kc is None:
            return True
        return upper_bb <= upper_kc and lower_bb >= lower_kc

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Update the trailing bandwidth record + prior-bar squeeze flag BEFORE the
        # base loop runs entry, so _entry_decision sees the freshest squeeze/
        # expansion state.  Mirror the base time-key dedup so a repeated boundary
        # firing never double-updates.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._refresh_squeeze(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _refresh_squeeze(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        """Append this bar's bandwidth and latch the PRIOR-bar squeeze state.

        The base loop appends the live close to the OHLC ring AFTER this runs, so
        the bandwidth is computed on the history INCLUDING the current close by
        folding it in transiently.
        """
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        # Build the close series that the current bar WILL have once the base
        # appends it, so the bandwidth reflects the live bar.
        closes = [*list(item.closes), close]
        bandwidth = bollinger_bandwidth(
            closes, window=self.bandwidth_window, num_std=self.bandwidth_num_std
        )
        history = self._bandwidths.setdefault(
            symbol, deque(maxlen=max(self.squeeze_percentile_window, self.bandwidth_window) + 8)
        )
        # Latch whether the PRIOR completed bar was a squeeze (before appending the
        # new bandwidth) and the bandwidth that defined that squeeze baseline.
        prior_squeeze = False
        prior_bandwidth: float | None = None
        if len(history) >= self.squeeze_percentile_window:
            rank = percentile_rank(list(history), window=self.squeeze_percentile_window)
            # The percentile-low bandwidth is the primary contraction gate; when
            # require_bb_in_kc is set, additionally demand the classic TTM-squeeze
            # condition (Bollinger band INSIDE the Keltner channel) on the same
            # prior-bar state.  _bb_inside_kc returns True during warmup so the
            # optional gate never silences the sleeve before it has band data.
            if (
                rank is not None
                and rank <= self.squeeze_percentile
                and not (self.require_bb_in_kc and not self._bb_inside_kc(item))
            ):
                prior_squeeze = True
                prior_bandwidth = history[-1]
        self._prev_squeeze[symbol] = prior_squeeze
        self._squeeze_bandwidth[symbol] = prior_bandwidth
        if bandwidth is not None:
            history.append(float(bandwidth))

    def _breakout_direction(self, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` for a prior-N-bar range break."""
        highs = list(item.highs)
        lows = list(item.lows)
        closes = list(item.closes)
        if len(highs) <= self.breakout_window or len(lows) <= self.breakout_window:
            return ""
        # Exclude the live bar so the break level is the PRIOR N-bar extreme.
        upper, _middle, lower = donchian_channel(highs[:-1], lows[:-1], window=self.breakout_window)
        if upper is None or lower is None:
            return ""
        close = closes[-1]
        if close >= upper:
            return "LONG"
        if close <= lower:
            return "SHORT"
        return ""

    def _expanding(self, symbol: str) -> bool:
        """Return whether the latest bandwidth expanded beyond the squeeze baseline."""
        history = self._bandwidths.get(symbol)
        baseline = self._squeeze_bandwidth.get(symbol)
        if history is None or len(history) == 0 or baseline is None or baseline <= _EPS:
            return False
        return history[-1] >= self.expansion_mult * baseline

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        # Require a squeeze on the PRIOR bar (the contraction regime precondition)
        # AND a volatility expansion on the current bar.
        if not self._prev_squeeze.get(symbol, False):
            return ""
        if not self._expanding(symbol):
            return ""
        return self._breakout_direction(item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Keep compounding only while the breakout continues in our direction; the
        # base ATR add-step gate already enforces favorable extension.
        direction = self._breakout_direction(item)
        return direction == item.mode

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        baseline = self._squeeze_bandwidth.get(symbol) if symbol is not None else None
        history = self._bandwidths.get(symbol) if symbol is not None else None
        current = history[-1] if history else None
        return {
            "squeeze_bandwidth": float(baseline) if baseline is not None else None,
            "current_bandwidth": float(current) if current is not None else None,
            "breakout_window": int(self.breakout_window),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["squeeze_state"] = {
            symbol: {
                "bandwidths": list(history),
                "prev_squeeze": bool(self._prev_squeeze.get(symbol, False)),
                "squeeze_bandwidth": self._squeeze_bandwidth.get(symbol),
            }
            for symbol, history in self._bandwidths.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("squeeze_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            target = self._bandwidths.get(symbol)
            if target is None or not isinstance(payload, dict):
                continue
            target.clear()
            for value in list(payload.get("bandwidths") or [])[-int(target.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)
            self._prev_squeeze[symbol] = bool(payload.get("prev_squeeze", False))
            self._squeeze_bandwidth[symbol] = safe_float(payload.get("squeeze_bandwidth"))


__all__ = [
    "CarryTrendConfluenceRiderStrategy",
    "VolatilitySqueezeBreakoutRiderStrategy",
]
