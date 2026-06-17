"""Aggressive per-symbol directional RETURN-maximizing alpha sleeves (>=30m).

These three sleeves push the return / compound-return objective further than the
incumbent return-rider family by improving the ENTRY edge (multi-horizon trend
alignment, buy-the-dip continuation entries) and by harvesting a steady
directional funding-carry income stream — while keeping the proven RIDE machinery
(ATR/percent trailing stop, pyramiding into continuation, vol-targeted sizing at
the TopCap scale) inherited from :class:`_ReturnRiderBase`.

- :class:`MultiTimeframeTrendEnsembleStrategy`: per symbol it measures the trend
  sign/strength at THREE lookback horizons (short/medium/long, via
  :func:`rate_of_change`) on the candidate base timeframe and ENTERS only when at
  least ``align_threshold`` of the horizons agree on direction.  Size scales with
  the agreement strength (conviction) times the vol-target.  Aligned multi-horizon
  trends are the strongest, most-persistent moves, so they ride furthest for the
  highest compound return.  This is DISTINCT from the single-horizon
  ``TopCapTimeSeriesMomentumStrategy`` / ``CompositeTrendStrategy`` (one lookback)
  and from ``AdaptiveTrendRiderStrategy`` (single adaptive KAMA slope): the edge
  here is the cross-horizon AGREEMENT gate, not one trend measure.

- :class:`PullbackTrendContinuationStrategy`: it first establishes the prevailing
  trend with a long-horizon MA slope / ROC sign, then ENTERS on a PULLBACK against
  the very-short-term move IN the trend direction (close dips toward/below a short
  MA, or short-term oversold, while the long trend is up — buy-the-dip; the mirror
  for sell-the-rally in a downtrend).  Buying the dip rather than chasing the
  breakout high gives a better entry price and a larger run from entry to exit, so
  return-per-trade is higher.  This is DISTINCT from the breakout/TSMOM riders
  purely on ENTRY TIMING (it deliberately enters on weakness within strength, not
  on a fresh N-bar high / momentum thrust).

- :class:`FundingHarvestCarryStrategy`: a per-symbol DIRECTIONAL funding-carry
  HARVEST.  Using the ``funding_rate`` feature (own 3-arg ``_extract_feature``),
  when funding is persistently NEGATIVE (longs are paid the funding) it goes/stays
  LONG; when persistently POSITIVE (shorts are paid) it goes/stays SHORT; size
  scales with ``|funding|`` magnitude.  A trend-no-fight filter blocks holding a
  short into a strongly rising price (and vice-versa).  The carry is then ridden
  with the same ATR trailing stop and pyramided while the funding sign persists,
  turning a steady CARRY-INCOME stream into compounded return.  This is DISTINCT
  from :class:`FundingDislocationTrendCarryStrategy` (which trades funding
  DISLOCATION blended with multi-horizon TREND TIMING and rebalances a
  cross-sectional book) and from ``PerpCrowdingCarryStrategy`` (which FADES extreme
  crowding/positioning): this sleeve simply collects the persistent funding sign as
  directional income, per symbol, with no dislocation/crowding-fade logic.

All three: an ATR/percent TRAILING stop is the primary return lever (winners run;
there is no tight fixed take-profit capping gains); a meaningful TopCap-scale
``target_allocation`` (``tunable=False``); vol-targeted sizing; a ``max_hold_bars``
cap; per-symbol bounded-deque state with pyramiding bookkeeping; ``None``-safe
graceful guards (never raise).  ``decision_cadence_seconds`` defaults to 1800
(>=30m floor).  They are data-local (no fetching, no artifact writes) and must
still be validated on the data-bearing machine.
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
    _Snapshot,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


def _roc_sign(closes: list[float], period: int) -> tuple[int, float]:
    """Return ``(sign, magnitude)`` of the trailing rate-of-change over ``period``.

    ``sign`` is ``+1``/``-1``/``0``; ``magnitude`` is ``|roc|`` (0.0 when unknown).
    """
    roc = rate_of_change(closes, period=max(1, int(period)))
    if roc is None:
        return 0, 0.0
    if roc > 0.0:
        return 1, abs(roc)
    if roc < 0.0:
        return -1, abs(roc)
    return 0, 0.0


@register("strategy", "MultiTimeframeTrendEnsembleStrategy", interface="event_driven")
class MultiTimeframeTrendEnsembleStrategy(_ReturnRiderBase):
    """Enter only when SHORT/MEDIUM/LONG horizon trends agree; ride + pyramid.

    Per symbol the sleeve computes the trend sign at three lookback horizons
    (``short_lookback`` / ``mid_lookback`` / ``long_lookback`` via
    :func:`rate_of_change`) on the candidate base timeframe and enters only when at
    least ``align_threshold`` of them agree on direction.  The entry strength
    (conviction) scales with the agreement count and the average horizon magnitude,
    and is folded into the vol-targeted base allocation so the cleanest aligned
    trends carry the most risk.  The move is ridden with an ATR trailing stop and
    pyramided while the horizons remain aligned.  Long and short.  The cross-horizon
    AGREEMENT gate is what makes this distinct from single-horizon TSMOM/composite
    trend and from the single-slope adaptive-trend rider.
    """

    strategy_name = "MultiTimeframeTrendEnsembleStrategy"
    strategy_id = "multi_timeframe_trend_ensemble"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "short_lookback": HyperParam.integer("short_lookback", default=6, low=1, high=4096),
            "mid_lookback": HyperParam.integer("mid_lookback", default=18, low=2, high=8192),
            "long_lookback": HyperParam.integer("long_lookback", default=48, low=3, high=20000),
            "align_threshold": HyperParam.integer("align_threshold", default=3, low=1, high=3),
            "min_horizon_roc": HyperParam.floating(
                "min_horizon_roc", default=0.0, low=0.0, high=1.0
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

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.short_lookback = max(1, int(resolved["short_lookback"]))
        self.mid_lookback = max(self.short_lookback + 1, int(resolved["mid_lookback"]))
        self.long_lookback = max(self.mid_lookback + 1, int(resolved["long_lookback"]))
        self.align_threshold = max(1, min(3, int(resolved["align_threshold"])))
        self.min_horizon_roc = max(0.0, float(resolved["min_horizon_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=self.long_lookback + 2)

    def _alignment(self, closes: list[float]) -> tuple[int, int, float]:
        """Return ``(net_sign, agree_count, avg_magnitude)`` across the 3 horizons.

        ``net_sign`` is the sign of the dominant direction; ``agree_count`` is how
        many of the three horizons exceed ``min_horizon_roc`` in that direction.
        """
        signs: list[int] = []
        mags: list[float] = []
        for lookback in (self.short_lookback, self.mid_lookback, self.long_lookback):
            sign, mag = _roc_sign(closes, lookback)
            if sign != 0 and mag >= self.min_horizon_roc:
                signs.append(sign)
                mags.append(mag)
            else:
                signs.append(0)
                mags.append(0.0)
        up = sum(1 for sign in signs if sign > 0)
        down = sum(1 for sign in signs if sign < 0)
        if up > down:
            net, agree = 1, up
        elif down > up:
            net, agree = -1, down
        else:
            # True tie (incl. up == down) -> no directional edge, no entry.
            return 0, 0, 0.0
        active = [mag for sign, mag in zip(signs, mags, strict=False) if sign == net]
        avg_mag = sum(active) / float(len(active)) if active else 0.0
        return net, agree, avg_mag

    def _entry_decision(self, item: _RiderState) -> str:
        closes = list(item.closes)
        net, agree, _avg = self._alignment(closes)
        if net == 0 or agree < self.align_threshold:
            return ""
        return "LONG" if net > 0 else "SHORT"

    def _should_pyramid(self, item: _RiderState) -> bool:
        closes = list(item.closes)
        net, agree, _avg = self._alignment(closes)
        if net == 0 or agree < self.align_threshold:
            return False
        return net > 0 if item.mode == "LONG" else net < 0

    def _vol_scaled_allocation(self, closes: list[float]) -> float:
        # Fold the multi-horizon agreement conviction into the base vol-target
        # allocation: full agreement (3/3) with strong magnitude scales up, the
        # minimum agreement scales down, while still respecting the base clamp.
        base = super()._vol_scaled_allocation(closes)
        net, agree, avg_mag = self._alignment(closes)
        if net == 0:
            return base
        agree_frac = agree / 3.0
        conviction = max(0.5, min(1.5, agree_frac + min(0.5, avg_mag * 5.0)))
        return max(0.0, min(self.target_allocation * 2.0, base * conviction))

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        net, agree, avg_mag = self._alignment(list(item.closes))
        return {
            "horizon_net_sign": int(net),
            "horizon_agree_count": int(agree),
            "horizon_avg_roc": float(avg_mag),
        }


@register("strategy", "PullbackTrendContinuationStrategy", interface="event_driven")
class PullbackTrendContinuationStrategy(_ReturnRiderBase):
    """Buy the dip in an uptrend / sell the rally in a downtrend; ride + pyramid.

    The prevailing trend is established from a long-horizon ROC sign (and an MA
    slope confirmation).  Once the long trend is UP, the sleeve waits for a PULLBACK
    — the close dipping to/below a short moving average by at least
    ``pullback_atr_mult`` ATRs (or a short-term ROC turning negative) — and then
    enters LONG into that dip; the mirror (rally into a downtrend) enters SHORT.
    Entering on weakness within strength yields a better average entry price and a
    longer run to the trailing-stop exit, lifting return-per-trade.  The move is
    ridden with an ATR trailing stop and pyramided on continuation.  Distinct from
    the breakout/TSMOM riders purely on ENTRY TIMING.
    """

    strategy_name = "PullbackTrendContinuationStrategy"
    strategy_id = "pullback_trend_continuation"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_lookback": HyperParam.integer("trend_lookback", default=48, low=3, high=20000),
            "trend_ma_window": HyperParam.integer("trend_ma_window", default=48, low=2, high=8192),
            "short_ma_window": HyperParam.integer("short_ma_window", default=10, low=2, high=4096),
            "pullback_roc_period": HyperParam.integer(
                "pullback_roc_period", default=3, low=1, high=512
            ),
            "pullback_atr_mult": HyperParam.floating(
                "pullback_atr_mult", default=0.5, low=0.0, high=20.0
            ),
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

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.short_ma_window = max(2, int(resolved["short_ma_window"]))
        self.pullback_roc_period = max(1, int(resolved["pullback_roc_period"]))
        self.pullback_atr_mult = max(0.0, float(resolved["pullback_atr_mult"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.trend_lookback, self.trend_ma_window, self.short_ma_window) + 2,
        )

    def _trend_sign(self, closes: list[float]) -> int:
        """Return ``+1``/``-1``/``0`` for the prevailing long-horizon trend.

        Requires the long-horizon ROC and the short-vs-long MA relationship to
        agree, so a flat/whipsaw regime yields ``0`` and blocks entry.
        """
        sign, mag = _roc_sign(closes, self.trend_lookback)
        if sign == 0 or mag < self.min_trend_roc:
            return 0
        long_ma = simple_moving_average(closes, self.trend_ma_window)
        if long_ma is None:
            return 0
        close = closes[-1]
        if sign > 0 and close >= long_ma:
            return 1
        if sign < 0 and close <= long_ma:
            return -1
        return 0

    def _is_pullback(self, closes: list[float], item: _RiderState, trend: int) -> bool:
        """Return whether price is pulling back AGAINST ``trend`` (dip/rally)."""
        short_ma = simple_moving_average(closes, self.short_ma_window)
        if short_ma is None:
            return False
        close = closes[-1]
        atr = self._current_atr(item)
        band = (atr * self.pullback_atr_mult) if (atr is not None and atr > _EPS) else 0.0
        short_sign, _short_mag = _roc_sign(closes, self.pullback_roc_period)
        if trend > 0:
            # Buy-the-dip: close dips to/below the short MA (by the ATR band) or
            # the very-short ROC has turned down while the long trend stays up.
            return close <= short_ma - band or short_sign < 0
        # Sell-the-rally: close rallies to/above the short MA or short ROC up.
        return close >= short_ma + band or short_sign > 0

    def _entry_decision(self, item: _RiderState) -> str:
        closes = list(item.closes)
        trend = self._trend_sign(closes)
        if trend == 0 or not self._is_pullback(closes, item, trend):
            return ""
        return "LONG" if trend > 0 else "SHORT"

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Keep compounding only while the prevailing trend still points our way;
        # the base ATR add-step gate already enforces favorable continuation.
        trend = self._trend_sign(list(item.closes))
        if trend == 0:
            return False
        return trend > 0 if item.mode == "LONG" else trend < 0

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        closes = list(item.closes)
        return {
            "trend_sign": int(self._trend_sign(closes)),
            "short_ma_window": int(self.short_ma_window),
        }


@register("strategy", "FundingHarvestCarryStrategy", interface="event_driven")
class FundingHarvestCarryStrategy(_ReturnRiderBase):
    """Per-symbol DIRECTIONAL funding-carry HARVEST with a trend-no-fight guard.

    Using the ``funding_rate`` perp feature (own 3-arg ``_extract_feature``), the
    sleeve tracks the trailing-average funding sign per symbol.  When funding is
    persistently NEGATIVE (longs are PAID the funding) it goes/stays LONG; when
    persistently POSITIVE (shorts are paid) it goes/stays SHORT.  Entry strength and
    the vol-targeted size scale with ``|funding|``.  A trend-no-fight filter blocks
    holding a short into a strongly rising price (and a long into a strongly falling
    price).  The directional carry position is then ridden with the inherited ATR
    trailing stop and pyramided while the funding sign persists — turning a steady
    CARRY-INCOME stream into compounded return.

    DISTINCT from :class:`FundingDislocationTrendCarryStrategy`, which trades the
    funding DISLOCATION blended with explicit multi-horizon TREND timing across a
    rebalanced cross-sectional book; and from ``PerpCrowdingCarryStrategy``, which
    FADES extreme crowding/positioning.  This sleeve has no dislocation score and no
    crowding fade: it simply collects the persistent funding sign as per-symbol
    directional income.
    """

    strategy_name = "FundingHarvestCarryStrategy"
    strategy_id = "funding_harvest_carry"
    required_features = ("funding_rate",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "funding_window": HyperParam.integer("funding_window", default=8, low=1, high=4096),
            "entry_funding": HyperParam.floating(
                "entry_funding", default=0.00005, low=0.0, high=0.05
            ),
            "exit_funding": HyperParam.floating("exit_funding", default=0.0, low=0.0, high=0.05),
            "funding_scale": HyperParam.floating(
                "funding_scale", default=0.0003, low=0.000001, high=0.10
            ),
            "no_fight_roc_period": HyperParam.integer(
                "no_fight_roc_period", default=6, low=1, high=4096
            ),
            "no_fight_roc": HyperParam.floating("no_fight_roc", default=0.05, low=0.0, high=1.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.5, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=480, low=1, high=200000),
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
        size = max(self.funding_window, self.no_fight_roc_period) + 8
        self._funding: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.funding_window = max(1, int(resolved["funding_window"]))
        self.entry_funding = max(0.0, float(resolved["entry_funding"]))
        self.exit_funding = max(0.0, float(resolved["exit_funding"]))
        self.funding_scale = max(_EPS, float(resolved["funding_scale"]))
        self.no_fight_roc_period = max(1, int(resolved["no_fight_roc_period"]))
        self.no_fight_roc = max(0.0, float(resolved["no_fight_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(int(resolved["funding_window"]), int(resolved["no_fight_roc_period"]))
            + 2,
        )

    def _extract_feature(self, symbol: str, field: str) -> float | None:
        """Resolve a per-symbol feature value (here: ``funding_rate``).

        Tries the bars feature/value getters in turn; returns ``None`` (skip, never
        raise) when no source provides a finite value.  Own 3-arg signature.
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
        ``persistent`` rather than reacting to a single first print.
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
                        symbol,
                        deque(maxlen=max(self.funding_window, self.no_fight_roc_period) + 8),
                    ).append(float(funding))
        super()._process_symbol(symbol, snapshot)

    def _carry_direction(self, symbol: str, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` from persistent funding sign.

        Negative funding -> longs are paid -> LONG; positive -> SHORT.  Blocked by
        the trend-no-fight filter when price is moving strongly the other way.
        """
        avg = self._avg_funding(symbol)
        if avg is None or abs(avg) < self.entry_funding:
            return ""
        closes = list(item.closes)
        roc = rate_of_change(closes, period=self.no_fight_roc_period)
        if avg < 0.0:
            # Want LONG; do not fight a strong downtrend.
            if roc is not None and roc <= -self.no_fight_roc:
                return ""
            return "LONG"
        # Want SHORT; do not fight a strong uptrend.
        if roc is not None and roc >= self.no_fight_roc:
            return ""
        return "SHORT"

    def _entry_decision(self, item: _RiderState) -> str:
        # The base passes only the state; resolve the owning symbol from the map.
        for symbol, candidate in self._state.items():
            if candidate is item:
                return self._carry_direction(symbol, item)
        return ""

    def _should_pyramid(self, item: _RiderState) -> bool:
        for symbol, candidate in self._state.items():
            if candidate is item:
                avg = self._avg_funding(symbol)
                if avg is None:
                    return False
                if item.mode == "LONG":
                    return avg < 0.0 and abs(avg) >= self.exit_funding
                return avg > 0.0 and abs(avg) >= self.exit_funding
        return False

    def _vol_scaled_allocation(self, closes: list[float]) -> float:
        # The owning symbol is unknown to the base sizing call, so fold the
        # most-recent strong funding magnitude across symbols into a conviction
        # multiplier (clamped) on top of the vol-target base allocation.
        base = super()._vol_scaled_allocation(closes)
        best = 0.0
        for symbol in self._state:
            avg = self._avg_funding(symbol)
            if avg is not None:
                best = max(best, abs(avg))
        if best <= 0.0:
            return base
        conviction = max(0.5, min(2.0, best / self.funding_scale))
        return max(0.0, min(self.target_allocation * 2.0, base * conviction))

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        for symbol, candidate in self._state.items():
            if candidate is item:
                avg = self._avg_funding(symbol)
                return {"avg_funding_rate": float(avg) if avg is not None else None}
        return {"avg_funding_rate": None}

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


__all__ = [
    "FundingHarvestCarryStrategy",
    "MultiTimeframeTrendEnsembleStrategy",
    "PullbackTrendContinuationStrategy",
]
