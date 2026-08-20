"""CUSUM change-point and variance-ratio gated trend riders.

Two per-symbol single-asset, return-first DIRECTIONAL sleeves that subclass the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and add a fresh entry trigger the
existing registry does not carry.  Both are wired ONLY at >=30m timeframes and
stay ``candidate_mix_type=="single"`` so they bypass the multi-asset admission
gate.

- :class:`CusumChangePointTrendRiderStrategy`: OHLCV-only.  THEORY (Page 1954
  cumulative-sum change-point detection).  Returns are standardized by a rolling
  volatility, then a two-sided CUSUM control chart accumulates the standardized
  drift: ``S_hi = max(0, S_hi + z - k)`` and ``S_lo = min(0, S_lo + z + k)``.
  When ``S_hi`` crosses the decision threshold ``h`` an UPWARD drift regime shift
  is declared (enter LONG and reset ``S_hi``); when ``S_lo`` crosses ``-h`` a
  DOWNWARD shift is declared (enter SHORT and reset ``S_lo``).  The detected shift
  direction IS the entry; the winner is ridden with the inherited ATR trailing
  stop.  DISTINCT from :class:`KalmanTrendRiderStrategy` (a continuous state-space
  slope), :class:`AdaptiveTrendRiderStrategy` (KAMA), and the channel/MA trend
  sleeves: the trigger is a DISCRETE sequential change-point detector, which no
  other sleeve uses.

- :class:`VarianceRatioTrendRiderStrategy`: OHLCV-only.  THEORY (Lo-MacKinlay
  1988 variance-ratio test of the random walk).  The variance ratio
  ``VR(k) = Var(k-period returns) / (k * Var(1-period returns))`` is ``~1`` under a
  random walk, ``>1`` under positive return autocorrelation (a persistent/trending
  regime), and ``<1`` under mean reversion.  The sleeve rides the prevailing trend
  ONLY when ``VR(k) >= 1 + vr_threshold`` (the random walk is rejected toward
  PERSISTENCE) AND the trend is confirmed.  DISTINCT from
  :class:`HurstRegimeGatedStrategy` (rescaled-range Hurst),
  :class:`PermutationEntropyTrendRiderStrategy` (ordinal-pattern entropy), and
  :class:`TrendEfficiencyMomentumStrategy` (Kaufman efficiency): the gate is the
  Lo-MacKinlay variance-ratio statistic, which no other sleeve computes.

Both: an ATR trailing stop is the primary return lever; a TopCap-scale
``target_allocation`` (``tunable=False``); vol-targeted sizing; a
``max_hold_bars`` cap; per-symbol bounded state; ``None``-safe graceful guards
(never raise).  ``decision_cadence_seconds`` defaults to 1800 (>=30m floor).
They are data-local (no fetching, no artifact writes) and must still be validated
on the data-bearing machine.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.indicators.variance_ratio import variance_ratio as _variance_ratio
from lumina_quant.strategies.external_alpha_sleeves import _EPS, _Snapshot
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


def _rolling_std(values: list[float]) -> float | None:
    """Population standard deviation of ``values`` (``None`` if degenerate)."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) * (v - mean) for v in values) / n
    if var <= 0.0:
        return None
    return math.sqrt(var)


# The Lo-MacKinlay variance ratio now lives at indicator level
# (``indicators/variance_ratio.py``); the BIASED overlapping estimator that this
# sleeve always used is that indicator's DEFAULT, so the ``_variance_ratio``
# import alias above is parity-locked byte-identical to the previous private
# helper.  Unbiased / heteroskedasticity-robust z-stat variants are opt-in at
# the indicator and unused here.


def _trend_sign(
    closes: list[float], *, trend_lookback: int, trend_ma_window: int, min_trend_roc: float
) -> int:
    """Return ``+1``/``-1``/``0`` for the prevailing long-horizon trend."""
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


@register("strategy", "CusumChangePointTrendRiderStrategy", interface="event_driven")
class CusumChangePointTrendRiderStrategy(_ReturnRiderBase):
    """Ride the drift regime shift declared by a two-sided CUSUM control chart.

    Returns are standardized by a rolling volatility; a two-sided CUSUM
    accumulates the standardized drift with slack ``cusum_k``.  An UPWARD shift
    (``S_hi >= cusum_h``) opens a LONG and resets ``S_hi``; a DOWNWARD shift
    (``S_lo <= -cusum_h``) opens a SHORT and resets ``S_lo`` (when ``allow_short``).
    The detected direction is the entry; the winner is ridden with the inherited
    ATR trailing stop + pyramiding.

    DISTINCT from :class:`KalmanTrendRiderStrategy` (continuous state-space slope)
    and the MA/channel trend sleeves: the trigger is a DISCRETE sequential
    change-point detector.  OHLCV-only, per-symbol single-asset,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "CusumChangePointTrendRiderStrategy"
    strategy_id = "cusum_change_point_trend_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "cusum_vol_window": HyperParam.integer(
                "cusum_vol_window", default=48, low=4, high=8192
            ),
            "cusum_k": HyperParam.floating("cusum_k", default=0.5, low=0.0, high=10.0),
            "cusum_h": HyperParam.floating("cusum_h", default=5.0, low=0.1, high=100.0),
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
        size = max(self.cusum_vol_window, 2) + 8
        self._returns: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }
        self._s_hi: dict[str, float] = dict.fromkeys(self.symbol_list, 0.0)
        self._s_lo: dict[str, float] = dict.fromkeys(self.symbol_list, 0.0)
        self._signal: dict[str, str] = dict.fromkeys(self.symbol_list, "")

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.cusum_vol_window = max(4, int(resolved["cusum_vol_window"]))
        self.cusum_k = max(0.0, float(resolved["cusum_k"]))
        self.cusum_h = max(0.1, float(resolved["cusum_h"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=int(resolved["cusum_vol_window"]) + 2)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Update the CUSUM detector on this bar's realized return BEFORE the base
        # runs ride+entry, mirroring the base time-key dedup.  Causal: the return
        # uses the current and previous closes, both known at the decision close.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._update_cusum(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _update_cusum(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        closes = list(item.closes)
        self._signal[symbol] = ""
        if not closes or closes[-1] <= _EPS:
            return
        ret = math.log(close / closes[-1])
        if not math.isfinite(ret):
            return
        history = self._returns.setdefault(symbol, deque(maxlen=max(self.cusum_vol_window, 2) + 8))
        history.append(ret)
        if len(history) < self.cusum_vol_window:
            return
        std = _rolling_std(list(history)[-self.cusum_vol_window :])
        if std is None or std <= _EPS:
            return
        z = ret / std
        s_hi = max(0.0, self._s_hi.get(symbol, 0.0) + z - self.cusum_k)
        s_lo = min(0.0, self._s_lo.get(symbol, 0.0) + z + self.cusum_k)
        if s_hi >= self.cusum_h:
            self._signal[symbol] = "LONG"
            s_hi = 0.0
            s_lo = 0.0
        elif s_lo <= -self.cusum_h:
            self._signal[symbol] = "SHORT"
            s_hi = 0.0
            s_lo = 0.0
        self._s_hi[symbol] = s_hi
        self._s_lo[symbol] = s_lo

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._signal.get(symbol, "")

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Scale in only on a fresh same-direction change-point re-trigger.
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._signal.get(symbol, "") == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        return {
            "cusum_s_hi": float(self._s_hi.get(symbol, 0.0)) if symbol is not None else None,
            "cusum_s_lo": float(self._s_lo.get(symbol, 0.0)) if symbol is not None else None,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cusum_state"] = {
            symbol: {
                "returns": list(self._returns.get(symbol, ())),
                "s_hi": float(self._s_hi.get(symbol, 0.0)),
                "s_lo": float(self._s_lo.get(symbol, 0.0)),
            }
            for symbol in self._state
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("cusum_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._returns or not isinstance(payload, dict):
                continue
            target = self._returns[symbol]
            target.clear()
            for value in list(payload.get("returns") or [])[-int(target.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)
            self._s_hi[symbol] = float(safe_float(payload.get("s_hi")) or 0.0)
            self._s_lo[symbol] = float(safe_float(payload.get("s_lo")) or 0.0)


@register("strategy", "VarianceRatioTrendRiderStrategy", interface="event_driven")
class VarianceRatioTrendRiderStrategy(_ReturnRiderBase):
    """Ride the trend ONLY when the Lo-MacKinlay variance ratio signals persistence.

    Over a window of log returns the variance ratio
    ``VR(k) = Var(k-period) / (k * Var(1-period))`` is computed.  A LONG/SHORT ride
    is armed only when ``VR(k) >= 1 + vr_threshold`` (the random walk is rejected
    toward POSITIVE autocorrelation / persistence) AND the prevailing trend is
    confirmed; a random-walk or mean-reverting regime suppresses entry.  The winner
    is ridden with the inherited ATR trailing stop + pyramiding.

    DISTINCT from :class:`HurstRegimeGatedStrategy`,
    :class:`PermutationEntropyTrendRiderStrategy`, and
    :class:`TrendEfficiencyMomentumStrategy`: the regime gate is the Lo-MacKinlay
    variance-ratio statistic.  OHLCV-only, per-symbol single-asset,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "VarianceRatioTrendRiderStrategy"
    strategy_id = "variance_ratio_trend_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "vr_window": HyperParam.integer("vr_window", default=96, low=8, high=8192),
            "vr_k": HyperParam.integer("vr_k", default=4, low=2, high=512),
            "vr_threshold": HyperParam.floating("vr_threshold", default=0.10, low=0.0, high=10.0),
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
        size = max(self.vr_window, 4) + 8
        self._returns: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.vr_window = max(8, int(resolved["vr_window"]))
        self.vr_k = max(2, int(resolved["vr_k"]))
        self.vr_threshold = max(0.0, float(resolved["vr_threshold"]))
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(
                int(resolved["vr_window"]),
                int(resolved["trend_lookback"]),
                int(resolved["trend_ma_window"]),
            )
            + 2,
        )

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Append this bar's realized log return BEFORE the base runs ride+entry,
        # mirroring the base time-key dedup.  Causal (current vs previous close).
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._append_return(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _append_return(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        closes = list(item.closes)
        if closes and closes[-1] > _EPS:
            ret = math.log(close / closes[-1])
            if math.isfinite(ret):
                self._returns.setdefault(symbol, deque(maxlen=max(self.vr_window, 4) + 8)).append(
                    ret
                )

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _vr(self, symbol: str) -> float | None:
        history = self._returns.get(symbol)
        if not history or len(history) < self.vr_window:
            return None
        return _variance_ratio(list(history)[-self.vr_window :], self.vr_k)

    def _gated_direction(self, symbol: str, item: _RiderState) -> str:
        vr = self._vr(symbol)
        if vr is None or vr < 1.0 + self.vr_threshold:
            # Random-walk or mean-reverting regime -> no trend ride.
            return ""
        trend = _trend_sign(
            list(item.closes),
            trend_lookback=self.trend_lookback,
            trend_ma_window=self.trend_ma_window,
            min_trend_roc=self.min_trend_roc,
        )
        if trend > 0:
            return "LONG"
        if trend < 0:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._gated_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._gated_direction(symbol, item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        vr = self._vr(symbol) if symbol is not None else None
        return {
            "variance_ratio": float(vr) if vr is not None else None,
            "vr_k": int(self.vr_k),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["vr_returns"] = {symbol: list(history) for symbol, history in self._returns.items()}
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("vr_returns") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            target = self._returns.get(symbol)
            if target is None:
                continue
            target.clear()
            for value in list(payload or [])[-int(target.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)


__all__ = [
    "CusumChangePointTrendRiderStrategy",
    "VarianceRatioTrendRiderStrategy",
]
