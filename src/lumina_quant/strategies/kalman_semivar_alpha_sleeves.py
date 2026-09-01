"""Kalman state-space trend and realized-semivariance asymmetry riders.

Two per-symbol single-asset, return-first DIRECTIONAL sleeves that subclass the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and add a fresh signal the existing
registry does not carry.  Both are wired ONLY at >=30m timeframes and stay
``candidate_mix_type=="single"`` so they bypass the multi-asset admission gate.

- :class:`KalmanTrendRiderStrategy`: OHLCV-only.  THEORY (state-space / Kalman
  trend estimation).  A LOCAL-LINEAR-TREND Kalman filter is run on the LOG price
  (so the slope is a scale-free per-bar drift): the recursive level+slope filter
  adapts its responsiveness from the process/observation-noise ratio, giving a
  lower-lag, uncertainty-aware trend estimate than a fixed-window MA/KAMA.  A ride
  is armed when the filtered SLOPE is statistically significant (``slope /
  sqrt(slope_variance) >= slope_t``) and economically meaningful (``|slope| >=
  min_slope_frac`` per bar).  DISTINCT from :class:`AdaptiveTrendRiderStrategy`
  (KAMA efficiency ratio), :class:`TrendEfficiencyMomentumStrategy`,
  :class:`DonchianAtrTrendStrategy`, and :class:`MovingAverageCrossStrategy`: the
  trend ESTIMATOR is a recursive Bayesian state-space slope, not a moving average
  or channel — there is no Kalman estimator anywhere else in the registry.

- :class:`RealizedSemivarianceTrendRiderStrategy`: OHLCV-only.  THEORY
  (Barndorff-Nielsen realized semivariance; Patton-Sheppard "Good Volatility, Bad
  Volatility" — upside vs downside realized variance carry different information,
  and the SIGNED jump variation predicts returns).  Over a trailing window of
  decision-bar log returns the sleeve accumulates the upside semivariance
  ``RS+ = sum(r^2 | r>0)`` and downside semivariance ``RS- = sum(r^2 | r<0)`` and
  forms the signed asymmetry ``SJ = (RS+ - RS-) / (RS+ + RS-)`` in ``[-1, 1]``.  A
  ride is armed only when ``|SJ| >= semivar_threshold`` (good vol clearly dominates
  bad vol, or vice-versa) AND the prevailing trend AGREES — a continuation ride
  driven by the variance-asymmetry, not a knife-catch.  DISTINCT from
  :class:`LotterySkewnessStrategy` / :class:`RareEventScoreStrategy`
  (cross-sectional lottery / tail score) and from the vol-managed sleeves (which
  SIZE by realized vol): the entry trigger is the upside/downside realized
  semivariance asymmetry, which no other sleeve uses.

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
from lumina_quant.strategies.external_alpha_sleeves import _EPS, _Snapshot
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam

# A local-linear-trend Kalman state: (level, slope, P00, P01, P11).
_KalmanState = tuple[float, float, float, float, float]


def _kalman_llt_step(
    state: _KalmanState | None,
    obs: float,
    *,
    obs_noise: float,
    level_process_noise: float,
    slope_process_noise: float,
    init_var: float,
) -> _KalmanState:
    """Advance one local-linear-trend Kalman step for a scalar observation.

    Standard LLT model: ``level_t = level + slope (+w_l)``, ``slope_t = slope
    (+w_s)``, observation ``z = level (+v)``.  Returns the posterior
    ``(level, slope, P00, P01, P11)``.  Numerically safe: the innovation variance
    ``S = P00_pred + obs_noise`` is always positive (``obs_noise > 0``).
    """
    if state is None:
        # Diffuse init: anchor the level at the first observation, zero slope.
        return (obs, 0.0, init_var, 0.0, init_var)
    level, slope, p00, p01, p11 = state
    # Predict.
    level_pred = level + slope
    slope_pred = slope
    p00_pred = p00 + 2.0 * p01 + p11 + level_process_noise
    p01_pred = p01 + p11
    p11_pred = p11 + slope_process_noise
    # Update with observation matrix H = [1, 0].
    s = p00_pred + obs_noise
    if s <= _EPS:
        return (level_pred, slope_pred, p00_pred, p01_pred, p11_pred)
    k0 = p00_pred / s
    k1 = p01_pred / s
    innovation = obs - level_pred
    new_level = level_pred + k0 * innovation
    new_slope = slope_pred + k1 * innovation
    new_p00 = (1.0 - k0) * p00_pred
    new_p01 = (1.0 - k0) * p01_pred
    new_p11 = p11_pred - k1 * p01_pred
    return (new_level, new_slope, new_p00, new_p01, new_p11)


def _signed_semivariance_asymmetry(returns: list[float]) -> float | None:
    """Return ``(RS+ - RS-) / (RS+ + RS-)`` in ``[-1, 1]`` over the returns.

    ``RS+``/``RS-`` are the upside/downside realized semivariances (sums of
    squared positive / negative returns).  ``None`` when there is no realized
    variance (an all-flat window) so the caller suppresses a signal.
    """
    rs_plus = 0.0
    rs_minus = 0.0
    for r in returns:
        if r > 0.0:
            rs_plus += r * r
        elif r < 0.0:
            rs_minus += r * r
    total = rs_plus + rs_minus
    if total <= _EPS:
        return None
    return (rs_plus - rs_minus) / total


def _trend_sign(
    closes: list[float], *, trend_lookback: int, trend_ma_window: int, min_trend_roc: float
) -> int:
    """Return ``+1``/``-1``/``0`` for the prevailing long-horizon trend.

    Requires the long-horizon ROC and the close-vs-trend-MA relationship to agree.
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


@register("strategy", "KalmanTrendRiderStrategy", interface="event_driven")
class KalmanTrendRiderStrategy(_ReturnRiderBase):
    """Ride the trend when a local-linear-trend Kalman SLOPE is significantly up/down.

    A local-linear-trend Kalman filter runs on the LOG close, yielding a
    lower-lag, uncertainty-aware estimate of the per-bar drift (the filter slope).
    A LONG ride is armed when the filtered slope is significantly positive
    (``slope / sqrt(slope_variance) >= slope_t``) and economically meaningful
    (``|slope| >= min_slope_frac``); a SHORT when significantly negative (and
    ``allow_short``).  The winner is ridden with the inherited ATR trailing stop +
    pyramiding.

    DISTINCT from :class:`AdaptiveTrendRiderStrategy` (KAMA efficiency ratio),
    :class:`TrendEfficiencyMomentumStrategy`, :class:`DonchianAtrTrendStrategy`,
    and :class:`MovingAverageCrossStrategy`: the trend ESTIMATOR is a recursive
    Bayesian state-space slope (no moving average / channel).  OHLCV-only,
    per-symbol single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "KalmanTrendRiderStrategy"
    strategy_id = "kalman_trend_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "obs_noise": HyperParam.floating("obs_noise", default=0.0001, low=1e-9, high=1.0),
            "level_process_noise": HyperParam.floating(
                "level_process_noise", default=1e-5, low=1e-12, high=1.0
            ),
            "slope_process_noise": HyperParam.floating(
                "slope_process_noise", default=1e-7, low=1e-14, high=1.0
            ),
            "slope_t": HyperParam.floating("slope_t", default=2.0, low=0.0, high=20.0),
            "min_slope_frac": HyperParam.floating("min_slope_frac", default=0.0, low=0.0, high=1.0),
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
        self._kalman: dict[str, _KalmanState | None] = dict.fromkeys(self.symbol_list)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.obs_noise = max(1e-9, float(resolved["obs_noise"]))
        self.level_process_noise = max(1e-12, float(resolved["level_process_noise"]))
        self.slope_process_noise = max(1e-14, float(resolved["slope_process_noise"]))
        self.slope_t = max(0.0, float(resolved["slope_t"]))
        self.min_slope_frac = max(0.0, float(resolved["min_slope_frac"]))
        # Diffuse-init variance for the filter's first observation.
        self.kalman_init_var = 1.0

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=int(resolved["atr_period"]) + 2)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Advance the Kalman filter on this bar's close BEFORE the base runs
        # ride+entry, mirroring the base time-key dedup so a repeated boundary
        # firing never double-updates the filter.  The filter is causal (it uses
        # only the current and past closes), so deciding on the posterior slope at
        # this bar's close is not look-ahead.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._advance_filter(symbol, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _advance_filter(self, symbol: str, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        self._kalman[symbol] = _kalman_llt_step(
            self._kalman.get(symbol),
            math.log(close),
            obs_noise=self.obs_noise,
            level_process_noise=self.level_process_noise,
            slope_process_noise=self.slope_process_noise,
            init_var=self.kalman_init_var,
        )

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _kalman_direction(self, symbol: str) -> str:
        state = self._kalman.get(symbol)
        if state is None:
            return ""
        _level, slope, _p00, _p01, p11 = state
        if abs(slope) < self.min_slope_frac:
            return ""
        if p11 > _EPS:
            t_stat = slope / math.sqrt(p11)
        else:
            t_stat = math.copysign(float("inf"), slope) if slope != 0.0 else 0.0
        if slope > 0.0 and t_stat >= self.slope_t:
            return "LONG"
        if slope < 0.0 and -t_stat >= self.slope_t:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._kalman_direction(symbol)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._kalman_direction(symbol) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        state = self._kalman.get(symbol) if symbol is not None else None
        return {
            "kalman_slope": float(state[1]) if state is not None else None,
            "kalman_slope_var": float(state[4]) if state is not None else None,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["kalman_state"] = {
            symbol: (list(payload) if payload is not None else None)
            for symbol, payload in self._kalman.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("kalman_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._kalman:
                continue
            if payload is None:
                self._kalman[symbol] = None
                continue
            try:
                seq = [float(v) for v in list(payload)]
            except Exception:
                continue
            if len(seq) == 5:
                self._kalman[symbol] = (seq[0], seq[1], seq[2], seq[3], seq[4])


@register("strategy", "RealizedSemivarianceTrendRiderStrategy", interface="event_driven")
class RealizedSemivarianceTrendRiderStrategy(_ReturnRiderBase):
    """Ride a trend confirmed by an upside/downside realized-semivariance asymmetry.

    Over a trailing window of decision-bar log returns the sleeve accumulates the
    upside semivariance ``RS+`` and downside semivariance ``RS-`` and forms the
    signed asymmetry ``SJ = (RS+ - RS-) / (RS+ + RS-)``.  A LONG ride is armed only
    when ``SJ >= semivar_threshold`` (good vol clearly dominates) AND the prevailing
    trend is up; a SHORT when ``SJ <= -semivar_threshold`` and the trend is down
    (and ``allow_short``).  The winner is ridden with the inherited ATR trailing
    stop + pyramiding.

    DISTINCT from :class:`LotterySkewnessStrategy` / :class:`RareEventScoreStrategy`
    (cross-sectional lottery / tail score) and from the vol-managed sleeves (which
    SIZE by realized vol): the entry trigger is the realized-semivariance asymmetry
    (Patton-Sheppard good/bad volatility, Barndorff-Nielsen signed jump variation).
    OHLCV-only, per-symbol single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "RealizedSemivarianceTrendRiderStrategy"
    strategy_id = "realized_semivariance_trend_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "semivar_window": HyperParam.integer("semivar_window", default=48, low=4, high=8192),
            "semivar_threshold": HyperParam.floating(
                "semivar_threshold", default=0.20, low=0.0, high=1.0
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
        size = max(self.semivar_window, 2) + 8
        self._returns: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.semivar_window = max(4, int(resolved["semivar_window"]))
        self.semivar_threshold = max(0.0, min(1.0, float(resolved["semivar_threshold"])))
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(
                int(resolved["semivar_window"]),
                int(resolved["trend_lookback"]),
                int(resolved["trend_ma_window"]),
            )
            + 2,
        )

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Append this bar's realized log return BEFORE the base runs ride+entry,
        # mirroring the base time-key dedup.  At refresh time item.closes holds bars
        # up to the PREVIOUS one, so the return folded is the CURRENT bar's realized
        # return (current close vs previous close) -- both are known at the decision
        # bar's close, which is the repo's decision point, so the asymmetry at entry
        # includes the current bar's return and is causal (no dependence on any
        # FUTURE bar).
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
        if len(closes) >= 1 and closes[-1] > _EPS:
            ret = math.log(close / closes[-1])
            if math.isfinite(ret):
                self._returns.setdefault(
                    symbol, deque(maxlen=max(self.semivar_window, 2) + 8)
                ).append(ret)

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _asymmetry(self, symbol: str) -> float | None:
        history = self._returns.get(symbol)
        if not history or len(history) < self.semivar_window:
            return None
        window = list(history)[-self.semivar_window :]
        return _signed_semivariance_asymmetry(window)

    def _confirmed_direction(self, symbol: str, item: _RiderState) -> str:
        sj = self._asymmetry(symbol)
        if sj is None or abs(sj) < self.semivar_threshold:
            return ""
        trend = _trend_sign(
            list(item.closes),
            trend_lookback=self.trend_lookback,
            trend_ma_window=self.trend_ma_window,
            min_trend_roc=self.min_trend_roc,
        )
        if sj > 0.0 and trend > 0:
            return "LONG"
        if sj < 0.0 and trend < 0:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._confirmed_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._confirmed_direction(symbol, item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        sj = self._asymmetry(symbol) if symbol is not None else None
        return {
            "signed_semivariance_asymmetry": float(sj) if sj is not None else None,
            "semivar_window": int(self.semivar_window),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["semivar_returns"] = {
            symbol: list(history) for symbol, history in self._returns.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("semivar_returns") if isinstance(state, dict) else None
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
    "KalmanTrendRiderStrategy",
    "RealizedSemivarianceTrendRiderStrategy",
]
