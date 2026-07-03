"""Hill tail-index regime rider: distributional TAIL-SHAPE, not level/direction.

A per-symbol single-asset, return-first DIRECTIONAL sleeve that subclasses the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and adds a fresh entry trigger the
existing registry does not carry.  It is wired ONLY at >=30m timeframes and
stays ``candidate_mix_type=="single"`` so it bypasses the multi-asset admission
gate.

- :class:`TailIndexRegimeRiderStrategy`: OHLCV-only.  THEORY (Hill 1975
  extreme-value-theory tail-index estimator; Kelly & Jiang 2014 "Tail Risk and
  Asset Prices" -- a fattening common tail-risk factor predicts elevated crash
  risk and carries a return premium).  Over a trailing window of decision-bar
  log returns, the sleeve isolates the LOSS tail (absolute value of negative
  returns) and estimates the Hill tail index twice: ``hill_short`` over a
  short RECENT sub-window (``k_short`` upper order statistics) and
  ``hill_base`` over the FULL ``tail_window`` baseline (``k_long`` upper order
  statistics).  The regime ratio ``hill_short / hill_base`` says whether the
  recent loss tail is FATTENING (ratio below 1) or THINNING (ratio at/above 1)
  relative to its own baseline.  A SHORT is armed when the tail is fattening
  (``ratio <= fatten_ratio``) AND the prevailing trend is down (crash-risk
  continuation); a LONG is armed when the tail is thinning/stable (``ratio >=
  thin_ratio``) AND the prevailing trend is up (risk-on once tail risk has
  abated).  The winner is ridden with the inherited ATR trailing stop +
  pyramiding.

  DISTINCT from every other sleeve in the registry that touches extremes or
  regime detection: :mod:`lumina_quant.indicators.rare_event` (used by
  ``RareEventScoreStrategy`` / ``LotterySkewnessStrategy``) scores how extreme
  a SINGLE observation is relative to its own history -- a per-bar rarity
  percentile.  The Hill estimator here instead fits the POWER-LAW EXPONENT of
  the entire loss-tail DISTRIBUTION over many order statistics -- a
  continuous distributional-SHAPE regime, not an event score, and it never
  reduces to a rescaled max/percentile of one observation.  It is also
  DISTINCT from :class:`~lumina_quant.strategies.cusum_varratio_alpha_sleeves.
  CusumChangePointTrendRiderStrategy` (a discrete sequential change-point
  detector on the return MEAN), :class:`~lumina_quant.strategies.
  kalman_semivar_alpha_sleeves.KalmanTrendRiderStrategy` (a continuous
  state-space slope of the price LEVEL), and :class:`~lumina_quant.strategies.
  cusum_varratio_alpha_sleeves.VarianceRatioTrendRiderStrategy` (the
  Lo-MacKinlay variance-ratio test of return AUTOCORRELATION): none of those
  estimate the tail's power-law shape, so this sleeve's entry trigger has
  essentially zero mechanical overlap with level, trend, or autocorrelation
  statistics already in the book.

An ATR trailing stop is the primary return lever; a TopCap-scale
``target_allocation`` (``tunable=False``); vol-targeted sizing; a
``max_hold_bars`` cap; per-symbol bounded state; ``None``-safe graceful guards
(never raise).  ``decision_cadence_seconds`` defaults to 1800 (>=30m floor).
It is data-local (no fetching, no artifact writes) and must still be validated
on the data-bearing machine.

Registration is intentionally deferred (see the commented ``@register`` line
below the class): this sleeve ships alongside a batch of same-day siblings and
is wired into the plugin registry and ``candidate_library`` atomically, with a
``research_only`` tier hint, in a follow-up integration step -- never before.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.indicators.tail_index import hill_tail_index
from lumina_quant.strategies.external_alpha_sleeves import _EPS, _Snapshot
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


def _trend_sign(closes: list[float], *, trend_lookback: int, min_trend_roc: float) -> int:
    """Return ``+1``/``-1``/``0`` for the prevailing trend.

    The sign of the return over ``trend_lookback`` bars, gated by a minimum
    absolute magnitude ``min_trend_roc`` so a flat/noisy path reads as neutral.
    """
    roc = rate_of_change(closes, period=trend_lookback)
    if roc is None or abs(roc) < min_trend_roc:
        return 0
    return 1 if roc > 0.0 else -1


def _loss_sample(returns: list[float], window: int) -> list[float]:
    """Return absolute values of the negative returns in the trailing ``window``."""
    window_i = max(1, int(window))
    tail = returns[-window_i:] if window_i < len(returns) else returns
    return [abs(r) for r in tail if r < 0.0]


class TailIndexRegimeRiderStrategy(_ReturnRiderBase):
    """Ride the trend confirmed by a Hill tail-index fattening/thinning regime.

    Over the trailing ``tail_window`` of decision-bar log returns, the sleeve
    isolates the loss tail (absolute negative returns) and estimates the Hill
    tail index over a short ``recent_window`` sub-sample (``hill_short``,
    ``k_short`` order statistics) and over the full ``tail_window`` baseline
    (``hill_base``, ``k_long`` order statistics).  The regime ratio
    ``hill_short / hill_base`` gates entry: ``ratio <= fatten_ratio`` (the loss
    tail is fattening) with a confirmed downtrend arms a SHORT; ``ratio >=
    thin_ratio`` (the loss tail is thinning/stable) with a confirmed uptrend
    arms a LONG (and ``allow_short``).  The winner is ridden with the inherited
    ATR trailing stop + pyramiding.

    DISTINCT from :class:`~lumina_quant.strategies.cusum_varratio_alpha_sleeves.
    CusumChangePointTrendRiderStrategy`, :class:`~lumina_quant.strategies.
    kalman_semivar_alpha_sleeves.KalmanTrendRiderStrategy`, and
    :class:`~lumina_quant.strategies.cusum_varratio_alpha_sleeves.
    VarianceRatioTrendRiderStrategy`: the regime gate is the Hill (1975)
    extreme-value tail-index estimator applied to the LOSS-TAIL distribution
    shape, which no other sleeve computes.  OHLCV-only, per-symbol
    single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "TailIndexRegimeRiderStrategy"
    strategy_id = "tail_index_regime_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "tail_window": HyperParam.integer("tail_window", default=96, low=16, high=8192),
            "recent_window": HyperParam.integer("recent_window", default=24, low=4, high=4096),
            "k_short": HyperParam.integer("k_short", default=8, low=2, high=256),
            "k_long": HyperParam.integer("k_long", default=20, low=2, high=1024),
            "fatten_ratio": HyperParam.floating("fatten_ratio", default=0.85, low=0.05, high=2.0),
            "thin_ratio": HyperParam.floating("thin_ratio", default=1.15, low=0.5, high=5.0),
            "trend_lookback": HyperParam.integer("trend_lookback", default=20, low=2, high=20000),
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
        size = max(self.tail_window, 2) + 8
        self._returns: dict[str, deque[float]] = {
            symbol: deque(maxlen=size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.tail_window = max(16, int(resolved["tail_window"]))
        self.recent_window = min(self.tail_window, max(4, int(resolved["recent_window"])))
        self.k_short = max(2, int(resolved["k_short"]))
        self.k_long = max(2, int(resolved["k_long"]))
        self.fatten_ratio = max(0.0, float(resolved["fatten_ratio"]))
        self.thin_ratio = max(0.0, float(resolved["thin_ratio"]))
        self.trend_lookback = max(2, int(resolved["trend_lookback"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(int(resolved["tail_window"]), int(resolved["trend_lookback"])) + 2,
        )

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Append this bar's realized log return BEFORE the base runs ride+entry,
        # mirroring the base time-key dedup.  Causal: uses the current and
        # previous decision-bar closes, both known at this bar's close.
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
                self._returns.setdefault(symbol, deque(maxlen=max(self.tail_window, 2) + 8)).append(
                    ret
                )

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _hill_metrics(self, symbol: str) -> tuple[float | None, float | None, float | None]:
        """Return ``(hill_short, hill_base, ratio)`` for the symbol's loss tail."""
        history = self._returns.get(symbol)
        if not history or len(history) < self.tail_window:
            return None, None, None
        returns = list(history)
        long_losses = _loss_sample(returns, self.tail_window)
        short_losses = _loss_sample(returns, self.recent_window)
        hill_base = hill_tail_index(long_losses, k=self.k_long)
        hill_short = hill_tail_index(short_losses, k=self.k_short)
        if hill_base is None or hill_short is None or hill_base <= _EPS:
            return hill_short, hill_base, None
        ratio = hill_short / hill_base
        return hill_short, hill_base, (ratio if math.isfinite(ratio) else None)

    def _regime_direction(self, symbol: str, item: _RiderState) -> str:
        _hill_short, _hill_base, ratio = self._hill_metrics(symbol)
        if ratio is None:
            return ""
        trend = _trend_sign(
            list(item.closes),
            trend_lookback=self.trend_lookback,
            min_trend_roc=self.min_trend_roc,
        )
        if ratio <= self.fatten_ratio and trend < 0:
            return "SHORT"
        if ratio >= self.thin_ratio and trend > 0:
            return "LONG"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._regime_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._regime_direction(symbol, item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        if symbol is None:
            return {"hill_short": None, "hill_base": None, "tail_index_ratio": None}
        hill_short, hill_base, ratio = self._hill_metrics(symbol)
        return {
            "hill_short": float(hill_short) if hill_short is not None else None,
            "hill_base": float(hill_base) if hill_base is not None else None,
            "tail_index_ratio": float(ratio) if ratio is not None else None,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["tail_index_returns"] = {
            symbol: list(history) for symbol, history in self._returns.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("tail_index_returns") if isinstance(state, dict) else None
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


# @register("strategy", "TailIndexRegimeRiderStrategy", interface="event_driven")  # applied atomically with the research_only tier hint in integration — do NOT apply earlier (live-safety)


# Per-timeframe candidate variants for TailIndexRegimeRiderStrategy, prepared
# for a later atomic wiring pass into strategy_factory.candidate_library (see
# the deferred @register comment above).  Mirrors the format of the existing
# _ADAPTIVE_TREND_RIDER_SLICE / _VOLATILITY_BREAKOUT_RIDER_SLICE tables:
# EXACTLY ONE variant per timeframe, windows scaled down and trail/hold tuned
# wider as the bar length grows so each timeframe still covers a comparable
# wall-clock lookback.
_TAIL_INDEX_REGIME_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "tail_window": 120,
            "recent_window": 24,
            "k_short": 8,
            "k_long": 24,
            "fatten_ratio": 0.85,
            "thin_ratio": 1.15,
            "trend_lookback": 20,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "tail_window": 112,
            "recent_window": 24,
            "k_short": 8,
            "k_long": 22,
            "fatten_ratio": 0.85,
            "thin_ratio": 1.15,
            "trend_lookback": 18,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "tail_window": 96,
            "recent_window": 20,
            "k_short": 7,
            "k_long": 20,
            "fatten_ratio": 0.85,
            "thin_ratio": 1.15,
            "trend_lookback": 14,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "tail_window": 80,
            "recent_window": 16,
            "k_short": 6,
            "k_long": 18,
            "fatten_ratio": 0.85,
            "thin_ratio": 1.15,
            "trend_lookback": 10,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


__all__ = [
    "TailIndexRegimeRiderStrategy",
]
