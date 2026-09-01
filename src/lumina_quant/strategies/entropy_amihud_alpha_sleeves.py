"""Permutation-entropy and Amihud-illiquidity gated trend riders.

Two per-symbol single-asset, return-first DIRECTIONAL sleeves that subclass the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and gate a trend ride on a fresh
signal the existing registry does not carry.  Both are wired ONLY at >=30m
timeframes and stay ``candidate_mix_type=="single"`` so they bypass the
multi-asset admission gate.

- :class:`PermutationEntropyTrendRiderStrategy`: OHLCV-only.  THEORY
  (Bandt-Pompe permutation entropy — an ordinal-pattern measure of time-series
  complexity / predictability).  A LOW normalized permutation entropy means the
  recent path is structured/predictable (a tradeable trend regime); a value near
  1 means it is essentially random.  The sleeve rides the prevailing trend ONLY
  when the normalized permutation entropy is BELOW ``pe_threshold`` (a predictable
  regime) AND the trend is confirmed; a high-entropy (random) regime SUPPRESSES
  the entry.  DISTINCT from :class:`HurstRegimeGatedStrategy` (rescaled-range
  Hurst), :class:`TrendEfficiencyMomentumStrategy` (Kaufman efficiency ratio),
  :class:`AdaptiveTrendRiderStrategy` (KAMA), and :class:`KalmanTrendRiderStrategy`
  (state-space slope): the gate is an INFORMATION-THEORETIC ordinal-pattern
  entropy, which no other sleeve computes.

- :class:`AmihudIlliquidityMomentumRiderStrategy`: OHLCV-only.  THEORY (Amihud
  2002 illiquidity premium — returns/momentum are stronger when illiquidity, the
  price impact per unit dollar volume, is elevated).  Using the existing
  :func:`amihud_illiquidity` proxy (``mean |log return| / dollar-volume``), the
  sleeve rides the prevailing trend ONLY when the current Amihud illiquidity is
  ELEVATED relative to its own rolling median (an illiquidity-premium regime) AND
  the trend is confirmed.  DISTINCT from :class:`LiquidityShockReversionStrategy` /
  :class:`OrderBookImbalanceReversionStrategy` (microstructure mean-reversion) and
  the pure trend riders (no liquidity conditioning): the gate is the Amihud
  illiquidity-premium regime, which no other sleeve uses.

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
from statistics import median
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import amihud_illiquidity
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


def _permutation_entropy(values: list[float], *, dim: int, delay: int = 1) -> float | None:
    """Return the normalized Bandt-Pompe permutation entropy in ``[0, 1]``.

    Counts the frequency of each ordinal pattern of ``dim`` consecutive values
    (spaced by ``delay``), takes the Shannon entropy of that distribution, and
    normalizes by ``log(dim!)``.  ``0`` means perfectly structured, ``1`` means
    maximally random.  Returns ``None`` when there are too few points.
    """
    m = max(2, int(dim))
    d = max(1, int(delay))
    n = len(values)
    span = (m - 1) * d
    last = n - span
    if last <= 0:
        return None
    counts: dict[tuple[int, ...], int] = {}
    total = 0
    for i in range(last):
        window = [values[i + k * d] for k in range(m)]
        # Ordinal pattern: the index order that sorts the window (ties by index).
        pattern = tuple(sorted(range(m), key=lambda k: window[k]))
        counts[pattern] = counts.get(pattern, 0) + 1
        total += 1
    if total <= 0:
        return None
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p)
    max_entropy = math.log(math.factorial(m))
    if max_entropy <= _EPS:
        return None
    return entropy / max_entropy


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


@register("strategy", "PermutationEntropyTrendRiderStrategy", interface="event_driven")
class PermutationEntropyTrendRiderStrategy(_ReturnRiderBase):
    """Ride the trend ONLY in a low-permutation-entropy (predictable) regime.

    The normalized Bandt-Pompe permutation entropy of the recent closes measures
    path complexity: low = structured/predictable, near 1 = random.  A LONG/SHORT
    ride is armed only when the entropy is BELOW ``pe_threshold`` AND the
    prevailing trend is confirmed; a high-entropy (random) regime suppresses entry.
    The winner is ridden with the inherited ATR trailing stop + pyramiding.

    DISTINCT from :class:`HurstRegimeGatedStrategy`,
    :class:`TrendEfficiencyMomentumStrategy`, :class:`AdaptiveTrendRiderStrategy`,
    and :class:`KalmanTrendRiderStrategy`: the regime gate is an
    information-theoretic ordinal-pattern entropy.  OHLCV-only, per-symbol
    single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "PermutationEntropyTrendRiderStrategy"
    strategy_id = "permutation_entropy_trend_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "pe_dim": HyperParam.integer("pe_dim", default=3, low=2, high=7, tunable=False),
            "pe_window": HyperParam.integer("pe_window", default=64, low=8, high=8192),
            "pe_threshold": HyperParam.floating("pe_threshold", default=0.85, low=0.0, high=1.0),
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

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.pe_dim = max(2, min(7, int(resolved["pe_dim"])))
        self.pe_window = max(8, int(resolved["pe_window"]))
        self.pe_threshold = max(0.0, min(1.0, float(resolved["pe_threshold"])))
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(
                int(resolved["pe_window"]),
                int(resolved["trend_lookback"]),
                int(resolved["trend_ma_window"]),
            )
            + 2,
        )

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _gated_direction(self, item: _RiderState) -> str:
        closes = list(item.closes)
        if len(closes) < self.pe_window:
            return ""
        pe = _permutation_entropy(closes[-self.pe_window :], dim=self.pe_dim)
        if pe is None or pe > self.pe_threshold:
            # Too random / not enough structure -> no ride.
            return ""
        trend = _trend_sign(
            closes,
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
        return self._gated_direction(item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        return self._gated_direction(item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        closes = list(item.closes)
        pe = (
            _permutation_entropy(closes[-self.pe_window :], dim=self.pe_dim)
            if len(closes) >= self.pe_window
            else None
        )
        return {
            "permutation_entropy": float(pe) if pe is not None else None,
            "pe_dim": int(self.pe_dim),
        }


@register("strategy", "AmihudIlliquidityMomentumRiderStrategy", interface="event_driven")
class AmihudIlliquidityMomentumRiderStrategy(_ReturnRiderBase):
    """Ride the trend ONLY when Amihud illiquidity is elevated (illiquidity premium).

    Using the Amihud proxy ``mean |log return| / dollar-volume``, a LONG/SHORT ride
    is armed only when the current illiquidity is ELEVATED relative to its own
    rolling median (``amihud >= illiquidity_rel * median``) AND the prevailing
    trend is confirmed — capturing the Amihud (2002) illiquidity premium where
    momentum persists more strongly.  The winner is ridden with the inherited ATR
    trailing stop + pyramiding.  Volume is read from each accepted bar; the sleeve
    degrades gracefully (no entries) until enough volume/illiquidity history.

    DISTINCT from :class:`LiquidityShockReversionStrategy` /
    :class:`OrderBookImbalanceReversionStrategy` (microstructure mean-reversion) and
    the pure trend riders (no liquidity conditioning): the gate is the Amihud
    illiquidity-premium regime.  OHLCV-only, per-symbol single-asset,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "AmihudIlliquidityMomentumRiderStrategy"
    strategy_id = "amihud_illiquidity_momentum_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "amihud_window": HyperParam.integer("amihud_window", default=48, low=4, high=8192),
            "amihud_history_window": HyperParam.integer(
                "amihud_history_window", default=64, low=4, high=8192
            ),
            "illiquidity_rel": HyperParam.floating(
                "illiquidity_rel", default=1.0, low=0.0, high=20.0
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
        vol_size = max(self.amihud_window, 2) + 8
        hist_size = max(self.amihud_history_window, 2) + 8
        self._volumes: dict[str, deque[float]] = {
            symbol: deque(maxlen=vol_size) for symbol in self.symbol_list
        }
        self._amihud_hist: dict[str, deque[float]] = {
            symbol: deque(maxlen=hist_size) for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.amihud_window = max(4, int(resolved["amihud_window"]))
        self.amihud_history_window = max(4, int(resolved["amihud_history_window"]))
        self.illiquidity_rel = max(0.0, float(resolved["illiquidity_rel"]))
        self.trend_lookback = max(3, int(resolved["trend_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(
                int(resolved["amihud_window"]),
                int(resolved["trend_lookback"]),
                int(resolved["trend_ma_window"]),
            )
            + 2,
        )

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Capture this bar's volume + refresh the trailing Amihud history BEFORE
        # the base runs ride+entry, mirroring the base time-key dedup.  At refresh
        # time item.closes holds bars up to the PREVIOUS one, so the Amihud computed
        # here (over closes+volumes seen so far) uses only closed bars.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._refresh_amihud(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _refresh_amihud(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        volume = safe_float(snapshot.volume)
        vols = self._volumes.setdefault(symbol, deque(maxlen=max(self.amihud_window, 2) + 8))
        vols.append(float(volume) if volume is not None and volume > 0.0 else 0.0)
        # closes seen so far + this bar's close gives the series the base will hold.
        closes = [*list(item.closes), close]
        amihud = amihud_illiquidity(closes, list(vols), window=self.amihud_window)
        if amihud is not None and math.isfinite(amihud):
            self._amihud_hist.setdefault(
                symbol, deque(maxlen=max(self.amihud_history_window, 2) + 8)
            ).append(float(amihud))

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _illiquid_regime(self, symbol: str) -> bool:
        history = self._amihud_hist.get(symbol)
        if not history or len(history) < max(2, self.amihud_history_window // 2):
            return False
        current = history[-1]
        med = median(history)
        if med <= _EPS:
            return False
        return current >= self.illiquidity_rel * med

    def _gated_direction(self, symbol: str, item: _RiderState) -> str:
        if not self._illiquid_regime(symbol):
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
        history = self._amihud_hist.get(symbol) if symbol is not None else None
        current = history[-1] if history else None
        med = median(history) if history else None
        return {
            "amihud_illiquidity": float(current) if current is not None else None,
            "amihud_median": float(med) if med is not None else None,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["amihud_state"] = {
            symbol: {
                "volumes": list(self._volumes.get(symbol, ())),
                "amihud_hist": list(self._amihud_hist.get(symbol, ())),
            }
            for symbol in self._state
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("amihud_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._volumes or not isinstance(payload, dict):
                continue
            vols = self._volumes[symbol]
            vols.clear()
            for value in list(payload.get("volumes") or [])[-int(vols.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    vols.append(parsed)
            hist = self._amihud_hist[symbol]
            hist.clear()
            for value in list(payload.get("amihud_hist") or [])[-int(hist.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None:
                    hist.append(parsed)


__all__ = [
    "AmihudIlliquidityMomentumRiderStrategy",
    "PermutationEntropyTrendRiderStrategy",
]
