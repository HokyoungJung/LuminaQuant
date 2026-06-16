"""Robustness meta-overlay alpha sleeves (regime + confidence gating).

The classes in this module are standalone, self-contained, param-driven
event-driven strategies (NOT wrappers over other registered strategies and NOT
portfolio-sets): the codebase has no strategy-wrapping composition primitive, so
each overlay embeds its own primary signal and selects/sizes it via an internal
gate.  Both register as drop-in candidates and remain data-local (no data
fetching, no artifact writes, no hidden environment state).  They must still be
validated on the data-bearing research machine.

- ``HurstRegimeGatedStrategy`` (S9): a multi-symbol basket that switches between
  an internal trend child (EMA20-EMA60 sign) and an internal mean-reversion child
  (z-score band) according to a rolling Hurst-exponent regime gate with
  enter/exit hysteresis (so it does not thrash at H~0.5); the mid-band blends both
  children at a configurable weight.  Rebalances the active book only on a banded
  score-delta to keep turnover bounded.
- ``ConfidenceGatedTrendStrategy`` (S13): a per-symbol single-asset rule-based
  meta-gate (NO ML).  Each candidate trades exactly one symbol, so it stays
  ``candidate_mix_type=="single"`` and outside the multi-asset admission gate.  It
  executes a trend primary only when a heuristic confidence score (built from a
  funding-rate z-score, a short/long realized-vol ratio, and a recent win-rate)
  clears a threshold, sizes the position by that confidence, and bands rebalances
  to suppress churn.  Fewer, higher-conviction trades lift deflated Sharpe and
  lower PBO.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    rolling_zscore,
    simple_return,
    volatility_ratio,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import hurst_exponent
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


def _restore_deque(target: deque[float], payload: Any) -> None:
    target.clear()
    for value in list(payload or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


def _mode(raw: Any) -> str:
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _ema(values: list[float], span: int) -> float | None:
    """Return the exponential moving average over the trailing ``span`` window."""
    if span <= 0 or len(values) < span:
        return None
    alpha = 2.0 / (float(span) + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    if not isinstance(ema, float):
        ema = float(ema)
    return ema if ema == ema else None  # NaN guard


@dataclass(slots=True)
class _RegimeState:
    closes: deque[float]
    mode: str = "OUT"
    regime: str = "MID"
    entry_price: float | None = None
    target_score: float = 0.0
    bars_held: int = 0
    last_time_key: str = ""


@dataclass(slots=True)
class _ConfidenceState:
    closes: deque[float]
    returns: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    target_score: float = 0.0
    bars_held: int = 0
    last_time_key: str = ""


@register("strategy", "HurstRegimeGatedStrategy", interface="event_driven")
class HurstRegimeGatedStrategy(Strategy):
    """Basket overlay that gates a trend child vs a mean-reversion child by Hurst.

    Per symbol it estimates a rolling Hurst exponent; H above ``trend_threshold``
    enables the trend child (EMA-fast minus EMA-slow sign), H below
    ``mr_threshold`` enables the mean-reversion child (faded z-score band), and the
    mid-band blends both children at ``mid_weight``.  Hysteresis (separate enter
    and exit Hurst thresholds derived from the gap) prevents regime thrash near
    H~0.5.  The active book is only rebalanced when the per-symbol score moves by
    more than ``rebalance_band`` to keep turnover bounded.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "hurst_window": HyperParam.integer("hurst_window", default=60, low=20, high=2000),
            "trend_threshold": HyperParam.floating(
                "trend_threshold", default=0.55, low=0.5, high=0.95
            ),
            "mr_threshold": HyperParam.floating("mr_threshold", default=0.45, low=0.05, high=0.5),
            "hysteresis": HyperParam.floating("hysteresis", default=0.03, low=0.0, high=0.25),
            "mid_weight": HyperParam.floating("mid_weight", default=0.3, low=0.0, high=1.0),
            "ema_fast": HyperParam.integer("ema_fast", default=20, low=2, high=2000),
            "ema_slow": HyperParam.integer("ema_slow", default=60, low=4, high=4000),
            "zscore_window": HyperParam.integer("zscore_window", default=60, low=8, high=4000),
            "entry_z": HyperParam.floating("entry_z", default=1.5, low=0.25, high=10.0),
            "lookback_bars": HyperParam.integer("lookback_bars", default=120, low=8, high=10080),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.20, low=0.0, high=5.0
            ),
            "max_positions": HyperParam.integer("max_positions", default=6, low=1, high=128),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.060, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=720, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=False, grid=[True, False]),
            "min_symbols": HyperParam.integer(
                "min_symbols", default=4, low=1, high=128, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.hurst_window = max(8, int(resolved["hurst_window"]))
        self.trend_threshold = max(0.5, min(0.99, float(resolved["trend_threshold"])))
        self.mr_threshold = max(0.01, min(0.5, float(resolved["mr_threshold"])))
        self.hysteresis = max(0.0, min(0.25, float(resolved["hysteresis"])))
        self.mid_weight = max(0.0, min(1.0, float(resolved["mid_weight"])))
        self.ema_fast = max(2, int(resolved["ema_fast"]))
        self.ema_slow = max(self.ema_fast + 1, int(resolved["ema_slow"]))
        self.zscore_window = max(4, int(resolved["zscore_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.lookback_bars = max(4, int(resolved["lookback_bars"]))
        self.rebalance_band = max(0.0, float(resolved["rebalance_band"]))
        self.max_positions = max(1, int(resolved["max_positions"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(1, int(resolved["min_symbols"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.hurst_window,
                self.ema_slow,
                self.zscore_window,
                self.lookback_bars,
                self.max_hold_bars,
            )
            + 8
        )
        self._state = {symbol: _RegimeState(deque(maxlen=size)) for symbol in self.symbol_list}
        self._last_eval_time_key = ""

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "regime": item.regime,
                    "entry_price": item.entry_price,
                    "target_score": float(item.target_score),
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            _restore_deque(item.closes, payload.get("closes"))
            item.mode = _mode(payload.get("mode"))
            regime = str(payload.get("regime", "MID")).upper()
            item.regime = regime if regime in {"TREND", "MR", "MID"} else "MID"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.target_score = float(safe_float(payload.get("target_score")) or 0.0)
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._rebalance(snapshot.time)

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        return True

    def _resolve_regime(self, prior: str, hurst: float | None) -> str:
        """Apply enter/exit hysteresis around the trend/MR thresholds."""
        if hurst is None:
            return "MID"
        trend_exit = self.trend_threshold - self.hysteresis
        mr_exit = self.mr_threshold + self.hysteresis
        if prior == "TREND" and hurst >= trend_exit:
            return "TREND"
        if prior == "MR" and hurst <= mr_exit:
            return "MR"
        if hurst >= self.trend_threshold:
            return "TREND"
        if hurst <= self.mr_threshold:
            return "MR"
        return "MID"

    def _child_scores(self, closes: list[float]) -> tuple[float | None, float | None]:
        """Return (trend_score, mr_score); ``None`` legs lack history."""
        trend_score: float | None = None
        ema_fast = (
            _ema(closes[-self.ema_slow :], self.ema_fast) if len(closes) >= self.ema_slow else None
        )
        ema_slow = (
            _ema(closes[-self.ema_slow :], self.ema_slow) if len(closes) >= self.ema_slow else None
        )
        if ema_fast is not None and ema_slow is not None and ema_slow > _EPS:
            trend_score = (ema_fast - ema_slow) / max(abs(ema_slow), _EPS)
        mr_score: float | None = None
        z = rolling_zscore(closes, window=self.zscore_window)
        if z is not None and self.entry_z > 0.0:
            # Fade extremes: positive score means "go long" (price below mean).
            mr_score = -z / max(self.entry_z, _EPS)
        return trend_score, mr_score

    def _score_symbol(self, item: _RegimeState) -> tuple[float, str] | None:
        closes = list(item.closes)
        if len(closes) < max(self.hurst_window, self.ema_slow, self.zscore_window) + 1:
            return None
        hurst = hurst_exponent(
            closes[-(self.hurst_window + 1) :],
            min_lag=2,
            max_lag=max(3, min(20, self.hurst_window // 3)),
        )
        regime = self._resolve_regime(item.regime, hurst)
        item.regime = regime
        trend_score, mr_score = self._child_scores(closes)
        if regime == "TREND":
            if trend_score is None:
                return None
            return float(trend_score), "trend"
        if regime == "MR":
            if mr_score is None:
                return None
            return float(mr_score), "mr"
        # Mid-band: blend both children at mid_weight, skip if neither ready.
        parts: list[float] = []
        if trend_score is not None:
            parts.append(trend_score)
        if mr_score is not None:
            parts.append(mr_score)
        if not parts:
            return None
        if trend_score is not None and mr_score is not None:
            blended = (1.0 - self.mid_weight) * trend_score + self.mid_weight * mr_score
        else:
            blended = self.mid_weight * parts[0]
        return float(blended), "mid"

    def _rebalance(self, event_time: Any) -> None:
        rows: list[tuple[float, str, str]] = []
        for symbol, item in self._state.items():
            scored = self._score_symbol(item)
            if scored is None:
                continue
            score, child = scored
            rows.append((score, symbol, child))
        if len(rows) < self.min_symbols:
            self._age_positions(event_time)
            return
        longs = sorted((row for row in rows if row[0] > 0.0), key=lambda r: r[0], reverse=True)
        shorts: list[tuple[float, str, str]] = []
        if self.allow_short:
            shorts = sorted((row for row in rows if row[0] < 0.0), key=lambda r: r[0])
        targets: dict[str, tuple[str, float, str]] = {}
        for score, symbol, child in longs[: self.max_positions]:
            targets[symbol] = ("LONG", score, child)
        for score, symbol, child in shorts[: self.max_positions]:
            targets[symbol] = ("SHORT", score, child)
        for symbol, item in self._state.items():
            price = item.closes[-1] if item.closes else None
            target = targets.get(symbol)
            if target is None:
                self._exit_position(symbol, item, event_time, price, "rebalance_removed")
                continue
            target_mode, score, child = target
            if item.mode == target_mode and abs(score - item.target_score) < self.rebalance_band:
                # Hold: score delta within band, no churn.
                item.bars_held += 1
                continue
            if item.mode != "OUT" and item.mode != target_mode:
                self._exit_position(symbol, item, event_time, price, "side_flip")
            self._enter_position(symbol, item, event_time, price, target_mode, score, child)
        # Age any non-target positions through stop/max-hold.
        for symbol, item in self._state.items():
            if symbol in targets or item.mode == "OUT":
                continue
            self._maybe_stop(symbol, item, event_time)

    def _enter_position(
        self,
        symbol: str,
        item: _RegimeState,
        event_time: Any,
        price: float | None,
        target_mode: str,
        score: float,
        child: str,
    ) -> None:
        stop_loss = None
        if price is not None and self.stop_loss_pct > 0.0:
            stop_loss = price * (
                1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
            )
        metadata = _target_metadata(
            strategy="HurstRegimeGatedStrategy",
            target_allocation=max(0.0, self.target_allocation / float(max(1, self.max_positions))),
            max_order_value=self.max_order_value,
            score=score,
            regime=item.regime,
            child=child,
            target_mode=target_mode,
        )
        _emit(
            self.events,
            strategy_id="hurst_regime_gated",
            symbol=symbol,
            event_time=event_time,
            signal_type=target_mode,
            strength=max(0.25, min(3.0, abs(score))),
            price=price,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = target_mode
        item.entry_price = price
        item.target_score = float(score)
        item.bars_held = 0

    def _exit_position(
        self,
        symbol: str,
        item: _RegimeState,
        event_time: Any,
        price: float | None,
        reason: str,
    ) -> None:
        if item.mode == "OUT":
            return
        _emit(
            self.events,
            strategy_id="hurst_regime_gated",
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={"strategy": "HurstRegimeGatedStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.target_score = 0.0
        item.bars_held = 0

    def _maybe_stop(self, symbol: str, item: _RegimeState, event_time: Any) -> None:
        if item.mode == "OUT":
            return
        price = item.closes[-1] if item.closes else None
        item.bars_held += 1
        stop = False
        if price is not None and item.entry_price is not None and self.stop_loss_pct > 0.0:
            if item.mode == "LONG" and price <= item.entry_price * (1.0 - self.stop_loss_pct):
                stop = True
            if item.mode == "SHORT" and price >= item.entry_price * (1.0 + self.stop_loss_pct):
                stop = True
        if stop or item.bars_held >= self.max_hold_bars:
            self._exit_position(
                symbol, item, event_time, price, "stop_loss" if stop else "max_hold"
            )

    def _age_positions(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            self._maybe_stop(symbol, item, event_time)


@register("strategy", "ConfidenceGatedTrendStrategy", interface="event_driven")
class ConfidenceGatedTrendStrategy(Strategy):
    """Per-symbol trend sleeve executed only when a rule-based confidence clears a bar.

    This is a single-asset overlay (one symbol per candidate) so it is
    ``candidate_mix_type=="single"`` and outside the multi-asset admission gate.
    The trend primary (EMA-fast minus EMA-slow sign) only fires when a heuristic
    confidence score -- built from a funding-rate z-score (calm funding favors
    continuation), a short/long realized-vol ratio (low vol favors trend), and a
    recent directional win-rate -- exceeds ``confidence_threshold``.  Position size
    scales with that confidence and a ``rebalance_band`` on the primary score
    suppresses churn.  No machine learning is involved.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "ema_fast": HyperParam.integer("ema_fast", default=20, low=2, high=2000),
            "ema_slow": HyperParam.integer("ema_slow", default=60, low=4, high=4000),
            "confidence_threshold": HyperParam.floating(
                "confidence_threshold", default=0.55, low=0.0, high=1.0
            ),
            "vol_ratio_short_window": HyperParam.integer(
                "vol_ratio_short_window", default=16, low=2, high=4096
            ),
            "vol_ratio_long_window": HyperParam.integer(
                "vol_ratio_long_window", default=96, low=4, high=20000
            ),
            "max_vol_ratio": HyperParam.floating("max_vol_ratio", default=1.5, low=0.1, high=20.0),
            "funding_z_window": HyperParam.integer(
                "funding_z_window", default=96, low=8, high=4096
            ),
            "max_funding_z": HyperParam.floating("max_funding_z", default=2.0, low=0.1, high=20.0),
            "win_rate_window": HyperParam.integer("win_rate_window", default=24, low=4, high=4096),
            "trend_lookback_bars": HyperParam.integer(
                "trend_lookback_bars", default=24, low=1, high=10080
            ),
            "lookback_bars": HyperParam.integer("lookback_bars", default=120, low=8, high=10080),
            "rebalance_band": HyperParam.floating(
                "rebalance_band", default=0.15, low=0.0, high=5.0
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.040, low=0.0, high=0.50
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.0, low=0.0, high=1.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=480, low=1, high=200000),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.ema_fast = max(2, int(resolved["ema_fast"]))
        self.ema_slow = max(self.ema_fast + 1, int(resolved["ema_slow"]))
        self.confidence_threshold = max(0.0, min(1.0, float(resolved["confidence_threshold"])))
        self.vol_ratio_short_window = max(2, int(resolved["vol_ratio_short_window"]))
        self.vol_ratio_long_window = max(
            self.vol_ratio_short_window + 1, int(resolved["vol_ratio_long_window"])
        )
        self.max_vol_ratio = max(0.1, float(resolved["max_vol_ratio"]))
        self.funding_z_window = max(4, int(resolved["funding_z_window"]))
        self.max_funding_z = max(0.1, float(resolved["max_funding_z"]))
        self.win_rate_window = max(2, int(resolved["win_rate_window"]))
        self.trend_lookback_bars = max(1, int(resolved["trend_lookback_bars"]))
        self.lookback_bars = max(4, int(resolved["lookback_bars"]))
        self.rebalance_band = max(0.0, float(resolved["rebalance_band"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.ema_slow,
                self.vol_ratio_long_window,
                self.funding_z_window,
                self.win_rate_window,
                self.lookback_bars,
                self.max_hold_bars,
            )
            + 8
        )
        self._state = {
            symbol: _ConfidenceState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._funding: dict[str, deque[float]] = {
            symbol: deque(maxlen=self.funding_z_window + 8) for symbol in self.symbol_list
        }

    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        """Own 3-arg feature accessor (event -> latest feature -> latest bar)."""
        direct = safe_float(getattr(event, field, None))
        if direct is not None:
            return direct
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

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "returns": list(item.returns),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "target_score": float(item.target_score),
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "funding": list(self._funding.get(symbol, deque())),
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            _restore_deque(item.closes, payload.get("closes"))
            _restore_deque(item.returns, payload.get("returns"))
            item.mode = _mode(payload.get("mode"))
            item.entry_price = safe_float(payload.get("entry_price"))
            item.target_score = float(safe_float(payload.get("target_score")) or 0.0)
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))
            if symbol in self._funding:
                _restore_deque(self._funding[symbol], payload.get("funding"))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._process_symbol(event, symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                self._process_symbol(event, str(symbol), snapshot)

    def _confidence(self, event: Any, symbol: str, item: _ConfidenceState) -> float | None:
        """Combine funding-calm, low-vol, and recent win-rate into [0, 1]."""
        components: list[float] = []
        # Funding z-score: calm funding (small |z|) favors trend continuation.
        funding = self._extract_feature(event, symbol, "funding_rate")
        funding_dq = self._funding.get(symbol)
        if funding is not None and funding_dq is not None:
            funding_dq.append(float(funding))
            z = rolling_zscore(funding_dq, window=self.funding_z_window)
            if z is not None:
                components.append(max(0.0, 1.0 - abs(z) / max(self.max_funding_z, _EPS)))
        # Vol ratio: low short/long realized vol favors a stable trend.
        vr = volatility_ratio(
            item.closes,
            fast_window=self.vol_ratio_short_window,
            slow_window=self.vol_ratio_long_window,
        )
        if vr is not None:
            components.append(max(0.0, 1.0 - vr / max(self.max_vol_ratio, _EPS)))
        # Recent win-rate: fraction of finite recent returns in the trend direction.
        rets = list(item.returns)[-self.win_rate_window :]
        if len(rets) >= 2:
            ups = sum(1 for value in rets if value > 0.0)
            win_rate = ups / float(len(rets))
            components.append(win_rate)
        if not components:
            return None
        return max(0.0, min(1.0, sum(components) / float(len(components))))

    def _process_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        prev_close = item.closes[-1] if item.closes else None
        item.closes.append(close)
        if prev_close is not None and prev_close > 0.0:
            item.returns.append(float(close / prev_close - 1.0))
        # Exit management first.
        self._maybe_exit(symbol, item, snapshot, close)
        closes = list(item.closes)
        if len(closes) < self.ema_slow + 1:
            return
        ema_fast = _ema(closes[-self.ema_slow :], self.ema_fast)
        ema_slow = _ema(closes[-self.ema_slow :], self.ema_slow)
        if ema_fast is None or ema_slow is None or ema_slow <= _EPS:
            return
        trend = (ema_fast - ema_slow) / max(abs(ema_slow), _EPS)
        mom = simple_return(closes, lookback=self.trend_lookback_bars)
        direction = ""
        if trend > 0.0 and (mom is None or mom >= 0.0):
            direction = "LONG"
        elif self.allow_short and trend < 0.0 and (mom is None or mom <= 0.0):
            direction = "SHORT"
        if not direction:
            return
        confidence = self._confidence(event, symbol, item)
        if confidence is None or confidence < self.confidence_threshold:
            return
        score = float(confidence if direction == "LONG" else -confidence)
        if item.mode == direction and abs(score - item.target_score) < self.rebalance_band:
            return
        if item.mode != "OUT" and item.mode != direction:
            self._emit_exit(symbol, item, snapshot.time, close, "side_flip")
        self._enter(symbol, item, snapshot, close, direction, confidence, trend)

    def _enter(
        self,
        symbol: str,
        item: _ConfidenceState,
        snapshot: _Snapshot,
        close: float,
        direction: str,
        confidence: float,
        trend: float,
    ) -> None:
        alloc = self.target_allocation * max(0.0, min(1.0, confidence))
        stop_loss = None
        take_profit = None
        if self.stop_loss_pct > 0.0:
            stop_loss = close * (
                1.0 - self.stop_loss_pct if direction == "LONG" else 1.0 + self.stop_loss_pct
            )
        if self.take_profit_pct > 0.0:
            take_profit = close * (
                1.0 + self.take_profit_pct if direction == "LONG" else 1.0 - self.take_profit_pct
            )
        metadata = _target_metadata(
            strategy="ConfidenceGatedTrendStrategy",
            target_allocation=max(0.0, alloc),
            max_order_value=self.max_order_value,
            confidence=confidence,
            trend=trend,
            target_mode=direction,
        )
        _emit(
            self.events,
            strategy_id="confidence_gated_trend",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=direction,
            strength=max(0.25, min(3.0, confidence * 3.0)),
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )
        item.mode = direction
        item.entry_price = close
        item.target_score = float(confidence if direction == "LONG" else -confidence)
        item.bars_held = 0

    def _maybe_exit(
        self, symbol: str, item: _ConfidenceState, snapshot: _Snapshot, close: float
    ) -> None:
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None:
            return
        item.bars_held += 1
        reason = ""
        if item.mode == "LONG":
            if self.stop_loss_pct > 0.0 and close <= item.entry_price * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close >= item.entry_price * (
                1.0 + self.take_profit_pct
            ):
                reason = "take_profit"
        else:
            if self.stop_loss_pct > 0.0 and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close <= item.entry_price * (
                1.0 - self.take_profit_pct
            ):
                reason = "take_profit"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if reason:
            self._emit_exit(symbol, item, snapshot.time, close, reason)

    def _emit_exit(
        self,
        symbol: str,
        item: _ConfidenceState,
        event_time: Any,
        close: float | None,
        reason: str,
    ) -> None:
        _emit(
            self.events,
            strategy_id="confidence_gated_trend",
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "ConfidenceGatedTrendStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.target_score = 0.0
        item.bars_held = 0


__all__ = [
    "ConfidenceGatedTrendStrategy",
    "HurstRegimeGatedStrategy",
]
