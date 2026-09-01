"""Volatility-term-structure recovery and breadth-gated trend-timer sleeves.

These two event-driven sleeves are deliberately data-local (no I/O, no hidden
configuration bus) and reuse the proven shared primitives so they slot into the
existing candidate pipeline as drop-in directional research hypotheses:

- :class:`RealizedVolTermStructureStrategy`: a per-symbol single-asset sleeve
  that keys on the REALIZED-VOLATILITY TERM-STRUCTURE ratio
  ``RV_s / RV_l`` (short-horizon realized vol over a short window — optionally
  micro-informed by the finer intrabar 1s tape inside the latest decision
  window — divided by the long-horizon realized vol on decision bars).  When the
  ratio spikes far above 1 (a vol shock / backwardated term structure = a panic
  vol spike) crypto tends to V-rebound and vol normalizes, so the sleeve enters
  LONG the recovery sized by the shock magnitude and RIDES it with the shared
  :class:`_ReturnRiderBase` ATR trailing stop; it can optionally FADE an extreme
  UPSIDE vol spike short when ``fade_upside_short`` is set.  It exits on vol
  normalization (ratio falls back toward ``exit_ratio``), the trailing stop, or
  ``max_hold_bars``.  This is DISTINCT from
  :class:`PanicReboundMeanReversionStrategy` (which triggers on a price-drawdown
  + volume shock, not the vol ratio), from :class:`HourlyShockReversionStrategy`
  (single completed-bar price shock), and from vol-managed momentum overlays
  (which only SIZE an existing momentum signal by vol) — here the RV term
  structure ratio IS the entry signal.  OHLCV-only, >= 30m only.

- :class:`BreadthRegimeTrendTimerStrategy`: a top-down market-timing sleeve that
  scales TOTAL net-long directional exposure by cross-sectional BREADTH.  Each
  decision bar it computes breadth = the fraction of the basket trading above its
  own trend (``close > SMA(trend_window)`` and/or a positive simple return over
  ``trend_window``).  Strong breadth (>= ``risk_on_breadth``) holds a long basket
  of the up-trending names with a breadth-scaled gross; weak/collapsing breadth
  (<= ``risk_off_breadth``) reduces or flattens the book (risk-off).  Targets are
  emitted through the shared cross-sectional rebalancer.  This is DISTINCT from
  :class:`CrossAssetDiversifiedTrendStrategy` (inverse-vol risk-parity weights +
  portfolio vol-target, but NO breadth gate on TOTAL exposure), from
  :class:`DualMomentumIndexRotationStrategy` (single-instrument absolute/relative
  rotation), and from TopCap TSMOM (no breadth gate) — the novelty is
  BREADTH-gated total exposure, i.e. a regime risk-on/off timer.  Multi-symbol
  basket -> ``family="cross_sectional"`` with admission tags.  >= 30m only.

Both sleeves trade through ``_emit`` / ``_target_metadata``, keep bounded deques,
are ``None``-safe, and never raise from ``calculate_signals``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    realized_volatility,
    simple_return,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _ranked_targets,
    _state_size,
)
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
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _RiderCommon,
    _RiderState,
    _ReturnRiderBase,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


_PARKINSON_RANGE_TO_SIGMA = 1.0 / (2.0 * math.sqrt(math.log(2.0)))


def _micro_realized_volatility(snapshot: _Snapshot) -> float | None:
    """Return a micro-informed per-bar volatility estimate from the latest window.

    The window snapshot already aggregates the finer (1s) intrabar tape of the
    latest decision window into a single OHLC bar; we convert that bar's high-low
    range into a Parkinson single-bar volatility estimate
    ``sigma = log(high/low) / (2*sqrt(ln 2))``.  That puts the micro short-leg on
    the SAME per-bar units as the close-to-close realized volatility used for the
    long leg, so the term-structure ratio stays meaningful.  Returns ``None`` when
    the range is unavailable so the caller falls back to the close-to-close short
    window.
    """
    if snapshot is None:
        return None
    high = safe_float(snapshot.high)
    low = safe_float(snapshot.low)
    if high is None or low is None or high <= _EPS or low <= _EPS or high < low:
        return None
    ratio = high / low
    if ratio <= 1.0:
        return 0.0
    return float(math.log(ratio) * _PARKINSON_RANGE_TO_SIGMA)


@register("strategy", "RealizedVolTermStructureStrategy", interface="event_driven")
class RealizedVolTermStructureStrategy(_ReturnRiderBase):
    """Ride the V-recovery after a realized-vol-term-structure backwardation spike.

    Per symbol the sleeve maintains a bounded close history and computes the
    short-horizon realized vol ``RV_s`` over ``short_window`` (optionally blended
    with a micro intrabar range proxy from the latest window when
    ``use_micro_short`` is set) and the long-horizon realized vol ``RV_l`` over
    ``long_window``.  The term-structure ratio ``RV_s / RV_l`` measures vol
    backwardation: a ratio far above 1 is a panic vol spike that historically
    mean-reverts as crypto V-rebounds.

    Entry: when ``ratio >= shock_ratio`` (a vol-shock backwardation) the sleeve
    enters LONG the recovery and rides it with the inherited ATR trailing stop.
    When ``fade_upside_short`` is set and the spike coincides with a strong
    UPSIDE move over ``short_window`` (return >= ``upside_return``), it instead
    FADES the blow-off SHORT.  The shock magnitude scales sizing through the
    inherited vol-target.

    Exit: the inherited ride logic handles the trailing stop and ``max_hold_bars``;
    additionally the position is flattened on vol normalization once the ratio
    falls back to ``exit_ratio`` (term structure re-flattens).  OHLCV-only,
    per-symbol single-asset, ``decision_cadence_seconds >= 1800``.
    """

    strategy_name = "RealizedVolTermStructureStrategy"
    strategy_id = "realized_vol_term_structure"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "short_window": HyperParam.integer("short_window", default=6, low=2, high=4096),
            "long_window": HyperParam.integer("long_window", default=48, low=4, high=20000),
            "shock_ratio": HyperParam.floating("shock_ratio", default=1.8, low=1.0, high=20.0),
            "exit_ratio": HyperParam.floating("exit_ratio", default=1.1, low=0.0, high=20.0),
            "use_micro_short": HyperParam.boolean(
                "use_micro_short", default=True, grid=[True, False]
            ),
            "fade_upside_short": HyperParam.boolean(
                "fade_upside_short", default=False, grid=[True, False]
            ),
            "upside_return": HyperParam.floating("upside_return", default=0.04, low=0.0, high=1.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.025, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=120, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=False, grid=[True, False]),
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
        self.short_window = max(2, int(resolved["short_window"]))
        self.long_window = max(self.short_window + 1, int(resolved["long_window"]))
        self.shock_ratio = max(1.0, float(resolved["shock_ratio"]))
        self.exit_ratio = max(0.0, float(resolved["exit_ratio"]))
        self.use_micro_short = bool(resolved["use_micro_short"])
        self.fade_upside_short = bool(resolved["fade_upside_short"])
        self.upside_return = max(0.0, float(resolved["upside_return"]))
        # Latest per-symbol micro short-leg vol proxy (refreshed each window).
        self._micro_short: dict[str, float] = {}

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.long_window, self.short_window) + 2,
        )

    def _ratio_for(self, item: _RiderState, symbol: str) -> float | None:
        """Return the realized-vol term-structure ratio ``RV_s / RV_l`` or ``None``."""
        closes = list(item.closes)
        rv_long = realized_volatility(closes, window=self.long_window)
        if rv_long is None or rv_long <= _EPS:
            return None
        rv_short = realized_volatility(closes, window=self.short_window)
        micro = self._micro_short.get(symbol)
        if self.use_micro_short and micro is not None and micro > 0.0:
            # Blend the micro intrabar-range proxy into the short leg so a fresh
            # panic spike inside the latest window is reflected immediately. Take
            # the more-informed (larger) of the two so a real spike is not damped.
            rv_short = micro if rv_short is None else max(rv_short, micro)
        if rv_short is None or rv_short <= _EPS:
            return None
        return float(rv_short / rv_long)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Refresh the micro short-leg vol proxy BEFORE the dedup/append in the
        # base loop runs (the base keys on time and short-circuits on repeats).
        micro = _micro_realized_volatility(snapshot)
        if micro is not None:
            self._micro_short[symbol] = micro
        item = self._state.get(symbol)
        if item is not None and item.mode in {"LONG", "SHORT"}:
            ratio = self._ratio_for(item, symbol)
            if ratio is not None and ratio <= self.exit_ratio:
                # Vol normalized: flatten the recovery leg cleanly.
                close = safe_float(snapshot.close)
                key = time_key(snapshot.time)
                if key and key != item.last_time_key and close is not None:
                    item.last_time_key = key
                    self._append_snapshot(item, snapshot, close)
                    self._emit_exit(symbol, item, snapshot.time, close, "vol_normalized")
                    return
        super()._process_symbol(symbol, snapshot)

    def _entry_decision(self, item: _RiderState) -> str:
        # The base loop already mapped this symbol; recover the key from the
        # latest stored time so the micro proxy lookup stays per-symbol correct.
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        ratio = self._ratio_for(item, symbol)
        if ratio is None or ratio < self.shock_ratio:
            return ""
        if self.fade_upside_short and self.allow_short:
            ret = simple_return(list(item.closes), lookback=self.short_window)
            if ret is not None and ret >= self.upside_return:
                return "SHORT"
        return "LONG"

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        ratio = self._ratio_for(item, symbol)
        # Keep compounding the recovery only while the term structure remains
        # backwardated (ratio still elevated above the exit threshold).
        return ratio is not None and ratio > self.exit_ratio

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        ratio = self._ratio_for(item, symbol) if symbol is not None else None
        return {
            "vol_term_ratio": float(ratio) if ratio is not None else None,
            "shock_ratio": float(self.shock_ratio),
        }

    def _symbol_for(self, item: _RiderState) -> str | None:
        """Return the symbol owning ``item`` (identity match on the state map)."""
        for symbol, state in self._state.items():
            if state is item:
                return symbol
        return None

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["micro_short"] = dict(self._micro_short)
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("micro_short") if isinstance(state, dict) else None
        if isinstance(raw, dict):
            for symbol, value in raw.items():
                parsed = safe_float(value)
                if symbol in self._state and parsed is not None and parsed >= 0.0:
                    self._micro_short[str(symbol)] = float(parsed)


def _pack_cross(item: _CrossSectionalState) -> dict[str, Any]:
    return {
        "closes": list(item.closes),
        "volumes": list(item.volumes),
        "mode": item.mode,
        "entry_price": item.entry_price,
        "bars_held": int(item.bars_held),
        "last_time_key": item.last_time_key,
    }


def _restore_cross(item: _CrossSectionalState, payload: dict[str, Any]) -> None:
    _restore_deque(item.closes, payload.get("closes"))
    _restore_deque(item.volumes, payload.get("volumes"))
    item.mode = _mode(payload.get("mode"))
    item.entry_price = safe_float(payload.get("entry_price"))
    item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
    item.last_time_key = str(payload.get("last_time_key", ""))


@register("strategy", "BreadthRegimeTrendTimerStrategy", interface="event_driven")
class BreadthRegimeTrendTimerStrategy(Strategy):
    """Breadth-gated total-exposure trend timer over a crypto basket.

    Each decision bar the sleeve computes cross-sectional BREADTH: the fraction
    of the basket trading above its own trend, where a name is "above trend" when
    its ``close > SMA(trend_window)`` AND (if ``require_positive_return``) its
    simple return over ``trend_window`` is positive.  Breadth in ``[0, 1]`` is the
    regime gauge:

    - ``breadth >= risk_on_breadth``: RISK-ON.  Hold a long basket of the
      up-trending names, with TOTAL gross exposure scaled by breadth
      (``gross = max_gross * breadth``) so a broadly strong tape carries more
      risk than a marginal one.
    - ``risk_off_breadth < breadth < risk_on_breadth``: stay in the current
      regime (hysteresis) — no risk-on/off flip is triggered.  An existing long
      book is re-formed and re-scaled to the current (lower) breadth on
      rebalance ticks (drop-outs rotate out, gross tracks breadth) and merely
      aged on non-rebalance ticks; a flat book stays flat until breadth clears
      ``risk_on_breadth``.
    - ``breadth <= risk_off_breadth``: RISK-OFF.  Flatten the book.

    The up-trending names are ranked by trend strength and the top
    ``max_positions`` become the long basket; per-name allocation is the
    breadth-scaled gross split evenly across the held names.  Targets are emitted
    through the shared cross-sectional rebalancer helpers.  The breadth gate on
    TOTAL exposure (not just per-name selection) is the novelty: it is a regime
    risk-on/off TIMER, not an inverse-vol risk-parity book.  Multi-symbol basket,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    strategy_name = "BreadthRegimeTrendTimerStrategy"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "trend_window": HyperParam.integer("trend_window", default=50, low=4, high=20000),
            "risk_on_breadth": HyperParam.floating(
                "risk_on_breadth", default=0.60, low=0.0, high=1.0
            ),
            "risk_off_breadth": HyperParam.floating(
                "risk_off_breadth", default=0.40, low=0.0, high=1.0
            ),
            "require_positive_return": HyperParam.boolean(
                "require_positive_return", default=True, grid=[True, False]
            ),
            "max_positions": HyperParam.integer("max_positions", default=8, low=1, high=256),
            "min_trend": HyperParam.floating("min_trend", default=0.0, low=0.0, high=5.0),
            "max_gross": HyperParam.floating("max_gross", default=0.80, low=0.0, high=5.0),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=3, low=1, high=10080),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.12, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
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
        self.trend_window = max(2, int(resolved["trend_window"]))
        self.risk_on_breadth = max(0.0, min(1.0, float(resolved["risk_on_breadth"])))
        self.risk_off_breadth = max(
            0.0, min(self.risk_on_breadth, float(resolved["risk_off_breadth"]))
        )
        self.require_positive_return = bool(resolved["require_positive_return"])
        self.max_positions = max(1, int(resolved["max_positions"]))
        self.min_trend = max(0.0, float(resolved["min_trend"]))
        self.max_gross = max(0.0, float(resolved["max_gross"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.trend_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        # Persisted regime flag for hysteresis between the two breadth bands.
        self._risk_on = False
        self._last_breadth = 0.0
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "risk_on": bool(self._risk_on),
            "last_breadth": float(self._last_breadth),
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        self._risk_on = bool(state.get("risk_on", False))
        parsed_breadth = safe_float(state.get("last_breadth"))
        if parsed_breadth is not None:
            self._last_breadth = max(0.0, min(1.0, float(parsed_breadth)))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    _restore_cross(self._state[symbol], payload)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
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
            if snapshot is not None and self._update(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    def _update(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        return True

    def _name_trend(self, item: _CrossSectionalState) -> tuple[bool, float | None]:
        """Return ``(above_trend, trend_return)`` for one name.

        ``above_trend`` is ``close > SMA(trend_window)`` AND (when
        ``require_positive_return``) a positive simple return over
        ``trend_window``.  ``trend_return`` is the simple return used for ranking.
        """
        closes = list(item.closes)
        sma = simple_moving_average(closes, self.trend_window)
        ret = simple_return(closes, lookback=self.trend_window)
        if sma is None or not closes:
            return False, ret
        close = closes[-1]
        above = close > sma
        if self.require_positive_return:
            above = above and ret is not None and ret > 0.0
        return bool(above), ret

    def _breadth_rows(self) -> tuple[float, list[tuple[float, str, dict[str, Any]]]]:
        """Return ``(breadth, ranked up-trend rows)`` across the basket."""
        eligible = 0
        above = 0
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            is_above, ret = self._name_trend(item)
            if ret is None:
                continue
            eligible += 1
            if is_above:
                above += 1
                if ret >= self.min_trend:
                    rows.append((float(ret), symbol, {"trend_return": float(ret)}))
        breadth = (above / eligible) if eligible > 0 else 0.0
        rows.sort(key=lambda row: row[0], reverse=True)
        return float(breadth), rows[: self.max_positions]

    def _flatten(self, event_time: Any, reason: str) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            price = item.closes[-1] if item.closes else None
            _emit(
                self.events,
                strategy_id="breadth_regime_trend_timer",
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata={"strategy": self.strategy_name, "reason": reason},
            )
            item.mode = "OUT"
            item.entry_price = None
            item.bars_held = 0

    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        breadth, rows = self._breadth_rows()
        self._last_breadth = breadth
        # Regime gate with hysteresis on TOTAL exposure.
        if breadth <= self.risk_off_breadth:
            self._risk_on = False
            self._flatten(event_time, "breadth_risk_off")
            return
        if breadth >= self.risk_on_breadth:
            self._risk_on = True
        # In the neutral band keep the current regime; if risk-off, do nothing
        # but age existing (already-flat) positions, else fall through to hold.
        if not self._risk_on:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="breadth_regime_trend_timer",
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        # Only re-form the basket on rebalance ticks; otherwise just age/hold so
        # turnover stays bounded (held same-side names are not churned).
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="breadth_regime_trend_timer",
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        if not rows:
            self._flatten(event_time, "no_uptrend_names")
            return
        # Long-only timer: rank up-trenders, take the top names. Total gross is
        # scaled by breadth so a broadly strong tape carries more risk.
        targets = _ranked_targets(
            rows,
            threshold=self.min_trend,
            max_longs=self.max_positions,
            max_shorts=0,
            allow_short=False,
        )
        gross = self.target_allocation * min(1.0, self.max_gross) * breadth
        self._emit_breadth_targets(targets, gross, breadth, event_time)

    def _emit_breadth_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        gross: float,
        breadth: float,
        event_time: Any,
    ) -> None:
        selected = len(targets)
        per_name = (gross / float(selected)) if selected else 0.0
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id="breadth_regime_trend_timer",
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": self.strategy_name, "reason": "rebalance_removed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.bars_held += 1
                continue
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (1.0 - self.stop_loss_pct)
            metadata = _target_metadata(
                strategy=self.strategy_name,
                target_allocation=max(0.0, per_name),
                max_order_value=self.max_order_value,
                score=score,
                target_mode=target_mode,
                breadth=float(breadth),
                breadth_scaled_gross=float(gross),
                **meta,
            )
            _emit(
                self.events,
                strategy_id="breadth_regime_trend_timer",
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, breadth * 2.0)),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
        # Age any held name that dropped out of the target basket.
        for symbol, item in self._state.items():
            if item.mode == "OUT" or symbol in targets:
                continue
            price = item.closes[-1] if item.closes else None
            item.bars_held += 1
            stop = (
                price is not None
                and item.entry_price is not None
                and self.stop_loss_pct > 0.0
                and item.mode == "LONG"
                and price <= item.entry_price * (1.0 - self.stop_loss_pct)
            )
            if stop or item.bars_held >= self.max_hold_bars:
                _emit(
                    self.events,
                    strategy_id="breadth_regime_trend_timer",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={
                        "strategy": self.strategy_name,
                        "reason": "stop_loss" if stop else "max_hold",
                    },
                )
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0


__all__ = [
    "BreadthRegimeTrendTimerStrategy",
    "RealizedVolTermStructureStrategy",
]
