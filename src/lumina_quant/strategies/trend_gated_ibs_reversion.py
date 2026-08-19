"""Research-only, long-only internal-bar-strength mean reversion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators import simple_moving_average
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.oscillators import internal_bar_strength
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _State:
    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


@register("strategy", "TrendGatedIbsReversionStrategy", interface="event_driven")
class TrendGatedIbsReversionStrategy(Strategy):
    """Buy a low-IBS pullback in an established uptrend for one full bar."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "sma_window": HyperParam.integer("sma_window", default=20, low=2, high=1000),
            "entry_ibs": HyperParam.floating("entry_ibs", default=0.30, low=0.0, high=1.0),
            "exit_ibs": HyperParam.floating("exit_ibs", default=0.70, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1, low=1, high=1000),
            "exit_on_ibs_rebound": HyperParam.boolean("exit_on_ibs_rebound", default=True),
            "exit_on_trend_fail": HyperParam.boolean("exit_on_trend_fail", default=True),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.03, low=0.0, high=1.0, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.10, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.sma_window = max(2, int(resolved["sma_window"]))
        self.entry_ibs = min(1.0, max(0.0, float(resolved["entry_ibs"])))
        self.exit_ibs = min(1.0, max(0.0, float(resolved["exit_ibs"])))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.exit_on_ibs_rebound = bool(resolved["exit_on_ibs_rebound"])
        self.exit_on_trend_fail = bool(resolved["exit_on_trend_fail"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = self.sma_window + 2
        self._state = {
            symbol: _State(
                opens=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": item.bars_held,
                    "last_time_key": item.last_time_key,
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
            for name in ("opens", "highs", "lows", "closes"):
                target = getattr(item, name)
                target.clear()
                for value in list(payload.get(name) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            item.mode = "LONG" if str(payload.get("mode", "OUT")).upper() == "LONG" else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except TypeError, ValueError:
                item.bars_held = 0
            item.last_time_key = str(payload.get("last_time_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._process(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event)
            return
        if event_type != "MARKET":
            return
        symbol = str(getattr(event, "symbol", ""))
        if symbol in self._state and (snapshot := _market_snapshot(event)) is not None:
            self._process(symbol, snapshot)

    def _process(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        close = safe_float(snapshot.close)
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        if close is None or high is None or low is None:
            return
        item.last_time_key = key
        item.opens.append(float(snapshot.open if snapshot.open is not None else close))
        item.highs.append(high)
        item.lows.append(low)
        item.closes.append(close)
        ibs = internal_bar_strength(high, low, close)
        lagged_sma = simple_moving_average(list(item.closes)[:-1], self.sma_window)

        if item.mode == "LONG":
            item.bars_held += 1
            reason = ""
            if self.exit_on_ibs_rebound and ibs is not None and ibs >= self.exit_ibs:
                reason = "ibs_rebound"
            elif self.exit_on_trend_fail and lagged_sma is not None and close <= lagged_sma:
                reason = "trend_fail"
            elif item.bars_held >= self.max_hold_bars:
                reason = "max_hold"
            if reason:
                _emit(
                    self.events,
                    strategy_id="trend_gated_ibs_reversion",
                    symbol=symbol,
                    event_time=snapshot.time,
                    signal_type="EXIT",
                    price=close,
                    metadata={"strategy": self.__class__.__name__, "reason": reason},
                )
                item.mode, item.entry_price, item.bars_held = "OUT", None, 0
            return

        if (
            ibs is None
            or ibs > self.entry_ibs
            or lagged_sma is None
            or close <= lagged_sma
            or len(item.closes) < 2
            or item.closes[-2] >= item.opens[-2]
        ):
            return
        stop_loss = close * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
        metadata = _target_metadata(
            strategy=self.__class__.__name__,
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            ibs=ibs,
            lagged_sma=lagged_sma,
            previous_bar_bearish=True,
        )
        _emit(
            self.events,
            strategy_id="trend_gated_ibs_reversion",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="LONG",
            strength=self.target_allocation or 1.0,
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode, item.entry_price, item.bars_held = "LONG", close, 0
