"""Regime-aware breakout candidate strategy for futures research."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.factory_fast import (
    composite_momentum_latest,
    rolling_range_position_latest,
    rolling_slope_latest,
    volatility_ratio_latest,
)
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    highs: deque
    lows: deque
    closes: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class RegimeBreakoutCandidateStrategy(Strategy):
    """Trade breakouts only when trend and volatility regime align."""

    def __init__(
        self,
        bars,
        events,
        lookback_window: int = 48,
        slope_window: int = 21,
        volatility_fast_window: int = 20,
        volatility_slow_window: int = 96,
        range_entry_threshold: float = 0.70,
        slope_entry_threshold: float = 0.0,
        momentum_floor: float = 0.0,
        max_volatility_ratio: float = 1.80,
        stop_loss_pct: float = 0.03,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)

        self.lookback_window = max(10, int(lookback_window))
        self.slope_window = max(5, int(slope_window))
        self.volatility_fast_window = max(5, int(volatility_fast_window))
        self.volatility_slow_window = max(self.volatility_fast_window + 1, int(volatility_slow_window))
        self.range_entry_threshold = min(0.99, max(0.51, float(range_entry_threshold)))
        self.slope_entry_threshold = float(slope_entry_threshold)
        self.momentum_floor = float(momentum_floor)
        self.max_volatility_ratio = max(0.2, float(max_volatility_ratio))
        self.stop_loss_pct = min(0.50, max(0.002, float(stop_loss_pct)))
        self.allow_short = bool(allow_short)

        maxlen = max(
            self.lookback_window,
            self.slope_window,
            self.volatility_fast_window,
            self.volatility_slow_window,
            55,
        )
        self._state = {
            symbol: _SymbolState(
                highs=deque(maxlen=maxlen),
                lows=deque(maxlen=maxlen),
                closes=deque(maxlen=maxlen),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        symbol_state = state.get("symbol_state")
        if not isinstance(symbol_state, dict):
            return

        for symbol, raw in symbol_state.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            item = self._state[symbol]
            item.highs.clear()
            item.lows.clear()
            item.closes.clear()

            for value in list(raw.get("highs") or [])[-item.highs.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.highs.append(parsed)
            for value in list(raw.get("lows") or [])[-item.lows.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.lows.append(parsed)
            for value in list(raw.get("closes") or [])[-item.closes.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.closes.append(parsed)

            restored_mode = str(raw.get("mode", "OUT")).upper()
            item.mode = restored_mode if restored_mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_bar(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            high = safe_float(getattr(event, "high", None))
            low = safe_float(getattr(event, "low", None))
            close = safe_float(getattr(event, "close", None))
            if high is not None and low is not None and close is not None:
                return event_time, high, low, close

        event_time = self.bars.get_latest_bar_datetime(symbol)
        high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if high is None or low is None or close is None:
            return None, None, None, None
        return event_time, high, low, close

    def _emit(self, symbol, event_time, signal_type, *, stop_loss=None, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="candidate_regime_breakout",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata=metadata,
            )
        )

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return

        symbol = getattr(event, "symbol", None)
        if symbol not in self._state:
            return

        item = self._state[symbol]
        event_time, high, low, close = self._resolve_bar(symbol, event)
        if high is None or low is None or close is None:
            return

        bar_key = time_key(event_time)
        if bar_key and bar_key == item.last_time_key:
            return
        item.last_time_key = bar_key

        item.highs.append(high)
        item.lows.append(low)
        item.closes.append(close)

        if len(item.closes) < self.volatility_slow_window:
            return

        closes = list(item.closes)
        highs = list(item.highs)
        lows = list(item.lows)

        slope = rolling_slope_latest(closes, window=self.slope_window)
        range_pos = rolling_range_position_latest(
            highs,
            lows,
            closes,
            window=self.lookback_window,
        )
        vol_ratio = volatility_ratio_latest(
            closes,
            fast_window=self.volatility_fast_window,
            slow_window=self.volatility_slow_window,
        )
        momentum = composite_momentum_latest(closes)

        if slope is None or range_pos is None or vol_ratio is None or momentum is None:
            return

        metadata = {
            "strategy": "RegimeBreakoutCandidateStrategy",
            "slope": float(slope),
            "range_position": float(range_pos),
            "volatility_ratio": float(vol_ratio),
            "momentum": float(momentum),
        }

        if item.mode == "LONG":
            if (
                close <= (item.entry_price or close) * (1.0 - self.stop_loss_pct)
                or slope < 0.0
                or range_pos < 0.50
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "long_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if item.mode == "SHORT":
            if (
                close >= (item.entry_price or close) * (1.0 + self.stop_loss_pct)
                or slope > 0.0
                or range_pos > 0.50
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "short_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if vol_ratio > self.max_volatility_ratio:
            return

        if (
            slope >= self.slope_entry_threshold
            and range_pos >= self.range_entry_threshold
            and momentum >= self.momentum_floor
        ):
            stop_loss = close * (1.0 - self.stop_loss_pct)
            self._emit(symbol, event_time, "LONG", stop_loss=stop_loss, metadata=metadata)
            item.mode = "LONG"
            item.entry_price = close
            return

        if self.allow_short and (
            slope <= -self.slope_entry_threshold
            and range_pos <= (1.0 - self.range_entry_threshold)
            and momentum <= -self.momentum_floor
        ):
            stop_loss = close * (1.0 + self.stop_loss_pct)
            self._emit(symbol, event_time, "SHORT", stop_loss=stop_loss, metadata=metadata)
            item.mode = "SHORT"
            item.entry_price = close
