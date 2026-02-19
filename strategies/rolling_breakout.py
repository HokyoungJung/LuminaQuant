"""Rolling channel breakout strategy with optional short support."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import average_true_range, safe_float, true_range
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    highs: deque
    lows: deque
    trs: deque
    state: str = "OUT"
    entry_price: float | None = None
    prev_close: float | None = None
    last_time_key: str = ""


class RollingBreakoutStrategy(Strategy):
    """Channel breakout strategy with simple ATR-aware protective stops."""

    def __init__(
        self,
        bars,
        events,
        lookback_bars=48,
        breakout_buffer=0.0,
        atr_window=14,
        atr_stop_multiplier=2.5,
        stop_loss_pct=0.03,
        allow_short=False,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        self.lookback_bars = max(5, int(lookback_bars))
        self.breakout_buffer = max(0.0, float(breakout_buffer))
        self.atr_window = max(1, int(atr_window))
        self.atr_stop_multiplier = max(0.1, float(atr_stop_multiplier))
        self.stop_loss_pct = min(0.5, max(0.001, float(stop_loss_pct)))
        self.allow_short = bool(allow_short)

        self._state = {
            symbol: _SymbolState(
                highs=deque(maxlen=self.lookback_bars),
                lows=deque(maxlen=self.lookback_bars),
                trs=deque(maxlen=self.atr_window),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "trs": list(item.trs),
                    "state": item.state,
                    "entry_price": item.entry_price,
                    "prev_close": item.prev_close,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        raw_symbol_state = state.get("symbol_state")
        if not isinstance(raw_symbol_state, dict):
            return

        for symbol, raw in raw_symbol_state.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            item = self._state[symbol]
            item.highs.clear()
            item.lows.clear()
            item.trs.clear()

            for value in list(raw.get("highs") or [])[-self.lookback_bars :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.highs.append(parsed)
            for value in list(raw.get("lows") or [])[-self.lookback_bars :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.lows.append(parsed)
            for value in list(raw.get("trs") or [])[-self.atr_window :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.trs.append(max(0.0, parsed))

            restored_state = str(raw.get("state", "OUT")).upper()
            item.state = restored_state if restored_state in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.prev_close = safe_float(raw.get("prev_close"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_bar(self, symbol, event):
        event_symbol = getattr(event, "symbol", None)
        if event_symbol == symbol:
            bar_time = getattr(event, "time", None)
            high = safe_float(getattr(event, "high", None))
            low = safe_float(getattr(event, "low", None))
            close = safe_float(getattr(event, "close", None))
            if high is not None and low is not None and close is not None:
                return bar_time, high, low, close

        bar_time = self.bars.get_latest_bar_datetime(symbol)
        high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if high is None or low is None or close is None:
            return None, None, None, None
        return bar_time, high, low, close

    def _emit(self, symbol, event_time, signal_type, close_price, metadata, stop_loss=None):
        self.events.put(
            SignalEvent(
                strategy_id="rolling_breakout",
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
        event_time, high_price, low_price, close_price = self._resolve_bar(symbol, event)
        if high_price is None or low_price is None or close_price is None:
            return

        time_key = "" if event_time is None else str(event_time)
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        has_channel = len(item.highs) >= self.lookback_bars and len(item.lows) >= self.lookback_bars
        channel_high = max(item.highs) if has_channel else None
        channel_low = min(item.lows) if has_channel else None

        prev_close = item.prev_close
        current_tr = true_range(high_price, low_price, prev_close)
        item.trs.append(max(0.0, current_tr))
        atr_value = average_true_range(item.trs, self.atr_window)

        stop_reason = None
        if item.state == "LONG" and item.entry_price is not None:
            if close_price <= item.entry_price * (1.0 - self.stop_loss_pct):
                stop_reason = "stop_loss"
            elif channel_low is not None and close_price < channel_low:
                stop_reason = "breakdown_exit"
            if stop_reason is not None:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    close_price,
                    {
                        "strategy": "RollingBreakoutStrategy",
                        "reason": stop_reason,
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        elif item.state == "SHORT" and item.entry_price is not None:
            if close_price >= item.entry_price * (1.0 + self.stop_loss_pct):
                stop_reason = "stop_loss"
            elif channel_high is not None and close_price > channel_high:
                stop_reason = "breakout_exit"
            if stop_reason is not None:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    close_price,
                    {
                        "strategy": "RollingBreakoutStrategy",
                        "reason": stop_reason,
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        if item.state == "OUT" and channel_high is not None and channel_low is not None:
            upper_trigger = channel_high * (1.0 + self.breakout_buffer)
            lower_trigger = channel_low * (1.0 - self.breakout_buffer)

            if close_price > upper_trigger:
                stop_loss = close_price * (1.0 - self.stop_loss_pct)
                if atr_value is not None:
                    stop_loss = max(stop_loss, close_price - atr_value * self.atr_stop_multiplier)
                self._emit(
                    symbol,
                    event_time,
                    "LONG",
                    close_price,
                    {
                        "strategy": "RollingBreakoutStrategy",
                        "channel_high": channel_high,
                        "channel_low": channel_low,
                        "atr": atr_value,
                    },
                    stop_loss=stop_loss,
                )
                item.state = "LONG"
                item.entry_price = close_price

            elif self.allow_short and close_price < lower_trigger:
                stop_loss = close_price * (1.0 + self.stop_loss_pct)
                if atr_value is not None:
                    stop_loss = min(stop_loss, close_price + atr_value * self.atr_stop_multiplier)
                self._emit(
                    symbol,
                    event_time,
                    "SHORT",
                    close_price,
                    {
                        "strategy": "RollingBreakoutStrategy",
                        "channel_high": channel_high,
                        "channel_low": channel_low,
                        "atr": atr_value,
                    },
                    stop_loss=stop_loss,
                )
                item.state = "SHORT"
                item.entry_price = close_price

        item.highs.append(high_price)
        item.lows.append(low_price)
        item.prev_close = close_price
