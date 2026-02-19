from __future__ import annotations

from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import RollingMeanWindow, safe_float
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    short_window: RollingMeanWindow
    long_window: RollingMeanWindow
    position: str = "OUT"
    last_time_key: str = ""


class MovingAverageCrossStrategy(Strategy):
    def __init__(self, bars, events, short_window=10, long_window=30, allow_short=True):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        self.short_window = max(2, int(short_window))
        self.long_window = max(self.short_window + 1, int(long_window))
        self.allow_short = bool(allow_short)
        self._state = {
            symbol: _SymbolState(
                short_window=RollingMeanWindow(self.short_window),
                long_window=RollingMeanWindow(self.long_window),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "position": item.position,
                    "last_time_key": item.last_time_key,
                    "short_state": item.short_window.to_state(),
                    "long_state": item.long_window.to_state(),
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
            position = str(raw.get("position", "OUT")).upper()
            item.position = position if position in {"OUT", "LONG", "SHORT"} else "OUT"
            item.last_time_key = str(raw.get("last_time_key", ""))
            item.short_window.load_state(list((raw.get("short_state") or {}).get("values") or []))
            item.long_window.load_state(list((raw.get("long_state") or {}).get("values") or []))

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        symbol_obj = getattr(event, "symbol", None)
        if symbol_obj not in self._state:
            return
        symbol = str(symbol_obj)

        item = self._state[symbol]
        event_time = getattr(event, "time", None)
        time_key = "" if event_time is None else str(event_time)
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        close_price = safe_float(getattr(event, "close", None))
        if close_price is None:
            close_price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close_price is None:
            return

        item.short_window.append(close_price)
        item.long_window.append(close_price)
        short_ma = item.short_window.mean()
        long_ma = item.long_window.mean()
        if short_ma is None or long_ma is None:
            return

        if item.position == "OUT" and short_ma > long_ma:
            self.events.put(
                SignalEvent(
                    strategy_id="moving_average_cross",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="LONG",
                    strength=1.0,
                    metadata={"strategy": "MovingAverageCrossStrategy"},
                )
            )
            item.position = "LONG"
        elif item.position == "OUT" and self.allow_short and short_ma < long_ma:
            self.events.put(
                SignalEvent(
                    strategy_id="moving_average_cross",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="SHORT",
                    strength=1.0,
                    metadata={"strategy": "MovingAverageCrossStrategy"},
                )
            )
            item.position = "SHORT"
        elif item.position == "LONG" and short_ma < long_ma:
            self.events.put(
                SignalEvent(
                    strategy_id="moving_average_cross",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="EXIT",
                    strength=1.0,
                    metadata={"strategy": "MovingAverageCrossStrategy"},
                )
            )
            if self.allow_short:
                self.events.put(
                    SignalEvent(
                        strategy_id="moving_average_cross",
                        symbol=symbol,
                        datetime=event_time,
                        signal_type="SHORT",
                        strength=1.0,
                        metadata={"strategy": "MovingAverageCrossStrategy"},
                    )
                )
                item.position = "SHORT"
            else:
                item.position = "OUT"
        elif item.position == "SHORT" and short_ma > long_ma:
            self.events.put(
                SignalEvent(
                    strategy_id="moving_average_cross",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="EXIT",
                    strength=1.0,
                    metadata={"strategy": "MovingAverageCrossStrategy"},
                )
            )
            self.events.put(
                SignalEvent(
                    strategy_id="moving_average_cross",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="LONG",
                    strength=1.0,
                    metadata={"strategy": "MovingAverageCrossStrategy"},
                )
            )
            item.position = "LONG"
