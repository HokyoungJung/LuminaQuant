from __future__ import annotations

from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import IncrementalRsi, safe_float
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    rsi: IncrementalRsi
    position: str = "OUT"
    last_time_key: str = ""


class RsiStrategy(Strategy):
    def __init__(
        self,
        bars,
        events,
        rsi_period=14,
        oversold=30,
        overbought=70,
        allow_short=True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        self.rsi_period = max(2, int(rsi_period))
        self.oversold = min(50.0, max(1.0, float(oversold)))
        self.overbought = max(self.oversold + 1.0, min(99.0, float(overbought)))
        self.allow_short = bool(allow_short)
        self._state = {
            symbol: _SymbolState(rsi=IncrementalRsi(self.rsi_period)) for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "position": item.position,
                    "last_time_key": item.last_time_key,
                    "rsi_state": item.rsi.to_state(),
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
            item.rsi.load_state(dict(raw.get("rsi_state") or {}))

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

        rsi_value = item.rsi.update(close_price)
        if rsi_value is None:
            return

        if item.position == "OUT" and rsi_value <= self.oversold:
            self.events.put(
                SignalEvent(
                    strategy_id="rsi",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="LONG",
                    strength=1.0,
                    metadata={"strategy": "RsiStrategy", "rsi": float(rsi_value)},
                )
            )
            item.position = "LONG"
        elif item.position == "OUT" and self.allow_short and rsi_value >= self.overbought:
            self.events.put(
                SignalEvent(
                    strategy_id="rsi",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="SHORT",
                    strength=1.0,
                    metadata={"strategy": "RsiStrategy", "rsi": float(rsi_value)},
                )
            )
            item.position = "SHORT"
        elif item.position == "LONG" and rsi_value >= self.overbought:
            self.events.put(
                SignalEvent(
                    strategy_id="rsi",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="EXIT",
                    strength=1.0,
                    metadata={"strategy": "RsiStrategy", "rsi": float(rsi_value)},
                )
            )
            if self.allow_short:
                self.events.put(
                    SignalEvent(
                        strategy_id="rsi",
                        symbol=symbol,
                        datetime=event_time,
                        signal_type="SHORT",
                        strength=1.0,
                        metadata={"strategy": "RsiStrategy", "rsi": float(rsi_value)},
                    )
                )
                item.position = "SHORT"
            else:
                item.position = "OUT"
        elif item.position == "SHORT" and rsi_value <= self.oversold:
            self.events.put(
                SignalEvent(
                    strategy_id="rsi",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="EXIT",
                    strength=1.0,
                    metadata={"strategy": "RsiStrategy", "rsi": float(rsi_value)},
                )
            )
            self.events.put(
                SignalEvent(
                    strategy_id="rsi",
                    symbol=symbol,
                    datetime=event_time,
                    signal_type="LONG",
                    strength=1.0,
                    metadata={"strategy": "RsiStrategy", "rsi": float(rsi_value)},
                )
            )
            item.position = "LONG"
