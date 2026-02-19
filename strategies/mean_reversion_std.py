"""Rolling mean-reversion strategy using standard deviation bands."""

from __future__ import annotations

from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import RollingZScoreWindow, safe_float
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    zscore_window: RollingZScoreWindow
    state: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class MeanReversionStdStrategy(Strategy):
    """Single-asset z-score mean reversion with optional shorts."""

    def __init__(
        self,
        bars,
        events,
        window=64,
        entry_z=2.0,
        exit_z=0.5,
        stop_loss_pct=0.03,
        allow_short=True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        self.window = max(8, int(window))
        self.entry_z = max(0.2, float(entry_z))
        self.exit_z = max(0.0, float(exit_z))
        self.stop_loss_pct = min(0.5, max(0.001, float(stop_loss_pct)))
        self.allow_short = bool(allow_short)

        self._state = {
            symbol: _SymbolState(zscore_window=RollingZScoreWindow(self.window))
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "prices": list(item.zscore_window.values),
                    "sum_price": item.zscore_window.sum_value,
                    "sum_squares": item.zscore_window.sum_squares,
                    "state": item.state,
                    "entry_price": item.entry_price,
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
            prices = []
            for value in list(raw.get("prices") or [])[-self.window :]:
                parsed = safe_float(value)
                if parsed is not None:
                    prices.append(parsed)
            item.zscore_window = RollingZScoreWindow(self.window)
            item.zscore_window.load_state(prices)

            restored_state = str(raw.get("state", "OUT")).upper()
            item.state = restored_state if restored_state in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_close(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            close_price = safe_float(getattr(event, "close", None))
            if close_price is not None:
                return close_price
        return safe_float(self.bars.get_latest_bar_value(symbol, "close"))

    def _append_price(self, item, close_price):
        item.zscore_window.append(close_price)

    @staticmethod
    def _emit(events, symbol, event_time, signal_type, metadata, stop_loss=None) -> None:
        events.put(
            SignalEvent(
                strategy_id="mean_reversion_std",
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
        close_price = self._resolve_close(symbol, event)
        if close_price is None:
            return

        event_time = getattr(event, "time", None)
        time_key = "" if event_time is None else str(event_time)
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        zscore = None
        if len(item.zscore_window.values) >= self.window:
            zscore = item.zscore_window.zscore(close_price)

        if item.state == "LONG" and item.entry_price is not None:
            stop_hit = close_price <= item.entry_price * (1.0 - self.stop_loss_pct)
            revert_hit = zscore is not None and zscore >= -self.exit_z
            if stop_hit or revert_hit:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "EXIT",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "reason": "stop_loss" if stop_hit else "mean_reversion",
                        "zscore": zscore,
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        elif item.state == "SHORT" and item.entry_price is not None:
            stop_hit = close_price >= item.entry_price * (1.0 + self.stop_loss_pct)
            revert_hit = zscore is not None and zscore <= self.exit_z
            if stop_hit or revert_hit:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "EXIT",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "reason": "stop_loss" if stop_hit else "mean_reversion",
                        "zscore": zscore,
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        if item.state == "OUT" and zscore is not None:
            if zscore <= -self.entry_z:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "LONG",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "zscore": zscore,
                    },
                    stop_loss=close_price * (1.0 - self.stop_loss_pct),
                )
                item.state = "LONG"
                item.entry_price = close_price
            elif self.allow_short and zscore >= self.entry_z:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "SHORT",
                    {
                        "strategy": "MeanReversionStdStrategy",
                        "zscore": zscore,
                    },
                    stop_loss=close_price * (1.0 + self.stop_loss_pct),
                )
                item.state = "SHORT"
                item.entry_price = close_price

        self._append_price(item, close_price)
