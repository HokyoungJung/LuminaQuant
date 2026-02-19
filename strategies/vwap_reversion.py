"""VWAP deviation mean-reversion strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import safe_float, vwap_deviation, vwap_from_sums
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    prices: deque
    volumes: deque
    price_sum: float = 0.0
    price_squares: float = 0.0
    value_sum: float = 0.0
    volume_sum: float = 0.0
    state: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class VwapReversionStrategy(Strategy):
    """Revert-to-VWAP strategy using rolling deviation thresholds."""

    def __init__(
        self,
        bars,
        events,
        window=64,
        entry_dev=0.02,
        exit_dev=0.005,
        stop_loss_pct=0.03,
        allow_short=True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        self.window = max(8, int(window))
        self.entry_dev = max(0.001, float(entry_dev))
        self.exit_dev = max(0.0, float(exit_dev))
        self.stop_loss_pct = min(0.5, max(0.001, float(stop_loss_pct)))
        self.allow_short = bool(allow_short)

        self._state = {
            symbol: _SymbolState(
                prices=deque(maxlen=self.window),
                volumes=deque(maxlen=self.window),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "prices": list(item.prices),
                    "volumes": list(item.volumes),
                    "price_sum": item.price_sum,
                    "price_squares": item.price_squares,
                    "value_sum": item.value_sum,
                    "volume_sum": item.volume_sum,
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
            item.prices.clear()
            item.volumes.clear()
            item.price_sum = 0.0
            item.price_squares = 0.0
            item.value_sum = 0.0
            item.volume_sum = 0.0

            prices = list(raw.get("prices") or [])[-self.window :]
            volumes = list(raw.get("volumes") or [])[-self.window :]
            paired = min(len(prices), len(volumes))
            for idx in range(paired):
                price = safe_float(prices[idx])
                volume = safe_float(volumes[idx])
                if price is None or volume is None:
                    continue
                vol = max(0.0, volume)
                item.prices.append(price)
                item.volumes.append(vol)
                item.price_sum += price
                item.price_squares += price * price
                item.value_sum += price * vol
                item.volume_sum += vol

            restored_state = str(raw.get("state", "OUT")).upper()
            item.state = restored_state if restored_state in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_bar(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            close_price = safe_float(getattr(event, "close", None))
            volume = safe_float(getattr(event, "volume", None))
            event_time = getattr(event, "time", None)
            if close_price is not None and volume is not None:
                return event_time, close_price, max(0.0, volume)

        close_price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))
        event_time = self.bars.get_latest_bar_datetime(symbol)
        if close_price is None:
            return None, None, None
        return event_time, close_price, max(0.0, volume or 0.0)

    def _append(self, item, close_price, volume):
        if len(item.prices) == self.window and len(item.volumes) == self.window:
            dropped_price = item.prices[0]
            dropped_volume = item.volumes[0]
            item.price_sum -= dropped_price
            item.price_squares -= dropped_price * dropped_price
            item.value_sum -= dropped_price * dropped_volume
            item.volume_sum -= dropped_volume

        item.prices.append(close_price)
        item.volumes.append(volume)
        item.price_sum += close_price
        item.price_squares += close_price * close_price
        item.value_sum += close_price * volume
        item.volume_sum += volume

    @staticmethod
    def _emit(events, symbol, event_time, signal_type, metadata, stop_loss=None) -> None:
        events.put(
            SignalEvent(
                strategy_id="vwap_reversion",
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
        event_time, close_price, volume = self._resolve_bar(symbol, event)
        if close_price is None or volume is None:
            return

        time_key = "" if event_time is None else str(event_time)
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        deviation = None
        vwap = None
        if len(item.prices) >= self.window and item.volume_sum > 0.0:
            vwap = vwap_from_sums(item.value_sum, item.volume_sum)
            deviation = vwap_deviation(close_price, vwap)

        if item.state == "LONG" and item.entry_price is not None:
            stop_hit = close_price <= item.entry_price * (1.0 - self.stop_loss_pct)
            revert_hit = deviation is not None and deviation >= -self.exit_dev
            if stop_hit or revert_hit:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "EXIT",
                    {
                        "strategy": "VwapReversionStrategy",
                        "reason": "stop_loss" if stop_hit else "mean_reversion",
                        "deviation": deviation,
                        "vwap": vwap,
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        elif item.state == "SHORT" and item.entry_price is not None:
            stop_hit = close_price >= item.entry_price * (1.0 + self.stop_loss_pct)
            revert_hit = deviation is not None and deviation <= self.exit_dev
            if stop_hit or revert_hit:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "EXIT",
                    {
                        "strategy": "VwapReversionStrategy",
                        "reason": "stop_loss" if stop_hit else "mean_reversion",
                        "deviation": deviation,
                        "vwap": vwap,
                    },
                )
                item.state = "OUT"
                item.entry_price = None

        if item.state == "OUT" and deviation is not None:
            if deviation <= -self.entry_dev:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "LONG",
                    {
                        "strategy": "VwapReversionStrategy",
                        "deviation": deviation,
                        "vwap": vwap,
                    },
                    stop_loss=close_price * (1.0 - self.stop_loss_pct),
                )
                item.state = "LONG"
                item.entry_price = close_price
            elif self.allow_short and deviation >= self.entry_dev:
                self._emit(
                    self.events,
                    symbol,
                    event_time,
                    "SHORT",
                    {
                        "strategy": "VwapReversionStrategy",
                        "deviation": deviation,
                        "vwap": vwap,
                    },
                    stop_loss=close_price * (1.0 + self.stop_loss_pct),
                )
                item.state = "SHORT"
                item.entry_price = close_price

        self._append(item, close_price, volume)
