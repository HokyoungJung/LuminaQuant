import logging
import math
from collections import deque

from lumina_quant.events import SignalEvent
from lumina_quant.strategy import Strategy

LOGGER = logging.getLogger(__name__)


class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a short/long simple weighted moving average.
    """

    def __init__(self, bars, events, short_window=10, long_window=30):
        self.bars = bars
        self.events = events
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.symbol_list = list(self.bars.symbol_list)
        self.bought = dict.fromkeys(self.symbol_list, "OUT")
        self._short_buffers = {s: deque(maxlen=self.short_window) for s in self.symbol_list}
        self._long_buffers = {s: deque(maxlen=self.long_window) for s in self.symbol_list}
        self._short_sums = dict.fromkeys(self.symbol_list, 0.0)
        self._long_sums = dict.fromkeys(self.symbol_list, 0.0)
        self._last_timestamps = dict.fromkeys(self.symbol_list)
        self._warmup_from_history()

    def _warmup_from_history(self):
        for symbol in self.symbol_list:
            closes = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)
            for close in closes:
                price = float(close)
                if math.isfinite(price):
                    self._append_close(symbol, price)

    def _append_close(self, symbol, close_price):
        short_buf = self._short_buffers[symbol]
        long_buf = self._long_buffers[symbol]

        if len(short_buf) == self.short_window:
            self._short_sums[symbol] -= float(short_buf[0])
        short_buf.append(close_price)
        self._short_sums[symbol] += close_price

        if len(long_buf) == self.long_window:
            self._long_sums[symbol] -= float(long_buf[0])
        long_buf.append(close_price)
        self._long_sums[symbol] += close_price

    def _current_ma(self, symbol):
        short_buf = self._short_buffers[symbol]
        long_buf = self._long_buffers[symbol]
        if len(short_buf) < self.short_window or len(long_buf) < self.long_window:
            return None, None
        short_ma = self._short_sums[symbol] / float(self.short_window)
        long_ma = self._long_sums[symbol] / float(self.long_window)
        return short_ma, long_ma

    def get_state(self):
        ma_state = {}
        for symbol in self.symbol_list:
            ma_state[symbol] = {
                "short_values": list(self._short_buffers[symbol]),
                "long_values": list(self._long_buffers[symbol]),
                "short_sum": self._short_sums[symbol],
                "long_sum": self._long_sums[symbol],
                "last_timestamp": self._last_timestamps[symbol],
            }
        return {
            "bought": dict(self.bought),
            "ma_state": ma_state,
        }

    def set_state(self, state):
        if "bought" in state:
            self.bought = state["bought"]

        raw_ma_state = state.get("ma_state")
        if not isinstance(raw_ma_state, dict):
            return

        for symbol, symbol_state in raw_ma_state.items():
            if symbol not in self._short_buffers or not isinstance(symbol_state, dict):
                continue

            short_vals = symbol_state.get("short_values") or []
            long_vals = symbol_state.get("long_values") or []

            clean_short = [float(v) for v in short_vals if isinstance(v, (int, float))]
            clean_long = [float(v) for v in long_vals if isinstance(v, (int, float))]

            self._short_buffers[symbol] = deque(
                clean_short[-self.short_window :], maxlen=self.short_window
            )
            self._long_buffers[symbol] = deque(
                clean_long[-self.long_window :], maxlen=self.long_window
            )
            self._short_sums[symbol] = float(sum(self._short_buffers[symbol]))
            self._long_sums[symbol] = float(sum(self._long_buffers[symbol]))
            self._last_timestamps[symbol] = symbol_state.get("last_timestamp")

    def _resolve_price(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            close = getattr(event, "close", None)
            if close is not None:
                close_price = float(close)
                if math.isfinite(close_price):
                    return close_price
        close = self.bars.get_latest_bar_value(symbol, "close")
        close_price = float(close)
        if not math.isfinite(close_price):
            return None
        return close_price

    def calculate_signals(self, event):
        if event.type != "MARKET":
            return

        symbol = getattr(event, "symbol", None)
        symbols_to_update = [symbol] if symbol in self.symbol_list else self.symbol_list
        event_time = getattr(event, "time", None)

        for current_symbol in symbols_to_update:
            if event_time is not None and self._last_timestamps.get(current_symbol) == event_time:
                continue

            close_price = self._resolve_price(current_symbol, event)
            if close_price is None:
                continue

            self._append_close(current_symbol, close_price)
            if event_time is not None:
                self._last_timestamps[current_symbol] = event_time

            curr_short, curr_long = self._current_ma(current_symbol)
            if curr_short is None or curr_long is None:
                continue

            if curr_short > curr_long and self.bought[current_symbol] == "OUT":
                LOGGER.info(
                    "LONG signal %s | short %.5f > long %.5f",
                    current_symbol,
                    curr_short,
                    curr_long,
                )
                signal = SignalEvent("ma_cross", str(current_symbol), event.time, "LONG", 1.0)
                self.events.put(signal)
                self.bought[current_symbol] = "LONG"
            elif curr_short < curr_long and self.bought[current_symbol] == "LONG":
                LOGGER.info(
                    "EXIT signal %s | short %.5f < long %.5f",
                    current_symbol,
                    curr_short,
                    curr_long,
                )
                signal = SignalEvent("ma_cross", str(current_symbol), event.time, "EXIT", 1.0)
                self.events.put(signal)
                self.bought[current_symbol] = "OUT"
