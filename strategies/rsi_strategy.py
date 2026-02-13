"""Relative Strength Index strategy implementation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.strategy import Strategy

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _RsiTracker:
    """Per-symbol incremental RSI state."""

    prev_close: float | None = None
    avg_gain: float | None = None
    avg_loss: float | None = None
    gain_sum: float = 0.0
    loss_sum: float = 0.0
    samples: int = 0


class RsiStrategy(Strategy):
    """RSI strategy with incremental, allocation-light updates.

    Trading rules:
    - Enter long when RSI < oversold and current state is OUT.
    - Exit long when RSI > overbought and current state is LONG.
    """

    def __init__(
        self,
        bars,
        events,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
    ) -> None:
        self.bars = bars
        self.events = events
        self.rsi_period = int(rsi_period)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        self.symbol_list = list(self.bars.symbol_list)
        self.bought = dict.fromkeys(self.symbol_list, "OUT")
        self._trackers = {symbol: _RsiTracker() for symbol in self.symbol_list}

    def get_state(self) -> dict:
        """Serialize strategy state for persistence."""
        tracker_state = {
            symbol: {
                "prev_close": tracker.prev_close,
                "avg_gain": tracker.avg_gain,
                "avg_loss": tracker.avg_loss,
                "gain_sum": tracker.gain_sum,
                "loss_sum": tracker.loss_sum,
                "samples": tracker.samples,
            }
            for symbol, tracker in self._trackers.items()
        }
        return {
            "bought": dict(self.bought),
            "rsi_state": tracker_state,
        }

    def set_state(self, state: dict) -> None:
        """Restore strategy state from persistence."""
        bought_state = state.get("bought")
        if isinstance(bought_state, dict):
            self.bought.update(bought_state)

        tracker_state = state.get("rsi_state")
        if not isinstance(tracker_state, dict):
            return

        for symbol, raw_tracker in tracker_state.items():
            if symbol not in self._trackers or not isinstance(raw_tracker, dict):
                continue
            tracker = self._trackers[symbol]
            prev_close = raw_tracker.get("prev_close")
            avg_gain = raw_tracker.get("avg_gain")
            avg_loss = raw_tracker.get("avg_loss")
            gain_sum = raw_tracker.get("gain_sum")
            loss_sum = raw_tracker.get("loss_sum")
            samples = raw_tracker.get("samples")
            tracker.prev_close = float(prev_close) if prev_close is not None else None
            tracker.avg_gain = float(avg_gain) if avg_gain is not None else None
            tracker.avg_loss = float(avg_loss) if avg_loss is not None else None
            tracker.gain_sum = float(gain_sum) if gain_sum is not None else 0.0
            tracker.loss_sum = float(loss_sum) if loss_sum is not None else 0.0
            tracker.samples = int(samples) if samples is not None else 0

    def _resolve_close(self, symbol: str, event) -> float | None:
        if getattr(event, "symbol", None) == symbol:
            close = getattr(event, "close", None)
            if close is not None:
                close_price = float(close)
                if math.isfinite(close_price):
                    return close_price
        close_price = self.bars.get_latest_bar_value(symbol, "close")
        if close_price is None:
            return None
        close_price = float(close_price)
        if not math.isfinite(close_price):
            return None
        return close_price

    def _update_rsi(self, symbol: str, close_price: float) -> float | None:
        tracker = self._trackers[symbol]
        prev_close = tracker.prev_close
        tracker.prev_close = close_price
        if prev_close is None:
            return None

        delta = close_price - prev_close
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0

        if tracker.avg_gain is None or tracker.avg_loss is None:
            tracker.gain_sum += gain
            tracker.loss_sum += loss
            tracker.samples += 1
            if tracker.samples < self.rsi_period:
                return None

            tracker.avg_gain = tracker.gain_sum / float(self.rsi_period)
            tracker.avg_loss = tracker.loss_sum / float(self.rsi_period)
        else:
            period = float(self.rsi_period)
            keep_weight = float(self.rsi_period - 1)
            tracker.avg_gain = (tracker.avg_gain * keep_weight + gain) / period
            tracker.avg_loss = (tracker.avg_loss * keep_weight + loss) / period

        avg_gain = tracker.avg_gain
        avg_loss = tracker.avg_loss
        if avg_gain is None or avg_loss is None:
            return None

        if avg_loss <= 0.0:
            return 100.0 if avg_gain > 0.0 else 0.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _emit_signal(self, symbol: str, event_time, signal_type: str, rsi_value: float) -> None:
        signal = SignalEvent(1, symbol, event_time, signal_type, 1.0)
        self.events.put(signal)
        LOGGER.info("%s signal %s | RSI %.2f", signal_type, symbol, rsi_value)

    def calculate_signals(self, event) -> None:
        """Calculate signal updates for incoming market events."""
        if getattr(event, "type", None) != "MARKET":
            return

        symbol = getattr(event, "symbol", None)
        if symbol in self._trackers:
            symbols_to_update = (symbol,)
        else:
            symbols_to_update = tuple(self.symbol_list)

        event_time = getattr(event, "time", None)
        for current_symbol in symbols_to_update:
            close_price = self._resolve_close(current_symbol, event)
            if close_price is None:
                continue

            current_rsi = self._update_rsi(current_symbol, close_price)
            if current_rsi is None:
                continue

            position_state = self.bought[current_symbol]
            if current_rsi < self.oversold and position_state == "OUT":
                self._emit_signal(current_symbol, event_time, "LONG", current_rsi)
                self.bought[current_symbol] = "LONG"
            elif current_rsi > self.overbought and position_state == "LONG":
                self._emit_signal(current_symbol, event_time, "EXIT", current_rsi)
                self.bought[current_symbol] = "OUT"
