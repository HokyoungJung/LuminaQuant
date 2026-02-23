"""Volatility-compression mean-reversion candidate strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.factory_fast import volatility_ratio_latest
from lumina_quant.indicators.oscillators import zscore
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    closes: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class VolatilityCompressionReversionStrategy(Strategy):
    """Fade short-term extremes only during volatility compression phases."""

    def __init__(
        self,
        bars,
        events,
        z_window: int = 48,
        fast_vol_window: int = 12,
        slow_vol_window: int = 72,
        compression_threshold: float = 0.75,
        entry_z: float = 1.6,
        exit_z: float = 0.35,
        stop_loss_pct: float = 0.025,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)

        self.z_window = max(6, int(z_window))
        self.fast_vol_window = max(5, int(fast_vol_window))
        self.slow_vol_window = max(self.fast_vol_window + 1, int(slow_vol_window))
        self.compression_threshold = max(0.1, float(compression_threshold))
        self.entry_z = max(0.2, float(entry_z))
        self.exit_z = min(self.entry_z - 0.05, max(0.05, float(exit_z)))
        self.stop_loss_pct = min(0.50, max(0.002, float(stop_loss_pct)))
        self.allow_short = bool(allow_short)

        maxlen = max(self.z_window, self.slow_vol_window) + 4
        self._state = {
            symbol: _SymbolState(closes=deque(maxlen=maxlen)) for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
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
            item.closes.clear()
            for value in list(raw.get("closes") or [])[-item.closes.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.closes.append(parsed)

            restored_mode = str(raw.get("mode", "OUT")).upper()
            item.mode = restored_mode if restored_mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_close(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            close = safe_float(getattr(event, "close", None))
            if close is not None:
                return event_time, close

        event_time = self.bars.get_latest_bar_datetime(symbol)
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close is None:
            return None, None
        return event_time, close

    def _emit(self, symbol, event_time, signal_type, *, stop_loss=None, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="candidate_vol_compression_reversion",
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
        event_time, close = self._resolve_close(symbol, event)
        if close is None:
            return

        bar_key = time_key(event_time)
        if bar_key and bar_key == item.last_time_key:
            return
        item.last_time_key = bar_key

        item.closes.append(close)
        if len(item.closes) < self.slow_vol_window:
            return

        closes = list(item.closes)
        z_value = zscore(closes, window=self.z_window)
        vol_ratio = volatility_ratio_latest(
            closes,
            fast_window=self.fast_vol_window,
            slow_window=self.slow_vol_window,
        )
        if z_value is None or vol_ratio is None:
            return

        metadata = {
            "strategy": "VolatilityCompressionReversionStrategy",
            "zscore": float(z_value),
            "volatility_ratio": float(vol_ratio),
        }

        if item.mode == "LONG":
            if (
                close <= (item.entry_price or close) * (1.0 - self.stop_loss_pct)
                or z_value >= -self.exit_z
                or vol_ratio > self.compression_threshold * 1.20
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "long_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if item.mode == "SHORT":
            if (
                close >= (item.entry_price or close) * (1.0 + self.stop_loss_pct)
                or z_value <= self.exit_z
                or vol_ratio > self.compression_threshold * 1.20
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "short_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if vol_ratio > self.compression_threshold:
            return

        if z_value <= -self.entry_z:
            stop_loss = close * (1.0 - self.stop_loss_pct)
            self._emit(symbol, event_time, "LONG", stop_loss=stop_loss, metadata=metadata)
            item.mode = "LONG"
            item.entry_price = close
            return

        if self.allow_short and z_value >= self.entry_z:
            stop_loss = close * (1.0 + self.stop_loss_pct)
            self._emit(symbol, event_time, "SHORT", stop_loss=stop_loss, metadata=metadata)
            item.mode = "SHORT"
            item.entry_price = close
