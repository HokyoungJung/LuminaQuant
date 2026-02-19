"""Two-asset lag-convergence strategy using relative momentum spread."""

from __future__ import annotations

from collections import deque

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import momentum_return, momentum_spread, safe_float
from lumina_quant.strategy import Strategy


class LagConvergenceStrategy(Strategy):
    """Pair strategy that trades convergence of lagged momentum spread."""

    def __init__(
        self,
        bars,
        events,
        symbol_x=None,
        symbol_y=None,
        lag_bars=3,
        entry_threshold=0.015,
        exit_threshold=0.004,
        stop_threshold=0.05,
        max_hold_bars=96,
        stop_loss_pct=0.03,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        if len(self.symbol_list) < 2:
            raise ValueError("LagConvergenceStrategy requires at least two symbols.")

        self.symbol_x = str(symbol_x) if symbol_x else str(self.symbol_list[0])
        self.symbol_y = str(symbol_y) if symbol_y else str(self.symbol_list[1])
        if self.symbol_x == self.symbol_y:
            raise ValueError("symbol_x and symbol_y must be different.")

        self.lag_bars = max(1, int(lag_bars))
        self.entry_threshold = max(0.001, float(entry_threshold))
        self.exit_threshold = max(0.0, float(exit_threshold))
        self.stop_threshold = max(self.entry_threshold + 1e-9, float(stop_threshold))
        self.max_hold_bars = max(1, int(max_hold_bars))
        self.stop_loss_pct = min(0.5, max(0.001, float(stop_loss_pct)))

        history_len = max(16, self.lag_bars + 8)
        self._x_history = deque(maxlen=history_len)
        self._y_history = deque(maxlen=history_len)

        self._mode = "OUT"
        self._bars_in_position = 0
        self._last_pair_time_key = ""
        self._last_spread = None

    def get_state(self):
        return {
            "x_history": list(self._x_history),
            "y_history": list(self._y_history),
            "mode": self._mode,
            "bars_in_position": int(self._bars_in_position),
            "last_pair_time_key": str(self._last_pair_time_key),
            "last_spread": self._last_spread,
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return

        self._x_history.clear()
        self._y_history.clear()

        x_maxlen = int(self._x_history.maxlen) if self._x_history.maxlen is not None else 0
        y_maxlen = int(self._y_history.maxlen) if self._y_history.maxlen is not None else 0

        for value in list(state.get("x_history") or [])[-x_maxlen:]:
            parsed = safe_float(value)
            if parsed is not None and parsed > 0.0:
                self._x_history.append(parsed)

        for value in list(state.get("y_history") or [])[-y_maxlen:]:
            parsed = safe_float(value)
            if parsed is not None and parsed > 0.0:
                self._y_history.append(parsed)

        mode = str(state.get("mode", "OUT")).upper()
        self._mode = mode if mode in {"OUT", "LONG_X_SHORT_Y", "SHORT_X_LONG_Y"} else "OUT"
        try:
            self._bars_in_position = max(0, int(state.get("bars_in_position", 0)))
        except Exception:
            self._bars_in_position = 0
        self._last_pair_time_key = str(state.get("last_pair_time_key", ""))
        self._last_spread = safe_float(state.get("last_spread"))

    def _aligned_pair_timestamp(self):
        tx = self.bars.get_latest_bar_datetime(self.symbol_x)
        ty = self.bars.get_latest_bar_datetime(self.symbol_y)
        if tx is None or ty is None or tx != ty:
            return None
        return tx

    def _resolve_pair_closes(self):
        close_x = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "close"))
        close_y = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "close"))
        if close_x is None or close_y is None or close_x <= 0.0 or close_y <= 0.0:
            return None, None
        return close_x, close_y

    def _emit(self, symbol, event_time, signal_type, metadata, stop_loss=None):
        self.events.put(
            SignalEvent(
                strategy_id="lag_convergence",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata=metadata,
            )
        )

    def _emit_entry(self, event_time, spread, close_x, close_y, mode):
        metadata = {
            "strategy": "LagConvergenceStrategy",
            "mode": mode,
            "spread": float(spread),
            "lag_bars": int(self.lag_bars),
            "entry_threshold": float(self.entry_threshold),
        }

        if mode == "LONG_X_SHORT_Y":
            self._emit(
                self.symbol_x,
                event_time,
                "LONG",
                metadata,
                stop_loss=close_x * (1.0 - self.stop_loss_pct),
            )
            self._emit(
                self.symbol_y,
                event_time,
                "SHORT",
                metadata,
                stop_loss=close_y * (1.0 + self.stop_loss_pct),
            )
        else:
            self._emit(
                self.symbol_x,
                event_time,
                "SHORT",
                metadata,
                stop_loss=close_x * (1.0 + self.stop_loss_pct),
            )
            self._emit(
                self.symbol_y,
                event_time,
                "LONG",
                metadata,
                stop_loss=close_y * (1.0 - self.stop_loss_pct),
            )

    def _emit_exit(self, event_time, spread, reason):
        metadata = {
            "strategy": "LagConvergenceStrategy",
            "mode": self._mode,
            "spread": float(spread),
            "reason": reason,
        }
        self._emit(self.symbol_x, event_time, "EXIT", metadata)
        self._emit(self.symbol_y, event_time, "EXIT", metadata)

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        if getattr(event, "symbol", None) not in {self.symbol_x, self.symbol_y}:
            return

        pair_time = self._aligned_pair_timestamp()
        if pair_time is None:
            return
        time_key = str(pair_time)
        if time_key == self._last_pair_time_key:
            return
        self._last_pair_time_key = time_key

        close_x, close_y = self._resolve_pair_closes()
        if close_x is None or close_y is None:
            return

        self._x_history.append(close_x)
        self._y_history.append(close_y)
        if len(self._x_history) <= self.lag_bars or len(self._y_history) <= self.lag_bars:
            return

        base_x = self._x_history[-1 - self.lag_bars]
        base_y = self._y_history[-1 - self.lag_bars]
        if base_x <= 0.0 or base_y <= 0.0:
            return

        momentum_x = momentum_return(close_x, base_x)
        momentum_y = momentum_return(close_y, base_y)
        if momentum_x is None or momentum_y is None:
            return
        spread = momentum_spread(momentum_x, momentum_y)
        self._last_spread = float(spread)

        if self._mode == "OUT":
            if spread <= -self.entry_threshold:
                self._emit_entry(pair_time, spread, close_x, close_y, "LONG_X_SHORT_Y")
                self._mode = "LONG_X_SHORT_Y"
                self._bars_in_position = 0
            elif spread >= self.entry_threshold:
                self._emit_entry(pair_time, spread, close_x, close_y, "SHORT_X_LONG_Y")
                self._mode = "SHORT_X_LONG_Y"
                self._bars_in_position = 0
            return

        self._bars_in_position += 1
        reason = None
        if abs(spread) <= self.exit_threshold:
            reason = "converged"
        elif abs(spread) >= self.stop_threshold:
            reason = "spread_stop"
        elif self._bars_in_position >= self.max_hold_bars:
            reason = "max_hold"

        if reason is None:
            return

        self._emit_exit(pair_time, spread, reason)
        self._mode = "OUT"
        self._bars_in_position = 0
