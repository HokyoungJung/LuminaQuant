"""Rare-event composite-score strategy.

The strategy consumes only the latest close per event and keeps a bounded
in-memory deque per symbol, avoiding repeated full-history loads.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators import rare_event_scores_latest, safe_float, time_key
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _SymbolState:
    closes: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class RareEventScoreStrategy(Strategy):
    """Trade on rare directional events using a 0~1 composite rarity score."""

    def __init__(
        self,
        bars,
        events,
        history_bars: int = 512,
        lookbacks: tuple[int, ...] = (1, 2, 3, 4, 5),
        return_factor: float = 1.0,
        trend_rolling_window: int = 20,
        local_extremum_window: int = 200,
        entry_score: float = 0.18,
        exit_score: float = 0.55,
        entry_streak: int = 3,
        stop_loss_pct: float = 0.03,
        allow_short: bool = True,
        diff: bool = False,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)

        self.lookbacks = tuple(max(1, int(x)) for x in lookbacks if int(x) > 0) or (1, 2, 3, 4, 5)
        self.return_factor = float(return_factor)
        self.trend_rolling_window = max(5, int(trend_rolling_window))
        self.local_extremum_window = max(10, int(local_extremum_window))
        self.entry_score = min(0.95, max(0.01, float(entry_score)))
        self.exit_score = min(0.99, max(self.entry_score + 0.02, float(exit_score)))
        self.entry_streak = max(2, int(entry_streak))
        self.stop_loss_pct = min(0.40, max(0.002, float(stop_loss_pct)))
        self.allow_short = bool(allow_short)
        self.diff = bool(diff)

        base_history = max(64, int(history_bars))
        min_required = max(
            max(self.lookbacks) + 4,
            self.trend_rolling_window + 12,
            self.local_extremum_window + 2,
        )
        self._history_bars = max(base_history, min_required)

        self._state = {
            symbol: _SymbolState(closes=deque(maxlen=self._history_bars)) for symbol in self.symbol_list
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
                strategy_id="rare_event_score",
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
        scores = rare_event_scores_latest(
            item.closes,
            lookbacks=self.lookbacks,
            return_factor=self.return_factor,
            trend_rolling_window=self.trend_rolling_window,
            local_extremum_window=self.local_extremum_window,
            diff=self.diff,
            max_points=self._history_bars,
        )
        if scores is None:
            return

        rarity = float(scores.composite_score)
        streak = int(scores.rare_streak_value)
        metadata = {
            "strategy": "RareEventScoreStrategy",
            "rarity_score": rarity,
            "rare_return_score": float(scores.rare_return_score),
            "rare_return_lookback": int(scores.rare_return_lookback),
            "rare_streak_score": float(scores.rare_streak_score),
            "rare_streak_value": streak,
            "trend_break_score": float(scores.trend_break_score),
            "local_extremum_score": float(scores.local_extremum_score),
            "local_extremum_side": int(scores.local_extremum_side),
        }

        if item.mode == "LONG":
            stop_hit = close <= (item.entry_price or close) * (1.0 - self.stop_loss_pct)
            normalize_hit = rarity >= self.exit_score or streak >= -1
            if stop_hit or normalize_hit:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    metadata={**metadata, "reason": "long_stop" if stop_hit else "long_normalize"},
                )
                item.mode = "OUT"
                item.entry_price = None
            return

        if item.mode == "SHORT":
            stop_hit = close >= (item.entry_price or close) * (1.0 + self.stop_loss_pct)
            normalize_hit = rarity >= self.exit_score or streak <= 1
            if stop_hit or normalize_hit:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    metadata={**metadata, "reason": "short_stop" if stop_hit else "short_normalize"},
                )
                item.mode = "OUT"
                item.entry_price = None
            return

        if rarity > self.entry_score:
            return

        if streak <= -self.entry_streak:
            stop_loss = close * (1.0 - self.stop_loss_pct)
            self._emit(symbol, event_time, "LONG", stop_loss=stop_loss, metadata=metadata)
            item.mode = "LONG"
            item.entry_price = close
            return

        if self.allow_short and streak >= self.entry_streak:
            stop_loss = close * (1.0 + self.stop_loss_pct)
            self._emit(symbol, event_time, "SHORT", stop_loss=stop_loss, metadata=metadata)
            item.mode = "SHORT"
            item.entry_price = close

