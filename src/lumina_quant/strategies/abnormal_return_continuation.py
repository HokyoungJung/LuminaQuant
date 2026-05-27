"""Abnormal-return continuation strategy.

This sleeve follows large one-day moves for a short holding window, matching the
literature on crypto post-shock continuation while staying low-memory.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _State:
    closes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


class AbnormalReturnContinuationStrategy(Strategy):
    """Continue large daily moves for a short bounded holding period."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "return_z_window": HyperParam.integer(
                "return_z_window", default=20, low=4, high=1024, tunable=False
            ),
            "entry_z": HyperParam.floating(
                "entry_z", default=1.5, low=0.1, high=10.0, tunable=False
            ),
            "exit_z": HyperParam.floating("exit_z", default=0.25, low=0.0, high=5.0, tunable=False),
            "hold_bars": HyperParam.integer("hold_bars", default=2, low=1, high=128, tunable=False),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.06, low=0.001, high=0.5, tunable=False
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, tunable=False),
        }

    def __init__(
        self,
        bars,
        events,
        return_z_window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.25,
        hold_bars: int = 2,
        stop_loss_pct: float = 0.06,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "return_z_window": return_z_window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "hold_bars": hold_bars,
                "stop_loss_pct": stop_loss_pct,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )
        self.return_z_window = int(resolved["return_z_window"])
        self.entry_z = float(resolved["entry_z"])
        self.exit_z = float(resolved["exit_z"])
        self.hold_bars = int(resolved["hold_bars"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.allow_short = bool(resolved["allow_short"])
        history_len = self.return_z_window + self.hold_bars + 4
        self._state = {
            symbol: _State(closes=deque(maxlen=history_len)) for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": item.bars_held,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            item.closes.clear()
            for value in list(payload.get("closes") or [])[-item.closes.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None and parsed > 0.0:
                    item.closes.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = max(0, int(payload.get("bars_held", 0)))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def _resolve_close(self, symbol: str, event: Any) -> float | None:
        if getattr(event, "symbol", None) == symbol:
            close = safe_float(getattr(event, "close", None))
            if close is not None and close > 0.0:
                return close
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close is None or close <= 0.0:
            return None
        return close

    def _return_z(self, closes: deque[float]) -> float | None:
        values = list(closes)
        win = max(4, self.return_z_window)
        if len(values) <= win:
            return None
        returns = []
        for prev, current in zip(values[-(win + 1) : -1], values[-win:], strict=True):
            if prev <= 0.0 or current <= 0.0:
                continue
            returns.append((current / prev) - 1.0)
        if len(returns) < win:
            return None
        latest = returns[-1]
        hist = returns[:-1]
        avg = mean(hist)
        variance = sum((value - avg) ** 2 for value in hist) / float(max(1, len(hist) - 1))
        std = math.sqrt(max(0.0, variance))
        if std <= 1e-12:
            return 0.0
        value = (latest - avg) / std
        return value if math.isfinite(value) else None

    def _emit(
        self, symbol: str, event_time: Any, signal_type: str, *, zscore: float | None
    ) -> None:
        close_price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        stop_loss = None
        if close_price is not None and close_price > 0.0:
            if signal_type == "LONG":
                stop_loss = close_price * (1.0 - self.stop_loss_pct)
            elif signal_type == "SHORT":
                stop_loss = close_price * (1.0 + self.stop_loss_pct)
        self.events.put(
            SignalEvent(
                strategy_id="abnormal_return_continuation",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata={
                    "strategy": "AbnormalReturnContinuationStrategy",
                    "return_z": zscore,
                    "entry_z": self.entry_z,
                    "exit_z": self.exit_z,
                    "hold_bars": self.hold_bars,
                },
            )
        )

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol not in self._state:
            return
        item = self._state[symbol]
        event_time = self.bars.get_latest_bar_datetime(symbol)
        key = time_key(event_time)
        if not key or item.last_time_key == key:
            return
        item.last_time_key = key
        close = self._resolve_close(symbol, event)
        if close is None:
            return
        item.closes.append(close)
        zscore = self._return_z(item.closes)
        if zscore is None:
            return

        if item.mode == "LONG":
            item.bars_held += 1
            if (
                item.bars_held >= self.hold_bars
                or zscore <= self.exit_z
                or (item.entry_price and close <= item.entry_price * (1.0 - self.stop_loss_pct))
            ):
                self._emit(symbol, event_time, "EXIT", zscore=zscore)
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return
        if item.mode == "SHORT":
            item.bars_held += 1
            if (
                item.bars_held >= self.hold_bars
                or zscore >= -self.exit_z
                or (item.entry_price and close >= item.entry_price * (1.0 + self.stop_loss_pct))
            ):
                self._emit(symbol, event_time, "EXIT", zscore=zscore)
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return

        if zscore >= self.entry_z:
            self._emit(symbol, event_time, "LONG", zscore=zscore)
            item.mode = "LONG"
            item.entry_price = close
            item.bars_held = 0
        elif self.allow_short and zscore <= -self.entry_z:
            self._emit(symbol, event_time, "SHORT", zscore=zscore)
            item.mode = "SHORT"
            item.entry_price = close
            item.bars_held = 0
