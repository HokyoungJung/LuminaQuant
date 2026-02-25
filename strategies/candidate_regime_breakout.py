"""Regime-aware breakout candidate strategy for futures research."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.factory_fast import (
    composite_momentum_latest,
    rolling_range_position_latest,
    rolling_slope_latest,
    volatility_ratio_latest,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    highs: deque
    lows: deque
    closes: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class RegimeBreakoutCandidateStrategy(Strategy):
    """Trade breakouts only when trend and volatility regime align."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_window": HyperParam.integer(
                "lookback_window",
                default=48,
                low=10,
                high=4096,
                optuna={"type": "int", "low": 16, "high": 128},
                grid=[24, 48, 72, 96],
            ),
            "slope_window": HyperParam.integer(
                "slope_window",
                default=21,
                low=5,
                high=4096,
                optuna={"type": "int", "low": 8, "high": 64},
                grid=[13, 21, 34],
            ),
            "volatility_fast_window": HyperParam.integer(
                "volatility_fast_window",
                default=20,
                low=5,
                high=4096,
                optuna={"type": "int", "low": 6, "high": 48},
                grid=[12, 20, 30],
            ),
            "volatility_slow_window": HyperParam.integer(
                "volatility_slow_window",
                default=96,
                low=6,
                high=8192,
                optuna={"type": "int", "low": 24, "high": 240},
                grid=[48, 96, 144],
            ),
            "range_entry_threshold": HyperParam.floating(
                "range_entry_threshold",
                default=0.70,
                low=0.51,
                high=0.99,
                optuna={"type": "float", "low": 0.55, "high": 0.90, "step": 0.01},
                grid=[0.6, 0.7, 0.8],
            ),
            "slope_entry_threshold": HyperParam.floating(
                "slope_entry_threshold",
                default=0.0,
                low=-1.0,
                high=1.0,
                optuna={"type": "float", "low": -0.001, "high": 0.003, "step": 0.0005},
                grid=[0.0, 0.001, 0.002],
            ),
            "momentum_floor": HyperParam.floating(
                "momentum_floor",
                default=0.0,
                low=-1.0,
                high=1.0,
                optuna={"type": "float", "low": -0.02, "high": 0.05, "step": 0.002},
                grid=[0.0, 0.01, 0.02],
            ),
            "max_volatility_ratio": HyperParam.floating(
                "max_volatility_ratio",
                default=1.80,
                low=0.2,
                high=20.0,
                optuna={"type": "float", "low": 0.5, "high": 3.0, "step": 0.05},
                grid=[1.2, 1.8, 2.4],
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.03,
                low=0.002,
                high=0.5,
                optuna={"type": "float", "low": 0.005, "high": 0.10, "step": 0.005},
                grid=[0.01, 0.02, 0.03, 0.05],
            ),
            "allow_short": HyperParam.boolean(
                "allow_short",
                default=True,
                optuna={"type": "categorical", "choices": [True, False]},
                grid=[True, False],
            ),
        }

    def __init__(
        self,
        bars,
        events,
        lookback_window: int = 48,
        slope_window: int = 21,
        volatility_fast_window: int = 20,
        volatility_slow_window: int = 96,
        range_entry_threshold: float = 0.70,
        slope_entry_threshold: float = 0.0,
        momentum_floor: float = 0.0,
        max_volatility_ratio: float = 1.80,
        stop_loss_pct: float = 0.03,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback_window": lookback_window,
                "slope_window": slope_window,
                "volatility_fast_window": volatility_fast_window,
                "volatility_slow_window": volatility_slow_window,
                "range_entry_threshold": range_entry_threshold,
                "slope_entry_threshold": slope_entry_threshold,
                "momentum_floor": momentum_floor,
                "max_volatility_ratio": max_volatility_ratio,
                "stop_loss_pct": stop_loss_pct,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )

        self.lookback_window = int(resolved["lookback_window"])
        self.slope_window = int(resolved["slope_window"])
        self.volatility_fast_window = int(resolved["volatility_fast_window"])
        self.volatility_slow_window = max(
            self.volatility_fast_window + 1,
            int(resolved["volatility_slow_window"]),
        )
        self.range_entry_threshold = float(resolved["range_entry_threshold"])
        self.slope_entry_threshold = float(resolved["slope_entry_threshold"])
        self.momentum_floor = float(resolved["momentum_floor"])
        self.max_volatility_ratio = float(resolved["max_volatility_ratio"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.allow_short = bool(resolved["allow_short"])

        maxlen = (
            max(
                self.lookback_window,
                self.slope_window,
                self.volatility_fast_window,
                self.volatility_slow_window,
            )
            + 2
        )
        self._state = {
            symbol: _SymbolState(
                highs=deque(maxlen=maxlen),
                lows=deque(maxlen=maxlen),
                closes=deque(maxlen=maxlen),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "highs": list(item.highs),
                    "lows": list(item.lows),
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
            item.highs.clear()
            item.lows.clear()
            item.closes.clear()

            for value in list(raw.get("highs") or [])[-item.highs.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.highs.append(parsed)
            for value in list(raw.get("lows") or [])[-item.lows.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.lows.append(parsed)
            for value in list(raw.get("closes") or [])[-item.closes.maxlen :]:
                parsed = safe_float(value)
                if parsed is not None:
                    item.closes.append(parsed)

            restored_mode = str(raw.get("mode", "OUT")).upper()
            item.mode = restored_mode if restored_mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(raw.get("entry_price"))
            item.last_time_key = str(raw.get("last_time_key", ""))

    def _resolve_bar(self, symbol, event):
        if getattr(event, "symbol", None) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            high = safe_float(getattr(event, "high", None))
            low = safe_float(getattr(event, "low", None))
            close = safe_float(getattr(event, "close", None))
            if high is not None and low is not None and close is not None:
                return event_time, high, low, close

        event_time = self.bars.get_latest_bar_datetime(symbol)
        high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if high is None or low is None or close is None:
            return None, None, None, None
        return event_time, high, low, close

    def _emit(self, symbol, event_time, signal_type, *, stop_loss=None, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="candidate_regime_breakout",
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
        event_time, high, low, close = self._resolve_bar(symbol, event)
        if high is None or low is None or close is None:
            return

        bar_key = time_key(event_time)
        if bar_key and bar_key == item.last_time_key:
            return
        item.last_time_key = bar_key

        item.highs.append(high)
        item.lows.append(low)
        item.closes.append(close)

        if len(item.closes) < self.volatility_slow_window:
            return

        closes = list(item.closes)
        highs = list(item.highs)
        lows = list(item.lows)

        slope = rolling_slope_latest(closes, window=self.slope_window)
        range_pos = rolling_range_position_latest(
            highs,
            lows,
            closes,
            window=self.lookback_window,
        )
        vol_ratio = volatility_ratio_latest(
            closes,
            fast_window=self.volatility_fast_window,
            slow_window=self.volatility_slow_window,
        )
        momentum = composite_momentum_latest(closes)

        if slope is None or range_pos is None or vol_ratio is None or momentum is None:
            return

        metadata = {
            "strategy": "RegimeBreakoutCandidateStrategy",
            "slope": float(slope),
            "range_position": float(range_pos),
            "volatility_ratio": float(vol_ratio),
            "momentum": float(momentum),
        }

        if item.mode == "LONG":
            if (
                close <= (item.entry_price or close) * (1.0 - self.stop_loss_pct)
                or slope < 0.0
                or range_pos < 0.50
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "long_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if item.mode == "SHORT":
            if (
                close >= (item.entry_price or close) * (1.0 + self.stop_loss_pct)
                or slope > 0.0
                or range_pos > 0.50
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "short_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if vol_ratio > self.max_volatility_ratio:
            return

        if (
            slope >= self.slope_entry_threshold
            and range_pos >= self.range_entry_threshold
            and momentum >= self.momentum_floor
        ):
            stop_loss = close * (1.0 - self.stop_loss_pct)
            self._emit(symbol, event_time, "LONG", stop_loss=stop_loss, metadata=metadata)
            item.mode = "LONG"
            item.entry_price = close
            return

        if self.allow_short and (
            slope <= -self.slope_entry_threshold
            and range_pos <= (1.0 - self.range_entry_threshold)
            and momentum <= -self.momentum_floor
        ):
            stop_loss = close * (1.0 + self.stop_loss_pct)
            self._emit(symbol, event_time, "SHORT", stop_loss=stop_loss, metadata=metadata)
            item.mode = "SHORT"
            item.entry_price = close
