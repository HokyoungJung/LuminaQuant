"""Volatility-compression mean-reversion candidate strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.factory_fast import volatility_ratio_latest
from lumina_quant.indicators.oscillators import zscore
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    closes: deque
    mode: str = "OUT"
    entry_price: float | None = None
    last_time_key: str = ""


class VolatilityCompressionReversionStrategy(Strategy):
    """Fade short-term extremes only during volatility compression phases."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "z_window": HyperParam.integer(
                "z_window",
                default=48,
                low=6,
                high=4096,
                optuna={"type": "int", "low": 12, "high": 192},
                grid=[24, 36, 48, 64, 96],
            ),
            "fast_vol_window": HyperParam.integer(
                "fast_vol_window",
                default=12,
                low=5,
                high=4096,
                optuna={"type": "int", "low": 6, "high": 48},
                grid=[8, 12, 20],
            ),
            "slow_vol_window": HyperParam.integer(
                "slow_vol_window",
                default=72,
                low=6,
                high=8192,
                optuna={"type": "int", "low": 24, "high": 320},
                grid=[48, 72, 120],
            ),
            "compression_threshold": HyperParam.floating(
                "compression_threshold",
                default=0.75,
                low=0.1,
                high=10.0,
                optuna={"type": "float", "low": 0.20, "high": 1.20, "step": 0.02},
                grid=[0.55, 0.7, 0.85],
            ),
            "entry_z": HyperParam.floating(
                "entry_z",
                default=1.6,
                low=0.2,
                high=20.0,
                optuna={"type": "float", "low": 0.6, "high": 3.0, "step": 0.05},
                grid=[1.2, 1.6, 2.0, 2.4],
            ),
            "exit_z": HyperParam.floating(
                "exit_z",
                default=0.35,
                low=0.05,
                high=20.0,
                optuna={"type": "float", "low": 0.05, "high": 1.2, "step": 0.05},
                grid=[0.2, 0.35, 0.5, 0.75],
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.025,
                low=0.002,
                high=0.5,
                optuna={"type": "float", "low": 0.005, "high": 0.12, "step": 0.005},
                grid=[0.01, 0.02, 0.03, 0.05],
            ),
            "exit_compression_multiplier": HyperParam.floating(
                "exit_compression_multiplier",
                default=1.20,
                low=1.0,
                high=10.0,
                optuna={"type": "float", "low": 1.0, "high": 2.0, "step": 0.05},
                grid=[1.0, 1.2, 1.4, 1.8],
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
        z_window: int = 48,
        fast_vol_window: int = 12,
        slow_vol_window: int = 72,
        compression_threshold: float = 0.75,
        entry_z: float = 1.6,
        exit_z: float = 0.35,
        stop_loss_pct: float = 0.025,
        exit_compression_multiplier: float = 1.20,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "z_window": z_window,
                "fast_vol_window": fast_vol_window,
                "slow_vol_window": slow_vol_window,
                "compression_threshold": compression_threshold,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_loss_pct": stop_loss_pct,
                "exit_compression_multiplier": exit_compression_multiplier,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )

        self.z_window = int(resolved["z_window"])
        self.fast_vol_window = int(resolved["fast_vol_window"])
        self.slow_vol_window = max(self.fast_vol_window + 1, int(resolved["slow_vol_window"]))
        self.compression_threshold = float(resolved["compression_threshold"])
        self.entry_z = float(resolved["entry_z"])
        self.exit_z = float(resolved["exit_z"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.exit_compression_multiplier = float(resolved["exit_compression_multiplier"])
        self.allow_short = bool(resolved["allow_short"])

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
                or vol_ratio > self.compression_threshold * self.exit_compression_multiplier
            ):
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "long_exit"})
                item.mode = "OUT"
                item.entry_price = None
            return

        if item.mode == "SHORT":
            if (
                close >= (item.entry_price or close) * (1.0 + self.stop_loss_pct)
                or z_value <= self.exit_z
                or vol_ratio > self.compression_threshold * self.exit_compression_multiplier
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
