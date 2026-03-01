"""Volatility-compression VWAP reversion strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.advanced_alpha import volcomp_vwap_pressure
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


class VolatilityCompressionReversionStrategy(Strategy):
    """Fade VWAP deviations only when compression regime is active."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "vwap_window": HyperParam.integer("vwap_window", default=60, low=8, high=4096),
            "z_window": HyperParam.integer("z_window", default=120, low=16, high=4096),
            "entry_z": HyperParam.floating("entry_z", default=1.5, low=0.2, high=20.0),
            "exit_z": HyperParam.floating("exit_z", default=0.35, low=0.01, high=20.0),
            "compression_percentile": HyperParam.floating(
                "compression_percentile",
                default=0.30,
                low=0.05,
                high=0.95,
            ),
            "compression_vol_ratio": HyperParam.floating(
                "compression_vol_ratio",
                default=0.85,
                low=0.1,
                high=3.0,
            ),
            "atr_stop_pct": HyperParam.floating("atr_stop_pct", default=0.02, low=0.001, high=0.5),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=64, low=1, high=200_000),
            "allow_short": HyperParam.boolean("allow_short", default=True),
            # Backwards-compatible aliases.
            "compression_threshold": HyperParam.floating(
                "compression_threshold",
                default=0.85,
                low=0.1,
                high=3.0,
                tunable=False,
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.02, low=0.001, high=0.5, tunable=False),
            "exit_compression_multiplier": HyperParam.floating(
                "exit_compression_multiplier",
                default=1.2,
                low=1.0,
                high=10.0,
                tunable=False,
            ),
            "fast_vol_window": HyperParam.integer("fast_vol_window", default=12, low=2, high=4096, tunable=False),
            "slow_vol_window": HyperParam.integer("slow_vol_window", default=60, low=4, high=8192, tunable=False),
        }

    def __init__(
        self,
        bars,
        events,
        vwap_window: int = 60,
        z_window: int = 120,
        entry_z: float = 1.5,
        exit_z: float = 0.35,
        compression_percentile: float = 0.30,
        compression_vol_ratio: float = 0.85,
        fast_vol_window: int = 12,
        slow_vol_window: int = 60,
        atr_stop_pct: float = 0.02,
        max_hold_bars: int = 64,
        allow_short: bool = True,
        **legacy: object,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)

        # Legacy parameter mapping.
        if "compression_threshold" in legacy and "compression_vol_ratio" not in legacy:
            compression_vol_ratio = float(legacy.get("compression_threshold") or compression_vol_ratio)
        if "stop_loss_pct" in legacy and "atr_stop_pct" not in legacy:
            atr_stop_pct = float(legacy.get("stop_loss_pct") or atr_stop_pct)
        if "fast_vol_window" in legacy:
            fast_vol_window = int(legacy.get("fast_vol_window") or fast_vol_window)
        if "slow_vol_window" in legacy:
            slow_vol_window = int(legacy.get("slow_vol_window") or slow_vol_window)

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "vwap_window": vwap_window,
                "z_window": z_window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "compression_percentile": compression_percentile,
                "compression_vol_ratio": compression_vol_ratio,
                "fast_vol_window": fast_vol_window,
                "slow_vol_window": slow_vol_window,
                "atr_stop_pct": atr_stop_pct,
                "max_hold_bars": max_hold_bars,
                "allow_short": allow_short,
            },
            keep_unknown=True,
        )

        self.vwap_window = int(resolved["vwap_window"])
        self.z_window = int(resolved["z_window"])
        self.entry_z = float(resolved["entry_z"])
        self.exit_z = float(resolved["exit_z"])
        self.compression_percentile = float(resolved["compression_percentile"])
        self.compression_vol_ratio = float(resolved["compression_vol_ratio"])
        self.fast_vol_window = int(resolved["fast_vol_window"])
        self.slow_vol_window = int(resolved["slow_vol_window"])
        self.atr_stop_pct = float(resolved["atr_stop_pct"])
        self.max_hold_bars = int(resolved["max_hold_bars"])
        self.allow_short = bool(resolved["allow_short"])

        size = max(256, self.z_window + 64)
        self._state = {
            symbol: _SymbolState(
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
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
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
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

        for symbol, payload in symbol_state.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("highs", "lows", "closes", "volumes"):
                target = getattr(item, attr)
                target.clear()
                raw_values = list(payload.get(attr) or [])
                keep = target.maxlen if target.maxlen is not None else len(raw_values)
                for value in raw_values[-int(keep) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)

            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except Exception:
                item.bars_held = 0
            item.last_time_key = str(payload.get("last_time_key", ""))

    def _resolve_bar(self, symbol, event):
        if str(getattr(event, "symbol", "")) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            high = safe_float(getattr(event, "high", None))
            low = safe_float(getattr(event, "low", None))
            close = safe_float(getattr(event, "close", None))
            volume = safe_float(getattr(event, "volume", None))
            if high is not None and low is not None and close is not None:
                return event_time, high, low, close, (0.0 if volume is None else volume)

        event_time = self.bars.get_latest_bar_datetime(symbol)
        high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))
        if high is None or low is None or close is None:
            return None, None, None, None, None
        return event_time, high, low, close, (0.0 if volume is None else volume)

    def _emit(self, symbol, event_time, signal_type, *, stop_loss=None, take_profit=None, strength=1.0, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="vol_compression_vwap_reversion",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )
        )

    def calculate_signals(self, event):
        if str(getattr(event, "type", "")).upper() != "MARKET":
            return

        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return

        item = self._state[symbol]
        event_time, high, low, close, volume = self._resolve_bar(symbol, event)
        if close is None:
            return

        bar_key = time_key(event_time)
        if bar_key and bar_key == item.last_time_key:
            return
        item.last_time_key = bar_key

        item.highs.append(float(high))
        item.lows.append(float(low))
        item.closes.append(float(close))
        item.volumes.append(float(volume if volume is not None else 0.0))

        min_history = max(
            24,
            int(self.fast_vol_window) + 4,
            int(self.slow_vol_window) + 4,
            max(8, int(self.z_window)),
        )
        if len(item.closes) < min_history:
            return

        adaptive_vwap_window = min(
            int(self.vwap_window),
            max(8, len(item.closes) - max(8, int(self.z_window)) + 1),
        )

        factor = volcomp_vwap_pressure(
            item.highs,
            item.lows,
            item.closes,
            item.volumes,
            vwap_window=adaptive_vwap_window,
            z_window=self.z_window,
            compression_percentile=self.compression_percentile,
            compression_vol_ratio=self.compression_vol_ratio,
            vol_fast_window=self.fast_vol_window,
            vol_slow_window=self.slow_vol_window,
        )
        if not bool(factor.get("available", False)):
            return

        active = bool(factor.get("active", False))
        score = float(factor.get("score", 0.0))
        dev_z = float(factor.get("deviation_z", 0.0))

        metadata = {
            "strategy": "VolCompressionVWAPReversionStrategy",
            "active": active,
            "score": score,
            "deviation_z": dev_z,
            "bandwidth_percentile": factor.get("bandwidth_percentile"),
            "volatility_ratio": factor.get("volatility_ratio"),
            "rare_event_score": factor.get("rare_event_score"),
        }

        stop_loss_pct = max(0.001, float(self.atr_stop_pct))

        if item.mode == "LONG":
            item.bars_held += 1
            should_exit = (
                (not active)
                or (dev_z >= -self.exit_z)
                or (float(close) <= float(item.entry_price or close) * (1.0 - stop_loss_pct))
                or (item.bars_held >= self.max_hold_bars)
            )
            if should_exit:
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "long_exit"})
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return

        if item.mode == "SHORT":
            item.bars_held += 1
            should_exit = (
                (not active)
                or (dev_z <= self.exit_z)
                or (float(close) >= float(item.entry_price or close) * (1.0 + stop_loss_pct))
                or (item.bars_held >= self.max_hold_bars)
            )
            if should_exit:
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "short_exit"})
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return

        if not active:
            return

        # Reversion direction: positive z => price above VWAP => short.
        if dev_z <= -self.entry_z:
            stop = float(close) * (1.0 - stop_loss_pct)
            take = float(close) * (1.0 + stop_loss_pct * 0.9)
            strength = min(2.0, 0.4 + abs(score))
            self._emit(
                symbol,
                event_time,
                "LONG",
                stop_loss=stop,
                take_profit=take,
                strength=strength,
                metadata={**metadata, "reason": "long_entry"},
            )
            item.mode = "LONG"
            item.entry_price = float(close)
            item.bars_held = 0
            return

        if self.allow_short and dev_z >= self.entry_z:
            stop = float(close) * (1.0 + stop_loss_pct)
            take = float(close) * (1.0 - stop_loss_pct * 0.9)
            strength = min(2.0, 0.4 + abs(score))
            self._emit(
                symbol,
                event_time,
                "SHORT",
                stop_loss=stop,
                take_profit=take,
                strength=strength,
                metadata={**metadata, "reason": "short_entry"},
            )
            item.mode = "SHORT"
            item.entry_price = float(close)
            item.bars_held = 0
