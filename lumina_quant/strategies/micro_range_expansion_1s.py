"""Research-only 1-second micro range expansion strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, pstdev

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _State:
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    bars_held: int = 0
    last_time_key: str = ""


class MicroRangeExpansion1sStrategy(Strategy):
    """Capture micro breakouts after compressed 1-second ranges."""

    required_timeframes = ("1s",)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback": HyperParam.integer("lookback", default=30, low=6, high=1024, grid=[20, 30, 45]),
            "range_z_threshold": HyperParam.floating(
                "range_z_threshold", default=1.5, low=0.1, high=6.0, grid=[1.2, 1.5, 2.0]
            ),
            "volume_z_threshold": HyperParam.floating(
                "volume_z_threshold", default=1.0, low=0.0, high=6.0, grid=[0.8, 1.0, 1.5]
            ),
            "signal_strength": HyperParam.floating("signal_strength", default=1.0, low=0.0, high=10.0),
            "min_range_samples": HyperParam.integer("min_range_samples", default=5, low=3, high=256),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=20, low=1, high=2000, grid=[10, 20, 30]),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
        }

    def __init__(
        self,
        bars,
        events,
        lookback: int = 30,
        range_z_threshold: float = 1.5,
        volume_z_threshold: float = 1.0,
        signal_strength: float = 1.0,
        min_range_samples: int = 5,
        max_hold_bars: int = 20,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback": lookback,
                "range_z_threshold": range_z_threshold,
                "volume_z_threshold": volume_z_threshold,
                "signal_strength": signal_strength,
                "min_range_samples": min_range_samples,
                "max_hold_bars": max_hold_bars,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )

        self.lookback = int(resolved["lookback"])
        self.range_z_threshold = float(resolved["range_z_threshold"])
        self.volume_z_threshold = float(resolved["volume_z_threshold"])
        self.signal_strength = float(resolved["signal_strength"])
        self.min_range_samples = int(resolved["min_range_samples"])
        self.max_hold_bars = int(resolved["max_hold_bars"])
        self.allow_short = bool(resolved["allow_short"])

        size = self.lookback + 4
        self._state = {
            symbol: _State(
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
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        raw_state = state.get("symbol_state")
        if not isinstance(raw_state, dict):
            return

        for symbol, payload in raw_state.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("highs", "lows", "closes", "volumes"):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or []):
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except Exception:
                item.bars_held = 0
            item.last_time_key = str(payload.get("last_time_key", ""))

    def _emit(self, symbol, event_time, signal_type, *, range_z, volume_z):
        self.events.put(
            SignalEvent(
                strategy_id="micro_range_expansion_1s",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(self.signal_strength),
                metadata={
                    "strategy": "MicroRangeExpansion1sStrategy",
                    "range_z": float(range_z),
                    "volume_z": float(volume_z),
                },
            )
        )

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return

        symbol = getattr(event, "symbol", None)
        if symbol not in self._state:
            return

        item = self._state[symbol]
        time_key = str(getattr(event, "time", "") or "")
        if time_key and time_key == item.last_time_key:
            return
        if time_key:
            item.last_time_key = time_key

        high = safe_float(getattr(event, "high", None))
        low = safe_float(getattr(event, "low", None))
        close = safe_float(getattr(event, "close", None))
        volume = safe_float(getattr(event, "volume", None))
        if high is None:
            high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        if low is None:
            low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        if close is None:
            close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if volume is None:
            volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))
        if high is None or low is None or close is None or volume is None:
            return

        item.highs.append(high)
        item.lows.append(low)
        item.closes.append(close)
        item.volumes.append(volume)

        if len(item.closes) < self.lookback:
            return

        ranges = [
            (high_px - low_px) / max(close_px, 1e-12)
            for high_px, low_px, close_px in zip(item.highs, item.lows, item.closes, strict=True)
            if close_px > 0.0
        ]
        if len(ranges) < self.min_range_samples:
            return

        latest_range = ranges[-1]
        range_sigma = pstdev(ranges)
        range_mu = mean(ranges)
        range_z = 0.0 if range_sigma <= 1e-12 else (latest_range - range_mu) / range_sigma

        vols = list(item.volumes)
        vol_sigma = pstdev(vols)
        vol_mu = mean(vols)
        volume_z = 0.0 if vol_sigma <= 1e-12 else (vols[-1] - vol_mu) / vol_sigma

        if item.mode != "OUT":
            item.bars_held += 1
            if item.bars_held >= self.max_hold_bars:
                self._emit(symbol, getattr(event, "time", None), "EXIT", range_z=range_z, volume_z=volume_z)
                item.mode = "OUT"
                item.bars_held = 0
            return

        if range_z < self.range_z_threshold or volume_z < self.volume_z_threshold:
            return

        prev_close = item.closes[-2]
        if close >= prev_close:
            self._emit(symbol, getattr(event, "time", None), "LONG", range_z=range_z, volume_z=volume_z)
            item.mode = "LONG"
            item.bars_held = 0
        elif self.allow_short:
            self._emit(symbol, getattr(event, "time", None), "SHORT", range_z=range_z, volume_z=volume_z)
            item.mode = "SHORT"
            item.bars_held = 0
