"""RG_PVTM composite trend strategy.

Primary sleeve:
- Regime-gated price-volume trend momentum
- 1s base feed compatible (decision cadence handled by backtest gating)
- ATR/trailing stop risk controls
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.advanced_alpha import pv_trend_score
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    closes: deque[float]
    highs: deque[float]
    lows: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    trailing_stop: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
    last_score: float = 0.0
    last_gate: bool = False


class CompositeTrendStrategy(Strategy):
    """Regime-gated trend sleeve using multi-horizon price-volume momentum."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "long_threshold": HyperParam.floating("long_threshold", default=0.55, low=0.05, high=5.0),
            "short_threshold": HyperParam.floating("short_threshold", default=0.55, low=0.05, high=5.0),
            "exit_score_cross": HyperParam.floating("exit_score_cross", default=0.05, low=0.0, high=2.0),
            "te_min": HyperParam.floating("te_min", default=0.25, low=0.0, high=1.0),
            "vr_min": HyperParam.floating("vr_min", default=0.85, low=0.1, high=3.0),
            "chop_max": HyperParam.floating("chop_max", default=62.0, low=10.0, high=100.0),
            "vol_window": HyperParam.integer("vol_window", default=120, low=16, high=4000),
            "risk_target_vol": HyperParam.floating("risk_target_vol", default=0.004, low=0.0001, high=0.5),
            "max_signal_strength": HyperParam.floating("max_signal_strength", default=2.0, low=0.1, high=10.0),
            "atr_window": HyperParam.integer("atr_window", default=32, low=4, high=2000),
            "atr_stop_mult": HyperParam.floating("atr_stop_mult", default=2.0, low=0.2, high=20.0),
            "trail_atr_mult": HyperParam.floating("trail_atr_mult", default=2.8, low=0.2, high=20.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=640, low=1, high=200_000),
            "crowding_reduce_threshold": HyperParam.floating(
                "crowding_reduce_threshold",
                default=0.55,
                low=0.0,
                high=1.0,
            ),
            "crowding_block_threshold": HyperParam.floating(
                "crowding_block_threshold",
                default=0.85,
                low=0.0,
                high=1.0,
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True),
            # Backwards-compatible legacy knobs retained for saved param payloads.
            "short_window": HyperParam.integer("short_window", default=20, low=2, high=8192, tunable=False),
            "long_window": HyperParam.integer("long_window", default=72, low=3, high=8192, tunable=False),
            "breakout_window": HyperParam.integer(
                "breakout_window",
                default=48,
                low=5,
                high=8192,
                tunable=False,
            ),
            "breakout_buffer": HyperParam.floating(
                "breakout_buffer",
                default=0.001,
                low=0.0,
                high=1.0,
                tunable=False,
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.03,
                low=0.001,
                high=1.0,
                tunable=False,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        long_threshold: float = 0.55,
        short_threshold: float = 0.55,
        exit_score_cross: float = 0.05,
        te_min: float = 0.25,
        vr_min: float = 0.85,
        chop_max: float = 62.0,
        vol_window: int = 120,
        risk_target_vol: float = 0.004,
        max_signal_strength: float = 2.0,
        atr_window: int = 32,
        atr_stop_mult: float = 2.0,
        trail_atr_mult: float = 2.8,
        max_hold_bars: int = 640,
        crowding_reduce_threshold: float = 0.55,
        crowding_block_threshold: float = 0.85,
        allow_short: bool = True,
        **legacy: object,
    ):
        _ = legacy
        self.bars = bars
        self.events = events
        self.symbol_list = [canonical_symbol(symbol) for symbol in list(self.bars.symbol_list)]
        self.symbol_list = [symbol for symbol in self.symbol_list if symbol]

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "exit_score_cross": exit_score_cross,
                "te_min": te_min,
                "vr_min": vr_min,
                "chop_max": chop_max,
                "vol_window": vol_window,
                "risk_target_vol": risk_target_vol,
                "max_signal_strength": max_signal_strength,
                "atr_window": atr_window,
                "atr_stop_mult": atr_stop_mult,
                "trail_atr_mult": trail_atr_mult,
                "max_hold_bars": max_hold_bars,
                "crowding_reduce_threshold": crowding_reduce_threshold,
                "crowding_block_threshold": crowding_block_threshold,
                "allow_short": allow_short,
            },
            keep_unknown=True,
        )

        self.long_threshold = float(resolved["long_threshold"])
        self.short_threshold = float(resolved["short_threshold"])
        self.exit_score_cross = float(resolved["exit_score_cross"])
        self.te_min = float(resolved["te_min"])
        self.vr_min = float(resolved["vr_min"])
        self.chop_max = float(resolved["chop_max"])
        self.vol_window = int(resolved["vol_window"])
        self.risk_target_vol = float(resolved["risk_target_vol"])
        self.max_signal_strength = float(resolved["max_signal_strength"])
        self.atr_window = int(resolved["atr_window"])
        self.atr_stop_mult = float(resolved["atr_stop_mult"])
        self.trail_atr_mult = float(resolved["trail_atr_mult"])
        self.max_hold_bars = int(resolved["max_hold_bars"])
        self.crowding_reduce_threshold = float(resolved["crowding_reduce_threshold"])
        self.crowding_block_threshold = float(resolved["crowding_block_threshold"])
        self.allow_short = bool(resolved["allow_short"])

        maxlen = max(256, self.vol_window + 64, self.atr_window + 64)
        self._state = {
            symbol: _SymbolState(
                closes=deque(maxlen=maxlen),
                highs=deque(maxlen=maxlen),
                lows=deque(maxlen=maxlen),
                volumes=deque(maxlen=maxlen),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "trailing_stop": item.trailing_stop,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "last_score": float(item.last_score),
                    "last_gate": bool(item.last_gate),
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        symbol_state = state.get("symbol_state")
        if not isinstance(symbol_state, dict):
            return

        for symbol, payload in symbol_state.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("closes", "highs", "lows", "volumes"):
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
            item.trailing_stop = safe_float(payload.get("trailing_stop"))
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except Exception:
                item.bars_held = 0
            item.last_time_key = str(payload.get("last_time_key", ""))
            item.last_score = float(safe_float(payload.get("last_score")) or 0.0)
            item.last_gate = bool(payload.get("last_gate", False))

    def _resolve_bar(self, symbol, event):
        if str(getattr(event, "symbol", "")) == symbol:
            event_time = getattr(event, "time", getattr(event, "datetime", None))
            high = safe_float(getattr(event, "high", None))
            low = safe_float(getattr(event, "low", None))
            close = safe_float(getattr(event, "close", None))
            volume = safe_float(getattr(event, "volume", None))
            if high is not None and low is not None and close is not None and volume is not None:
                return event_time, high, low, close, volume

        event_time = self.bars.get_latest_bar_datetime(symbol)
        high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
        low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))
        if high is None or low is None or close is None:
            return None, None, None, None, None
        return event_time, high, low, close, (volume if volume is not None else 0.0)

    def _extract_crowding_score(self, symbol, event) -> float | None:
        direct = safe_float(getattr(event, "crowding_score", None))
        if direct is not None:
            return direct
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                return safe_float(getter(symbol, "crowding_score"))
            except Exception:
                return None
        try:
            return safe_float(self.bars.get_latest_bar_value(symbol, "crowding_score"))
        except Exception:
            return None

    @staticmethod
    def _rolling_volatility(closes: deque[float], window: int) -> float:
        arr = np.asarray(list(closes), dtype=float)
        if arr.size < max(8, int(window)):
            return 0.0
        returns = np.diff(np.log(np.clip(arr[-int(window) :], 1e-12, np.inf)))
        if returns.size < 2:
            return 0.0
        return float(np.std(returns, ddof=1))

    @staticmethod
    def _atr_abs(highs: deque[float], lows: deque[float], closes: deque[float], window: int) -> float:
        if len(closes) < max(3, int(window)):
            return 0.0
        h = np.asarray(list(highs)[-int(window) :], dtype=float)
        low_arr = np.asarray(list(lows)[-int(window) :], dtype=float)
        c = np.asarray(list(closes)[-int(window) :], dtype=float)
        prev_close = np.r_[c[0], c[:-1]]
        tr = np.maximum.reduce(
            [
                h - low_arr,
                np.abs(h - prev_close),
                np.abs(low_arr - prev_close),
            ]
        )
        tr = np.asarray(tr, dtype=float)
        tr = tr[np.isfinite(tr)]
        if tr.size == 0:
            return 0.0
        return float(np.mean(tr))

    def _signal_strength(self, sigma: float, crowding_score: float | None) -> float:
        sigma_floor = max(1e-6, float(sigma))
        strength = float(self.risk_target_vol) / sigma_floor
        strength = min(self.max_signal_strength, max(0.10, strength))
        if crowding_score is not None and abs(float(crowding_score)) >= self.crowding_reduce_threshold:
            strength *= 0.5
        return float(max(0.05, strength))

    def _emit(
        self,
        symbol: str,
        event_time,
        signal_type: str,
        *,
        strength: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.events.put(
            SignalEvent(
                strategy_id="composite_trend",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )
        )

    def calculate_signals(self, event) -> None:
        if str(getattr(event, "type", "")).upper() != "MARKET":
            return

        symbol = canonical_symbol(str(getattr(event, "symbol", "")))
        if symbol not in self._state:
            return

        item = self._state[symbol]
        event_time, high, low, close, volume = self._resolve_bar(symbol, event)
        if close is None:
            return

        key = time_key(event_time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key

        item.highs.append(float(high))
        item.lows.append(float(low))
        item.closes.append(float(close))
        item.volumes.append(float(volume if volume is not None else 0.0))

        if len(item.closes) < 96:
            return

        factor = pv_trend_score(
            item.closes,
            item.volumes,
            high=item.highs,
            low=item.lows,
            te_min=self.te_min,
            vr_min=self.vr_min,
            chop_max=self.chop_max,
        )
        score = float(factor.get("score", 0.0))
        gate = bool(factor.get("gate", False))
        item.last_score = score
        item.last_gate = gate

        sigma = self._rolling_volatility(item.closes, self.vol_window)
        atr_abs = max(1e-8, self._atr_abs(item.highs, item.lows, item.closes, self.atr_window))
        crowding = self._extract_crowding_score(symbol, event)

        strength = self._signal_strength(sigma, crowding)
        if crowding is not None and abs(float(crowding)) >= self.crowding_block_threshold and item.mode == "OUT":
            return

        metadata = {
            "strategy": "CompositeTrendStrategy",
            "pv_trend_score": score,
            "regime_gate": gate,
            "trend_efficiency": factor.get("trend_efficiency"),
            "volatility_ratio": factor.get("volatility_ratio"),
            "choppiness": factor.get("choppiness"),
            "crowding_score": crowding,
            "bars_held": int(item.bars_held),
        }

        if item.mode == "LONG":
            item.bars_held += 1
            next_stop = float(close) - (self.trail_atr_mult * atr_abs)
            item.trailing_stop = max(float(item.trailing_stop or next_stop), next_stop)
            should_exit = (
                (not gate)
                or (score <= self.exit_score_cross)
                or (float(close) <= float(item.trailing_stop or -math.inf))
                or (item.bars_held >= self.max_hold_bars)
            )
            if should_exit:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    strength=strength,
                    metadata={**metadata, "reason": "long_exit"},
                )
                item.mode = "OUT"
                item.entry_price = None
                item.trailing_stop = None
                item.bars_held = 0
            return

        if item.mode == "SHORT":
            item.bars_held += 1
            next_stop = float(close) + (self.trail_atr_mult * atr_abs)
            item.trailing_stop = min(float(item.trailing_stop or next_stop), next_stop)
            should_exit = (
                (not gate)
                or (score >= -self.exit_score_cross)
                or (float(close) >= float(item.trailing_stop or math.inf))
                or (item.bars_held >= self.max_hold_bars)
            )
            if should_exit:
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    strength=strength,
                    metadata={**metadata, "reason": "short_exit"},
                )
                item.mode = "OUT"
                item.entry_price = None
                item.trailing_stop = None
                item.bars_held = 0
            return

        # Entry logic.
        if gate and score >= self.long_threshold:
            stop = float(close) - (self.atr_stop_mult * atr_abs)
            take = float(close) + (self.atr_stop_mult * atr_abs * 1.8)
            self._emit(
                symbol,
                event_time,
                "LONG",
                strength=strength,
                stop_loss=stop,
                take_profit=take,
                metadata={**metadata, "reason": "long_entry"},
            )
            item.mode = "LONG"
            item.entry_price = float(close)
            item.trailing_stop = stop
            item.bars_held = 0
            return

        if self.allow_short and gate and score <= -self.short_threshold:
            stop = float(close) + (self.atr_stop_mult * atr_abs)
            take = float(close) - (self.atr_stop_mult * atr_abs * 1.8)
            self._emit(
                symbol,
                event_time,
                "SHORT",
                strength=strength,
                stop_loss=stop,
                take_profit=take,
                metadata={**metadata, "reason": "short_entry"},
            )
            item.mode = "SHORT"
            item.entry_price = float(close)
            item.trailing_stop = stop
            item.bars_held = 0
