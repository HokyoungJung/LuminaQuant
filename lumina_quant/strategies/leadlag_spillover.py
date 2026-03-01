"""Lead-lag spillover strategy using cross-asset predictor scores."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.advanced_alpha import cross_leadlag_spillover
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_LEADERS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")
_METALS = {"XAU/USDT", "XAG/USDT"}


@dataclass(slots=True)
class _SymbolState:
    closes: deque[float]
    highs: deque[float]
    lows: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


class LeadLagSpilloverStrategy(Strategy):
    """Trade laggards using ridge-style leader lag spillover predictions."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "window": HyperParam.integer("window", default=180, low=32, high=4096),
            "max_lag": HyperParam.integer("max_lag", default=3, low=1, high=8),
            "ridge_alpha": HyperParam.floating("ridge_alpha", default=1.0, low=1e-6, high=100.0),
            "entry_score": HyperParam.floating("entry_score", default=0.35, low=0.01, high=5.0),
            "exit_score": HyperParam.floating("exit_score", default=0.08, low=0.0, high=2.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=32, low=1, high=100_000),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.02, low=0.001, high=0.5),
            "max_realized_vol": HyperParam.floating("max_realized_vol", default=0.08, low=0.001, high=2.0),
            "min_range_pct": HyperParam.floating("min_range_pct", default=0.0001, low=0.0, high=1.0),
            "allow_short": HyperParam.boolean("allow_short", default=True),
            # Backwards-compatible aliases.
            "entry_spillover": HyperParam.floating(
                "entry_spillover",
                default=0.35,
                low=0.01,
                high=5.0,
                tunable=False,
            ),
            "exit_spillover": HyperParam.floating(
                "exit_spillover",
                default=0.08,
                low=0.0,
                high=2.0,
                tunable=False,
            ),
            "leader_momentum_floor": HyperParam.floating(
                "leader_momentum_floor",
                default=0.0,
                low=0.0,
                high=1.0,
                tunable=False,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        window: int = 180,
        max_lag: int = 3,
        ridge_alpha: float = 1.0,
        entry_score: float = 0.35,
        exit_score: float = 0.08,
        max_hold_bars: int = 32,
        stop_loss_pct: float = 0.02,
        max_realized_vol: float = 0.08,
        min_range_pct: float = 0.0001,
        allow_short: bool = True,
        **legacy: object,
    ):
        self.bars = bars
        self.events = events
        canonical_symbols = [canonical_symbol(symbol) for symbol in list(self.bars.symbol_list)]
        self.symbol_list = [symbol for symbol in canonical_symbols if symbol]

        if "entry_spillover" in legacy:
            entry_score = float(legacy.get("entry_spillover") or entry_score)
        if "exit_spillover" in legacy:
            exit_score = float(legacy.get("exit_spillover") or exit_score)

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "window": window,
                "max_lag": max_lag,
                "ridge_alpha": ridge_alpha,
                "entry_score": entry_score,
                "exit_score": exit_score,
                "max_hold_bars": max_hold_bars,
                "stop_loss_pct": stop_loss_pct,
                "max_realized_vol": max_realized_vol,
                "min_range_pct": min_range_pct,
                "allow_short": allow_short,
            },
            keep_unknown=True,
        )

        self.window = int(resolved["window"])
        self.max_lag = int(resolved["max_lag"])
        self.ridge_alpha = float(resolved["ridge_alpha"])
        self.entry_score = float(resolved["entry_score"])
        self.exit_score = float(resolved["exit_score"])
        self.max_hold_bars = int(resolved["max_hold_bars"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.max_realized_vol = float(resolved["max_realized_vol"])
        self.min_range_pct = float(resolved["min_range_pct"])
        self.allow_short = bool(resolved["allow_short"])

        size = max(256, self.window + 32)
        self._state = {
            symbol: _SymbolState(
                closes=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
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
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
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
            for attr in ("closes", "highs", "lows"):
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
        event_symbol = canonical_symbol(str(getattr(event, "symbol", "")))
        if event_symbol == symbol:
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

    @staticmethod
    def _realized_vol(closes: deque[float], window: int = 48) -> float:
        arr = np.asarray(list(closes), dtype=float)
        if arr.size < max(8, int(window)):
            return 0.0
        rets = np.diff(np.log(np.clip(arr[-int(window) :], 1e-12, np.inf)))
        if rets.size < 2:
            return 0.0
        return float(np.std(rets, ddof=1))

    @staticmethod
    def _liquidity_proxy(high: float, low: float, close: float) -> float:
        if close <= 0.0:
            return 0.0
        return max(0.0, float(high - low) / float(close))

    def _emit(self, symbol, event_time, signal_type, *, stop_loss=None, strength=1.0, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="leadlag_spillover",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                stop_loss=stop_loss,
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
        event_time, high, low, close = self._resolve_bar(symbol, event)
        if close is None:
            return

        key = time_key(event_time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key

        item.highs.append(float(high))
        item.lows.append(float(low))
        item.closes.append(float(close))

        if len(item.closes) < max(64, self.window // 2):
            return

        price_map = {
            sym: list(state.closes)
            for sym, state in self._state.items()
            if sym not in _METALS and len(state.closes) >= max(32, self.window // 3)
        }
        if symbol in _METALS:
            return
        if len(price_map) < 3:
            return

        spill = cross_leadlag_spillover(
            price_map,
            leaders=_LEADERS,
            max_lag=self.max_lag,
            ridge_alpha=self.ridge_alpha,
            window=self.window,
        )
        preds = dict(spill.get("predictions") or {})
        sym_pred = preds.get(symbol)
        if not isinstance(sym_pred, dict):
            return

        score = float(sym_pred.get("score", 0.0))
        predicted_return = float(sym_pred.get("predicted_return", 0.0))

        realized_vol = self._realized_vol(item.closes)
        range_pct = self._liquidity_proxy(float(high), float(low), float(close))
        liquidity_ok = range_pct >= self.min_range_pct
        vol_ok = realized_vol <= self.max_realized_vol

        metadata = {
            "strategy": "LeadLagSpilloverStrategy",
            "score": score,
            "predicted_return": predicted_return,
            "method": sym_pred.get("method"),
            "leaders": sym_pred.get("leaders_used"),
            "liquidity_proxy": range_pct,
            "realized_vol": realized_vol,
            "filters_passed": bool(liquidity_ok and vol_ok),
        }

        if item.mode == "LONG":
            item.bars_held += 1
            should_exit = (
                (score <= self.exit_score)
                or (item.bars_held >= self.max_hold_bars)
                or (float(close) <= float(item.entry_price or close) * (1.0 - self.stop_loss_pct))
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
                (score >= -self.exit_score)
                or (item.bars_held >= self.max_hold_bars)
                or (float(close) >= float(item.entry_price or close) * (1.0 + self.stop_loss_pct))
            )
            if should_exit:
                self._emit(symbol, event_time, "EXIT", metadata={**metadata, "reason": "short_exit"})
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return

        if not liquidity_ok or not vol_ok:
            return

        strength = min(2.0, max(0.2, abs(score)))
        if score >= self.entry_score:
            self._emit(
                symbol,
                event_time,
                "LONG",
                stop_loss=float(close) * (1.0 - self.stop_loss_pct),
                strength=strength,
                metadata={**metadata, "reason": "long_entry"},
            )
            item.mode = "LONG"
            item.entry_price = float(close)
            item.bars_held = 0
            return

        if self.allow_short and score <= -self.entry_score:
            self._emit(
                symbol,
                event_time,
                "SHORT",
                stop_loss=float(close) * (1.0 + self.stop_loss_pct),
                strength=strength,
                metadata={**metadata, "reason": "short_entry"},
            )
            item.mode = "SHORT"
            item.entry_price = float(close)
            item.bars_held = 0
            return

        _ = math.copysign(1.0, predicted_return) if predicted_return != 0.0 else 0.0
