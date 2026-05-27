"""Panic-rebound mean-reversion strategy for profit-reboot research.

The strategy is intentionally event-driven and bounded-memory: it consumes the
latest bar only, waits for a confirmed rebound after a downside liquidation
shock, and exits quickly through hard stop, take-profit, trailing, or time-stop
rules.  It never enters on the shock bar itself, which keeps the confirmation
path live-equivalent and avoids "catching the falling knife" without a later
bar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, safe_int, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _PanicSymbolState:
    closes: deque[float]
    highs: deque[float]
    lows: deque[float]
    volumes: deque[float]
    returns: deque[float]
    mode: str = "OUT"
    entry_price: float = 0.0
    high_watermark: float = 0.0
    bars_held: int = 0
    last_time_key: str = ""
    pending_shock: dict[str, float] = field(default_factory=dict)


class PanicReboundMeanReversionStrategy(Strategy):
    """Long-only liquidation rebound strategy with explicit confirmation.

    Rules:
    - Maintain one bounded history per symbol.
    - Detect a large negative one-bar return with elevated volume.
    - Wait for a later confirmation bar that reclaims local VWAP and rebounds
      from the shock close before entering long.
    - Exit via hard stop, take profit, trailing exit, VWAP failure, or max hold.
    """

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "history_bars": HyperParam.integer(
                "history_bars", default=160, low=16, high=20000, tunable=False
            ),
            "return_window": HyperParam.integer(
                "return_window", default=32, low=4, high=4096, tunable=False
            ),
            "volume_window": HyperParam.integer(
                "volume_window", default=32, low=4, high=4096, tunable=False
            ),
            "vwap_window": HyperParam.integer(
                "vwap_window", default=24, low=2, high=4096, tunable=False
            ),
            "shock_return_z": HyperParam.floating(
                "shock_return_z", default=2.0, low=0.0, high=20.0, tunable=False
            ),
            "shock_return_pct": HyperParam.floating(
                "shock_return_pct", default=0.025, low=0.0, high=1.0, tunable=False
            ),
            "volume_z": HyperParam.floating(
                "volume_z", default=1.0, low=0.0, high=20.0, tunable=False
            ),
            "confirmation_bars": HyperParam.integer(
                "confirmation_bars", default=3, low=1, high=120, tunable=False
            ),
            "min_rebound_pct": HyperParam.floating(
                "min_rebound_pct", default=0.006, low=0.0, high=1.0, tunable=False
            ),
            "vwap_recovery_pct": HyperParam.floating(
                "vwap_recovery_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.018, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.035, low=0.0, high=2.0, tunable=False
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.018, low=0.0, high=1.0, tunable=False
            ),
            "max_hold_bars": HyperParam.integer(
                "max_hold_bars", default=18, low=1, high=10000, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.08, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating(
                "min_price", default=0.10, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        history_bars: int = 160,
        return_window: int = 32,
        volume_window: int = 32,
        vwap_window: int = 24,
        shock_return_z: float = 2.0,
        shock_return_pct: float = 0.025,
        volume_z: float = 1.0,
        confirmation_bars: int = 3,
        min_rebound_pct: float = 0.006,
        vwap_recovery_pct: float = 0.0,
        stop_loss_pct: float = 0.018,
        take_profit_pct: float = 0.035,
        trailing_exit_pct: float = 0.018,
        max_hold_bars: int = 18,
        target_allocation: float = 0.08,
        max_order_value: float = 300.0,
        min_price: float = 0.10,
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError("PanicReboundMeanReversionStrategy requires at least one symbol.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "history_bars": history_bars,
                "return_window": return_window,
                "volume_window": volume_window,
                "vwap_window": vwap_window,
                "shock_return_z": shock_return_z,
                "shock_return_pct": shock_return_pct,
                "volume_z": volume_z,
                "confirmation_bars": confirmation_bars,
                "min_rebound_pct": min_rebound_pct,
                "vwap_recovery_pct": vwap_recovery_pct,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_exit_pct": trailing_exit_pct,
                "max_hold_bars": max_hold_bars,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "min_price": min_price,
            },
            keep_unknown=False,
        )

        self.return_window = max(4, int(resolved["return_window"]))
        self.volume_window = max(4, int(resolved["volume_window"]))
        self.vwap_window = max(2, int(resolved["vwap_window"]))
        self.shock_return_z = max(0.0, float(resolved["shock_return_z"]))
        self.shock_return_pct = max(0.0, float(resolved["shock_return_pct"]))
        self.volume_z = max(0.0, float(resolved["volume_z"]))
        self.confirmation_bars = max(1, int(resolved["confirmation_bars"]))
        self.min_rebound_pct = max(0.0, float(resolved["min_rebound_pct"]))
        self.vwap_recovery_pct = max(0.0, float(resolved["vwap_recovery_pct"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))

        history_len = max(
            int(resolved["history_bars"]),
            self.return_window + self.confirmation_bars + 4,
            self.volume_window + self.confirmation_bars + 4,
            self.vwap_window + self.confirmation_bars + 4,
        )
        self._state: dict[str, _PanicSymbolState] = {
            symbol: _PanicSymbolState(
                closes=deque(maxlen=history_len),
                highs=deque(maxlen=history_len),
                lows=deque(maxlen=history_len),
                volumes=deque(maxlen=history_len),
                returns=deque(maxlen=history_len),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(state.closes),
                    "highs": list(state.highs),
                    "lows": list(state.lows),
                    "volumes": list(state.volumes),
                    "returns": list(state.returns),
                    "mode": state.mode,
                    "entry_price": state.entry_price,
                    "high_watermark": state.high_watermark,
                    "bars_held": state.bars_held,
                    "last_time_key": state.last_time_key,
                    "pending_shock": dict(state.pending_shock),
                }
                for symbol, state in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        raw_symbols = state.get("symbol_state")
        if not isinstance(raw_symbols, dict):
            return
        for symbol, raw in raw_symbols.items():
            if symbol not in self._state or not isinstance(raw, dict):
                continue
            item = self._state[symbol]
            for attr in ("closes", "highs", "lows", "volumes", "returns"):
                target = getattr(item, attr)
                target.clear()
                keep = int(target.maxlen) if target.maxlen is not None else 0
                for value in list(raw.get(attr) or [])[-keep:]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(float(parsed))
            mode = str(raw.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG"} else "OUT"
            item.entry_price = float(safe_float(raw.get("entry_price")) or 0.0)
            item.high_watermark = float(safe_float(raw.get("high_watermark")) or 0.0)
            item.bars_held = max(0, safe_int(raw.get("bars_held"), 0))
            item.last_time_key = str(raw.get("last_time_key", ""))
            pending = raw.get("pending_shock")
            item.pending_shock = {
                str(key): float(value)
                for key, value in dict(pending or {}).items()
                if safe_float(value) is not None
            }

    @staticmethod
    def _zscore(value: float, history: deque[float], window: int) -> float | None:
        values = list(history)[-window:]
        if len(values) < max(4, window // 2):
            return None
        sigma = stdev(values) if len(values) > 1 else 0.0
        if sigma <= 1e-12:
            return None
        return (float(value) - mean(values)) / sigma

    def _latest_bar(
        self, symbol: str, row: Any | None = None
    ) -> tuple[float, float, float, float] | None:
        if isinstance(row, dict):
            close = safe_float(row.get("close"))
            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            volume = safe_float(row.get("volume"))
        elif isinstance(row, (tuple, list)) and len(row) >= 6:
            high = safe_float(row[2])
            low = safe_float(row[3])
            close = safe_float(row[4])
            volume = safe_float(row[5])
        else:
            close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
            high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
            low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
            volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))

        if close is None or close <= self.min_price:
            return None
        high = float(high if high is not None and high > 0.0 else close)
        low = float(low if low is not None and low > 0.0 else close)
        volume = float(volume if volume is not None and volume >= 0.0 else 0.0)
        return float(close), max(high, float(close)), min(low, float(close)), volume

    def _local_vwap(self, item: _PanicSymbolState) -> float | None:
        closes = list(item.closes)[-self.vwap_window :]
        volumes = list(item.volumes)[-self.vwap_window :]
        if not closes:
            return None
        weighted = sum(
            close * max(0.0, volume) for close, volume in zip(closes, volumes, strict=False)
        )
        total_volume = sum(max(0.0, volume) for volume in volumes)
        if total_volume <= 1e-12:
            return mean(closes)
        return weighted / total_volume

    def _emit(
        self,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float,
        *,
        reason: str,
        return_z: float | None = None,
        volume_z: float | None = None,
        vwap: float | None = None,
    ) -> None:
        stop_loss = price * (1.0 - self.stop_loss_pct) if signal_type == "LONG" else None
        take_profit = price * (1.0 + self.take_profit_pct) if signal_type == "LONG" else None
        metadata: dict[str, Any] = {
            "strategy": "PanicReboundMeanReversionStrategy",
            "reason": reason,
            "return_z": float(return_z) if return_z is not None else None,
            "volume_z": float(volume_z) if volume_z is not None else None,
            "local_vwap": float(vwap) if vwap is not None else None,
            "target_allocation": float(self.target_allocation),
            "max_symbol_exposure_pct": float(self.target_allocation),
            "max_order_value": float(self.max_order_value),
        }
        self.events.put(
            SignalEvent(
                strategy_id="panic_rebound_mean_reversion",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(self.target_allocation if signal_type == "LONG" else 1.0),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_percent=self.trailing_exit_pct if signal_type == "LONG" else None,
                metadata=metadata,
            )
        )

    def _process_bar(
        self,
        symbol: str,
        event_time: Any,
        close: float,
        high: float,
        low: float,
        volume: float,
    ) -> None:
        item = self._state[symbol]
        event_time_key = time_key(event_time)
        if not event_time_key or item.last_time_key == event_time_key:
            return
        item.last_time_key = event_time_key

        previous_close = float(item.closes[-1]) if item.closes else None
        current_return = (
            (float(close) / previous_close) - 1.0
            if previous_close is not None and previous_close > 0.0
            else None
        )
        return_z = (
            self._zscore(float(current_return), item.returns, self.return_window)
            if current_return is not None
            else None
        )
        volume_z = self._zscore(float(volume), item.volumes, self.volume_window)

        item.closes.append(float(close))
        item.highs.append(float(high))
        item.lows.append(float(low))
        item.volumes.append(float(volume))
        if current_return is not None:
            item.returns.append(float(current_return))
        vwap = self._local_vwap(item)

        if item.mode == "LONG":
            item.bars_held += 1
            item.high_watermark = max(item.high_watermark, float(high), float(close))
            stop_hit = close <= item.entry_price * (1.0 - self.stop_loss_pct)
            take_profit_hit = close >= item.entry_price * (1.0 + self.take_profit_pct)
            trailing_hit = (
                self.trailing_exit_pct > 0.0
                and item.high_watermark > item.entry_price
                and close <= item.high_watermark * (1.0 - self.trailing_exit_pct)
            )
            vwap_failure = vwap is not None and close < vwap * (1.0 - self.stop_loss_pct)
            time_stop = item.bars_held >= self.max_hold_bars
            if stop_hit or take_profit_hit or trailing_hit or vwap_failure or time_stop:
                reason = "stop_loss"
                if take_profit_hit:
                    reason = "take_profit"
                elif trailing_hit:
                    reason = "trailing_exit"
                elif vwap_failure:
                    reason = "vwap_failure"
                elif time_stop:
                    reason = "max_hold"
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    close,
                    reason=reason,
                    return_z=return_z,
                    volume_z=volume_z,
                    vwap=vwap,
                )
                item.mode = "OUT"
                item.entry_price = 0.0
                item.high_watermark = 0.0
                item.bars_held = 0
            return

        pending = item.pending_shock
        if pending:
            pending["age"] = float(pending.get("age", 0.0) + 1.0)
            shock_close = float(pending.get("shock_close", close))
            shock_low = float(pending.get("shock_low", low))
            reclaimed_vwap = vwap is None or close >= vwap * (1.0 + self.vwap_recovery_pct)
            rebounded = close >= shock_close * (1.0 + self.min_rebound_pct)
            invalidated = close <= shock_low * (1.0 - self.stop_loss_pct)
            if reclaimed_vwap and rebounded and not invalidated:
                self._emit(
                    symbol,
                    event_time,
                    "LONG",
                    close,
                    reason="confirmed_panic_rebound",
                    return_z=return_z,
                    volume_z=volume_z,
                    vwap=vwap,
                )
                item.mode = "LONG"
                item.entry_price = float(close)
                item.high_watermark = float(close)
                item.bars_held = 0
                item.pending_shock = {}
                return
            if invalidated or pending["age"] >= self.confirmation_bars:
                item.pending_shock = {}

        enough_history = (
            len(item.returns) >= self.return_window and len(item.volumes) >= self.volume_window
        )
        if not enough_history or current_return is None:
            return
        negative_shock = current_return <= -self.shock_return_pct
        abnormal_return = return_z is not None and return_z <= -self.shock_return_z
        abnormal_volume = volume_z is not None and volume_z >= self.volume_z
        if negative_shock and abnormal_return and abnormal_volume:
            item.pending_shock = {
                "age": 0.0,
                "shock_close": float(close),
                "shock_low": float(low),
                "return_z": float(return_z),
                "volume_z": float(volume_z),
            }

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        _ = aggregator
        if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
            self.calculate_signals(event)
            return

        event_time = getattr(event, "time", None)
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        for symbol in self.symbol_list:
            rows = list(bars_1s.get(symbol) or [])
            if not rows:
                continue
            latest = self._latest_bar(symbol, rows[-1])
            if latest is None:
                continue
            self._process_bar(symbol, event_time, *latest)

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = str(getattr(event, "symbol", "") or "")
        if symbol not in self._state:
            return
        event_time = getattr(event, "time", None) or self.bars.get_latest_bar_datetime(symbol)
        latest = self._latest_bar(symbol, event)
        if latest is None:
            return
        self._process_bar(symbol, event_time, *latest)


__all__ = ["PanicReboundMeanReversionStrategy"]
