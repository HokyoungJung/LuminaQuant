"""Compression-breakout continuation strategy for profit-reboot research."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import quantiles
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, safe_int, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _CompressionBreakoutState:
    closes: deque[float]
    highs: deque[float]
    lows: deque[float]
    ranges: deque[float]
    mode: str = "OUT"
    entry_price: float = 0.0
    high_watermark: float = 0.0
    bars_held: int = 0
    last_time_key: str = ""


class CompressionBreakoutContinuationStrategy(Strategy):
    """Long-only breakout after volatility compression and BTC confirmation."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer(
                "lookback_bars", default=48, low=4, high=10000, tunable=False
            ),
            "compression_window": HyperParam.integer(
                "compression_window", default=24, low=4, high=10000, tunable=False
            ),
            "compression_history_bars": HyperParam.integer(
                "compression_history_bars", default=160, low=20, high=20000, tunable=False
            ),
            "compression_percentile": HyperParam.floating(
                "compression_percentile", default=0.25, low=0.0, high=1.0, tunable=False
            ),
            "breakout_buffer": HyperParam.floating(
                "breakout_buffer", default=0.002, low=0.0, high=1.0, tunable=False
            ),
            "broad_lookback_bars": HyperParam.integer(
                "broad_lookback_bars", default=24, low=2, high=10000, tunable=False
            ),
            "broad_threshold": HyperParam.floating(
                "broad_threshold", default=0.0, low=-1.0, high=1.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.025, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.060, low=0.0, high=2.0, tunable=False
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.030, low=0.0, high=1.0, tunable=False
            ),
            "max_hold_bars": HyperParam.integer(
                "max_hold_bars", default=72, low=1, high=10000, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.10, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=350.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "btc_symbol": HyperParam.string("btc_symbol", default="BTC/USDT", tunable=False),
            "min_price": HyperParam.floating(
                "min_price", default=0.10, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        lookback_bars: int = 48,
        compression_window: int = 24,
        compression_history_bars: int = 160,
        compression_percentile: float = 0.25,
        breakout_buffer: float = 0.002,
        broad_lookback_bars: int = 24,
        broad_threshold: float = 0.0,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.060,
        trailing_exit_pct: float = 0.030,
        max_hold_bars: int = 72,
        target_allocation: float = 0.10,
        max_order_value: float = 350.0,
        btc_symbol: str | None = None,
        min_price: float = 0.10,
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError(
                "CompressionBreakoutContinuationStrategy requires at least one symbol."
            )

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback_bars": lookback_bars,
                "compression_window": compression_window,
                "compression_history_bars": compression_history_bars,
                "compression_percentile": compression_percentile,
                "breakout_buffer": breakout_buffer,
                "broad_lookback_bars": broad_lookback_bars,
                "broad_threshold": broad_threshold,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_exit_pct": trailing_exit_pct,
                "max_hold_bars": max_hold_bars,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "btc_symbol": btc_symbol,
                "min_price": min_price,
            },
            keep_unknown=False,
        )

        self.lookback_bars = max(4, int(resolved["lookback_bars"]))
        self.compression_window = max(4, int(resolved["compression_window"]))
        self.compression_history_bars = max(20, int(resolved["compression_history_bars"]))
        self.compression_percentile = min(1.0, max(0.0, float(resolved["compression_percentile"])))
        self.breakout_buffer = max(0.0, float(resolved["breakout_buffer"]))
        self.broad_lookback_bars = max(2, int(resolved["broad_lookback_bars"]))
        self.broad_threshold = float(resolved["broad_threshold"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        default_btc = "BTC/USDT" if "BTC/USDT" in self.symbol_list else self.symbol_list[0]
        candidate_btc = str(resolved["btc_symbol"] or "").strip()
        self.btc_symbol = candidate_btc if candidate_btc in self.symbol_list else default_btc

        history_len = (
            max(
                self.lookback_bars,
                self.compression_window,
                self.broad_lookback_bars,
                self.compression_history_bars,
            )
            + 4
        )
        self._state = {
            symbol: _CompressionBreakoutState(
                closes=deque(maxlen=history_len),
                highs=deque(maxlen=history_len),
                lows=deque(maxlen=history_len),
                ranges=deque(maxlen=self.compression_history_bars),
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
                    "ranges": list(state.ranges),
                    "mode": state.mode,
                    "entry_price": state.entry_price,
                    "high_watermark": state.high_watermark,
                    "bars_held": state.bars_held,
                    "last_time_key": state.last_time_key,
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
            for attr in ("closes", "highs", "lows", "ranges"):
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

    def _latest_bar(self, symbol: str, row: Any | None = None) -> tuple[float, float, float] | None:
        if isinstance(row, dict):
            high = safe_float(row.get("high"))
            low = safe_float(row.get("low"))
            close = safe_float(row.get("close"))
        elif isinstance(row, (tuple, list)) and len(row) >= 5:
            high = safe_float(row[2])
            low = safe_float(row[3])
            close = safe_float(row[4])
        else:
            high = safe_float(self.bars.get_latest_bar_value(symbol, "high"))
            low = safe_float(self.bars.get_latest_bar_value(symbol, "low"))
            close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close is None or close <= self.min_price:
            return None
        high = float(high if high is not None and high > 0.0 else close)
        low = float(low if low is not None and low > 0.0 else close)
        return float(close), max(high, float(close)), min(low, float(close))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        if len(values) < 4:
            return sorted(values)[0]
        buckets = quantiles(values, n=100, method="inclusive")
        index = int(max(0, min(98, round(percentile * 99) - 1)))
        return float(buckets[index])

    def _broad_momentum(self) -> float | None:
        state = self._state.get(self.btc_symbol)
        if state is None or len(state.closes) <= self.broad_lookback_bars:
            return None
        base = float(list(state.closes)[-self.broad_lookback_bars - 1])
        current = float(state.closes[-1])
        if base <= 0.0:
            return None
        return (current / base) - 1.0

    def _compression_active(self, state: _CompressionBreakoutState) -> tuple[bool, float | None]:
        if len(state.highs) < self.compression_window or len(state.ranges) < 8:
            return False, None
        highs = list(state.highs)[-self.compression_window :]
        lows = list(state.lows)[-self.compression_window :]
        close = float(state.closes[-1])
        if close <= 0.0:
            return False, None
        current_range = (max(highs) - min(lows)) / close
        threshold = self._percentile(list(state.ranges), self.compression_percentile)
        if threshold is None:
            return False, current_range
        return current_range <= threshold, current_range

    def _emit(
        self,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float,
        *,
        reason: str,
        broad_momentum: float | None,
        compression_range: float | None,
    ) -> None:
        stop_loss = price * (1.0 - self.stop_loss_pct) if signal_type == "LONG" else None
        take_profit = price * (1.0 + self.take_profit_pct) if signal_type == "LONG" else None
        metadata = {
            "strategy": "CompressionBreakoutContinuationStrategy",
            "reason": reason,
            "broad_momentum": broad_momentum,
            "compression_range": compression_range,
            "lookback_bars": int(self.lookback_bars),
            "compression_window": int(self.compression_window),
            "target_allocation": float(self.target_allocation),
            "max_symbol_exposure_pct": float(self.target_allocation),
            "max_order_value": float(self.max_order_value),
        }
        self.events.put(
            SignalEvent(
                strategy_id="compression_breakout_continuation",
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
        self, symbol: str, event_time: Any, close: float, high: float, low: float
    ) -> None:
        item = self._state[symbol]
        event_time_key = time_key(event_time)
        if not event_time_key or item.last_time_key == event_time_key:
            return
        item.last_time_key = event_time_key

        item.closes.append(float(close))
        item.highs.append(float(high))
        item.lows.append(float(low))
        if len(item.highs) >= self.compression_window:
            recent_highs = list(item.highs)[-self.compression_window :]
            recent_lows = list(item.lows)[-self.compression_window :]
            item.ranges.append((max(recent_highs) - min(recent_lows)) / float(close))

        broad_momentum = self._broad_momentum()
        broad_ok = broad_momentum is not None and broad_momentum >= self.broad_threshold
        compression_ok, compression_range = self._compression_active(item)

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
            time_stop = item.bars_held >= self.max_hold_bars
            broad_failure = broad_momentum is not None and broad_momentum < self.broad_threshold
            if stop_hit or take_profit_hit or trailing_hit or time_stop or broad_failure:
                reason = "stop_loss"
                if take_profit_hit:
                    reason = "take_profit"
                elif trailing_hit:
                    reason = "trailing_exit"
                elif time_stop:
                    reason = "max_hold"
                elif broad_failure:
                    reason = "broad_momentum_failure"
                self._emit(
                    symbol,
                    event_time,
                    "EXIT",
                    close,
                    reason=reason,
                    broad_momentum=broad_momentum,
                    compression_range=compression_range,
                )
                item.mode = "OUT"
                item.entry_price = 0.0
                item.high_watermark = 0.0
                item.bars_held = 0
            return

        if not broad_ok or not compression_ok or len(item.highs) <= self.lookback_bars:
            return
        prior_highs = list(item.highs)[-self.lookback_bars - 1 : -1]
        if not prior_highs:
            return
        breakout_level = max(prior_highs) * (1.0 + self.breakout_buffer)
        if close > breakout_level:
            self._emit(
                symbol,
                event_time,
                "LONG",
                close,
                reason="compression_breakout",
                broad_momentum=broad_momentum,
                compression_range=compression_range,
            )
            item.mode = "LONG"
            item.entry_price = float(close)
            item.high_watermark = float(close)
            item.bars_held = 0

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
            if latest is not None:
                self._process_bar(symbol, event_time, *latest)

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = str(getattr(event, "symbol", "") or "")
        if symbol not in self._state:
            return
        event_time = getattr(event, "time", None) or self.bars.get_latest_bar_datetime(symbol)
        latest = self._latest_bar(symbol, event)
        if latest is not None:
            self._process_bar(symbol, event_time, *latest)


__all__ = ["CompressionBreakoutContinuationStrategy"]
