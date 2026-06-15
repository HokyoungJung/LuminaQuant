"""Dormant calendar-seasonality overlay alpha sleeve (S11, per-index single-asset).

``CalendarSeasonalityOverlayStrategy`` tilts a single index perp (SPY or QQQ) by
two well-documented calendar effects derived purely from the bar timestamp:

- Turn-of-month: go long around the last/first trading days of the month.
- Day-of-week: a configurable per-weekday tilt (long/flat).

Because each candidate trades exactly ONE symbol, it is
``candidate_mix_type == "single"`` and is wired per-index (one candidate per
index), so it sidesteps the multi-asset admission gate entirely.  It is a no-op
when the configured index perp is absent.  The sleeve uses only ``datetime`` —
no external features — keeps a bounded close deque, and self-skips on missing
prices.  It auto-discovers through the plugin registry; validate scoring on the
data-bearing machine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_DEFAULT_INDEX_PREFERENCES = ("SPYUSDT", "SPY/USDT", "QQQUSDT", "QQQ/USDT")


@dataclass(slots=True)
class _CalendarState:
    closes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


def _mode(raw: Any) -> str:
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _restore_deque(target: deque[float], payload: Any) -> None:
    target.clear()
    for value in list(payload or [])[-int(target.maxlen or 0) :]:
        parsed = safe_float(value)
        if parsed is not None:
            target.append(parsed)


def _is_turn_of_month(dt: datetime, *, pre_days: int, post_days: int) -> bool:
    # Last ``pre_days`` calendar days of the month, or first ``post_days`` days.
    if dt.day <= max(0, int(post_days)):
        return True
    # Days remaining until month end.
    days_to_end = 0
    cursor = dt
    while True:
        cursor = cursor + timedelta(days=1)
        if cursor.month == dt.month:
            days_to_end += 1
        else:
            break
    return days_to_end < max(0, int(pre_days))


@register("strategy", "CalendarSeasonalityOverlayStrategy", interface="event_driven")
class CalendarSeasonalityOverlayStrategy(Strategy):
    """Per-index single-asset turn-of-month and day-of-week seasonality tilt."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "index_symbol": HyperParam.string("index_symbol", default="SPYUSDT", tunable=False),
            "turn_of_month_pre_days": HyperParam.integer(
                "turn_of_month_pre_days", default=3, low=0, high=15
            ),
            "turn_of_month_post_days": HyperParam.integer(
                "turn_of_month_post_days", default=3, low=0, high=15
            ),
            "enable_turn_of_month": HyperParam.boolean(
                "enable_turn_of_month", default=True, grid=[True, False]
            ),
            "enable_day_of_week": HyperParam.boolean(
                "enable_day_of_week", default=True, grid=[True, False]
            ),
            # Per-weekday long tilt mask (Mon..Sun); default: long Mon/Tue/Fri.
            "weekday_long_mask": HyperParam.string(
                "weekday_long_mask", default="1,1,0,0,1,0,0", tunable=False
            ),
            "hold_bars": HyperParam.integer("hold_bars", default=2, low=1, high=4096),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.05, low=0.0, high=0.50
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        preferred = str(resolved["index_symbol"])
        self.index_symbol = preferred if preferred in self.symbol_list else ""
        if not self.index_symbol:
            for candidate in _DEFAULT_INDEX_PREFERENCES:
                if candidate in self.symbol_list:
                    self.index_symbol = candidate
                    break
        self.turn_of_month_pre_days = max(0, int(resolved["turn_of_month_pre_days"]))
        self.turn_of_month_post_days = max(0, int(resolved["turn_of_month_post_days"]))
        self.enable_turn_of_month = bool(resolved["enable_turn_of_month"])
        self.enable_day_of_week = bool(resolved["enable_day_of_week"])
        self.weekday_long_mask = self._parse_mask(str(resolved["weekday_long_mask"] or ""))
        self.hold_bars = max(1, int(resolved["hold_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = max(8, self.hold_bars + 8)
        self._state = _CalendarState(deque(maxlen=size))

    @staticmethod
    def _parse_mask(raw: str) -> list[bool]:
        out: list[bool] = []
        for chunk in raw.split(","):
            text = chunk.strip()
            if not text:
                continue
            out.append(text not in {"0", "false", "False", "no"})
        while len(out) < 7:
            out.append(False)
        return out[:7]

    def get_state(self) -> dict[str, Any]:
        item = self._state
        return {
            "closes": list(item.closes),
            "mode": item.mode,
            "entry_price": item.entry_price,
            "bars_held": int(item.bars_held),
            "last_time_key": item.last_time_key,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        item = self._state
        _restore_deque(item.closes, state.get("closes"))
        item.mode = _mode(state.get("mode"))
        item.entry_price = safe_float(state.get("entry_price"))
        item.bars_held = _safe_non_negative_int(state.get("bars_held"))
        item.last_time_key = str(state.get("last_time_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        if not self.index_symbol:
            return
        for symbol in _event_symbols(event, [self.index_symbol]):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._process(snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        if not self.index_symbol:
            return
        symbol = getattr(event, "symbol", None)
        if symbol == self.index_symbol:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                self._process(snapshot)

    def _exit_if_needed(self, snapshot: _Snapshot, tilt_long: bool) -> None:
        item = self._state
        close = safe_float(snapshot.close)
        if item.mode != "LONG" or item.entry_price is None or close is None:
            return
        item.bars_held += 1
        reason = ""
        if self.stop_loss_pct > 0.0 and close <= item.entry_price * (1.0 - self.stop_loss_pct):
            reason = "stop_loss"
        elif item.bars_held >= self.hold_bars:
            reason = "max_hold"
        elif not tilt_long:
            reason = "seasonality_off"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id="calendar_seasonality_overlay",
            symbol=self.index_symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "CalendarSeasonalityOverlayStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0

    def _seasonal_long(self, raw_time: Any) -> bool:
        dt = _event_datetime_utc(raw_time)
        if dt is None:
            return False
        dt = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        long_signal = False
        if self.enable_turn_of_month and _is_turn_of_month(
            dt,
            pre_days=self.turn_of_month_pre_days,
            post_days=self.turn_of_month_post_days,
        ):
            long_signal = True
        if self.enable_day_of_week:
            weekday = dt.weekday()
            if 0 <= weekday < len(self.weekday_long_mask) and self.weekday_long_mask[weekday]:
                long_signal = True
        return long_signal

    def _process(self, snapshot: _Snapshot) -> None:
        item = self._state
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        item.closes.append(close)
        tilt_long = self._seasonal_long(snapshot.time)
        self._exit_if_needed(snapshot, tilt_long)
        if item.mode != "OUT" or not tilt_long:
            return
        stop_loss = None
        if self.stop_loss_pct > 0.0:
            stop_loss = close * (1.0 - self.stop_loss_pct)
        metadata = _target_metadata(
            strategy="CalendarSeasonalityOverlayStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            target_mode="LONG",
            reason="seasonality_tilt",
        )
        _emit(
            self.events,
            strategy_id="calendar_seasonality_overlay",
            symbol=self.index_symbol,
            event_time=snapshot.time,
            signal_type="LONG",
            strength=1.0,
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = "LONG"
        item.entry_price = close
        item.bars_held = 0


__all__ = ["CalendarSeasonalityOverlayStrategy"]
