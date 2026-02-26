from __future__ import annotations

from abc import ABC
from typing import Any

from lumina_quant.core.events import MarketEvent
from lumina_quant.event_clock import EventSequencer, assign_event_identity
from lumina_quant.message_bus import MessageBus


def _event_time_to_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = int(float(value))
        if abs(numeric) < 100_000_000_000:
            return numeric * 1000
        return numeric
    ts_fn = getattr(value, "timestamp", None)
    if callable(ts_fn):
        try:
            ts_value = ts_fn()
            if isinstance(ts_value, (int, float)):
                return int(float(ts_value) * 1000)
        except Exception:
            return None
    return None


class TradingEngine(ABC):
    """Abstract base class for trading engines (Backtest and LiveTrader).
    Encapsulates common event processing logic (The "Kernel").
    """

    def __init__(self, events, data_handler, strategy, portfolio, execution_handler):
        self.events = events
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self._event_sequencer = EventSequencer()
        self.message_bus = MessageBus()
        self.timeframe_aggregator = None
        self._window_decision_last_bucket: int | None = None

        # Stats
        self.market_events = 0
        self.signals = 0
        self.orders = 0
        self.fills = 0

    def process_event(self, event):
        """Routing logic for events."""
        if event is not None:
            assign_event_identity(event, self._event_sequencer)
            event_type = str(getattr(event, "type", "UNKNOWN")).upper()
            self.message_bus.publish(f"event.{event_type}", event)
            if event.type == "MARKET":
                self.handle_market_event(event)
            elif event.type == "MARKET_BATCH":
                self.handle_market_batch_event(event)
            elif event.type == "MARKET_WINDOW":
                self.handle_market_window_event(event)
            elif event.type == "SIGNAL":
                self.handle_signal_event(event)
            elif event.type == "ORDER":
                self.handle_order_event(event)
            elif event.type == "FILL":
                self.handle_fill_event(event)

    def handle_market_event(self, event):
        self.market_events += 1
        should_process = True
        strategy_guard = getattr(self.strategy, "should_process_market_event", None)
        if callable(strategy_guard):
            try:
                should_process = bool(strategy_guard(event))
            except Exception:
                should_process = True
        if should_process:
            self.strategy.calculate_signals(event)
        self.portfolio.update_timeindex(event)
        # Optional: Simulated execution handler might need to check open orders
        if hasattr(self.execution_handler, "check_open_orders"):
            self.execution_handler.check_open_orders(event)

    def handle_market_batch_event(self, event):
        batch_events = tuple(getattr(event, "bars", ()) or ())
        if not batch_events:
            return

        strategy_guard = getattr(self.strategy, "should_process_market_event", None)
        for market_event in batch_events:
            self.market_events += 1
            should_process = True
            if callable(strategy_guard):
                try:
                    should_process = bool(strategy_guard(market_event))
                except Exception:
                    should_process = True
            if should_process:
                self.strategy.calculate_signals(market_event)

        # Timestamp-level accounting update (once per second).
        self.portfolio.update_timeindex(event)

        if hasattr(self.execution_handler, "check_open_orders"):
            for market_event in batch_events:
                self.execution_handler.check_open_orders(market_event)

    def _resolve_required_timeframes(self) -> list[str]:
        default_timeframes = ["20s", "1m", "5m", "15m", "1h", "4h", "1d"]
        resolved = list(default_timeframes)

        raw = getattr(self.strategy, "required_timeframes", None)
        if callable(raw):
            try:
                raw = raw()
            except Exception:
                raw = None
        if isinstance(raw, (list, tuple, set)):
            for token in raw:
                item = str(token or "").strip()
                if item:
                    resolved.append(item)
        return resolved

    def _resolve_required_lookbacks(self) -> dict[str, int]:
        raw = getattr(self.strategy, "required_lookbacks", None)
        if callable(raw):
            try:
                raw = raw()
            except Exception:
                raw = None
        if not isinstance(raw, dict):
            return {}

        out: dict[str, int] = {}
        for key, value in raw.items():
            try:
                out[str(key)] = max(1, int(value))
            except Exception:
                continue
        return out

    def _ensure_timeframe_aggregator(self):
        if self.timeframe_aggregator is not None:
            return self.timeframe_aggregator
        try:
            from lumina_quant.timeframe_aggregator import TimeframeAggregator

            self.timeframe_aggregator = TimeframeAggregator(
                timeframes=self._resolve_required_timeframes(),
                lookbacks=self._resolve_required_lookbacks(),
            )
        except Exception:
            self.timeframe_aggregator = None
        return self.timeframe_aggregator

    @staticmethod
    def _coerce_market_event(symbol: str, row: Any, fallback_time: Any) -> MarketEvent | None:
        if isinstance(row, MarketEvent):
            return row
        if isinstance(row, dict):
            return MarketEvent(
                time=row.get("time") or row.get("datetime") or fallback_time,
                symbol=str(row.get("symbol") or symbol),
                open=float(row.get("open", 0.0)),
                high=float(row.get("high", 0.0)),
                low=float(row.get("low", 0.0)),
                close=float(row.get("close", 0.0)),
                volume=float(row.get("volume", 0.0)),
            )
        if isinstance(row, (tuple, list)) and len(row) >= 6:
            return MarketEvent(
                time=row[0] if row[0] is not None else fallback_time,
                symbol=str(symbol),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        return None

    def _should_process_market_window_event(self, event: Any) -> bool:
        raw_cadence = getattr(self.strategy, "decision_cadence_seconds", None)
        if callable(raw_cadence):
            try:
                raw_cadence = raw_cadence()
            except Exception:
                raw_cadence = None
        try:
            cadence_seconds = int(raw_cadence) if raw_cadence is not None else 0
        except Exception:
            cadence_seconds = 0

        if cadence_seconds <= 0:
            return True

        event_ms = _event_time_to_ms(getattr(event, "time", None))
        if event_ms is None:
            return True

        cadence_ms = max(1_000, cadence_seconds * 1000)
        bucket = int(event_ms // cadence_ms)
        if self._window_decision_last_bucket == bucket:
            return False
        self._window_decision_last_bucket = bucket
        return True

    def handle_market_window_event(self, event):
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        total_bars = sum(len(values or ()) for values in bars_1s.values())
        self.market_events += int(total_bars if total_bars > 0 else 1)

        aggregator = self._ensure_timeframe_aggregator()
        if aggregator is not None:
            aggregator.update_from_1s_batch(bars_1s)

        if self._should_process_market_window_event(event):
            window_fn = getattr(self.strategy, "calculate_signals_window", None)
            if callable(window_fn):
                try:
                    window_fn(event, aggregator)
                except TypeError:
                    self.strategy.calculate_signals(event)
            else:
                self.strategy.calculate_signals(event)

        # Update portfolio once per decision tick.
        self.portfolio.update_timeindex(event)

        if hasattr(self.execution_handler, "check_open_orders"):
            for symbol, rows in bars_1s.items():
                if not rows:
                    continue
                market_event = self._coerce_market_event(
                    symbol=str(symbol),
                    row=rows[-1],
                    fallback_time=getattr(event, "time", None),
                )
                if market_event is not None:
                    self.execution_handler.check_open_orders(market_event)

    def handle_signal_event(self, event):
        self.signals += 1
        self.portfolio.update_signal(event)

    def handle_order_event(self, event):
        self.orders += 1
        self.execution_handler.execute_order(event)

    def handle_fill_event(self, event):
        self.fills += 1
        self.portfolio.update_fill(event)
        # Hook for state saving (LiveTrader can override or we add a hook)
        self.on_fill(event)

    def get_engine_state(self) -> dict[str, Any]:
        """Capture engine-level state for chunk boundaries."""
        state: dict[str, Any] = {
            "window_decision_last_bucket": self._window_decision_last_bucket,
        }
        aggregator = self._ensure_timeframe_aggregator()
        if aggregator is not None:
            get_state_fn = getattr(aggregator, "get_state", None)
            if callable(get_state_fn):
                try:
                    state["timeframe_aggregator"] = get_state_fn()
                except Exception:
                    pass
        return state

    def set_engine_state(self, state: dict[str, Any]) -> None:
        """Restore engine-level state from `get_engine_state()` output."""
        if not isinstance(state, dict):
            return

        if "window_decision_last_bucket" in state:
            raw = state.get("window_decision_last_bucket")
            try:
                self._window_decision_last_bucket = int(raw) if raw is not None else None
            except Exception:
                self._window_decision_last_bucket = None

        aggregator_state = state.get("timeframe_aggregator")
        if isinstance(aggregator_state, dict):
            aggregator = self._ensure_timeframe_aggregator()
            if aggregator is not None:
                set_state_fn = getattr(aggregator, "set_state", None)
                if callable(set_state_fn):
                    set_state_fn(dict(aggregator_state))

    def on_fill(self, event):
        """Hook for post-fill actions (e.g. logging, saving state).
        Override in subclasses.
        """
        pass
