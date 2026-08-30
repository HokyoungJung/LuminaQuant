from __future__ import annotations

import math
from abc import ABC
from inspect import Signature, signature
from typing import Any

from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.core.strategy_input import StrategyInputContext
from lumina_quant.data.feature_points import FEATURE_COLUMNS
from lumina_quant.event_clock import EventSequencer, assign_event_identity
from lumina_quant.message_bus import MessageBus
from lumina_quant.utils.timeutil import utc_epoch_ms


def _event_time_to_ms(value: Any) -> int | None:
    """Coerce event time with the shared strict UTC conversion contract."""
    return utc_epoch_ms(value)


def _warmup_time_to_ms(value: Any) -> int | None:
    """Coerce warmup time with the shared strict UTC conversion contract."""
    return utc_epoch_ms(value)


def _accepts_positional_call(function: Any, *args: Any) -> bool:
    """Whether a callable can receive these positional arguments.

    Signature inspection distinguishes an argument mismatch from a TypeError
    raised inside strategy code. Uninspectable callables are invoked normally
    so their contract failures remain visible.
    """
    try:
        contract: Signature = signature(function)
    except TypeError, ValueError:
        return True
    try:
        contract.bind(*args)
    except TypeError:
        return False
    return True


class TradingEngine(ABC):
    """Abstract base class for trading engines (Backtest and LiveTrader).
    Encapsulates common event processing logic (The "Kernel").
    """

    def __init__(
        self,
        events,
        data_handler,
        strategy,
        portfolio,
        execution_handler,
        *,
        live_start_ms: int | None = None,
    ):
        self.events = events
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self._event_sequencer = EventSequencer()
        self.message_bus = MessageBus()
        self.timeframe_aggregator = None
        self._window_decision_last_bucket: int | None = None
        # Warmup boundary: events strictly before this epoch-ms only prime
        # strategy/indicator state — no portfolio accounting, no orders, no fills.
        # None disables suppression entirely (live trading, plain backtests).
        self._live_start_ms = int(live_start_ms) if live_start_ms is not None else None
        self._warmup_active = False
        # One-shot warmup->live transition hook guard.  Persisted across chunk
        # boundaries via get_engine_state/set_engine_state so a chunked run
        # neither double-fires nor skips the hook when the warmup boundary
        # falls between chunks.  ``_had_warmup`` records that THIS RUN was
        # warmup-configured at some point: continuation chunks are constructed
        # with warmup_bars=0 (live_start_ms=None), so without the carried
        # provenance the hook could never fire when chunk 1 held only warmup
        # bars and the first live event lands in a later chunk.
        self._warmup_end_hook_fired = False
        self._had_warmup = self._live_start_ms is not None

        # Stats
        self.market_events = 0
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self._snapshot_fill_count = 0

    def _is_warmup_time(self, time_value: Any) -> bool:
        if self._live_start_ms is None:
            return False
        event_ms = _warmup_time_to_ms(time_value)
        # A bad clock must never turn a warmup event into a live-routing event.
        # Suppress it until a valid timestamp proves it is at/after the cutoff.
        return event_ms is None or event_ms < self._live_start_ms

    def _update_warmup_state(self, time_value: Any) -> bool:
        """Track whether the current market event sits in the warmup region.

        Signals/orders/fills are emitted into the queue by the market handlers
        and processed afterwards, so the flag set here governs suppression of
        the downstream events spawned by this market event.

        On the first LIVE event of a warmup-configured run this fires the
        optional one-shot ``strategy.on_warmup_end()`` hook (before the
        strategy sees the live bar), so stateful sleeves can drop "positions"
        they believe they entered during warmup — the engine suppressed those
        signals, so no portfolio position ever existed (ghost-position
        desynchronization).  Strategies without the hook are untouched.
        """
        self._warmup_active = self._is_warmup_time(time_value)
        if (
            (self._live_start_ms is not None or self._had_warmup)
            and not self._warmup_active
            and not self._warmup_end_hook_fired
        ):
            hook = getattr(self.strategy, "on_warmup_end", None)
            if callable(hook):
                # A deliberately-defined hook must not fail silently.
                hook()
            # Only a successful hook invocation commits the one-shot state.
            # A transient hook failure is retried before the next valid live bar.
            self._warmup_end_hook_fired = True
        return self._warmup_active

    def _required_inputs(self) -> tuple[str, ...]:
        raw = getattr(self.strategy, "required_inputs", ())
        return tuple(str(item).strip().lower() for item in tuple(raw or ()) if str(item).strip())

    def _required_features(self) -> tuple[str, ...]:
        raw = getattr(self.strategy, "required_features", ())
        return tuple(str(item).strip().lower() for item in tuple(raw or ()) if str(item).strip())

    def _assert_strategy_requirements(
        self,
        *,
        available_inputs: set[str],
        feature_lookup: Any,
    ) -> None:
        required_inputs = set(self._required_inputs())
        missing_inputs = sorted(required_inputs - set(available_inputs))
        if missing_inputs:
            raise RuntimeError(
                "Strategy required_inputs are unavailable for this execution path: "
                + ", ".join(missing_inputs)
            )

        required_features = set(self._required_features())
        if not required_features:
            return
        supported_features = {str(item).lower() for item in FEATURE_COLUMNS} | {"feature_points"}
        unknown_features = sorted(required_features - supported_features)
        if unknown_features:
            raise RuntimeError(
                "Strategy declared unsupported required_features: " + ", ".join(unknown_features)
            )
        if feature_lookup is None or not str(getattr(feature_lookup, "db_path", "") or "").strip():
            raise RuntimeError(
                "Strategy required_features are unavailable because feature lookup is not configured."
            )

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

    def _check_open_orders_with_equity_context(self, event: Any) -> None:
        """Run one order sweep with the exact pre-queued-fill portfolio equity."""
        check = getattr(self.execution_handler, "check_open_orders", None)
        if not callable(check):
            return
        equity_before = float(self.portfolio.current_holdings["total"])
        if bool(getattr(self.execution_handler, "record_cost_attribution", False)) and (
            not math.isfinite(equity_before) or equity_before <= 0.0
        ):
            # A terminal/wiped-out alpha run must remain evidence-producing.  No
            # finite-positive capacity proxy exists after ruin, and executing
            # another open-order sweep would create an invalid observation.
            return
        set_context = getattr(self.execution_handler, "set_capacity_equity_context", None)
        clear_context = getattr(self.execution_handler, "clear_capacity_equity_context", None)
        if callable(set_context):
            set_context(equity_before)
        try:
            check(event)
        finally:
            if callable(clear_context):
                clear_context()

    def handle_market_event(self, event):
        self.market_events += 1
        warmup = self._update_warmup_state(getattr(event, "time", None))
        should_process = True
        strategy_guard = getattr(self.strategy, "should_process_market_event", None)
        if callable(strategy_guard):
            should_process = bool(strategy_guard(event))
        if should_process:
            self._assert_strategy_requirements(
                available_inputs={
                    "market_event",
                    "ohlcv",
                    "data_handler",
                    "execution_handler",
                    "exchange",
                },
                feature_lookup=getattr(self.data_handler, "_feature_lookup", None),
            )
            self.strategy.calculate_signals(event)
        if warmup:
            # Warmup bars only prime indicator state — no equity rows, no order checks.
            return
        self.portfolio.update_timeindex(event)
        self._snapshot_fill_count = self.fills
        # Optional: Simulated execution handler might need to check open orders
        if hasattr(self.execution_handler, "check_open_orders"):
            self._check_open_orders_with_equity_context(event)

    def handle_market_batch_event(self, event):
        batch_events = tuple(getattr(event, "bars", ()) or ())
        if not batch_events:
            return

        warmup = self._update_warmup_state(getattr(event, "time", None))
        strategy_guard = getattr(self.strategy, "should_process_market_event", None)
        batch_fn = getattr(self.strategy, "calculate_signals_batch", None)
        accepted_events = []
        for market_event in batch_events:
            self.market_events += 1
            should_process = True
            if callable(strategy_guard):
                should_process = bool(strategy_guard(market_event))
            if should_process:
                self._assert_strategy_requirements(
                    available_inputs={
                        "market_event",
                        "ohlcv",
                        "data_handler",
                        "execution_handler",
                        "exchange",
                    },
                    feature_lookup=getattr(self.data_handler, "_feature_lookup", None),
                )
                if callable(batch_fn):
                    accepted_events.append(market_event)
                else:
                    self.strategy.calculate_signals(market_event)

        if callable(batch_fn) and accepted_events:
            batch_fn(
                MarketBatchEvent(
                    time=getattr(event, "time", None),
                    bars=tuple(accepted_events),
                    timestamp_ns=getattr(event, "timestamp_ns", None),
                    sequence=getattr(event, "sequence", None),
                )
            )

        if warmup:
            # Batch events share one timestamp, so the event-level warmup check
            # is exact per-bar suppression: indicators advanced above, but no
            # accounting rows and no order checks in the warmup region.
            return

        # Timestamp-level accounting update (once per second).
        self.portfolio.update_timeindex(event)
        self._snapshot_fill_count = self.fills

        if hasattr(self.execution_handler, "check_open_orders"):
            for market_event in batch_events:
                if self._is_warmup_time(getattr(market_event, "time", None)):
                    continue
                self._check_open_orders_with_equity_context(market_event)

    def _resolve_required_timeframes(self) -> list[str]:
        default_timeframes = ["20s", "1m", "5m", "15m", "1h", "4h", "1d"]
        resolved = list(default_timeframes)

        raw = getattr(self.strategy, "required_timeframes", None)
        if callable(raw):
            raw = raw()
        if raw is None:
            return resolved
        if not isinstance(raw, (list, tuple, set)):
            raise TypeError("strategy required_timeframes must be a sequence")
        for token in raw:
            if type(token) is not str or not token.strip():
                raise ValueError("strategy required_timeframes contains an invalid token")
            resolved.append(token.strip())
        return resolved

    def _resolve_required_lookbacks(self) -> dict[str, int]:
        raw = getattr(self.strategy, "required_lookbacks", None)
        if callable(raw):
            raw = raw()
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise TypeError("strategy required_lookbacks must be a mapping")

        out: dict[str, int] = {}
        for key, value in raw.items():
            if type(key) is not str or not key.strip() or type(value) is not int or value <= 0:
                raise ValueError("strategy required_lookbacks contains an invalid entry")
            out[key.strip()] = value
        return out

    def _strategy_uses_timeframe_aggregator(self) -> bool:
        raw = getattr(self.strategy, "uses_timeframe_aggregator", False)
        if callable(raw):
            raw = raw()
        if type(raw) is not bool:
            raise TypeError("strategy uses_timeframe_aggregator must be boolean")
        if raw:
            return True

        if "aggregator" in self._required_inputs():
            return True

        raw_timeframes = getattr(self.strategy, "required_timeframes", None)
        if callable(raw_timeframes):
            raw_timeframes = raw_timeframes()
        if raw_timeframes is not None and not isinstance(raw_timeframes, (list, tuple, set)):
            raise TypeError("strategy required_timeframes must be a sequence")
        if isinstance(raw_timeframes, (list, tuple, set)):
            if any(type(token) is not str or not token.strip() for token in raw_timeframes):
                raise ValueError("strategy required_timeframes contains an invalid token")
            return bool(raw_timeframes)
        return False

    def _ensure_timeframe_aggregator(self):
        if self.timeframe_aggregator is not None:
            return self.timeframe_aggregator
        from lumina_quant.timeframe_aggregator import TimeframeAggregator

        self.timeframe_aggregator = TimeframeAggregator(
            timeframes=self._resolve_required_timeframes(),
            lookbacks=self._resolve_required_lookbacks(),
        )
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
            raw_cadence = raw_cadence()
        if raw_cadence is None:
            cadence_seconds = 0
        elif type(raw_cadence) is not int:
            raise TypeError("strategy decision_cadence_seconds must be an integer")
        else:
            cadence_seconds = raw_cadence

        if cadence_seconds <= 0:
            return True

        event_ms = _event_time_to_ms(getattr(event, "time", None))
        if event_ms is None:
            raise ValueError("market window decision timestamp is invalid")

        cadence_ms = max(1_000, cadence_seconds * 1000)
        bucket = int(event_ms // cadence_ms)
        if self._window_decision_last_bucket == bucket:
            return False
        self._window_decision_last_bucket = bucket
        return True

    def handle_market_window_event(self, event):
        bars_1s = getattr(event, "bars_1s", {}) or {}
        total_bars = sum(len(values or ()) for values in bars_1s.values())
        self.market_events += int(total_bars if total_bars > 0 else 1)
        # Warmup is decided on the window watermark: a window whose watermark is
        # still before live_start only primes state; a straddling window already
        # decides at a live timestamp and is processed normally (its pre-live
        # bars are still excluded per-bar from open-order checks below).
        warmup = self._update_warmup_state(getattr(event, "time", None))

        aggregator = (
            self._ensure_timeframe_aggregator()
            if self._strategy_uses_timeframe_aggregator()
            else None
        )
        if aggregator is not None:
            aggregator.update_from_1s_batch(bars_1s)

        if self._should_process_market_window_event(event):
            preferred_contract = (
                str(
                    getattr(self.strategy, "preferred_contract", "market_window") or "market_window"
                )
                .strip()
                .lower()
            )
            window_fn = getattr(self.strategy, "calculate_signals_window", None)
            context_fn = getattr(self.strategy, "calculate_signals_context", None)
            available_inputs = {
                "market_window",
                "ohlcv",
                "data_handler",
                "execution_handler",
                "exchange",
            }
            if aggregator is not None:
                available_inputs.add("aggregator")
            if callable(context_fn):
                available_inputs.add("context")
            self._assert_strategy_requirements(
                available_inputs=available_inputs,
                feature_lookup=getattr(self.data_handler, "_feature_lookup", None),
            )
            if callable(context_fn) and preferred_contract == "context":
                context = StrategyInputContext(
                    event=event,
                    aggregator=aggregator,
                    feature_lookup=getattr(self.data_handler, "_feature_lookup", None),
                    data_handler=self.data_handler,
                    execution_handler=self.execution_handler,
                    exchange=getattr(self.execution_handler, "exchange", None),
                    provider_metadata={
                        "data_handler_class": self.data_handler.__class__.__name__,
                        "execution_handler_class": self.execution_handler.__class__.__name__,
                        "market_data_source": getattr(self, "market_data_source", None),
                    },
                )
                if _accepts_positional_call(context_fn, context):
                    context_fn(context)
                elif callable(window_fn) and _accepts_positional_call(window_fn, event, aggregator):
                    window_fn(event, aggregator)
                elif callable(window_fn) and _accepts_positional_call(window_fn, event):
                    window_fn(event)
                else:
                    self.strategy.calculate_signals(event)
            elif callable(window_fn) and _accepts_positional_call(window_fn, event, aggregator):
                window_fn(event, aggregator)
            elif callable(window_fn) and _accepts_positional_call(window_fn, event):
                window_fn(event)
            else:
                self.strategy.calculate_signals(event)

        if warmup:
            # Warmup windows advance strategy/aggregator state above, but emit
            # no equity rows and perform no order checks.
            return

        # Update portfolio once per decision tick.
        self.portfolio.update_timeindex(event)
        self._snapshot_fill_count = self.fills

        if hasattr(self.execution_handler, "check_open_orders"):
            # 2026-07-03 audit perf fix: with no working conditional orders the
            # per-row check is a guaranteed no-op (check_open_orders early-outs),
            # yet a MarketEvent used to be constructed for EVERY 1s row in the
            # window. Skip the whole sweep when the order book is empty; handlers
            # without an active_orders contract keep the legacy sweep.
            active_orders = getattr(self.execution_handler, "active_orders", None)
            if active_orders is not None and not active_orders:
                return
            fallback_time = getattr(event, "time", None)
            for symbol, rows in bars_1s.items():
                if not rows:
                    continue
                # Evaluate open orders (STOP / LMT / TAKE_PROFIT / TRAIL_STOP)
                # against EVERY 1s bar in the window, not just the last one.
                # Checking only rows[-1] silently skips any stop/limit level
                # touched in the other ~19s of a ~20s window, understating tail
                # risk and biasing optimizer parameter selection. This mirrors
                # handle_market_batch_event, which already checks every bar.
                for row in rows:
                    market_event = self._coerce_market_event(
                        symbol=str(symbol),
                        row=row,
                        fallback_time=fallback_time,
                    )
                    if market_event is None:
                        continue
                    # Per-bar warmup suppression: a straddling window may carry
                    # bars from before live_start — those must not trigger
                    # stop/limit evaluation.
                    if self._is_warmup_time(getattr(market_event, "time", None)):
                        continue
                    self._check_open_orders_with_equity_context(market_event)

    def handle_signal_event(self, event):
        if self._warmup_active:
            return
        self.signals += 1
        self.portfolio.update_signal(event)

    def admit_order_event(self, event) -> bool:
        """Return whether an order may reach the configured execution handler.

        Backtests retain their historical behavior.  Live engines override this
        single hook so queued and direct ``process_event(OrderEvent)`` calls
        share the same execution boundary.
        """
        return True

    def handle_order_event(self, event):
        if self._warmup_active:
            return
        if not self.admit_order_event(event):
            return
        self.orders += 1
        self.execution_handler.execute_order(event)

    def handle_fill_event(self, event):
        if self._warmup_active:
            return
        self.fills += 1
        self.portfolio.update_fill(event)
        # Hook for state saving (LiveTrader can override or we add a hook)
        self.on_fill(event)

    def reconcile_final_portfolio_snapshot(self):
        if self.fills == self._snapshot_fill_count:
            return
        reconcile = getattr(self.portfolio, "reconcile_final_snapshot", None)
        if callable(reconcile):
            reconcile()
            self._snapshot_fill_count = self.fills

    def get_engine_state(self) -> dict[str, Any]:
        """Capture engine-level state for chunk boundaries."""
        state: dict[str, Any] = {
            "event_sequencer": self._event_sequencer.get_state(),
            "window_decision_last_bucket": self._window_decision_last_bucket,
            "warmup_end_hook_fired": bool(self._warmup_end_hook_fired),
            "had_warmup": bool(self._live_start_ms is not None or self._had_warmup),
            "live_start_ms": self._live_start_ms,
        }
        if self.timeframe_aggregator is None and not self._strategy_uses_timeframe_aggregator():
            return state

        aggregator = self._ensure_timeframe_aggregator()
        if aggregator is not None:
            get_state_fn = getattr(aggregator, "get_state", None)
            if callable(get_state_fn):
                state["timeframe_aggregator"] = get_state_fn()
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

        sequencer_state = state.get("event_sequencer")
        if sequencer_state is not None:
            self._event_sequencer.set_state(sequencer_state)

        if "warmup_end_hook_fired" in state:
            self._warmup_end_hook_fired = bool(state.get("warmup_end_hook_fired"))
        if "had_warmup" in state:
            self._had_warmup = bool(state.get("had_warmup"))
        if "live_start_ms" in state:
            raw = state.get("live_start_ms")
            try:
                restored_live_start_ms = int(raw) if raw is not None else None
            except TypeError, ValueError:
                restored_live_start_ms = None
            self._live_start_ms = restored_live_start_ms
            if restored_live_start_ms is not None:
                self._had_warmup = True

        aggregator_state = state.get("timeframe_aggregator")
        if isinstance(aggregator_state, dict) and self._strategy_uses_timeframe_aggregator():
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
