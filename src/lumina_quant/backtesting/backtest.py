"""Event-driven backtest engine (Phase 4).

Bar engine pipeline:
    data_handler  →  events queue  →  strategy / portfolio / execution_handler
    BacktestConfigView (permanent engine-private adapter, backtesting/_config_view.py)
    carries uppercase-attr config to Portfolio and ExecutionHandler.  Pass a plain
    ``RuntimeConfig`` and this module wraps it automatically.

Compute backends (Phase 2):
    Metric evaluation routes through ``optimization.native_backend`` →
    ``lumina_quant._compute.evaluate_metrics`` (pyo3).  The bar engine itself
    processes OHLCV tuples in Python; heavy numerical work stays in pyo3 kernels.

Phase 4 execution model:
    ``SimulatedExecutionHandler`` delegates all fill / funding / liquidation logic to
    ``backtesting.execution_model.ExecutionModel``.
"""

from __future__ import annotations

import collections
import logging
import os
import queue
from datetime import UTC, timedelta
from pprint import pprint
from typing import Any

from lumina_quant.backtesting._config_view import BacktestConfigView
from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.core.engine import TradingEngine
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds

LOGGER = logging.getLogger(__name__)


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
            return None
        except Exception:
            return None
    return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


class TimeframeGatedStrategy:
    """Gate strategy signal evaluations to aligned timeframe boundaries."""

    def __init__(self, strategy, timeframe):
        self._strategy = strategy
        self._timeframe = normalize_timeframe_token(timeframe)
        self._timeframe_ms = max(1, int(timeframe_to_milliseconds(self._timeframe)))
        self._last_bucket_per_symbol: dict[str, int] = {}

    def __getattr__(self, name):
        return getattr(self._strategy, name)

    def should_process_market_event(self, event):
        if getattr(event, "type", None) != "MARKET":
            return True
        event_ms = _event_time_to_ms(getattr(event, "time", None))
        if event_ms is None:
            return True
        if event_ms % self._timeframe_ms != 0:
            return False

        symbol = str(getattr(event, "symbol", "") or "")
        if not symbol:
            return True
        bucket = event_ms // self._timeframe_ms
        previous = self._last_bucket_per_symbol.get(symbol)
        if previous == bucket:
            return False
        self._last_bucket_per_symbol[symbol] = bucket
        return True

    def calculate_signals(self, event):
        self._strategy.calculate_signals(event)

    def get_state(self):
        state: dict[Any, Any] = {}
        get_state_fn = getattr(self._strategy, "get_state", None)
        if callable(get_state_fn):
            raw_state = get_state_fn()
            if isinstance(raw_state, dict):
                state.update(raw_state)
        state["_tf_gate_last_bucket"] = dict(self._last_bucket_per_symbol)
        return state

    def set_state(self, state):
        if isinstance(state, dict):
            raw = state.get("_tf_gate_last_bucket")
            if isinstance(raw, dict):
                self._last_bucket_per_symbol = {k: int(v) for k, v in raw.items() if v is not None}
        set_state_fn = getattr(self._strategy, "set_state", None)
        if callable(set_state_fn):
            set_state_fn(state)


class FastQueue:
    """A lock-free wrapper around collections.deque for single-threaded backtests.
    Speeds up event loops by avoiding thread locking overhead (~30% faster).
    """

    def __init__(self):
        self._deque = collections.deque()

    def put(self, item, block=True, timeout=None):
        self._deque.append(item)

    def get(self, block=True, timeout=None):
        try:
            return self._deque.popleft()
        except IndexError:
            raise queue.Empty

    def qsize(self):
        return len(self._deque)

    def empty(self):
        return len(self._deque) == 0


class Backtest(TradingEngine):
    """Encapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
        self,
        csv_dir,
        symbol_list,
        start_date,
        data_handler_cls,
        execution_handler_cls,
        portfolio_cls,
        strategy_cls,
        strategy_params=None,
        end_date=None,
        data_dict=None,
        record_history=True,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe=None,
        data_handler_kwargs=None,
        config=None,
        warmup_bars=0,
    ):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        # Accept RuntimeConfig directly — wrap in BacktestConfigView so all
        # downstream getattr(config, "UPPER_ATTR") consumers work unchanged.
        if config is None:
            self.config: Any = BacktestConfigView(get_default_runtime_config())
        elif isinstance(config, BacktestConfigView):
            self.config = config
        elif hasattr(config, "execution") and hasattr(config, "backtest"):
            # Looks like a RuntimeConfig — wrap it.
            self.config = BacktestConfigView(config)
        else:
            # Duck-typed config bag (legacy or test-stub) — pass through.
            self.config = config
        self.heartbeat = 0.0
        self.start_date = start_date
        self.end_date = end_date  # Override config if specific
        self.data_dict = data_dict
        self.record_history = bool(record_history)
        self.track_metrics = bool(track_metrics)
        self.record_trades = bool(record_trades)
        self.data_handler_kwargs = dict(data_handler_kwargs or {})
        raw_strategy_timeframe = str(strategy_timeframe or getattr(self.config, "TIMEFRAME", "1m"))
        try:
            self.strategy_timeframe = normalize_timeframe_token(raw_strategy_timeframe)
        except Exception:
            self.strategy_timeframe = "1m"
        try:
            self._strategy_timeframe_ms = int(timeframe_to_milliseconds(self.strategy_timeframe))
        except Exception:
            self._strategy_timeframe_ms = 60_000
        skip_env = os.getenv("LQ__BACKTEST__SKIP_AHEAD_ENABLED")
        default_skip_ahead = bool(getattr(self.config, "SKIP_AHEAD_ENABLED", True))
        self._skip_ahead_enabled = _as_bool(
            default_skip_ahead if skip_env is None else skip_env,
            default_skip_ahead,
        )
        self.skip_ahead_jumps = 0
        self.skip_ahead_rows_skipped = 0

        # Warmup context (docs/TODO.md item 1): `start_date` stays the trading
        # boundary (live_start); the data handler loads from
        # `start_date - warmup_bars x strategy_timeframe` so stateful indicators
        # are warm at the first in-window bar. warmup_bars=0 keeps every code
        # path bit-identical to the pre-warmup engine.
        self.warmup_bars = max(0, int(warmup_bars or 0))
        self._live_start_ms: int | None = None
        self._effective_load_start = start_date
        if self.warmup_bars > 0:
            live_start_dt = OHLCVFrameLoader._coerce_bound(start_date)
            if live_start_dt is None:
                raise ValueError(
                    f"warmup_bars={self.warmup_bars} requires a parseable start_date, "
                    f"got {start_date!r}"
                )
            # Naive datetimes are treated as UTC throughout the data layer.
            self._live_start_ms = int(live_start_dt.replace(tzinfo=UTC).timestamp() * 1000)
            warmup_ms = self.warmup_bars * max(1, int(self._strategy_timeframe_ms))
            self._effective_load_start = live_start_dt - timedelta(milliseconds=warmup_ms)

        self.data_handler_cls = data_handler_cls
        self.execution_handler_cls = execution_handler_cls
        self.portfolio_cls = portfolio_cls
        self.strategy_cls = strategy_cls

        self.strategy_params = strategy_params or {}
        # Use FastQueue for Backtest (No Locks)
        self.events = FastQueue()
        self._generate_trading_instances()
        if self.warmup_bars > 0:
            self._assert_warmup_available()

        # Initialize Base Engine
        super().__init__(
            self.events,
            self.data_handler,
            self.strategy,
            self.portfolio,
            self.execution_handler,
            live_start_ms=self._live_start_ms,
        )

    def _generate_trading_instances(self):
        """Generates the trading instance objects from
        their class types.
        """
        LOGGER.debug("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        try:
            self.data_handler = self.data_handler_cls(
                self.events,
                self.csv_dir,
                self.symbol_list,
                self._effective_load_start,
                self.end_date,
                self.data_dict,
                **self.data_handler_kwargs,
            )
        except TypeError:
            self.data_handler = self.data_handler_cls(
                self.events,
                self.csv_dir,
                self.symbol_list,
                self._effective_load_start,
                self.end_date,
                self.data_dict,
            )
        self.strategy = self.strategy_cls(self.bars, self.events, **self.strategy_params)
        configured_cadence = int(getattr(self.config, "DECISION_CADENCE_SECONDS", 20))
        raw_cadence = os.getenv("LQ__BACKTEST__DECISION_CADENCE_SECONDS", "").strip()
        if raw_cadence:
            try:
                configured_cadence = max(1, int(raw_cadence))
            except Exception:
                configured_cadence = max(1, configured_cadence)
        else:
            configured_cadence = max(1, configured_cadence)
        try:
            if int(getattr(self.strategy, "decision_cadence_seconds", 0) or 0) <= 0:
                self.strategy.decision_cadence_seconds = int(configured_cadence)
        except Exception:
            pass
        if self.strategy_timeframe != "1s":
            self.strategy = TimeframeGatedStrategy(self.strategy, self.strategy_timeframe)
        try:
            self.portfolio = self.portfolio_cls(
                self.bars,
                self.events,
                self.start_date,
                self.config,
                record_history=self.record_history,
                track_metrics=self.track_metrics,
                record_trades=self.record_trades,
                sampling_timeframe=self.strategy_timeframe,
            )
        except TypeError:
            try:
                self.portfolio = self.portfolio_cls(
                    self.bars,
                    self.events,
                    self.start_date,
                    self.config,
                    record_history=self.record_history,
                    record_trades=self.record_trades,
                )
            except TypeError:
                self.portfolio = self.portfolio_cls(
                    self.bars,
                    self.events,
                    self.start_date,
                    self.config,
                )
        # Pass config to execution handler for slippage/commission
        self.execution_handler = self.execution_handler_cls(self.events, self.bars, self.config)

    @property
    def bars(self):
        return self.data_handler

    def _assert_warmup_available(self) -> None:
        """Raise ``InsufficientWarmupError`` when requested warmup data is missing.

        Counts distinct strategy-timeframe buckets before ``start_date`` per
        symbol; each loaded symbol must provide at least ``warmup_bars`` of
        context. Symbols with no rows at all keep the legacy no-data path.
        """
        if self._live_start_ms is None:
            return
        from lumina_quant.optimization.walkers import InsufficientWarmupError

        tf_ms = max(1, int(self._strategy_timeframe_ms))
        required = int(self.warmup_bars)
        live_start_ms = int(self._live_start_ms)
        timestamps_by_symbol = getattr(self.data_handler, "symbol_timestamps_ms", None)
        if not isinstance(timestamps_by_symbol, dict):
            return
        for ts_list in timestamps_by_symbol.values():
            buckets: set[int] = set()
            for ts in ts_list or ():
                if ts is None:
                    continue
                if int(ts) >= live_start_ms:
                    break  # timestamps are sorted ascending
                buckets.add(int(ts) // tf_ms)
            if len(buckets) < required:
                raise InsufficientWarmupError(
                    window_bars=len(buckets),
                    required_bars=required,
                )

    def _supports_skip_ahead(self) -> bool:
        if not bool(self._skip_ahead_enabled):
            return False
        step_ms = getattr(self.data_handler, "skip_ahead_step_ms", None)
        if step_ms is not None:
            try:
                if int(step_ms) <= 1_000:
                    return False
            except Exception:
                return False
        elif int(self._strategy_timeframe_ms) <= 1_000:
            return False
        return callable(getattr(self.data_handler, "skip_to_timestamp_ms", None))

    def _can_skip_ahead_now(self) -> bool:
        if not self._supports_skip_ahead():
            return False
        # Never skip inside the warmup region: warmup bars exist solely to feed
        # indicator state, and skip-ahead would drop them (positions are always
        # flat during warmup, so the flat-position fast path would fire).
        if self._live_start_ms is not None:
            current_ms = getattr(self.data_handler, "last_emitted_timestamp_ms", None)
            if current_ms is None or int(current_ms) < int(self._live_start_ms):
                return False
        if bool(getattr(self.strategy, "requires_intrabar_checks", False)):
            return False
        active_orders = getattr(self.execution_handler, "active_orders", None)
        if active_orders:
            return False
        try:
            positions = getattr(self.portfolio, "current_positions", {}) or {}
            return all(abs(float(qty)) < 1e-12 for qty in positions.values())
        except Exception:
            return False

    def _apply_skip_ahead(self) -> None:
        if not self._can_skip_ahead_now():
            return
        current_ms = getattr(self.data_handler, "last_emitted_timestamp_ms", None)
        if current_ms is None:
            return
        raw_step = getattr(self.data_handler, "skip_ahead_step_ms", None)
        if raw_step is None:
            raw_step = int(self._strategy_timeframe_ms)
        try:
            step = max(1_000, int(raw_step))
        except Exception:
            step = max(1_000, int(self._strategy_timeframe_ms))
        target_ms = ((int(current_ms) // step) + 1) * step
        skipped = int(self.data_handler.skip_to_timestamp_ms(target_ms))
        if skipped > 0:
            self.skip_ahead_jumps += 1
            self.skip_ahead_rows_skipped += skipped

    def _run_backtest(self):
        """Executes the backtest."""
        i = 0
        while True:
            i += 1
            # Update the market bars
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    self.process_event(event)

            self._apply_skip_ahead()

            # time.sleep(self.heartbeat)

    def _output_performance(self, persist_output=True, verbose=True):
        """Outputs the strategy performance from the backtest."""
        self.portfolio.create_equity_curve_dataframe()
        if persist_output:
            self.portfolio.output_trade_log(os.path.join("data", "trades.csv"))

        stats = self.portfolio.output_summary_stats()

        if persist_output:
            self.portfolio.save_equity_curve(os.path.join("data", "equity.csv"))

        if verbose:
            print("Creating summary stats...")
            print("Creating equity curve...")
            print(self.portfolio.equity_curve.tail(10).to_dict(as_series=False))
            pprint(stats)
        return stats

    def simulate_trading(self, output=True, persist_output=None, verbose=True):
        """Simulates the backtest and outputs portfolio performance."""
        self._run_backtest()
        if not output:
            return None

        if persist_output is None:
            persist_output = getattr(self.config, "PERSIST_OUTPUT", True)
        return self._output_performance(
            persist_output=persist_output,
            verbose=verbose,
        )


__all__ = ["Backtest", "FastQueue", "TimeframeGatedStrategy"]
