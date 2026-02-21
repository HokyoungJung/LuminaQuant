import collections
import logging
import os
import queue
from pprint import pprint
from typing import Any

from lumina_quant.config import BacktestConfig
from lumina_quant.engine import TradingEngine
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
    ):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.config = BacktestConfig
        self.heartbeat = 0.0
        self.start_date = start_date
        self.end_date = end_date  # Override config if specific
        self.data_dict = data_dict
        self.record_history = bool(record_history)
        self.track_metrics = bool(track_metrics)
        self.record_trades = bool(record_trades)
        raw_strategy_timeframe = str(strategy_timeframe or getattr(self.config, "TIMEFRAME", "1m"))
        try:
            self.strategy_timeframe = normalize_timeframe_token(raw_strategy_timeframe)
        except Exception:
            self.strategy_timeframe = "1m"

        self.data_handler_cls = data_handler_cls
        self.execution_handler_cls = execution_handler_cls
        self.portfolio_cls = portfolio_cls
        self.strategy_cls = strategy_cls

        self.strategy_params = strategy_params or {}
        # Use FastQueue for Backtest (No Locks)
        self.events = FastQueue()
        self._generate_trading_instances()

        # Initialize Base Engine
        super().__init__(
            self.events,
            self.data_handler,
            self.strategy,
            self.portfolio,
            self.execution_handler,
        )

    def _generate_trading_instances(self):
        """Generates the trading instance objects from
        their class types.
        """
        LOGGER.debug("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self.data_handler = self.data_handler_cls(
            self.events,
            self.csv_dir,
            self.symbol_list,
            self.start_date,
            self.end_date,
            self.data_dict,
        )
        self.strategy = self.strategy_cls(self.bars, self.events, **self.strategy_params)
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
