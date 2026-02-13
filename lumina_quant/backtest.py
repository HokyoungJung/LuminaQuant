import queue
import collections
from pprint import pprint
from lumina_quant.config import BacktestConfig
from lumina_quant.engine import TradingEngine


class FastQueue:
    """
    A lock-free wrapper around collections.deque for single-threaded backtests.
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
    """
    Encapsulates the settings and components for carrying out
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
    ):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.config = BacktestConfig
        self.heartbeat = 0.0
        self.start_date = start_date
        self.end_date = end_date  # Override config if specific
        self.data_dict = data_dict

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
        """
        Generates the trading instance objects from
        their class types.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self.data_handler = self.data_handler_cls(
            self.events,
            self.csv_dir,
            self.symbol_list,
            self.start_date,
            self.end_date,
            self.data_dict,
        )
        self.strategy = self.strategy_cls(
            self.bars, self.events, **self.strategy_params
        )
        self.portfolio = self.portfolio_cls(
            self.bars, self.events, self.start_date, self.config
        )
        # Pass config to execution handler for slippage/commission
        self.execution_handler = self.execution_handler_cls(
            self.events, self.bars, self.config
        )

    @property
    def bars(self):
        return self.data_handler

    def _run_backtest(self):
        """
        Executes the backtest.
        """
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
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()
        if persist_output:
            self.portfolio.output_trade_log()

        stats = self.portfolio.output_summary_stats()

        if persist_output:
            self.portfolio.save_equity_curve("equity.csv")

        if verbose:
            print("Creating summary stats...")
            print("Creating equity curve...")
            print(self.portfolio.equity_curve.tail(10).to_dict(as_series=False))
            pprint(stats)
        return stats

    def simulate_trading(self, output=True, persist_output=None, verbose=True):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        if not output:
            return None

        if persist_output is None:
            persist_output = getattr(self.config, "PERSIST_OUTPUT", True)
        return self._output_performance(
            persist_output=persist_output,
            verbose=verbose,
        )
