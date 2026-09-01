from datetime import datetime, timedelta

import polars as pl

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import FillEvent
from lumina_quant.strategy import Strategy


class _FinalBarFillStrategy(Strategy):
    def __init__(self, bars, events):
        self.events = events

    def calculate_signals(self, event):
        self.events.put(
            FillEvent(
                timeindex=event.time,
                symbol=event.symbol,
                exchange="TEST",
                quantity=1.0,
                direction="BUY",
                fill_cost=100.0,
                commission=1.0,
            )
        )


class _MetricBars:
    symbol_list = ["BTC/USDT"]

    def __init__(self, current_dt, close):
        self.current_dt = current_dt
        self.close = close

    def get_latest_bar_datetime(self, symbol):
        return self.current_dt

    def get_latest_bar_value(self, symbol, field):
        return self.close


class _MetricConfig:
    INITIAL_CAPITAL = 10_000.0
    LEVERAGE = 1


def test_final_bar_fill_is_reconciled_into_last_history_snapshot():
    start = datetime(2026, 1, 1)
    bar_time = start + timedelta(minutes=1)
    frame = pl.DataFrame(
        {
            "datetime": [bar_time],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [10.0],
        }
    )
    backtest = Backtest(
        csv_dir="data",
        symbol_list=["BTC/USDT"],
        start_date=start,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=_FinalBarFillStrategy,
        data_dict={"BTC/USDT": frame},
        strategy_timeframe="1m",
    )

    backtest.simulate_trading(output=False)

    assert backtest.events.empty()
    assert len(backtest.portfolio.all_holdings) == 2
    assert backtest.portfolio._metric_totals == [10_000.0, 9_999.0]
    final = backtest.portfolio.all_holdings[-1]
    assert final[0] == bar_time
    assert final[1:7] == (9899.0, 1.0, 0.0, 9999.0, 100.0, 100.0)

    backtest.reconcile_final_portfolio_snapshot()
    assert len(backtest.portfolio.all_holdings) == 2
    assert backtest.portfolio._metric_totals == [10_000.0, 9_999.0]


def test_off_cadence_final_metric_is_appended_once_without_history():
    start = datetime(2026, 1, 1)
    bars = _MetricBars(start, 100.0)
    portfolio = Portfolio(
        bars,
        events=None,
        start_date=start,
        config=_MetricConfig,
        record_history=False,
        sampling_timeframe="1h",
    )
    bars.current_dt = start + timedelta(hours=1)
    portfolio.update_timeindex(None)
    bars.current_dt += timedelta(minutes=30)
    bars.close = 120.0
    portfolio.current_positions["BTC/USDT"] = 1.0
    portfolio.current_holdings["cash"] = 9_900.0

    portfolio.reconcile_final_snapshot()
    portfolio.reconcile_final_snapshot()

    assert portfolio._metric_totals == [10_000.0, 10_000.0, 10_020.0]
    assert portfolio._metric_benchmarks == [0.0, 100.0, 120.0]
