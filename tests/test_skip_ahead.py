from __future__ import annotations

from datetime import datetime, timedelta
from time import perf_counter

import polars as pl
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.strategy import Strategy


class _NoopStrategy(Strategy):
    def __init__(self, bars, events):
        self.bars = bars
        self.events = events

    def calculate_signals(self, event):
        _ = event
        return None


def _frame(start: datetime, seconds: int, offset: float) -> pl.DataFrame:
    datetimes = [start + timedelta(seconds=i) for i in range(seconds)]
    base = [100.0 + offset + (i * 0.001) for i in range(seconds)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": base,
            "high": [v + 0.05 for v in base],
            "low": [v - 0.05 for v in base],
            "close": base,
            "volume": [1000.0 for _ in range(seconds)],
        }
    )


def _run(skip_enabled: bool, data_dict: dict[str, pl.DataFrame], monkeypatch):
    monkeypatch.setenv("LQ__BACKTEST__SKIP_AHEAD_ENABLED", "1" if skip_enabled else "0")
    backtest = Backtest(
        csv_dir="data",
        symbol_list=list(data_dict.keys()),
        start_date=datetime(2026, 1, 1, 0, 0, 0),
        end_date=datetime(2026, 1, 1, 2, 0, 0),
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=_NoopStrategy,
        strategy_params={},
        data_dict=data_dict,
        record_history=False,
        track_metrics=True,
        record_trades=False,
        strategy_timeframe="1m",
    )
    start = perf_counter()
    backtest.simulate_trading(output=False)
    return backtest, perf_counter() - start


def test_skip_ahead_matches_baseline_and_reduces_work(monkeypatch):
    start = datetime(2026, 1, 1, 0, 0, 0)
    seconds = 7_200  # 2 hours of 1-second bars
    data_dict = {
        "BTC/USDT": _frame(start, seconds, 0.0),
        "ETH/USDT": _frame(start, seconds, 10.0),
        "BNB/USDT": _frame(start, seconds, 20.0),
    }

    baseline, baseline_elapsed = _run(False, data_dict, monkeypatch)
    optimized, optimized_elapsed = _run(True, data_dict, monkeypatch)

    assert float(optimized.portfolio.current_holdings["total"]) == float(
        baseline.portfolio.current_holdings["total"]
    )
    assert int(optimized.portfolio.trade_count) == int(baseline.portfolio.trade_count)
    assert all(
        abs(float(optimized.portfolio.current_positions[s]))
        == abs(float(baseline.portfolio.current_positions[s]))
        for s in data_dict
    )

    assert optimized.skip_ahead_jumps > 0
    assert optimized.skip_ahead_rows_skipped > 0
    assert optimized.market_events < baseline.market_events

    # Performance check (allow slack for noisy CI environments).
    if baseline_elapsed > 0.01:
        assert optimized_elapsed <= baseline_elapsed * 0.95
