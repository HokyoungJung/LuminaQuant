from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.chunked_runner import run_backtest_chunked
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.events import SignalEvent
from lumina_quant.strategy import Strategy


class _FlipStrategy(Strategy):
    def __init__(self, bars, events):
        self.bars = bars
        self.events = events
        self.symbol = next(iter(self.bars.symbol_list))
        self.position = "OUT"
        self.last_hour = None

    def get_state(self):
        return {"position": self.position, "last_hour": self.last_hour}

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        self.position = str(state.get("position", self.position))
        self.last_hour = state.get("last_hour", self.last_hour)

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        if getattr(event, "symbol", None) != self.symbol:
            return

        dt = getattr(event, "time", None)
        if not hasattr(dt, "hour"):
            return
        hour_key = (dt.year, dt.month, dt.day, dt.hour)
        if self.last_hour == hour_key:
            return
        self.last_hour = hour_key

        if self.position == "OUT":
            self.events.put(
                SignalEvent(
                    strategy_id="flip",
                    symbol=self.symbol,
                    datetime=dt,
                    signal_type="LONG",
                    strength=1.0,
                )
            )
            self.position = "LONG"
        else:
            self.events.put(
                SignalEvent(
                    strategy_id="flip",
                    symbol=self.symbol,
                    datetime=dt,
                    signal_type="EXIT",
                    strength=1.0,
                )
            )
            self.position = "OUT"


def _build_frame(start: datetime, days: int) -> pl.DataFrame:
    rows = days * 24 * 60  # 1-minute bars over N days
    datetimes = [start + timedelta(minutes=i) for i in range(rows)]
    values = [100.0 + (i * 0.01) for i in range(rows)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": values,
            "high": [v + 0.05 for v in values],
            "low": [v - 0.05 for v in values],
            "close": values,
            "volume": [10_000.0 for _ in range(rows)],
        }
    )


def test_chunked_runner_matches_full_run_with_fills(monkeypatch):
    monkeypatch.setenv("LQ_SKIP_AHEAD", "0")

    symbol = "BTC/USDT"
    start = datetime(2026, 1, 1, 0, 0, 0)
    end = datetime(2026, 1, 3, 23, 59, 0)
    frame = _build_frame(start, days=3)

    baseline = Backtest(
        csv_dir="data",
        symbol_list=[symbol],
        start_date=start,
        end_date=end,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=_FlipStrategy,
        strategy_params={},
        data_dict={symbol: frame},
        record_history=False,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe="1m",
    )
    baseline.simulate_trading(output=False)

    def _loader(chunk_start: datetime, chunk_end: datetime):
        part = frame.filter((pl.col("datetime") >= chunk_start) & (pl.col("datetime") <= chunk_end))
        return {symbol: part} if part.height > 0 else {}

    chunked = run_backtest_chunked(
        csv_dir="data",
        symbol_list=[symbol],
        start_date=start,
        end_date=end,
        strategy_cls=_FlipStrategy,
        strategy_params={},
        data_loader=_loader,
        chunk_days=1,
        strategy_timeframe="1m",
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        record_history=False,
        track_metrics=True,
        record_trades=True,
    )

    assert int(chunked.portfolio.trade_count) == int(baseline.portfolio.trade_count)
    assert abs(
        float(chunked.portfolio.current_holdings["total"])
        - float(baseline.portfolio.current_holdings["total"])
    ) <= 1e-9
