"""Benchmark skip-ahead vs full stepping on synthetic 1s OHLCV data."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from time import perf_counter

import polars as pl

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.strategy import Strategy


class NoopStrategy(Strategy):
    def __init__(self, bars, events):
        self.bars = bars
        self.events = events

    def calculate_signals(self, event):
        _ = event
        return None


def build_symbol_frame(start: datetime, seconds: int, offset: float) -> pl.DataFrame:
    datetimes = [start + timedelta(seconds=i) for i in range(seconds)]
    values = [100.0 + offset + (i * 0.001) for i in range(seconds)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": values,
            "high": [v + 0.1 for v in values],
            "low": [v - 0.1 for v in values],
            "close": values,
            "volume": [1000.0 for _ in range(seconds)],
        }
    )


def run_once(skip_ahead: bool, data_dict: dict[str, pl.DataFrame], start: datetime, end: datetime):
    os.environ["LQ_SKIP_AHEAD"] = "1" if skip_ahead else "0"
    bt = Backtest(
        csv_dir="data",
        symbol_list=list(data_dict.keys()),
        start_date=start,
        end_date=end,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=NoopStrategy,
        strategy_params={},
        data_dict=data_dict,
        record_history=False,
        track_metrics=True,
        record_trades=False,
        strategy_timeframe="1m",
    )
    t0 = perf_counter()
    bt.simulate_trading(output=False)
    elapsed = perf_counter() - t0
    return bt, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark skip-ahead speed")
    parser.add_argument("--symbols", type=int, default=12, help="Number of synthetic symbols")
    parser.add_argument("--hours", type=int, default=24, help="Simulation hours at 1-second bars")
    args = parser.parse_args()

    start = datetime(2026, 1, 1, 0, 0, 0)
    seconds = max(1, int(args.hours)) * 3600
    end = start + timedelta(seconds=seconds - 1)

    data_dict: dict[str, pl.DataFrame] = {}
    for idx in range(max(1, int(args.symbols))):
        symbol = f"SYM{idx:02d}/USDT"
        data_dict[symbol] = build_symbol_frame(start, seconds, offset=float(idx) * 10.0)

    baseline, baseline_elapsed = run_once(False, data_dict, start, end)
    optimized, optimized_elapsed = run_once(True, data_dict, start, end)

    speedup = baseline_elapsed / max(optimized_elapsed, 1e-9)
    print("=== Skip-ahead Benchmark ===")
    print(f"symbols={len(data_dict)} seconds={seconds}")
    print(
        f"baseline: elapsed={baseline_elapsed:.4f}s market_events={baseline.market_events} "
        f"skip_jumps={baseline.skip_ahead_jumps} skipped_rows={baseline.skip_ahead_rows_skipped}"
    )
    print(
        f"skip-ahead: elapsed={optimized_elapsed:.4f}s market_events={optimized.market_events} "
        f"skip_jumps={optimized.skip_ahead_jumps} skipped_rows={optimized.skip_ahead_rows_skipped}"
    )
    print(f"speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
