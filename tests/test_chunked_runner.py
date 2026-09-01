from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl
from lumina_quant.backtesting.chunked_runner import (
    _capture_backtest_state,
    iter_chunk_windows,
    run_backtest_chunked,
)
from lumina_quant.strategy import Strategy


class _NoopStrategy(Strategy):
    def __init__(self, bars, events):
        self.bars = bars
        self.events = events

    def calculate_signals(self, event):
        _ = event
        return None


def _build_frame(start: datetime, days: int) -> pl.DataFrame:
    rows = days * 24
    datetimes = [start + timedelta(hours=i) for i in range(rows)]
    values = [100.0 + i * 0.1 for i in range(rows)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": values,
            "high": [v + 0.2 for v in values],
            "low": [v - 0.2 for v in values],
            "close": values,
            "volume": [10.0 for _ in range(rows)],
        }
    )


def test_iter_chunk_windows_respects_chunk_days():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 21)
    windows = iter_chunk_windows(start_date=start, end_date=end, chunk_days=7)

    assert len(windows) == 3
    for chunk_start, chunk_end in windows:
        assert chunk_end >= chunk_start
        assert (chunk_end - chunk_start) <= timedelta(days=7)


def test_run_backtest_chunked_calls_loader_per_chunk():
    symbol = "BTC/USDT"
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 21)
    full = _build_frame(start, days=21)

    calls: list[tuple[datetime, datetime]] = []

    def _loader(chunk_start: datetime, chunk_end: datetime):
        calls.append((chunk_start, chunk_end))
        frame = full.filter((pl.col("datetime") >= chunk_start) & (pl.col("datetime") <= chunk_end))
        return {symbol: frame} if frame.height > 0 else {}

    backtest = run_backtest_chunked(
        csv_dir="data",
        symbol_list=[symbol],
        start_date=start,
        end_date=end,
        strategy_cls=_NoopStrategy,
        strategy_params={},
        data_loader=_loader,
        chunk_days=7,
        strategy_timeframe="1m",
        record_history=False,
        track_metrics=True,
        record_trades=False,
    )

    assert len(calls) == 3
    assert int(backtest.portfolio.trade_count) == 0
    assert float(backtest.portfolio.current_holdings["total"]) > 0.0


def test_capture_state_skips_heavy_lists_when_recording_disabled():
    fake = SimpleNamespace(
        strategy=SimpleNamespace(get_state=lambda: {"s": 1}),
        portfolio=SimpleNamespace(
            get_state=lambda: {"p": 1},
            all_positions=[("p", 1)],
            all_holdings=[("h", 1)],
            trades=[{"id": 1}],
            trade_count=3,
            _metric_totals=[1.0, 2.0],
            _metric_benchmarks=[1.0, 1.1],
        ),
        execution_handler=SimpleNamespace(
            get_state=lambda: {"e": 1},
            active_orders=[{"id": "o"}],
            _order_seq=7,
        ),
    )

    carry = _capture_backtest_state(
        fake,
        record_history=False,
        track_metrics=True,
        record_trades=False,
    )

    assert "all_positions" not in carry
    assert "all_holdings" not in carry
    assert "trades" not in carry
    assert "metric_totals" in carry
    assert "metric_benchmarks" in carry
