from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import SignalEvent
from lumina_quant.strategy import Strategy
from lumina_quant.timeframe_aggregator import TimeframeAggregator


class _CadenceFlipStrategy(Strategy):
    decision_cadence_seconds = 20
    required_timeframes = ("20s", "1m")
    required_lookbacks = {"20s": 8, "1m": 8}

    def __init__(self, bars, events):
        self.bars = bars
        self.events = events
        self.symbol = next(iter(self.bars.symbol_list))
        self.position = "OUT"
        self._last_bucket = None
        self.decisions = []

    def _maybe_emit(self, dt, bucket):
        if self._last_bucket == bucket:
            return
        self._last_bucket = bucket
        self.decisions.append(int(bucket))
        signal_type = "LONG" if self.position == "OUT" else "EXIT"
        self.position = "LONG" if signal_type == "LONG" else "OUT"
        self.events.put(
            SignalEvent(
                strategy_id="cadence_flip",
                symbol=self.symbol,
                datetime=dt,
                signal_type=signal_type,
                strength=1.0,
            )
        )

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        if getattr(event, "symbol", None) != self.symbol:
            return
        ts = getattr(event, "time", None)
        if ts is None:
            return
        sec = int(ts.timestamp())
        if sec % int(self.decision_cadence_seconds) != 0:
            return
        self._maybe_emit(ts, bucket=sec // int(self.decision_cadence_seconds))

    def calculate_signals_window(self, event, aggregator):
        _ = aggregator.get_last_bar(self.symbol, "20s") if aggregator is not None else None
        ts = getattr(event, "time", None)
        if ts is None:
            return
        sec = int(ts.timestamp())
        if sec % int(self.decision_cadence_seconds) != 0:
            return
        self._maybe_emit(ts, bucket=sec // int(self.decision_cadence_seconds))


class _WindowNoopStrategy(Strategy):
    decision_cadence_seconds = 20

    def __init__(self, bars, events):
        self.bars = bars
        self.events = events

    def calculate_signals(self, event):
        _ = event
        return None


def _build_1s_frame(start: datetime, seconds: int, offset: float = 0.0) -> pl.DataFrame:
    datetimes = [start + timedelta(seconds=i) for i in range(seconds)]
    base = [100.0 + offset + (i * 0.01) for i in range(seconds)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": base,
            "high": [v + 0.05 for v in base],
            "low": [v - 0.05 for v in base],
            "close": base,
            "volume": [100.0 for _ in range(seconds)],
        }
    )


def test_timeframe_aggregator_correctness_with_overlapping_windows():
    start = datetime(2026, 1, 1, 0, 0, 0)
    frame = _build_1s_frame(start, seconds=65)
    rows = list(frame.iter_rows(named=False))

    aggregator = TimeframeAggregator(timeframes=["20s", "1m"], lookbacks={"20s": 8, "1m": 8})
    aggregator.update_from_1s_batch({"BTC/USDT": tuple(rows[:40])})
    # Overlapping update must be deduplicated internally.
    aggregator.update_from_1s_batch({"BTC/USDT": tuple(rows[20:])})

    bars_20s = aggregator.get_bars("BTC/USDT", "20s", n=4)
    assert len(bars_20s) == 4
    assert float(bars_20s[0][1]) == float(rows[0][1])
    assert float(bars_20s[-1][4]) == float(rows[-1][4])

    bars_1m = aggregator.get_bars("BTC/USDT", "1m", n=2)
    assert len(bars_1m) == 2
    assert float(bars_1m[0][1]) == float(rows[0][1])
    assert float(bars_1m[0][4]) == float(rows[59][4])
    assert float(bars_1m[-1][4]) == float(rows[-1][4])


def test_windowed_mode_matches_legacy_cadence_behavior(monkeypatch):
    monkeypatch.setenv("LQ__BACKTEST__SKIP_AHEAD_ENABLED", "0")

    symbol = "BTC/USDT"
    start = datetime(2026, 1, 1, 0, 0, 0)
    seconds = 1_180
    frame = _build_1s_frame(start, seconds=seconds)

    baseline = Backtest(
        csv_dir="data",
        symbol_list=[symbol],
        start_date=start,
        end_date=start + timedelta(seconds=seconds - 1),
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=_CadenceFlipStrategy,
        strategy_params={},
        data_dict={symbol: frame},
        record_history=False,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe="1s",
    )
    baseline.simulate_trading(output=False)

    windowed = Backtest(
        csv_dir="data",
        symbol_list=[symbol],
        start_date=start,
        end_date=start + timedelta(seconds=seconds - 1),
        data_handler_cls=HistoricParquetWindowedDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=_CadenceFlipStrategy,
        strategy_params={},
        data_dict={symbol: frame},
        data_handler_kwargs={"backtest_poll_seconds": 20, "backtest_window_seconds": 300},
        record_history=False,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe="1s",
    )
    windowed.simulate_trading(output=False)

    assert list(windowed.strategy.decisions) == list(baseline.strategy.decisions)
    assert int(windowed.signals) == int(baseline.signals)
    assert abs(int(windowed.portfolio.trade_count) - int(baseline.portfolio.trade_count)) <= 1


def test_windowed_skip_ahead_keeps_results_and_reduces_ticks(monkeypatch):
    symbol = "BTC/USDT"
    start = datetime(2026, 1, 1, 0, 0, 0)
    seconds = 7_200
    frame = _build_1s_frame(start, seconds=seconds)

    def _run(skip_enabled: bool):
        monkeypatch.setenv("LQ__BACKTEST__SKIP_AHEAD_ENABLED", "1" if skip_enabled else "0")
        backtest = Backtest(
            csv_dir="data",
            symbol_list=[symbol],
            start_date=start,
            end_date=start + timedelta(seconds=seconds - 1),
            data_handler_cls=HistoricParquetWindowedDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=_WindowNoopStrategy,
            strategy_params={},
            data_dict={symbol: frame},
            data_handler_kwargs={"backtest_poll_seconds": 20, "backtest_window_seconds": 120},
            record_history=False,
            track_metrics=True,
            record_trades=False,
            strategy_timeframe="1m",
        )
        backtest.simulate_trading(output=False)
        return backtest

    baseline = _run(skip_enabled=False)
    optimized = _run(skip_enabled=True)

    assert float(optimized.portfolio.current_holdings["total"]) == float(
        baseline.portfolio.current_holdings["total"]
    )
    assert int(optimized.portfolio.trade_count) == int(baseline.portfolio.trade_count)
    assert optimized.skip_ahead_jumps > 0
    assert optimized.skip_ahead_rows_skipped > 0
    assert optimized.market_events < baseline.market_events
