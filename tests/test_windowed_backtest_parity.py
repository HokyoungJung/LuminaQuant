from __future__ import annotations

import queue
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import SignalEvent
from lumina_quant.core.market_window_contract import normalize_bars_1s
from lumina_quant.strategy import Strategy
from lumina_quant.timeframe_aggregator import NativeBarRelease, TimeframeAggregator


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

    @staticmethod
    def _event_time_parts(ts):
        if hasattr(ts, "timestamp"):
            dt = ts
            sec = int(ts.timestamp())
            return dt, sec
        sec = int(int(ts) // 1000)
        return datetime.fromtimestamp(sec), sec

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
        dt, sec = self._event_time_parts(ts)
        if sec % int(self.decision_cadence_seconds) != 0:
            return
        self._maybe_emit(dt, bucket=sec // int(self.decision_cadence_seconds))

    def calculate_signals_window(self, event, aggregator):
        _ = aggregator.get_last_bar(self.symbol, "20s") if aggregator is not None else None
        ts = getattr(event, "time", None)
        if ts is None:
            return
        dt, sec = self._event_time_parts(ts)
        if sec % int(self.decision_cadence_seconds) != 0:
            return
        self._maybe_emit(dt, bucket=sec // int(self.decision_cadence_seconds))


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


def _normalize_buckets(values: list[int]) -> list[int]:
    if not values:
        return []
    origin = int(values[0])
    return [int(value) - origin for value in values]


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


def test_timeframe_aggregator_batched_update_matches_incremental_rows():
    start = datetime(2026, 1, 1, 0, 0, 0)
    frame = _build_1s_frame(start, seconds=125)
    rows = list(frame.iter_rows(named=False))

    batched = TimeframeAggregator(
        timeframes=["20s", "1m", "5m"], lookbacks={"20s": 16, "1m": 16, "5m": 16}
    )
    incremental = TimeframeAggregator(
        timeframes=["20s", "1m", "5m"], lookbacks={"20s": 16, "1m": 16, "5m": 16}
    )

    batched.update_from_1s_batch({"BTC/USDT": tuple(rows[:80])})
    batched.update_from_1s_batch({"BTC/USDT": tuple(rows[60:])})
    for row in rows:
        incremental.update_from_1s_batch({"BTC/USDT": (row,)})

    for timeframe in ("1s", "20s", "1m", "5m"):
        assert batched.get_bars("BTC/USDT", timeframe, n=16) == incremental.get_bars(
            "BTC/USDT",
            timeframe,
            n=16,
        )


def test_timeframe_aggregator_canonical_epoch_rows_match_datetime_rows():
    start = datetime(2026, 1, 1, 0, 0, 0)
    frame = _build_1s_frame(start, seconds=125)
    datetime_rows = list(frame.iter_rows(named=False))
    canonical_rows = [
        (
            int(row[0].replace(tzinfo=UTC).timestamp() * 1000),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
        for row in datetime_rows
    ]

    datetime_aggregator = TimeframeAggregator(
        timeframes=["20s", "1m", "5m"],
        lookbacks={"20s": 16, "1m": 16, "5m": 16},
    )
    canonical_aggregator = TimeframeAggregator(
        timeframes=["20s", "1m", "5m"],
        lookbacks={"20s": 16, "1m": 16, "5m": 16},
    )

    # Use overlapping batches to exercise the dedupe path as it appears in
    # MARKET_WINDOW replay.
    normalized_datetime_rows = normalize_bars_1s({"BTC/USDT": tuple(datetime_rows)})["BTC/USDT"]
    datetime_aggregator.update_from_1s_batch({"BTC/USDT": tuple(normalized_datetime_rows[:80])})
    datetime_aggregator.update_from_1s_batch({"BTC/USDT": tuple(normalized_datetime_rows[60:])})
    canonical_aggregator.update_from_1s_batch({"BTC/USDT": tuple(canonical_rows[:80])})
    canonical_aggregator.update_from_1s_batch({"BTC/USDT": tuple(canonical_rows[60:])})

    for timeframe in ("1s", "20s", "1m", "5m"):
        assert canonical_aggregator.get_bars(
            "BTC/USDT", timeframe, n=16
        ) == datetime_aggregator.get_bars(
            "BTC/USDT",
            timeframe,
            n=16,
        )


def test_exact_native_releases_match_one_second_left_fold() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        (
            int((start + timedelta(seconds=index)).timestamp() * 1000),
            100.0,
            101.0,
            99.0,
            100.0,
            1e16 if index == 0 else 1.0,
        )
        for index in range((8 * 60 * 60) + 1)
    ]
    exact = TimeframeAggregator(timeframes=["4h", "1d"])
    releases = exact.update_from_canonical_1s_rows_exact("BTCUSDT", rows)

    reference = TimeframeAggregator(timeframes=["4h", "1d"])
    for row in rows:
        reference.update_from_1s_batch({"BTCUSDT": (row,)})

    assert exact.get_state() == reference.get_state()
    assert releases == (
        NativeBarRelease(
            release_timestamp_ms=rows[14_400][0],
            symbol="BTCUSDT",
            timeframe="4h",
            bar=reference.get_bars("BTCUSDT", "4h", n=3)[0],
        ),
        NativeBarRelease(
            release_timestamp_ms=rows[28_800][0],
            symbol="BTCUSDT",
            timeframe="4h",
            bar=reference.get_bars("BTCUSDT", "4h", n=3)[1],
        ),
    )


def test_alpha_max_columnar_advance_matches_ordinary_handler_tails() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(seconds=100)
    data = {
        symbol: _build_1s_frame(
            start,
            100,
            offset=float(index),
        )
        for index, symbol in enumerate(("BTC/USDT", "ETH/USDT"))
    }
    reference_events = queue.Queue()
    reference = HistoricParquetWindowedDataHandler(
        reference_events,
        ".",
        ["BTC/USDT", "ETH/USDT"],
        start,
        end,
        data,
        backtest_poll_seconds=1,
        backtest_window_seconds=1,
        market_window_parity_v2_enabled=True,
    )
    while reference.continue_backtest:
        reference.update_bars()
        while not reference_events.empty():
            reference_events.get_nowait()

    batched = HistoricParquetWindowedDataHandler(
        queue.Queue(),
        ".",
        ["BTC/USDT", "ETH/USDT"],
        start,
        end,
        data,
        backtest_poll_seconds=1,
        backtest_window_seconds=1,
        market_window_parity_v2_enabled=True,
    )
    view = batched.alpha_max_exact_columnar_view()
    assert tuple(view) == ("BTC/USDT", "ETH/USDT")
    assert np.array_equal(view["BTC/USDT"][0], view["ETH/USDT"][0])
    batched.alpha_max_advance_without_event(0, 49)
    batched.alpha_max_advance_without_event(50, 99)

    assert batched.symbol_index == reference.symbol_index
    assert batched.next_bar == reference.next_bar
    assert batched.continue_backtest is reference.continue_backtest is False
    assert batched.last_emitted_timestamp_ms == reference.last_emitted_timestamp_ms
    assert {symbol: tuple(rows) for symbol, rows in batched._window_rows.items()} == {
        symbol: tuple(rows) for symbol, rows in reference._window_rows.items()
    }
    assert {symbol: tuple(rows) for symbol, rows in batched.latest_symbol_data.items()} == {
        symbol: tuple(rows) for symbol, rows in reference.latest_symbol_data.items()
    }


def test_market_window_normalize_canonical_epoch_rows_match_datetime_rows():
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = list(_build_1s_frame(start, seconds=21).iter_rows(named=False))
    canonical_rows = [
        (
            int(row[0].replace(tzinfo=UTC).timestamp() * 1000),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
        for row in rows
    ]

    assert normalize_bars_1s({"BTC/USDT": tuple(canonical_rows)}) == normalize_bars_1s(
        {"BTC/USDT": tuple(rows)}
    )


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

    assert _normalize_buckets(list(windowed.strategy.decisions)) == _normalize_buckets(
        list(baseline.strategy.decisions)
    )
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
