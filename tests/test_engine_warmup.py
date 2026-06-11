"""Warmup-context end-to-end acceptance tests (docs/TODO.md item 1).

Falsifiable criteria covered here:
- With ``warmup_bars >= lookback`` the strategy's indicator state at the first
  in-window bar equals the state from a run fed the full prior history.
- Zero orders/fills/equity rows timestamped before ``start_date``.
- ``warmup_bars=0`` is byte-identical to the pre-warmup engine.
- ``InsufficientWarmupError`` is raised (not ``-999``) when requested warmup
  data is unavailable; unexpected exceptions in fold evaluation propagate.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.chunked_runner import run_backtest_chunked
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.events import SignalEvent
from lumina_quant.optimization.walkers import InsufficientWarmupError
from lumina_quant.strategy import Strategy

SYMBOL = "BTC/USDT"
T0 = datetime(2026, 1, 1)


def _to_ms(value) -> int:
    if isinstance(value, (int, float)):
        numeric = int(value)
        return numeric * 1000 if abs(numeric) < 100_000_000_000 else numeric
    dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _build_minute_frame(start: datetime, minutes: int) -> pl.DataFrame:
    datetimes = [start + timedelta(minutes=i) for i in range(minutes)]
    closes = [100.0 + (i % 37) * 0.5 - (i % 11) * 0.3 for i in range(minutes)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": closes,
            "high": [v + 0.4 for v in closes],
            "low": [v - 0.4 for v in closes],
            "close": closes,
            "volume": [10.0] * minutes,
        }
    )


def _build_second_frame(start: datetime, seconds: int) -> pl.DataFrame:
    datetimes = [start + timedelta(seconds=i) for i in range(seconds)]
    closes = [100.0 + (i % 91) * 0.05 - (i % 17) * 0.04 for i in range(seconds)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": closes,
            "high": [v + 0.02 for v in closes],
            "low": [v - 0.02 for v in closes],
            "close": closes,
            "volume": [5.0] * seconds,
        }
    )


class _ProbeSmaStrategy(Strategy):
    """Stateful incremental SMA probe that also flips LONG/EXIT for fills."""

    def __init__(self, bars, events, lookback=45, flip_minutes=10):
        self.bars = bars
        self.events = events
        self.lookback = int(lookback)
        self.flip_minutes = int(flip_minutes)
        self._closes = deque(maxlen=self.lookback)
        self.samples: list[tuple[int, int, float]] = []
        self.position = "OUT"
        self._last_flip_key = None

    def get_state(self):
        return {"position": self.position}

    def set_state(self, state):
        if isinstance(state, dict):
            self.position = str(state.get("position", self.position))

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        close = float(event.close)
        self._closes.append(close)
        sma = sum(self._closes) / len(self._closes)
        self.samples.append((_to_ms(event.time), len(self._closes), sma))

        dt = event.time
        flip_key = (dt.year, dt.month, dt.day, dt.hour, dt.minute // self.flip_minutes)
        if self._last_flip_key == flip_key:
            return
        self._last_flip_key = flip_key
        signal_type = "LONG" if self.position == "OUT" else "EXIT"
        self.position = "LONG" if self.position == "OUT" else "OUT"
        self.events.put(
            SignalEvent(
                strategy_id="probe",
                symbol=event.symbol,
                datetime=dt,
                signal_type=signal_type,
                strength=1.0,
            )
        )


class _ProbeWindowStrategy(Strategy):
    """MARKET_WINDOW probe: incremental SMA over 1s closes + LONG/EXIT flips."""

    def __init__(self, bars, events, lookback=300):
        self.bars = bars
        self.events = events
        self.lookback = int(lookback)
        self._closes = deque(maxlen=self.lookback)
        self._last_seen_ms: int | None = None
        self.samples: list[tuple[int, int, float]] = []
        self.position = "OUT"

    def calculate_signals(self, event):
        # Window consumption happens in calculate_signals_window; the legacy
        # per-row fallback path is deliberately a no-op for this probe.
        _ = event

    def calculate_signals_window(self, event, aggregator):
        _ = aggregator
        if getattr(event, "type", None) != "MARKET_WINDOW":
            return
        for rows in (getattr(event, "bars_1s", {}) or {}).values():
            for row in rows or ():
                ts_ms = int(row[0])
                if self._last_seen_ms is not None and ts_ms <= self._last_seen_ms:
                    continue
                self._last_seen_ms = ts_ms
                self._closes.append(float(row[4]))
        if not self._closes:
            return
        sma = sum(self._closes) / len(self._closes)
        self.samples.append((_to_ms(event.time), len(self._closes), sma))

        signal_type = "LONG" if self.position == "OUT" else "EXIT"
        self.position = "LONG" if self.position == "OUT" else "OUT"
        self.events.put(
            SignalEvent(
                strategy_id="probe-window",
                symbol=SYMBOL,
                datetime=event.time,
                signal_type=signal_type,
                strength=1.0,
            )
        )


def _run_backtest(
    frame: pl.DataFrame,
    *,
    start_date: datetime,
    warmup_bars: int = 0,
    strategy_cls=_ProbeSmaStrategy,
    strategy_params: dict | None = None,
    data_handler_cls=HistoricCSVDataHandler,
    data_handler_kwargs: dict | None = None,
    strategy_timeframe: str = "1m",
) -> Backtest:
    backtest = Backtest(
        csv_dir="data",
        symbol_list=[SYMBOL],
        start_date=start_date,
        end_date=None,
        data_handler_cls=data_handler_cls,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=strategy_cls,
        strategy_params=strategy_params or {},
        data_dict={SYMBOL: frame},
        data_handler_kwargs=data_handler_kwargs or {},
        record_history=True,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe=strategy_timeframe,
        warmup_bars=warmup_bars,
    )
    backtest.simulate_trading(output=False)
    return backtest


def _first_sample_at_or_after(samples, boundary_ms):
    for sample in samples:
        if sample[0] >= boundary_ms:
            return sample
    return None


def test_indicator_state_parity_with_full_history_run():
    frame = _build_minute_frame(T0, minutes=180)
    live_start = T0 + timedelta(minutes=60)
    live_start_ms = _to_ms(live_start)

    warm = _run_backtest(frame, start_date=live_start, warmup_bars=60)
    full = _run_backtest(frame, start_date=T0, warmup_bars=0)

    warm_samples = list(warm.strategy.samples)
    full_samples = list(full.strategy.samples)

    # Warmup bars were actually fed to the strategy.
    assert warm_samples and warm_samples[0][0] < live_start_ms

    first_warm = _first_sample_at_or_after(warm_samples, live_start_ms)
    first_full = _first_sample_at_or_after(full_samples, live_start_ms)
    assert first_warm is not None and first_full is not None
    # Identical bar stream => bit-identical incremental indicator state.
    assert first_warm == first_full
    # lookback=45 <= warmup_bars=60: the indicator is fully warm at bar one.
    assert first_warm[1] == 45


def test_no_orders_fills_or_equity_rows_before_start_date():
    frame = _build_minute_frame(T0, minutes=180)
    live_start = T0 + timedelta(minutes=60)
    live_start_ms = _to_ms(live_start)

    warm = _run_backtest(frame, start_date=live_start, warmup_bars=60)

    assert int(warm.fills) > 0, "expected live-region fills to prove suppression is scoped"
    for trade in warm.portfolio.trades:
        assert _to_ms(trade["datetime"]) >= live_start_ms
    for row in warm.portfolio.all_holdings:
        assert _to_ms(row[0]) >= live_start_ms
    for row in warm.portfolio.all_positions:
        assert _to_ms(row[0]) >= live_start_ms


def test_warmup_zero_is_bit_identical_to_default():
    frame = _build_minute_frame(T0, minutes=180)

    default_run = _run_backtest(frame, start_date=T0)
    explicit_zero = _run_backtest(frame, start_date=T0, warmup_bars=0)

    assert default_run.portfolio.all_holdings == explicit_zero.portfolio.all_holdings
    assert default_run.portfolio.trades == explicit_zero.portfolio.trades
    assert int(default_run.portfolio.trade_count) == int(explicit_zero.portfolio.trade_count)
    assert default_run.strategy.samples == explicit_zero.strategy.samples


def test_windowed_handler_warmup_parity_and_suppression(monkeypatch):
    # Keep both bar streams identical regardless of position state.
    monkeypatch.setenv("LQ__BACKTEST__SKIP_AHEAD_ENABLED", "0")
    frame = _build_second_frame(T0, seconds=40 * 60)
    # Live boundary intentionally NOT aligned to the 20s poll grid so one
    # MARKET_WINDOW event straddles it (per-bar suppression crux). It is also
    # mid-bucket on the 1m strategy tf, so the straddling partial bucket is NOT
    # warm context (item 1a strict count) — warmup_bars=20 therefore needs 20
    # COMPLETE 1m buckets before the boundary.
    live_start = T0 + timedelta(minutes=20, seconds=30)
    live_start_ms = _to_ms(live_start)
    handler_kwargs = {"backtest_poll_seconds": 20, "backtest_window_seconds": 20}

    warm = _run_backtest(
        frame,
        start_date=live_start,
        warmup_bars=20,
        strategy_cls=_ProbeWindowStrategy,
        data_handler_cls=HistoricParquetWindowedDataHandler,
        data_handler_kwargs=handler_kwargs,
        strategy_timeframe="1m",
    )
    full = _run_backtest(
        frame,
        start_date=T0,
        warmup_bars=0,
        strategy_cls=_ProbeWindowStrategy,
        data_handler_cls=HistoricParquetWindowedDataHandler,
        data_handler_kwargs=handler_kwargs,
        strategy_timeframe="1m",
    )

    warm_samples = list(warm.strategy.samples)
    assert warm_samples and warm_samples[0][0] < live_start_ms

    first_warm = _first_sample_at_or_after(warm_samples, live_start_ms)
    first_full = _first_sample_at_or_after(list(full.strategy.samples), live_start_ms)
    assert first_warm is not None and first_full is not None
    assert first_warm == first_full

    assert int(warm.fills) > 0
    for trade in warm.portfolio.trades:
        assert _to_ms(trade["datetime"]) >= live_start_ms
    for row in warm.portfolio.all_holdings:
        assert _to_ms(row[0]) >= live_start_ms


def test_insufficient_warmup_raises_loudly():
    frame = _build_minute_frame(T0, minutes=120)

    # No context at all before start_date.
    with pytest.raises(InsufficientWarmupError):
        _run_backtest(frame, start_date=T0, warmup_bars=10)

    # Partial context (5 bars available, 10 requested).
    with pytest.raises(InsufficientWarmupError):
        _run_backtest(frame, start_date=T0 + timedelta(minutes=5), warmup_bars=10)


def test_partial_bucket_straddling_live_start_is_not_warm_context():
    """docs/TODO.md item 1a: strict warmup-availability count.

    1m base data, 5m strategy timeframe, live_start mid-bucket at 00:13. The
    bucket [00:10, 00:15) only holds base rows BEFORE live_start, so it is a
    partial strategy-tf bar and must not count toward warmup availability.
    """
    frame = _build_minute_frame(T0, minutes=120)
    mid_bucket_start = T0 + timedelta(minutes=13)

    # Two COMPLETE 5m buckets fit before 00:13 — warmup_bars=2 is satisfiable.
    _run_backtest(
        frame,
        start_date=mid_bucket_start,
        warmup_bars=2,
        strategy_timeframe="5m",
        strategy_params={"flip_minutes": 30},
    )

    # A third bucket exists only as the partial straddling bucket — reject it.
    with pytest.raises(InsufficientWarmupError):
        _run_backtest(
            frame,
            start_date=mid_bucket_start,
            warmup_bars=3,
            strategy_timeframe="5m",
            strategy_params={"flip_minutes": 30},
        )

    # Bucket-aligned live_start keeps its exact pre-strictness semantics.
    _run_backtest(
        frame,
        start_date=T0 + timedelta(minutes=15),
        warmup_bars=3,
        strategy_timeframe="5m",
        strategy_params={"flip_minutes": 30},
    )


def test_chunked_runner_extends_first_loader_call_by_warmup():
    frame = _build_minute_frame(T0, minutes=3 * 24 * 60)
    runner_start = T0 + timedelta(hours=4)
    runner_end = T0 + timedelta(days=2, hours=23)
    calls: list[tuple[datetime, datetime]] = []

    def _loader(load_start: datetime, load_end: datetime):
        calls.append((load_start, load_end))
        sliced = frame.filter((pl.col("datetime") >= load_start) & (pl.col("datetime") <= load_end))
        return {SYMBOL: sliced} if sliced.height > 0 else {}

    backtest = run_backtest_chunked(
        csv_dir="data",
        symbol_list=[SYMBOL],
        start_date=runner_start,
        end_date=runner_end,
        strategy_cls=_ProbeSmaStrategy,
        strategy_params={},
        data_loader=_loader,
        chunk_days=1,
        strategy_timeframe="1m",
        data_handler_cls=HistoricCSVDataHandler,
        record_history=False,
        track_metrics=True,
        record_trades=False,
        warmup_bars=120,
    )

    assert len(calls) >= 2
    # First chunk's load window is extended backwards by warmup span only.
    assert calls[0][0] == runner_start - timedelta(minutes=120)
    for load_start, _ in calls[1:]:
        assert load_start > runner_start
    assert float(backtest.portfolio.current_holdings["total"]) > 0.0


def test_chunked_runner_raises_when_loader_omits_warmup_rows():
    frame = _build_minute_frame(T0, minutes=24 * 60)
    runner_start = T0 + timedelta(hours=4)
    runner_end = T0 + timedelta(hours=23)

    def _loader(load_start: datetime, load_end: datetime):
        _ = load_start  # deliberately ignore the warmup-extended request
        sliced = frame.filter(
            (pl.col("datetime") >= runner_start) & (pl.col("datetime") <= load_end)
        )
        return {SYMBOL: sliced} if sliced.height > 0 else {}

    with pytest.raises(InsufficientWarmupError):
        run_backtest_chunked(
            csv_dir="data",
            symbol_list=[SYMBOL],
            start_date=runner_start,
            end_date=runner_end,
            strategy_cls=_ProbeSmaStrategy,
            strategy_params={},
            data_loader=_loader,
            chunk_days=1,
            strategy_timeframe="1m",
            data_handler_cls=HistoricCSVDataHandler,
            record_history=False,
            track_metrics=True,
            record_trades=False,
            warmup_bars=120,
        )
