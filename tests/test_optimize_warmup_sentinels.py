"""Fold-evaluator warmup wiring + sentinel-semantics tests (docs/TODO.md item 1/1a).

Contract under test (``cli/optimize.py:_execute_backtest``):
- ``BT_CHUNK_WARMUP_BARS`` is honoured end-to-end (warmup bars reach the
  strategy, orders stay suppressed before ``start_date``).
- ``InsufficientWarmupError`` propagates (never a ``-999`` sentinel).
- Infeasible parameter combinations (``ValueError``) and no-data windows keep
  the worst-score sentinel; any other exception propagates.

Per-fold policy under test (``cli/optimize.py:_run_fold_under_policy``, item 1a):
- ``InsufficientWarmupError`` skips just that fold with a loud log line.
- Unexpected exceptions drop the fold with a full traceback (run continues).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from lumina_quant.cli import optimize
from lumina_quant.optimization.walkers import InsufficientWarmupError
from lumina_quant.strategy import Strategy

SYMBOL = "BTC/USDT"
T0 = datetime(2026, 1, 1)


def _build_minute_frame(start: datetime, minutes: int) -> pl.DataFrame:
    datetimes = [start + timedelta(minutes=i) for i in range(minutes)]
    closes = [100.0 + (i % 23) * 0.4 for i in range(minutes)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": closes,
            "high": [v + 0.2 for v in closes],
            "low": [v - 0.2 for v in closes],
            "close": closes,
            "volume": [7.0] * minutes,
        }
    )


class _RecordingStrategy(Strategy):
    seen_ms: list[int] = []

    def __init__(self, bars, events):
        self.bars = bars
        self.events = events

    def calculate_signals(self, event):
        if getattr(event, "type", None) != "MARKET":
            return
        dt = event.time
        aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        type(self).seen_ms.append(int(aware.timestamp() * 1000))


class _InfeasibleParamsStrategy(Strategy):
    def __init__(self, bars, events):
        _ = bars, events
        raise ValueError("short_window must be < long_window")

    def calculate_signals(self, event):  # pragma: no cover - never constructed
        _ = event


class _BuggyStrategy(Strategy):
    def __init__(self, bars, events):
        self.bars = bars
        self.events = events

    def calculate_signals(self, event):
        _ = event
        raise RuntimeError("real engine bug — must propagate")


@pytest.fixture
def _non_parquet_mode(monkeypatch):
    monkeypatch.setattr(optimize, "PARQUET_MODE", False)
    monkeypatch.setattr(optimize, "STRATEGY_TIMEFRAME", "1m")
    monkeypatch.setattr(optimize, "BT_CHUNK_WARMUP_BARS", 0)


def test_execute_backtest_honours_chunk_warmup_bars(monkeypatch, _non_parquet_mode):
    monkeypatch.setattr(optimize, "BT_CHUNK_WARMUP_BARS", 30)
    _RecordingStrategy.seen_ms = []
    frame = _build_minute_frame(T0, minutes=180)
    start = T0 + timedelta(hours=1)
    start_ms = int(start.replace(tzinfo=UTC).timestamp() * 1000)

    result = optimize._execute_backtest(
        _RecordingStrategy,
        {},
        "data",
        [SYMBOL],
        start,
        T0 + timedelta(hours=3),
        data_dict={SYMBOL: frame},
    )

    assert "error" not in result
    warmup_seen = [ts for ts in _RecordingStrategy.seen_ms if ts < start_ms]
    assert len(warmup_seen) == 30, "warmup bars must reach the strategy"


def test_execute_backtest_propagates_insufficient_warmup(monkeypatch, _non_parquet_mode):
    monkeypatch.setattr(optimize, "BT_CHUNK_WARMUP_BARS", 30)
    frame = _build_minute_frame(T0, minutes=120)  # no context before T0

    with pytest.raises(InsufficientWarmupError):
        optimize._execute_backtest(
            _RecordingStrategy,
            {},
            "data",
            [SYMBOL],
            T0,
            T0 + timedelta(hours=2),
            data_dict={SYMBOL: frame},
        )


def test_execute_backtest_keeps_sentinel_for_infeasible_params(_non_parquet_mode):
    frame = _build_minute_frame(T0, minutes=120)

    result = optimize._execute_backtest(
        _InfeasibleParamsStrategy,
        {},
        "data",
        [SYMBOL],
        T0,
        T0 + timedelta(hours=2),
        data_dict={SYMBOL: frame},
    )

    assert result["sharpe"] == -999.0
    assert result.get("infeasible") is True
    assert "error" in result


def test_execute_backtest_keeps_sentinel_for_no_data(_non_parquet_mode):
    result = optimize._execute_backtest(
        _RecordingStrategy,
        {},
        "data",
        [SYMBOL],
        T0,
        T0 + timedelta(hours=2),
        data_dict={},
    )

    assert result["sharpe"] == -999.0
    assert result.get("no_data") is True


def test_execute_backtest_propagates_unexpected_exceptions(_non_parquet_mode):
    frame = _build_minute_frame(T0, minutes=120)

    with pytest.raises(RuntimeError, match="must propagate"):
        optimize._execute_backtest(
            _BuggyStrategy,
            {},
            "data",
            [SYMBOL],
            T0,
            T0 + timedelta(hours=2),
            data_dict={SYMBOL: frame},
        )


def test_fold_policy_skips_warmup_gap_loudly(capsys):
    def _gap(split):
        raise InsufficientWarmupError(window_bars=3, required_bars=10, fold=split["fold"])

    report = optimize._run_fold_under_policy({"fold": 2}, fold_runner=_gap)

    assert report is None
    out = capsys.readouterr().out
    assert "[Fold 2] SKIPPED" in out
    assert "warmup" in out
    assert "-999" in out  # the policy line documents the no-sentinel contract


def test_fold_policy_logs_unexpected_failures_with_traceback(capsys):
    def _bug(split):
        _ = split
        raise RuntimeError("engine bug surfaced at fold level")

    report = optimize._run_fold_under_policy({"fold": 1}, fold_runner=_bug)

    assert report is None
    captured = capsys.readouterr()
    assert "[Fold 1] FAILED" in captured.out
    assert "engine bug surfaced at fold level" in captured.err
    assert "Traceback" in captured.err


def test_fold_policy_passes_reports_through_and_flags_empty_folds(capsys):
    sentinel = {"fold": 3, "selected": {}}

    assert optimize._run_fold_under_policy({"fold": 3}, fold_runner=lambda s: sentinel) is sentinel
    assert optimize._run_fold_under_policy({"fold": 4}, fold_runner=lambda s: None) is None
    assert "[Fold 4] No valid optimization results." in capsys.readouterr().out
