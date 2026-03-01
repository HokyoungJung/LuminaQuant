from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_OPTIMIZE_PATH = Path(__file__).resolve().parents[1] / "optimize.py"
sys.path.insert(0, str(_OPTIMIZE_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("optimize_module", _OPTIMIZE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
optimize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(optimize)


def test_resolve_topk_bounds():
    topk = optimize._resolve_topk(100)
    assert 1 <= topk <= 100
    assert topk >= optimize.TWO_STAGE_MIN_TOPK
    assert topk <= optimize.TWO_STAGE_MAX_TOPK


def test_resolve_prefilter_window_bounds():
    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 1)
    fast_start, fast_end = optimize._resolve_prefilter_window(start, end)

    assert fast_start == start
    assert start < fast_end <= end


def test_recent_validation_split_anchors_before_oos():
    data_start = datetime(2025, 1, 1)
    oos_start = datetime(2026, 2, 1)
    oos_end = datetime(2026, 3, 1)
    in_sample_end = datetime(2026, 1, 31, 23, 59)

    split = optimize._build_recent_validation_split(
        data_start,
        in_sample_end=in_sample_end,
        oos_start=oos_start,
        oos_end=oos_end,
        validation_days=30,
        timeframe="1m",
    )

    assert split is not None
    assert split["val_end"] == in_sample_end
    assert split["test_start"] == oos_start
    assert split["test_end"] == oos_end
    assert (oos_start - split["val_start"]).days == 30


def test_recent_validation_split_can_be_disabled():
    split = optimize._build_recent_validation_split(
        datetime(2025, 1, 1),
        in_sample_end=datetime(2026, 1, 31),
        oos_start=datetime(2026, 2, 1),
        oos_end=datetime(2026, 3, 1),
        validation_days=0,
        timeframe="1m",
    )
    assert split is None


def test_optimize_uses_windowed_handler_in_parquet_mode(monkeypatch):
    captured: dict[str, object] = {}

    class _StubStrategy:
        __name__ = "StubStrategy"

    class _StubBacktest:
        def __init__(self):
            self.portfolio = SimpleNamespace(
                output_summary_stats_fast=lambda: {
                    "status": "ok",
                    "sharpe": 1.5,
                    "cagr": 0.1,
                    "max_drawdown": 0.05,
                },
                trade_count=7,
            )

    def _stub_run_backtest_chunked(**kwargs):
        captured.update(kwargs)
        return _StubBacktest()

    monkeypatch.setattr(optimize, "PARQUET_MODE", True)
    monkeypatch.setattr(optimize, "ACTIVE_MARKET_DB_PATH", "data/market_parquet")
    monkeypatch.setattr(optimize, "ACTIVE_MARKET_EXCHANGE", "binance")
    monkeypatch.setattr(optimize, "ACTIVE_BASE_TIMEFRAME", "1s")
    monkeypatch.setattr(optimize, "STRATEGY_TIMEFRAME", "1m")
    monkeypatch.setattr(optimize, "run_backtest_chunked", _stub_run_backtest_chunked)
    monkeypatch.setattr(
        optimize,
        "load_data_dict_from_parquet",
        lambda *_args, **_kwargs: {"BTC/USDT": object()},
    )

    result = optimize._execute_backtest(
        _StubStrategy,
        {},
        "data",
        ["BTC/USDT"],
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
    )

    assert result["sharpe"] == 1.5
    assert captured["data_handler_cls"] is optimize.HistoricParquetWindowedDataHandler
    assert captured["backtest_mode"] == "windowed"
    assert dict(captured["data_handler_kwargs"]) == {
        "backtest_poll_seconds": int(optimize.BACKTEST_POLL_SECONDS),
        "backtest_window_seconds": int(optimize.BACKTEST_WINDOW_SECONDS),
    }


def test_parquet_mode_forces_single_worker_in_parallel_runner(monkeypatch):
    monkeypatch.setattr(optimize, "PARQUET_MODE", True)
    monkeypatch.setattr(optimize, "run_single_backtest_train", lambda args: ("ok", args))

    results = optimize._run_backtests_parallel([("a",), ("b",)], worker_count=8)

    assert results == [("ok", ("a",)), ("ok", ("b",))]
