from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from lumina_quant.cli import optimize


def test_optimize_uses_windowed_handler_in_parquet_mode(monkeypatch):
    captured: dict[str, object] = {}

    class _StubStrategy:
        __name__ = "StubStrategy"

    class _StubBacktest:
        def __init__(self):
            self.portfolio = SimpleNamespace(
                output_summary_stats_fast=lambda: {
                    "status": "ok",
                    "sharpe": 1.0,
                    "cagr": 0.0,
                    "max_drawdown": 0.0,
                },
                trade_count=0,
            )

    def _stub_run_backtest_chunked(**kwargs):
        captured.update(kwargs)
        return _StubBacktest()

    monkeypatch.setattr(optimize, "PARQUET_MODE", True)
    monkeypatch.setattr(optimize, "ACTIVE_MARKET_DB_PATH", "data/market_parquet")
    monkeypatch.setattr(optimize, "ACTIVE_MARKET_EXCHANGE", "binance")
    monkeypatch.setattr(optimize, "ACTIVE_BASE_TIMEFRAME", "1s")
    monkeypatch.setattr(optimize, "run_backtest_chunked", _stub_run_backtest_chunked)
    monkeypatch.setattr(
        optimize,
        "load_data_dict_from_parquet",
        lambda *_args, **_kwargs: {"BTC/USDT": object()},
    )

    optimize._execute_backtest(
        _StubStrategy,
        {},
        "data",
        ["BTC/USDT"],
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
    )

    assert captured["data_handler_cls"] is optimize.HistoricParquetWindowedDataHandler
    assert captured["backtest_mode"] == "windowed"
    assert dict(captured["data_handler_kwargs"]) == {
        "backtest_poll_seconds": int(optimize.BacktestConfig.POLL_SECONDS),
        "backtest_window_seconds": int(optimize.BacktestConfig.WINDOW_SECONDS),
    }
