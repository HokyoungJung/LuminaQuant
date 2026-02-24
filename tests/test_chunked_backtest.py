from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from lumina_quant.parquet_market_data import ParquetMarketDataRepository

_RUN_BACKTEST_PATH = Path(__file__).resolve().parents[1] / "run_backtest.py"
_RUN_BACKTEST_SPEC = importlib.util.spec_from_file_location("run_backtest_module", _RUN_BACKTEST_PATH)
if _RUN_BACKTEST_SPEC is None or _RUN_BACKTEST_SPEC.loader is None:
    raise RuntimeError(f"Failed to load run_backtest module from {_RUN_BACKTEST_PATH}")
run_backtest = importlib.util.module_from_spec(_RUN_BACKTEST_SPEC)
_RUN_BACKTEST_SPEC.loader.exec_module(run_backtest)


def _cross_day_1s_frame() -> pl.DataFrame:
    start = datetime(2026, 1, 1, 23, 59, 0)
    rows = 180
    datetimes = [start + timedelta(seconds=idx) for idx in range(rows)]
    return pl.DataFrame(
        {
            "datetime": datetimes,
            "open": [100.0 + idx * 0.1 for idx in range(rows)],
            "high": [100.5 + idx * 0.1 for idx in range(rows)],
            "low": [99.5 + idx * 0.1 for idx in range(rows)],
            "close": [100.2 + idx * 0.1 for idx in range(rows)],
            "volume": [1.0 for _ in range(rows)],
        }
    )


def test_chunked_resample_matches_single_pass(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_cross_day_1s_frame())

    start = datetime(2026, 1, 1, 23, 59, 0)
    end = datetime(2026, 1, 2, 0, 1, 59)

    single = repo.load_ohlcv(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date=start,
        end_date=end,
    )
    chunked = repo.load_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date=start,
        end_date=end,
        chunk_days=1,
        warmup_bars=60,
    )

    assert chunked.equals(single)


def test_run_backtest_loader_uses_parquet_when_detected(tmp_path: Path, monkeypatch):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_cross_day_1s_frame())

    monkeypatch.setattr(run_backtest, "SYMBOL_LIST", ["BTC/USDT"])
    monkeypatch.setattr(run_backtest, "START_DATE", datetime(2026, 1, 1, 23, 59, 0))
    monkeypatch.setattr(run_backtest, "END_DATE", datetime(2026, 1, 2, 0, 1, 59))
    monkeypatch.setattr(run_backtest, "MARKET_DB_BACKEND", "parquet")
    monkeypatch.setattr(run_backtest, "BT_CHUNK_DAYS", 1)
    monkeypatch.setattr(run_backtest, "BT_CHUNK_WARMUP_BARS", 60)

    data_dict = run_backtest._load_data_dict(
        "db",
        str(tmp_path),
        "binance",
        base_timeframe="1m",
        auto_collect_db=False,
    )

    assert data_dict is not None
    assert "BTC/USDT" in data_dict
    assert data_dict["BTC/USDT"].height >= 2
