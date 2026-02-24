from __future__ import annotations

from pathlib import Path

import polars as pl
from lumina_quant.parquet_market_data import ParquetMarketDataRepository


def _build_rows(order: str = "base") -> pl.DataFrame:
    rows = [
        {
            "datetime": "2026-01-04T23:59:58Z",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        {
            "datetime": "2026-01-04T23:59:59Z",
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 1.0,
        },
        {
            "datetime": "2026-01-05T00:00:00Z",
            "open": 3.0,
            "high": 3.0,
            "low": 3.0,
            "close": 3.0,
            "volume": 1.0,
        },
        {
            "datetime": "2026-01-05T00:00:01Z",
            "open": 4.0,
            "high": 4.0,
            "low": 4.0,
            "close": 4.0,
            "volume": 1.0,
        },
    ]
    if order == "reversed":
        rows = list(reversed(rows))
    return pl.DataFrame(rows)


def _frame_signature(frame: pl.DataFrame) -> list[tuple]:
    normalized = frame.sort("datetime")
    return [tuple(row) for row in normalized.iter_rows()]


def test_week_chunk_resample_is_deterministic_before_and_after_compaction(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)

    # Write same semantic bars in different row orders and files.
    repo.upsert_1s(exchange="binance", symbol="ETH/USDT", rows=_build_rows("base"))
    repo.upsert_1s(exchange="binance", symbol="ETH/USDT", rows=_build_rows("reversed"))

    pre_a = repo.load_ohlcv(exchange="binance", symbol="ETH/USDT", timeframe="1m")
    pre_b = repo.load_ohlcv(exchange="binance", symbol="ETH/USDT", timeframe="1m")
    assert _frame_signature(pre_a) == _frame_signature(pre_b)

    repo.compact_all(exchange="binance", symbol="ETH/USDT", timeframe="1s", remove_sources=True)

    post_a = repo.load_ohlcv(exchange="binance", symbol="ETH/USDT", timeframe="1m")
    post_b = repo.load_ohlcv(exchange="binance", symbol="ETH/USDT", timeframe="1m")
    assert _frame_signature(post_a) == _frame_signature(post_b)
    assert post_a["volume"].to_list() == [2.0, 2.0]
