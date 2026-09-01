from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from lumina_quant.storage.parquet import ParquetMarketDataRepository
from lumina_quant.services.materialize_from_raw import materialize_raw_aggtrades


def _checkpoint_payload(row, *, observed_until_ms):
    return {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "last_timestamp_ms": row["timestamp_ms"],
        "last_trade_id": row["agg_trade_id"],
        "observed_until_ms": observed_until_ms,
        "updated_at_utc": "2025-01-01T00:00:00+00:00",
        "batch_rows": 1,
        "last_row": row,
        "last_row_sha256": hashlib.sha256(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def test_materialize_from_raw_produces_deterministic_committed_manifest(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_700_000_000_500,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_700_000_001_000,
                "price": 101.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
        ],
    )

    checkpoint_row = {
        "agg_trade_id": 2,
        "timestamp_ms": 1_700_000_000_500,
        "price": 101.0,
        "quantity": 0.2,
        "is_buyer_maker": True,
    }
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload=_checkpoint_payload(
            checkpoint_row,
            observed_until_ms=1_700_000_000_999,
        ),
    )

    first = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date=None,
        end_date=None,
    )
    second = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date=None,
        end_date=None,
    )

    assert first
    assert second
    assert first[0].canonical_row_checksum == second[0].canonical_row_checksum

    loaded = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
    )
    assert loaded.height >= 1


def test_materialize_explicit_historical_range_uses_bound_checkpoint(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    start_ms = int(datetime(2026, 3, 19, tzinfo=UTC).timestamp() * 1000)
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 10,
                "timestamp_ms": start_ms,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 11,
                "timestamp_ms": start_ms + 1_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
        ],
    )
    checkpoint_row = {
        "agg_trade_id": 11,
        "timestamp_ms": start_ms + 1_000,
        "price": 101.0,
        "quantity": 0.2,
        "is_buyer_maker": True,
    }
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload=_checkpoint_payload(
            checkpoint_row,
            observed_until_ms=start_ms + 1_000,
        ),
    )

    commits = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date="2026-03-19T00:00:00+00:00",
        end_date="2026-03-19T00:00:02+00:00",
    )

    assert commits
    assert commits[0].partition == "2026-03-19"
