from __future__ import annotations

import hashlib

import threading
from pathlib import Path
import polars as pl
import json

import pytest
from lumina_quant import market_data
from lumina_quant.market_data import (
    load_futures_feature_points_from_db,
    upsert_futures_feature_points_rows,
)


def test_feature_point_upsert_handles_sparse_mixed_numeric_rows(tmp_path):
    db_path = tmp_path / "market_parquet"

    upserted = upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
        rows=[
            {"timestamp_ms": 1_700_000_000_000, "open_interest": 1},
            {
                "timestamp_ms": 1_700_000_060_000,
                "funding_rate": 0.00031,
                "funding_fee_rate": 0.00031,
                "funding_fee_quote_per_unit": 0.62,
            },
            {
                "timestamp_ms": 1_700_000_120_000,
                "liquidation_long_qty": 2,
                "liquidation_long_notional": 1.5,
            },
        ],
    )

    assert upserted == 3

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
    )
    assert frame.height == 3
    assert frame.get_column("funding_rate").drop_nulls().to_list() == [0.00031]
    assert frame.get_column("funding_fee_rate").drop_nulls().to_list() == [0.00031]
    assert frame.get_column("funding_fee_quote_per_unit").drop_nulls().to_list() == [0.62]


def test_feature_point_load_respects_date_partition_bounds(tmp_path):
    db_path = tmp_path / "market_parquet"

    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": 1_735_689_600_000, "funding_rate": 0.00010},  # 2025-01-02
            {"timestamp_ms": 1_741_910_400_000, "funding_rate": 0.00020},  # 2025-03-15
            {"timestamp_ms": 1_749_600_000_000, "funding_rate": 0.00030},  # 2025-06-12
        ],
    )

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        start_date="2025-03-01",
        end_date="2025-04-01",
    )

    assert frame.height == 1
    assert frame.get_column("funding_rate").to_list() == [0.00020]


def test_feature_point_load_emits_partition_and_collect_progress(tmp_path, monkeypatch):
    db_path = tmp_path / "market_parquet"

    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": 1_741_910_400_000, "funding_rate": 0.00020},
        ],
    )
    events: list[tuple[str, dict[str, object]]] = []
    counter = iter([10.0, 10.2, 20.0, 20.3])
    monkeypatch.setattr(market_data, "perf_counter", lambda: next(counter))

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        start_date="2025-03-01",
        end_date="2025-04-01",
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert frame.height == 1
    assert [name for name, _ in events] == [
        "resource_feature_partition_scan_completed",
        "resource_feature_collect_started",
        "resource_feature_collect_completed",
    ]
    assert events[0][1]["partition_count"] == 1
    assert events[0][1]["parquet_file_count"] == 1
    assert events[0][1]["elapsed_seconds"] == 0.2
    assert events[2][1]["row_count"] == 1
    assert events[2][1]["elapsed_seconds"] == 0.3


def test_official_funding_merge_preserves_existing_feature_columns(tmp_path):
    root = tmp_path / "market_parquet"
    timestamp_ms = 1_704_067_200_000
    upsert_futures_feature_points_rows(
        str(root),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[{"timestamp_ms": timestamp_ms, "open_interest": 42.0}],
    )
    source = tmp_path / "funding.parquet"
    pl.DataFrame(
        {
            "timestamp_ms": [timestamp_ms],
            "source_timestamp_ms": [timestamp_ms],
            "exchange": ["binance"],
            "symbol": ["BTC/USDT"],
            "funding_rate": [0.0001],
        },
        schema={
            "timestamp_ms": pl.Int64,
            "source_timestamp_ms": pl.Int64,
            "exchange": pl.Utf8,
            "symbol": pl.Utf8,
            "funding_rate": pl.Float64,
        },
    ).write_parquet(source)
    repo = market_data.MarketDataRepository(str(root))
    repo.publish_official_funding_day(
        exchange="binance",
        symbol="BTC/USDT",
        day="2024-01-01",
        source=source,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_byte_count=source.stat().st_size,
        expected_row_count=1,
        provenance_receipt_sha256="0" * 64,
    )
    frame = repo.load_futures_feature_points(
        exchange="binance", symbol="BTC/USDT", start_date="2024-01-01", end_date="2024-01-01"
    )
    assert frame.get_column("open_interest").to_list() == [42.0]
    assert frame.get_column("funding_rate").to_list() == [0.0001]


def test_feature_upserts_share_canonical_schema_across_days_and_preserve_nulls(tmp_path):
    root = tmp_path / "market_parquet"
    upsert_futures_feature_points_rows(
        str(root),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": 1_704_067_200_000, "open_interest": 4.0},
            {"timestamp_ms": 1_704_153_600_000, "funding_rate": 0.0002},
        ],
    )
    frame = load_futures_feature_points_from_db(str(root), exchange="binance", symbol="BTC/USDT")
    assert frame.columns == [
        "exchange",
        "symbol",
        "timestamp_ms",
        "datetime",
        "source",
        *market_data._FEATURE_COLUMNS,
    ]
    assert frame.get_column("open_interest").to_list() == [4.0, None]
    assert frame.get_column("funding_rate").to_list() == [None, 0.0002]


def test_same_day_feature_upserts_do_not_lose_disjoint_updates(tmp_path):
    root = tmp_path / "market_parquet"
    barrier = threading.Barrier(2)

    def write(row):
        barrier.wait()
        upsert_futures_feature_points_rows(
            str(root), exchange="binance", symbol="BTC/USDT", rows=[row]
        )

    first = {"timestamp_ms": 1_704_067_200_000, "open_interest": 7.0}
    second = {"timestamp_ms": 1_704_067_201_000, "funding_rate": 0.0001}
    threads = [threading.Thread(target=write, args=(row,)) for row in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    frame = load_futures_feature_points_from_db(str(root), exchange="binance", symbol="BTC/USDT")
    assert frame.height == 2
    assert frame.get_column("open_interest").drop_nulls().to_list() == [7.0]
    assert frame.get_column("funding_rate").drop_nulls().to_list() == [0.0001]


def test_official_funding_seal_is_idempotent_recovers_pending_and_rejects_replacement(tmp_path):
    root = tmp_path / "market_parquet"
    timestamp_ms = 1_704_067_200_000
    source = tmp_path / "funding.parquet"
    official = pl.DataFrame(
        {
            "timestamp_ms": [timestamp_ms],
            "source_timestamp_ms": [timestamp_ms],
            "exchange": ["binance"],
            "symbol": ["BTC/USDT"],
            "funding_rate": [0.0001],
        },
        schema={
            "timestamp_ms": pl.Int64,
            "source_timestamp_ms": pl.Int64,
            "exchange": pl.Utf8,
            "symbol": pl.Utf8,
            "funding_rate": pl.Float64,
        },
    )
    official.write_parquet(source)
    repo = market_data.MarketDataRepository(str(root))
    receipt = {
        "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "expected_byte_count": source.stat().st_size,
        "expected_row_count": 1,
        "provenance_receipt_sha256": "0" * 64,
    }
    kwargs = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "day": "2024-01-01",
        "source": source,
        **receipt,
    }
    output = repo.publish_official_funding_day(**kwargs)
    assert repo.publish_official_funding_day(**kwargs) == output
    seal = output.parent / "alpha_max_official_funding_seal.v1"
    pending = json.loads(seal.read_text())
    pending.update(state="pending", output_sha256="", output_byte_count=0)
    seal.write_text(json.dumps(pending))
    assert repo.publish_official_funding_day(**kwargs) == output
    official.with_columns(pl.lit(0.0002).alias("funding_rate")).write_parquet(source)
    replacement = {
        **kwargs,
        "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "expected_byte_count": source.stat().st_size,
    }
    with pytest.raises(ValueError, match="seal conflicts"):
        repo.publish_official_funding_day(**replacement)


def test_atomic_feature_write_surfaces_obsolete_cleanup_failure(tmp_path, monkeypatch):
    date_path = tmp_path / "date=2024-01-01"
    date_path.mkdir()
    output = date_path / "compact-2024-01-01.parquet"
    obsolete = date_path / "obsolete.parquet"
    frame = pl.DataFrame(
        [{"timestamp_ms": 1_704_067_200_000}],
        schema={"timestamp_ms": pl.Int64},
    )
    frame.write_parquet(obsolete)
    original_unlink = Path.unlink

    def fail_only_obsolete(path, *args, **kwargs):
        if path == obsolete:
            raise OSError("cleanup failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_only_obsolete)
    with pytest.raises(OSError, match="cleanup failed"):
        market_data._atomic_feature_write(date_path, output, frame)
