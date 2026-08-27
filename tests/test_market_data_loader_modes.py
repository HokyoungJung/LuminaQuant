from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.data.raw_first_lineage import resample_1s_frame
from lumina_quant.market_data import MarketDataRepository, load_data_dict_from_parquet
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def _seed_legacy_and_manifest(root) -> None:
    repo = ParquetMarketDataRepository(str(root))
    repo.upsert_1s(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "datetime": datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.2,
                "volume": 10.0,
            }
        ],
    )

    partition_date = "2026-03-01"
    partition_root = repo.materialized_partition_root(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
    )
    commit_id = "seed-commit"
    commit_dir = partition_root / f"commit={commit_id}"
    commit_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 3, 1, 0, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.2],
            "volume": [10.0],
        }
    ).with_columns(pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime(time_unit="ms")))
    data_file = commit_dir / "part-0000.parquet"
    frame.write_parquet(data_file)
    dt = frame["datetime"][0]
    ts_ms = int(dt.timestamp() * 1000)
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
        payload={
            "manifest_version": 1,
            "commit_id": commit_id,
            "symbol": "BTC/USDT",
            "timeframe": "1s",
            "partition": str(partition_root),
            "window_start_ms": ts_ms,
            "window_end_ms": ts_ms,
            "event_time_watermark_ms": ts_ms,
            "source_checkpoint_start": ts_ms,
            "source_checkpoint_end": ts_ms,
            "row_count": 1,
            "canonical_row_checksum": repo.canonical_row_checksum(frame),
            "data_files": [f"commit={commit_id}/part-0000.parquet"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": "pytest",
            "status": "committed",
        },
    )


def test_facade_preserves_existing_dotted_directory_root(tmp_path):
    candidate = tmp_path / ".market_parquet.g003-candidate"
    candidate.mkdir()

    facade = MarketDataRepository(str(candidate))

    assert facade.logical_root_path == candidate


def test_facade_preserves_dotted_directory_symlink_for_repository_validation(tmp_path):
    target = tmp_path / "external"
    target.mkdir()
    candidate = tmp_path / ".market_parquet.g003-candidate"
    candidate.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="owned sibling"):
        MarketDataRepository(str(candidate))


def test_facade_preserves_broken_dotted_symlink_for_fail_closed_resolution(tmp_path):
    candidate = tmp_path / ".market_parquet.g003-candidate"
    candidate.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ValueError, match="owned sibling"):
        MarketDataRepository(str(candidate))


def test_facade_keeps_legacy_file_symlink_fallback(tmp_path):
    legacy = tmp_path / "actual.sqlite"
    legacy.write_bytes(b"legacy")
    configured = tmp_path / "market_data.sqlite"
    configured.symlink_to(legacy)

    facade = MarketDataRepository(str(configured))

    assert facade.logical_root_path == tmp_path / "market_parquet"


def test_facade_keeps_legacy_file_path_fallback(tmp_path):
    facade = MarketDataRepository(str(tmp_path / "market_data.sqlite"))

    assert facade.logical_root_path == tmp_path / "market_parquet"


def test_parquet_loader_uses_canonical_resampler_exactly(tmp_path):
    repo = ParquetMarketDataRepository(tmp_path)
    start = datetime(2026, 6, 1, tzinfo=UTC)
    repo.upsert_1s(
        exchange="binance",
        symbol="BNB/USDT",
        rows=[
            {
                "datetime": start + timedelta(seconds=offset),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 0.1 + (offset % 11) * 0.01,
            }
            for offset in range(120)
        ],
    )
    raw = repo.load_ohlcv(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1s",
        start_date=start,
        end_date=start + timedelta(seconds=119),
    )

    loaded = repo.load_ohlcv(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1m",
        start_date=start,
        end_date=start + timedelta(seconds=119),
    )
    expected = resample_1s_frame(
        raw,
        timeframe="1m",
        complete_through_ms=int((start + timedelta(seconds=120)).timestamp() * 1000) - 1,
    )

    assert loaded.equals(expected)


def test_chunked_and_unchunked_resampling_match_across_unaligned_boundary(tmp_path):
    repo = ParquetMarketDataRepository(tmp_path)
    start = datetime(2026, 6, 1, 0, 30, tzinfo=UTC)
    offsets = (
        timedelta(minutes=0),
        timedelta(minutes=20),
        timedelta(hours=23, minutes=45),
        timedelta(days=1, minutes=-15),
        timedelta(days=1),
        timedelta(days=1, minutes=20),
        timedelta(days=1, hours=23),
        timedelta(days=2),
    )
    repo.upsert_1s(
        exchange="binance",
        symbol="BNB/USDT",
        rows=[
            {
                "datetime": start + offset,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1.0 + index,
            }
            for index, offset in enumerate(offsets)
        ],
    )
    end = start + timedelta(days=2)

    unchunked = repo.load_ohlcv(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1h",
        start_date=start,
        end_date=end,
    )
    chunked = repo.load_ohlcv_chunked(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1h",
        start_date=start,
        end_date=end,
        chunk_days=1,
    )
    chunked_with_warmup = repo.load_ohlcv_chunked(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1h",
        start_date=start,
        end_date=end,
        chunk_days=1,
        warmup_bars=2,
    )
    raw_unchunked = repo.load_ohlcv(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1s",
        start_date=start,
        end_date=end,
    )
    raw_chunked = repo.load_ohlcv_chunked(
        exchange="binance",
        symbol="BNB/USDT",
        timeframe="1s",
        start_date=start,
        end_date=end,
        chunk_days=1,
        warmup_bars=2,
    )

    assert chunked.equals(unchunked)
    assert chunked_with_warmup.equals(unchunked)
    assert raw_chunked.equals(raw_unchunked)


def test_loader_supports_legacy_and_raw_first_modes(tmp_path):
    _seed_legacy_and_manifest(tmp_path)

    legacy = load_data_dict_from_parquet(
        str(tmp_path),
        exchange="binance",
        symbol_list=["BTC/USDT"],
        timeframe="1s",
        data_mode="legacy",
    )
    raw_first = load_data_dict_from_parquet(
        str(tmp_path),
        exchange="binance",
        symbol_list=["BTC/USDT"],
        timeframe="1s",
        data_mode="raw-first",
    )

    assert "BTC/USDT" in legacy
    assert "BTC/USDT" in raw_first
    assert raw_first["BTC/USDT"].height == 1


def test_raw_first_missing_symbol_is_fail_fast(tmp_path):
    _seed_legacy_and_manifest(tmp_path)

    with pytest.raises(RawFirstDataMissingError):
        load_data_dict_from_parquet(
            str(tmp_path),
            exchange="binance",
            symbol_list=["BTC/USDT", "ETH/USDT"],
            timeframe="1s",
            data_mode="raw-first",
        )


def test_facade_merges_direct_bars_with_explicit_source_lineage(tmp_path, monkeypatch):
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    second_minute = datetime(2026, 6, 1, 0, 1, tzinfo=UTC)
    monkeypatch.setenv("LQ_PREFER_1S_DERIVED", "1")
    facade = MarketDataRepository(str(tmp_path))
    facade.upsert_ohlcv(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        rows=[
            {
                "timestamp_ms": int(start.timestamp() * 1000),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            },
            {
                "timestamp_ms": int(second_minute.timestamp() * 1000),
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 12.0,
            },
        ],
    )
    ParquetMarketDataRepository(str(tmp_path)).upsert_1s(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "datetime": start,
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "volume": 2.0,
            }
        ],
    )

    loaded, audit = facade.load_ohlcv_with_source_audit(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date=start,
        end_date=second_minute,
    )

    assert loaded.height == 2
    assert loaded["close"].to_list() == [110.5, 101.5]
    assert audit["precedence"] == "resampled_1s_derived_over_direct_1m"
    assert audit["direct_rows"] == 2
    assert audit["resampled_rows"] == 1
    assert audit["direct_only_rows"] == 1
    assert audit["resampled_only_rows"] == 0
    assert audit["overlap_rows"] == 1
    assert audit["overlap_equal_rows"] == 0
    assert audit["overlap_conflict_rows"] == 1
    assert audit["first_overlap_conflict_timestamp_ms"] == int(start.timestamp() * 1000)
    assert audit["effective_direct_rows"] == 1
    assert audit["effective_resampled_rows"] == 1
    assert len(audit["overlap_conflict_sha256"]) == 64

    monkeypatch.setenv("LQ_PREFER_1S_DERIVED", "0")
    direct_preferred = MarketDataRepository(str(tmp_path))
    direct_loaded, direct_audit = direct_preferred.load_ohlcv_with_source_audit(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date=start,
        end_date=second_minute,
    )

    assert direct_loaded["close"].to_list() == [100.5, 101.5]
    assert direct_audit["precedence"] == "direct_1m_over_resampled_1s_derived"
    assert direct_audit["effective_direct_rows"] == 2
    assert direct_audit["effective_resampled_rows"] == 0
