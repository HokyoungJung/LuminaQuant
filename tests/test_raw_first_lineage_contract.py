from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from lumina_quant.data.raw_first_lineage import (
    coerce_raw_aggtrades_frame,
    normalize_exchange_timestamp_ms,
    raw_aggtrades_to_1s_frame,
    resolve_raw_aggtrades_backend_name,
)
import lumina_quant.data.native_raw_first_backend as native_raw_first_backend
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def test_normalize_exchange_timestamp_ms_rejects_non_ms_units() -> None:
    assert normalize_exchange_timestamp_ms(1_700_000_000_000, source="rest") == 1_700_000_000_000
    with pytest.raises(ValueError, match="seconds-like"):
        normalize_exchange_timestamp_ms(1_700_000_000, source="rest")
    with pytest.raises(ValueError, match="microseconds-like"):
        normalize_exchange_timestamp_ms(1_700_000_000_000_000, source="rest")


def test_coerce_raw_aggtrades_frame_polars_matches_iterable_path() -> None:
    rows = [
        {
            "agg_trade_id": 10,
            "timestamp_ms": 1_700_000_000_100,
            "price": 100.0,
            "quantity": 1.0,
            "is_buyer_maker": False,
        },
        {
            "agg_trade_id": 10,
            "timestamp_ms": 1_700_000_000_100,
            "price": 101.0,
            "quantity": 2.0,
            "is_buyer_maker": True,
        },
        {
            "agg_trade_id": 11,
            "timestamp_ms": 1_700_000_000_900,
            "price": 0.0,
            "quantity": 1.0,
            "is_buyer_maker": False,
        },
        {
            "agg_trade_id": 12,
            "timestamp_ms": 1_700_000_001_100,
            "price": 102.0,
            "quantity": 3.0,
            "is_buyer_maker": False,
        },
    ]
    iterable = coerce_raw_aggtrades_frame(rows, source="iterable")
    polars = coerce_raw_aggtrades_frame(pl.DataFrame(rows), source="polars")

    assert polars.to_dicts() == iterable.to_dicts()
    assert polars["price"].to_list() == pytest.approx([101.0, 102.0])
    assert polars["quantity"].to_list() == pytest.approx([2.0, 3.0])


def test_coerce_raw_aggtrades_frame_polars_rejects_non_ms_units() -> None:
    frame = pl.DataFrame(
        {
            "agg_trade_id": [1],
            "timestamp_ms": [1_700_000_000],
            "price": [100.0],
            "quantity": [1.0],
            "is_buyer_maker": [False],
        }
    )

    with pytest.raises(ValueError, match="seconds-like"):
        coerce_raw_aggtrades_frame(frame, source="polars")


def test_raw_aggtrades_to_1s_frame_drops_incomplete_last_second() -> None:
    frame = raw_aggtrades_to_1s_frame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_700_000_000_500,
                "price": 101.0,
                "quantity": 2.0,
                "is_buyer_maker": True,
            },
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_700_000_001_100,
                "price": 102.0,
                "quantity": 3.0,
                "is_buyer_maker": False,
            },
        ],
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_001_100,
        complete_through_ms=1_700_000_001_100,
    )

    assert frame.height == 1
    assert frame["open"][0] == pytest.approx(100.0)
    assert frame["close"][0] == pytest.approx(101.0)
    assert frame["volume"][0] == pytest.approx(3.0)


def test_raw_aggtrades_to_1s_frame_skips_leading_gaps_without_previous_close() -> None:
    frame = raw_aggtrades_to_1s_frame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_002_100,
                "price": 102.0,
                "quantity": 1.5,
                "is_buyer_maker": False,
            }
        ],
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_002_999,
        complete_through_ms=1_700_000_002_999,
    )

    assert frame.height == 1
    assert frame["datetime"][0] == datetime(2023, 11, 14, 22, 13, 22)
    assert frame["close"][0] == pytest.approx(102.0)
    assert frame["volume"][0] == pytest.approx(1.5)


def test_raw_aggtrades_to_1s_frame_fills_leading_gaps_with_previous_close() -> None:
    frame = raw_aggtrades_to_1s_frame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_002_100,
                "price": 102.0,
                "quantity": 1.5,
                "is_buyer_maker": False,
            }
        ],
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_002_999,
        previous_close=99.0,
        complete_through_ms=1_700_000_002_999,
    )

    assert frame.height == 3
    assert frame["close"].to_list() == pytest.approx([99.0, 99.0, 102.0])
    assert frame["volume"].to_list() == pytest.approx([0.0, 0.0, 1.5])


def test_raw_aggtrades_to_1s_frame_explicit_rust_mode_errors_when_backend_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "lumina_quant.data.native_raw_first_backend._load_native_function",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="native library is unavailable"):
        raw_aggtrades_to_1s_frame(
            [
                {
                    "agg_trade_id": 1,
                    "timestamp_ms": 1_700_000_000_000,
                    "price": 100.0,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                }
            ],
            source="pytest",
            range_start_ms=1_700_000_000_000,
            range_end_ms=1_700_000_000_999,
            complete_through_ms=1_700_000_000_999,
            backend="rust",
        )


def test_resolve_raw_aggtrades_backend_name_reports_python_without_native(monkeypatch) -> None:
    monkeypatch.setattr(
        "lumina_quant.data.native_raw_first_backend._load_native_function",
        lambda: None,
    )
    assert resolve_raw_aggtrades_backend_name("auto") == "python"


def test_raw_first_backend_diagnostics_reports_load_error(monkeypatch) -> None:
    native_raw_first_backend._AUTO_FALLBACK_WARNED.clear()
    native_raw_first_backend._NATIVE_HANDLE = None
    native_raw_first_backend._NATIVE_FN = None
    native_raw_first_backend._NATIVE_DLL = ""
    monkeypatch.setattr(
        "lumina_quant.data.native_raw_first_backend._load_native_function",
        lambda: None,
    )
    monkeypatch.setattr(
        native_raw_first_backend,
        "_NATIVE_LOAD_ERROR",
        "failed to load test library",
    )
    native_raw_first_backend._AUTO_FALLBACK_WARNED.add("failed to load test library")

    diagnostics = native_raw_first_backend.raw_first_backend_diagnostics("auto")

    assert diagnostics["requested_backend"] == "auto"
    assert diagnostics["resolved_backend"] == "python"
    assert diagnostics["description"] == "python"
    assert diagnostics["native_load_error"] == "failed to load test library"
    assert diagnostics["auto_fallback_warning_count"] == 1
    assert diagnostics["auto_fallback_warning_reasons"] == ["failed to load test library"]


def test_raw_aggtrades_auto_backend_logs_once_when_native_unavailable(monkeypatch, caplog) -> None:
    native_raw_first_backend._AUTO_FALLBACK_WARNED.clear()
    monkeypatch.setattr(
        "lumina_quant.data.native_raw_first_backend._load_native_function",
        lambda: None,
    )
    monkeypatch.setattr(
        native_raw_first_backend,
        "_NATIVE_LOAD_ERROR",
        "failed to load test library",
    )
    caplog.set_level("WARNING")

    payload = [
        {
            "agg_trade_id": 1,
            "timestamp_ms": 1_700_000_000_000,
            "price": 100.0,
            "quantity": 1.0,
            "is_buyer_maker": False,
        }
    ]

    first = raw_aggtrades_to_1s_frame(
        payload,
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_000_999,
        complete_through_ms=1_700_000_000_999,
        backend="auto",
    )
    second = raw_aggtrades_to_1s_frame(
        payload,
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_000_999,
        complete_through_ms=1_700_000_000_999,
        backend="auto",
    )

    assert first.height == 1
    assert second.height == 1
    records = [
        record.message for record in caplog.records if "falling back to Python" in record.message
    ]
    assert len(records) == 1
    assert "failed to load test library" in records[0]


def test_raw_aggtrades_auto_backend_logs_once_when_native_status_fails(monkeypatch, caplog) -> None:
    native_raw_first_backend._AUTO_FALLBACK_WARNED.clear()
    monkeypatch.setattr(
        "lumina_quant.data.native_raw_first_backend._load_native_function",
        lambda: lambda *args, **kwargs: 7,
    )
    caplog.set_level("WARNING")

    frame = raw_aggtrades_to_1s_frame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
            }
        ],
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_000_999,
        complete_through_ms=1_700_000_000_999,
        backend="auto",
    )

    assert frame.height == 1
    records = [record.message for record in caplog.records if "status=7" in record.message]
    assert len(records) == 1


def test_committed_loader_rebuilds_higher_timeframe_from_1s_and_truncates_incomplete_tail(
    tmp_path,
) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    partition_date = "2026-03-01"
    partition_root = repo.materialized_partition_root(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
    )
    commit_id = "20260301-1s"
    commit_dir = partition_root / f"commit={commit_id}"
    commit_dir.mkdir(parents=True, exist_ok=True)

    base_dt = datetime(2026, 3, 1, 0, 0, tzinfo=UTC).replace(tzinfo=None)
    frame = pl.DataFrame(
        {
            "datetime": [base_dt + timedelta(seconds=index) for index in range(90)],
            "open": [100.0 + (index * 0.1) for index in range(90)],
            "high": [100.5 + (index * 0.1) for index in range(90)],
            "low": [99.5 + (index * 0.1) for index in range(90)],
            "close": [100.2 + (index * 0.1) for index in range(90)],
            "volume": [1.0 for _ in range(90)],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
    data_file = commit_dir / "part-0000.parquet"
    frame.write_parquet(data_file)

    checksum = repo.canonical_row_checksum(frame)
    manifest_payload = {
        "manifest_version": 1,
        "commit_id": commit_id,
        "symbol": "BTC/USDT",
        "timeframe": "1s",
        "partition": str(partition_root),
        "window_start_ms": int(frame["datetime"].min().timestamp() * 1000),
        "window_end_ms": int(frame["datetime"].max().timestamp() * 1000),
        "event_time_watermark_ms": int(frame["datetime"].max().timestamp() * 1000),
        "source_checkpoint_start": int(frame["datetime"].min().timestamp() * 1000),
        "source_checkpoint_end": int(frame["datetime"].max().timestamp() * 1000),
        "row_count": int(frame.height),
        "canonical_row_checksum": checksum,
        "data_files": [f"commit={commit_id}/part-0000.parquet"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer": "pytest",
        "status": "committed",
    }
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
        payload=manifest_payload,
    )

    rebuilt = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date="2026-03-01T00:00:00Z",
        end_date="2026-03-01T00:01:29Z",
        chunk_days=1,
        warmup_bars=0,
        staleness_threshold_seconds=None,
    )

    assert rebuilt.height == 1
    assert rebuilt["datetime"][0] == base_dt
    assert rebuilt["open"][0] == pytest.approx(100.0)
    assert rebuilt["close"][0] == pytest.approx(106.1)
    assert rebuilt["volume"][0] == pytest.approx(60.0)


def test_raw_aggtrades_to_1s_frame_rust_matches_python_when_native_available() -> None:
    if resolve_raw_aggtrades_backend_name("auto") == "python":
        pytest.skip("Rust raw-first backend is not built")

    rows = [
        {
            "agg_trade_id": 1,
            "timestamp_ms": 1_700_000_000_100,
            "price": 100.0,
            "quantity": 0.1,
            "is_buyer_maker": False,
        },
        {
            "agg_trade_id": 2,
            "timestamp_ms": 1_700_000_000_800,
            "price": 101.0,
            "quantity": 0.2,
            "is_buyer_maker": True,
        },
        {
            "agg_trade_id": 3,
            "timestamp_ms": 1_700_000_002_100,
            "price": 102.0,
            "quantity": 0.3,
            "is_buyer_maker": False,
        },
    ]
    python_frame = raw_aggtrades_to_1s_frame(
        rows,
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_002_999,
        previous_close=99.0,
        complete_through_ms=1_700_000_002_999,
        backend="python",
    )
    rust_frame = raw_aggtrades_to_1s_frame(
        rows,
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_002_999,
        previous_close=99.0,
        complete_through_ms=1_700_000_002_999,
        backend="rust",
    )

    assert rust_frame.shape == python_frame.shape
    assert rust_frame["datetime"].to_list() == python_frame["datetime"].to_list()
    assert rust_frame["open"].to_list() == pytest.approx(python_frame["open"].to_list())
    assert rust_frame["high"].to_list() == pytest.approx(python_frame["high"].to_list())
    assert rust_frame["low"].to_list() == pytest.approx(python_frame["low"].to_list())
    assert rust_frame["close"].to_list() == pytest.approx(python_frame["close"].to_list())
    assert rust_frame["volume"].to_list() == pytest.approx(python_frame["volume"].to_list())
