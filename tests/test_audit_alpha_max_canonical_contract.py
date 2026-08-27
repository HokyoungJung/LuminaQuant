from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from scripts.research import audit_alpha_max_canonical_contract as subject


def _raw_frame(timestamps: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.5] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        },
        schema=subject.RAW_SCHEMA,
    )


def test_deep_raw_audit_accepts_exact_second_grid(tmp_path: Path) -> None:
    start_ms = 1_704_067_200_000
    path = tmp_path / "2024-01.parquet"
    _raw_frame(
        [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 0, 0, 1),
            datetime(2024, 1, 1, 0, 0, 2),
        ]
    ).write_parquet(path)

    audit = subject.audit_raw_partition(
        path,
        symbol="BTCUSDT",
        month="2024-01",
        start_ms=start_ms,
        end_ms=start_ms + 3000,
        deep=True,
    )

    assert audit["status"] == "complete"
    assert audit["actual_rows"] == 3
    assert audit["deep_failures"] == {
        "timestamp_sequence_failures": 0,
        "non_finite_values": 0,
        "non_positive_prices": 0,
        "negative_volume": 0,
        "ohlc_failures": 0,
    }


def test_deep_raw_audit_rejects_duplicate_and_missing_second(tmp_path: Path) -> None:
    start_ms = 1_704_067_200_000
    path = tmp_path / "2024-01.parquet"
    _raw_frame(
        [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 0, 0, 2),
            datetime(2024, 1, 1, 0, 0, 2),
        ]
    ).write_parquet(path)

    audit = subject.audit_raw_partition(
        path,
        symbol="BTCUSDT",
        month="2024-01",
        start_ms=start_ms,
        end_ms=start_ms + 3000,
        deep=True,
    )

    assert audit["status"] == "incomplete"
    assert audit["deep_failures"]["timestamp_sequence_failures"] > 0
    shallow = subject.audit_raw_partition(
        path,
        symbol="BTCUSDT",
        month="2024-01",
        start_ms=start_ms,
        end_ms=start_ms + 3000,
        deep=False,
    )
    assert shallow["status"] == "inventory-complete"


def test_funding_audit_accepts_exchange_jitter_and_rejects_settlement_collision(
    tmp_path: Path,
) -> None:
    symbol_root = (
        tmp_path / "feature_points" / "exchange=binance" / "symbol=BTCUSDT" / "date=1970-01-01"
    )
    symbol_root.mkdir(parents=True)
    path = symbol_root / "funding.parquet"
    interval = subject.funding_interval_ms("BTCUSDT")
    pl.DataFrame(
        {
            "timestamp_ms": [5, interval + 14],
            "funding_rate": [0.0001, 0.0002],
        }
    ).write_parquet(path)

    complete = subject.audit_funding(
        tmp_path,
        symbol="BTCUSDT",
        start_ms=0,
        end_ms=2 * interval,
    )
    assert complete["status"] == "complete"
    assert complete["max_observed_jitter_ms"] == 14

    pl.DataFrame(
        {
            "timestamp_ms": [0, 60_000, interval],
            "funding_rate": [0.0001, 0.0003, 0.0002],
        }
    ).write_parquet(path)
    incomplete = subject.audit_funding(
        tmp_path,
        symbol="BTCUSDT",
        start_ms=0,
        end_ms=2 * interval,
    )

    assert incomplete["status"] == "incomplete"
    assert incomplete["missing_rows"] == 0
    assert incomplete["extra_rows_in_window"] == 0
    assert incomplete["duplicate_rows_in_window"] == 1
    assert incomplete["jitter_violation_rows"] == 1


def test_partial_raw_audit_preserves_valid_rows_outside_contract(tmp_path: Path) -> None:
    partition_start = 1_704_067_200_000
    start_ms = partition_start + 86_400_000
    path = tmp_path / "2024-01.parquet"
    _raw_frame(
        [
            datetime(2024, 1, 1, 23, 59, 59),
            datetime(2024, 1, 2, 0, 0, 0),
            datetime(2024, 1, 2, 0, 0, 1),
            datetime(2024, 1, 2, 0, 0, 2),
            datetime(2024, 1, 2, 0, 0, 3),
        ]
    ).write_parquet(path)

    audit = subject.audit_raw_partition(
        path,
        symbol="BTCUSDT",
        month="2024-01",
        start_ms=start_ms,
        end_ms=start_ms + 3_000,
        deep=True,
    )

    assert audit["status"] == "complete"
    assert audit["actual_rows"] == 3
    assert audit["outside_contract_rows"] == 2
    assert audit["mispartitioned_rows"] == 0


def test_funding_audit_rejects_duplicate_across_feature_files(tmp_path: Path) -> None:
    symbol_root = (
        tmp_path / "feature_points" / "exchange=binance" / "symbol=BTCUSDT" / "date=1970-01-01"
    )
    symbol_root.mkdir(parents=True)
    for filename in ("compact.parquet", "funding.parquet"):
        pl.DataFrame(
            {"timestamp_ms": [0], "funding_rate": [0.0001]},
            schema={"timestamp_ms": pl.Int64, "funding_rate": pl.Float64},
        ).write_parquet(symbol_root / filename)
    pl.DataFrame({"timestamp_ms": [0], "mark_price": [100.0]}).write_parquet(
        symbol_root / "mark.parquet"
    )

    audit = subject.audit_funding(
        tmp_path,
        symbol="BTCUSDT",
        start_ms=0,
        end_ms=subject.funding_interval_ms("BTCUSDT"),
    )

    assert audit["status"] == "incomplete"
    assert audit["missing_rows"] == 0
    assert audit["duplicate_rows_in_window"] == 1
    assert audit["errors"] == []
