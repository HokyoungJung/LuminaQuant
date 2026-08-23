from __future__ import annotations

import hashlib
import os
import stat

import polars as pl
import pytest

from lumina_quant.data.feature_points import (
    FEATURE_POINT_MAX_STALE_MS,
    FeaturePointLookup,
    SealedFeatureFile,
)
from lumina_quant.market_data import upsert_futures_feature_points_rows


def _sealed_feature_fixture(tmp_path, *, rate: float = 0.0001):
    db_path = tmp_path / "market_parquet"
    timestamp_ms = 1_700_000_000_000
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[{"timestamp_ms": timestamp_ms, "funding_rate": rate}],
    )
    partition = next(db_path.rglob("*.parquet"))
    observed = partition.stat(follow_symlinks=False)
    entry = SealedFeatureFile(
        relative_path=partition.relative_to(db_path).as_posix(),
        byte_count=observed.st_size,
        mode=stat.S_IMODE(observed.st_mode),
        mtime_ns=observed.st_mtime_ns,
        sha256=hashlib.sha256(partition.read_bytes()).hexdigest(),
    )
    return db_path, partition, entry, timestamp_ms


def test_feature_point_lookup_forward_fills_latest_non_null_value(tmp_path):
    db_path = tmp_path / "market_parquet"
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "timestamp_ms": 1_700_000_000_000,
                "funding_rate": 0.0001,
                "funding_fee_rate": 0.0001,
                "funding_fee_quote_per_unit": 5.0,
            },
            {"timestamp_ms": 1_700_000_060_000, "mark_price": 50_000.0},
            {"timestamp_ms": 1_700_000_120_000, "open_interest": 1_250_000.0},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=1_700_000_060_000) == 0.0001
    assert (
        lookup.get_latest("BTC/USDT", "funding_fee_rate", timestamp_ms=1_700_000_060_000) == 0.0001
    )
    assert (
        lookup.get_latest("BTC/USDT", "funding_fee_quote_per_unit", timestamp_ms=1_700_000_060_000)
        == 5.0
    )
    assert lookup.get_latest("BTC/USDT", "mark_price", timestamp_ms=1_700_000_060_000) == 50_000.0
    assert lookup.get_latest("BTC/USDT", "open_interest", timestamp_ms=1_700_000_060_000) is None
    assert (
        lookup.get_latest("BTC/USDT", "open_interest", timestamp_ms=1_700_000_120_000)
        == 1_250_000.0
    )
    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=1_699_999_000_000) is None


def test_feature_point_lookup_sums_raw_feature_window(tmp_path):
    db_path = tmp_path / "market_parquet"
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="ETH/USDT",
        rows=[
            {"timestamp_ms": 1_700_000_000_000, "taker_buy_quote_volume": 10.0},
            {"timestamp_ms": 1_700_000_060_000, "taker_buy_quote_volume": 15.0},
            {"timestamp_ms": 1_700_000_120_000, "taker_buy_quote_volume": None},
            {"timestamp_ms": 1_700_000_180_000, "taker_buy_quote_volume": 5.0},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert (
        lookup.sum_between(
            "ETH/USDT",
            "taker_buy_quote_volume",
            start_timestamp_ms=1_700_000_000_000,
            end_timestamp_ms=1_700_000_120_000,
        )
        == 25.0
    )
    assert (
        lookup.sum_between(
            "ETH/USDT",
            "taker_buy_quote_volume",
            start_timestamp_ms=1_700_000_120_001,
            end_timestamp_ms=1_700_000_180_000,
        )
        == 5.0
    )
    assert (
        lookup.sum_between(
            "ETH/USDT",
            "taker_buy_quote_volume",
            start_timestamp_ms=1_700_000_240_000,
            end_timestamp_ms=1_700_000_300_000,
        )
        is None
    )


def test_feature_point_lookup_does_not_forward_fill_beyond_staleness_limit(tmp_path):
    db_path = tmp_path / "market_parquet"
    start_ms = 1_700_000_000_000
    stale_ms = start_ms + FEATURE_POINT_MAX_STALE_MS + 1
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": start_ms, "funding_rate": 0.0001},
            {"timestamp_ms": stale_ms, "mark_price": 50_000.0},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=start_ms) == 0.0001
    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=stale_ms) is None


def test_feature_point_lookup_waits_for_jittered_source_timestamp(tmp_path):
    db_path = tmp_path / "market_parquet"
    prior_ms = 1_700_000_000_000
    boundary_ms = prior_ms + FEATURE_POINT_MAX_STALE_MS
    source_ms = boundary_ms + 8
    partition = (
        db_path
        / "feature_points"
        / "exchange=binance"
        / "symbol=XRPUSDT"
        / "date=2023-11-14"
        / "part-0.parquet"
    )
    partition.parent.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "timestamp_ms": prior_ms,
                "source_timestamp_ms": prior_ms,
                "funding_rate": 0.0001,
            },
            {
                "timestamp_ms": boundary_ms,
                "source_timestamp_ms": source_ms,
                "funding_rate": -0.0002,
            },
        ]
    ).write_parquet(partition)
    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    at_boundary = lookup.get_latest_point(
        "XRP/USDT",
        "funding_rate",
        timestamp_ms=boundary_ms,
    )
    after_source = lookup.get_latest_point(
        "XRP/USDT",
        "funding_rate",
        timestamp_ms=source_ms,
    )

    assert at_boundary is not None
    assert at_boundary.value == pytest.approx(0.0001)
    assert at_boundary.source_timestamp_ms == prior_ms
    assert after_source is not None
    assert after_source.value == pytest.approx(-0.0002)
    assert after_source.source_timestamp_ms == source_ms


def test_sealed_feature_lookup_rejects_post_binding_content_replacement(tmp_path):
    db_path, partition, entry, timestamp_ms = _sealed_feature_fixture(tmp_path)
    lookup = FeaturePointLookup(
        db_path=str(db_path.resolve()),
        exchange="binance",
        sealed_files=(entry,),
    )

    replacement = partition.with_name("replacement.parquet")
    pl.DataFrame({"timestamp_ms": [timestamp_ms], "funding_rate": [0.999]}).write_parquet(
        replacement
    )
    os.replace(replacement, partition)

    with pytest.raises(ValueError, match=r"sealed_feature_(metadata|content)_mismatch"):
        lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=timestamp_ms)


def test_sealed_feature_lookup_keeps_original_root_capability_after_path_swap(tmp_path):
    db_path, _partition, entry, timestamp_ms = _sealed_feature_fixture(tmp_path)
    lookup = FeaturePointLookup(
        db_path=str(db_path.resolve()),
        exchange="binance",
        sealed_files=(entry,),
    )

    original_root = tmp_path / "original-root"
    db_path.rename(original_root)
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[{"timestamp_ms": timestamp_ms, "funding_rate": 0.999}],
    )

    assert lookup.get_latest(
        "BTC/USDT", "funding_rate", timestamp_ms=timestamp_ms
    ) == pytest.approx(0.0001)


def test_sealed_feature_lookup_rejects_post_binding_hardlink(tmp_path):
    db_path, partition, entry, timestamp_ms = _sealed_feature_fixture(tmp_path)
    lookup = FeaturePointLookup(
        db_path=str(db_path.resolve()),
        exchange="binance",
        sealed_files=(entry,),
    )
    os.link(partition, partition.with_name("hardlink.parquet"))

    with pytest.raises(ValueError, match="sealed_feature_hardlink_rejected"):
        lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=timestamp_ms)


def test_sealed_feature_lookup_rejects_symlinked_root(tmp_path):
    db_path, _partition, entry, _timestamp_ms = _sealed_feature_fixture(tmp_path)
    alias = tmp_path / "feature-alias"
    alias.symlink_to(db_path, target_is_directory=True)

    with pytest.raises(ValueError, match=r"sealed_feature_root_(symlink|open_failed)_rejected"):
        FeaturePointLookup(
            db_path=str(alias.absolute()),
            exchange="binance",
            sealed_files=(entry,),
        )
