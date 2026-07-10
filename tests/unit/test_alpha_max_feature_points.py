from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lumina_quant.data.feature_points import (
    FEATURE_POINT_MAX_STALE_MS,
    FeaturePoint,
    FeaturePointLookup,
)
from lumina_quant.market_data import upsert_futures_feature_points_rows


def test_feature_point_lookup_exposes_immutable_source_timestamp_and_scalar_parity(tmp_path):
    db_path = tmp_path / "market_parquet"
    start_ms = 1_700_000_000_000
    next_ms = start_ms + 60_000
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": start_ms, "funding_rate": 0.0001},
            {"timestamp_ms": next_ms, "mark_price": 50_000.0},
        ],
    )
    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    point = lookup.get_latest_point("BTC/USDT", "funding_rate", timestamp_ms=next_ms)

    assert point == FeaturePoint(value=0.0001, source_timestamp_ms=start_ms)
    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=next_ms) == point.value
    with pytest.raises(FrozenInstanceError):
        point.value = 0.0  # type: ignore[misc]


def test_feature_point_lookup_point_keeps_existing_future_stale_and_finite_semantics(tmp_path):
    db_path = tmp_path / "market_parquet"
    start_ms = 1_700_000_000_000
    stale_boundary_ms = start_ms + FEATURE_POINT_MAX_STALE_MS
    stale_after_ms = stale_boundary_ms + 1
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="ETH/USDT",
        rows=[
            {"timestamp_ms": start_ms, "funding_rate": 0.0003},
            {"timestamp_ms": stale_after_ms, "mark_price": 1800.0},
            {"timestamp_ms": stale_after_ms + 60_000, "open_interest": float("nan")},
        ],
    )
    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert lookup.get_latest_point("ETH/USDT", "funding_rate", timestamp_ms=start_ms - 1) is None
    assert lookup.get_latest_point(
        "ETH/USDT", "funding_rate", timestamp_ms=stale_boundary_ms
    ) == FeaturePoint(
        value=0.0003,
        source_timestamp_ms=start_ms,
    )
    assert lookup.get_latest_point("ETH/USDT", "funding_rate", timestamp_ms=stale_after_ms) is None
    assert lookup.get_latest("ETH/USDT", "funding_rate", timestamp_ms=stale_after_ms) is None
    assert (
        lookup.get_latest_point(
            "ETH/USDT",
            "open_interest",
            timestamp_ms=stale_after_ms + 60_000,
        )
        is None
    )
    assert lookup.get_latest_point("ETH/USDT", "not_a_feature", timestamp_ms=stale_after_ms) is None
