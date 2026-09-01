from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import polars as pl

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "backfill_raw_taker_flow_feature_points.py"
)
SPEC = importlib.util.spec_from_file_location("backfill_raw_taker_flow_feature_points", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_aggregate_raw_taker_flow_uses_buyer_maker_side_and_cadence_end() -> None:
    raw = pl.DataFrame(
        {
            "timestamp_ms": [1_700_000_001_000, 1_700_000_005_000, 1_700_000_021_000],
            "price": [100.0, 110.0, 120.0],
            "quantity": [2.0, 3.0, 4.0],
            "is_buyer_maker": [False, True, False],
        }
    )

    out = MODULE.aggregate_raw_taker_flow(
        raw,
        exchange="binance",
        symbol="BTC/USDT",
        cadence_seconds=20,
        source="unit",
    )

    rows = out.to_dicts()
    assert [row["timestamp_ms"] for row in rows] == [1_700_000_020_000, 1_700_000_040_000]
    assert rows[0]["taker_buy_base_volume"] == 2.0
    assert rows[0]["taker_sell_base_volume"] == 3.0
    assert rows[0]["taker_buy_quote_volume"] == 200.0
    assert rows[0]["taker_sell_quote_volume"] == 330.0
    assert rows[1]["taker_buy_base_volume"] == 4.0
    assert rows[1]["taker_sell_base_volume"] == 0.0


def test_merge_feature_points_preserves_existing_funding_and_adds_taker_flow() -> None:
    existing = pl.DataFrame(
        {
            "exchange": ["binance"],
            "symbol": ["BTC/USDT"],
            "timestamp_ms": [1_700_000_020_000],
            "datetime": ["2023-11-14T22:13:40+00:00"],
            "source": ["funding"],
            "funding_rate": [0.0001],
        }
    )
    incoming = pl.DataFrame(
        {
            "exchange": ["binance"],
            "symbol": ["BTC/USDT"],
            "timestamp_ms": [1_700_000_020_000],
            "datetime": ["2023-11-14T22:13:40+00:00"],
            "source": ["raw_aggtrades_taker_flow_20s"],
            "taker_buy_quote_volume": [200.0],
            "taker_sell_quote_volume": [330.0],
        }
    )

    merged = MODULE.merge_feature_points(existing, incoming)

    row = merged.to_dicts()[0]
    assert row["funding_rate"] == 0.0001
    assert row["taker_buy_quote_volume"] == 200.0
    assert row["taker_sell_quote_volume"] == 330.0
