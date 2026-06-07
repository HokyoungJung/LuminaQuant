from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from lumina_quant.market_data import load_futures_feature_points_from_db
from scripts import import_binance_book_ticker_history as module


def test_normalize_bbo_frame_with_symbol_override() -> None:
    frame = pl.DataFrame(
        {
            "exchange_ts_ms": [1_700_000_000_000],
            "bid": [100.0],
            "B": [2.0],
            "ask": [100.2],
            "A": [3.0],
        }
    )

    normalized = module.normalize_bbo_frame(frame, symbol_override="BTC/USDT")

    assert normalized.to_dicts()[0]["symbol"] == "BTC/USDT"
    assert normalized.to_dicts()[0]["bbo_mid_price"] == 100.1
    assert normalized.to_dicts()[0]["bbo_spread_bps"] > 0.0


def test_import_bbo_history_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "bbo.csv"
    csv_path.write_text(
        "timestamp_ms,symbol,bid_price,bid_qty,ask_price,ask_qty\n"
        "1700000000000,BTC/USDT,100,2,100.2,3\n"
        "1700000005000,BTC/USDT,100.1,2,100.3,3\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "market_parquet"

    payload = module.import_bbo_history(db_path=str(db_path), input_path=csv_path)

    assert payload["total_persisted_rows"] == 2
    loaded = load_futures_feature_points_from_db(
        str(db_path), exchange="binance", symbol="BTC/USDT"
    )
    rows = (
        loaded.sort("timestamp_ms")
        .select(["timestamp_ms", "best_bid_price", "best_ask_price", "bbo_spread_bps"])
        .to_dicts()
    )
    assert rows[0]["best_bid_price"] == 100.0
    assert rows[1]["best_ask_price"] == 100.3
    assert rows[0]["bbo_spread_bps"] > 0.0


def test_import_bbo_history_jsonl_with_aliases(tmp_path: Path) -> None:
    path = tmp_path / "bbo.jsonl"
    path.write_text(
        json.dumps(
            {"time": 1700000000000, "s": "ETH/USDT", "b": 50.0, "B": 1.0, "a": 50.1, "A": 1.5}
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "market_parquet"

    payload = module.import_bbo_history(db_path=str(db_path), input_path=path)

    assert payload["symbols"] == ["ETH/USDT"]
    loaded = load_futures_feature_points_from_db(
        str(db_path), exchange="binance", symbol="ETH/USDT"
    )
    row = loaded.to_dicts()[0]
    assert row["best_bid_price"] == 50.0
    assert row["best_ask_price"] == 50.1


def test_normalize_bbo_frame_accepts_official_binance_archive_columns() -> None:
    frame = pl.DataFrame(
        {
            "transaction_time": [1_711_756_800_007],
            "best_bid_price": [0.12004],
            "best_bid_qty": [108662.0],
            "best_ask_price": [0.12005],
            "best_ask_qty": [31803.0],
        }
    )
    normalized = module.normalize_bbo_frame(frame, symbol_override="TRXUSDT")
    row = normalized.to_dicts()[0]
    assert row["timestamp_ms"] == 1_711_756_800_007
    assert row["symbol"] == "TRX/USDT"
    assert row["best_bid_price"] == 0.12004
    assert row["best_ask_price"] == 0.12005
