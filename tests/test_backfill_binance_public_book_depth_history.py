from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from lumina_quant.market_data import load_futures_feature_points_from_db
from scripts import backfill_binance_public_book_depth_history as module


def _zip_csv(name: str, body: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def test_public_book_depth_archive_url() -> None:
    url = module.public_book_depth_archive_url("BTC/USDT", date(2026, 5, 31))
    assert url.endswith("/BTCUSDT/BTCUSDT-bookDepth-2026-05-31.zip")


def test_rows_from_book_depth_zip_derives_nearest_depth_imbalance() -> None:
    blob = _zip_csv(
        "BTCUSDT-bookDepth-2026-05-31.csv",
        "timestamp,percentage,depth,notional\n"
        "2026-05-31 00:00:07,-5.00,10,500\n"
        "2026-05-31 00:00:07,-1.00,20,700\n"
        "2026-05-31 00:00:07,1.00,30,300\n"
        "2026-05-31 00:00:07,5.00,40,900\n",
    )

    rows, scanned = module._rows_from_book_depth_zip(blob, cadence_seconds=3600)

    assert scanned == 4
    assert rows == [
        {
            "timestamp_ms": 1780185600000,
            "book_depth_bid_notional_1pct": 700.0,
            "book_depth_ask_notional_1pct": 300.0,
            "book_depth_imbalance_1pct": 0.4,
        }
    ]


def test_backfill_public_book_depth_history_persists_features(monkeypatch, tmp_path: Path) -> None:
    blob = _zip_csv(
        "SOLUSDT-bookDepth-2026-05-31.csv",
        "timestamp,percentage,depth,notional\n"
        "2026-05-31 00:00:07,-1.00,20,700\n"
        "2026-05-31 00:00:07,1.00,30,300\n",
    )
    monkeypatch.setattr(module, "_download_zip_bytes", lambda url, *, retries, base_wait_sec: blob)

    db_path = tmp_path / "market_parquet"
    payload = module.backfill_public_book_depth_history(
        db_path=str(db_path),
        symbols=["SOLUSDT"],
        start_date=date(2026, 5, 31),
        end_date=date(2026, 5, 31),
    )

    assert payload["total_persisted_rows"] == 1
    loaded = load_futures_feature_points_from_db(
        str(db_path), exchange="binance", symbol="SOL/USDT"
    )
    row = loaded.to_dicts()[0]
    assert row["book_depth_bid_notional_1pct"] == 700.0
    assert row["book_depth_ask_notional_1pct"] == 300.0
    assert row["book_depth_imbalance_1pct"] == 0.4


def test_backfill_public_book_depth_history_batches_symbol_upsert(monkeypatch) -> None:
    blob = _zip_csv(
        "BTCUSDT-bookDepth-2026-05-31.csv",
        "timestamp,percentage,depth,notional\n"
        "2026-05-31 00:00:07,-1.00,20,700\n"
        "2026-05-31 00:00:07,1.00,30,300\n",
    )
    calls: list[list[dict[str, object]]] = []
    monkeypatch.setattr(module, "_download_zip_bytes", lambda url, *, retries, base_wait_sec: blob)

    def fake_upsert(
        db_path: str,
        *,
        exchange: str,
        symbol: str,
        rows: list[dict[str, object]],
        source: str,
    ) -> int:
        calls.append(rows)
        return len(rows)

    monkeypatch.setattr(module, "upsert_futures_feature_points_rows", fake_upsert)

    payload = module.backfill_public_book_depth_history(
        db_path="unused",
        symbols=["BTCUSDT"],
        start_date=date(2026, 5, 31),
        end_date=date(2026, 6, 1),
    )

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert payload["persisted_rows_by_symbol"]["BTC/USDT"] == 2
    assert [entry["queued_rows"] for entry in payload["imported_archives"]] == [1, 1]
