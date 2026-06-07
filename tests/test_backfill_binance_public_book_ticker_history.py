from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from lumina_quant.market_data import load_futures_feature_points_from_db
from scripts import backfill_binance_public_book_ticker_history as module


def _zip_csv(name: str, body: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def test_public_book_ticker_archive_url() -> None:
    url = module.public_book_ticker_archive_url("BTC/USDT", date(2024, 3, 30))
    assert url.endswith("/BTCUSDT/BTCUSDT-bookTicker-2024-03-30.zip")


def test_backfill_public_book_ticker_history_imports_official_archive_shape(
    monkeypatch, tmp_path: Path
) -> None:
    blob = _zip_csv(
        "TRXUSDT-bookTicker-2024-03-30.csv",
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
        "1,0.12004,108662,0.12005,31803,1711756800007,1711756800013\n"
        "2,0.12005,108600,0.12006,31700,1711756800100,1711756800105\n",
    )

    def fake_download(url: str, *, retries: int, base_wait_sec: float) -> bytes | None:
        assert "TRXUSDT-bookTicker-2024-03-30.zip" in url
        return blob

    monkeypatch.setattr(module, "_download_zip_bytes", fake_download)
    db_path = tmp_path / "market_parquet"

    payload = module.backfill_public_book_ticker_history(
        db_path=str(db_path),
        symbols=["TRXUSDT"],
        start_date=date(2024, 3, 30),
        end_date=date(2024, 3, 30),
    )

    assert payload["total_persisted_rows"] == 2
    loaded = load_futures_feature_points_from_db(
        str(db_path), exchange="binance", symbol="TRX/USDT"
    )
    rows = (
        loaded.sort("timestamp_ms")
        .select(["timestamp_ms", "best_bid_price", "best_ask_price"])
        .to_dicts()
    )
    assert rows == [
        {"timestamp_ms": 1711756800007, "best_bid_price": 0.12004, "best_ask_price": 0.12005},
        {"timestamp_ms": 1711756800100, "best_bid_price": 0.12005, "best_ask_price": 0.12006},
    ]


def test_backfill_public_book_ticker_history_can_cadence_sample_archive(
    monkeypatch, tmp_path: Path
) -> None:
    blob = _zip_csv(
        "BTCUSDT-bookTicker-2024-03-30.csv",
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
        "1,100.0,1,100.1,1,1711756800007,1711756800013\n"
        "2,100.0,1,100.1,1,1711756800500,1711756800505\n"
        "3,101.0,1,101.1,1,1711756805000,1711756805005\n",
    )

    monkeypatch.setattr(
        module,
        "_download_zip_bytes",
        lambda url, *, retries, base_wait_sec: blob,
    )

    payload = module.backfill_public_book_ticker_history(
        db_path=str(tmp_path / "market_parquet"),
        symbols=["BTCUSDT"],
        start_date=date(2024, 3, 30),
        end_date=date(2024, 3, 30),
        cadence_seconds=5,
    )

    assert payload["imported_archives"][0]["archive_rows"] == 3
    assert payload["imported_archives"][0]["normalized_rows"] == 2
    assert payload["total_persisted_rows"] == 2


def test_backfill_public_book_ticker_history_records_missing_archives(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        module,
        "_download_zip_bytes",
        lambda url, *, retries, base_wait_sec: None,
    )

    payload = module.backfill_public_book_ticker_history(
        db_path=str(tmp_path / "market_parquet"),
        symbols=["BTCUSDT"],
        start_date=date(2024, 3, 30),
        end_date=date(2024, 3, 31),
    )

    assert payload["imported_archive_count"] == 0
    assert payload["missing_archive_count"] == 2
    assert payload["persisted_rows_by_symbol"] == {"BTC/USDT": 0}
