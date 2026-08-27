from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1] / "scripts/backfill_binance_public_metrics_history.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_binance_public_metrics_history", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
DAY = date(2026, 5, 29)


def _archive(*rows: str, member_symbol: str = "BTCUSDT") -> bytes:
    content = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        "count_long_short_ratio,sum_taker_long_short_vol_ratio\n" + "\n".join(rows) + "\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{member_symbol}-metrics-{DAY.isoformat()}.csv", content)
    return buffer.getvalue()


def test_public_metrics_url_uses_official_daily_archive() -> None:
    assert MODULE.public_metrics_archive_url("BTC/USDT", DAY) == (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2026-05-29.zip"
    )


def test_metrics_archive_normalizes_quote_open_interest() -> None:
    blob = _archive(
        "2026-05-29 00:05:00,BTCUSDT,106592.279,7849530040.1553,1,1,1,1",
        "2026-05-29 00:10:00,BTCUSDT,106589.522,7851616993.622329,1,1,1,1",
        "2026-05-30 00:00:00,BTCUSDT,106500,7800000000,1,1,1,1",
    )

    rows, member = MODULE.metrics_rows_from_zip(
        blob,
        expected_symbol="BTC/USDT",
        expected_day=DAY,
    )

    assert member == "BTCUSDT-metrics-2026-05-29.csv"
    assert rows == [
        {"timestamp_ms": 1_780_013_100_000, "open_interest": 7_849_530_040.1553},
        {"timestamp_ms": 1_780_013_400_000, "open_interest": 7_851_616_993.622329},
        {"timestamp_ms": 1_780_099_200_000, "open_interest": 7_800_000_000.0},
    ]


def test_metrics_archive_rejects_wrong_symbol() -> None:
    blob = _archive("2026-05-29 00:05:00,ETHUSDT,1,2,1,1,1,1")

    with pytest.raises(ValueError, match="symbol does not match"):
        MODULE.metrics_rows_from_zip(blob, expected_symbol="BTC/USDT", expected_day=DAY)


def test_backfill_persists_rows_and_archive_provenance(monkeypatch, tmp_path: Path) -> None:
    blob = _archive("2026-05-29 00:05:00,BTCUSDT,1,2,1,1,1,1")
    captured: dict[str, object] = {}
    monkeypatch.setattr(MODULE, "_download_zip_bytes", lambda *_args, **_kwargs: blob)

    def upsert(db_path, *, exchange, symbol, rows, source):
        captured.update(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            rows=list(rows),
            source=source,
        )
        return len(captured["rows"])

    monkeypatch.setattr(MODULE, "upsert_futures_feature_points_rows", upsert)

    payload = MODULE.backfill_public_metrics_history(
        db_path=str(tmp_path),
        symbols=["BTCUSDT"],
        start_date=DAY,
        end_date=DAY,
        retries=0,
        base_wait_sec=0.0,
    )

    assert captured["symbol"] == "BTC/USDT"
    assert captured["source"] == "binance_public_metrics_archive"
    assert captured["rows"] == [{"timestamp_ms": 1_780_013_100_000, "open_interest": 2.0}]
    assert payload["missing_archive_count"] == 0
    assert payload["total_persisted_rows"] == 1
    [receipt] = payload["imported_archives"]
    assert receipt["archive_byte_count"] == len(blob)
    assert len(receipt["archive_sha256"]) == 64
