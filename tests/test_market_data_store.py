"""Tests for market OHLCV storage helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lumina_quant.influx_market_data import InfluxMarketDataRepository
from lumina_quant.market_data import (
    MarketDataRepository,
    _build_market_data_repository,
    connect_market_data_1s_db,
    connect_market_data_db,
    ensure_market_ohlcv_schema,
    get_last_ohlcv_timestamp_ms,
    load_data_dict_from_db,
    load_ohlcv_from_db,
    normalize_symbol,
    upsert_ohlcv_rows,
    upsert_ohlcv_rows_1s,
)


class TestMarketDataStore(unittest.TestCase):
    """Validate SQLite OHLCV upsert and load behavior."""

    def test_symbol_normalization(self):
        self.assertEqual(normalize_symbol("btcusdt"), "BTC/USDT")
        self.assertEqual(normalize_symbol("BTC-USDT"), "BTC/USDT")
        self.assertEqual(normalize_symbol("eth_usdt"), "ETH/USDT")

    def test_builder_inferred_influx_missing_credentials_falls_back_to_sqlite(self):
        with patch.dict(
            "os.environ",
            {
                "LQ__STORAGE__BACKEND": "influxdb",
            },
            clear=True,
        ):
            repo = _build_market_data_repository("data/lq_market.sqlite3")
        self.assertIsInstance(repo, MarketDataRepository)

    def test_builder_explicit_influx_missing_credentials_is_strict(self):
        with patch.dict("os.environ", {}, clear=True):
            for backend in ("influxdb", "influx"):
                with self.subTest(backend=backend), self.assertRaises(ValueError):
                    _build_market_data_repository("data/lq_market.sqlite3", backend=backend)

    def test_builder_inferred_influx_with_credentials_uses_influx_repo(self):
        with patch.dict(
            "os.environ",
            {
                "LQ__STORAGE__BACKEND": "influxdb",
                "LQ__STORAGE__INFLUX_URL": "http://localhost:8086",
                "LQ__STORAGE__INFLUX_ORG": "test-org",
                "LQ__STORAGE__INFLUX_BUCKET": "test-bucket",
                "INFLUXDB_TOKEN": "test-token",
            },
            clear=True,
        ):
            repo = _build_market_data_repository("data/lq_market.sqlite3")
        self.assertIsInstance(repo, InfluxMarketDataRepository)

    def test_upsert_1s_empty_rows_explicit_influx_missing_credentials_is_strict(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            upsert_ohlcv_rows_1s(
                "data/lq_market.sqlite3",
                exchange="binance",
                symbol="BTC/USDT",
                rows=[],
                backend="influxdb",
            )

    def test_upsert_1s_empty_rows_inferred_influx_missing_credentials_returns_zero(self):
        with patch.dict("os.environ", {"LQ__STORAGE__BACKEND": "influxdb"}, clear=True):
            inserted = upsert_ohlcv_rows_1s(
                "data/lq_market.sqlite3",
                exchange="binance",
                symbol="BTC/USDT",
                rows=[],
            )
        self.assertEqual(inserted, 0)

    def test_upsert_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            conn = connect_market_data_db(db_path)
            try:
                ensure_market_ohlcv_schema(conn)
                rows = [
                    (1704067200000, 100.0, 101.0, 99.0, 100.5, 10.0),
                    (1704067260000, 100.5, 102.0, 100.0, 101.5, 12.0),
                ]
                count = upsert_ohlcv_rows(
                    conn,
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    rows=rows,
                    source="test",
                )
                self.assertEqual(count, 2)

                # Idempotent update on same primary key
                updated_rows = [
                    (1704067260000, 100.5, 102.0, 100.0, 111.5, 12.0),
                ]
                upsert_ohlcv_rows(
                    conn,
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    rows=updated_rows,
                    source="test",
                )
            finally:
                conn.close()

            df = load_ohlcv_from_db(
                db_path,
                exchange="binance",
                symbol="BTCUSDT",
                timeframe="1m",
            )
            self.assertEqual(df.height, 2)
            self.assertAlmostEqual(float(df["close"][1]), 111.5)

            conn2 = connect_market_data_db(db_path)
            try:
                last_ts = get_last_ohlcv_timestamp_ms(
                    conn2,
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                )
            finally:
                conn2.close()
            self.assertEqual(last_ts, 1704067260000)

            data_dict = load_data_dict_from_db(
                db_path,
                exchange="binance",
                symbol_list=["BTC/USDT", "ETH/USDT"],
                timeframe="1m",
            )
            self.assertIn("BTC/USDT", data_dict)
            self.assertNotIn("ETH/USDT", data_dict)

    def test_repository_facade_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            repo = MarketDataRepository(db_path)

            conn = connect_market_data_db(db_path)
            try:
                ensure_market_ohlcv_schema(conn)
                upsert_ohlcv_rows(
                    conn,
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    rows=[(1704067200000, 100.0, 101.0, 99.0, 100.5, 10.0)],
                    source="test",
                )
            finally:
                conn.close()

            self.assertTrue(
                repo.market_data_exists(exchange="binance", symbol="BTCUSDT", timeframe="1m")
            )
            frame = repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")
            self.assertEqual(frame.height, 1)

    def test_load_derived_timeframe_from_1s_respects_window(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            conn = connect_market_data_1s_db(db_path)
            try:
                # Ensure DB file exists and table can be touched by upsert helper.
                conn.execute("SELECT 1")
            finally:
                conn.close()

            upsert_ohlcv_rows_1s(
                db_path,
                exchange="binance",
                symbol="BTC/USDT",
                rows=[
                    (1704067200000, 100.0, 101.0, 99.0, 100.5, 1.0),
                    (1704067201000, 100.5, 102.0, 100.0, 101.5, 2.0),
                    (1704067202000, 101.5, 103.0, 101.0, 102.5, 3.0),
                ],
            )

            frame = load_ohlcv_from_db(
                db_path,
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                start_date="2024-01-01T00:00:00+00:00",
                end_date="2024-01-01T00:00:10+00:00",
            )
            self.assertEqual(frame.height, 1)
            self.assertAlmostEqual(float(frame["open"][0]), 100.0)
            self.assertAlmostEqual(float(frame["high"][0]), 103.0)
            self.assertAlmostEqual(float(frame["low"][0]), 99.0)
            self.assertAlmostEqual(float(frame["close"][0]), 102.5)
            self.assertAlmostEqual(float(frame["volume"][0]), 6.0)


if __name__ == "__main__":
    unittest.main()
