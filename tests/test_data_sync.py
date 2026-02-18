"""Tests for Binance OHLCV sync helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lumina_quant.data_sync import (
    MarketDataSyncService,
    SyncRequest,
    ensure_market_data_coverage,
    get_symbol_ohlcv_coverage,
    parse_timestamp_input,
    sync_symbol_ohlcv,
)
from lumina_quant.market_data import (
    connect_market_data_db,
    ensure_market_ohlcv_schema,
    upsert_ohlcv_rows,
)


class _FakeExchange:
    """Deterministic fake exchange for pagination tests."""

    rateLimit = 0

    def __init__(self):
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        _ = (symbol, timeframe, limit)
        self.calls.append(int(since or 0))
        cursor = int(since or 0)
        if cursor <= 0:
            return [
                [0, 100.0, 101.0, 99.0, 100.5, 10.0],
                [60000, 100.5, 102.0, 100.0, 101.0, 9.0],
            ]
        if cursor <= 120000:
            return [
                [60000, 100.5, 102.0, 100.0, 101.0, 9.0],
                [120000, 101.0, 103.0, 100.5, 102.5, 11.0],
            ]
        return []


class _FakeTradeFallbackExchange:
    """Exchange that rejects 1s klines and serves trades for fallback aggregation."""

    rateLimit = 0

    def __init__(self):
        self.trade_calls = []

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        _ = (symbol, timeframe, since, limit)
        raise RuntimeError("Invalid interval")

    def fetch_trades(self, symbol, since=None, limit=1000):
        _ = (symbol, limit)
        cursor = int(since or 0)
        self.trade_calls.append(cursor)
        if cursor <= 0:
            return [
                {"timestamp": 1000, "price": 100.0, "amount": 1.0},
                {"timestamp": 1600, "price": 101.0, "amount": 2.0},
                {"timestamp": 2200, "price": 99.0, "amount": 1.5},
            ]
        if cursor <= 3000:
            return [
                {"timestamp": 3000, "price": 102.0, "amount": 3.0},
            ]
        return []


class TestDataSync(unittest.TestCase):
    """Validate sync pagination and timestamp parsing."""

    def test_parse_timestamp_input(self):
        self.assertEqual(parse_timestamp_input(1700000000), 1700000000000)
        self.assertEqual(parse_timestamp_input("1700000000000"), 1700000000000)
        self.assertEqual(parse_timestamp_input("1970-01-01T00:00:01+00:00"), 1000)

    def test_sync_symbol_ohlcv_dedupes_overlapping_batches(self):
        fake = _FakeExchange()
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            conn = connect_market_data_db(db_path)
            try:
                ensure_market_ohlcv_schema(conn)
            finally:
                conn.close()

            stats = sync_symbol_ohlcv(
                exchange=fake,
                db_path=db_path,
                exchange_id="binance",
                symbol="BTC/USDT",
                timeframe="1m",
                start_ms=0,
                end_ms=180000,
                limit=2,
                max_batches=10,
            )
            self.assertEqual(stats.fetched_rows, 3)
            self.assertEqual(stats.upserted_rows, 3)
            self.assertEqual(stats.first_timestamp_ms, 0)
            self.assertEqual(stats.last_timestamp_ms, 120000)
            self.assertGreaterEqual(len(fake.calls), 2)

    def test_ensure_market_data_coverage_fills_head_and_tail_gaps(self):
        fake = _FakeExchange()
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            conn = connect_market_data_db(db_path)
            try:
                ensure_market_ohlcv_schema(conn)
                upsert_ohlcv_rows(
                    conn,
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    rows=[(60000, 100.5, 102.0, 100.0, 101.0, 9.0)],
                    source="seed",
                )
            finally:
                conn.close()

            stats = ensure_market_data_coverage(
                exchange=fake,
                db_path=db_path,
                exchange_id="binance",
                symbol_list=["BTC/USDT"],
                timeframe="1m",
                since_ms=0,
                until_ms=180000,
                limit=2,
                max_batches=10,
            )
            self.assertEqual(len(stats), 1)
            self.assertGreaterEqual(int(stats[0].fetched_rows), 2)

            first_ts, last_ts, row_count = get_symbol_ohlcv_coverage(
                db_path=db_path,
                exchange_id="binance",
                symbol="BTC/USDT",
                timeframe="1m",
            )
            self.assertEqual(first_ts, 0)
            self.assertEqual(last_ts, 120000)
            self.assertEqual(row_count, 3)

    def test_sync_symbol_1s_falls_back_to_trades_when_ohlcv_unsupported(self):
        fake = _FakeTradeFallbackExchange()
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            conn = connect_market_data_db(db_path)
            try:
                ensure_market_ohlcv_schema(conn)
            finally:
                conn.close()

            stats = sync_symbol_ohlcv(
                exchange=fake,
                db_path=db_path,
                exchange_id="binance",
                symbol="BTC/USDT",
                timeframe="1s",
                start_ms=0,
                end_ms=3500,
                limit=1000,
                max_batches=20,
            )
            self.assertGreaterEqual(len(fake.trade_calls), 1)
            self.assertEqual(stats.fetched_rows, 3)
            self.assertEqual(stats.upserted_rows, 3)
            self.assertEqual(stats.first_timestamp_ms, 1000)
            self.assertEqual(stats.last_timestamp_ms, 3000)

    def test_sync_service_facade(self):
        fake = _FakeExchange()
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "market_data.db")
            conn = connect_market_data_db(db_path)
            try:
                ensure_market_ohlcv_schema(conn)
            finally:
                conn.close()

            service = MarketDataSyncService(exchange=fake, db_path=db_path, exchange_id="binance")
            stats = service.sync_symbol(
                SyncRequest(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    start_ms=0,
                    end_ms=180000,
                    limit=2,
                    max_batches=10,
                )
            )
            self.assertEqual(stats.fetched_rows, 3)
            coverage = service.get_symbol_coverage(symbol="BTC/USDT", timeframe="1m")
            self.assertEqual(coverage[2], 3)


if __name__ == "__main__":
    unittest.main()
