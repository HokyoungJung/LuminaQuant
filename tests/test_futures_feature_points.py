from __future__ import annotations

import os
import tempfile
import unittest

from lumina_quant.market_data import (
    FUTURES_FEATURE_POINTS_TABLE,
    connect_market_data_db,
    ensure_futures_feature_points_schema,
    upsert_futures_feature_points,
)


class TestFuturesFeaturePoints(unittest.TestCase):
    def test_upsert_and_merge_preserves_existing_non_null_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "feature_store")
            conn = connect_market_data_db(db_path)
            try:
                ensure_futures_feature_points_schema(conn)
                upserted = upsert_futures_feature_points(
                    conn,
                    exchange="binance",
                    symbol="BTC/USDT",
                    rows=[
                        {
                            "timestamp_ms": 1_700_000_000_000,
                            "funding_rate": 0.0001,
                            "mark_price": 50000.0,
                        }
                    ],
                )
                self.assertEqual(upserted, 1)

                upserted_again = upsert_futures_feature_points(
                    conn,
                    exchange="binance",
                    symbol="BTC/USDT",
                    rows=[
                        {
                            "timestamp_ms": 1_700_000_000_000,
                            "index_price": 49990.0,
                        }
                    ],
                )
                self.assertEqual(upserted_again, 1)

                row = conn.execute(
                    f"SELECT funding_rate, mark_price, index_price FROM {FUTURES_FEATURE_POINTS_TABLE}"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertAlmostEqual(float(row[0]), 0.0001)
                self.assertAlmostEqual(float(row[1]), 50000.0)
                self.assertAlmostEqual(float(row[2]), 49990.0)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
