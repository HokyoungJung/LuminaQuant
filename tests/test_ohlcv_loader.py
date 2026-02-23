from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime

import polars as pl
from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader


class TestOHLCVFrameLoader(unittest.TestCase):
    def test_normalize_selects_filters_and_sorts(self):
        frame = pl.DataFrame(
            {
                "datetime": [
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ],
                "open": [3.0, 1.0, 2.0],
                "high": [3.1, 1.1, 2.1],
                "low": [2.9, 0.9, 1.9],
                "close": [3.0, 1.0, 2.0],
                "volume": [30.0, 10.0, 20.0],
                "ignored": [1, 2, 3],
            }
        )
        loader = OHLCVFrameLoader(
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 3),
        )

        out = loader.normalize(frame)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.columns, ["datetime", "open", "high", "low", "close", "volume"])
        self.assertEqual(out.height, 2)
        values = out["datetime"].to_list()
        self.assertEqual(values[0], datetime(2024, 1, 2))
        self.assertEqual(values[1], datetime(2024, 1, 3))

    def test_load_csv_returns_none_on_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "invalid.csv")
            pl.DataFrame(
                {
                    "datetime": ["2024-01-01"],
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.0],
                    # Missing volume
                }
            ).write_csv(csv_path)

            loader = OHLCVFrameLoader()
            out = loader.load_csv(csv_path)
            self.assertIsNone(out)

    def test_normalize_handles_timezone_aware_datetime_with_naive_bounds(self):
        frame = pl.DataFrame(
            {
                "datetime": [
                    datetime(2025, 1, 1, tzinfo=UTC),
                    datetime(2025, 1, 2, tzinfo=UTC),
                ],
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.0, 2.0],
                "volume": [10.0, 20.0],
            }
        )
        loader = OHLCVFrameLoader(start_date=datetime(2025, 1, 2), end_date=datetime(2025, 1, 2))
        out = loader.normalize(frame)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.height, 1)
        self.assertEqual(out["datetime"][0], datetime(2025, 1, 2))


if __name__ == "__main__":
    unittest.main()
