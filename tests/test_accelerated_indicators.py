import unittest

from lumina_quant.indicators import (
    NUMBA_AVAILABLE,
    POLARS_AVAILABLE,
    TALIB_AVAILABLE,
    close_to_close_volatility,
    compute_fast_alpha_bundle,
    garman_klass_volatility,
    linear_decay_latest,
    rogers_satchell_volatility,
    rolling_corr_latest_numpy,
    rolling_feature_frame_polars,
    rolling_mean_latest_numpy,
    rolling_std_latest_numpy,
    talib_feature_pack,
    yang_zhang_volatility,
)


class TestAcceleratedIndicators(unittest.TestCase):
    def setUp(self):
        self.closes = [100.0 + float(idx) + (0.1 if idx % 4 == 0 else -0.05) for idx in range(160)]
        self.opens = [value - 0.2 for value in self.closes]
        self.highs = [value + 0.9 for value in self.closes]
        self.lows = [value - 1.0 for value in self.closes]
        self.volumes = [1200.0 + 3.0 * idx for idx in range(160)]

    def test_acceleration_flags(self):
        self.assertTrue(isinstance(TALIB_AVAILABLE, bool))
        self.assertTrue(isinstance(POLARS_AVAILABLE, bool))
        self.assertTrue(isinstance(NUMBA_AVAILABLE, bool))

    def test_numpy_and_range_estimators(self):
        self.assertIsNotNone(linear_decay_latest(self.closes, window=10))
        self.assertIsNotNone(rolling_mean_latest_numpy(self.closes, window=20))
        self.assertIsNotNone(rolling_std_latest_numpy(self.closes, window=20))
        self.assertIsNotNone(rolling_corr_latest_numpy(self.closes, self.volumes, window=20))
        self.assertIsNotNone(close_to_close_volatility(self.closes, window=20))
        self.assertIsNotNone(
            garman_klass_volatility(self.opens, self.highs, self.lows, self.closes)
        )
        self.assertIsNotNone(
            rogers_satchell_volatility(self.opens, self.highs, self.lows, self.closes)
        )
        self.assertIsNotNone(yang_zhang_volatility(self.opens, self.highs, self.lows, self.closes))

    def test_feature_pack_and_alpha_bundle(self):
        pack = talib_feature_pack(self.opens, self.highs, self.lows, self.closes, self.volumes)
        self.assertIn("rsi", pack)
        self.assertIn("macd", pack)

        bundle = compute_fast_alpha_bundle(
            self.opens,
            self.highs,
            self.lows,
            self.closes,
            self.volumes,
        )
        self.assertIn("alpha_001", bundle)
        self.assertIn("vol_yang_zhang", bundle)

    def test_polars_feature_frame(self):
        if POLARS_AVAILABLE:
            frame = rolling_feature_frame_polars(
                self.opens,
                self.highs,
                self.lows,
                self.closes,
                self.volumes,
            )
            self.assertGreater(frame.height, 0)
            self.assertIn("ma_spread", frame.columns)
            self.assertIn("atr_percent", frame.columns)


if __name__ == "__main__":
    unittest.main()
