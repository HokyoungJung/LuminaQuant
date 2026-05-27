import math
import unittest

from lumina_quant.indicators import (
    IncrementalRsi,
    RollingMeanWindow,
    RollingZScoreWindow,
    average_true_range,
    momentum_return,
    momentum_spread,
    rolling_beta,
    rolling_corr,
    rolling_vwap,
    safe_float,
    safe_int,
    sample_std,
    time_key,
    true_range,
    vwap_deviation,
    vwap_from_sums,
)


class TestIndicatorCore(unittest.TestCase):
    def test_common_helpers(self):
        self.assertEqual(safe_float("1.25"), 1.25)
        self.assertIsNone(safe_float(float("inf")))
        self.assertIsNone(safe_float("nan"))
        self.assertEqual(safe_int("12"), 12)
        self.assertEqual(safe_int("x", default=7), 7)
        self.assertEqual(time_key(None), "")
        self.assertEqual(time_key(123), "123")

    def test_rolling_mean_window(self):
        window = RollingMeanWindow(3)
        window.append(1.0)
        window.append(2.0)
        self.assertIsNone(window.mean())
        window.append(3.0)
        self.assertAlmostEqual(window.mean(), 2.0)
        window.append(4.0)
        self.assertAlmostEqual(window.mean(), 3.0)

    def test_incremental_rsi_produces_bounded_values(self):
        rsi = IncrementalRsi(period=5)
        closes = [100, 101, 102, 103, 104, 103, 102, 101, 102, 103]
        outputs = []
        for close_price in closes:
            value = rsi.update(close_price)
            if value is not None:
                outputs.append(value)

        self.assertGreater(len(outputs), 0)
        for value in outputs:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

        saved = rsi.to_state()
        restored = IncrementalRsi(period=5)
        restored.load_state(saved)
        self.assertEqual(saved, restored.to_state())

    def test_rolling_zscore_window(self):
        z_window = RollingZScoreWindow(window=4)
        for value in [1.0, 2.0, 3.0, 4.0]:
            z_window.append(value)

        z_value = z_window.zscore(5.0)
        self.assertIsNotNone(z_value)
        self.assertAlmostEqual(z_value, 2.2360679, places=6)

    def test_vwap_helpers(self):
        self.assertAlmostEqual(vwap_from_sums(1000.0, 10.0), 100.0)
        self.assertIsNone(vwap_from_sums(1000.0, 0.0))
        self.assertAlmostEqual(vwap_deviation(98.0, 100.0), -0.02)
        self.assertIsNone(vwap_deviation(98.0, None))

        prices = [100.0, 101.0, 102.0, 103.0]
        volumes = [1.0, 1.0, 1.0, 1.0]
        self.assertAlmostEqual(rolling_vwap(prices, volumes, 4), 101.5)
        self.assertIsNone(rolling_vwap(prices, volumes, 1))

    def test_atr_helpers(self):
        tr1 = true_range(105.0, 100.0, None)
        tr2 = true_range(110.0, 104.0, 103.0)
        tr3 = true_range(111.0, 107.0, 112.0)
        self.assertAlmostEqual(tr1, 5.0)
        self.assertAlmostEqual(tr2, 7.0)
        self.assertAlmostEqual(tr3, 5.0)

        atr = average_true_range([tr1, tr2, tr3], 3)
        self.assertAlmostEqual(atr, 17.0 / 3.0)

    def test_rolling_stat_helpers(self):
        values = [1.0, 2.0, 3.0, 4.0]
        std_value = sample_std(values)
        self.assertIsNotNone(std_value)
        self.assertAlmostEqual(std_value, 1.2909944487358056)

        y_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        x_values = [3.0, 5.0, 7.0, 9.0, 11.0]  # x = 2y + 1
        beta = rolling_beta(x_values, y_values)
        corr = rolling_corr(x_values, y_values)
        self.assertIsNotNone(beta)
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(beta, 2.0, places=6)
        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_momentum_helpers(self):
        self.assertAlmostEqual(momentum_return(110.0, 100.0), 0.1)
        self.assertIsNone(momentum_return(110.0, 0.0))
        self.assertAlmostEqual(momentum_spread(0.15, -0.05), 0.2)


if __name__ == "__main__":
    unittest.main()
