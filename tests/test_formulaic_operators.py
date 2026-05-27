import unittest

from lumina_quant.indicators import (
    decay_linear,
    delay,
    delta,
    rank_pct,
    returns_from_close,
    signed_power,
    ts_argmax,
    ts_argmin,
    ts_correlation,
    ts_covariance,
    ts_max,
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
)


class TestFormulaicOperators(unittest.TestCase):
    def test_delay_and_delta(self):
        values = [1.0, 2.0, 4.0, 7.0]
        self.assertEqual(delay(values, 1), 4.0)
        self.assertEqual(delay(values, 2), 2.0)
        self.assertEqual(delta(values, 1), 3.0)
        self.assertEqual(delta(values, 2), 5.0)
        self.assertIsNone(delay([1.0], 1))

    def test_ts_window_ops(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(ts_sum(values, 3), 12.0)
        self.assertEqual(ts_product(values, 3), 60.0)
        self.assertEqual(ts_min(values, 3), 3.0)
        self.assertEqual(ts_max(values, 3), 5.0)
        self.assertEqual(ts_argmin(values, 3), 1.0)
        self.assertEqual(ts_argmax(values, 3), 3.0)

    def test_rank_decay_and_stats(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(rank_pct(values, 5), 0.9)
        self.assertAlmostEqual(ts_rank(values, 5), 0.9)
        self.assertAlmostEqual(decay_linear(values, 5), 55.0 / 15.0)
        self.assertEqual(signed_power(-2.0, 3.0), -8.0)
        self.assertIsNotNone(ts_stddev(values, 5))

    def test_corr_cov_and_returns(self):
        x_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr = ts_correlation(x_values, y_values, 5)
        cov = ts_covariance(x_values, y_values, 5)
        self.assertIsNotNone(corr)
        self.assertIsNotNone(cov)
        self.assertAlmostEqual(corr, 1.0, places=6)
        self.assertGreater(cov, 0.0)

        returns = returns_from_close([100.0, 102.0, 101.0, 103.0])
        self.assertEqual(len(returns), 3)
        self.assertAlmostEqual(returns[0], 0.02)


if __name__ == "__main__":
    unittest.main()
