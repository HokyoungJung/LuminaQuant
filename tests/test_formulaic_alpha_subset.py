import unittest

from lumina_quant.indicators import (
    alpha_001,
    alpha_002,
    alpha_003,
    alpha_004,
    alpha_005,
    alpha_006,
    alpha_007,
    alpha_008,
    alpha_009,
    alpha_010,
    alpha_011,
    alpha_012,
    alpha_013,
    alpha_014,
    alpha_015,
    alpha_016,
    alpha_018,
    alpha_019,
    alpha_020,
    alpha_025,
    alpha_041,
    alpha_042,
    alpha_043,
    alpha_044,
    alpha_053,
    alpha_054,
    alpha_055,
    alpha_101,
)
from lumina_quant.indicators.formulaic_alpha import ALPHA_101_FUNCTIONS, compute_alpha101


class TestFormulaicAlphaSubset(unittest.TestCase):
    def setUp(self):
        self.closes = [100.0 + float(idx) + (0.2 if idx % 3 == 0 else -0.1) for idx in range(120)]
        self.opens = [value - 0.3 for value in self.closes]
        self.highs = [value + 0.8 for value in self.closes]
        self.lows = [value - 0.9 for value in self.closes]
        self.vwaps = [value - 0.05 for value in self.closes]
        self.volumes = [1000.0 + 5.0 * idx for idx in range(120)]

    def test_alpha_subset_returns_values(self):
        values = [
            alpha_001(self.closes),
            alpha_002(self.closes, self.opens, self.volumes),
            alpha_003(self.opens, self.volumes),
            alpha_004(self.lows),
            alpha_005(self.opens, self.closes, self.vwaps),
            alpha_006(self.opens, self.volumes),
            alpha_007(self.closes, self.volumes),
            alpha_008(self.opens, self.closes),
            alpha_009(self.closes),
            alpha_010(self.closes),
            alpha_011(self.closes, self.vwaps, self.volumes),
            alpha_012(self.closes, self.volumes),
            alpha_013(self.closes, self.volumes),
            alpha_014(self.closes, self.opens, self.volumes),
            alpha_015(self.highs, self.volumes),
            alpha_016(self.highs, self.volumes),
            alpha_018(self.closes, self.opens),
            alpha_019(self.closes),
            alpha_020(self.opens, self.highs, self.lows, self.closes),
            alpha_025(self.highs, self.closes, self.volumes, self.vwaps),
            alpha_041(self.highs, self.lows, self.vwaps),
            alpha_042(self.closes, self.vwaps),
            alpha_043(self.closes, self.volumes),
            alpha_044(self.highs, self.volumes),
            alpha_053(self.highs, self.lows, self.closes),
            alpha_054(self.opens, self.highs, self.lows, self.closes),
            alpha_055(self.highs, self.lows, self.closes, self.volumes),
            alpha_101(self.opens, self.highs, self.lows, self.closes),
        ]
        for value in values:
            self.assertIsNotNone(value)

    def test_alpha_tunable_parameters(self):
        default_value = alpha_001(self.closes)
        tuned_value = alpha_001(self.closes, std_window=15, argmax_window=7)
        self.assertIsNotNone(default_value)
        self.assertIsNotNone(tuned_value)

        alt_default = alpha_005(self.opens, self.closes, self.vwaps, mean_window=10, rank_window=20)
        alt_tuned = alpha_005(self.opens, self.closes, self.vwaps, mean_window=6, rank_window=30)
        self.assertIsNotNone(alt_default)
        self.assertIsNotNone(alt_tuned)
        self.assertTrue(isinstance(alt_default, (int, float)))
        self.assertTrue(isinstance(alt_tuned, (int, float)))

        default_alpha_101 = alpha_101(self.opens, self.highs, self.lows, self.closes)
        tuned_alpha_101 = alpha_101(self.opens, self.highs, self.lows, self.closes, eps=0.01)
        self.assertIsNotNone(default_alpha_101)
        self.assertIsNotNone(tuned_alpha_101)
        self.assertNotEqual(default_alpha_101, tuned_alpha_101)

    def test_formula_evaluator_covers_all_101_ids(self):
        closes = [100.0 + (0.3 * idx) + (0.1 if idx % 2 == 0 else -0.2) for idx in range(420)]
        opens = [close - 0.25 for close in closes]
        highs = [close + 0.85 for close in closes]
        lows = [close - 0.9 for close in closes]
        volumes = [2000.0 + (10.0 * idx) for idx in range(420)]
        vwaps = [((high + low + close) / 3.0) for high, low, close in zip(highs, lows, closes)]

        self.assertEqual(len(ALPHA_101_FUNCTIONS), 101)

        non_null_count = 0
        for alpha_id in range(1, 102):
            value = compute_alpha101(
                alpha_id,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                vwaps=vwaps,
            )
            if value is not None:
                non_null_count += 1
            self.assertTrue(value is None or isinstance(float(value), float))

        self.assertGreaterEqual(non_null_count, 80)


if __name__ == "__main__":
    unittest.main()
