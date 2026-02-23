from __future__ import annotations

import random
import unittest

from lumina_quant.utils.performance import PerformanceMetrics, create_drawdowns


def _legacy_create_drawdowns(pnl):
    hwm = [0]
    drawdown = [0.0] * len(pnl)
    duration = [0] * len(pnl)

    for t in range(1, len(pnl)):
        cur_val = pnl[t]
        hwm.append(max(hwm[t - 1], cur_val))

        div = hwm[t]
        if div == 0:
            div = 1

        dd = (hwm[t] - cur_val) / div
        drawdown[t] = dd
        duration[t] = 0 if dd == 0 else duration[t - 1] + 1

    return drawdown, max(duration) if duration else 0


class TestCreateDrawdowns(unittest.TestCase):
    def test_matches_legacy_behavior_on_fixed_cases(self):
        cases = [
            [],
            [100.0],
            [100.0, 110.0, 108.0, 120.0, 90.0, 95.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [-1.0, -2.0, -1.5, -3.0],
        ]
        for pnl in cases:
            expected_dd, expected_duration = _legacy_create_drawdowns(pnl)
            actual_dd, actual_duration = create_drawdowns(pnl)
            self.assertEqual(len(expected_dd), len(actual_dd))
            for left, right in zip(expected_dd, actual_dd, strict=False):
                self.assertAlmostEqual(left, right, places=12)
            self.assertEqual(expected_duration, actual_duration)

    def test_matches_legacy_behavior_on_random_walks(self):
        rng = random.Random(42)
        for _ in range(20):
            value = 100.0
            series = []
            for _ in range(300):
                value += rng.uniform(-2.0, 2.0)
                series.append(value)
            expected_dd, expected_duration = _legacy_create_drawdowns(series)
            actual_dd, actual_duration = create_drawdowns(series)
            for left, right in zip(expected_dd, actual_dd, strict=False):
                self.assertAlmostEqual(left, right, places=12)
            self.assertEqual(expected_duration, actual_duration)

    def test_oop_facade_matches_function_output(self):
        pnl = [100.0, 102.0, 101.5, 103.0, 99.0, 98.0, 100.0]
        expected_dd, expected_duration = create_drawdowns(pnl)
        actual_dd, actual_duration = PerformanceMetrics.drawdowns(pnl)
        self.assertEqual(expected_duration, actual_duration)
        for left, right in zip(expected_dd, actual_dd, strict=False):
            self.assertAlmostEqual(left, right, places=12)


if __name__ == "__main__":
    unittest.main()
