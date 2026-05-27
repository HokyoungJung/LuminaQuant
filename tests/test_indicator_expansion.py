import unittest

from lumina_quant.indicators import (
    chande_momentum_oscillator,
    conditional_value_at_risk,
    cumulative_return,
    detrended_price_oscillator,
    fisher_transform,
    kaufman_efficiency_ratio,
    linear_regression_slope,
    max_drawdown,
    negative_volume_index,
    positive_volume_index,
    price_volume_correlation,
    rolling_sharpe_ratio,
    rolling_sortino_ratio,
    value_at_risk,
    volume_oscillator,
    vortex_indicator,
)


class TestIndicatorExpansion(unittest.TestCase):
    def setUp(self):
        self.closes = [100.0 + (0.6 * idx) + (0.15 if idx % 3 == 0 else -0.1) for idx in range(260)]
        self.highs = [close + 1.1 for close in self.closes]
        self.lows = [close - 1.2 for close in self.closes]
        self.volumes = [
            1000.0 + (7.0 * idx) + (50.0 if idx % 5 == 0 else 0.0) for idx in range(260)
        ]

    def test_extended_momentum_indicators(self):
        self.assertIsNotNone(cumulative_return(self.closes))
        self.assertIsNotNone(cumulative_return(self.closes, period=30))
        self.assertIsNotNone(kaufman_efficiency_ratio(self.closes, period=20))

        cmo = chande_momentum_oscillator(self.closes, period=14)
        self.assertIsNotNone(cmo)
        self.assertGreaterEqual(float(cmo), -100.0)
        self.assertLessEqual(float(cmo), 100.0)

        self.assertIsNotNone(detrended_price_oscillator(self.closes, period=20))
        self.assertIsNotNone(fisher_transform(self.closes, period=10))

    def test_extended_trend_indicators(self):
        vi_plus, vi_minus = vortex_indicator(self.highs, self.lows, self.closes, period=14)
        self.assertIsNotNone(vi_plus)
        self.assertIsNotNone(vi_minus)
        self.assertGreater(float(vi_plus), 0.0)
        self.assertGreater(float(vi_minus), 0.0)

        slope_norm = linear_regression_slope(self.closes, window=30, normalize=True)
        slope_raw = linear_regression_slope(self.closes, window=30, normalize=False)
        self.assertIsNotNone(slope_norm)
        self.assertIsNotNone(slope_raw)

    def test_extended_volatility_and_risk(self):
        volatile_closes = [
            100.0 + (0.2 * idx) + (2.5 if idx % 7 == 0 else (-3.0 if idx % 5 == 0 else 0.0))
            for idx in range(260)
        ]

        mdd = max_drawdown(volatile_closes)
        self.assertIsNotNone(mdd)
        self.assertLessEqual(float(mdd), 0.0)

        var95 = value_at_risk(volatile_closes, window=120, confidence=0.95)
        cvar95 = conditional_value_at_risk(volatile_closes, window=120, confidence=0.95)
        self.assertIsNotNone(var95)
        self.assertIsNotNone(cvar95)
        self.assertGreaterEqual(float(var95), 0.0)
        self.assertGreaterEqual(float(cvar95), float(var95))

        sharpe = rolling_sharpe_ratio(volatile_closes, window=63)
        sortino = rolling_sortino_ratio(volatile_closes, window=63)
        self.assertIsNotNone(sharpe)
        self.assertIsNotNone(sortino)

    def test_extended_volume_indicators(self):
        pvi = positive_volume_index(self.closes, self.volumes, initial_value=1000.0)
        nvi = negative_volume_index(self.closes, self.volumes, initial_value=1000.0)
        vo = volume_oscillator(self.volumes, short_period=10, long_period=30)
        pvc = price_volume_correlation(self.closes, self.volumes, window=30)

        self.assertIsNotNone(pvi)
        self.assertIsNotNone(nvi)
        self.assertIsNotNone(vo)
        self.assertIsNotNone(pvc)


if __name__ == "__main__":
    unittest.main()
