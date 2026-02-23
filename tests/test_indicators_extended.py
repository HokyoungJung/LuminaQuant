import unittest

from lumina_quant.indicators import (
    accumulation_distribution_line,
    aroon_indicator,
    atr_percent,
    awesome_oscillator,
    bollinger_bands,
    bollinger_bandwidth,
    chaikin_money_flow,
    chaikin_oscillator,
    choppiness_index,
    commodity_channel_index,
    donchian_channel,
    downside_volatility,
    historical_volatility,
    ichimoku_cloud,
    keltner_channel,
    log_returns,
    money_flow_index,
    moving_average_convergence_divergence,
    on_balance_volume,
    parkinson_volatility,
    percentage_price_oscillator,
    rate_of_change,
    relative_strength_index,
    stochastic_oscillator,
    stochastic_rsi,
    supertrend,
    triple_exponential_average_rate_of_change,
    ulcer_index,
    ultimate_oscillator,
    volume_price_trend,
    volume_weighted_moving_average,
    williams_r,
    zscore,
)


class TestIndicatorsExtended(unittest.TestCase):
    def setUp(self):
        self.closes = [
            100.0,
            101.0,
            102.5,
            101.5,
            103.0,
            104.0,
            103.5,
            105.0,
            106.0,
            107.5,
            108.0,
            109.0,
            110.5,
            111.0,
            112.0,
            111.5,
            113.0,
            114.0,
            115.0,
            116.0,
            117.5,
            118.0,
            119.0,
            120.0,
            121.0,
            122.0,
            123.0,
            124.0,
            125.0,
            126.0,
            127.0,
            128.0,
            129.0,
            130.0,
            131.0,
            132.0,
            133.0,
            134.0,
            135.0,
            136.0,
        ]
        self.closes.extend([137.0 + float(idx) for idx in range(30)])
        self.highs = [value + 1.2 for value in self.closes]
        self.lows = [value - 1.0 for value in self.closes]
        self.volumes = [1000.0 + 10.0 * idx for idx in range(len(self.closes))]

    def test_band_indicators(self):
        middle, upper, lower = bollinger_bands(self.closes, window=20, num_std=2.0)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)

        upper_d, middle_d, lower_d = donchian_channel(self.highs, self.lows, window=20)
        self.assertGreater(upper_d, lower_d)
        self.assertAlmostEqual(middle_d, (upper_d + lower_d) / 2.0)

        middle_k, upper_k, lower_k = keltner_channel(
            self.highs,
            self.lows,
            self.closes,
            window=20,
            atr_window=10,
            atr_multiplier=1.5,
        )
        self.assertIsNotNone(middle_k)
        self.assertIsNotNone(upper_k)
        self.assertIsNotNone(lower_k)

    def test_oscillator_indicators(self):
        rsi = relative_strength_index(self.closes, period=14)
        self.assertIsNotNone(rsi)
        self.assertGreaterEqual(rsi, 0.0)
        self.assertLessEqual(rsi, 100.0)

        roc = rate_of_change(self.closes, period=12)
        self.assertIsNotNone(roc)

        cci = commodity_channel_index(self.highs, self.lows, self.closes, period=20)
        self.assertIsNotNone(cci)

        k_value, d_value = stochastic_oscillator(self.highs, self.lows, self.closes)
        self.assertIsNotNone(k_value)
        self.assertIsNotNone(d_value)

        wr = williams_r(self.highs, self.lows, self.closes, period=14)
        self.assertIsNotNone(wr)
        self.assertGreaterEqual(wr, -100.0)
        self.assertLessEqual(wr, 0.0)

        mfi = money_flow_index(self.highs, self.lows, self.closes, self.volumes, period=14)
        self.assertIsNotNone(mfi)

        stoch_rsi_k, stoch_rsi_d = stochastic_rsi(self.closes, rsi_period=14, stoch_period=14)
        self.assertIsNotNone(stoch_rsi_k)
        self.assertIsNotNone(stoch_rsi_d)

        z_value = zscore(self.closes, window=20)
        self.assertIsNotNone(z_value)

        ao = awesome_oscillator(self.highs, self.lows)
        self.assertIsNotNone(ao)

        uo = ultimate_oscillator(self.highs, self.lows, self.closes)
        self.assertIsNotNone(uo)

    def test_trend_indicators(self):
        macd, signal, hist = moving_average_convergence_divergence(self.closes)
        self.assertIsNotNone(macd)
        self.assertIsNotNone(signal)
        self.assertIsNotNone(hist)

        aroon_up, aroon_down, aroon_osc = aroon_indicator(self.highs, self.lows, period=25)
        self.assertIsNotNone(aroon_up)
        self.assertIsNotNone(aroon_down)
        self.assertIsNotNone(aroon_osc)

        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_cloud(
            self.highs,
            self.lows,
            self.closes,
            tenkan_period=9,
            kijun_period=26,
            senkou_b_period=52,
        )
        self.assertIsNotNone(tenkan)
        self.assertIsNotNone(kijun)
        self.assertIsNotNone(senkou_a)
        self.assertIsNotNone(senkou_b)
        self.assertIsNotNone(chikou)

        st_value, direction, upper_band, lower_band = supertrend(
            self.highs,
            self.lows,
            self.closes,
            atr_period=10,
            multiplier=3.0,
        )
        self.assertIsNotNone(st_value)
        self.assertIn(direction, ("LONG", "SHORT"))
        self.assertIsNotNone(upper_band)
        self.assertIsNotNone(lower_band)

        ppo, ppo_signal, ppo_hist = percentage_price_oscillator(self.closes)
        self.assertIsNotNone(ppo)
        self.assertIsNotNone(ppo_signal)
        self.assertIsNotNone(ppo_hist)

        trix, trix_signal = triple_exponential_average_rate_of_change(self.closes)
        self.assertIsNotNone(trix)
        self.assertIsNotNone(trix_signal)

    def test_volume_indicators(self):
        obv = on_balance_volume(self.closes, self.volumes)
        self.assertIsNotNone(obv)

        adl = accumulation_distribution_line(self.highs, self.lows, self.closes, self.volumes)
        self.assertIsNotNone(adl)

        cmf = chaikin_money_flow(self.highs, self.lows, self.closes, self.volumes, period=20)
        self.assertIsNotNone(cmf)

        vwma = volume_weighted_moving_average(self.closes, self.volumes, period=20)
        self.assertIsNotNone(vwma)

        cho = chaikin_oscillator(self.highs, self.lows, self.closes, self.volumes)
        self.assertIsNotNone(cho)

        vpt = volume_price_trend(self.closes, self.volumes)
        self.assertIsNotNone(vpt)

    def test_volatility_indicators(self):
        returns = log_returns(self.closes)
        self.assertGreater(len(returns), 0)

        hv = historical_volatility(self.closes, window=20)
        self.assertIsNotNone(hv)
        self.assertGreater(hv, 0.0)

        atrp = atr_percent(self.highs, self.lows, self.closes, period=14)
        self.assertIsNotNone(atrp)

        bbw = bollinger_bandwidth(self.closes, window=20, num_std=2.0)
        self.assertIsNotNone(bbw)

        chop = choppiness_index(self.highs, self.lows, self.closes, period=14)
        self.assertIsNotNone(chop)

        ui = ulcer_index(self.closes, window=14)
        self.assertIsNotNone(ui)

        downside = downside_volatility(self.closes, window=20)
        self.assertIsNotNone(downside)

        park = parkinson_volatility(self.highs, self.lows, window=20)
        self.assertIsNotNone(park)


if __name__ == "__main__":
    unittest.main()
