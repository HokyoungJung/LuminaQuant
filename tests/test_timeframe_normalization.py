import unittest

from lumina_quant.backtest import TimeframeGatedStrategy
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds


class TestTimeframeNormalization(unittest.TestCase):
    def test_month_and_minute_tokens_are_distinct(self):
        self.assertEqual(normalize_timeframe_token("1m"), "1m")
        self.assertEqual(normalize_timeframe_token("1M"), "1M")
        self.assertEqual(timeframe_to_milliseconds("1m"), 60_000)
        self.assertEqual(timeframe_to_milliseconds("1M"), 2_592_000_000)

    def test_normalization_preserves_supported_units(self):
        self.assertEqual(normalize_timeframe_token("5H"), "5h")
        self.assertEqual(normalize_timeframe_token("7d"), "7d")
        self.assertEqual(normalize_timeframe_token("2W"), "2w")

    def test_backtest_timeframe_gate_preserves_month_token(self):
        class _DummyStrategy:
            def calculate_signals(self, event):
                _ = event

        gated = TimeframeGatedStrategy(_DummyStrategy(), "1M")
        self.assertEqual(gated._timeframe, "1M")
        self.assertEqual(gated._timeframe_ms, 2_592_000_000)


if __name__ == "__main__":
    unittest.main()
