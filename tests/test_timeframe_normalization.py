import unittest

from lumina_quant.backtesting.backtest import TimeframeGatedStrategy
from lumina_quant.data.raw_first_lineage import (
    normalize_timeframe_token as raw_first_normalize_timeframe_token,
)
from lumina_quant.data.raw_first_lineage import (
    timeframe_to_milliseconds as raw_first_timeframe_to_milliseconds,
)
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

    def test_raw_first_timeframe_helpers_match_market_data_contract(self):
        for token in ("1m", "1M", "5H", "7d", "2W", "12h", "30m"):
            self.assertEqual(
                raw_first_normalize_timeframe_token(token),
                normalize_timeframe_token(token),
            )
            self.assertEqual(
                raw_first_timeframe_to_milliseconds(token),
                timeframe_to_milliseconds(token),
            )


if __name__ == "__main__":
    unittest.main()
