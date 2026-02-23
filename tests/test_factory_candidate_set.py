import unittest

from strategies.factory_candidate_set import build_candidate_set, summarize_candidate_set


class TestFactoryCandidateSet(unittest.TestCase):
    def test_build_candidate_set_for_small_universe(self):
        candidates = build_candidate_set(
            symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT"],
            timeframes=["1s", "1m"],
            max_candidates=0,
        )
        self.assertGreater(len(candidates), 0)

        first = candidates[0]
        self.assertIn("candidate_id", first)
        self.assertIn("strategy", first)
        self.assertIn("timeframe", first)
        self.assertIn("symbols", first)
        self.assertIn("params", first)

    def test_summary_contains_strategy_and_timeframe_counts(self):
        candidates = build_candidate_set(
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframes=["1m"],
            max_candidates=120,
        )
        summary = summarize_candidate_set(candidates)

        self.assertEqual(summary["count"], len(candidates))
        self.assertIn("strategies", summary)
        self.assertIn("timeframes", summary)
        self.assertIn("families", summary)
        self.assertIn("1m", summary["timeframes"])


if __name__ == "__main__":
    unittest.main()
