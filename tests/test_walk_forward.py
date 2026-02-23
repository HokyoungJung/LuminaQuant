import unittest
from datetime import datetime

from lumina_quant.optimization.walkers import build_walk_forward_splits


class TestWalkForwardSplits(unittest.TestCase):
    def test_split_count_and_order(self):
        base = datetime(2024, 1, 1)
        splits = build_walk_forward_splits(
            base_start=base,
            folds=3,
            train_months=12,
            val_months=6,
            test_months=6,
            step_months=6,
        )
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0]["train_start"], base)
        for s in splits:
            self.assertLess(s["train_start"], s["train_end"])
            self.assertLess(s["train_end"], s["val_end"])
            self.assertLess(s["val_end"], s["test_end"])
        self.assertLess(splits[0]["train_start"], splits[1]["train_start"])
        self.assertLess(splits[1]["train_start"], splits[2]["train_start"])


if __name__ == "__main__":
    unittest.main()
